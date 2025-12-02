"""
Self-Supervised Waypoint Prediction Losses

This module implements a hybrid loss function for training the waypoint
prediction head of the hierarchical policy. The key insight is that we can
use the agent's own trajectories as supervision signal:

1. Imitation Loss: For successful (safe) trajectories, minimize the distance
   between predicted waypoints and actual future positions.

2. Repulsion Loss: For failed (crash) trajectories, maximize the distance
   between predicted waypoints and the crash locations.

3. Goal-Directed Loss: Encourage waypoints to point toward the goal.

This approach requires no human demonstration data - the agent learns from
its own experience what paths are achievable and which lead to failure.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class WaypointLoss:
    """
    Hybrid loss combining imitation, repulsion, and goal-directed components.

    The loss function adapts based on trajectory safety:
    - Safe trajectories (high reward): imitate the actual path taken
    - Unsafe trajectories (crash/low reward): repel from the failure path
    """

    def __init__(
            self,
            self_supervised_weight: float = 1.0,
            goal_directed_weight: float = 0.2,
            repulsion_weight: float = 2.0,
            repulsion_margin: float = 2.5,
            num_waypoints: int = 5,
            waypoint_spacing: float = 0.5,
    ):
        """
        Initialize the waypoint loss function.

        Args:
            self_supervised_weight: Weight for imitation loss on safe trajectories
            goal_directed_weight: Weight for goal alignment loss
            repulsion_weight: Weight for repulsion loss on unsafe trajectories
            repulsion_margin: Minimum distance to maintain from unsafe waypoints (meters)
            num_waypoints: Number of waypoints predicted by the policy
            waypoint_spacing: Time spacing between waypoints (seconds)
        """
        self.self_supervised_weight = self_supervised_weight
        self.goal_directed_weight = goal_directed_weight
        self.repulsion_weight = repulsion_weight
        self.repulsion_margin = repulsion_margin
        self.num_waypoints = num_waypoints
        self.waypoint_spacing = waypoint_spacing

    def compute_loss(
            self,
            predicted_waypoints: torch.Tensor,
            trajectory_buffer: Dict,
            current_indices: torch.Tensor,
            goal_telemetry: torch.Tensor,
            speeds: torch.Tensor,
            safety_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the combined waypoint prediction loss.

        Args:
            predicted_waypoints: (batch, num_waypoints, 2) predicted positions
            trajectory_buffer: Dict containing 'positions' and 'yaws' tensors
            current_indices: (batch,) indices into trajectory buffer
            goal_telemetry: (batch, 3) goal info [cos, sin, distance]
            speeds: (batch,) current vehicle speeds
            safety_mask: (batch,) binary mask, 1=safe trajectory, 0=unsafe

        Returns:
            total_loss: Scalar loss tensor
            loss_dict: Dictionary of individual loss components for logging
        """
        device = predicted_waypoints.device
        batch_size = predicted_waypoints.shape[0]

        # Extract actual future positions from trajectory buffer
        future_positions, valid_mask = self._extract_future_positions(
            trajectory_buffer, current_indices, device, batch_size
        )

        # Default to all safe if no mask provided
        if safety_mask is None:
            safety_mask = torch.ones(batch_size, device=device)

        # Expand masks for broadcasting
        safety_mask_expanded = safety_mask.view(batch_size, 1, 1).expand(-1, self.num_waypoints, 1)
        valid_mask_expanded = valid_mask.unsqueeze(-1).float()

        # 1. Imitation Loss (applied to safe trajectories only)
        mse = F.mse_loss(predicted_waypoints, future_positions, reduction='none')
        mse_per_point = mse.mean(dim=-1, keepdim=True)
        weighted_imitation = mse_per_point * safety_mask_expanded
        loss_imitation = (weighted_imitation * valid_mask_expanded).sum() / (valid_mask_expanded.sum() + 1e-6)

        # 2. Repulsion Loss (applied to unsafe trajectories only)
        # Penalize predictions that are too close to crash locations
        dist_to_crash = torch.norm(predicted_waypoints - future_positions, dim=-1, keepdim=True)
        repulsion = torch.clamp(self.repulsion_margin - dist_to_crash, min=0.0)
        crash_mask_expanded = 1.0 - safety_mask_expanded
        weighted_repulsion = repulsion * crash_mask_expanded
        loss_repulsion = (weighted_repulsion * valid_mask_expanded).sum() / (valid_mask_expanded.sum() + 1e-6)

        # 3. Goal-Directed Loss (always applied)
        loss_goal_directed = self._compute_goal_directed_loss(
            predicted_waypoints, goal_telemetry, speeds
        )

        # Combine losses
        total_loss = (
            self.self_supervised_weight * loss_imitation +
            self.repulsion_weight * loss_repulsion +
            self.goal_directed_weight * loss_goal_directed
        )

        loss_dict = {
            "loss_imitation": loss_imitation.item(),
            "loss_repulsion": loss_repulsion.item(),
            "loss_goal_directed": loss_goal_directed.item()
        }

        return total_loss, loss_dict

    def _extract_future_positions(
            self,
            trajectory_buffer: Dict,
            current_indices: torch.Tensor,
            device: torch.device,
            batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract future positions from trajectory buffer and transform to vehicle frame.

        Args:
            trajectory_buffer: Contains 'positions' and 'yaws' arrays
            current_indices: Current timestep indices
            device: Target device for tensors
            batch_size: Batch size

        Returns:
            future_positions: (batch, num_waypoints, 2) future positions in vehicle frame
            valid_mask: (batch, num_waypoints) boolean mask for valid entries
        """
        positions = trajectory_buffer['positions']
        yaws = trajectory_buffer['yaws']

        future_positions = torch.zeros(batch_size, self.num_waypoints, 2, device=device)
        valid_mask = torch.zeros(batch_size, self.num_waypoints, dtype=torch.bool, device=device)

        for b in range(batch_size):
            current_idx = int(current_indices[b].item())
            car_pos_t = positions[current_idx]
            car_yaw_t = yaws[current_idx]

            for w in range(self.num_waypoints):
                # Calculate future index based on waypoint spacing
                # Assuming 50Hz (0.02s) physics timestep
                step_offset = (w + 1) * int(self.waypoint_spacing / 0.02)
                future_idx = current_idx + step_offset

                if future_idx < len(positions):
                    future_pos_world = positions[future_idx]
                    future_positions[b, w] = self._world_to_car_frame(
                        future_pos_world, car_pos_t, car_yaw_t
                    )
                    valid_mask[b, w] = True

        return future_positions, valid_mask

    def _compute_goal_directed_loss(
            self,
            predicted_waypoints: torch.Tensor,
            goal_telemetry: torch.Tensor,
            speeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss encouraging waypoints to point toward the goal.

        Creates target waypoints along the goal direction, with distances
        based on current speed and time to each waypoint.
        """
        goal_cos = goal_telemetry[:, 0]
        goal_sin = goal_telemetry[:, 1]
        goal_dist = goal_telemetry[:, 2]

        target_waypoints = torch.zeros_like(predicted_waypoints)

        for w in range(self.num_waypoints):
            time_to_waypoint = (w + 1) * self.waypoint_spacing
            dist_to_waypoint = speeds * time_to_waypoint
            # Don't predict beyond the goal
            dist_to_waypoint = torch.minimum(dist_to_waypoint, goal_dist)

            # Convert goal direction to waypoint position
            # X = lateral (goal_sin), Y = forward (goal_cos)
            target_waypoints[:, w, 0] = dist_to_waypoint * goal_sin
            target_waypoints[:, w, 1] = dist_to_waypoint * goal_cos

        return F.mse_loss(predicted_waypoints, target_waypoints)

    @staticmethod
    def _world_to_car_frame(
            world_pos: torch.Tensor,
            car_world_pos: torch.Tensor,
            car_yaw: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform a world position to vehicle-local coordinates.

        Args:
            world_pos: Position in world frame (x, y)
            car_world_pos: Vehicle position in world frame
            car_yaw: Vehicle heading in world frame (radians)

        Returns:
            Local position (x_local, y_local) where:
            - x_local: lateral offset (positive = right)
            - y_local: longitudinal offset (positive = forward)
        """
        dx = world_pos[0] - car_world_pos[0]
        dy = world_pos[1] - car_world_pos[1]

        cos_yaw = torch.cos(car_yaw)
        sin_yaw = torch.sin(car_yaw)

        x_local = dx * cos_yaw + dy * sin_yaw
        y_local = -dx * sin_yaw + dy * cos_yaw

        return torch.tensor([x_local, y_local], device=world_pos.device)


class TrajectoryBuffer:
    """
    Circular buffer for storing vehicle trajectory data.

    Used for self-supervised learning: we record where the vehicle actually
    went, then use those positions as supervision targets for waypoint prediction.
    """

    def __init__(self, buffer_size: int = 500, num_envs: int = 1):
        """
        Initialize the trajectory buffer.

        Args:
            buffer_size: Maximum number of timesteps to store
            num_envs: Number of parallel environments
        """
        self.buffer_size = buffer_size
        self.num_envs = num_envs

        # Preallocate arrays
        self.positions = np.zeros((num_envs, buffer_size, 3), dtype=np.float32)
        self.yaws = np.zeros((num_envs, buffer_size), dtype=np.float32)
        self.speeds = np.zeros((num_envs, buffer_size), dtype=np.float32)

        # Tracking state
        self.current_idx = np.zeros(num_envs, dtype=np.int32)
        self.filled = np.zeros(num_envs, dtype=np.bool_)

    def add(self, env_idx: int, position: np.ndarray, yaw: float, speed: float) -> None:
        """Add a new data point to the buffer for a specific environment."""
        idx = self.current_idx[env_idx]

        self.positions[env_idx, idx] = position
        self.yaws[env_idx, idx] = yaw
        self.speeds[env_idx, idx] = speed

        # Update index (circular)
        self.current_idx[env_idx] = (idx + 1) % self.buffer_size

        # Mark as filled once we've wrapped around
        if idx == self.buffer_size - 1:
            self.filled[env_idx] = True

    def reset(self, env_idx: int) -> None:
        """Reset the buffer for a specific environment (e.g., on episode end)."""
        self.current_idx[env_idx] = 0
        self.filled[env_idx] = False

    def get_trajectory(self, env_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get the trajectory data for a specific environment.

        Returns data in chronological order, handling the circular buffer wrap.
        """
        idx = self.current_idx[env_idx]

        if self.filled[env_idx]:
            # Buffer has wrapped - concatenate old and new portions
            pos = np.concatenate([
                self.positions[env_idx, idx:],
                self.positions[env_idx, :idx]
            ])
            yaw = np.concatenate([
                self.yaws[env_idx, idx:],
                self.yaws[env_idx, :idx]
            ])
        else:
            # Buffer hasn't wrapped - just take up to current index
            pos = self.positions[env_idx, :idx]
            yaw = self.yaws[env_idx, :idx]

        return {
            'positions': torch.from_numpy(pos),
            'yaws': torch.from_numpy(yaw)
        }

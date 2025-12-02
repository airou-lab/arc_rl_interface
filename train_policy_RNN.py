#!/usr/bin/env python3
"""
train_policy_RNN.py - FIXED VERSION
====================================
Properly implements the waypoint auxiliary loss for hierarchical training.

Key Fixes:
1. Actually computes waypoint loss (was placeholder = 0.0)
2. Uses TrajectoryStore to access trajectory data
3. Proper safety mask handling for repulsion loss
4. Goal-directed loss uses turn_bias (not geometric bearing)
"""
import argparse
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

# SB3 Imports
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import explained_variance

# Project Imports
from unity_dense_env import UnityDenseEnv, DenseRewardConfig
from action_repeat_wrapper import ActionRepeatWrapper
from wrappers.waypoint_tracking_wrapper import WaypointTrackingWrapper, get_trajectory_store
from policies.hierarchical_policy import HierarchicalPathPlanningPolicy
from policies.fusion_policy import FusionFeaturesExtractor


class WaypointAuxiliaryLoss:
    """
    Computes the self-supervised waypoint auxiliary loss.

    Components:
    1. Imitation Loss: Predicted waypoints should match actual future trajectory (safe paths)
    2. Repulsion Loss: Predicted waypoints should avoid crash trajectories (unsafe paths)
    3. Goal-Directed Loss: Final waypoint should align with navigation command
    """

    def __init__(
            self,
            num_waypoints: int = 5,
            waypoint_spacing_seconds: float = 0.5,
            physics_dt: float = 0.02,
            imitation_weight: float = 1.0,
            repulsion_weight: float = 2.0,
            repulsion_margin: float = 2.5,
            goal_weight: float = 0.2,
    ):
        self.num_waypoints = num_waypoints
        self.waypoint_spacing_seconds = waypoint_spacing_seconds
        self.physics_dt = physics_dt
        self.steps_per_waypoint = int(waypoint_spacing_seconds / physics_dt)

        self.imitation_weight = imitation_weight
        self.repulsion_weight = repulsion_weight
        self.repulsion_margin = repulsion_margin
        self.goal_weight = goal_weight

    def compute(
            self,
            predicted_waypoints: torch.Tensor,
            trajectory_data: dict,
            current_indices: torch.Tensor,
            turn_bias: torch.Tensor,
            speeds: torch.Tensor,
            device: torch.device,
    ) -> tuple:
        """
        Compute the waypoint auxiliary loss.

        Args:
            predicted_waypoints: (batch_size, num_waypoints, 2) in vehicle frame
            trajectory_data: dict with 'positions', 'yaws', 'safety' arrays
            current_indices: (batch_size,) index into trajectory for each sample
            turn_bias: (batch_size,) navigation command [-1, 1]
            speeds: (batch_size,) current vehicle speed
            device: torch device

        Returns:
            total_loss: scalar tensor
            loss_dict: dict of individual loss components
        """
        batch_size = predicted_waypoints.shape[0]

        # Extract future positions in vehicle frame
        future_positions, valid_mask, safety_mask = self._extract_future_positions(
            trajectory_data, current_indices, device, batch_size
        )

        if valid_mask.sum() < 1:
            # No valid data - return zero loss
            zero = torch.tensor(0.0, device=device)
            return zero, {'imitation': 0.0, 'repulsion': 0.0, 'goal': 0.0}

        # 1. Imitation Loss (safe trajectories only)
        safe_mask = (safety_mask > 0.5).float().unsqueeze(-1)  # (batch, waypoints, 1)
        valid_safe = valid_mask.unsqueeze(-1).float() * safe_mask

        mse = F.mse_loss(predicted_waypoints, future_positions, reduction='none')
        imitation_loss = (mse * valid_safe).sum() / (valid_safe.sum() + 1e-6)

        # 2. Repulsion Loss (unsafe trajectories only)
        unsafe_mask = (safety_mask < 0.5).float().unsqueeze(-1)
        valid_unsafe = valid_mask.unsqueeze(-1).float() * unsafe_mask

        dist_to_unsafe = torch.norm(predicted_waypoints - future_positions, dim=-1, keepdim=True)
        repulsion = torch.clamp(self.repulsion_margin - dist_to_unsafe, min=0.0)
        repulsion_loss = (repulsion * valid_unsafe).sum() / (valid_unsafe.sum() + 1e-6)

        # 3. Goal-Directed Loss (align with turn command)
        goal_loss = self._compute_goal_loss(predicted_waypoints, turn_bias, speeds)

        # Total
        total_loss = (
                self.imitation_weight * imitation_loss +
                self.repulsion_weight * repulsion_loss +
                self.goal_weight * goal_loss
        )

        return total_loss, {
            'imitation': imitation_loss.item(),
            'repulsion': repulsion_loss.item(),
            'goal': goal_loss.item(),
        }

    def _extract_future_positions(
            self,
            trajectory_data: dict,
            current_indices: torch.Tensor,
            device: torch.device,
            batch_size: int,
    ) -> tuple:
        """Extract future positions in vehicle frame from trajectory buffer."""

        positions = trajectory_data.get('positions')
        yaws = trajectory_data.get('yaws')
        safety = trajectory_data.get('safety')

        if positions is None or len(positions) < 2:
            # Return empty tensors
            empty_pos = torch.zeros(batch_size, self.num_waypoints, 2, device=device)
            empty_mask = torch.zeros(batch_size, self.num_waypoints, dtype=torch.bool, device=device)
            empty_safety = torch.ones(batch_size, self.num_waypoints, device=device)
            return empty_pos, empty_mask, empty_safety

        # Convert to tensors
        positions = torch.tensor(positions, dtype=torch.float32, device=device)
        yaws = torch.tensor(yaws, dtype=torch.float32, device=device)
        safety = torch.tensor(safety, dtype=torch.float32, device=device)

        traj_len = positions.shape[0]

        future_positions = torch.zeros(batch_size, self.num_waypoints, 2, device=device)
        valid_mask = torch.zeros(batch_size, self.num_waypoints, dtype=torch.bool, device=device)
        safety_mask = torch.ones(batch_size, self.num_waypoints, device=device)

        for b in range(batch_size):
            current_idx = int(current_indices[b].item()) if current_indices.numel() > 1 else int(current_indices.item())
            current_idx = min(current_idx, traj_len - 1)

            if current_idx < 0:
                continue

            # Current position and yaw (for frame transformation)
            car_pos = positions[current_idx, :2]  # XZ plane
            car_yaw = yaws[current_idx]

            for w in range(self.num_waypoints):
                future_idx = current_idx + (w + 1) * self.steps_per_waypoint

                if future_idx < traj_len:
                    future_pos_world = positions[future_idx, :2]

                    # Transform to vehicle frame
                    local_pos = self._world_to_vehicle_frame(
                        future_pos_world, car_pos, car_yaw
                    )

                    future_positions[b, w] = local_pos
                    valid_mask[b, w] = True
                    safety_mask[b, w] = safety[min(future_idx, len(safety) - 1)]

        return future_positions, valid_mask, safety_mask

    def _world_to_vehicle_frame(
            self,
            world_pos: torch.Tensor,
            car_pos: torch.Tensor,
            car_yaw: torch.Tensor,
    ) -> torch.Tensor:
        """Transform world position to vehicle-local frame."""
        # Offset from car
        dx = world_pos[0] - car_pos[0]
        dy = world_pos[1] - car_pos[1]  # Note: using XZ plane (Unity Y=up)

        # Rotate to vehicle frame
        cos_yaw = torch.cos(car_yaw)
        sin_yaw = torch.sin(car_yaw)

        # X = lateral (right positive), Y = forward
        local_x = dx * cos_yaw + dy * sin_yaw
        local_y = -dx * sin_yaw + dy * cos_yaw

        return torch.stack([local_x, local_y])

    def _compute_goal_loss(
            self,
            predicted_waypoints: torch.Tensor,
            turn_bias: torch.Tensor,
            speeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute goal-directed loss based on turn command.

        In the Passive Visual paradigm, we don't have geometric goal bearing.
        Instead, we use the turn_bias command:
        - turn_bias = -1: Turn left (final waypoint should be left of center)
        - turn_bias = 0: Go straight (final waypoint should be ahead)
        - turn_bias = 1: Turn right (final waypoint should be right of center)
        """
        batch_size = predicted_waypoints.shape[0]

        # Get final waypoint
        final_wp = predicted_waypoints[:, -1, :]  # (batch, 2)

        # Expected lateral direction based on turn bias
        # turn_bias positive -> expect positive X (right)
        # turn_bias negative -> expect negative X (left)
        expected_lateral_sign = turn_bias  # [-1, 1]

        # Actual lateral position of final waypoint
        actual_lateral = final_wp[:, 0]  # X component

        # Penalty when signs don't match or magnitude is too small
        # We want: sign(actual_lateral) ≈ sign(turn_bias) when |turn_bias| > 0.3

        # For strong turn commands (|bias| > 0.3), penalize wrong direction
        strong_command_mask = (torch.abs(turn_bias) > 0.3).float()

        # Desired lateral = turn_bias * typical_turn_radius
        typical_turn_lateral = 4.0  # meters
        target_lateral = turn_bias * typical_turn_lateral

        # Loss: MSE between target and actual lateral for strong commands
        lateral_error = (actual_lateral - target_lateral) ** 2
        goal_loss = (lateral_error * strong_command_mask).mean()

        # Also encourage forward progress (Y should be positive)
        forward_pos = final_wp[:, 1]
        min_forward = 8.0  # Final waypoint should be at least 8m ahead
        forward_penalty = torch.clamp(min_forward - forward_pos, min=0.0).mean()

        return goal_loss + 0.1 * forward_penalty


class CustomHierarchicalPPO(RecurrentPPO):
    """
    Custom PPO that properly integrates the Waypoint Auxiliary Loss.
    """

    def __init__(self, *args, waypoint_loss_weight: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)

        self.waypoint_loss_weight = waypoint_loss_weight

        # Initialize the auxiliary loss
        self.waypoint_criterion = WaypointAuxiliaryLoss(
            num_waypoints=kwargs.get('policy_kwargs', {}).get('num_waypoints', 5),
            imitation_weight=1.0,
            repulsion_weight=kwargs.get('policy_kwargs', {}).get('repulsion_weight', 2.0),
            goal_weight=0.2,
        )

        # Get trajectory store
        self._trajectory_store = get_trajectory_store()

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        Includes waypoint auxiliary loss computation.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        waypoint_losses = []
        clip_fractions = []

        continue_training = True

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):

                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                # Standard PPO evaluation
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )

                # --- WAYPOINT AUXILIARY LOSS ---
                aux_loss = self._compute_waypoint_auxiliary_loss(
                    rollout_data.observations,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )
                waypoint_losses.append(aux_loss.item() if aux_loss.numel() > 0 else 0.0)

                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # Total loss with auxiliary waypoint loss
                loss = (
                        policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss
                        + self.waypoint_loss_weight * aux_loss
                )

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten()
        )

        # Logging
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)

        # Waypoint loss logging
        if waypoint_losses:
            self.logger.record("train/waypoint_aux_loss", np.mean(waypoint_losses))

        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def _compute_waypoint_auxiliary_loss(
            self,
            observations,
            lstm_states,
            episode_starts,
    ) -> torch.Tensor:
        """
        Compute the waypoint auxiliary loss for the current batch.
        """
        # Get trajectory data from the store
        trajectory_data = self._trajectory_store.get_trajectory(env_id=0)
        safety_mask = self._trajectory_store.get_safety_mask(env_id=0)

        if trajectory_data is None or len(trajectory_data.get('positions', [])) < 10:
            return torch.tensor(0.0, device=self.device)

        # Re-compute waypoints for this batch
        features = self.policy.extract_features(observations)

        latent_pi, _ = self.policy._process_sequence(
            features,
            lstm_states.pi,
            episode_starts,
            self.policy.lstm_actor
        )

        if self.policy.mlp_extractor is not None:
            latent_pi = self.policy.mlp_extractor.forward_actor(latent_pi)

        # Get observation vector
        obs_vec = observations['vec'] if isinstance(observations, dict) else observations

        # Compute waypoints
        predicted_waypoints = self.policy._compute_waypoints(latent_pi, obs_vec)

        # Extract turn_bias and speed
        turn_bias = obs_vec[:, HierarchicalPathPlanningPolicy.IDX_TURN_BIAS]
        speeds = obs_vec[:, HierarchicalPathPlanningPolicy.IDX_SPEED]

        # Current indices (simplified - use trajectory length as proxy)
        batch_size = obs_vec.shape[0]
        traj_len = len(trajectory_data.get('positions', []))

        # Distribute indices across trajectory (heuristic for batch training)
        current_indices = torch.linspace(
            0, max(0, traj_len - 50),
            batch_size,
            device=self.device
        ).long()

        # Combine trajectory data with safety mask
        combined_trajectory = {
            'positions': trajectory_data['positions'],
            'yaws': trajectory_data['yaws'],
            'safety': safety_mask if safety_mask is not None else np.ones(traj_len),
        }

        # Compute loss
        loss, loss_dict = self.waypoint_criterion.compute(
            predicted_waypoints,
            combined_trajectory,
            current_indices,
            turn_bias,
            speeds,
            self.device,
        )

        return loss


class WaypointVisualizationCallback(BaseCallback):
    """Callback to send waypoints to Unity for visualization during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Get waypoints from policy
        if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'last_waypoints'):
            wp = self.model.policy.last_waypoints

            if wp is not None and isinstance(wp, torch.Tensor):
                wp_np = wp.detach().cpu().numpy()

                # Log mean forward distance
                if wp_np.ndim == 3:
                    self.logger.record("waypoints/mean_forward_m", np.mean(wp_np[:, :, 1]))
                    self.logger.record("waypoints/mean_lateral_m", np.mean(np.abs(wp_np[:, :, 0])))

                # Send to Unity for visualization
                if len(self.training_env.envs) > 0:
                    env = self.training_env.envs[0]

                    # Unwrap to find the waypoint setter
                    while hasattr(env, 'env'):
                        if hasattr(env, 'set_predicted_waypoints'):
                            break
                        env = env.env

                    if hasattr(env, 'set_predicted_waypoints'):
                        if wp_np.ndim == 3:
                            env.set_predicted_waypoints(wp_np[0])
                        else:
                            env.set_predicted_waypoints(wp_np)

        return True


def make_env(host, port, img_size, max_steps, repeat, reward_cfg, verbose, env_id=0):
    """Create environment with proper wrapper chain."""

    def _init():
        env = UnityDenseEnv(
            host, port, img_size[0], img_size[1], max_steps,
            reward_cfg, verbose
        )
        if repeat > 1:
            env = ActionRepeatWrapper(env, repeat)
        env = WaypointTrackingWrapper(env, env_id=env_id)
        return Monitor(env)

    return _init


def train(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"hppo_{timestamp}"
    model_dir = Path(args.model_dir) / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.tensorboard_log)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Reward config
    reward_cfg = DenseRewardConfig(
        stuck_pen_k=0.5,
        goal_approach_k=0.0,  # Passive visual - no geometric goal approach
    )

    # Create environment
    env_fn = make_env(
        args.host, args.port, tuple(args.img_size),
        args.max_steps, args.repeat, reward_cfg, args.verbose > 1
    )
    env = DummyVecEnv([env_fn])

    # Policy kwargs
    policy_kwargs = {
        "lstm_hidden_size": args.lstm_hidden_size,
        "n_lstm_layers": args.n_lstm_layers,
        "enable_critic_lstm": True,
        "features_extractor_class": FusionFeaturesExtractor,
        "features_extractor_kwargs": {},
        "num_waypoints": args.num_waypoints,
        "waypoint_horizon": args.waypoint_horizon,
        "repulsion_weight": args.repulsion_weight,
        "waypoint_loss_weight": args.waypoint_loss_weight,
    }

    # Create model
    model = CustomHierarchicalPPO(
        HierarchicalPathPlanningPolicy,
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),
        verbose=args.verbose,
        waypoint_loss_weight=args.waypoint_loss_weight,
    )

    # Callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=args.save_freq,
            save_path=str(model_dir / "checkpoints"),
            name_prefix="hppo"
        ),
        WaypointVisualizationCallback(),
    ]

    print(f"=" * 60)
    print(f"Starting Hierarchical PPO Training")
    print(f"=" * 60)
    print(f"Resolution: {args.img_size}")
    print(f"Waypoints: {args.num_waypoints} @ {args.waypoint_horizon}s horizon")
    print(f"Waypoint Loss Weight: {args.waypoint_loss_weight}")
    print(f"Repulsion Weight: {args.repulsion_weight}")
    print(f"Model dir: {model_dir}")
    print(f"=" * 60)

    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList(callbacks),
        tb_log_name=run_name,
        progress_bar=True
    )

    model.save(str(model_dir / "final_model"))
    print(f"Model saved to {model_dir / 'final_model'}")

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Hierarchical PPO with Waypoint Loss")

    # Training
    parser.add_argument('--timesteps', type=int, default=200000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--n_steps', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--ent_coef', type=float, default=0.005)

    # Network
    parser.add_argument('--lstm_hidden_size', type=int, default=256)
    parser.add_argument('--n_lstm_layers', type=int, default=1)

    # Hierarchical
    parser.add_argument('--num_waypoints', type=int, default=5)
    parser.add_argument('--waypoint_horizon', type=float, default=2.5)
    parser.add_argument('--waypoint_loss_weight', type=float, default=0.15)
    parser.add_argument('--repulsion_weight', type=float, default=2.0)

    # Environment
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5556)
    parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128])
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--repeat', type=int, default=1)

    # Logging
    parser.add_argument('--model_dir', default='./models')
    parser.add_argument('--tensorboard_log', default='./tb')
    parser.add_argument('--save_freq', type=int, default=25000)
    parser.add_argument('--verbose', type=int, default=1)

    train(parser.parse_args())
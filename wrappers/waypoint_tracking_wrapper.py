"""
Waypoint Tracking Wrapper - FIXED VERSION
==========================================
Changes from original:
1. Safety backfill: Marks last N steps as unsafe when crash occurs
2. Global trajectory store: Allows training loop to access trajectory data
3. Position tracking from Unity (if available) or dead-reckoning fallback
4. Proper episode boundary handling
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional
from collections import deque
import threading


class TrajectoryStore:
    """
    Thread-safe global store for trajectory data.
    Allows the training loop to access trajectory data that would otherwise
    be discarded by SB3's rollout buffer.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Store trajectories per environment
        self._trajectories: Dict[int, Dict] = {}
        self._episode_safety: Dict[int, np.ndarray] = {}
        self._data_lock = threading.Lock()

    def store_trajectory(self, env_id: int, trajectory: Dict, safety_mask: np.ndarray):
        """Store trajectory data for an environment."""
        with self._data_lock:
            self._trajectories[env_id] = {
                'positions': trajectory['positions'].copy(),
                'yaws': trajectory['yaws'].copy(),
                'speeds': trajectory['speeds'].copy(),
            }
            self._episode_safety[env_id] = safety_mask.copy()

    def get_trajectory(self, env_id: int) -> Optional[Dict]:
        """Get stored trajectory for an environment."""
        with self._data_lock:
            return self._trajectories.get(env_id)

    def get_safety_mask(self, env_id: int) -> Optional[np.ndarray]:
        """Get safety mask for an environment."""
        with self._data_lock:
            return self._episode_safety.get(env_id)

    def clear(self, env_id: Optional[int] = None):
        """Clear stored data."""
        with self._data_lock:
            if env_id is not None:
                self._trajectories.pop(env_id, None)
                self._episode_safety.pop(env_id, None)
            else:
                self._trajectories.clear()
                self._episode_safety.clear()


# Global instance
_trajectory_store = TrajectoryStore()


def get_trajectory_store() -> TrajectoryStore:
    """Get the global trajectory store instance."""
    return _trajectory_store


class WaypointTrackingWrapper(gym.Wrapper):
    """
    Wrapper that tracks:
    1. Predicted waypoints from policy
    2. Actual trajectory (for self-supervised learning)
    3. Safety flags with backfill for crash trajectories

    Key fix: When a crash occurs, the last N steps are marked as unsafe,
    not just the final crash frame.
    """

    # Number of steps to backfill as unsafe when crash occurs
    SAFETY_BACKFILL_STEPS = 25  # ~0.5 seconds at 50Hz

    # Telemetry indices for position extraction
    IDX_SPEED = 3
    IDX_YAW_RATE = 4
    IDX_DS = 11

    def __init__(self, env: gym.Env, env_id: int = 0):
        super().__init__(env)
        self.env_id = env_id

        # Trajectory buffers
        self.position_history: list = []
        self.yaw_history: list = []
        self.speed_history: list = []
        self.reward_history: list = []

        # Safety tracking
        self.safety_history: list = []

        # Waypoint tracking
        self.last_predicted_waypoints: Optional[np.ndarray] = None
        self.waypoint_prediction_history: list = []

        # Position estimation (dead-reckoning fallback)
        self._estimated_pos = np.zeros(3, dtype=np.float32)
        self._estimated_yaw = 0.0

        # Episode tracking
        self._step_count = 0

        # Get global store
        self._store = get_trajectory_store()

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        # Clear histories
        self.position_history = []
        self.yaw_history = []
        self.speed_history = []
        self.reward_history = []
        self.safety_history = []
        self.waypoint_prediction_history = []
        self.last_predicted_waypoints = None

        # Reset position estimate
        self._estimated_pos = np.zeros(3, dtype=np.float32)
        self._estimated_yaw = 0.0
        self._step_count = 0

        # Clear store for this env
        self._store.clear(self.env_id)

        ret = self.env.reset(**kwargs)
        if isinstance(ret, tuple) and len(ret) == 2:
            return ret
        return ret, {}

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self._step_count += 1

        # Extract position - prefer Unity's ground truth if available
        if 'car_position' in info and info['car_position'] is not None:
            position = np.array(info['car_position'], dtype=np.float32)
        else:
            # Dead-reckoning fallback
            position = self._update_position_estimate(obs)

        # Extract yaw
        if 'car_yaw' in info and info['car_yaw'] is not None:
            yaw = float(info['car_yaw'])
        else:
            yaw = self._estimated_yaw

        # Extract speed from observation
        if isinstance(obs, dict) and 'vec' in obs:
            speed = float(obs['vec'][self.IDX_SPEED])
        else:
            speed = 0.0

        # Store trajectory data
        self.position_history.append(position.copy())
        self.yaw_history.append(yaw)
        self.speed_history.append(speed)
        self.reward_history.append(float(reward))

        # Initially mark all steps as safe (1.0)
        # Backfill will mark unsafe steps when episode ends
        self.safety_history.append(1.0)

        # Store waypoint prediction if available
        if self.last_predicted_waypoints is not None:
            self.waypoint_prediction_history.append({
                'step': self._step_count,
                'waypoints': self.last_predicted_waypoints.copy(),
                'position': position.copy(),
                'yaw': yaw
            })

        # Handle episode end - apply safety backfill
        if done or truncated:
            self._handle_episode_end(done, truncated, reward)

        # Add trajectory info (still useful for debugging)
        info['trajectory'] = {
            'positions': np.array(self.position_history),
            'yaws': np.array(self.yaw_history),
            'speeds': np.array(self.speed_history),
            'safety': np.array(self.safety_history)
        }

        if self.last_predicted_waypoints is not None:
            info['predicted_waypoints'] = self.last_predicted_waypoints.copy()

        return obs, reward, done, truncated, info

    def _update_position_estimate(self, obs) -> np.ndarray:
        """Dead-reckoning position update."""
        if not isinstance(obs, dict) or 'vec' not in obs:
            return self._estimated_pos.copy()

        vec = obs['vec']
        yaw_rate = float(vec[self.IDX_YAW_RATE])
        ds = float(vec[self.IDX_DS])

        dt = 0.02  # Assume 50Hz physics

        # Update yaw
        self._estimated_yaw += yaw_rate * dt

        # Normalize to [-pi, pi]
        while self._estimated_yaw > np.pi:
            self._estimated_yaw -= 2 * np.pi
        while self._estimated_yaw < -np.pi:
            self._estimated_yaw += 2 * np.pi

        # Update position using ds
        if abs(ds) > 0.001:
            self._estimated_pos[0] += ds * np.sin(self._estimated_yaw)
            self._estimated_pos[2] += ds * np.cos(self._estimated_yaw)

        return self._estimated_pos.copy()

    def _handle_episode_end(self, done: bool, truncated: bool, final_reward: float):
        """
        Handle episode end - apply safety backfill if crash occurred.

        Safety logic:
        - done=True, truncated=False, reward > 0: SUCCESS (all safe)
        - done=True, truncated=False, reward < 0: CRASH (backfill unsafe)
        - truncated=True: TIMEOUT (all safe, just ran out of time)
        """
        is_crash = done and not truncated and final_reward < 0

        if is_crash:
            # Backfill last N steps as unsafe
            num_steps = len(self.safety_history)
            backfill_start = max(0, num_steps - self.SAFETY_BACKFILL_STEPS)

            for i in range(backfill_start, num_steps):
                # Gradual unsafe marking: more unsafe closer to crash
                # steps_to_crash = num_steps - i
                # Could use: safety = steps_to_crash / SAFETY_BACKFILL_STEPS
                # For now, binary: all backfilled steps are unsafe
                self.safety_history[i] = 0.0

        # Store in global trajectory store for training loop access
        trajectory_data = {
            'positions': np.array(self.position_history),
            'yaws': np.array(self.yaw_history),
            'speeds': np.array(self.speed_history),
        }
        safety_mask = np.array(self.safety_history)

        self._store.store_trajectory(self.env_id, trajectory_data, safety_mask)

    def set_predicted_waypoints(self, waypoints: np.ndarray):
        """Called by callback/policy to set the current predicted waypoints."""
        self.last_predicted_waypoints = waypoints.copy()

        # Forward to Unity visualization
        if hasattr(self.env, 'set_waypoints'):
            self.env.set_waypoints(waypoints)

    def get_waypoint_history(self) -> list:
        """Get history of waypoint predictions for analysis."""
        return self.waypoint_prediction_history.copy()

    def get_current_trajectory_for_loss(self) -> Optional[Dict]:
        """
        Get trajectory data formatted for WaypointLoss computation.

        Returns dict with:
        - positions: (N, 3) array of world positions
        - yaws: (N,) array of yaw angles
        - safety: (N,) array of safety flags (1.0=safe, 0.0=unsafe)
        """
        if len(self.position_history) < 2:
            return None

        return {
            'positions': np.array(self.position_history),
            'yaws': np.array(self.yaw_history),
            'safety': np.array(self.safety_history),
            'current_idx': len(self.position_history) - 1
        }
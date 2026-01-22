#!/usr/bin/env python3
"""
train_vision_lanes.py - Training with Vision-Based Lane Detection
==================================================================

This approach is TRULY PASSIVE VISUAL because:
1. Lane info comes from camera image (HSV or UFLD)
2. No privileged simulation data (no RouteProgress needed!)
3. Same perception works on real robot
4. The policy learns from what it can actually see

Key difference from train_steering_only.py:
- OLD: lat_err, hdg_err from privileged RouteProgress (simulation-only)
- NEW: lane_offset, heading from camera-based detection (transfers to real)

The reward signal is now based on what the camera sees, not what
the simulation knows. This is essential for sim-to-real transfer.
"""
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from live_unity_env import LiveUnityEnv


# =============================================================================
# Telemetry Indices (from Unity)
# =============================================================================
class Idx:
    TURN_BIAS = 0  # Navigation command: -1, 0, +1
    RESERVED = 1  # Always 0
    GOAL_DIST = 2  # Distance to goal (for progress tracking)
    SPEED = 3  # Current speed m/s
    YAW_RATE = 4  # Current yaw rate rad/s
    LAST_STEER = 5  # Last steering command
    LAST_THR = 6  # Last throttle command
    LAST_BRK = 7  # Last brake command
    # NOTE: We ignore indices 8-11 (lat_err, hdg_err, kappa, ds)
    # because they require RouteProgress which we don't want to use


# =============================================================================
# Vision-Based Lane Detection
# =============================================================================
class VisionLaneDetector:
    """
    Detects lane lines from camera image using classical CV.

    For more robust detection, replace with UFLD model.
    This HSV-based approach works for simulation with colored lane markings.
    """

    def __init__(
            self,
            # Yellow lane detection (common in simulation)
            yellow_lower=(15, 50, 50),
            yellow_upper=(35, 255, 255),
            # White lane detection
            white_lower=(0, 0, 160),
            white_upper=(180, 40, 255),
            # Road (gray/black) detection for finding road edges
            road_lower=(0, 0, 30),
            road_upper=(180, 50, 120),
            # Image geometry
            image_height=128,
            image_width=128,
    ):
        self.yellow_lower = np.array(yellow_lower)
        self.yellow_upper = np.array(yellow_upper)
        self.white_lower = np.array(white_lower)
        self.white_upper = np.array(white_upper)
        self.road_lower = np.array(road_lower)
        self.road_upper = np.array(road_upper)
        self.h = image_height
        self.w = image_width

    def detect(self, image: np.ndarray) -> dict:
        """
        Detect lanes and compute lane-relative metrics.

        Returns:
            dict with:
                - lane_offset: normalized offset from lane center (-1 to 1)
                - heading_error: heading relative to lane (radians)
                - curvature: estimated lane curvature
                - confidence: detection confidence (0 to 1)
                - left_x: x-coordinate of left lane at bottom
                - right_x: x-coordinate of right lane at bottom
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Detect lane markings
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        lane_mask = cv2.bitwise_or(yellow_mask, white_mask)

        # Focus on bottom portion (where lanes are visible)
        roi_top = int(0.5 * self.h)
        roi = lane_mask[roi_top:, :]

        # Split into left and right halves
        mid_x = self.w // 2
        left_roi = roi[:, :mid_x]
        right_roi = roi[:, mid_x:]

        # Find lane line positions at multiple heights
        left_points = []
        right_points = []

        for row_offset in range(0, roi.shape[0], 8):  # Sample every 8 pixels
            row_y = roi_top + row_offset

            # Left lane
            left_cols = np.where(lane_mask[row_y, :mid_x] > 0)[0]
            if len(left_cols) > 0:
                # Take rightmost point in left region (closest to lane edge)
                left_x = left_cols[-1]
                left_points.append((left_x, row_y))

            # Right lane
            right_cols = np.where(lane_mask[row_y, mid_x:] > 0)[0]
            if len(right_cols) > 0:
                # Take leftmost point in right region (closest to lane edge)
                right_x = right_cols[0] + mid_x
                right_points.append((right_x, row_y))

        # Compute lane metrics
        if len(left_points) >= 2 and len(right_points) >= 2:
            # Lane positions at bottom of image
            left_x_bottom = np.mean([p[0] for p in left_points[-3:]])
            right_x_bottom = np.mean([p[0] for p in right_points[-3:]])

            # Lane center
            lane_center = (left_x_bottom + right_x_bottom) / 2
            image_center = self.w / 2

            # Normalized offset (-1 to 1, positive = car is right of center)
            lane_width = max(right_x_bottom - left_x_bottom, 1)
            lane_offset = (image_center - lane_center) / (lane_width / 2)
            lane_offset = np.clip(lane_offset, -2.0, 2.0)

            # Heading from lane line slopes
            # Fit lines to lane points
            if len(left_points) >= 3:
                left_xs = [p[0] for p in left_points]
                left_ys = [p[1] for p in left_points]
                left_slope = np.polyfit(left_ys, left_xs, 1)[0]
            else:
                left_slope = 0

            if len(right_points) >= 3:
                right_xs = [p[0] for p in right_points]
                right_ys = [p[1] for p in right_points]
                right_slope = np.polyfit(right_ys, right_xs, 1)[0]
            else:
                right_slope = 0

            # Average slope indicates heading error
            avg_slope = (left_slope + right_slope) / 2
            heading_error = np.arctan(avg_slope)

            # Curvature from lane width change
            if len(left_points) >= 4 and len(right_points) >= 4:
                top_width = abs(right_points[0][0] - left_points[0][0])
                bottom_width = abs(right_points[-1][0] - left_points[-1][0])
                curvature = (bottom_width - top_width) / max(bottom_width, 1)
            else:
                curvature = 0

            confidence = min(len(left_points), len(right_points)) / 8.0
            confidence = min(confidence, 1.0)

            return {
                'lane_offset': float(lane_offset),
                'heading_error': float(heading_error),
                'curvature': float(curvature),
                'confidence': float(confidence),
                'left_x': float(left_x_bottom),
                'right_x': float(right_x_bottom),
                'lane_detected': True,
            }

        else:
            # No lanes detected - return defaults
            return {
                'lane_offset': 0.0,
                'heading_error': 0.0,
                'curvature': 0.0,
                'confidence': 0.0,
                'left_x': 0.0,
                'right_x': self.w,
                'lane_detected': False,
            }


# =============================================================================
# WRAPPER: Vision Lane Reward
# =============================================================================
class VisionLaneWrapper(gym.Wrapper):
    """
    Uses vision-based lane detection for reward shaping.

    This is TRULY PASSIVE VISUAL because:
    - Lane info comes from camera image
    - No privileged simulation data needed
    - Same approach works on real robot
    """

    def __init__(
            self,
            env: gym.Env,
            detector: VisionLaneDetector = None,
            # Reward parameters
            centered_bonus: float = 0.3,
            offset_penalty_scale: float = 0.2,
            heading_bonus: float = 0.2,
            heading_penalty_scale: float = 0.15,
            no_lane_penalty: float = 0.1,
            # Steering target (for dense reward)
            use_steering_target: bool = True,
            steering_p_gain: float = 0.4,
            steering_d_gain: float = 0.3,
            steering_reward_scale: float = 0.4,
            # Add lane info to observation
            add_to_obs: bool = True,
    ):
        super().__init__(env)

        self.detector = detector or VisionLaneDetector()
        self.centered_bonus = centered_bonus
        self.offset_penalty_scale = offset_penalty_scale
        self.heading_bonus = heading_bonus
        self.heading_penalty_scale = heading_penalty_scale
        self.no_lane_penalty = no_lane_penalty
        self.use_steering_target = use_steering_target
        self.steering_p_gain = steering_p_gain
        self.steering_d_gain = steering_d_gain
        self.steering_reward_scale = steering_reward_scale
        self.add_to_obs = add_to_obs

        # Modify observation space to include lane info
        if add_to_obs:
            orig_vec = env.observation_space["vec"]
            # Add: lane_offset, heading_error, curvature, confidence
            new_dim = orig_vec.shape[0] + 4
            self.observation_space = spaces.Dict({
                "image": env.observation_space["image"],
                "vec": spaces.Box(-np.inf, np.inf, shape=(new_dim,), dtype=np.float32)
            })

        self._last_lane_info = None
        self._last_action = None

    def _augment_obs(self, obs, lane_info):
        """Add lane info to observation vector."""
        if self.add_to_obs and isinstance(obs, dict) and 'vec' in obs:
            lane_vec = np.array([
                lane_info['lane_offset'],
                lane_info['heading_error'],
                lane_info['curvature'],
                lane_info['confidence'],
            ], dtype=np.float32)
            obs['vec'] = np.concatenate([obs['vec'], lane_vec])
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Detect lanes in initial frame
        if isinstance(obs, dict) and 'image' in obs:
            self._last_lane_info = self.detector.detect(obs['image'])
        else:
            self._last_lane_info = {'lane_offset': 0, 'heading_error': 0,
                                    'curvature': 0, 'confidence': 0, 'lane_detected': False}

        obs = self._augment_obs(obs, self._last_lane_info)
        info['lane_info'] = self._last_lane_info

        self._last_action = None
        return obs, info

    def step(self, action):
        self._last_action = action
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Detect lanes
        if isinstance(obs, dict) and 'image' in obs:
            lane_info = self.detector.detect(obs['image'])
        else:
            lane_info = {'lane_offset': 0, 'heading_error': 0,
                         'curvature': 0, 'confidence': 0, 'lane_detected': False}

        # Compute lane-based reward
        lane_reward = 0.0

        if lane_info['lane_detected'] and lane_info['confidence'] > 0.3:
            offset = abs(lane_info['lane_offset'])
            heading = abs(lane_info['heading_error'])
            conf = lane_info['confidence']

            # Centered bonus
            if offset < 0.2:
                lane_reward += self.centered_bonus * conf
                info['lane_status'] = 'centered'
            else:
                lane_reward -= self.offset_penalty_scale * offset * conf
                info['lane_status'] = 'off_center'

            # Heading alignment bonus
            if heading < 0.1:  # ~6 degrees
                lane_reward += self.heading_bonus * conf
            else:
                lane_reward -= self.heading_penalty_scale * heading * conf

            # Steering target reward
            if self.use_steering_target and self._last_action is not None:
                # Compute target steering from lane position
                target_steer = (
                        self.steering_p_gain * lane_info['lane_offset'] +
                        self.steering_d_gain * lane_info['heading_error']
                )
                target_steer = np.clip(target_steer, -1.0, 1.0)

                # Get actual steering
                actual_steer = float(self._last_action[0])
                steer_error = abs(actual_steer - target_steer)

                # Reward for matching target
                steer_reward = self.steering_reward_scale * np.exp(-3.0 * steer_error)
                lane_reward += steer_reward

                info['target_steer'] = target_steer
                info['actual_steer'] = actual_steer
                info['steer_error'] = steer_error
        else:
            # No lane detected
            lane_reward -= self.no_lane_penalty
            info['lane_status'] = 'no_lane'

        reward += lane_reward

        # Augment observation
        obs = self._augment_obs(obs, lane_info)

        # Store info
        info['lane_info'] = lane_info
        info['lane_reward'] = lane_reward
        self._last_lane_info = lane_info

        return obs, reward, terminated, truncated, info


# =============================================================================
# WRAPPER: Curvature-Adaptive Speed (unchanged)
# =============================================================================
class CurvatureSpeedWrapper(gym.Wrapper):
    """Convert 1D steering to 3D action with curvature-adaptive throttle."""

    def __init__(
            self,
            env: gym.Env,
            base_throttle: float = 0.42,
            min_throttle: float = 0.28,
            max_throttle: float = 0.52,
            steer_speed_factor: float = 0.25,
    ):
        super().__init__(env)
        self.base_throttle = base_throttle
        self.min_throttle = min_throttle
        self.max_throttle = max_throttle
        self.steer_speed_factor = steer_speed_factor

        # 1D action space
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

    def step(self, action):
        steer = float(action[0]) if hasattr(action, '__len__') else float(action)
        steer = np.clip(steer, -1.0, 1.0)

        # Adaptive throttle
        throttle = self.base_throttle - self.steer_speed_factor * abs(steer)
        throttle = np.clip(throttle, self.min_throttle, self.max_throttle)

        full_action = np.array([steer, throttle, 0.0], dtype=np.float32)
        return self.env.step(full_action)


# =============================================================================
# WRAPPER: Progress Reward (based on speed and goal distance)
# =============================================================================
class ProgressWrapper(gym.Wrapper):
    """Reward for making progress toward goal."""

    def __init__(self, env: gym.Env, speed_bonus: float = 0.5, progress_bonus: float = 1.0):
        super().__init__(env)
        self.speed_bonus = speed_bonus
        self.progress_bonus = progress_bonus
        self._last_goal_dist = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(obs, dict) and 'vec' in obs:
            self._last_goal_dist = obs['vec'][Idx.GOAL_DIST]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if isinstance(obs, dict) and 'vec' in obs:
            speed = obs['vec'][Idx.SPEED]
            goal_dist = obs['vec'][Idx.GOAL_DIST]

            # Speed bonus
            if speed > 0.2:
                reward += self.speed_bonus * min(speed / 2.0, 1.0)

            # Progress toward goal
            if self._last_goal_dist is not None:
                progress = self._last_goal_dist - goal_dist
                if progress > 0:
                    reward += self.progress_bonus * progress

            self._last_goal_dist = goal_dist
            info['speed'] = speed
            info['goal_dist'] = goal_dist

        return obs, reward, terminated, truncated, info


# =============================================================================
# WRAPPER: Safety (collision detection, off-road)
# =============================================================================
class SafetyWrapper(gym.Wrapper):
    """Terminate on collision or stuck."""

    def __init__(
            self,
            env: gym.Env,
            crash_penalty: float = -5.0,
            stuck_threshold: float = 0.1,
            stuck_steps: int = 30,
    ):
        super().__init__(env)
        self.crash_penalty = crash_penalty
        self.stuck_threshold = stuck_threshold
        self.stuck_steps = stuck_steps
        self._stuck_counter = 0
        self._last_speed = 0.0

    def reset(self, **kwargs):
        self._stuck_counter = 0
        self._last_speed = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if isinstance(obs, dict) and 'vec' in obs:
            speed = obs['vec'][Idx.SPEED]

            # Detect stuck
            if speed < self.stuck_threshold:
                self._stuck_counter += 1
                if self._stuck_counter > self.stuck_steps:
                    terminated = True
                    reward += self.crash_penalty
                    info['terminated_reason'] = 'stuck'
            else:
                self._stuck_counter = 0

            # Detect sudden stop (collision)
            if self._last_speed > 0.5 and speed < 0.1:
                terminated = True
                reward += self.crash_penalty
                info['terminated_reason'] = 'collision'

            self._last_speed = speed

        return obs, reward, terminated, truncated, info


# =============================================================================
# Feature Extractor (simpler, for 1D action)
# =============================================================================
class VisionFusionExtractor(BaseFeaturesExtractor):
    """CNN + physics fusion for steering policy."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # Get dimensions
        vec_dim = observation_space["vec"].shape[0]
        cnn_dim = 128
        total_dim = cnn_dim + vec_dim

        super().__init__(observation_space, features_dim=total_dim)

        # CNN for images
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size
        with torch.no_grad():
            sample = torch.zeros(1, 3, 128, 128)
            cnn_out = self.cnn(sample).shape[1]

        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_out, cnn_dim),
            nn.ReLU(),
        )

        self.fusion_norm = nn.LayerNorm(total_dim)

    def forward(self, obs):
        # Image: (B, H, W, C) -> (B, C, H, W)
        img = obs["image"]
        if img.dim() == 4 and img.shape[-1] == 3:
            img = img.permute(0, 3, 1, 2)
        img = img.float() / 255.0

        # CNN forward
        cnn_out = self.cnn(img)
        visual_feats = self.cnn_fc(cnn_out)

        # Physics
        phys_feats = obs["vec"].float()

        # Fuse
        fused = torch.cat([visual_feats, phys_feats], dim=1)
        return self.fusion_norm(fused)


# =============================================================================
# Callbacks
# =============================================================================
class TrainingCallback(BaseCallback):
    """Training progress logger."""

    def __init__(self, log_freq=50, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._ep_rewards = []
        self._ep_lengths = []
        self._lane_detections = []

    def _on_step(self):
        # Track lane detection rate
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                if 'lane_info' in info:
                    self._lane_detections.append(info['lane_info'].get('lane_detected', False))

        # Episode end
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    ep = self.locals['infos'][i].get('episode', {})
                    if 'r' in ep:
                        self._ep_rewards.append(ep['r'])
                        self._ep_lengths.append(ep['l'])

        # Periodic log
        if self.n_calls % self.log_freq == 0 and self._ep_rewards:
            recent_rew = np.mean(self._ep_rewards[-10:])
            recent_len = np.mean(self._ep_lengths[-10:])
            lane_rate = np.mean(self._lane_detections[-100:]) if self._lane_detections else 0

            print(f"\n{'=' * 60}")
            print(f"Step {self.num_timesteps:,} | Episodes: {len(self._ep_rewards)}")
            print(f"  Reward (last 10): {recent_rew:.1f}")
            print(f"  Length (last 10): {recent_len:.0f}")
            print(f"  Lane detection rate: {100 * lane_rate:.1f}%")
            print(f"{'=' * 60}")

        return True


# =============================================================================
# Environment Factory
# =============================================================================
def make_env(host, port, img_size, max_steps, verbose=False):
    def _init():
        # Base environment
        env = LiveUnityEnv(
            host=host, port=port,
            img_width=img_size[0], img_height=img_size[1],
            max_steps=max_steps, verbose=verbose,
        )

        # Safety
        env = SafetyWrapper(env)

        # Progress reward
        env = ProgressWrapper(env)

        # Vision-based lane detection (THE KEY DIFFERENCE!)
        env = VisionLaneWrapper(
            env,
            detector=VisionLaneDetector(),
            centered_bonus=0.3,
            heading_bonus=0.2,
            use_steering_target=True,
            add_to_obs=True,
        )

        # 1D steering with adaptive throttle
        env = CurvatureSpeedWrapper(env)

        return Monitor(env)

    return _init


# =============================================================================
# Main
# =============================================================================
def train(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"vision_lanes_{timestamp}"
    model_dir = Path(args.model_dir) / run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  VISION-BASED LANE FOLLOWING (Truly Passive Visual)")
    print("=" * 70)
    print(f"\nModel directory: {model_dir}")
    print()
    print("KEY DIFFERENCE FROM PREVIOUS APPROACH:")
    print("  OLD: Used RouteProgress lat_err/hdg_err (simulation-only, doesn't transfer)")
    print("  NEW: Uses camera-based lane detection (works on real robot!)")
    print()
    print("LANE DETECTION:")
    print("  - HSV color filtering for lane markings")
    print("  - Computes: lane_offset, heading_error from IMAGE")
    print("  - Same detection can run on real robot's camera")
    print()
    print("NO ROUTE SETUP REQUIRED:")
    print("  - Works on any road with visible lane markings")
    print("  - No manual RoutePath needed")
    print("=" * 70)

    # Create env
    env = DummyVecEnv([make_env(
        args.host, args.port,
        (args.img_size[0], args.img_size[1]),
        args.max_steps,
        args.verbose > 1,
    )])

    # Policy
    policy_kwargs = {
        "lstm_hidden_size": 128,
        "n_lstm_layers": 1,
        "enable_critic_lstm": True,
        "features_extractor_class": VisionFusionExtractor,
        "net_arch": dict(pi=[64], vf=[64]),
    }

    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=5,
        gamma=0.99,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.tensorboard_log,
        verbose=args.verbose,
    )

    callbacks = [
        CheckpointCallback(save_freq=10000, save_path=str(model_dir / "ckpt"), name_prefix="vis"),
        TrainingCallback(log_freq=50),
    ]

    print(f"\nStarting training for {args.timesteps:,} steps...")
    print("Watch Unity window to see lane detection in action!")
    print()

    try:
        model.learn(args.timesteps, callback=callbacks, tb_log_name=run_name, progress_bar=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted, saving...")

    model.save(str(model_dir / "final_model"))
    print(f"\n✓ Model saved to {model_dir}")

    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--timesteps', type=int, default=200000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--n_steps', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=5556)
    p.add_argument('--img_size', type=int, nargs=2, default=[128, 128])
    p.add_argument('--max_steps', type=int, default=500)
    p.add_argument('--model_dir', default='./models')
    p.add_argument('--tensorboard_log', default='./tb_logs')
    p.add_argument('--verbose', type=int, default=1)

    train(p.parse_args())
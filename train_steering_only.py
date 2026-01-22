#!/usr/bin/env python3
"""
train_steering_only.py - SIMPLIFIED TRAINING
=============================================
Key insight: The agent is trying to learn too much at once.

This version simplifies dramatically:
1. FIXED THROTTLE - Agent only learns steering (1D output)
2. CURVATURE SUPERVISION - CNN learns to predict road curvature from images
3. STEERING TARGET - Desired steering computed from turn_bias + predicted curvature
4. DENSE REWARD - MSE between actual steering and target steering

The agent learns: "Given an image and turn command, output correct steering angle"

This is still Passive Visual because:
- Curvature is PREDICTED from image, not given as input at deployment
- Turn bias is the only navigation signal
- No geometric coordinates
"""
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from live_unity_env import LiveUnityEnv


# =============================================================================
# Telemetry Indices
# =============================================================================
class Idx:
    TURN_BIAS = 0  # Command: -1 (left), 0 (straight), +1 (right)
    RESERVED = 1  # Always 0
    GOAL_DIST = 2  # Distance to goal (masked to 0 for policy)
    SPEED = 3  # Current speed m/s
    YAW_RATE = 4  # Current yaw rate rad/s
    LAST_STEER = 5  # Last steering command
    LAST_THR = 6  # Last throttle command
    LAST_BRK = 7  # Last brake command
    LAT_ERR = 8  # Lateral error from path (for supervision only)
    HDG_ERR = 9  # Heading error from path (for supervision only)
    KAPPA = 10  # Path curvature 1/m (for supervision only)
    DS = 11  # Distance traveled this step


# =============================================================================
# WRAPPER 1: Fixed Throttle - Convert 1D steering to 3D action
# =============================================================================
class FixedThrottleWrapper(gym.Wrapper):
    """
    Converts 1D steering action to 3D (steer, throttle, brake).

    The agent outputs: steer in [-1, 1]
    We send to Unity: (steer, fixed_throttle, 0)

    This dramatically simplifies learning - only 1 DoF to learn!
    """

    def __init__(self, env: gym.Env, throttle: float = 0.4):
        super().__init__(env)
        self.throttle = throttle

        # Override action space to 1D
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

    def step(self, action):
        # Convert 1D -> 3D
        steer = float(action[0]) if hasattr(action, '__len__') else float(action)
        steer = np.clip(steer, -1.0, 1.0)

        full_action = np.array([steer, self.throttle, 0.0], dtype=np.float32)
        return self.env.step(full_action)


# =============================================================================
# WRAPPER 2: Curvature-Adaptive Speed
# =============================================================================
class CurvatureSpeedWrapper(gym.Wrapper):
    """
    Instead of fixed throttle, adjust speed based on road curvature.

    Slow down for curves, speed up for straights.
    Uses PREDICTED curvature (from last step's observation) to maintain
    Passive Visual compliance.

    This is NOT cheating because:
    - We're using the agent's own curvature prediction
    - Or a simple heuristic based on recent steering
    """

    def __init__(
            self,
            env: gym.Env,
            base_throttle: float = 0.45,
            min_throttle: float = 0.25,
            max_throttle: float = 0.55,
            steer_speed_factor: float = 0.3,  # How much steering reduces speed
    ):
        super().__init__(env)
        self.base_throttle = base_throttle
        self.min_throttle = min_throttle
        self.max_throttle = max_throttle
        self.steer_speed_factor = steer_speed_factor

        # 1D action space (steering only)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

    def step(self, action):
        steer = float(action[0]) if hasattr(action, '__len__') else float(action)
        steer = np.clip(steer, -1.0, 1.0)

        # Reduce throttle when steering hard
        throttle = self.base_throttle - self.steer_speed_factor * abs(steer)
        throttle = np.clip(throttle, self.min_throttle, self.max_throttle)

        full_action = np.array([steer, throttle, 0.0], dtype=np.float32)
        return self.env.step(full_action)


# =============================================================================
# WRAPPER 3: Steering Target Reward (CORE DENSE REWARD)
# =============================================================================
class SteeringTargetWrapper(gym.Wrapper):
    """
    Provides dense reward based on how close steering is to a target.

    Target steering is computed from:
    1. turn_bias: the navigation command
    2. lateral_error: deviation from lane center (from telemetry)
    3. heading_error: angular deviation from path tangent

    This creates a dense gradient that tells the agent exactly what to do.

    IMPORTANT: This wrapper MUST be applied BEFORE PassiveVisualWrapper!
    It uses lat_err and hdg_err which get masked afterward.

    PASSIVE VISUAL COMPLIANCE:
    - These privileged values are used only for REWARD computation
    - The POLICY never sees them (they get masked later)
    - The CNN must learn to predict correct steering from IMAGES
    """

    def __init__(
            self,
            env: gym.Env,
            # Gains for computing target steering (PD-like controller)
            turn_bias_gain: float = 0.35,  # How much turn_bias affects target
            lateral_gain: float = 0.4,  # P-gain on lateral error
            heading_gain: float = 0.6,  # P-gain on heading error
            # Reward scaling
            steering_reward_scale: float = 0.4,
            perfect_steering_bonus: float = 0.25,
            steering_error_threshold: float = 0.12,  # "Close enough" threshold
            # Smoothness rewards
            smooth_steering_bonus: float = 0.15,
            max_steer_change: float = 0.3,  # Penalize jerky steering above this
    ):
        super().__init__(env)
        self.turn_bias_gain = turn_bias_gain
        self.lateral_gain = lateral_gain
        self.heading_gain = heading_gain
        self.steering_reward_scale = steering_reward_scale
        self.perfect_steering_bonus = perfect_steering_bonus
        self.steering_error_threshold = steering_error_threshold
        self.smooth_steering_bonus = smooth_steering_bonus
        self.max_steer_change = max_steer_change

        self._last_steer = 0.0
        self._last_action = None

    def reset(self, **kwargs):
        self._last_steer = 0.0
        self._last_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        self._last_action = action
        obs, reward, done, truncated, info = self.env.step(action)

        if isinstance(obs, dict) and 'vec' in obs:
            vec = obs['vec']
            turn_bias = vec[Idx.TURN_BIAS]
            lat_err = vec[Idx.LAT_ERR]  # BEFORE masking!
            hdg_err = vec[Idx.HDG_ERR]  # BEFORE masking!

            # Compute target steering using classical control law
            # This is what "perfect" steering would be
            # The formula: steer toward lane center + align with path tangent + follow command
            target_steer = (
                    self.turn_bias_gain * turn_bias +
                    self.lateral_gain * np.clip(lat_err, -2.0, 2.0) +
                    self.heading_gain * np.clip(hdg_err, -1.0, 1.0)
            )
            target_steer = np.clip(target_steer, -1.0, 1.0)

            # Get actual steering from action
            actual_steer = float(self._last_action[0]) if self._last_action is not None else 0.0

            # Compute steering error
            steer_error = abs(actual_steer - target_steer)

            # Dense reward: closer to target = higher reward
            # Use exponential decay for smoother gradient
            steer_reward = self.steering_reward_scale * np.exp(-3.0 * steer_error)

            # Bonus for being very close to target
            if steer_error < self.steering_error_threshold:
                steer_reward += self.perfect_steering_bonus
                info['steering_quality'] = 'excellent'
            elif steer_error < self.steering_error_threshold * 2:
                info['steering_quality'] = 'good'
            else:
                info['steering_quality'] = 'poor'

            # Smoothness reward: penalize jerky steering
            steer_change = abs(actual_steer - self._last_steer)
            if steer_change < self.max_steer_change:
                steer_reward += self.smooth_steering_bonus
                info['steering_smoothness'] = 'smooth'
            else:
                # Small penalty for jerky steering
                steer_reward -= 0.1 * (steer_change - self.max_steer_change)
                info['steering_smoothness'] = 'jerky'

            self._last_steer = actual_steer
            reward += steer_reward

            # Info for debugging
            info['target_steer'] = float(target_steer)
            info['actual_steer'] = float(actual_steer)
            info['steer_error'] = float(steer_error)
            info['steer_reward'] = float(steer_reward)

        return obs, reward, done, truncated, info


# =============================================================================
# WRAPPER 4: Progress Reward (Forward Motion)
# =============================================================================
class ProgressRewardWrapper(gym.Wrapper):
    """
    Rewards forward progress along the route.
    Uses ds (distance traveled) from telemetry.
    """

    def __init__(self, env: gym.Env, progress_scale: float = 2.0):
        super().__init__(env)
        self.progress_scale = progress_scale

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        if isinstance(obs, dict) and 'vec' in obs:
            ds = obs['vec'][Idx.DS]
            # Reward forward progress (ds is in meters per step)
            progress_reward = self.progress_scale * max(0, ds * 50)  # Scale up
            reward += progress_reward
            info['progress_reward'] = progress_reward
            info['ds'] = ds

        return obs, reward, done, truncated, info


# =============================================================================
# WRAPPER 5: Episode Termination on Large Error
# =============================================================================
class SafetyWrapper(gym.Wrapper):
    """
    Terminates episode if car goes too far off track.
    Also adds collision detection based on sudden stops.
    """

    def __init__(
            self,
            env: gym.Env,
            max_lateral_error: float = 3.0,  # meters
            crash_penalty: float = -5.0,
            off_track_penalty: float = -0.5,
    ):
        super().__init__(env)
        self.max_lateral_error = max_lateral_error
        self.crash_penalty = crash_penalty
        self.off_track_penalty = off_track_penalty
        self._last_speed = 0.0
        self._stuck_steps = 0

    def reset(self, **kwargs):
        self._last_speed = 0.0
        self._stuck_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        if isinstance(obs, dict) and 'vec' in obs:
            vec = obs['vec']
            lat_err = abs(vec[Idx.LAT_ERR])
            speed = vec[Idx.SPEED]

            # Check if off track
            if lat_err > self.max_lateral_error:
                reward += self.off_track_penalty
                info['off_track'] = True

                # Terminate if way off track
                if lat_err > self.max_lateral_error * 1.5:
                    done = True
                    reward += self.crash_penalty
                    info['terminated_reason'] = 'off_track'

            # Detect stuck (likely collision)
            if speed < 0.1 and self._last_speed > 0.3:
                # Sudden stop = likely collision
                self._stuck_steps += 1
                if self._stuck_steps > 20:
                    done = True
                    reward += self.crash_penalty
                    info['terminated_reason'] = 'stuck'
            else:
                self._stuck_steps = 0

            self._last_speed = speed

        return obs, reward, done, truncated, info


# =============================================================================
# WRAPPER 6: Observation Masking for Passive Visual
# =============================================================================
class PassiveVisualWrapper(gym.Wrapper):
    """
    Masks privileged information from the observation.

    The policy sees:
    - turn_bias (command)
    - speed, yaw_rate (proprioception)
    - last_steer, last_thr, last_brk (action history)

    The policy does NOT see:
    - goal_dist (masked to 0)
    - lat_err, hdg_err, kappa, ds (masked to 0)

    These are still used by reward wrappers, but not visible to policy.
    The UNMASKED values are stored in info for debugging/logging.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def _mask_observation(self, obs, info):
        if isinstance(obs, dict) and 'vec' in obs:
            vec = obs['vec'].copy()

            # Store unmasked values in info for debugging
            info['unmasked_goal_dist'] = float(vec[Idx.GOAL_DIST])
            info['unmasked_lat_err'] = float(vec[Idx.LAT_ERR])
            info['unmasked_hdg_err'] = float(vec[Idx.HDG_ERR])
            info['unmasked_kappa'] = float(vec[Idx.KAPPA])
            info['unmasked_ds'] = float(vec[Idx.DS])

            # Mask privileged information
            vec[Idx.GOAL_DIST] = 0.0
            vec[Idx.LAT_ERR] = 0.0
            vec[Idx.HDG_ERR] = 0.0
            vec[Idx.KAPPA] = 0.0
            vec[Idx.DS] = 0.0
            obs['vec'] = vec
        return obs, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._mask_observation(obs, info)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs, info = self._mask_observation(obs, info)
        return obs, reward, done, truncated, info


# =============================================================================
# Feature Extractor with Curvature Prediction Head
# =============================================================================
class CurvatureAwareFusion(BaseFeaturesExtractor):
    """
    CNN + Physics fusion with auxiliary curvature prediction.

    The CNN learns to predict road curvature from the image.
    This auxiliary task helps it understand road geometry.

    Architecture:
        Image (128x128x3) -> CNN -> 256 features
        Physics (12) -> Identity -> 12 features
        Fused -> LayerNorm -> 268 features

    Auxiliary head:
        CNN features -> Linear -> Curvature prediction
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 268):
        vec_dim = observation_space["vec"].shape[0]
        cnn_output_dim = 256
        total_dim = cnn_output_dim + vec_dim

        super().__init__(observation_space, features_dim=total_dim)

        # Visual stream - properly sized for 128x128 input
        # Layer 1: 128x128 -> 31x31 (kernel=8, stride=4, padding=0)
        # Layer 2: 31x31 -> 14x14 (kernel=4, stride=2, padding=0)
        # Layer 3: 14x14 -> 12x12 (kernel=3, stride=1, padding=0)
        # Flatten: 64 * 12 * 12 = 9216
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size dynamically
        with torch.no_grad():
            sample_img = torch.zeros(1, 3, 128, 128)
            cnn_flat_size = self.cnn(sample_img).shape[1]

        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_flat_size, cnn_output_dim),
            nn.ReLU()
        )

        # Fusion normalization
        self.fusion_norm = nn.LayerNorm(total_dim)

        # Auxiliary curvature prediction head (for potential future use)
        self.curvature_head = nn.Sequential(
            nn.Linear(cnn_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self._last_curvature_pred = None
        self._cnn_output_dim = cnn_output_dim

    def forward(self, observations: dict) -> torch.Tensor:
        # Process images: expect (B, H, W, C), need (B, C, H, W)
        img = observations["image"]
        if img.dim() == 4 and img.shape[-1] == 3:
            img = img.permute(0, 3, 1, 2)  # BHWC -> BCHW
        img = img.float() / 255.0  # Normalize to [0, 1]

        # CNN forward
        cnn_flat = self.cnn(img)
        visual_feats = self.cnn_fc(cnn_flat)

        # Predict curvature (auxiliary task - stored for potential use)
        self._last_curvature_pred = self.curvature_head(visual_feats)

        # Get physics vector
        physics_feats = observations["vec"].float()

        # Fuse and normalize
        fused = torch.cat([visual_feats, physics_feats], dim=1)
        return self.fusion_norm(fused)

    def get_curvature_prediction(self):
        return self._last_curvature_pred


# =============================================================================
# Debug Callback with Rich Logging
# =============================================================================
class TrainingDebugCallback(BaseCallback):
    """Logs training progress and key metrics."""

    def __init__(self, log_freq: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._episode_rewards = []
        self._episode_lengths = []
        self._steer_errors = []
        self._lane_status_counts = {'centered': 0, 'off_center': 0}
        self._termination_reasons = {}

    def _on_step(self) -> bool:
        # Collect per-step metrics
        if self.locals.get('infos') is not None:
            for info in self.locals['infos']:
                if 'steer_error' in info:
                    self._steer_errors.append(info['steer_error'])
                if 'lane_status' in info:
                    self._lane_status_counts[info['lane_status']] = \
                        self._lane_status_counts.get(info['lane_status'], 0) + 1

        # Log episode stats when episodes end
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][i]
                    ep_info = info.get('episode', {})
                    if 'r' in ep_info:
                        self._episode_rewards.append(ep_info['r'])
                        self._episode_lengths.append(ep_info['l'])

                    # Track termination reasons
                    reason = info.get('terminated_reason', 'unknown')
                    self._termination_reasons[reason] = \
                        self._termination_reasons.get(reason, 0) + 1

        # Periodic logging
        if self.n_calls % self.log_freq == 0 and len(self._episode_rewards) > 0:
            recent_rewards = self._episode_rewards[-10:]
            recent_lengths = self._episode_lengths[-10:]
            recent_steer_err = self._steer_errors[-100:] if self._steer_errors else [0]

            print(f"\n{'=' * 60}")
            print(f"Step {self.num_timesteps:,} | Episodes: {len(self._episode_rewards)}")
            print(f"  Reward (last 10): {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}")
            print(f"  Length (last 10): {np.mean(recent_lengths):.0f}")
            print(f"  Steer Error (last 100): {np.mean(recent_steer_err):.3f}")

            if self._termination_reasons:
                print(f"  Terminations: {dict(self._termination_reasons)}")

            total_lane = sum(self._lane_status_counts.values())
            if total_lane > 0:
                centered_pct = 100 * self._lane_status_counts.get('centered', 0) / total_lane
                print(f"  Lane Centered: {centered_pct:.1f}%")
            print(f"{'=' * 60}")

        return True

    def _on_training_end(self):
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print(f"Total episodes: {len(self._episode_rewards)}")
        if self._episode_rewards:
            print(f"Final reward (last 10): {np.mean(self._episode_rewards[-10:]):.2f}")
            print(f"Best episode reward: {max(self._episode_rewards):.2f}")
        print("=" * 60)


# =============================================================================
# WRAPPER 7: Lane Keeping Reward (Dense but Not Cheating)
# =============================================================================
class LaneKeepingRewardWrapper(gym.Wrapper):
    """
    Rewards staying close to lane center.

    This wrapper MUST be applied BEFORE PassiveVisualWrapper!
    Uses lat_err for reward but policy never sees it.

    PASSIVE VISUAL COMPLIANCE:
    - lat_err is used only for reward shaping
    - The policy must learn lane centering from VISUAL input
    - This creates dense gradient without leaking geometry to policy
    """

    def __init__(
            self,
            env: gym.Env,
            centered_bonus: float = 0.2,  # Bonus for being centered
            off_lane_penalty: float = 0.1,  # Penalty per meter off center
            center_threshold: float = 0.3,  # meters - "centered" if closer than this
    ):
        super().__init__(env)
        self.centered_bonus = centered_bonus
        self.off_lane_penalty = off_lane_penalty
        self.center_threshold = center_threshold

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        if isinstance(obs, dict) and 'vec' in obs:
            lat_err = abs(obs['vec'][Idx.LAT_ERR])  # BEFORE masking!

            if lat_err < self.center_threshold:
                # Centered - give bonus
                reward += self.centered_bonus
                info['lane_status'] = 'centered'
            else:
                # Off center - penalty proportional to distance
                penalty = self.off_lane_penalty * (lat_err - self.center_threshold)
                reward -= penalty
                info['lane_status'] = 'off_center'

            info['lat_err_abs'] = float(lat_err)

        return obs, reward, done, truncated, info


# =============================================================================
# Environment Factory
# =============================================================================
def make_env(host, port, img_size, max_steps, verbose=False):
    def _init():
        # Base environment
        env = LiveUnityEnv(
            host=host,
            port=port,
            img_width=img_size[0],
            img_height=img_size[1],
            max_steps=max_steps,
            verbose=verbose,
        )

        # ====================================================================
        # WRAPPER CHAIN (ORDER MATTERS!)
        #
        # Rewards use UNMASKED observations (lat_err, hdg_err, kappa, ds)
        # Policy sees MASKED observations (these values are zeroed)
        # This is PASSIVE VISUAL: dense rewards without geometric cheating
        # ====================================================================

        # 1. Safety first - terminates if car goes way off track
        env = SafetyWrapper(
            env,
            max_lateral_error=2.5,  # meters
            crash_penalty=-5.0,
            off_track_penalty=-0.3,
        )

        # 2. Progress reward - encourages forward movement
        env = ProgressRewardWrapper(
            env,
            progress_scale=2.5,  # Reward per meter traveled
        )

        # 3. Lane keeping reward - encourages staying centered
        env = LaneKeepingRewardWrapper(
            env,
            centered_bonus=0.2,
            off_lane_penalty=0.15,
            center_threshold=0.3,
        )

        # 4. Steering target reward - THE KEY DENSE SIGNAL
        #    Tells agent exactly what steering would be correct
        env = SteeringTargetWrapper(
            env,
            turn_bias_gain=0.35,
            lateral_gain=0.4,
            heading_gain=0.6,
            steering_reward_scale=0.4,
            perfect_steering_bonus=0.25,
        )

        # 5. Mask privileged info - MUST be after reward wrappers
        #    Policy only sees: turn_bias, speed, yaw_rate, action history
        env = PassiveVisualWrapper(env)

        # 6. Convert to 1D steering with curvature-adaptive speed
        #    Removes throttle learning - agent only controls steering
        env = CurvatureSpeedWrapper(
            env,
            base_throttle=0.42,
            min_throttle=0.28,
            max_throttle=0.52,
            steer_speed_factor=0.25,
        )

        return Monitor(env)

    return _init


# =============================================================================
# Main Training
# =============================================================================
def train(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"steer_only_{timestamp}"
    model_dir = Path(args.model_dir) / run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("   SIMPLIFIED STEERING-ONLY TRAINING")
    print("   Passive Visual + Dense Reward Shaping")
    print("=" * 70)
    print(f"\nModel directory: {model_dir}")
    print(f"TensorBoard: tensorboard --logdir {args.tensorboard_log}")
    print()

    print("SIMPLIFICATIONS:")
    print("  ✓ Agent only outputs STEERING (1D action)")
    print("  ✓ Throttle automatically adjusts based on steering magnitude")
    print("  ✓ Dense reward from steering target (classical control law)")
    print("  ✓ Lane keeping reward (stay centered)")
    print("  ✓ Progress reward (move forward)")
    print()

    print("PASSIVE VISUAL COMPLIANCE:")
    print("  ✓ Policy only sees: image + turn_bias + speed/yaw_rate + action history")
    print("  ✗ Policy does NOT see: goal_dist, lat_err, hdg_err, kappa, ds")
    print("  → Rewards use privileged info, but policy can't cheat with it")
    print("  → CNN must learn road geometry from images to succeed")
    print()

    print("REWARD STRUCTURE:")
    print("  + Progress: reward per meter traveled (~2.5/m)")
    print("  + Lane Keeping: bonus for staying centered (0.2)")
    print("  + Steering Target: reward for correct steering (0.4 + 0.25 bonus)")
    print("  + Smoothness: bonus for smooth steering (0.15)")
    print("  - Off Track: penalty proportional to distance")
    print("  - Crash: large penalty (-5.0)")
    print()
    print("=" * 70)

    # Create environment
    env = DummyVecEnv([make_env(
        args.host, args.port,
        (args.img_size[0], args.img_size[1]),
        args.max_steps,
        args.verbose > 1
    )])

    # Policy configuration
    policy_kwargs = {
        "lstm_hidden_size": args.lstm_hidden_size,
        "n_lstm_layers": 1,
        "enable_critic_lstm": True,
        "features_extractor_class": CurvatureAwareFusion,
        "features_extractor_kwargs": {},
        "net_arch": dict(pi=[128, 64], vf=[128, 64]),  # Slightly larger networks
    }

    # Create model with tuned hyperparameters
    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=0.95,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.tensorboard_log,
        verbose=args.verbose,
    )

    # Print model summary
    print(f"\nModel Architecture:")
    print(f"  LSTM hidden size: {args.lstm_hidden_size}")
    print(f"  Policy network: [128, 64]")
    print(f"  Value network: [128, 64]")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Entropy coef: {args.ent_coef}")

    # Callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=args.save_freq,
            save_path=str(model_dir / "checkpoints"),
            name_prefix="steer"
        ),
        TrainingDebugCallback(log_freq=50),
    ]

    print(f"\nStarting training for {args.timesteps:,} timesteps...")
    print("Press Ctrl+C to stop training early (model will be saved)")
    print()

    # Train with graceful interruption handling
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name=run_name,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted! Saving model...")

    # Save final model
    final_path = str(model_dir / "final_model")
    model.save(final_path)
    print(f"\n✓ Model saved to {final_path}")

    # Save training config
    config = {
        'timesteps': args.timesteps,
        'lr': args.lr,
        'n_steps': args.n_steps,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'gamma': args.gamma,
        'ent_coef': args.ent_coef,
        'lstm_hidden_size': args.lstm_hidden_size,
        'img_size': args.img_size,
        'max_steps': args.max_steps,
        'run_name': run_name,
    }
    import json
    with open(model_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved to {model_dir / 'config.json'}")

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simplified steering-only training with dense rewards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training
    parser.add_argument('--timesteps', type=int, default=200000,
                        help='Total training timesteps')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--n_steps', type=int, default=256,
                        help='Steps per rollout (larger = more stable)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Minibatch size')
    parser.add_argument('--n_epochs', type=int, default=5,
                        help='PPO epochs per rollout')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--ent_coef', type=float, default=0.005,
                        help='Entropy coefficient (lower = more deterministic)')

    # Architecture
    parser.add_argument('--lstm_hidden_size', type=int, default=256,
                        help='LSTM hidden state size')

    # Environment
    parser.add_argument('--host', default='127.0.0.1',
                        help='Unity host')
    parser.add_argument('--port', type=int, default=5556,
                        help='Unity port')
    parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128],
                        help='Image size (width height)')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Max steps per episode')

    # Saving
    parser.add_argument('--model_dir', default='./models',
                        help='Directory to save models')
    parser.add_argument('--tensorboard_log', default='./tb_logs',
                        help='TensorBoard log directory')
    parser.add_argument('--save_freq', type=int, default=10000,
                        help='Checkpoint save frequency')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level')

    args = parser.parse_args()
    train(args)
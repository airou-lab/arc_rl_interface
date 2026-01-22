#!/usr/bin/env python3
"""
train_policy_RNN_v3.py - NON-CHEATING FIXES
============================================
The previous versions didn't work because the untrained control head
outputs random actions that cancel each other out (throttle + brake).

This version fixes the learning problem WITHOUT geometric cheating:

1. ACTION CONSTRAINTS:
   - Throttle and brake are mutually exclusive (can't apply both)
   - Minimum throttle floor during exploration phase
   - Brake only engages when throttle is explicitly low

2. EXPLORATION THAT FAVORS MOVEMENT:
   - Bias action noise toward positive throttle
   - Gradually reduce bias as policy improves

3. SIMPLIFIED REWARD:
   - Strong reward for forward speed
   - Penalty for being stationary
   - Alignment with turn_bias (the ONLY navigation signal allowed)

4. NO GEOMETRIC INFORMATION:
   - Waypoints are used for self-supervised learning only
   - Control decisions based on visual features + turn_bias
   - No Pure Pursuit, no goal coordinates

The key insight: The policy needs to learn that:
   turn_bias = -1 → steer left
   turn_bias = +1 → steer right
   turn_bias = 0 → go straight

This is the ONLY navigation signal in Passive Visual Navigation.
"""
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import explained_variance

from unity_dense_env import UnityDenseEnv, DenseRewardConfig
from action_repeat_wrapper import ActionRepeatWrapper
from wrappers.waypoint_tracking_wrapper import WaypointTrackingWrapper, get_trajectory_store
from policies.hierarchical_policy import HierarchicalPathPlanningPolicy
from policies.fusion_policy import FusionFeaturesExtractor


# ============================================================================
# FIX 1: Action Constraint Wrapper - Throttle/Brake Mutual Exclusion
# ============================================================================
class ActionConstraintWrapper(gym.Wrapper):
    """
    Constrains actions to prevent self-sabotage by untrained policy.

    Key constraints:
    1. Throttle and brake cannot both be high simultaneously
    2. Minimum throttle floor to ensure movement during exploration
    3. Brake only engages when throttle is explicitly commanded low

    This is NOT cheating because:
    - We're not providing any geometric information
    - We're just preventing physically nonsensical actions
    - Real cars can't accelerate and brake simultaneously either
    """

    def __init__(
            self,
            env: gym.Env,
            min_throttle: float = 0.3,  # Minimum throttle during exploration
            throttle_decay_steps: int = 50000,  # Steps to decay min throttle
            final_min_throttle: float = 0.0,  # Final minimum (full policy control)
            brake_throttle_threshold: float = 0.2,  # Only brake if throttle below this
    ):
        super().__init__(env)
        self.initial_min_throttle = min_throttle
        self.final_min_throttle = final_min_throttle
        self.throttle_decay_steps = throttle_decay_steps
        self.brake_throttle_threshold = brake_throttle_threshold
        self._total_steps = 0

    def step(self, action):
        self._total_steps += 1
        action = np.asarray(action, dtype=np.float32).copy()

        if len(action) >= 3:
            steer, throttle, brake = action[0], action[1], action[2]

            # 1. Compute current minimum throttle (decays over training)
            progress = min(1.0, self._total_steps / self.throttle_decay_steps)
            min_throttle = self.initial_min_throttle + \
                           (self.final_min_throttle - self.initial_min_throttle) * progress

            # 2. Apply minimum throttle
            throttle = max(throttle, min_throttle)

            # 3. Mutual exclusion: if throttle is significant, suppress brake
            if throttle > self.brake_throttle_threshold:
                brake = 0.0

            action[0] = steer
            action[1] = throttle
            action[2] = brake

        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# ============================================================================
# FIX 2: Turn Bias Alignment Reward
# ============================================================================
class TurnBiasAlignmentWrapper(gym.Wrapper):
    """
    Rewards steering that aligns with the turn_bias navigation command.

    This is the CORE of Passive Visual Navigation:
    - turn_bias is the ONLY navigation signal (like GPS "turn left/right")
    - The car must learn to interpret this command visually
    - Reward alignment between commanded direction and actual steering

    This is NOT cheating because:
    - turn_bias is a high-level command, not geometric coordinates
    - It's equivalent to a human driver hearing "turn left at next intersection"
    - The car still needs to visually determine WHEN and HOW MUCH to turn
    """

    def __init__(
            self,
            env: gym.Env,
            alignment_bonus: float = 0.3,  # Reward for correct steering direction
            misalignment_penalty: float = 0.1,  # Penalty for wrong direction
            command_threshold: float = 0.3,  # |turn_bias| > this triggers alignment check
    ):
        super().__init__(env)
        self.alignment_bonus = alignment_bonus
        self.misalignment_penalty = misalignment_penalty
        self.command_threshold = command_threshold

        self.IDX_TURN_BIAS = 0
        self._last_action = None

    def step(self, action):
        self._last_action = np.asarray(action, dtype=np.float32).copy()
        obs, reward, done, truncated, info = self.env.step(action)

        # Get turn_bias from observation
        if isinstance(obs, dict) and 'vec' in obs:
            turn_bias = float(obs['vec'][self.IDX_TURN_BIAS])
        else:
            turn_bias = 0.0

        # Get steering from action
        steer = float(self._last_action[0]) if self._last_action is not None else 0.0

        # Check alignment only when turn command is strong
        if abs(turn_bias) > self.command_threshold:
            # Correct alignment: signs match and magnitude is reasonable
            if np.sign(steer) == np.sign(turn_bias) and abs(steer) > 0.1:
                reward += self.alignment_bonus
                info['turn_alignment'] = 'correct'
            # Wrong direction
            elif np.sign(steer) == -np.sign(turn_bias) and abs(steer) > 0.1:
                reward -= self.misalignment_penalty
                info['turn_alignment'] = 'wrong'
            else:
                info['turn_alignment'] = 'weak'
        else:
            # No strong turn command - reward going relatively straight
            if abs(steer) < 0.3:
                reward += self.alignment_bonus * 0.5
                info['turn_alignment'] = 'straight_ok'
            else:
                info['turn_alignment'] = 'straight_excess_steer'

        info['turn_bias'] = turn_bias
        info['actual_steer'] = steer

        return obs, reward, done, truncated, info


# ============================================================================
# FIX 3: Speed-Based Reward (encourages movement)
# ============================================================================
class SpeedRewardWrapper(gym.Wrapper):
    """
    Rewards maintaining forward speed.

    This is NOT cheating because:
    - Speed is vehicle state, not navigation information
    - All vehicles need to maintain speed to navigate
    - We're rewarding movement, not direction
    """

    def __init__(
            self,
            env: gym.Env,
            target_speed: float = 1.5,
            speed_bonus: float = 0.4,
            stationary_penalty: float = 0.3,
            min_speed_threshold: float = 0.2,
    ):
        super().__init__(env)
        self.target_speed = target_speed
        self.speed_bonus = speed_bonus
        self.stationary_penalty = stationary_penalty
        self.min_speed_threshold = min_speed_threshold
        self.IDX_SPEED = 3

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        if isinstance(obs, dict) and 'vec' in obs:
            speed = abs(obs['vec'][self.IDX_SPEED])
        else:
            speed = 0.0

        if speed >= self.min_speed_threshold:
            speed_ratio = min(speed / self.target_speed, 1.0)
            reward += self.speed_bonus * speed_ratio
            info['speed_status'] = 'moving'
        else:
            reward -= self.stationary_penalty
            info['speed_status'] = 'stationary'

        info['current_speed'] = speed
        return obs, reward, done, truncated, info


# ============================================================================
# FIX 4: Goal Heading Reward (NOT CHEATING - like compass bearing to goal)
# ============================================================================
class GoalHeadingWrapper(gym.Wrapper):
    """
    Rewards heading roughly toward the goal using hdg_err.

    This is NOT cheating because:
    - hdg_err in Goal Fallback mode = angle between car forward and goal direction
    - This is like having a compass bearing to destination (GPS provides this)
    - It does NOT tell the car where the road/lane is
    - Lane following must be learned from VISUAL input

    We explicitly DO NOT use lat_err because:
    - Goal Fallback lat_err = deviation from spawn→goal straight line
    - On curved roads, correct lane following INCREASES this error
    - Using it would penalize correct driving behavior

    The CNN must learn actual lane following from camera images.
    This wrapper only provides weak "heading toward goal" guidance.
    """

    def __init__(
            self,
            env: gym.Env,
            hdg_err_scale: float = 0.15,  # Weak penalty for facing wrong way
            aligned_bonus: float = 0.1,  # Small bonus for facing goal
            hdg_threshold: float = 0.3,  # "Aligned" if |hdg_err| < ~17 degrees
            large_hdg_threshold: float = 1.5,  # ~86 deg - only penalize if WAY off
    ):
        super().__init__(env)
        self.hdg_err_scale = hdg_err_scale
        self.aligned_bonus = aligned_bonus
        self.hdg_threshold = hdg_threshold
        self.large_hdg_threshold = large_hdg_threshold

        # Telemetry indices
        self.IDX_HDG_ERR = 9

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # Extract heading error from telemetry
        if isinstance(obs, dict) and 'vec' in obs:
            hdg_err = obs['vec'][self.IDX_HDG_ERR]  # Radians from goal direction
        else:
            hdg_err = 0.0

        # Only penalize if heading is significantly wrong (> ~86 degrees off)
        # This allows following curved roads without penalty
        if abs(hdg_err) > self.large_hdg_threshold:
            # Car is facing very wrong direction (maybe backwards?)
            hdg_penalty = self.hdg_err_scale * (abs(hdg_err) - self.large_hdg_threshold)
            reward -= hdg_penalty
            info['heading_status'] = 'wrong_way'
        elif abs(hdg_err) < self.hdg_threshold:
            # Car is roughly facing goal - small bonus
            reward += self.aligned_bonus
            info['heading_status'] = 'aligned'
        else:
            # Car is somewhat off but acceptable for curved roads
            info['heading_status'] = 'acceptable'

        info['hdg_err'] = float(hdg_err)

        return obs, reward, done, truncated, info


# ============================================================================
# FIX 5: Collision Penalty (NOT CHEATING - real robots have bumper sensors)
# ============================================================================
class CollisionPenaltyWrapper(gym.Wrapper):
    """
    Penalizes collisions and terminates episode on crash.

    This is NOT cheating because:
    - Real robots have bumper sensors, lidar proximity detection, IMU shock
    - We're learning from consequences (crash = bad), not privileged path info

    Detection methods:
    1. Unity sends 'collision' flag in info dict (preferred)
    2. Episode ends early with done=True + low speed (stuck against wall)
    3. Speed drops suddenly (impact deceleration)
    """

    def __init__(
            self,
            env: gym.Env,
            collision_penalty: float = 5.0,  # Large penalty for crashing
            stuck_penalty: float = 2.0,  # Penalty for being stuck
            stuck_speed_threshold: float = 0.1,  # Speed below this = potentially stuck
            stuck_steps_threshold: int = 30,  # Steps at low speed to count as stuck
            impact_decel_threshold: float = 1.0,  # Sudden speed drop = impact
    ):
        super().__init__(env)
        self.collision_penalty = collision_penalty
        self.stuck_penalty = stuck_penalty
        self.stuck_speed_threshold = stuck_speed_threshold
        self.stuck_steps_threshold = stuck_steps_threshold
        self.impact_decel_threshold = impact_decel_threshold

        self.IDX_SPEED = 3
        self._last_speed = 0.0
        self._stuck_counter = 0
        self._episode_collisions = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_speed = 0.0
        self._stuck_counter = 0
        self._episode_collisions = 0
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # Get current speed
        if isinstance(obs, dict) and 'vec' in obs:
            speed = abs(obs['vec'][self.IDX_SPEED])
        else:
            speed = 0.0

        collision_detected = False
        collision_type = None

        # Method 1: Check if Unity sent collision flag
        if info.get('collision', False):
            collision_detected = True
            collision_type = 'unity_collision'

        # Method 2: Check for sudden deceleration (impact)
        speed_drop = self._last_speed - speed
        if speed_drop > self.impact_decel_threshold and self._last_speed > 0.5:
            collision_detected = True
            collision_type = 'impact_decel'

        # Method 3: Check for stuck against wall (low speed + throttle)
        throttle = action[1] if len(action) > 1 else 0.0
        if speed < self.stuck_speed_threshold and throttle > 0.3:
            self._stuck_counter += 1
            if self._stuck_counter >= self.stuck_steps_threshold:
                collision_detected = True
                collision_type = 'stuck'
                reward -= self.stuck_penalty
        else:
            self._stuck_counter = max(0, self._stuck_counter - 1)

        # Apply collision penalty
        if collision_detected:
            self._episode_collisions += 1
            reward -= self.collision_penalty
            info['collision_detected'] = True
            info['collision_type'] = collision_type

            # Terminate episode on collision
            done = True
            info['terminal_reason'] = f'collision_{collision_type}'

        self._last_speed = speed
        info['episode_collisions'] = self._episode_collisions
        info['stuck_counter'] = self._stuck_counter

        return obs, reward, done, truncated, info
        super().__init__(env)
        self.target_speed = target_speed
        self.speed_bonus = speed_bonus
        self.stationary_penalty = stationary_penalty
        self.min_speed_threshold = min_speed_threshold

        self.IDX_SPEED = 3

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        if isinstance(obs, dict) and 'vec' in obs:
            speed = float(obs['vec'][self.IDX_SPEED])
        else:
            speed = 0.0

        if speed >= self.min_speed_threshold:
            # Bonus proportional to speed (capped at target)
            speed_ratio = min(speed / self.target_speed, 1.0)
            reward += self.speed_bonus * speed_ratio
            info['speed_reward'] = self.speed_bonus * speed_ratio
        else:
            # Penalty for being stationary
            reward -= self.stationary_penalty
            info['speed_reward'] = -self.stationary_penalty

        info['current_speed'] = speed

        return obs, reward, done, truncated, info


# ============================================================================
# FIX 4: Persistent Trajectory Store
# ============================================================================
class PersistentTrajectoryWrapper(gym.Wrapper):
    """Maintains trajectory data across episodes for stable aux loss."""

    def __init__(self, env, env_id: int = 0, max_length: int = 2000):
        super().__init__(env)
        self.env_id = env_id
        self.max_length = max_length

        self.positions = []
        self.yaws = []
        self.speeds = []
        self.safety = []

        self._pos = np.zeros(3, dtype=np.float32)
        self._yaw = 0.0

        self.IDX_SPEED = 3
        self.IDX_YAW_RATE = 4
        self.IDX_DS = 11

        self._store = get_trajectory_store()

    def reset(self, **kwargs):
        # Don't clear - keep rolling history
        self._pos = np.zeros(3, dtype=np.float32)
        self._yaw = 0.0

        ret = self.env.reset(**kwargs)
        obs = ret[0] if isinstance(ret, tuple) else ret
        self._record(obs, safe=True)

        return ret

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        is_crash = done and not truncated and reward < -1.0
        self._record(obs, safe=not is_crash)

        if is_crash:
            # Mark last 25 steps as unsafe
            for i in range(max(0, len(self.safety) - 25), len(self.safety)):
                self.safety[i] = 0.0

        self._update_store()

        return obs, reward, done, truncated, info

    def _record(self, obs, safe: bool):
        if isinstance(obs, dict) and 'vec' in obs:
            vec = obs['vec']
            speed = float(vec[self.IDX_SPEED])
            yaw_rate = float(vec[self.IDX_YAW_RATE])
            ds = float(vec[self.IDX_DS])
        else:
            speed, yaw_rate, ds = 0.0, 0.0, 0.0

        dt = 0.02
        self._yaw += yaw_rate * dt

        if abs(ds) > 0.001:
            self._pos[0] += ds * np.sin(self._yaw)
            self._pos[2] += ds * np.cos(self._yaw)

        self.positions.append(self._pos.copy())
        self.yaws.append(self._yaw)
        self.speeds.append(speed)
        self.safety.append(1.0 if safe else 0.0)

        # Trim
        if len(self.positions) > self.max_length:
            self.positions = self.positions[-self.max_length:]
            self.yaws = self.yaws[-self.max_length:]
            self.speeds = self.speeds[-self.max_length:]
            self.safety = self.safety[-self.max_length:]

    def _update_store(self):
        if len(self.positions) < 50:
            return

        self._store.store_trajectory(
            self.env_id,
            {
                'positions': np.array(self.positions),
                'yaws': np.array(self.yaws),
                'speeds': np.array(self.speeds),
            },
            np.array(self.safety)
        )


# ============================================================================
# Debug Callback
# ============================================================================
class DebugCallback(BaseCallback):
    """Logs diagnostics including heading and collision metrics."""

    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.speeds = []
        self.alignments = {'correct': 0, 'wrong': 0, 'weak': 0, 'straight_ok': 0}
        self.hdg_errs = []
        self.heading_status = {'aligned': 0, 'acceptable': 0, 'wrong_way': 0}
        self.collisions = 0
        self.stuck_events = 0

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])

        for info in infos:
            if 'current_speed' in info:
                self.speeds.append(info['current_speed'])
            if 'turn_alignment' in info:
                key = info['turn_alignment']
                if key in self.alignments:
                    self.alignments[key] += 1
            # Heading metrics
            if 'hdg_err' in info:
                self.hdg_errs.append(abs(info['hdg_err']))
            if 'heading_status' in info:
                status = info['heading_status']
                if status in self.heading_status:
                    self.heading_status[status] += 1
            # Collision metrics
            if info.get('collision_detected', False):
                self.collisions += 1
                collision_type = info.get('collision_type', 'unknown')
                if collision_type == 'stuck':
                    self.stuck_events += 1

        if self.num_timesteps % self.log_freq == 0 and len(self.speeds) > 0:
            self.logger.record("debug/mean_speed", np.mean(self.speeds[-100:]))
            self.logger.record("debug/max_speed", np.max(self.speeds[-100:]) if self.speeds else 0)

            total = sum(self.alignments.values())
            if total > 0:
                self.logger.record("debug/alignment_correct_pct",
                                   100 * self.alignments['correct'] / total)

            # Heading metrics
            if len(self.hdg_errs) > 0:
                self.logger.record("debug/mean_hdg_err", np.mean(self.hdg_errs[-100:]))

            hdg_total = sum(self.heading_status.values())
            if hdg_total > 0:
                self.logger.record("debug/heading_aligned_pct",
                                   100 * self.heading_status['aligned'] / hdg_total)

            # Collision metrics
            self.logger.record("debug/total_collisions", self.collisions)
            self.logger.record("debug/stuck_events", self.stuck_events)

            # Trajectory length
            store = get_trajectory_store()
            traj = store.get_trajectory(0)
            if traj:
                self.logger.record("debug/trajectory_length", len(traj.get('positions', [])))

        return True


# ============================================================================
# Waypoint Visualization Callback
# ============================================================================
class WaypointVisualizationCallback(BaseCallback):
    """
    Callback that:
    1. Logs waypoint metrics to TensorBoard
    2. Sends waypoints to Unity for visualization
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._send_counter = 0

    def _on_step(self) -> bool:
        if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'last_waypoints'):
            wp = self.model.policy.last_waypoints

            if wp is not None and isinstance(wp, torch.Tensor):
                wp_np = wp.detach().cpu().numpy()

                # Log metrics to TensorBoard
                if wp_np.ndim == 3:
                    self.logger.record("waypoints/mean_forward_m", np.mean(wp_np[:, :, 1]))
                    self.logger.record("waypoints/mean_lateral_m", np.mean(np.abs(wp_np[:, :, 0])))

                # ============================================
                # SEND WAYPOINTS TO UNITY FOR VISUALIZATION
                # ============================================
                self._send_counter += 1
                if self._send_counter % 2 == 0:  # Send every other step to reduce overhead
                    try:
                        if len(self.training_env.envs) > 0:
                            env = self.training_env.envs[0]

                            # Unwrap to find the environment with set_waypoints
                            while hasattr(env, 'env'):
                                if hasattr(env, 'set_waypoints'):
                                    break
                                env = env.env

                            # Send waypoints to Unity
                            if hasattr(env, 'set_waypoints'):
                                if wp_np.ndim == 3:
                                    env.set_waypoints(wp_np[0])  # First batch element
                                else:
                                    env.set_waypoints(wp_np)
                    except Exception as e:
                        pass  # Don't crash training if visualization fails

        return True


# ============================================================================
# Clamped Waypoint Aux Loss
# ============================================================================
class ClampedWaypointAuxLoss:
    """Waypoint auxiliary loss with clamping."""

    def __init__(self, num_waypoints=5, max_loss=30.0):
        self.num_waypoints = num_waypoints
        self.steps_per_wp = 25  # 0.5s at 50Hz
        self.max_loss = max_loss

    def compute(self, predicted_wp, traj_data, indices, device):
        positions = traj_data.get('positions')
        yaws = traj_data.get('yaws')

        if positions is None or len(positions) < 50:
            return torch.tensor(0.0, device=device)

        positions = torch.tensor(positions, dtype=torch.float32, device=device)
        yaws = torch.tensor(yaws, dtype=torch.float32, device=device)

        batch_size = predicted_wp.shape[0]
        traj_len = positions.shape[0]

        losses = []

        for b in range(batch_size):
            idx = int(indices[b].item())
            idx = min(max(0, idx), traj_len - 1)

            car_pos = positions[idx, :2]
            car_yaw = yaws[idx]

            for w in range(self.num_waypoints):
                future_idx = idx + (w + 1) * self.steps_per_wp
                if future_idx >= traj_len:
                    continue

                # Get future position in world frame
                future_pos = positions[future_idx, :2]

                # Transform to vehicle frame
                dx = future_pos[0] - car_pos[0]
                dy = future_pos[1] - car_pos[1]

                cos_yaw = torch.cos(car_yaw)
                sin_yaw = torch.sin(car_yaw)

                local_x = dx * cos_yaw + dy * sin_yaw
                local_y = -dx * sin_yaw + dy * cos_yaw

                # Compute loss
                pred = predicted_wp[b, w]
                target = torch.stack([local_x, local_y])

                loss = F.smooth_l1_loss(pred, target)
                losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device)

        total_loss = torch.stack(losses).mean()
        return torch.clamp(total_loss, max=self.max_loss)


# ============================================================================
# Custom PPO
# ============================================================================
class CustomHierarchicalPPO(RecurrentPPO):
    def __init__(self, *args, waypoint_loss_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.waypoint_loss_weight = waypoint_loss_weight
        self.waypoint_criterion = ClampedWaypointAuxLoss(
            num_waypoints=kwargs.get('policy_kwargs', {}).get('num_waypoints', 5),
            max_loss=30.0
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_trajectory_store', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, pg_losses, value_losses = [], [], []
        waypoint_losses, clip_fractions = [], []

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions,
                    rollout_data.lstm_states, rollout_data.episode_starts
                )

                aux_loss = self._compute_aux_loss(
                    rollout_data.observations,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts
                )
                waypoint_losses.append(aux_loss.item())

                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss = -torch.min(
                    advantages * ratio,
                    advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                ).mean()

                pg_losses.append(policy_loss.item())
                clip_fractions.append(
                    torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                )

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                entropy_loss = -torch.mean(entropy) if entropy is not None else -torch.mean(-log_prob)
                entropy_losses.append(entropy_loss.item())

                loss = (policy_loss + self.ent_coef * entropy_loss +
                        self.vf_coef * value_loss + self.waypoint_loss_weight * aux_loss)

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

        self._n_updates += self.n_epochs

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten()
        )

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/waypoint_aux_loss", np.mean(waypoint_losses))
        self.logger.record("train/n_updates", self._n_updates)

    def _compute_aux_loss(self, observations, lstm_states, episode_starts):
        store = get_trajectory_store()
        traj = store.get_trajectory(0)

        if traj is None or len(traj.get('positions', [])) < 50:
            return torch.tensor(0.0, device=self.device)

        features = self.policy.extract_features(observations)
        latent_pi, _ = self.policy._process_sequence(
            features, lstm_states.pi, episode_starts, self.policy.lstm_actor
        )

        if self.policy.mlp_extractor is not None:
            latent_pi = self.policy.mlp_extractor.forward_actor(latent_pi)

        obs_vec = observations['vec'] if isinstance(observations, dict) else observations
        predicted_wp = self.policy._compute_waypoints(latent_pi, obs_vec)

        batch_size = obs_vec.shape[0]
        traj_len = len(traj['positions'])

        # Random indices from recent trajectory
        max_idx = max(50, traj_len - 50)
        indices = torch.randint(0, max_idx, (batch_size,), device=self.device)

        return self.waypoint_criterion.compute(predicted_wp, traj, indices, self.device)


# ============================================================================
# Environment Factory
# ============================================================================
def make_env(host, port, img_size, max_steps, repeat, reward_cfg, verbose, env_id=0):
    def _init():
        env = UnityDenseEnv(
            host, port, img_size[0], img_size[1], max_steps,
            reward_cfg, verbose
        )

        # ENABLE WAYPOINT VISUALIZATION
        env.enable_waypoint_visualization()

        if repeat > 1:
            env = ActionRepeatWrapper(env, repeat)

        # FIX 1: Action constraints (throttle/brake mutual exclusion)
        env = ActionConstraintWrapper(
            env,
            min_throttle=0.35,  # Start with 35% throttle floor
            throttle_decay_steps=100000,  # Decay over 100k steps
            final_min_throttle=0.0,  # Eventually full policy control
        )

        # FIX 2: Turn bias alignment reward
        env = TurnBiasAlignmentWrapper(
            env,
            alignment_bonus=0.3,
            misalignment_penalty=0.1,
        )

        # FIX 3: Speed reward
        env = SpeedRewardWrapper(
            env,
            target_speed=1.5,
            speed_bonus=0.4,
            stationary_penalty=0.3,
        )

        # FIX 4: Goal heading reward (NOT lane following - that must come from vision)
        # Only penalizes heading very wrong direction, allows curved road following
        env = GoalHeadingWrapper(
            env,
            hdg_err_scale=0.15,  # Weak penalty for facing wrong way
            aligned_bonus=0.1,  # Small bonus for facing goal
            large_hdg_threshold=1.5,  # Only penalize if > ~86 degrees off
        )

        # FIX 5: Collision penalty (NOT CHEATING - real robots have bumper sensors)
        # Terminates episode on crash, teaches car to avoid obstacles
        env = CollisionPenaltyWrapper(
            env,
            collision_penalty=5.0,  # Large penalty for crashing
            stuck_penalty=2.0,  # Penalty for stuck against wall
            stuck_steps_threshold=30,  # Steps to count as stuck
        )

        # FIX 6: Persistent trajectory (for self-supervised waypoint learning)
        env = PersistentTrajectoryWrapper(env, env_id=env_id)

        # Standard waypoint wrapper (auxiliary learning, not control)
        env = WaypointTrackingWrapper(env, env_id=env_id)

        return Monitor(env)

    return _init


# ============================================================================
# Main
# ============================================================================
def train(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"hppo_v3_{timestamp}"
    model_dir = Path(args.model_dir) / run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.tensorboard_log)
    log_dir.mkdir(parents=True, exist_ok=True)

    reward_cfg = DenseRewardConfig(stuck_pen_k=0.5, goal_approach_k=0.0)

    env = DummyVecEnv([make_env(
        args.host, args.port, tuple(args.img_size),
        args.max_steps, args.repeat, reward_cfg, args.verbose > 1
    )])

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

    callbacks = [
        CheckpointCallback(save_freq=args.save_freq,
                           save_path=str(model_dir / "checkpoints"),
                           name_prefix="hppo"),
        WaypointVisualizationCallback(),
        DebugCallback(log_freq=100),
    ]

    print("=" * 60)
    print("Hierarchical PPO V3 - PASSIVE VISUAL NAVIGATION")
    print("=" * 60)
    print(f"Model dir: {model_dir}")
    print()
    print("NON-CHEATING REWARD SHAPING:")
    print("  1. ActionConstraintWrapper - throttle/brake mutual exclusion")
    print("  2. TurnBiasAlignmentWrapper - reward steering alignment with nav commands")
    print("  3. SpeedRewardWrapper - encourage movement")
    print("  4. GoalHeadingWrapper - weak penalty for facing wrong way (like compass)")
    print("  5. CollisionPenaltyWrapper - crash penalty + episode termination")
    print("  6. PersistentTrajectoryWrapper - stable aux loss")
    print()
    print("PASSIVE VISUAL PRINCIPLE:")
    print("  - NO geometric goal coordinates (masked)")
    print("  - NO lane centerline (would be cheating)")
    print("  - Navigation intent from turn_bias command only")
    print("  - Heading guidance like compass bearing (not path following)")
    print("  - Collision detection like bumper sensor (not privileged)")
    print("  - LANE FOLLOWING MUST BE LEARNED FROM VISUAL INPUT (CNN)")
    print("=" * 60)

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
    parser = argparse.ArgumentParser()

    parser.add_argument('--timesteps', type=int, default=200000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--n_steps', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--ent_coef', type=float, default=0.005)

    parser.add_argument('--lstm_hidden_size', type=int, default=256)
    parser.add_argument('--n_lstm_layers', type=int, default=1)

    parser.add_argument('--num_waypoints', type=int, default=5)
    parser.add_argument('--waypoint_horizon', type=float, default=2.5)
    parser.add_argument('--waypoint_loss_weight', type=float, default=0.1)
    parser.add_argument('--repulsion_weight', type=float, default=2.0)

    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5556)
    parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128])
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--repeat', type=int, default=1)

    parser.add_argument('--model_dir', default='./models')
    parser.add_argument('--tensorboard_log', default='./tb')
    parser.add_argument('--save_freq', type=int, default=25000)
    parser.add_argument('--verbose', type=int, default=1)

    train(parser.parse_args())
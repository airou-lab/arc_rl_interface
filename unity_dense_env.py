"""
Unity Dense Reward Environment Wrapper
Protocol v2.0 - Passive Visual Navigation
"""

from __future__ import annotations
import dataclasses
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from live_unity_env import LiveUnityEnv


# Telemetry indices (Protocol v2.0)
class TelemetryIdx:
    TURN_BIAS = 0
    RESERVED = 1
    GOAL_DIST = 2
    SPEED = 3
    YAW_RATE = 4
    LAST_STEER = 5
    LAST_THR = 6
    LAST_BRK = 7
    LAT_ERR = 8
    HDG_ERR = 9
    KAPPA = 10
    DS = 11


@dataclasses.dataclass
class DenseRewardConfig:
    forward_k: float = 2.0
    goal_approach_k: float = 0.0  # UPDATED: Set to 0.0 to prevent curve conflict
    goal_heading_k: float = 0.3
    spin_k: float = 0.1
    steer_k: float = 0.03
    steer_change_k: float = 0.05
    brake_k: float = 0.02
    lat_err_k: float = 0.3
    hdg_err_k: float = 0.2
    goal_reached_bonus: float = 5.0
    goal_reached_threshold: float = 3.0
    stuck_pen_k: float = 0.5
    stuck_speed_eps: float = 0.15
    stuck_thr_eps: float = 0.1


class UnityDenseEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
            self,
            host="127.0.0.1",
            port=5556,
            img_width=128,
            img_height=128,
            max_steps=500,
            reward_cfg=None,
            verbose=False,
            step_delay=0.0
    ):
        super().__init__()
        self.unity = LiveUnityEnv(
            host, port, img_width, img_height, max_steps,
            verbose=verbose, step_delay=step_delay
        )
        if hasattr(self.unity, 'enable_waypoint_visualization'):
            self.unity.enable_waypoint_visualization()

        self.max_steps = int(max_steps)
        self.rwd = reward_cfg or DenseRewardConfig()
        self.verbose = verbose
        self.H, self.W = int(img_height), int(img_width)

        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255,
                shape=(self.H, self.W, 3),
                dtype=np.uint8
            ),
            "vec": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(12,),
                dtype=np.float32
            ),
        })

        self.action_space = spaces.Box(
            low=np.array([-1., 0., 0.], dtype=np.float32),
            high=np.array([1., 1., 1.], dtype=np.float32),
            dtype=np.float32
        )

        self._steps = 0
        self._prev_goal_dist = None
        self._episode_reward = 0.0

    def _get_raw_vec(self, obs):
        return obs.get("vec", np.zeros(12, dtype=np.float32))

    def _mask_vec_for_policy(self, raw_vec):
        policy_vec = raw_vec.copy()
        policy_vec[TelemetryIdx.GOAL_DIST] = 0.0  # Hide goal distance
        return policy_vec

    def _reward(self, raw_vec, action):
        turn_bias = raw_vec[TelemetryIdx.TURN_BIAS]
        goal_dist = raw_vec[TelemetryIdx.GOAL_DIST]
        speed = raw_vec[TelemetryIdx.SPEED]
        yaw_rate = raw_vec[TelemetryIdx.YAW_RATE]
        last_steer = raw_vec[TelemetryIdx.LAST_STEER]
        last_thr = raw_vec[TelemetryIdx.LAST_THR]
        last_brk = raw_vec[TelemetryIdx.LAST_BRK]
        lat_err = raw_vec[TelemetryIdx.LAT_ERR]
        hdg_err = raw_vec[TelemetryIdx.HDG_ERR]
        ds = raw_vec[TelemetryIdx.DS]

        steer = float(action[0]) if action is not None else 0.0

        # Forward progress reward
        fwd_reward = self.rwd.forward_k * max(0.0, ds * 50.0)
        if ds * 50.0 < -0.1:
            fwd_reward = self.rwd.forward_k * (ds * 50.0) * 2.0

        # Goal approach reward (DISABLED by default in config)
        goal_reward = 0.0
        if self._prev_goal_dist is not None and 0 < goal_dist < 1000:
            delta_dist = self._prev_goal_dist - goal_dist
            goal_reward = self.rwd.goal_approach_k * max(0.0, delta_dist)

            if goal_dist < self.rwd.goal_reached_threshold:
                goal_reward += self.rwd.goal_reached_bonus

        self._prev_goal_dist = goal_dist

        # Stuck penalty
        stuck = 0.0
        if abs(speed) < self.rwd.stuck_speed_eps and last_thr > self.rwd.stuck_thr_eps:
            stuck = self.rwd.stuck_pen_k

        # Behavior penalties
        pen = (
                self.rwd.spin_k * abs(yaw_rate) +
                self.rwd.steer_k * (steer ** 2) +
                self.rwd.brake_k * last_brk +
                self.rwd.lat_err_k * min(abs(lat_err), 3.0) +
                self.rwd.hdg_err_k * min(abs(hdg_err), 1.5)
        )

        total_reward = float(fwd_reward + goal_reward - pen - stuck)

        return total_reward

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.unity.reset(seed=seed, options=options)

        self._steps = 0
        self._episode_reward = 0.0

        raw = self._get_raw_vec(obs)
        goal_dist = raw[TelemetryIdx.GOAL_DIST]
        self._prev_goal_dist = float(goal_dist) if goal_dist < 1000 else None

        obs['vec'] = self._mask_vec_for_policy(raw)
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        obs, r_unity, done, trunc, info = self.unity.step(action)

        self._steps += 1
        raw = self._get_raw_vec(obs)

        reward = self._reward(raw, action)
        self._episode_reward += reward

        obs['vec'] = self._mask_vec_for_policy(raw)

        if self.max_steps > 0 and self._steps >= self.max_steps and not done:
            trunc = True

        info['episode_reward'] = self._episode_reward
        return obs, reward, bool(done), bool(trunc), info

    def render(self):
        return self.unity.render()

    def close(self):
        self.unity.close()

    def enable_waypoint_visualization(self):
        self.unity.enable_waypoint_visualization()

    def set_waypoints(self, waypoints):
        self.unity.set_waypoints(waypoints)

    @property
    def unwrapped(self):
        return self.unity
# action_repeat_wrapper.py
from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
import gymnasium as gym
"""
ActionRepeatWrapper

A tiny gymnasium wrapper that repeats each action for N physics steps.
Unity in lockstep mode steps at higher rates than our agent. Repeating actions reduces control jitters
without changing the policy code.
"""
class ActionRepeatWrapper(gym.Wrapper):
    """
    Repeat the same action for 'repeat' consecutive env steps.
    Accumulates reward across the repeated steps and stops early if terminated of truncated.
    """
    def __init__(self, env: gym.Env, repeat: int = 1):
        assert repeat >= 1, "repeat must be >= 1"
        super().__init__(env)
        self.repeat = int(repeat)

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        total_reward = 0.0
        last_obs = None
        last_info: Dict[str, Any] = {}
        terminated = False
        truncated = False

        for i in range(self.repeat):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += float(reward)
            last_obs, last_info = obs, info
            terminated |= bool(term)
            truncated |= bool(trunc)
            if terminated or truncated:
                break

        return last_obs, total_reward, terminated, truncated, last_info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
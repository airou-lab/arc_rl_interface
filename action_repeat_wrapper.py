# action_repeat_wrapper.py
import gymnasium as gym

class ActionRepeatWrapper(gym.Wrapper):
    """Repeat each action for N steps (sum rewards, early-break on done/truncated)."""
    def __init__(self, env: gym.Env, repeat: int = 1):
        assert repeat >= 1
        super().__init__(env)
        self.repeat = int(repeat)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None
        for _ in range(self.repeat):
            obs, r, terminated, truncated, info = self.env.step(action)
            total_reward += float(r)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
# Save as train_simple_hierarchical.py
import gymnasium as gym
import numpy as np
from live_unity_env import LiveUnityEnv
from stable_baselines3 import PPO


class SingleResetEnv(gym.Env):
    """Simple wrapper that prevents double resets"""

    def __init__(self, host='127.0.0.1', port=5556):
        super().__init__()
        self.host = host
        self.port = port
        self._first_reset = True

        # Create env to get spaces
        self.env = LiveUnityEnv(host, port, 84, 84, 500, False)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None):
        if self._first_reset:
            # First reset - actually reset Unity
            self._first_reset = False
            return self.env.reset(seed=seed, options=options)
        else:
            # Subsequent reset - just return stored obs without resetting Unity
            if hasattr(self.env, 'last_obs') and self.env.last_obs is not None:
                return self.env.last_obs, {}
            else:
                # Fallback - do actual reset
                return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


# Create and train
print("Creating environment...")
env = SingleResetEnv()

print("Creating model...")
model = PPO("MultiInputPolicy", env, verbose=1, n_steps=128, batch_size=64)

print("Starting training...")
try:
    model.learn(total_timesteps=5000)
    print("Training successful!")
except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
finally:
    env.close()
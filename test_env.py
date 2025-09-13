"""
Quick smoke test for LiveUnityEnv without SB3.
"""
from __future__ import annotations
import numpy as np
from live_unity_env import LiveUnityEnv, UnityEnvConfig

def main():
    env = LiveUnityEnv(UnityEnvConfig())
    obs, info = env.reset()
    print("reset:", obs.shape, info)

    total = 0.0
    for t in range(50):
        action = np.array([0.2, 0.4], dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)
        total += float(reward)
        print(f"t={t} r={reward:+.3f} done={done} trunc={truncated}")
        if done or truncated:
            break
    print("return:", total)
    env.close()

if __name__ == "__main__":
    main()
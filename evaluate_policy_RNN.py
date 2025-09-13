"""
Roll out a trained model for N episodes and print summary stats.
"""
from __future__ import annotations
import argparse
import numpy as np
from sb3_contrib import RecurrentPPO
from live_unity_env import LiveUnityEnv, UnityEnvConfig

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--img_size", type=int, nargs=2, default=[84, 84])
    p.add_argument("--max_steps", type=int, default=500)
    args = p.parse_args()

    env = LiveUnityEnv(UnityEnvConfig(
        host=args.host, port=args.port,
        img_width=args.img_size[0], img_height=args.img_size[1],
        max_steps=args.max_steps
    ))
    model = RecurrentPPO.load(args.model)

    returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_rew = 0.0
        done = truncated = False
        lstm_state = None
        episode_start = True
        while not (done or truncated):
            action, lstm_state = model.predict(
                obs, state=lstm_state,
                episode_start=np.array([episode_start], dtype=bool),
                deterministic=True
            )
            episode_start = False
            obs, reward, done, truncated, _ = env.step(action)
            ep_rew += float(reward)
        returns.append(ep_rew)
        print(f"[ep {ep+1}/{args.episodes}] return={ep_rew:.3f}")

    arr = np.array(returns, dtype=np.float32)
    print(f"\nmean={arr.mean():.3f} std={arr.std():.3f} min={arr.min():.3f} max={arr.max():.3f}")
    env.close()

if __name__ == "__main__":
    main()
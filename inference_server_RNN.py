"""
Run a trained RecurrentPPO model against the live Unity scene.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict

import numpy as np
from sb3_contrib import RecurrentPPO
from live_unity_env import LiveUnityEnv, UnityEnvConfig
from stable_baselines3.common.utils import get_schedule_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--img_size", type=int, nargs=2, default=[128, 128])
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args()

    cfg = UnityEnvConfig(
        host=args.host, port=args.port,
        img_width=args.img_size[0], img_height=args.img_size[1],
    )
    env = LiveUnityEnv(**asdict(cfg))
    model = RecurrentPPO.load(
        args.model,
        custom_objects={
            "clip_range": get_schedule_fn(0.2), # Limits policy updates to +-20% change
            "lr_schedule": get_schedule_fn(3e-4), # Learning rate of 0.0003
            "waypoint_criterion": None, # Not needed for inference currently
        },
        device="cpu" # Force CPU
    )

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_rew = 0.0
        step_count = 0

        # Let model handle LSTM state initialization
        lstm_state = None
        episode_start = np.array([True], dtype=bool)

        while not (done or truncated):
            action, lstm_state = model.predict(
                observation=obs,
                state=lstm_state,
                episode_start=episode_start,
                deterministic=args.deterministic,
            )
            episode_start = np.array([False], dtype=bool)

            obs, reward, done, truncated, info = env.step(action)
            ep_rew += float(reward)
            step_count += 1

        print(f"[ep {ep+1}] steps={step_count} reward={ep_rew:.3f} done={done} trunc={truncated}")

    env.close()


if __name__ == "__main__":
    main()

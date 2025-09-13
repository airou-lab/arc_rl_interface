"""
Run a model and optionally record to MP4 + CSV.

Requires `imageio[ffmpeg]` to write mp4; will auto-disable video if unavailable.
"""
from __future__ import annotations
import argparse
import csv
import os
from typing import Optional
import numpy as np
from sb3_contrib import RecurrentPPO
from live_unity_env import LiveUnityEnv, UnityEnvConfig

try:
    import imageio.v3 as iio
    _HAS_VIDEO = True
except Exception:
    _HAS_VIDEO = False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--img_size", type=int, nargs=2, default=[84, 84])
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--out_mp4", type=str, default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out_csv", type=str, default="run.csv")
    args = p.parse_args()

    if args.out_mp4 and not _HAS_VIDEO:
        print("[warn] imageio[ffmpeg] not installed; video will not be saved.")
        args.out_mp4 = None

    env = LiveUnityEnv(UnityEnvConfig(
        host=args.host, port=args.port,
        img_width=args.img_size[0], img_height=args.img_size[1],
        max_steps=args.max_steps, action_repeat=args.repeat,
    ))
    model = RecurrentPPO.load(args.model)

    vw = None
    if args.out_mp4:
        os.makedirs(os.path.dirname(args.out_mp4) or ".", exist_ok=True)
        vw = iio.imopen(args.out_mp4, "w", plugin="ffmpeg", fps=args.fps)

    csv_fp = open(args.out_csv, "w", newline="")
    csv_writer = csv.writer(csv_fp)
    csv_writer.writerow(["step", "reward", "done", "truncated"])

    obs, _ = env.reset()
    lstm_state = None
    episode_start = True
    done = truncated = False
    step = 0
    total_reward = 0.0

    while not (done or truncated):
        action, lstm_state = model.predict(
            obs, state=lstm_state, episode_start=np.array([episode_start], dtype=bool),
            deterministic=True
        )
        episode_start = False
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += float(reward)

        csv_writer.writerow([step, float(reward), int(done), int(truncated)])
        if vw is not None:
            vw.write_frame(obs)

        step += 1

    csv_fp.close()
    if vw is not None:
        vw.close()
    env.close()
    print(f"[done] steps={step} return={total_reward:.3f} csv={args.out_csv} mp4={args.out_mp4 or 'n/a'}")

if __name__ == "__main__":
    main()
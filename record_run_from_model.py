#!/usr/bin/env python3
"""
Record video and trajectory data from a trained model.

This script runs a single episode with a trained policy and saves:
- MP4 video of the agent's camera view
- CSV file with trajectory data (actions, rewards, telemetry)
- Waypoint predictions (if using hierarchical policy)

Usage:
    python record_run_from_model.py \
        --model models/rppo_*/final_model.zip \
        --out_mp4 recordings/episode.mp4 \
        --out_csv recordings/episode.csv
"""

import argparse
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from sb3_contrib import RecurrentPPO

from unity_dense_env import UnityDenseEnv, DenseRewardConfig
from action_repeat_wrapper import ActionRepeatWrapper
from wrappers.waypoint_tracking_wrapper import WaypointTrackingWrapper
from policies.hierarchical_policy import HierarchicalPathPlanningPolicy


def record_run(
    model_path: str,
    out_mp4: str,
    out_csv: str,
    host: str = "127.0.0.1",
    port: int = 5556,
    img_size: tuple = (128, 128),
    max_steps: int = 500,
    repeat: int = 1,
    fps: int = 30,
    scale: int = 4,
    verbose: bool = True,
):
    """
    Record a single episode with the trained model.

    Args:
        model_path: Path to .zip model file
        out_mp4: Output video file path
        out_csv: Output trajectory CSV file path
        host: Unity server hostname
        port: Unity server port
        img_size: Image resolution (width, height)
        max_steps: Maximum steps per episode
        repeat: Action repeat factor
        fps: Video frames per second
        scale: Upscale factor for video (e.g., 4 makes 128x128 -> 512x512)
        verbose: Print detailed logs
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = RecurrentPPO.load(model_path)

    # Check if hierarchical
    is_hierarchical = isinstance(model.policy, HierarchicalPathPlanningPolicy)
    if is_hierarchical:
        print("[OK] Detected hierarchical path planning policy")
    else:
        print("[OK] Detected standard LSTM policy")

    # Create environment
    print(f"Connecting to Unity on {host}:{port}...")
    env = UnityDenseEnv(
        host=host,
        port=port,
        img_width=img_size[0],
        img_height=img_size[1],
        max_steps=max_steps,
        reward_cfg=DenseRewardConfig(),
        verbose=verbose,
    )

    if repeat > 1:
        env = ActionRepeatWrapper(env, repeat=repeat)

    # Wrap with waypoint tracking if hierarchical
    if is_hierarchical:
        env = WaypointTrackingWrapper(env)
        if hasattr(env.unwrapped, 'enable_waypoint_visualization'):
            env.unwrapped.enable_waypoint_visualization()
        print("[OK] Enabled waypoint tracking and visualization")

    # Prepare video writer
    H, W = img_size[1], img_size[0]
    out_h, out_w = H * scale, W * scale
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_mp4, fourcc, fps, (out_w, out_h))

    if not video_writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out_mp4}")

    print(f"Recording to {out_mp4} ({out_w}x{out_h} @ {fps}fps)...")

    # Data collection
    trajectory_data = []
    waypoint_data = []

    # Reset environment
    obs, info = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    step = 0
    episode_reward = 0.0
    done = False
    truncated = False

    print("Recording episode...")

    while not (done or truncated):
        frame = obs["image"]

        # Predict action
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True
        )

        # Extract waypoints if hierarchical
        waypoints_np = None
        if is_hierarchical and hasattr(model.policy, 'predict_waypoints'):
            waypoints, _ = model.policy.predict_waypoints(
                obs,
                state=lstm_states,
                episode_start=episode_starts.squeeze() if episode_starts.ndim > 0 else episode_starts,
                deterministic=True
            )

            if waypoints is not None:
                waypoints_np = waypoints.cpu().numpy().squeeze()

                if hasattr(env, 'set_predicted_waypoints'):
                    env.set_predicted_waypoints(waypoints_np)
                if hasattr(env.unwrapped, 'set_waypoints'):
                    env.unwrapped.set_waypoints(waypoints_np)

        # Step environment
        obs_next, reward, done, truncated, info = env.step(action)
        episode_starts = np.zeros((1,), dtype=bool)
        episode_reward += reward

        # Record trajectory
        trajectory_data.append({
            'step': step,
            'reward': reward,
            'cumulative_reward': episode_reward,
            'steer': action[0],
            'throttle': action[1],
            'brake': action[2],
            'speed': info.get('speed', 0.0),
            'yaw_rate': info.get('yaw_rate', 0.0),
            'goal_cos': info.get('goal_cos', 0.0),
            'goal_sin': info.get('goal_sin', 0.0),
            'goal_dist': info.get('goal_dist', 0.0),
            'lat_err': info.get('lat_err', 0.0),
            'hdg_err': info.get('hdg_err', 0.0),
            'kappa': info.get('kappa', 0.0),
            'ds': info.get('ds', 0.0),
            'done': done,
            'truncated': truncated,
        })

        # Record waypoints
        if waypoints_np is not None:
            waypoint_record = {'step': step}
            for i in range(waypoints_np.shape[0]):
                waypoint_record[f'waypoint_{i}_x'] = waypoints_np[i, 0]
                waypoint_record[f'waypoint_{i}_y'] = waypoints_np[i, 1]
            waypoint_data.append(waypoint_record)

        # Write video frame (upscale and convert RGB -> BGR)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_scaled = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        video_writer.write(frame_scaled)

        obs = obs_next
        step += 1

        if step % 50 == 0:
            print(f"  Step {step}, reward: {episode_reward:.2f}")

    # Finalize video
    video_writer.release()
    print(f"[OK] Video saved to {out_mp4}")

    # Save trajectory
    df = pd.DataFrame(trajectory_data)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Trajectory saved to {out_csv}")

    # Save waypoints if available
    if waypoint_data:
        waypoint_csv = out_csv.replace('.csv', '_waypoints.csv')
        df_waypoints = pd.DataFrame(waypoint_data)
        df_waypoints.to_csv(waypoint_csv, index=False)
        print(f"[OK] Waypoints saved to {waypoint_csv}")

    # Print summary
    if done and not truncated:
        status = "SUCCESS"
    elif truncated:
        status = "TRUNCATED"
    else:
        status = "FAILED"

    print(f"\nEpisode Summary:")
    print(f"  Steps: {step}")
    print(f"  Total Reward: {episode_reward:.2f}")
    print(f"  Status: {status}")

    if is_hierarchical and waypoint_data:
        print(f"  Waypoint data points: {len(waypoint_data)}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Record video from trained model")

    # Model
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.zip)')

    # Output
    parser.add_argument('--out_mp4', type=str, required=True,
                        help='Output video path')
    parser.add_argument('--out_csv', type=str, required=True,
                        help='Output trajectory CSV path')

    # Environment
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5556)
    parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128],
                        metavar=('W', 'H'))
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--repeat', type=int, default=1)

    # Video
    parser.add_argument('--fps', type=int, default=30,
                        help='Video frames per second')
    parser.add_argument('--scale', type=int, default=4,
                        help='Upscale factor (128x128 -> 512x512 if scale=4)')

    parser.add_argument('--verbose', action='store_true', default=True)

    args = parser.parse_args()

    record_run(
        model_path=args.model,
        out_mp4=args.out_mp4,
        out_csv=args.out_csv,
        host=args.host,
        port=args.port,
        img_size=tuple(args.img_size),
        max_steps=args.max_steps,
        repeat=args.repeat,
        fps=args.fps,
        scale=args.scale,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()

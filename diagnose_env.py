#!/usr/bin/env python3
"""
diagnose_env.py - Debug what's actually happening in the environment
====================================================================
Runs episodes and prints RAW telemetry to understand the signals.
"""
import argparse
import numpy as np
import time

from live_unity_env import LiveUnityEnv


# Telemetry indices
class Idx:
    TURN_BIAS = 0  # Command: -1 (left), 0 (straight), +1 (right)
    RESERVED = 1  # Always 0 (was goal_sin)
    GOAL_DIST = 2  # Distance to goal
    SPEED = 3  # Current speed m/s
    YAW_RATE = 4  # Current yaw rate rad/s
    LAST_STEER = 5  # Last steering command
    LAST_THR = 6  # Last throttle command
    LAST_BRK = 7  # Last brake command
    LAT_ERR = 8  # Lateral error from path
    HDG_ERR = 9  # Heading error from path
    KAPPA = 10  # Path curvature 1/m
    DS = 11  # Distance traveled this step


def diagnose(args):
    print("\n" + "=" * 70)
    print("  ENVIRONMENT DIAGNOSTICS")
    print("=" * 70)

    # Create raw environment (no wrappers)
    env = LiveUnityEnv(
        host=args.host,
        port=args.port,
        img_width=128,
        img_height=128,
        max_steps=args.max_steps,
        verbose=True,
    )

    print("\nRunning diagnostic episodes...")
    print("Watch the Unity window to see what the car is doing!\n")

    for ep in range(args.n_episodes):
        print(f"\n{'=' * 70}")
        print(f"EPISODE {ep + 1}")
        print(f"{'=' * 70}")

        obs, info = env.reset()
        vec = obs['vec']

        print(f"\n[RESET] Initial telemetry:")
        print(f"  turn_bias:  {vec[Idx.TURN_BIAS]:+.2f}  (should be -1, 0, or +1)")
        print(f"  goal_dist:  {vec[Idx.GOAL_DIST]:.1f} m")
        print(f"  speed:      {vec[Idx.SPEED]:.2f} m/s")
        print(f"  lat_err:    {vec[Idx.LAT_ERR]:+.3f} m  (+ = right of center?)")
        print(f"  hdg_err:    {vec[Idx.HDG_ERR]:+.3f} rad ({np.degrees(vec[Idx.HDG_ERR]):+.1f}°)")
        print(f"  kappa:      {vec[Idx.KAPPA]:.4f} 1/m")

        initial_goal_dist = vec[Idx.GOAL_DIST]

        done = False
        step = 0
        total_ds = 0

        # Track ranges
        lat_errs = []
        hdg_errs = []
        speeds = []
        turn_biases = []

        while not done and step < args.max_steps:
            # Simple policy: steer based on lat_err and hdg_err
            lat_err = vec[Idx.LAT_ERR]
            hdg_err = vec[Idx.HDG_ERR]
            turn_bias = vec[Idx.TURN_BIAS]

            # Classical steering controller
            steer = 0.3 * turn_bias + 0.4 * lat_err + 0.5 * hdg_err
            steer = np.clip(steer, -1.0, 1.0)

            # Fixed throttle
            action = np.array([steer, 0.4, 0.0], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            vec = obs['vec']

            # Track metrics
            lat_errs.append(vec[Idx.LAT_ERR])
            hdg_errs.append(vec[Idx.HDG_ERR])
            speeds.append(vec[Idx.SPEED])
            turn_biases.append(vec[Idx.TURN_BIAS])
            total_ds += vec[Idx.DS]

            step += 1

            # Print every N steps
            if step % 50 == 0:
                print(f"\n[Step {step}]")
                print(f"  turn_bias:  {vec[Idx.TURN_BIAS]:+.2f}")
                print(f"  goal_dist:  {vec[Idx.GOAL_DIST]:.1f} m (started at {initial_goal_dist:.1f})")
                print(f"  speed:      {vec[Idx.SPEED]:.2f} m/s")
                print(f"  lat_err:    {vec[Idx.LAT_ERR]:+.3f} m")
                print(f"  hdg_err:    {vec[Idx.HDG_ERR]:+.3f} rad ({np.degrees(vec[Idx.HDG_ERR]):+.1f}°)")
                print(f"  kappa:      {vec[Idx.KAPPA]:.4f}")
                print(f"  ds:         {vec[Idx.DS]:.4f}")
                print(f"  steer_cmd:  {steer:+.3f}")
                print(f"  reward:     {reward:.2f}")

        # Episode summary
        print(f"\n{'=' * 50}")
        print(f"EPISODE {ep + 1} SUMMARY")
        print(f"{'=' * 50}")
        print(f"  Steps: {step}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        print(f"  Total distance (ds): {total_ds:.2f} m")
        print(f"  Goal distance: {initial_goal_dist:.1f} → {vec[Idx.GOAL_DIST]:.1f} m")
        print(f"  Progress toward goal: {initial_goal_dist - vec[Idx.GOAL_DIST]:.1f} m")

        print(f"\n  lat_err  range: [{min(lat_errs):+.3f}, {max(lat_errs):+.3f}] m")
        print(f"  hdg_err  range: [{min(hdg_errs):+.3f}, {max(hdg_errs):+.3f}] rad")
        print(f"  hdg_err  range: [{np.degrees(min(hdg_errs)):+.1f}°, {np.degrees(max(hdg_errs)):+.1f}°]")
        print(f"  speed    range: [{min(speeds):.2f}, {max(speeds):.2f}] m/s")

        unique_biases = set([round(b, 1) for b in turn_biases])
        print(f"  turn_bias values seen: {sorted(unique_biases)}")

        # Check for issues
        print(f"\n  DIAGNOSIS:")

        if max(speeds) < 0.5:
            print(f"  ⚠️  Car barely moving! Max speed only {max(speeds):.2f} m/s")

        if abs(max(hdg_errs)) > 0.5 or abs(min(hdg_errs)) > 0.5:
            print(f"  ⚠️  Large heading errors! Car pointing wrong direction?")

        if initial_goal_dist - vec[Idx.GOAL_DIST] < 5:
            print(f"  ⚠️  No progress toward goal! Only moved {initial_goal_dist - vec[Idx.GOAL_DIST]:.1f}m closer")

        if len(unique_biases) == 1 and 0.0 in unique_biases:
            print(f"  ⚠️  turn_bias always 0! Navigation commands not changing")

        if max(lat_errs) - min(lat_errs) < 0.1:
            print(f"  ⚠️  lat_err barely changes! Is RouteProgress working?")

    env.close()
    print("\n" + "=" * 70)
    print("  DIAGNOSTICS COMPLETE")
    print("=" * 70)
    print("\nCheck the Unity window - what was the car actually doing?")
    print("If lat_err and hdg_err don't match visual behavior, there's a")
    print("problem with RouteProgress or the route setup in Unity.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose environment signals")
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5556)
    parser.add_argument('--n_episodes', type=int, default=2)
    parser.add_argument('--max_steps', type=int, default=300)
    args = parser.parse_args()
    diagnose(args)
#!/usr/bin/env python3
"""
sanity_check.py - Quick environment sanity check
================================================
Runs a few random episodes to verify everything is connected properly.
"""
import argparse
import numpy as np
import time

from live_unity_env import LiveUnityEnv
from train_steering_only import (
    Idx, SafetyWrapper, ProgressRewardWrapper, LaneKeepingRewardWrapper,
    SteeringTargetWrapper, PassiveVisualWrapper, CurvatureSpeedWrapper
)
from stable_baselines3.common.monitor import Monitor


def sanity_check(args):
    print("\n" + "=" * 60)
    print("  ENVIRONMENT SANITY CHECK")
    print("=" * 60)
    print(f"Host: {args.host}:{args.port}")
    print(f"Image size: {args.img_size}")
    print("=" * 60 + "\n")

    # Create environment with full wrapper stack
    print("Creating environment...")
    env = LiveUnityEnv(
        host=args.host,
        port=args.port,
        img_width=args.img_size[0],
        img_height=args.img_size[1],
        max_steps=200,
        verbose=True,
    )

    # Apply wrapper stack
    env = SafetyWrapper(env, max_lateral_error=2.5)
    env = ProgressRewardWrapper(env, progress_scale=2.5)
    env = LaneKeepingRewardWrapper(env)
    env = SteeringTargetWrapper(env)
    env = PassiveVisualWrapper(env)
    env = CurvatureSpeedWrapper(env)
    env = Monitor(env)

    print("✓ Environment created\n")

    # Check observation and action spaces
    print("Observation space:")
    for key, space in env.observation_space.spaces.items():
        print(f"  {key}: {space}")
    print(f"\nAction space: {env.action_space}")
    print()

    # Run test episodes
    print("Running test episodes with random actions...\n")

    for ep in range(args.n_episodes):
        print(f"--- Episode {ep + 1} ---")
        obs, info = env.reset()

        # Check observation shapes
        assert obs['image'].shape == (args.img_size[1], args.img_size[0], 3), \
            f"Image shape mismatch: {obs['image'].shape}"
        assert obs['vec'].shape == (12,), f"Vec shape mismatch: {obs['vec'].shape}"

        # Check that privileged info is masked
        assert obs['vec'][Idx.GOAL_DIST] == 0.0, "goal_dist not masked!"
        assert obs['vec'][Idx.LAT_ERR] == 0.0, "lat_err not masked!"
        assert obs['vec'][Idx.HDG_ERR] == 0.0, "hdg_err not masked!"
        assert obs['vec'][Idx.KAPPA] == 0.0, "kappa not masked!"

        print(f"  turn_bias: {obs['vec'][Idx.TURN_BIAS]:.1f}")
        print(f"  speed: {obs['vec'][Idx.SPEED]:.2f} m/s")

        # Run some steps
        total_reward = 0
        step = 0
        done = False

        while not done and step < 100:
            # Random steering action
            action = np.array([np.random.uniform(-0.3, 0.3)])

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

            if step % 20 == 0:
                print(f"    Step {step}: reward={reward:.2f}, "
                      f"steer_err={info.get('steer_error', 0):.3f}, "
                      f"target={info.get('target_steer', 0):.2f}")

        print(f"  Episode ended: steps={step}, total_reward={total_reward:.2f}")
        print(f"  Termination: {info.get('terminated_reason', 'completed')}")
        print()

    env.close()

    print("=" * 60)
    print("  ✓ SANITY CHECK PASSED")
    print("=" * 60)
    print("\nThe environment is working correctly!")
    print("You can now run training with:")
    print(f"  python train_steering_only.py --host {args.host} --port {args.port}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Environment sanity check")
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5556)
    parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128])
    parser.add_argument('--n_episodes', type=int, default=2)
    args = parser.parse_args()
    sanity_check(args)
#!/usr/bin/env python3
"""
evaluate_steering.py - Evaluate steering-only models
====================================================
Tests trained models and reports metrics.
"""
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Import the wrappers from training script
from train_steering_only import (
    make_env, Idx,
    SafetyWrapper, ProgressRewardWrapper, LaneKeepingRewardWrapper,
    SteeringTargetWrapper, PassiveVisualWrapper, CurvatureSpeedWrapper
)


def evaluate(args):
    print("\n" + "=" * 60)
    print("  STEERING-ONLY MODEL EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print("=" * 60 + "\n")

    # Load model
    print("Loading model...")
    model = RecurrentPPO.load(args.model_path)
    print("✓ Model loaded\n")

    # Create environment
    env = DummyVecEnv([make_env(
        args.host, args.port,
        (args.img_size[0], args.img_size[1]),
        args.max_steps,
        verbose=args.verbose
    )])

    # Metrics collection
    metrics = defaultdict(list)

    print("Running evaluation episodes...")
    print("-" * 60)

    for ep in range(args.n_episodes):
        obs = env.reset()
        lstm_states = None
        episode_done = [False]

        ep_reward = 0.0
        ep_length = 0
        ep_steer_errors = []
        ep_lat_errors = []
        ep_centered_steps = 0

        while not episode_done[0]:
            # Get action from model
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=np.array([ep_length == 0]),
                deterministic=args.deterministic
            )

            # Step environment
            obs, reward, done, info = env.step(action)

            ep_reward += reward[0]
            ep_length += 1

            # Collect step metrics
            info = info[0]
            if 'steer_error' in info:
                ep_steer_errors.append(info['steer_error'])
            if 'unmasked_lat_err' in info:
                ep_lat_errors.append(abs(info['unmasked_lat_err']))
            if info.get('lane_status') == 'centered':
                ep_centered_steps += 1

            episode_done = done

        # Store episode metrics
        metrics['reward'].append(ep_reward)
        metrics['length'].append(ep_length)
        metrics['mean_steer_error'].append(np.mean(ep_steer_errors) if ep_steer_errors else 0)
        metrics['mean_lat_error'].append(np.mean(ep_lat_errors) if ep_lat_errors else 0)
        metrics['centered_pct'].append(100 * ep_centered_steps / max(ep_length, 1))

        # Get termination reason
        term_reason = info.get('terminated_reason', 'completed')
        metrics['termination'].append(term_reason)

        print(f"Episode {ep + 1:3d}: Reward={ep_reward:7.2f}, Length={ep_length:4d}, "
              f"SteerErr={metrics['mean_steer_error'][-1]:.3f}, "
              f"Centered={metrics['centered_pct'][-1]:.1f}%, "
              f"End={term_reason}")

    env.close()

    # Summary statistics
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    def report_stat(name, values):
        print(f"{name:20s}: {np.mean(values):8.2f} ± {np.std(values):6.2f} "
              f"(min={np.min(values):.2f}, max={np.max(values):.2f})")

    report_stat("Reward", metrics['reward'])
    report_stat("Episode Length", metrics['length'])
    report_stat("Steering Error", metrics['mean_steer_error'])
    report_stat("Lateral Error (m)", metrics['mean_lat_error'])
    report_stat("Centered (%)", metrics['centered_pct'])

    # Termination breakdown
    print(f"\nTermination reasons:")
    term_counts = defaultdict(int)
    for t in metrics['termination']:
        term_counts[t] += 1
    for reason, count in sorted(term_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count} ({100 * count / args.n_episodes:.1f}%)")

    print("\n" + "=" * 60)

    # Return success rate (completed without crash)
    success_rate = term_counts.get('completed', 0) / args.n_episodes
    print(f"Success Rate: {100 * success_rate:.1f}%")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate steering-only model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('model_path', type=str,
                        help='Path to trained model (.zip)')
    parser.add_argument('--n_episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic actions (no exploration)')

    # Environment
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5556)
    parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128])
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    evaluate(args)
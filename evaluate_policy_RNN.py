#!/usr/bin/env python3
"""
Evaluate a trained RecurrentPPO policy.

This script loads a trained model and runs it in the Unity environment,
collecting statistics on episode rewards, lengths, and success rates.
For hierarchical policies, it also tracks and saves waypoint predictions.

Usage:
    python evaluate_policy_RNN.py --model models/rppo_*/final_model.zip --episodes 10
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sb3_contrib import RecurrentPPO

from unity_dense_env import UnityDenseEnv, DenseRewardConfig
from action_repeat_wrapper import ActionRepeatWrapper
from wrappers.waypoint_tracking_wrapper import WaypointTrackingWrapper
from policies.hierarchical_policy import HierarchicalPathPlanningPolicy


def evaluate_policy(
        model_path: str,
        num_episodes: int = 10,
        host: str = "127.0.0.1",
        port: int = 5556,
        img_size: tuple = (128, 128),
        max_steps: int = 500,
        repeat: int = 1,
        save_waypoints: bool = True,
        output_dir: str = "./eval_results",
        verbose: bool = True,
):
    """
    Evaluate a trained policy over multiple episodes.

    Args:
        model_path: Path to trained model (.zip)
        num_episodes: Number of episodes to evaluate
        host: Unity server hostname
        port: Unity server port
        img_size: Image resolution (width, height)
        max_steps: Maximum steps per episode
        repeat: Action repeat factor
        save_waypoints: Whether to save waypoint trajectories (hierarchical only)
        output_dir: Directory to save results
        verbose: Print detailed logs

    Returns:
        Dictionary containing evaluation statistics
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

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Evaluation statistics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    truncated_count = 0
    all_waypoint_histories = []

    print(f"\n{'=' * 60}")
    print(f"Starting evaluation: {num_episodes} episodes")
    print(f"{'=' * 60}\n")

    for ep in range(num_episodes):
        obs, info = env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        episode_reward = 0.0
        episode_length = 0
        done = False
        truncated = False

        print(f"Episode {ep + 1}/{num_episodes}...", end=" ", flush=True)

        while not (done or truncated):
            # Predict action
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True
            )

            # Extract and send waypoints if hierarchical
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
            obs, reward, done, truncated, info = env.step(action)
            episode_starts = np.zeros((1,), dtype=bool)

            episode_reward += reward
            episode_length += 1

        # Record statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if done and not truncated:
            success_count += 1
            status = "SUCCESS"
        elif truncated:
            truncated_count += 1
            status = "TRUNCATED"
        else:
            status = "FAILED"

        print(f"{status} | Reward: {episode_reward:.2f} | Length: {episode_length}")

        # Save waypoint history if hierarchical
        if is_hierarchical and save_waypoints and hasattr(env, 'get_waypoint_history'):
            waypoint_hist = env.get_waypoint_history()
            if waypoint_hist:
                all_waypoint_histories.append({
                    'episode': ep + 1,
                    'reward': episode_reward,
                    'length': episode_length,
                    'success': done and not truncated,
                    'waypoints': waypoint_hist,
                })

    # Compute statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    success_rate = success_count / num_episodes
    truncated_rate = truncated_count / num_episodes

    # Print results
    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Episodes:        {num_episodes}")
    print(f"Mean Reward:     {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean Length:     {mean_length:.1f} steps")
    print(f"Success Rate:    {success_rate * 100:.1f}% ({success_count}/{num_episodes})")
    print(f"Truncated Rate:  {truncated_rate * 100:.1f}% ({truncated_count}/{num_episodes})")
    print(f"{'=' * 60}\n")

    # Save results
    results = {
        'model_path': str(model_path),
        'num_episodes': num_episodes,
        'is_hierarchical': is_hierarchical,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'mean_length': float(mean_length),
        'success_count': success_count,
        'success_rate': float(success_rate),
        'truncated_count': truncated_count,
        'truncated_rate': float(truncated_rate),
    }

    results_file = output_path / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved results to {results_file}")

    # Save waypoint histories if hierarchical
    if is_hierarchical and save_waypoints and all_waypoint_histories:
        waypoints_file = output_path / "waypoint_histories.json"
        with open(waypoints_file, 'w') as f:
            json.dump(
                all_waypoint_histories, f, indent=2,
                default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x
            )
        print(f"[OK] Saved waypoint histories to {waypoints_file}")
        print(f"     Total waypoint data points: {sum(len(h['waypoints']) for h in all_waypoint_histories)}")

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RecurrentPPO policy")

    # Model
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.zip)')

    # Evaluation
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to evaluate')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                        help='Directory to save results')

    # Environment
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5556)
    parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128],
                        metavar=('W', 'H'))
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--repeat', type=int, default=1)

    # Waypoints
    parser.add_argument('--save_waypoints', action='store_true', default=True,
                        help='Save waypoint trajectories (if hierarchical)')
    parser.add_argument('--no_save_waypoints', dest='save_waypoints', action='store_false',
                        help='Do not save waypoint trajectories')

    parser.add_argument('--verbose', action='store_true', default=True)

    args = parser.parse_args()

    evaluate_policy(
        model_path=args.model,
        num_episodes=args.episodes,
        host=args.host,
        port=args.port,
        img_size=tuple(args.img_size),
        max_steps=args.max_steps,
        repeat=args.repeat,
        save_waypoints=args.save_waypoints,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
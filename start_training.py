#!/usr/bin/env python3
"""
Quick training launcher - Start training with one command!
Handles both standard and hierarchical policies.
"""

import subprocess
import sys
import os
from pathlib import Path


def check_unity():
    """Check if Unity is likely running"""
    import socket
    try:
        # Try to connect to Unity port
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        result = s.connect_ex(('127.0.0.1', 5556))
        s.close()
        return result == 0
    except:
        return False


def main():
    print("\n AUTONOMOUS DRIVING TRAINING LAUNCHER")
    print("=" * 60)

    # Check project setup
    if not Path("policies").exists():
        print("\n️  Project not set up yet. Running setup...")
        subprocess.run([sys.executable, "setup_project.py"], check=True)

    # Check Unity
    if not check_unity():
        print("\n Unity doesn't appear to be running on port 5556")
        print("   Please start Unity with RLClientSender and press Play")
        print("   Then run this script again.")
        sys.exit(1)
    else:
        print(" Unity detected on port 5556")

    # Choose training mode
    print("\n Training Mode:")
    print("  1) Standard Policy (Dense rewards only)")
    print("  2) Hierarchical Policy (Waypoint prediction) [RECOMMENDED]")
    print("  3) Quick Test (10k steps for debugging)")

    choice = input("\nSelect mode (1/2/3) [default=2]: ").strip() or "2"

    # Base command
    cmd = [
        sys.executable, "-u", "train_policy_RNN.py",
        "--host", "127.0.0.1",
        "--port", "5556",
        "--img_size", "84", "84",
        "--max_steps", "500",
        "--n_steps", "256",
        "--batch_size", "128",
        "--n_epochs", "5",
        "--gamma", "0.995",
        "--save_freq", "25000",
        "--verbose", "1",
        "--tensorboard_log", "./tb",
    ]

    if choice == "1":
        # Standard policy
        print("\n Training standard RecurrentPPO...")
        cmd.extend([
            "--timesteps", "200000",
            "--lr", "5e-4",
            "--ent_coef", "0.02",
        ])

    elif choice == "2":
        # Hierarchical policy
        print("\n Training hierarchical policy with waypoint prediction...")
        cmd.extend([
            "--hierarchical",
            "--timesteps", "200000",
            "--lr", "3e-4",  # Slightly lower LR for hierarchical
            "--ent_coef", "0.01",  # Lower entropy for precise waypoints
            "--num_waypoints", "5",
            "--waypoint_horizon", "2.5",
            "--self_supervised_weight", "0.8",
            "--goal_directed_weight", "0.05",
            "--waypoint_loss_weight", "0.15",
        ])

    elif choice == "3":
        # Quick test
        print("\n Quick test run (10k steps)...")
        cmd.extend([
            "--hierarchical",
            "--timesteps", "10000",
            "--lr", "3e-4",
            "--ent_coef", "0.01",
            "--save_freq", "5000",
        ])

    else:
        print("Invalid choice")
        sys.exit(1)

    print("\n Models will be saved to: ./models/")
    print(" TensorBoard logs: ./tb/")
    print("\nTo monitor: tensorboard --logdir ./tb")
    print("\n" + "=" * 60)
    print("Starting training...\n")

    try:
        # Run training
        subprocess.run(cmd, check=True)
        print("\n Training complete!")

        # Find latest model
        model_dirs = sorted(Path("models").glob("rppo_*"))
        if model_dirs:
            latest = model_dirs[-1]
            print(f"\n Latest model: {latest}/final_model.zip")
            print(f"\nTo evaluate: python evaluate_policy_RNN.py --model {latest}/final_model.zip")

    except KeyboardInterrupt:
        print("\n\n️  Training interrupted by user")
    except subprocess.CalledProcessError as e:
        print(f"\n Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
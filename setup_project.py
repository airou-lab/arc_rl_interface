#!/usr/bin/env python3
"""
Setup script to organize project structure and fix imports
Run this ONCE before training to set up everything correctly
"""

import os
import sys
from pathlib import Path


def setup_project():
    """Create proper directory structure and fix imports"""

    print(" Setting up project structure...")

    # Create required directories
    directories = [
        "models",
        "logs",
        "tb",
        "policies",
        "losses",
        "wrappers",
        "rollouts",
        "eval_results"
    ]

    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"  ✓ Created {dir_name}/")

    # Create __init__.py files for Python packages
    packages = ["policies", "losses", "wrappers"]
    for pkg in packages:
        init_file = Path(pkg) / "__init__.py"
        if not init_file.exists():
            init_file.write_text("")
            print(f"  ✓ Created {pkg}/__init__.py")

    # Move files to correct locations if they exist in root
    file_moves = {
        "hierarchical_policy.py": "policies/hierarchical_policy.py",
        "waypoint_losses.py": "losses/waypoint_losses.py",
        "waypoint_tracking_wrapper.py": "wrappers/waypoint_tracking_wrapper.py",
        "coach_reward_wrapper.py": "wrappers/coach_reward_wrapper.py",
        "lane_aux_wrapper.py": "wrappers/lane_aux_wrapper.py"
    }

    for src, dst in file_moves.items():
        src_path = Path(src)
        dst_path = Path(dst)
        if src_path.exists() and not dst_path.exists():
            src_path.rename(dst_path)
            print(f"  ✓ Moved {src} → {dst}")

    print("\n Project structure ready!")
    print("\n Directory layout:")
    print("  project/")
    print("  ├── train_policy_RNN.py      (main training)")
    print("  ├── policies/                (policy networks)")
    print("  │   └── hierarchical_policy.py")
    print("  ├── losses/                  (loss functions)")
    print("  │   └── waypoint_losses.py")
    print("  ├── wrappers/                (env wrappers)")
    print("  │   ├── waypoint_tracking_wrapper.py")
    print("  │   └── ...(other wrappers)")
    print("  ├── models/                  (saved models)")
    print("  ├── logs/                    (training logs)")
    print("  └── tb/                      (tensorboard)")


if __name__ == "__main__":
    setup_project()
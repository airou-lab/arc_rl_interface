import os
import sys
import argparse
import json
import shutil
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
from typing import cast

# Ensure unity_env module is accessible
sys.path.append(os.path.dirname(__file__))
from unity_camera_env import UnityCameraEnv

def get_next_eval_index(archive_root: str) -> int:
    os.makedirs(archive_root, exist_ok=True)
    existing = [d for d in os.listdir(archive_root) if d.startswith("eval_run")]
    indices = []
    for name in existing:
        parts = name.split("eval_run")
        if len(parts) == 2 and parts[1].isdigit():
            indices.append(int(parts[1]))
    return max(indices, default=0) + 1

def evaluate(model_path: str, capture_dir: str, episodes: int = 3,
             save_debug_masks: bool = False, mask_refresh_N: int = 5,
             deviation_refresh_threshold: float = 0.7, refresh_cooldown: int = 3):
    print("[DEBUG] Starting evaluation")

    if not os.path.isfile(model_path):
        print(f"[ERROR] Model not found at: {model_path}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"eval_{timestamp}"
    eval_log_dir = os.path.join("evaluation_logs", run_tag)
    os.makedirs(eval_log_dir, exist_ok=True)

    env = UnityCameraEnv(
        capture_dir=capture_dir,
        save_debug_masks=save_debug_masks,
        mask_refresh_N=mask_refresh_N,
        deviation_refresh_threshold=deviation_refresh_threshold,
        refresh_cooldown=refresh_cooldown
    )
    env.eval_mode = True
    monitor_path = os.path.join(eval_log_dir, "monitor.csv")
    env = Monitor(env, filename=monitor_path)
    env = TimeLimit(env, max_episode_steps=cast(UnityCameraEnv, env.unwrapped).max_steps)
    vec_env = DummyVecEnv([lambda: env])

    print(f"[INFO] Loading model: {model_path}")
    model = PPO.load(model_path)
    print("[INFO] Loaded model successfully")

    episode_rewards = []
    for ep in range(episodes):
        obs = vec_env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_vec, _ = vec_env.step(action)
            done = done_vec[0]
            total_reward += reward[0]

        print(f"[EPISODE {ep + 1}] Total reward: {total_reward:.2f}")
        episode_rewards.append(total_reward)

    summary = {
        "model": model_path,
        "capture_dir": capture_dir,
        "episodes": episodes,
        "rewards": [float(r) for r in episode_rewards],
        "average_reward": float(np.mean(episode_rewards))
    }
    summary_path = os.path.join(eval_log_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[SAVED] Evaluation summary saved to: {summary_path}")
    print(f"[INFO] Evaluation complete. Logs directory: {eval_log_dir}")

    archive_root = "models"
    eval_index = get_next_eval_index(archive_root)
    archive_dir = os.path.join(archive_root, f"eval_run{eval_index}")
    os.makedirs(archive_dir, exist_ok=True)

    try:
        shutil.copy2(summary_path, os.path.join(archive_dir, "summary.json"))
        shutil.copy2(monitor_path, os.path.join(archive_dir, "monitor.csv"))

        debug_dir = os.path.join("Logs", "DebugMasks")
        if os.path.exists(debug_dir):
            shutil.copytree(debug_dir, os.path.join(archive_dir, "DebugMasks"))

        print(f"[ARCHIVE] Evaluation results archived in: {archive_dir}")
    except Exception as e:
        print(f"[WARN] Failed to archive evaluation logs: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/final_ppo_unity_model_latest.zip",
                        help="Path to PPO model (.zip)")
    parser.add_argument("--capture_dir", type=str, default="../../Assets/Captures",
                        help="Directory containing Unity image captures")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of evaluation episodes")
    parser.add_argument("--save_debug_masks", action="store_true",
                        help="Save lane mask debug images to Logs/DebugMasks")
    parser.add_argument("--mask_refresh_N", type=int, default=5,
                        help="Refresh lane mask every N steps")
    parser.add_argument("--deviation_refresh_threshold", type=float, default=0.7,
                        help="Trigger lane mask refresh if deviation exceeds this threshold")
    parser.add_argument("--refresh_cooldown", type=int, default=3,
                        help="Minimum steps between deviation-triggered lane mask refresh")
    args = parser.parse_args()

    evaluate(model_path=args.model_path,
             capture_dir=args.capture_dir,
             episodes=args.episodes,
             save_debug_masks=args.save_debug_masks,
             mask_refresh_N=args.mask_refresh_N,
             deviation_refresh_threshold=args.deviation_refresh_threshold,
             refresh_cooldown=args.refresh_cooldown)

if __name__ == "__main__":
    main()
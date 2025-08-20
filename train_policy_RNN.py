"""
Recurrent (spatiotemporal) PPO training with strictly passive RGB observations.
- Uses sb3-contrib RecurrentPPO with CnnLstmPolicy (temporal memory).
- Works with either an offline image env (actions don't affect frames) or a live Unity socket env
  where actions affect the next frame (recommended for true spatiotemporal credit assignment).
"""
import os, sys, argparse
from datetime import datetime
from pathlib import Path

import torch.nn as nn
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gymnasium.wrappers import TimeLimit

from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import CnnLstmPolicy

# Local imports
sys.path.append(os.path.dirname(__file__))
from unity_camera_env import UnityCameraEnv
from live_unity_env import LiveUnityEnv

def make_env(args, eval_mode=False):
    if args.live:
        env = LiveUnityEnv(host=args.host, port=args.port, img_size=tuple(args.img_size), max_steps=args.max_steps)
    else:
        env = UnityCameraEnv(capture_dir=args.capture_dir, img_size=tuple(args.img_size), max_steps=args.max_steps)
    env.eval_mode = bool(eval_mode)
    return env

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true", default=False, help="Train against live Unity socket env")
    p.add_argument("--host", type=str, default="127.0.0.1", help="Unity host (live mode)")
    p.add_argument("--port", type=int, default=5556, help="Unity port (live mode)")

    p.add_argument("--capture_dir", type=str, default="../../Assets/Captures", help="Offline frames dir")
    p.add_argument("--img_size", type=int, nargs=2, default=[84, 84], help="(W H)")
    p.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    p.add_argument("--timesteps", type=int, default=200_000, help="Total training steps")
    p.add_argument("--tensorboard_log", type=str, default="./tensorboard_logs", help="TB dir")
    p.add_argument("--save_freq", type=int, default=20_000, help="Checkpoint freq")
    p.add_argument("--eval_freq", type=int, default=10_000, help="Eval freq")
    p.add_argument("--eval_episodes", type=int, default=5, help="Eval episodes")
    p.add_argument("--verbose", type=int, default=1, help="Verbosity")

    args = p.parse_args()

    run_tag = datetime.now().strftime("RNN_%Y%m%d_%H%M%S")
    tb_dir = Path(args.tensorboard_log) / f"PPO_{run_tag}"
    models_dir = Path("models") / run_tag
    logs_dir = Path("logs") / run_tag
    ckpt_dir = Path("checkpoints") / run_tag
    for d in [tb_dir, models_dir, logs_dir, ckpt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Build envs
    train_env = make_env(args, eval_mode=False)
    train_env = Monitor(train_env, filename=str(logs_dir / "monitor.csv"))
    train_env = TimeLimit(train_env, max_episode_steps=args.max_steps)
    train_env = DummyVecEnv([lambda: train_env])
    train_env = VecTransposeImage(train_env)  # CHW->HWC

    eval_env = make_env(args, eval_mode=True)
    eval_env = Monitor(eval_env)
    eval_env = TimeLimit(eval_env, max_episode_steps=args.max_steps)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecTransposeImage(eval_env)

    # Recurrent PPO with CNN+LSTM
    policy_kwargs = dict(activation_fn=nn.ReLU)

    model = RecurrentPPO(
        CnnLstmPolicy,
        train_env,
        learning_rate=2.5e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tb_dir),
        verbose=args.verbose
    )

    # Callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir),
        log_path=str(logs_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        verbose=args.verbose
    )
    ckpt_cb = CheckpointCallback(save_freq=args.save_freq, save_path=str(ckpt_dir), name_prefix="ppo_rnn")

    print("\nStarting training...")
    model.learn(total_timesteps=args.timesteps, callback=[eval_cb, ckpt_cb], progress_bar=True)

    final_path = models_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"Saved final model to {final_path}")

if __name__ == "__main__":
    main()
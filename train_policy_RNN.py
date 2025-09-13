#!/usr/bin/env python3
"""
Train a recurrent PPO (LSTM) agent on the Unity camera stream.

- Uses sb3_contrib.RecurrentPPO with CnnLstmPolicy
- Connects to Unity via LiveUnityEnv (pure RGB, passive)
- Optional action repeat (wrapper)
- Checkpoints and TensorBoard logging

Example:
python train_policy_RNN.py --host 127.0.0.1 --port 5556 \
  --img_size 84 84 --max_steps 500 --timesteps 200000 \
  --lr 5e-4 --n_steps 256 --batch_size 128 --n_epochs 5 \
  --ent_coef 0.02 --repeat 1 --verbose 1
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime
from typing import Callable, List
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv as DVE, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO
from live_unity_env import LiveUnityEnv, UnityEnvConfig
from action_repeat_wrapper import ActionRepeatWrapper

def make_env_fn(host: str, port: int, img_size, max_steps: int, repeat: int) -> Callable[[], gym.Env]:
    """Factory so DummyVecEnv can create/reset env cleanly."""
    def _thunk() -> gym.Env:
        cfg = UnityEnvConfig(
            host=host,
            port=port,
            img_width=img_size[0],
            img_height=img_size[1],
            max_steps=max_steps,
            action_repeat=1, # wrap below instead
        )
        env = LiveUnityEnv(cfg)
        env = Monitor(env)
        if max_steps is not None and max_steps > 0:
            env = TimeLimit(env, max_episode_steps=max_steps)
        if repeat and repeat > 1:
            env = ActionRepeatWrapper(env, repeat=repeat)
        return env
    return _thunk

def main():
    p = argparse.ArgumentParser(description="Train Recurrent PPO on Unity RGB stream (passive).")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--img_size", type=int, nargs=2, default=[84, 84], metavar=("W", "H"))
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--timesteps", type=int, default=200_000)

    # PPO hyperparams
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--n_steps", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_epochs", type=int, default=5)
    p.add_argument("--ent_coef", type=float, default=0.02)
    p.add_argument("--gamma", type=float, default=0.99)

    # Wrappers / misc
    p.add_argument("--repeat", type=int, default=1, help="Action repeat (use 1 for no repeat)")
    p.add_argument("--tensorboard_log", type=str, default="./tb")
    p.add_argument("--save_freq", type=int, default=25_000)
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args()

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = os.path.join("models", f"rppo_{run_tag}")
    os.makedirs(models_dir, exist_ok=True)

    print("[main] creating VecEnv...")
    env_fn = make_env_fn(args.host, args.port, tuple(args.img_size), args.max_steps, args.repeat)
    vec = DVE([env_fn])
    vec = VecMonitor(vec)

    print(f"[main] Vec obs_space={vec.observation_space} act_space={vec.action_space}")

    model = RecurrentPPO(
        policy="CnnLstmPolicy",
        env=vec,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        verbose=args.verbose,
        tensorboard_log=args.tensorboard_log,
        device="cpu", # change to "cuda" on GPU
    )

    ckpt = CheckpointCallback(save_freq=args.save_freq, save_path=models_dir, name_prefix="rppo")

    print("[main] === Training start ===")
    model.learn(total_timesteps=args.timesteps, callback=[ckpt], progress_bar=True)

    final_path = os.path.join(models_dir, "final_model.zip")
    model.save(final_path)
    print(f"[main] saved: {final_path}")

if __name__ == "__main__":
    main()
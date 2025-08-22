#!/usr/bin/env python3
"""
RecurrentPPO training for Unity live RGB (84x84) — purely passive (vision-only).

- Auto-detects CHW (3,H,W) observations and converts to HWC (H,W,3) for SB3 v2.
- Single Unity connection (no EvalCallback).
- Gymnasium + sb3-contrib 2.x.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import gymnasium as gym

from stable_baselines3.common.vec_env import DummyVecEnv as DVE, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO

# Your live env (must implement Gymnasium API reset/step; may output CHW or HWC)
from live_unity_env import LiveUnityEnv


# ---------- Wrappers ----------
class ToHWCWrapper(gym.ObservationWrapper):
    """Convert CHW (C,H,W) uint8 to HWC (H,W,C) uint8 and fix the space."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box), "obs space must be Box"
        shp = env.observation_space.shape
        assert len(shp) == 3 and shp[0] == 3, f"expected (3,H,W), got {shp}"
        H, W = shp[1], shp[2]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(H, W, 3), dtype=np.uint8
        )

    def observation(self, obs):
        if isinstance(obs, np.ndarray) and obs.ndim == 3 and obs.shape[0] == 3:
            return np.transpose(obs, (1, 2, 0))  # CHW -> HWC
        return obs


class ActionRepeatWrapper(gym.Wrapper):
    """Repeat each action for N steps (sum rewards, early break on done/truncated)."""
    def __init__(self, env: gym.Env, repeat: int = 1):
        assert repeat >= 1, "repeat must be >= 1"
        super().__init__(env)
        self.repeat = int(repeat)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None
        for _ in range(self.repeat):
            obs, r, terminated, truncated, info = self.env.step(action)
            total_reward += float(r)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def make_env(host: str, port: int, img_size: tuple[int, int], max_steps: int | None, repeat: int):
    """Factory for DummyVecEnv."""
    def _thunk():
        print(f"[make_env] host={host} port={port} img_size={img_size} max_steps={max_steps} repeat={repeat}", flush=True)
        env = LiveUnityEnv(host=host, port=port, img_size=tuple(img_size), max_steps=max_steps)

        # Auto-detect CHW and convert to HWC for SB3 v2
        assert isinstance(env.observation_space, gym.spaces.Box), "obs space must be Box"
        shp = env.observation_space.shape
        if len(shp) == 3 and shp[0] == 3 and shp[-1] != 3:
            print(f"[make_env] detected CHW {shp} -> applying ToHWCWrapper", flush=True)
            env = ToHWCWrapper(env)
            shp = env.observation_space.shape

        # Sanity checks (SB3 expects channel-last uint8)
        assert env.observation_space.dtype == np.uint8, "obs dtype must be uint8"
        H, W, C = shp
        assert (H, W, C) == (img_size[1], img_size[0], 3), f"expected {(img_size[1], img_size[0], 3)}, got {shp}"

        env = ActionRepeatWrapper(env, repeat=repeat)
        return env
    return _thunk


def main():
    parser = argparse.ArgumentParser(description="Train RecurrentPPO on Unity live 84x84 RGB")
    # Connection / env
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--img_size", type=int, nargs=2, default=[84, 84], help="(W H)")
    parser.add_argument("--max_steps", type=int, default=500, help="Episode horizon (Unity may end earlier)")
    parser.add_argument("--repeat", type=int, default=1, help="Action repeat (1 = no repeat)")

    # PPO hyperparams
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=256, help="Rollout length per update")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--ent_coef", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.99)

    # Logging / saving
    parser.add_argument("--tensorboard_log", type=str, default="./tb")
    parser.add_argument("--save_freq", type=int, default=50_000, help="Checkpoint frequency (timesteps)")
    parser.add_argument("--verbose", type=int, default=1)

    args = parser.parse_args()

    # Dirs
    Path("models").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    Path(args.tensorboard_log).mkdir(parents=True, exist_ok=True)

    # Build VecEnv (single Unity connection)
    print("[main] creating VecEnv...", flush=True)
    env_fn = make_env(args.host, args.port, tuple(args.img_size), args.max_steps, args.repeat)
    vec = DVE([env_fn])
    vec = VecMonitor(vec)
    print(f"[main] Vec obs_space={vec.observation_space} act_space={vec.action_space}", flush=True)

    # Model
    model = RecurrentPPO(
        "CnnLstmPolicy",
        vec,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        verbose=args.verbose,
        tensorboard_log=args.tensorboard_log,
    )

    # Checkpointing
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=f"checkpoints/{run_tag}",
        name_prefix="ppo_recurrent"
    )

    print("[main] === Training start ===", flush=True)
    model.learn(total_timesteps=args.timesteps, callback=[ckpt], progress_bar=True)
    print("[main] === Training end ===", flush=True)

    # Save final model
    final_path = Path("models") / f"final_model_{run_tag}.zip"
    model.save(str(final_path))
    print(f"[main] saved model to {final_path}", flush=True)


if __name__ == "__main__":
    main()
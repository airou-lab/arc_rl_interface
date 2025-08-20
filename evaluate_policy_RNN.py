import os, sys, argparse
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
from sb3_contrib import RecurrentPPO

sys.path.append(os.path.dirname(__file__))
from unity_camera_env import UnityCameraEnv
from live_unity_env import LiveUnityEnv

def make_env(args, eval_mode=True):
    if args.live:
        env = LiveUnityEnv(host=args.host, port=args.port, img_size=tuple(args.img_size), max_steps=args.max_steps)
    else:
        env = UnityCameraEnv(capture_dir=args.capture_dir, img_size=tuple(args.img_size), max_steps=args.max_steps)
    env.eval_mode = bool(eval_mode)
    return env

def run_episode(model, env, deterministic=True):
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0.0
    state = None
    episode_start = True
    while not (done or truncated):
        # VecTransposeImage expects HWC inside policy; here we pass CHW raw to env; model.predict expects HWC only when using VecEnv
        # For direct env use, transpose to HWC:
        obs_np = np.transpose(obs, (1,2,0))  # CHW->HWC
        obs_np = np.expand_dims(obs_np, axis=0)  # batch
        action, state = model.predict(obs_np, state=state, episode_start=np.array([episode_start]), deterministic=deterministic)
        episode_start = False
        obs, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
    return total_reward

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--live", action="store_true", default=False)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--capture_dir", type=str, default="../../Assets/Captures")
    p.add_argument("--img_size", type=int, nargs=2, default=[84,84])
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--episodes", type=int, default=5)
    args = p.parse_args()

    env = make_env(args, eval_mode=True)
    model = RecurrentPPO.load(args.model_path)

    rewards = []
    for ep in range(args.episodes):
        ep_r = run_episode(model, env, deterministic=True)
        print(f"Episode {ep+1}: reward={ep_r:.3f}")
        rewards.append(ep_r)

    print(f"\nMean reward over {args.episodes} eps: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")

if __name__ == "__main__":
    main()

"""
Run a trained RecurrentPPO model against the live Unity scene.
This is a *client* that connects to RLClientSender, reads frames,
and sends actions predicted by the model. (Name kept for continuity.)

Example:
python inference_server_RNN.py --model models/rppo_YYYYMMDD_HHMMSS/final_model.zip
"""
from __future__ import annotations
import argparse
import numpy as np
from sb3_contrib import RecurrentPPO
from live_unity_env import LiveUnityEnv, UnityEnvConfig

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--img_size", type=int, nargs=2, default=[84, 84])
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--episodes", type=int, default=10)
    args = p.parse_args()

    cfg = UnityEnvConfig(
        host=args.host, port=args.port,
        img_width=args.img_size[0], img_height=args.img_size[1],
        max_steps=args.max_steps, action_repeat=args.repeat,
    )
    env = LiveUnityEnv(cfg)

    model = RecurrentPPO.load(args.model)

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_rew = 0.0
        lstm_state = None
        episode_start = True

        while not (done or truncated):
            action, lstm_state = model.predict(
                observation=obs,
                state=lstm_state,
                episode_start=np.array([episode_start], dtype=bool),
                deterministic=False,
            )
            episode_start = False
            obs, reward, done, truncated, info = env.step(action)
            ep_rew += float(reward)

        print(f"[ep {ep+1}] reward={ep_rew:.3f} done={done} trunc={truncated}")

    env.close()

if __name__ == "__main__":
    main()
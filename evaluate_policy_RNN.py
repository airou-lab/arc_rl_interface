import os, sys, argparse, numpy as np
from sb3_contrib import RecurrentPPO

sys.path.append(os.path.dirname(__file__))
from live_unity_env_NO_CROP import LiveUnityEnv

class ActionRepeatWrapper:
    def __init__(self, env, repeat:int=1):
        self.env = env; self.repeat = max(1, int(repeat))
        self.action_space = env.action_space; self.observation_space = env.observation_space
    def reset(self, **kw): return self.env.reset(**kw)
    def step(self, action):
        total=0.0; info={}
        for i in range(self.repeat):
            obs, r, d, t, info = self.env.step(action)
            total += float(r)
            if d or t: break
        return obs, total, d, t, info
    def close(self): return self.env.close()
    def __getattr__(self,k): return getattr(self.env,k)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--img_size", type=int, nargs=2, default=[84,84])
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--repeat", type=int, default=1)
    args = p.parse_args()

    env = LiveUnityEnv(host=args.host, port=args.port, img_size=tuple(args.img_size), max_steps=args.max_steps)
    env = ActionRepeatWrapper(env, repeat=args.repeat)

    model = RecurrentPPO.load(args.model_path)

    for ep in range(args.episodes):
        obs, info = env.reset()
        state = None; episode_start = True; done=False; trunc=False; total=0.0
        while not (done or trunc):
            # model expects HWC; convert CHW->HWC and add batch
            x = np.transpose(obs, (1,2,0))[None, ...]
            action, state = model.predict(x, state=state, episode_start=np.array([episode_start]), deterministic=True)
            episode_start = False
            obs, r, done, trunc, info = env.step(action)
            total += float(r)
        print(f"Episode {ep+1}: reward={total:.3f}")

    env.close()

if __name__ == "__main__":
    main()

import argparse
from unity_camera_env import UnityCameraEnv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture_dir', type=str, default='../../Assets/Captures')
    ap.add_argument('--img_w', type=int, default=84)
    ap.add_argument('--img_h', type=int, default=84)
    ap.add_argument('--steps', type=int, default=10)
    args = ap.parse_args()

    env = UnityCameraEnv(capture_dir=args.capture_dir, img_size=(args.img_w, args.img_h), max_steps=args.steps)
    obs, info = env.reset()
    print('Observation shape:', obs.shape, 'info:', info)
    total = 0.0
    for t in range(args.steps):
        action = env.action_space.sample()
        obs, r, done, trunc, info = env.step(action)
        total += r
        print(f'Step {t+1}: reward={r:.3f} done={done} trunc={trunc}')
        if done or trunc:
            break
    print('Total reward:', total)

if __name__ == '__main__':
    main()
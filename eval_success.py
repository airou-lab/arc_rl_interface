import argparse, socket, struct
import numpy as np
import cv2

def recv_exact(s, n):
    data = b""
    while len(data) < n:
        chunk = s.recv(n - len(data))
        if not chunk:
            raise ConnectionError("socket closed")
        data += chunk
    return data

def run_episode(host, port, model=None, deterministic=True):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect((host, port))
        client.sendall(b"R")
        L = struct.unpack("!I", recv_exact(client, 4))[0]
        jpeg = recv_exact(client, L)
        reward = struct.unpack("!f", recv_exact(client, 4))[0]
        done = bool(recv_exact(client,1)[0]); truncated = bool(recv_exact(client,1)[0])

        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("decode fail")

        state = None
        episode_start = True
        total = 0.0
        steps = 0
        last_r = reward

        while True:
            if model is None:
                steer, throttle = float(np.random.uniform(-0.3, 0.3)), float(np.random.uniform(0.2, 0.6))
            else:
                obs = np.expand_dims(img, axis=0)
                action, state = model.predict(obs, state=state, episode_start=np.array([episode_start]), deterministic=deterministic)
                action = np.array(action).flatten()
                steer = float(np.clip(action[0], -1, 1))
                throttle = float(np.clip(action[1], 0, 1))
                episode_start = False

            client.sendall(struct.pack("!ff", steer, throttle))

            L = struct.unpack("!I", recv_exact(client, 4))[0]
            jpeg = recv_exact(client, L)
            tail = recv_exact(client, 6)
            r = struct.unpack("!f", tail[:4])[0]
            d = bool(tail[4]); t = bool(tail[5])

            total += r; steps += 1; last_r = r
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if d or t:
                return total, steps, last_r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--model_path", type=str, default=None)
    args = ap.parse_args()

    model = None
    if args.model_path:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(args.model_path)

    totals = []
    stepss = []
    goals = cols = timeouts = 0

    for i in range(args.episodes):
        tot, n, last_r = run_episode(args.host, args.port, model=model, deterministic=True)
        totals.append(tot); stepss.append(n)
        # Heuristic classification
        if last_r <= -0.5:
            term = "collision"; cols += 1
        elif last_r >= 1.0:
            term = "goal"; goals += 1
        else:
            term = "timeout"; timeouts += 1
        print(f"ep {i+1:02d}: return={tot:.3f} steps={n} term={term}")

    print("\nSummary")
    print(f"  mean return {np.mean(totals):.3f} ± {np.std(totals):.3f}")
    print(f"  mean steps  {np.mean(stepss):.1f}")
    print(f"  term counts {{'goal': {goals}, 'collision': {cols}, 'timeout': {timeouts}}}")

if __name__ == "__main__":
    main()

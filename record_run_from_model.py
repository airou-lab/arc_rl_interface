import argparse, socket, struct
import numpy as np
import cv2
from sb3_contrib import RecurrentPPO

def recv_exact(s, n):
    data = b""
    while len(data) < n:
        chunk = s.recv(n - len(data))
        if not chunk:
            raise ConnectionError("socket closed")
        data += chunk
    return data

def overlay_hud(img_bgr, steer, throttle, reward, step):
    h, w = img_bgr.shape[:2]
    hud = img_bgr.copy()

    # Steering indicator
    cx, cy = w // 2, h - 8
    bar_w = int((w - 20) * 0.5)
    cv2.rectangle(hud, (10, h - 16), (w - 10, h - 2), (255, 255, 255), 1)
    steer_x = int(cx + steer * bar_w)
    cv2.line(hud, (cx, h - 16), (cx, h - 2), (255, 255, 255), 1)
    cv2.circle(hud, (steer_x, h - 9), 4, (0, 255, 0), -1)

    # Throttle bar
    t_h = int((h - 20) * float(np.clip(throttle, 0.0, 1.0)))
    cv2.rectangle(hud, (4, h - 4 - t_h), (12, h - 4), (0, 255, 0), -1)
    cv2.rectangle(hud, (4, 16), (12, h - 4), (255, 255, 255), 1)

    # Text
    cv2.putText(hud, f"step {step}", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(hud, f"r {reward:+.3f}", (w-78, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)
    return hud

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--model_path", type=str, required=True, help="Path to RecurrentPPO .zip")
    ap.add_argument("--out_mp4", type=str, default="proof_run.mp4")
    ap.add_argument("--out_csv", type=str, default="proof_run.csv")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--deterministic", action="store_true", default=True)
    args = ap.parse_args()

    model = RecurrentPPO.load(args.model_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out_mp4, fourcc, args.fps, (84, 84))

    import csv
    fcsv = open(args.out_csv, "w", newline="")
    wcsv = csv.writer(fcsv)
    wcsv.writerow(["step","steer","throttle","reward","done","truncated"])

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((args.host, args.port))
        # Reset
        s.sendall(b"R")
        L = struct.unpack("!I", recv_exact(s, 4))[0]
        jpeg = recv_exact(s, L)
        reward = struct.unpack("!f", recv_exact(s, 4))[0]
        done = bool(recv_exact(s,1)[0]); truncated = bool(recv_exact(s,1)[0])
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to decode first JPEG")

        state = None
        episode_start = True
        step = 0

        while True:
            obs_hwc = img  # 84x84x3 uint8
            obs_batch = np.expand_dims(obs_hwc, axis=0)
            action, state = model.predict(obs_batch, state=state, episode_start=np.array([episode_start]), deterministic=args.deterministic)
            episode_start = False

            action = np.array(action).flatten()
            steer = float(np.clip(action[0], -1.0, 1.0))
            throttle = float(np.clip(action[1], 0.0, 1.0))

            s.sendall(struct.pack("!ff", steer, throttle))

            L = struct.unpack("!I", recv_exact(s, 4))[0]
            jpeg = recv_exact(s, L)
            tail = recv_exact(s, 6)
            reward = struct.unpack("!f", tail[:4])[0]
            done = bool(tail[4]); truncated = bool(tail[5])

            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                break

            hud = overlay_hud(img, steer, throttle, reward, step)
            vw.write(hud)
            wcsv.writerow([step, steer, throttle, reward, int(done), int(truncated)])

            step += 1
            if done or truncated:
                break

    vw.release()
    fcsv.close()
    print(f"Wrote {args.out_mp4} and {args.out_csv}")

if __name__ == "__main__":
    main()

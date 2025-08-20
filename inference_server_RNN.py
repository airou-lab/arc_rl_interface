import socket, struct, argparse
import numpy as np
import cv2
from sb3_contrib import RecurrentPPO

CROP_TOP_FRAC = 0.25

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--img_w", type=int, default=84)
parser.add_argument("--img_h", type=int, default=84)
args = parser.parse_args()

model = RecurrentPPO.load(args.model_path)

def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data

def preprocess(jpeg_bytes):
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h = img.shape[0]
    img = img[int(h * CROP_TOP_FRAC):]
    img = cv2.resize(img, (args.img_w, args.img_h))
    return img  # HWC uint8

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", args.port))
    s.listen(1)
    print(f"Listening on 0.0.0.0:{args.port} ...")

    while True:
        conn, addr = s.accept()
        print("Client connected:", addr)
        with conn:
            state = None
            episode_start = True
            while True:
                hdr = recv_exact(conn, 4)
                if hdr is None:
                    break
                (length,) = struct.unpack("!I", hdr)
                img_bytes = recv_exact(conn, length)
                if img_bytes is None:
                    break

                obs = preprocess(img_bytes)
                if obs is None:
                    break
                obs = np.expand_dims(obs, axis=0)  # batch dimension HWC
                action, state = model.predict(obs, state=state, episode_start=np.array([episode_start]), deterministic=True)
                episode_start = False

                action = np.array(action).flatten()
                steer = float(np.clip(action[0], -1.0, 1.0))
                throttle = float(np.clip(action[1], 0.0, 1.0))
                conn.sendall(struct.pack("!ff", steer, throttle))
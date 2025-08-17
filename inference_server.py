import os
import socket
import struct
import argparse
import numpy as np
import cv2
from stable_baselines3 import PPO
from unity_camera_env import UnityCameraEnv

CROP_TOP_FRAC = 0.25  # must match unity_camera_env

# -------------------
# Argument parsing
# -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to trained PPO model .zip file")
parser.add_argument("--capture_dir", type=str, default="Assets/Captures", help="Capture directory with camera_intrinsics.yaml")
parser.add_argument("--port", type=int, default=5555, help="TCP port to listen on")
args = parser.parse_args()

HOST = '0.0.0.0'
PORT = args.port
CAPTURE_DIR = args.capture_dir
MODEL_PATH = args.model_path

# -------------------
# Load PPO Model
# -------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
print(f"[INFO] Loading PPO model from: {MODEL_PATH}")
model = PPO.load(MODEL_PATH)
print("[INFO] Model loaded.")
print(f"[DEBUG] Action space from model: {model.action_space}")

# -------------------
# Create env for preprocessing
# -------------------
env_for_preproc = UnityCameraEnv(capture_dir=CAPTURE_DIR, save_debug_masks=False)
env_for_preproc.eval_mode = True

# -------------------
# Socket setup
# -------------------
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print(f"[READY] Listening on {HOST}:{PORT}")

# -------------------
# Utilities
# -------------------
def recv_exact(sock, length):
    data = b''
    while len(data) < length:
        packet = sock.recv(length - len(data))
        if not packet:
            return None
        data += packet
    return data

def preprocess_with_env(jpeg_bytes):
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG image")

    h = img.shape[0]
    cropped = img[int(h * CROP_TOP_FRAC):]
    resized = cv2.resize(cropped, env_for_preproc.img_size)

    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    chw = np.transpose(sharpened, (2, 0, 1)).astype(np.float32) / 255.0
    obs = np.expand_dims(chw, axis=0)
    return obs

# -------------------
# Main loop
# -------------------
while True:
    client_socket, addr = server_socket.accept()
    print(f"[CONNECTED] Client from {addr}")
    try:
        while True:
            length_bytes = recv_exact(client_socket, 4)
            if not length_bytes:
                break
            (length,) = struct.unpack('!I', length_bytes)

            image_data = recv_exact(client_socket, length)
            if image_data is None:
                break

            obs = preprocess_with_env(image_data)
            action, _ = model.predict(obs, deterministic=True)

            action = np.array(action).flatten()
            steering = float(np.clip(action[0], -1.0, 1.0))
            throttle = float(np.clip(action[1], 0.0, 1.0))

            # Send to Unity
            client_socket.send(struct.pack('!ff', steering, throttle))
            print(f"[ACTION] Sent steering={steering:.3f}, throttle={throttle:.3f}")

    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        client_socket.close()
        print(f"[DISCONNECTED] {addr}")
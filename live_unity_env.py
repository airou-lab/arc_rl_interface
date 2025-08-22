import socket, struct
from typing import Optional, Dict, Any, Tuple
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

class LiveUnityEnv(gym.Env):
    """
    Live passive env over TCP.
    Unity sends already-resized frames (e.g., 84x84). No Python crop.
    Protocol:
      RESET: send b'R' -> recv len|jpeg|reward|done|truncated
      STEP:  send !ff steer,throttle -> recv len|jpeg|reward|done|truncated
    """
    metadata = {"render_modes": []}

    def __init__(self, host="127.0.0.1", port=5556, img_size=(84,84), max_steps=500):
        super().__init__()
        self.host, self.port = host, port
        self.img_size = tuple(img_size)
        self.max_steps = max_steps
        self.sock = None
        self.steps = 0
        self.action_space = spaces.Box(low=np.array([-1.0,0.0],np.float32), high=np.array([1.0,1.0],np.float32), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, self.img_size[1], self.img_size[0]), dtype=np.uint8)

    def _connect(self):
        if self.sock: return
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        self.sock = s

    def _recv_exact(self, n:int):
        data = b""
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def _recv_obs_reward_done(self):
        hdr = self._recv_exact(4)
        if hdr is None: return None, 0.0, True, False
        (length,) = struct.unpack("!I", hdr)
        img_bytes = self._recv_exact(length)
        if img_bytes is None: return None, 0.0, True, False
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None: return None, 0.0, True, False
        # Ensure size matches declared
        if (img.shape[1], img.shape[0]) != self.img_size:
            img = cv2.resize(img, self.img_size)
        tail = self._recv_exact(6)
        if tail is None or len(tail)!=6: return None, 0.0, True, False
        reward = struct.unpack("!f", tail[:4])[0]
        done = bool(tail[4]); trunc = bool(tail[5])
        chw = np.transpose(img, (2,0,1)).astype(np.uint8)
        return chw, float(reward), done, trunc

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._connect()
        self.sock.sendall(b'R')
        self.steps = 0
        obs, _, done, _ = self._recv_obs_reward_done()
        if obs is None or done:
            # retry once
            self.sock.close(); self.sock=None
            self._connect(); self.sock.sendall(b'R')
            obs, _, _, _ = self._recv_obs_reward_done()
        return obs, {"reset": True}

    def step(self, action):
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))
        self.sock.sendall(struct.pack("!ff", steer, throttle))
        obs, reward, done, trunc = self._recv_obs_reward_done()
        self.steps += 1
        if self.max_steps and self.steps >= self.max_steps:
            trunc = True
        if obs is None:
            done = True
            obs = np.zeros((3, self.img_size[1], self.img_size[0]), np.uint8)
        return obs, reward, done, trunc, {}

    def close(self):
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            self.sock.close(); self.sock=None
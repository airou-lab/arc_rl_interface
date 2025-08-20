import socket, struct
from typing import Optional, Dict, Any, Tuple
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

CROP_TOP_FRAC = 0.25

class LiveUnityEnv(gym.Env):
    """
    Live passive env over a TCP socket. Agent sees RGB only (CHW in env; use VecTransposeImage outside).
    Protocol expected from Unity:
      - On reset: Python sends b'R'; Unity responds with (len, jpeg, reward, done, truncated).
      - On step: Python sends struct('!ff') steering, throttle; Unity responds with (len, jpeg, reward, done, truncated).
      Response format:
        struct('!I') length, then JPEG bytes, then struct('!f??') => (reward: float32, done: bool, truncated: bool)
    """
    metadata = {"render_modes": []}

    def __init__(self, host: str = "127.0.0.1", port: int = 5556, img_size: Tuple[int, int] = (84, 84), max_steps: Optional[int] = 500):
        super().__init__()
        self.host, self.port = host, port
        self.img_size = tuple(img_size)
        self.max_steps = max_steps or 500
        self.sock: Optional[socket.socket] = None
        self.steps = 0

        self.action_space = spaces.Box(low=np.array([-1.0, 0.0], np.float32),
                                       high=np.array([1.0, 1.0], np.float32),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, self.img_size[1], self.img_size[0]), dtype=np.uint8)

    def _connect(self):
        if self.sock: return
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        self.sock = s

    def _recv_exact(self, n: int) -> Optional[bytes]:
        data = b""
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def _recv_obs_reward_done(self):
        hdr = self._recv_exact(4)
        if hdr is None:
            return None, 0.0, True, False
        (length,) = struct.unpack("!I", hdr)
        img_bytes = self._recv_exact(length)
        if img_bytes is None:
            return None, 0.0, True, False
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, 0.0, True, False
        h = img.shape[0]
        img = img[int(h * CROP_TOP_FRAC):]
        img = cv2.resize(img, self.img_size)
        tail = self._recv_exact(6)  # 4 bytes float + 2 bool bytes
        if tail is None or len(tail) != 6:
            return None, 0.0, True, False
        reward = struct.unpack("!f", tail[:4])[0]
        done = bool(tail[4])
        truncated = bool(tail[5])
        chw = np.transpose(img, (2, 0, 1)).astype(np.uint8)
        return chw, float(reward), done, truncated

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._connect()
        self.sock.sendall(b'R')
        self.steps = 0
        obs, _, done, _ = self._recv_obs_reward_done()
        return obs, {"reset": True}

    def step(self, action: np.ndarray):
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))
        self.sock.sendall(struct.pack("!ff", steer, throttle))
        obs, reward, done, truncated = self._recv_obs_reward_done()
        self.steps += 1
        if self.max_steps and self.steps >= self.max_steps:
            truncated = True
        if obs is None:
            done = True
            obs = np.zeros((3, self.img_size[1], self.img_size[0]), np.uint8)
        return obs, reward, done, truncated, {}

    def close(self):
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            self.sock.close()
            self.sock = None

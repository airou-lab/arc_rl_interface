"""
LiveUnityEnv
Gymnasium-compatible environment that connects to the Unity scene
(RLClientSender.cs) over TCP and exchanges:

Client (Python) -> Unity:
    - Reset: send a single byte 'R'
    - Step: send 8 bytes = big-endian float32 steer, float32 throttle

Unity -> Client (Pythin):
    - 4-byte big-endian uint32: length N of JPEG payload (can be zero)
    - N bytes: JPEG frame (RGB)
    - 6-byte tail: big-endian float32 reward, 1 byte done, 1 byte truncated

Obs: unit8 RGB image, shape (H, W, 3), HWC
Action: Box([-1, 0], [1, 1]) -> (steer, throttle)
Reward/Done: Propagated from Unity

This env is minimal and passive: the policy only sees the RGB frame and sends
continuous actions.
"""
from __future__ import annotations
import socket
import struct
import time
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image
import io

# helpers
def _read_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes or raise ConnectionError."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while reading")
        buf.extend(chunk)
    return bytes(buf)

def _be_f32_to_bytes(x: float) -> bytes:
    return struct.pack(">f", float(x))

def _decode_jpeg_to_rgb(data: bytes, expected_hw: Tuple[int, int]) -> np.ndarray:
    """Decode a JPEG bytes object to RGB uint8 HWC."""
    if not data:
        # If Unity sends zero-length (shouldn't), return black frame
        h, w = expected_hw
        return np.zeros((h, w, 3), dtype=np.uint8)

    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        # Unity renders the exact size; resize only if mismatched
        if im.size != (expected_hw[1], expected_hw[0]):
            im = im.resize((expected_hw[1], expected_hw[0]), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
        assert arr.ndim == 3 and arr.shape[2] == 3, f"bad image shape {arr.shape}"
        return arr

# config
@dataclass
class UnityEnvConfig:
    host: str = "127.0.0.1"
    port: int = 5556
    img_width: int = 84
    img_height: int = 84
    max_steps: int = 500
    connect_timeout_s: float = 15.0
    socket_timeout_s: float = 15.0
    action_repeat: int = 1 # prefer using ActionRepeatWrapper, but kept here for convenience

# env
class LiveUnityEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: UnityEnvConfig | None = None, **kwargs):
        super().__init__()
        if cfg is None:
            cfg = UnityEnvConfig(**kwargs)
        # allow kwargs override
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        self.cfg = cfg
        h, w = self.cfg.img_height, self.cfg.img_width
        self.observation_space = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0], dtype=np.float32),
                                       high=np.array([+1.0, 1.0], dtype=np.float32),
                                       dtype=np.float32)
        self._sock: Optional[socket.socket] = None
        self._step_count: int = 0
        self._last_info: Dict[str, Any] = {}

    # socket lifecycle
    def _connect(self) -> None:
        if self._sock is not None:
            return
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.cfg.connect_timeout_s)
        s.connect((self.cfg.host, self.cfg.port))
        s.settimeout(self.cfg.socket_timeout_s)
        self._sock = s

    def _close_socket(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    # gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._connect()

        # Request a new episode
        assert self._sock is not None
        self._sock.sendall(b"R")

        # Receive the first observation immediately after reset
        obs, reward, done, trunc, info = self._recv_step()
        self._step_count = 0
        self._last_info = info
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(2)
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))

        # Repeat (kept here for convenience; can also wrap with ActionRepeatWrapper)
        total_reward = 0.0
        last_obs = None
        last_done = False
        last_trunc = False
        info = {}

        repeats = max(1, int(self.cfg.action_repeat))
        for _ in range(repeats):
            self._send_action(steer, throttle)
            obs, reward, done, trunc, info = self._recv_step()
            total_reward += float(reward)
            last_obs = obs
            last_done = done
            last_trunc = trunc
            self._step_count += 1
            if done or trunc:
                break
        return last_obs, total_reward, last_done, last_trunc, info

    def close(self):
        self._close_socket()
        return super().close()

    # protocol helpers
    def _send_action(self, steer: float, throttle: float) -> None:
        """Send big-endian float32 steer and throttle."""
        assert self._sock is not None
        self._sock.sendall(_be_f32_to_bytes(steer) + _be_f32_to_bytes(throttle))

    def _recv_step(self):
        """
        Receive one step:
          - 4-byte length N
          - N bytes JPEG
          - 4-byte float reward + 1B done + 1B truncated
        """
        assert self._sock is not None
        # length
        hdr = _read_exact(self._sock, 4)
        (n,) = struct.unpack(">I", hdr)
        # jpeg
        jpeg = _read_exact(self._sock, n) if n > 0 else b""
        # tail
        tail = _read_exact(self._sock, 6)
        (reward,) = struct.unpack(">f", tail[:4])
        done = bool(tail[4])
        truncated = bool(tail[5])

        obs = _decode_jpeg_to_rgb(jpeg, expected_hw=(self.cfg.img_height, self.cfg.img_width))
        info = {
            "step": self._step_count,
            "jpeg_bytes": n,
            "reward": float(reward),
            "done": done,
            "truncated": truncated,
        }
        return obs, float(reward), done, truncated, info
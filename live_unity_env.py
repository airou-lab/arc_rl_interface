"""
LiveUnityEnv - TCP client connecting to Unity simulation
UPDATED: Defaults to 128x128 resolution for robotics-grade vision.
"""

import socket
import struct
import io
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image
from typing import Optional, Tuple, Dict, Any


class LiveUnityEnv(gym.Env):
    """
    Gymnasium environment connecting to Unity via TCP.
    Protocol:
    - Commands: 'R' (reset), 'A' + action (step), 'W' + waypoints, 'Q' (quit)
    - Receives: JPEG image + telemetry floats + reward + done/truncated flags
    """

    metadata = {"render_modes": ["rgb_array"]}

    # Telemetry indices (12 floats from Unity)
    TEL_GOAL_COS = 0
    TEL_GOAL_SIN = 1
    TEL_GOAL_DIST = 2
    TEL_SPEED = 3
    TEL_YAW_RATE = 4
    TEL_LAST_STEER = 5
    TEL_LAST_THR = 6
    TEL_LAST_BRK = 7
    TEL_LAT_ERR = 8
    TEL_HDG_ERR = 9
    TEL_KAPPA = 10
    TEL_DS = 11

    # Extended telemetry for position (if Unity sends it)
    TEL_POS_X = 12
    TEL_POS_Y = 13
    TEL_POS_Z = 14
    TEL_YAW = 15

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        img_width: int = 128,  # UPDATED
        img_height: int = 128, # UPDATED
        max_steps: int = 500,
        timeout: float = 120.0,
        verbose: bool = False,
        step_delay: float = 0.0,
    ):
        super().__init__()

        self.host = host
        self.port = port
        self.img_width = img_width
        self.img_height = img_height
        self.max_steps = max_steps
        self.timeout = timeout
        self.verbose = verbose
        self.step_delay = step_delay

        # Connection state
        self.client_socket: Optional[socket.socket] = None
        self.connected = False

        # Episode state
        self.step_count = 0
        self.episode_count = 0
        self.last_obs: Optional[Dict[str, np.ndarray]] = None
        self.last_reward = 0.0
        self.last_done = False
        self.last_truncated = False

        # Position tracking
        self._car_position = np.zeros(3, dtype=np.float32)
        self._car_yaw = 0.0

        # Waypoint state
        self.last_waypoints: Optional[np.ndarray] = None
        self.waypoints_enabled = False

        # Define spaces
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255,
                shape=(img_height, img_width, 3),
                dtype=np.uint8
            ),
            "vec": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(12,),
                dtype=np.float32
            )
        })

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Connect on initialization
        self._connect()

    def _connect(self):
        if self.connected: return
        try:
            if self.verbose: print(f"[LiveUnityEnv] Connecting to Unity at {self.host}:{self.port}...")
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.settimeout(self.timeout)
            self.client_socket.connect((self.host, self.port))
            self.connected = True
            if self.verbose: print(f"[LiveUnityEnv] Connected to Unity at {self.host}:{self.port}")
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Failed to connect to Unity: {e}")

    def _reconnect(self):
        print("[LiveUnityEnv] Connection lost. Attempting to reconnect...")
        self._cleanup()
        for i in range(5):
            try:
                time.sleep(1.0)
                self._connect()
                print("[LiveUnityEnv] Reconnected successfully!")
                return
            except Exception as e:
                print(f"[LiveUnityEnv] Reconnection attempt {i+1} failed: {e}")
        raise RuntimeError("Failed to reconnect to Unity after multiple attempts.")

    def _send_command(self, command: bytes):
        if not self.connected or self.client_socket is None: self._reconnect()
        try: self.client_socket.sendall(command)
        except (BrokenPipeError, ConnectionResetError):
            self._reconnect()
            self.client_socket.sendall(command)
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Failed to send command: {e}")

    def _send_action(self, action: np.ndarray):
        if not self.connected or self.client_socket is None: self._reconnect()
        action = np.asarray(action, dtype=np.float32)
        msg = b'A' + struct.pack('<fff', float(action[0]), float(action[1]), float(action[2]))
        try: self.client_socket.sendall(msg)
        except (BrokenPipeError, ConnectionResetError):
            self._reconnect()
            self.client_socket.sendall(msg)
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Failed to send action: {e}")

    def _receive_observation(self) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if not self.connected or self.client_socket is None: self._reconnect()
        try:
            jpeg_len = struct.unpack('>I', self._recv_exactly(4))[0]
            jpeg_data = self._recv_exactly(jpeg_len)

            img = Image.open(io.BytesIO(jpeg_data))
            img_array = np.array(img, dtype=np.uint8)
            if img_array.ndim == 2: img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4: img_array = img_array[:, :, :3]

            if img_array.shape != (self.img_height, self.img_width, 3):
                raise RuntimeError(f"Image shape mismatch: got {img_array.shape}, expected ({self.img_height}, {self.img_width}, 3)")

            telemetry = struct.unpack('>12f', self._recv_exactly(12 * 4))
            vec_obs = np.array(telemetry, dtype=np.float32)

            reward = struct.unpack('>f', self._recv_exactly(4))[0]
            done = bool(self._recv_exactly(1)[0])
            truncated = bool(self._recv_exactly(1)[0])

            self._update_position_estimate(vec_obs[3], vec_obs[4], vec_obs[11])

            obs = {"image": img_array, "vec": vec_obs}
            info = {
                "goal_cos": vec_obs[0], "goal_sin": vec_obs[1], "goal_dist": vec_obs[2],
                "speed": vec_obs[3], "yaw_rate": vec_obs[4],
                "lat_err": vec_obs[8], "hdg_err": vec_obs[9], "kappa": vec_obs[10], "ds": vec_obs[11],
                "car_position": self._car_position.copy(), "car_yaw": self._car_yaw,
                "step": self.step_count, "episode_id": self.episode_count,
            }
            return obs, reward, done, truncated, info
        except (BrokenPipeError, ConnectionResetError):
            self._cleanup()
            raise RuntimeError("Connection reset by Unity during observation read.")
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Failed to receive observation: {e}")

    def _update_position_estimate(self, speed: float, yaw_rate: float, ds: float):
        dt = 0.02
        self._car_yaw += yaw_rate * dt
        while self._car_yaw > np.pi: self._car_yaw -= 2 * np.pi
        while self._car_yaw < -np.pi: self._car_yaw += 2 * np.pi
        if abs(ds) > 0.001:
            self._car_position[0] += ds * np.sin(self._car_yaw)
            self._car_position[2] += ds * np.cos(self._car_yaw)

    def _recv_exactly(self, num_bytes: int) -> bytes:
        data = b''
        while len(data) < num_bytes:
            chunk = self.client_socket.recv(num_bytes - len(data))
            if not chunk: raise ConnectionResetError("Connection closed by Unity")
            data += chunk
        return data

    def enable_waypoint_visualization(self): self.waypoints_enabled = True
    def disable_waypoint_visualization(self): self.waypoints_enabled = False

    def send_waypoints(self, waypoints: np.ndarray):
        if not self.waypoints_enabled or not self.connected or self.client_socket is None: return
        self.last_waypoints = waypoints.copy()
        waypoints = np.asarray(waypoints, dtype=np.float32)
        if waypoints.ndim != 2 or waypoints.shape[1] != 2: return
        num = waypoints.shape[0]
        if num == 0 or num > 255: return
        try:
            msg = b'W' + struct.pack('<B', num)
            for i in range(num): msg += struct.pack('<ff', float(waypoints[i,0]), float(waypoints[i,1]))
            self.client_socket.sendall(msg)
        except Exception: pass

    def set_waypoints(self, waypoints: np.ndarray): self.send_waypoints(waypoints)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._car_position = np.zeros(3, dtype=np.float32)
        self._car_yaw = 0.0
        try:
            self._send_command(b'R')
            obs, reward, done, truncated, info = self._receive_observation()
        except (RuntimeError, ConnectionResetError, BrokenPipeError):
            self._reconnect()
            self._send_command(b'R')
            obs, reward, done, truncated, info = self._receive_observation()

        self.step_count = 0
        self.episode_count += 1
        self.last_obs = obs
        self.last_reward = 0.0
        self.last_done = False
        self.last_truncated = False
        self.last_waypoints = None
        info['episode_id'] = self.episode_count
        return obs, info

    def step(self, action: np.ndarray):
        try:
            self._send_action(action)
            if self.waypoints_enabled and self.last_waypoints is not None:
                self.send_waypoints(self.last_waypoints)
            obs, reward, done, truncated, info = self._receive_observation()
        except (RuntimeError, ConnectionResetError, BrokenPipeError):
            self._reconnect()
            self._send_command(b'R')
            obs, reward, done, truncated, info = self._receive_observation()
            done = True; truncated = True; info['connection_reset'] = True
            self.step_count = 0
            self._car_position = np.zeros(3, dtype=np.float32)
            self._car_yaw = 0.0

        self.step_count += 1
        self.last_obs = obs
        self.last_reward = reward
        self.last_done = done
        self.last_truncated = truncated
        if self.step_delay > 0: time.sleep(self.step_delay)
        info['step'] = self.step_count
        info['episode_id'] = self.episode_count
        if self.step_count >= self.max_steps and not done:
            truncated = True
            info['TimeLimit.truncated'] = True
        return obs, reward, done, truncated, info

    def render(self): return self.last_obs["image"] if self.last_obs is not None else None
    def close(self):
        if self.connected:
            try: self._send_command(b'Q')
            except: pass
        self._cleanup()
    def _cleanup(self):
        self.connected = False
        if self.client_socket is not None:
            try: self.client_socket.close()
            except: pass
            self.client_socket = None
    def __del__(self): self.close()
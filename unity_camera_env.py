import os
import glob
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

# ---- Strict passive constants ----
CROP_TOP_FRAC = 0.25 # MUST match inference/training if used there

class UnityCameraEnv(gym.Env):
    """
    Strictly passive visual environment for offline frames.
    - Observation: RGB only (cropped/resized/enhanced), channel-first (C,H,W) uint8
    - No masks, no optical flow, no path planning, no action blending.
    - Reward: action-only placeholder (alive + forward intent + smoothness).
    - Termination: time-based only (offline), not a true MDP (good for smoke tests).
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        capture_dir: str = "Assets/Captures",
        img_size: Tuple[int, int] = (84, 84),
        max_steps: Optional[int] = None,
        use_augmentation: bool = True,
    ):
        super().__init__()
        self.capture_dir = os.path.abspath(capture_dir)
        self.img_size = tuple(img_size)
        self.max_steps = max_steps or 500 # fallback
        self.use_augmentation = bool(use_augmentation)

        # Action space: steer [-1,1], throttle [0,1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space: CHW uint8
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, self.img_size[1], self.img_size[0]), dtype=np.uint8
        )

        # State
        self.rgb_paths: List[str] = []
        self.step_idx: int = 0
        self.prev_steer: float = 0.0
        self.eval_mode: bool = False # can be flipped externally

    # ---- Utilities ----
    def _load_images(self) -> List[str]:
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        files: List[str] = []
        for e in exts:
            files.extend(glob.glob(os.path.join(self.capture_dir, e)))
        return sorted(files)

    def _read_obs(self, idx: int) -> np.ndarray:
        if idx >= len(self.rgb_paths):
            return np.zeros((3, self.img_size[1], self.img_size[0]), np.uint8)
        img = cv2.imread(self.rgb_paths[idx], cv2.IMREAD_COLOR)
        if img is None:
            return np.zeros((3, self.img_size[1], self.img_size[0]), np.uint8)
        h = img.shape[0]
        img = img[int(h * CROP_TOP_FRAC):]
        img = cv2.resize(img, self.img_size)

        # Passive enhancement: LAB equalization + mild sharpening (no semantics)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        img = cv2.filter2D(img, -1, kernel)

        chw = np.transpose(img, (2, 0, 1))
        return chw.astype(np.uint8)

    def _get_info(self) -> Dict[str, Any]:
        # Minimal info; no perception metrics.
        return {
            "step": int(self.step_idx),
            "total_frames": int(len(self.rgb_paths)),
            "eval_mode": bool(self.eval_mode),
        }

    # ---- Gym API ----
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.rgb_paths = self._load_images()
        if len(self.rgb_paths) == 0:
            self.max_steps = 1
        self.step_idx = 0
        self.prev_steer = 0.0
        obs = self._read_obs(self.step_idx)
        return obs, self._get_info()

    def step(self, action: np.ndarray):
        # Clamp and unpack action
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))

        # Passive, action-only reward
        reward = 0.0
        reward += 0.01 # alive
        reward += 0.02 * throttle # forward intent
        reward -= 0.01 * abs(steer - self.prev_steer) # smoothness
        reward -= 0.005 * abs(steer) # regularization

        self.prev_steer = steer

        # Advance
        self.step_idx += 1
        terminated = (self.step_idx >= min(self.max_steps, len(self.rgb_paths))) if len(self.rgb_paths) > 0 else True
        truncated = False

        obs = self._read_obs(self.step_idx) if not terminated else self._read_obs(max(0, len(self.rgb_paths) - 1))
        return obs, float(reward), bool(terminated), bool(truncated), self._get_info()

    def render(self):
        return None

    def close(self):
        pass
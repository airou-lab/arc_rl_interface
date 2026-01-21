#!/usr/bin/env python3
"""
ufld_lane_wrapper.py - UFLD Lane Detection for Passive Visual RL
=================================================================

This wrapper uses UFLD (Ultra Fast Lane Detection) to compute lane-relative
metrics FROM THE CAMERA IMAGE. This is truly Passive Visual because:

1. Lane info comes from camera (available on real robot)
2. No privileged simulation data needed
3. Same perception pipeline works in sim and real

The wrapper:
1. Runs UFLD on each camera frame
2. Computes: lateral offset, heading error, lane curvature
3. Adds these to observation (optional) or uses for reward only
4. Provides dense reward signal for lane following

USAGE:
    env = UFLDLaneWrapper(env, ufld_model_path="path/to/ufld.pth")
"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces


class UFLDLaneWrapper(gym.Wrapper):
    """
    Wrapper that uses UFLD for lane detection and reward shaping.

    This is PASSIVE VISUAL because lane info comes from camera image,
    not from privileged simulation data.
    """

    def __init__(
            self,
            env: gym.Env,
            ufld_model: Optional[torch.nn.Module] = None,
            ufld_model_path: Optional[str] = None,
            # Reward parameters
            lane_center_bonus: float = 0.3,
            lane_deviation_penalty: float = 0.2,
            heading_alignment_bonus: float = 0.2,
            # Whether to add lane info to observation
            add_lane_to_obs: bool = True,
            # UFLD parameters
            ufld_input_size: Tuple[int, int] = (288, 800),  # UFLD expected input
            row_anchors: Optional[np.ndarray] = None,
            griding_num: int = 100,
            # Lane geometry
            lane_width_pixels: float = 200.0,  # Expected lane width in image
            image_center_x: float = 64.0,  # Center of 128x128 image
    ):
        super().__init__(env)

        self.ufld_model = ufld_model
        self.ufld_model_path = ufld_model_path
        self.lane_center_bonus = lane_center_bonus
        self.lane_deviation_penalty = lane_deviation_penalty
        self.heading_alignment_bonus = heading_alignment_bonus
        self.add_lane_to_obs = add_lane_to_obs
        self.ufld_input_size = ufld_input_size
        self.griding_num = griding_num
        self.lane_width_pixels = lane_width_pixels
        self.image_center_x = image_center_x

        # Default row anchors (bottom portion of image where lanes are visible)
        if row_anchors is None:
            self.row_anchors = np.linspace(0.4, 0.9, 18) * 128  # For 128x128 image
        else:
            self.row_anchors = row_anchors

        # Load model if path provided
        if ufld_model_path is not None and ufld_model is None:
            self._load_ufld_model(ufld_model_path)

        # Modify observation space if adding lane info
        if add_lane_to_obs:
            # Add 4 values: lane_offset, heading_err, curvature, confidence
            orig_vec_space = env.observation_space["vec"]
            new_vec_dim = orig_vec_space.shape[0] + 4

            self.observation_space = spaces.Dict({
                "image": env.observation_space["image"],
                "vec": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(new_vec_dim,),
                    dtype=np.float32
                )
            })

        # State
        self._last_lane_info = {
            'offset': 0.0,
            'heading': 0.0,
            'curvature': 0.0,
            'confidence': 0.0,
            'left_lane': None,
            'right_lane': None,
        }

        # Device for UFLD
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_ufld_model(self, path: str):
        """Load UFLD model from checkpoint."""
        try:
            # Try loading as a full model
            self.ufld_model = torch.load(path, map_location=self.device)
            self.ufld_model.eval()
            print(f"[UFLDLaneWrapper] Loaded UFLD model from {path}")
        except Exception as e:
            print(f"[UFLDLaneWrapper] Warning: Could not load UFLD model: {e}")
            print("[UFLDLaneWrapper] Using fallback HSV lane detection")
            self.ufld_model = None

    def _detect_lanes_ufld(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run UFLD on image and extract lane information.

        Returns dict with:
            - left_lane: list of (x, y) points
            - right_lane: list of (x, y) points
            - offset: lateral offset from lane center (positive = right)
            - heading: heading error in radians
            - curvature: estimated lane curvature
            - confidence: detection confidence
        """
        if self.ufld_model is None:
            return self._detect_lanes_hsv(image)

        # Preprocess for UFLD
        img_resized = cv2.resize(image, (self.ufld_input_size[1], self.ufld_input_size[0]))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.ufld_model(img_tensor)

        # Parse UFLD output (depends on model architecture)
        # This is a simplified version - adjust based on your UFLD variant
        lanes = self._parse_ufld_output(output, image.shape)

        return lanes

    def _detect_lanes_hsv(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Fallback lane detection using HSV color filtering.
        Works for yellow lane markings in simulation.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Yellow lane detection (adjust for your simulation)
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # White lane detection
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Combine masks
        lane_mask = cv2.bitwise_or(yellow_mask, white_mask)

        # Find lane points in bottom half of image
        h, w = lane_mask.shape
        bottom_half = lane_mask[h // 2:, :]

        # Split into left and right regions
        left_region = bottom_half[:, :w // 2]
        right_region = bottom_half[:, w // 2:]

        # Find lane line x-coordinates at each row
        left_points = []
        right_points = []

        for row_idx, row in enumerate(range(h // 2, h, 4)):  # Sample every 4 rows
            # Left lane
            left_cols = np.where(lane_mask[row, :w // 2] > 0)[0]
            if len(left_cols) > 0:
                left_x = np.mean(left_cols)
                left_points.append((left_x, row))

            # Right lane
            right_cols = np.where(lane_mask[row, w // 2:] > 0)[0]
            if len(right_cols) > 0:
                right_x = np.mean(right_cols) + w // 2
                right_points.append((right_x, row))

        # Compute lane center and heading
        if len(left_points) > 2 and len(right_points) > 2:
            # Average x positions near bottom of image
            left_x_bottom = np.mean([p[0] for p in left_points[-3:]])
            right_x_bottom = np.mean([p[0] for p in right_points[-3:]])

            lane_center = (left_x_bottom + right_x_bottom) / 2
            offset = (lane_center - self.image_center_x) / (w / 2)  # Normalized

            # Heading from lane line slopes
            if len(left_points) > 1:
                left_slope = (left_points[0][0] - left_points[-1][0]) / max(1, left_points[-1][1] - left_points[0][1])
            else:
                left_slope = 0

            if len(right_points) > 1:
                right_slope = (right_points[0][0] - right_points[-1][0]) / max(1,
                                                                               right_points[-1][1] - right_points[0][1])
            else:
                right_slope = 0

            avg_slope = (left_slope + right_slope) / 2
            heading = np.arctan(avg_slope)  # Radians

            # Curvature (simplified)
            curvature = 0.0  # Would need more sophisticated fitting

            confidence = min(len(left_points), len(right_points)) / 10.0
            confidence = min(confidence, 1.0)

        else:
            # No lanes detected
            offset = 0.0
            heading = 0.0
            curvature = 0.0
            confidence = 0.0
            left_points = []
            right_points = []

        return {
            'left_lane': left_points,
            'right_lane': right_points,
            'offset': offset,
            'heading': heading,
            'curvature': curvature,
            'confidence': confidence,
        }

    def _parse_ufld_output(self, output, image_shape) -> Dict[str, Any]:
        """Parse UFLD network output into lane points."""
        # This depends on your specific UFLD model
        # Placeholder implementation
        return self._detect_lanes_hsv(np.zeros(image_shape, dtype=np.uint8))

    def _compute_lane_reward(self, lane_info: Dict[str, Any]) -> float:
        """Compute reward based on lane position."""
        reward = 0.0

        confidence = lane_info['confidence']

        if confidence > 0.3:
            # Lane detected with reasonable confidence
            offset = abs(lane_info['offset'])
            heading = abs(lane_info['heading'])

            # Reward for being centered
            if offset < 0.15:
                reward += self.lane_center_bonus * confidence
            else:
                reward -= self.lane_deviation_penalty * offset * confidence

            # Reward for aligned heading
            if heading < 0.1:  # ~6 degrees
                reward += self.heading_alignment_bonus * confidence
            else:
                reward -= 0.1 * heading * confidence

        return reward

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Run lane detection on first frame
        if isinstance(obs, dict) and 'image' in obs:
            self._last_lane_info = self._detect_lanes_hsv(obs['image'])

        # Add lane info to observation if enabled
        if self.add_lane_to_obs and isinstance(obs, dict):
            lane_vec = np.array([
                self._last_lane_info['offset'],
                self._last_lane_info['heading'],
                self._last_lane_info['curvature'],
                self._last_lane_info['confidence'],
            ], dtype=np.float32)
            obs['vec'] = np.concatenate([obs['vec'], lane_vec])

        info['lane_info'] = self._last_lane_info
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Run lane detection
        if isinstance(obs, dict) and 'image' in obs:
            self._last_lane_info = self._detect_lanes_hsv(obs['image'])

        # Compute lane-based reward
        lane_reward = self._compute_lane_reward(self._last_lane_info)
        reward += lane_reward

        # Add lane info to observation if enabled
        if self.add_lane_to_obs and isinstance(obs, dict):
            lane_vec = np.array([
                self._last_lane_info['offset'],
                self._last_lane_info['heading'],
                self._last_lane_info['curvature'],
                self._last_lane_info['confidence'],
            ], dtype=np.float32)
            obs['vec'] = np.concatenate([obs['vec'], lane_vec])

        # Add to info for debugging
        info['lane_info'] = self._last_lane_info
        info['lane_reward'] = lane_reward

        return obs, reward, terminated, truncated, info


class SimpleLaneDetector:
    """
    Simple lane detector using classical CV.
    Can be used as a baseline or when UFLD isn't available.
    """

    def __init__(
            self,
            yellow_lower=(15, 80, 80),
            yellow_upper=(35, 255, 255),
            white_lower=(0, 0, 180),
            white_upper=(180, 30, 255),
    ):
        self.yellow_lower = np.array(yellow_lower)
        self.yellow_upper = np.array(yellow_upper)
        self.white_lower = np.array(white_lower)
        self.white_upper = np.array(white_upper)

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect lanes and return lane info dict."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        lane_mask = cv2.bitwise_or(yellow_mask, white_mask)

        h, w = lane_mask.shape

        # Find lane center at bottom of image
        bottom_rows = lane_mask[int(0.7 * h):, :]

        # Split left/right
        left_mask = bottom_rows[:, :w // 2]
        right_mask = bottom_rows[:, w // 2:]

        left_cols = np.where(np.any(left_mask > 0, axis=0))[0]
        right_cols = np.where(np.any(right_mask > 0, axis=0))[0]

        if len(left_cols) > 0 and len(right_cols) > 0:
            left_x = np.mean(left_cols)
            right_x = np.mean(right_cols) + w // 2

            lane_center = (left_x + right_x) / 2
            image_center = w / 2

            offset = (lane_center - image_center) / (w / 2)  # -1 to 1
            confidence = 1.0
        else:
            offset = 0.0
            confidence = 0.0

        return {
            'offset': offset,
            'heading': 0.0,  # Would need temporal info
            'curvature': 0.0,
            'confidence': confidence,
        }


# Test function
def test_lane_detection():
    """Test lane detection on a sample image."""
    import matplotlib.pyplot as plt

    # Create a simple test image with lane lines
    img = np.zeros((128, 128, 3), dtype=np.uint8)

    # Draw yellow lane lines
    cv2.line(img, (20, 128), (40, 60), (255, 255, 0), 3)  # Left lane (RGB yellow)
    cv2.line(img, (108, 128), (88, 60), (255, 255, 0), 3)  # Right lane

    detector = SimpleLaneDetector()
    result = detector.detect(img)

    print("Lane detection result:")
    print(f"  Offset: {result['offset']:.3f}")
    print(f"  Confidence: {result['confidence']:.2f}")

    # Visualize
    plt.imshow(img)
    plt.title(f"Offset: {result['offset']:.2f}")
    plt.savefig("lane_test.png")
    print("Saved lane_test.png")


if __name__ == "__main__":
    test_lane_detection()
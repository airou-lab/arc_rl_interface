"""
UnityCameraEnv – Enhanced Gym-compatible environment with robust lane detection.

Enhanced Features:
- Edge-based detection with Hough transform for thin lines
- Optical flow for temporal consistency
- Bezier curve path planning
- Confidence-based fallback strategies
- Improved reward shaping with multiple components
- Data augmentation for training robustness

Original Features Maintained:
- Crop bottom 75% (keeps buildings in view)
- L-channel adaptive threshold + HSV yellow + LAB-B boost
- Horizon suppression + orientation filtering
- Adaptive morphology
- Trapezoid road ROI
- Connected component analysis
- Debug visualization support
"""

import os
import csv
import yaml
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from natsort import natsorted

# ---- Strict passive mode constants ----
STRICT_PASSIVE = True
CROP_TOP_FRAC = 0.25  # must match inference_server




# ========================= Optical Flow Navigator ========================= #
class OpticalFlowNavigator:
    """Temporal consistency through optical flow analysis"""

    def __init__(self):
        self.prev_gray = None
        self.flow_history = []
        self.max_history = 5

    def compute_flow_guidance(self, current_frame: np.ndarray) -> Tuple[float, float]:
        """
        Compute steering guidance from optical flow
        Returns: (steering_suggestion, confidence)
        """
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is not None:
            # Calculate dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            # Focus on road region (bottom half)
            h, w = gray.shape
            road_flow = flow[h//2:, :]

            # Analyze horizontal flow component for steering
            horizontal_flow = road_flow[:, :, 0]

            # Split into left and right regions
            left_flow = np.mean(horizontal_flow[:, :w//2])
            right_flow = np.mean(horizontal_flow[:, w//2:])

            # Compute flow divergence as steering signal
            flow_divergence = (right_flow - left_flow) / (abs(right_flow) + abs(left_flow) + 1e-6)

            # Estimate confidence from flow magnitude
            magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
            confidence = min(1.0, np.mean(magnitude) / 5.0)  # Normalize to [0, 1]

            # Smooth with history
            self.flow_history.append(flow_divergence)
            if len(self.flow_history) > self.max_history:
                self.flow_history.pop(0)

            smoothed_steering = np.mean(self.flow_history)

            self.prev_gray = gray
            return float(smoothed_steering), float(confidence)

        self.prev_gray = gray
        return 0.0, 0.0

    def reset(self):
        """Reset flow navigator state"""
        self.prev_gray = None
        self.flow_history = []


# ========================= Enhanced Lane Detection ========================= #

def enhanced_lane_detection(resized_bgr: np.ndarray, roi: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Enhanced detection combining edge detection, Hough transform, and color-based methods
    Returns: (mask, confidence)
    """
    h, w = roi.shape[:2]

    # 1. Edge-based detection for thin lines
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive edge detection with dynamic thresholds
    median_val = np.median(filtered)
    lower = int(max(0, 0.66 * median_val))
    upper = int(min(255, 1.33 * median_val))
    edges = cv2.Canny(filtered, lower, upper)

    # Apply ROI to edges
    edges = cv2.bitwise_and(edges, roi)

    # 2. Hough Line Transform for lane segments
    lane_mask = np.zeros((h, w), dtype=np.uint8)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=20,  # Lower threshold for thin lines
        minLineLength=15,  # Shorter segments OK
        maxLineGap=20  # Allow larger gaps
    )

    line_confidence = 0.0
    if lines is not None:
        valid_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate line angle
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)

            # Filter for roughly vertical lines (lanes in perspective)
            if 45 < angle < 135:  # Vertical-ish
                cv2.line(lane_mask, (x1, y1), (x2, y2), 255, 2)
                valid_lines.append(line[0])

        # Calculate confidence from line detection
        if len(valid_lines) > 0:
            line_confidence = min(1.0, len(valid_lines) / 10.0)

    # 3. Dilate to create continuous regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    lane_mask = cv2.dilate(lane_mask, kernel, iterations=1)

    # 4. Color-based detection (existing method)
    color_mask, color_confidence = color_based_detection(resized_bgr, roi)

    # 5. Combine masks with confidence weighting
    combined_mask = np.zeros_like(lane_mask)

    if line_confidence > 0.3:
        # Strong line detection - use primarily
        combined_mask = cv2.bitwise_or(combined_mask, lane_mask)
        combined_confidence = line_confidence
    else:
        # Weak line detection - blend with color
        combined_mask = cv2.bitwise_or(lane_mask, color_mask)
        combined_confidence = max(line_confidence, color_confidence)

    # 6. Clean up with morphological operations
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE,
                                     np.ones((3, 3), np.uint8), iterations=1)

    # 7. Keep largest connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask, 8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) > 0:
            largest_idx = 1 + np.argmax(areas)
            combined_mask = (labels == largest_idx).astype(np.uint8) * 255

            # Recover nearby small components
            cx, cy = centroids[largest_idx]
            radius = max(10, int(0.15 * min(h, w)))
            for k in range(1, num_labels):
                if k != largest_idx:
                    xk, yk = centroids[k]
                    dist = np.sqrt((xk - cx)**2 + (yk - cy)**2)
                    if dist <= radius and stats[k, cv2.CC_STAT_AREA] > 10:
                        combined_mask = cv2.bitwise_or(combined_mask,
                                                      (labels == k).astype(np.uint8) * 255)

    return combined_mask.astype(np.uint8), combined_confidence


def color_based_detection(resized_bgr: np.ndarray, roi: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Color-based lane detection (original method with confidence)
    Returns: (mask, confidence)
    """
    lab = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Adaptive threshold on L channel
    l_mask = cv2.adaptiveThreshold(l, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, -5)

    # HSV yellow detection
    hsv = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
    lower_y = np.array([10, 40, 60], np.uint8)
    upper_y = np.array([55, 255, 255], np.uint8)
    y_mask = (cv2.inRange(hsv, lower_y, upper_y) > 0).astype(np.uint8)

    # B channel threshold
    b_thr = np.percentile(b, 88)
    b_mask = (b >= b_thr).astype(np.uint8)

    # Combine color masks
    combined = np.maximum(l_mask, np.maximum(y_mask, b_mask))
    combined = cv2.bitwise_and(combined, roi)

    # Calculate confidence based on coverage
    coverage = cv2.countNonZero(combined)
    target_coverage = int(0.06 * roi.shape[0] * roi.shape[1])
    confidence = min(1.0, coverage / max(1, target_coverage))

    return combined, confidence


# ========================= Path Planning ========================= #

def bezier_curve(points: np.ndarray, num_points: int = 100) -> np.ndarray:
    """Generate Bezier curve through control points"""
    n = len(points) - 1
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2))

    for i, point in enumerate(points):
        # Bernstein polynomial
        bern = np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))
        poly = bern * (t ** i) * ((1 - t) ** (n - i))
        curve[:, 0] += poly * point[0]
        curve[:, 1] += poly * point[1]

    return curve


def plan_bezier_path(lane_mask: np.ndarray, lookahead_pixels: int = 30) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Plan smooth path using Bezier curves
    Returns: (steering_angle, path_points)
    """
    h, w = lane_mask.shape

    # Extract lane center points from bottom to top
    center_points = []
    for y in range(h - 1, max(h // 2, h - lookahead_pixels), -3):
        row = lane_mask[y, :]
        if np.sum(row) > 0:
            indices = np.where(row > 0)[0]
            center_x = int(np.mean(indices))
            center_points.append([center_x, y])

    if len(center_points) < 4:
        return 0.0, []

    # Generate Bezier curve
    control_points = np.array(center_points[:min(6, len(center_points))])
    curve = bezier_curve(control_points, num_points=50)

    # Calculate steering from curve direction near vehicle
    near_points = curve[-10:]  # Points closest to vehicle
    if len(near_points) >= 2:
        # Calculate average heading
        dx = near_points[-1, 0] - near_points[0, 0]
        dy = near_points[-1, 1] - near_points[0, 1]

        # Convert to steering angle relative to image center
        image_center = w / 2
        target_x = near_points[-1, 0]
        steering = (target_x - image_center) / image_center

        # Apply smoothing based on path curvature
        if len(near_points) >= 3:
            curvature = np.std([p[0] for p in near_points]) / w
            steering *= (1.0 - min(0.5, curvature))  # Reduce steering on curves

        path_points = [(int(p[0]), int(p[1])) for p in curve]
        return float(np.clip(steering, -1.0, 1.0)), path_points

    return 0.0, []


# ========================= Data Augmentation ========================= #

def augment_observation(obs: np.ndarray, training: bool = True, intensity: float = 0.3) -> np.ndarray:
    """
    Apply data augmentation to improve training robustness
    Only applied during training, not evaluation
    """
    if not training:
        return obs

    # Convert from CHW to HWC for OpenCV
    obs_hwc = np.transpose(obs, (1, 2, 0))

    # Random brightness/contrast adjustment
    if np.random.rand() < intensity:
        alpha = np.random.uniform(0.8, 1.2)  # Contrast
        beta = np.random.randint(-20, 20)    # Brightness
        obs_hwc = cv2.convertScaleAbs(obs_hwc, alpha=alpha, beta=beta)

    # Add Gaussian noise
    if np.random.rand() < intensity * 0.5:
        noise = np.random.randn(*obs_hwc.shape) * 5
        obs_hwc = np.clip(obs_hwc + noise, 0, 255).astype(np.uint8)

    # Simulate motion blur (horizontal only for driving)
    if np.random.rand() < intensity * 0.3:
        kernel_size = np.random.choice([3, 5])
        kernel = np.ones((1, kernel_size)) / kernel_size
        obs_hwc = cv2.filter2D(obs_hwc, -1, kernel)

    # Random shadow simulation
    if np.random.rand() < intensity * 0.4:
        # Create random shadow polygon
        h, w = obs_hwc.shape[:2]
        shadow_pts = np.array([
            [np.random.randint(0, w//2), 0],
            [np.random.randint(w//2, w), 0],
            [np.random.randint(w//2, w), h],
            [np.random.randint(0, w//2), h]
        ], np.int32)

        mask = np.ones(obs_hwc.shape[:2], dtype=np.float32)
        cv2.fillPoly(mask, [shadow_pts], 0.5)
        obs_hwc = (obs_hwc * mask[:, :, np.newaxis]).astype(np.uint8)

    # Convert back to CHW
    return np.transpose(obs_hwc, (2, 0, 1))


# ========================= Original Helper Functions ========================= #

def _roi_top_y_from_mask(roi: np.ndarray) -> int:
    rows = np.where(roi.sum(axis=1) > 0)[0]
    return int(rows[0]) if rows.size else 0


def _apply_horizon_suppression(mask: np.ndarray, roi: np.ndarray, extra: float = 0.10) -> np.ndarray:
    H, _ = roi.shape[:2]
    top_y = _roi_top_y_from_mask(roi)
    cut = max(0, min(H - 1, int(top_y * (1.0 - extra))))
    horizon_mask = np.zeros_like(roi, dtype=np.uint8)
    horizon_mask[cut:] = 1
    return cv2.bitwise_and(mask, horizon_mask)


def _orientation_filter(bin_mask: np.ndarray, max_deg: float = 50.0) -> np.ndarray:
    if cv2.countNonZero(bin_mask.astype(np.uint8)) == 0:
        return bin_mask
    m8 = (bin_mask * 255).astype(np.uint8)
    gx = cv2.Sobel(m8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(m8, cv2.CV_32F, 0, 1, ksize=3)
    ang = np.degrees(np.arctan2(np.abs(gy), np.abs(gx)))
    keep = (ang <= max_deg)
    out = np.zeros_like(m8)
    out[keep] = m8[keep]
    out = (out > 0).astype(np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return out


def _primary_lane_mask(resized_bgr: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Original primary lane mask function - kept for compatibility"""
    H, W = roi.shape[:2]
    lab = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_mask = cv2.adaptiveThreshold(l, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, -5)

    hsv = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
    lower_y = np.array([10, 40, 60], np.uint8)
    upper_y = np.array([55, 255, 255], np.uint8)
    y_mask = (cv2.inRange(hsv, lower_y, upper_y) > 0).astype(np.uint8)

    b_thr = np.percentile(b, 88)
    b_mask = (b >= b_thr).astype(np.uint8)

    pre = np.maximum(l_mask, np.maximum(y_mask, b_mask))
    pre = cv2.bitwise_and(pre, roi)
    pre = _apply_horizon_suppression(pre, roi, extra=0.10)

    cov = int(cv2.countNonZero(pre))
    if cov < 50:
        open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        close_k = np.ones((3, 3), np.uint8)
    elif cov < 200:
        open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        close_k = np.ones((3, 3), np.uint8)
    else:
        open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        close_k = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(pre, cv2.MORPH_OPEN, open_k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=1)

    if cv2.countNonZero(mask) == 0 or cv2.countNonZero(mask) < int(0.5 * max(1, cov)):
        mask = cv2.dilate(pre, np.ones((3, 3), np.uint8), iterations=1)

    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + int(np.argmax(areas))
        lane = (labels == largest_idx).astype(np.uint8)

        cx, cy = cents[largest_idx]
        radius = max(4, int(0.12 * min(H, W)))
        for k in range(1, num):
            if k == largest_idx:
                continue
            area_k = stats[k, cv2.CC_STAT_AREA]
            if 0 < area_k <= 0.15 * areas.max():
                xk, yk = cents[k]
                if (xk - cx) ** 2 + (yk - cy) ** 2 <= radius ** 2:
                    lane |= (labels == k).astype(np.uint8)
        mask = lane

    mask = _orientation_filter(mask, max_deg=50.0)
    return mask.astype(np.uint8)


def _fallback_lane_mask(
    resized_bgr: np.ndarray,
    roi: np.ndarray,
    percentiles=(90, 86, 82),
    min_coverage_px: int = 40
) -> np.ndarray:
    """Original fallback lane mask function"""
    lab = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2LAB)
    L, _, B = cv2.split(lab)

    for p in percentiles:
        thr = np.percentile(B, p)
        m = ((B >= thr) & (L >= 50)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
        m = cv2.bitwise_and(m, roi)
        m = _apply_horizon_suppression(m, roi, extra=0.10)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if num > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = (labels == largest).astype(np.uint8)

        if cv2.countNonZero(m) >= min_coverage_px:
            return m

    return np.zeros_like(roi, dtype=np.uint8)


def generate_lane_mask(
    capture_dir: str,
    rgb_paths: List[str],
    step_idx: int,
    img_size: Tuple[int, int],
    debug_dir: Optional[str] = None,
    roi_top_y_frac: float = 0.45,
    min_coverage_px: int = 60,
    use_enhanced: bool = True,  # New parameter to toggle enhanced detection
) -> Tuple[np.ndarray, float]:
    """
    Generate lane mask with confidence score
    Returns: (mask, confidence)
    """
    if step_idx >= len(rgb_paths):
        return np.zeros(img_size[::-1], np.uint8), 0.0

    path = os.path.join(capture_dir, rgb_paths[step_idx])
    img = cv2.imread(path)
    if img is None:
        return np.zeros(img_size[::-1], np.uint8), 0.0

    h = img.shape[0]
    cropped = img[int(h * 0.25):]
    resized = cv2.resize(cropped, img_size)
    H, W = img_size[1], img_size[0]

    # Create ROI
    roi = np.zeros((H, W), np.uint8)
    top_y = int(max(0, min(H - 1, roi_top_y_frac * H)))
    pts = np.array([[int(0.10 * W), top_y], [int(0.90 * W), top_y],
                    [W - 1, H - 1], [0, H - 1]], np.int32)
    cv2.fillPoly(roi, [pts], 1)

    # Use enhanced or original detection
    if use_enhanced:
        mask, confidence = enhanced_lane_detection(resized, roi)
    else:
        mask = _primary_lane_mask(resized, roi)
        coverage = cv2.countNonZero(mask)
        confidence = min(1.0, coverage / (0.06 * H * W))

    # Apply horizon suppression
    mask = _apply_horizon_suppression(mask, roi, extra=0.10)

    # Fallback if mask is too small
    if cv2.countNonZero(mask) < min_coverage_px:
        fb = _fallback_lane_mask(resized, roi)
        if cv2.countNonZero(fb) > 0:
            mask = fb
            confidence = 0.3  # Lower confidence for fallback

    # Last resort - pure color detection
    if cv2.countNonZero(mask) == 0:
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        lower_y = np.array([10, 40, 60], np.uint8)
        upper_y = np.array([55, 255, 255], np.uint8)
        y_raw = (cv2.inRange(hsv, lower_y, upper_y) > 0).astype(np.uint8)
        y_raw = cv2.bitwise_and(y_raw, roi)
        y_raw = _apply_horizon_suppression(y_raw, roi, extra=0.10)
        y_raw = cv2.dilate(y_raw, np.ones((3, 3), np.uint8), iterations=2)
        mask = y_raw
        confidence = 0.1  # Very low confidence

    # Debug visualization
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"mask_step_{step_idx}.png"), mask * 255)

        # Enhanced debug with path planning
        overlay = resized.copy()
        overlay[mask.astype(bool)] = (0, 255, 0)

        # Draw Bezier path if available
        steering, path_points = plan_bezier_path(mask)
        for i in range(len(path_points) - 1):
            cv2.line(overlay, path_points[i], path_points[i+1], (255, 0, 0), 2)

        cv2.imwrite(os.path.join(debug_dir, f"overlay_step_{step_idx}.png"), overlay)

    return mask.astype(np.uint8), confidence


# ========================= Enhanced Environment ========================= #

class UnityCameraEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        capture_dir: str = "Assets/Captures",
        img_size: Tuple[int, int] = (84, 84),
        max_steps: Optional[int] = None,
        save_debug_masks: bool = False,
        mask_refresh_N: int = 5,
        deviation_refresh_threshold: float = 0.6,
        refresh_cooldown: int = 2,
        off_lane_window: int = 2,
        roi_top_y_frac: float = 0.45,
        min_episode_steps: int = 30,
        low_coverage_patience: int = 5,
        use_enhanced_detection: bool = True,  # Toggle enhanced features
        use_optical_flow: bool = False,        # Toggle optical flow
        use_path_planning: bool = False,       # Toggle path planning
        use_augmentation: bool = True,        # Toggle data augmentation
    ):
        super().__init__()

        # Enhanced feature flags
        self.use_enhanced_detection = use_enhanced_detection
        self.use_optical_flow = use_optical_flow
        self.use_path_planning = use_path_planning
        self.use_augmentation = use_augmentation

        # Initialize optical flow navigator if enabled
        if self.use_optical_flow:
            self.flow_navigator = OpticalFlowNavigator()

        # Clean up logs
        os.makedirs("Logs", exist_ok=True)
        for f in os.listdir("Logs"):
            if f.endswith((".csv", ".png")):
                os.remove(os.path.join("Logs", f))

        self.capture_dir = os.path.abspath(capture_dir)
        self.img_size = img_size
        self.save_debug_masks = save_debug_masks

        self.refresh_cooldown = refresh_cooldown
        self.last_refresh_step = -refresh_cooldown

        self.off_lane_window = off_lane_window
        self.roi_top_y_frac = roi_top_y_frac

        self.min_episode_steps = int(min_episode_steps)
        self.low_coverage_patience = int(low_coverage_patience)

        # Load camera intrinsics
        intr_path = os.path.join(self.capture_dir, "camera_intrinsics.yaml")
        if not os.path.exists(intr_path):
            raise FileNotFoundError("camera_intrinsics.yaml not found in capture folder")
        with open(intr_path, "r") as f:
            intr = yaml.safe_load(f)
        cam = intr["camera_matrix"]
        self.fx, self.fy, self.cx, self.cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]

        self.rgb_paths = self._load_latest_images()
        if not self.rgb_paths:
            raise RuntimeError("No RGB captures found in capture_dir")
        self.max_steps = min(max_steps or len(self.rgb_paths), len(self.rgb_paths))

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], np.float32),
            high=np.array([1.0, 1.0], np.float32),
        )
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, img_size[1], img_size[0]), dtype=np.uint8
        )

        # Setup logging
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = open(os.path.join("Logs", f"episode_log_{ts}.csv"), "w", newline="")
        self.logger = csv.writer(self.log_file)
        self.logger.writerow(
            ["step", "action", "reward", "dev", "coverage", "lane_r", "steer_cost",
             "decay", "cum_r", "off_lane", "confidence", "flow_steer", "path_steer"]
        )
        self.debug_dir = os.path.join("Logs", "DebugMasks")
        os.makedirs(self.debug_dir, exist_ok=True)

        # State variables
        self.lane_mask: Optional[np.ndarray] = None
        self.prev_mask: Optional[np.ndarray] = None
        self.prev_coverage: int = 0
        self.low_cov_streak: int = 0
        self.off_lane_streak: int = 0

        # Enhanced state variables
        self.lane_confidence: float = 0.0
        self.prev_steering: float = 0.0
        self.path_steering: float = 0.0
        self.flow_steering: float = 0.0
        self.current_frame: Optional[np.ndarray] = None

        self.step_idx = 0
        self.cumulative_reward = 0.0

        # Evaluation mode toggle
        self.eval_mode: bool = os.environ.get("UNITY_ENV_EVAL", "").lower() in ("1", "true", "yes")
        # Optional demo controller for evaluation only
        self.demo_controller: bool = os.environ.get("UNITY_DEMO_CONTROLLER", "").lower() in ("1", "true", "yes")

        self._last_coverage: int = 0

    # ================ Helper Methods ================ #

    def _load_latest_images(self) -> List[str]:
        """Load latest image paths from capture directory"""
        return list(
            natsorted(
                [f for f in os.listdir(self.capture_dir)
                 if f.lower().endswith(".jpg") and "CameraRGB" in f]
            )
        )

    def _generate_lane_mask(self) -> Tuple[np.ndarray, float]:
        """Generate lane mask with confidence score"""
        mask, confidence = generate_lane_mask(
            self.capture_dir,
            self.rgb_paths,
            self.step_idx,
            self.img_size,
            self.debug_dir if self.save_debug_masks else None,
            roi_top_y_frac=self.roi_top_y_frac,
            use_enhanced=self.use_enhanced_detection,
        )

        # Coverage surge smoothing
        cov = int(cv2.countNonZero((mask * 255).astype(np.uint8)))
        if self.prev_coverage and cov > max(50, self.prev_coverage * 2):
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
            cov = int(cv2.countNonZero((mask * 255).astype(np.uint8)))
            confidence *= 0.7  # Reduce confidence after erosion

        # Reuse previous mask if current is too small
        if cov < 12 and self.prev_mask is not None and cv2.countNonZero(self.prev_mask.astype(np.uint8)) > 0:
            mask = self.prev_mask.copy()
            cov = int(cv2.countNonZero((mask * 255).astype(np.uint8)))
            confidence *= 0.5  # Low confidence for reused mask

        self.prev_coverage = cov
        self.prev_mask = mask.copy()
        return mask, confidence

    def _get_obs(self) -> np.ndarray:
        """Get current observation with optional augmentation"""
        if self.step_idx >= len(self.rgb_paths):
            return np.zeros((3, self.img_size[1], self.img_size[0]), np.uint8)

        img = cv2.imread(os.path.join(self.capture_dir, self.rgb_paths[self.step_idx]))
        if img is None:
            return np.zeros((3, self.img_size[1], self.img_size[0]), np.uint8)

        # Store current frame for optical flow
        self.current_frame = img.copy()

        # Crop and resize
        img = cv2.resize(img[int(img.shape[0] * CROP_TOP_FRAC):], self.img_size)

        # Apply LAB enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge((l, a, b))
        sharp = cv2.filter2D(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), -1,
                           np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

        obs = np.transpose(sharp, (2, 0, 1)).astype(np.uint8)

        # Apply augmentation during training
        if self.use_augmentation and not self.eval_mode:
            obs = augment_observation(obs, training=True, intensity=0.3)

        return obs

    # ================ Enhanced Reward System ================ #

    def _calculate_multi_component_reward(
        self,
        deviation: float,
        coverage: int,
        steering: float,
        throttle: float,
        confidence: float,
        off_lane: bool
    ) -> Dict[str, float]:
        """
        Calculate multi-component reward with detailed breakdown
        """
        rewards = {}

        # 1. Lane following reward (confidence-weighted)
        coverage_quality = self._coverage_quality(coverage)
        centering_quality = max(0.0, 1.0 - deviation)
        rewards['lane_following'] = coverage_quality * centering_quality * confidence * 0.4

        # 2. Forward progress reward
        rewards['progress'] = throttle * (1.0 - min(1.0, abs(steering) * 0.5)) * 0.3

        # 3. Smooth driving bonus
        steering_change = abs(steering - self.prev_steering)
        rewards['smoothness'] = -steering_change * 0.1

        # 4. Path planning bonus (if enabled)
        if self.use_path_planning and abs(self.path_steering) > 0.01:
            path_agreement = 1.0 - min(1.0, abs(steering - self.path_steering))
            rewards['path_following'] = path_agreement * confidence * 0.1
        else:
            rewards['path_following'] = 0.0

        # 5. Exploration bonus when confidence is low
        if confidence < 0.3:
            rewards['exploration'] = 0.05
        else:
            rewards['exploration'] = 0.0

        # 6. Penalties
        rewards['steering_cost'] = -abs(steering) * 0.02

        if off_lane:
            rewards['off_lane_penalty'] = -0.3
        else:
            rewards['off_lane_penalty'] = 0.0

        # 7. Time decay
        halfway = self.max_steps // 2
        if self.step_idx <= halfway:
            decay = 1.0
        else:
            decay = 1.0 - (self.step_idx - halfway) / max(1, halfway)

        # Apply decay to all rewards
        for key in rewards:
            rewards[key] *= decay

        rewards['total'] = sum(rewards.values())
        rewards['decay'] = decay

        return rewards

    def _coverage_quality(self, coverage: int) -> float:
        """Calculate coverage quality score"""
        target = int(0.06 * self.img_size[0] * self.img_size[1])
        return float(np.clip(coverage / max(1, target), 0.0, 1.0))

    def _demo_action(self) -> Tuple[float, float]:
        """Enhanced demo controller using multiple strategies"""
        # Try path planning first
        if self.use_path_planning and self.lane_mask is not None:
            path_steer, _ = plan_bezier_path(self.lane_mask)
            if abs(path_steer) > 0.01:
                throttle = float(np.clip(0.4 * (1.0 - abs(path_steer)), 0.15, 0.5))
                return float(path_steer), throttle

        # Fallback to centroid-based steering
        if self.lane_mask is None or cv2.countNonZero(self.lane_mask.astype(np.uint8)) == 0:
            return 0.0, 0.15

        m8 = (self.lane_mask * 255).astype(np.uint8)
        M = cv2.moments(m8)
        if M["m00"] == 0:
            return 0.0, 0.15

        cx = int(M["m10"] / M["m00"])
        center = self.img_size[0] // 2
        err = (cx - center) / max(1, center)
        steer = float(np.clip(-0.8 * err, -0.6, 0.6))
        throttle = float(np.clip(0.35 * (1.0 - abs(err)), 0.1, 0.5))
        return steer, throttle

    # ================ Main Step Function ================ #

    def step(self, action):
        """Enhanced step function with multiple navigation strategies"""

        # Generate or refresh lane mask
        if self.lane_mask is None or (self.step_idx - self.last_refresh_step) >= self.refresh_cooldown:
            self.lane_mask, self.lane_confidence = self._generate_lane_mask()
            self.last_refresh_step = self.step_idx

        # Demo controller override (eval only)
        if (not STRICT_PASSIVE) and self.eval_mode and self.demo_controller:
            action = self._demo_action()

        # Get observation
        obs = self._get_obs()

        # Compute navigation signals
        deviation, coverage = self._estimate_lane_deviation()
        self._last_coverage = int(coverage)

        # Optical flow guidance
        if self.use_optical_flow and self.current_frame is not None:
            resized_frame = cv2.resize(self.current_frame[int(self.current_frame.shape[0] * CROP_TOP_FRAC):],
                                      self.img_size)
            self.flow_steering, flow_confidence = self.flow_navigator.compute_flow_guidance(resized_frame)
        else:
            self.flow_steering, flow_confidence = 0.0, 0.0

        # Path planning guidance
        if self.use_path_planning and self.lane_mask is not None:
            self.path_steering, _ = plan_bezier_path(self.lane_mask)
        else:
            self.path_steering = 0.0

        # Blend action with guidance signals based on confidence
        steering = float(np.clip(action[0], -1, 1))
        throttle = float(np.clip(action[1], 0, 1))

        # Apply guidance blending when confidence is low
        if (not STRICT_PASSIVE) and (self.lane_confidence < 0.5):
            # Blend with optical flow
            if abs(self.flow_steering) > 0.01:
                steering = 0.7 * steering + 0.3 * self.flow_steering

            # Blend with path planning
            if abs(self.path_steering) > 0.01:
                steering = 0.6 * steering + 0.4 * self.path_steering

        # Calculate reward
        off_lane = self._check_off_lane()
        reward_components = self._calculate_multi_component_reward(
            deviation, coverage, steering, throttle, self.lane_confidence, off_lane
        )
        reward = reward_components['total']

        # Update off-lane streak
        if off_lane:
            self.off_lane_streak += 1
        else:
            self.off_lane_streak = 0

        # Update low coverage streak
        if coverage < 80 and self.step_idx >= self.min_episode_steps:
            self.low_cov_streak += 1
        else:
            self.low_cov_streak = 0

        # Update cumulative reward and previous steering
        self.cumulative_reward += reward
        self.prev_steering = steering

        # Log detailed metrics
        self.logger.writerow([
            self.step_idx,
            [float(steering), float(throttle)],
            reward,
            deviation,
            self._last_coverage,
            reward_components['lane_following'],
            reward_components['steering_cost'],
            reward_components['decay'],
            self.cumulative_reward,
            int(off_lane),
            self.lane_confidence,
            self.flow_steering,
            self.path_steering
        ])
        self.log_file.flush()

        self.step_idx += 1

        # Termination conditions
        early_stop = (self.off_lane_streak >= 10) or (
            self.step_idx > self.min_episode_steps and
            deviation > 0.95 and
            self.low_cov_streak >= self.low_coverage_patience
        )

        if self.eval_mode:
            terminated = (self.step_idx >= self.max_steps)
        else:
            terminated = (self.step_idx >= self.max_steps) or early_stop

        return obs, reward, terminated, False, self._get_info()

    # ================ Lane Analysis Methods ================ #

    def _estimate_lane_deviation(self) -> Tuple[float, int]:
        """Estimate deviation from lane center"""
        if self.lane_mask is None:
            return 1.0, 0
        m8 = (self.lane_mask * 255).astype(np.uint8)
        coverage = int(cv2.countNonZero(m8))
        M = cv2.moments(m8)
        if M["m00"] == 0 or coverage < 20:
            return 1.0, coverage
        cx = int(M["m10"] / M["m00"])
        dev = abs((cx - self.img_size[0] // 2) / (self.img_size[0] // 2))
        return dev, coverage

    def _check_off_lane(self) -> bool:
        """Check if vehicle is off the lane"""
        if self.lane_mask is None:
            return True
        h, w = self.lane_mask.shape
        cx = w // 2
        band_top = int(0.55 * h)
        band_bot = int(0.90 * h)
        band = self.lane_mask[band_top:band_bot, :]
        half = max(2, self.off_lane_window)
        x0, x1 = max(0, cx - 3*half), min(w, cx + 3*half + 1)
        strip = band[:, x0:x1]
        if np.count_nonzero(strip) > 0:
            return False
        cols = np.where(band.sum(axis=0) > 0)[0]
        if cols.size == 0:
            return True
        min_dx = int(np.min(np.abs(cols - cx)))
        return min_dx > 6

    # ================ Gym API Methods ================ #

    def _get_info(self) -> Dict:
        """Get environment info"""
        return {
            "step": self.step_idx,
            "coverage": int(self._last_coverage),
            "confidence": float(self.lane_confidence),
            "eval_mode": bool(self.eval_mode),
            "demo_controller": bool(self.demo_controller),
            "flow_steering": float(self.flow_steering),
            "path_steering": float(self.path_steering),
            "enhanced_features": {
                "detection": self.use_enhanced_detection,
                "optical_flow": self.use_optical_flow,
                "path_planning": self.use_path_planning,
                "augmentation": self.use_augmentation
            }
        }

    def reset(self, *, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)

        # Handle evaluation mode
        if options and "eval_mode" in options:
            self.eval_mode = bool(options["eval_mode"])
        elif os.environ.get("UNITY_ENV_EVAL", "").lower() in ("1", "true", "yes"):
            self.eval_mode = True

        # Demo controller only in eval mode
        self.demo_controller = self.eval_mode and (
            os.environ.get("UNITY_DEMO_CONTROLLER", "").lower() in ("1", "true", "yes")
        )

        # Reset state variables
        self.step_idx = 0
        self.cumulative_reward = 0.0
        self.off_lane_streak = 0
        self.low_cov_streak = 0
        self.prev_coverage = 0
        self.prev_mask = None
        self.lane_mask = None
        self._last_coverage = 0
        self.last_refresh_step = -self.refresh_cooldown

        # Reset enhanced state
        self.lane_confidence = 0.0
        self.prev_steering = 0.0
        self.path_steering = 0.0
        self.flow_steering = 0.0
        self.current_frame = None

        # Reset optical flow navigator
        if self.use_optical_flow:
            self.flow_navigator.reset()

        # Reload images
        self.rgb_paths = self._load_latest_images()
        if not self.rgb_paths:
            obs = np.zeros((3, self.img_size[1], self.img_size[0]), np.uint8)
            return obs, self._get_info()

        # Generate initial lane mask
        self.lane_mask, self.lane_confidence = self._generate_lane_mask()
        return self._get_obs(), self._get_info()

    def close(self):
        """Clean up resources"""
        if hasattr(self, "log_file") and not self.log_file.closed:
            self.log_file.close()

    def __del__(self):
        """Destructor"""
        self.close()
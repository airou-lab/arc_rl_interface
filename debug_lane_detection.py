#!/usr/bin/env python3
"""
debug_lane_detection.py - Visualize and tune lane detection
=============================================================
Connects to Unity and shows lane detection overlays in real-time.
Use this to tune HSV parameters for your specific road textures.
"""
import argparse
import numpy as np
import cv2
import time
from live_unity_env import LiveUnityEnv


class LaneDetectorDebug:
    """Lane detector with visualization."""

    def __init__(self):
        # Yellow lanes
        self.yellow_lower = np.array([15, 50, 50])
        self.yellow_upper = np.array([35, 255, 255])
        # White lanes
        self.white_lower = np.array([0, 0, 160])
        self.white_upper = np.array([180, 40, 255])

    def detect_and_visualize(self, image: np.ndarray) -> tuple:
        """
        Detect lanes and return visualization.

        Returns:
            (lane_info dict, visualization image)
        """
        h, w = image.shape[:2]
        vis = image.copy()

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Detect lane colors
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        lane_mask = cv2.bitwise_or(yellow_mask, white_mask)

        # ROI: bottom half
        roi_top = h // 2

        # Find lane points
        mid_x = w // 2
        left_points = []
        right_points = []

        for row in range(roi_top, h, 4):
            # Left lane
            left_cols = np.where(lane_mask[row, :mid_x] > 0)[0]
            if len(left_cols) > 0:
                left_x = left_cols[-1]  # Rightmost in left region
                left_points.append((left_x, row))
                cv2.circle(vis, (int(left_x), row), 2, (255, 0, 0), -1)  # Blue

            # Right lane
            right_cols = np.where(lane_mask[row, mid_x:] > 0)[0]
            if len(right_cols) > 0:
                right_x = right_cols[0] + mid_x  # Leftmost in right region
                right_points.append((right_x, row))
                cv2.circle(vis, (int(right_x), row), 2, (0, 0, 255), -1)  # Red

        # Compute metrics
        if len(left_points) >= 2 and len(right_points) >= 2:
            left_x_bottom = np.mean([p[0] for p in left_points[-3:]])
            right_x_bottom = np.mean([p[0] for p in right_points[-3:]])

            lane_center = (left_x_bottom + right_x_bottom) / 2
            image_center = w / 2

            lane_width = right_x_bottom - left_x_bottom
            lane_offset = (image_center - lane_center) / (lane_width / 2) if lane_width > 0 else 0

            # Draw lane center line
            cv2.line(vis, (int(lane_center), h), (int(lane_center), roi_top), (0, 255, 0), 2)

            # Draw image center line (where car should be)
            cv2.line(vis, (int(image_center), h), (int(image_center), roi_top), (255, 255, 0), 1)

            confidence = min(len(left_points), len(right_points)) / 8.0

            lane_info = {
                'lane_offset': float(lane_offset),
                'confidence': min(confidence, 1.0),
                'left_x': float(left_x_bottom),
                'right_x': float(right_x_bottom),
                'lane_detected': True,
            }
        else:
            lane_info = {
                'lane_offset': 0.0,
                'confidence': 0.0,
                'lane_detected': False,
            }

        # Add text overlay
        text_color = (255, 255, 255)
        if lane_info['lane_detected']:
            cv2.putText(vis, f"Offset: {lane_info['lane_offset']:+.2f}", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
            cv2.putText(vis, f"Conf: {lane_info['confidence']:.2f}", (5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        else:
            cv2.putText(vis, "NO LANES", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return lane_info, vis, lane_mask


def main(args):
    print("\n" + "=" * 60)
    print("  LANE DETECTION DEBUG")
    print("=" * 60)
    print("\nConnecting to Unity...")

    env = LiveUnityEnv(
        host=args.host,
        port=args.port,
        img_width=128,
        img_height=128,
        max_steps=1000,
        verbose=True,
    )

    detector = LaneDetectorDebug()

    print("\nRunning lane detection visualization...")
    print("Watch the window! Press Ctrl+C to stop.\n")

    # Create window
    cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lane Detection", 512, 256)

    obs, info = env.reset()
    step = 0

    try:
        while True:
            # Simple driving action (slight steering based on offset)
            if 'lane_info' in info and info['lane_info'].get('lane_detected'):
                offset = info['lane_info']['lane_offset']
                steer = 0.3 * offset
            else:
                steer = 0.0

            action = np.array([steer, 0.4, 0.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            # Detect and visualize
            image = obs['image']
            lane_info, vis, mask = detector.detect_and_visualize(image)

            # Show mask alongside visualization
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            combined = np.hstack([vis, mask_rgb])

            # Convert RGB to BGR for OpenCV display
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            cv2.imshow("Lane Detection", combined_bgr)

            # Print stats periodically
            if step % 50 == 0:
                print(f"Step {step}: offset={lane_info['lane_offset']:+.2f}, "
                      f"conf={lane_info['confidence']:.2f}, "
                      f"detected={lane_info['lane_detected']}")

            if terminated or truncated:
                print(f"Episode ended at step {step}")
                obs, info = env.reset()
                step = 0

            step += 1

            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        env.close()
        cv2.destroyAllWindows()

    print("\nDone!")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=5556)
    main(p.parse_args())
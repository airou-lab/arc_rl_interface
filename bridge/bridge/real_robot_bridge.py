import socket
import struct
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading

# --- CONFIGURATION ---
HOST = '0.0.0.0' # Listen on all interfaces
PORT = 5556
IMG_SIZE = (128, 128)  # Must match training
MAX_SPEED = 0.5 # We can change this later if it becomes weird
MAX_STEER = 0.4 # Radians (at steer=1.0) ~23 degrees

CAMERA_TOPIC = '/camera/camera/color/image_raw' # Confirm this later.. (realsense topic)
DRIVE_TOPIC = '/drive'


class RealRobotBridge(Node):
    def __init__(self):
        super().__init__('unity_impersonator_bridge')
        # ROS setup
        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # Telemetry (we'll send zeros, real values where possible)
        self.current_speed = 0.0
        self.current_steer = 0.0

        # Sub
        self.create_subscription(Image, CAMERA_TOPIC, self.image_callback, 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 1)

        self.get_logger().info(f"Bridge listening on port {PORT}")
        self.get_logger().info(f"Camera: {CAMERA_TOPIC}, Drive: {DRIVE_TOPIC}")

    def image_callback(self, msg):
        try:
            # Convert ROS image -> OpenCV Image
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Resize to agent expected size
            resized = cv2.resize(cv_img, IMG_SIZE)
            # Port to  JPEG
            _, jpeg_data = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

            with self.frame_lock:
                self.latest_frame = jpeg_data.tobytes()
        except Exception as e:
            self.get_logger().error(f"Image error: {e}")

    def publish_command(self, steer_norm, throttle_norm):
        """Translates normalized Agent actions to Real World units.
        This is where our actual drive commands happen!"""
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        # Map [-1, 1] -> [-MAX_STEER, +MAX_STEER]
        msg.drive.steering_angle = steer_norm * MAX_STEER
        # Map [0, 1] -> [0, MAX_SPEED]
        msg.drive.speed = throttle_norm * MAX_SPEED

        self.current_steer = steer_norm
        self.current_speed = throttle_norm * MAX_SPEED

        self.drive_pub.publish(msg)

# --- TCP SERVER LOGIC ---
def run_tcp_server(ros_node):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    print(f"TCP Server listening on port {PORT}...")

    while True:
        print("Waiting for agent to connect...")
        conn, addr = server.accept()
        print(f"Agent connected from: {addr}")

        try:
            while True:
                # Read command byte
                cmd = conn.recv(1)
                if not cmd:
                    break

                if cmd == b'R':
                    # Reset command
                    print("Reset command received")
                    # Send observation after reset
                    send_observation(conn, ros_node)

                elif cmd == b'A':
                    # Action command: 3 floats (steer, throttle, brake) little-endian
                    action_data = recv_exactly(conn, 12)  # 3 * 4 bytes
                    steer, throttle, brake = struct.unpack('<fff', action_data)

                    # Apply action (ignore brake for now, use throttle)
                    ros_node.publish_command(steer, throttle)

                    # Send observation
                    send_observation(conn, ros_node)

                elif cmd == b'W':
                    # Waypoint command - just consume and ignore
                    num_waypoints = struct.unpack('<B', conn.recv(1))[0]
                    waypoint_data = recv_exactly(conn, num_waypoints * 8)  # 2 floats per waypoint
                    # Ignore waypoints on real robot

                elif cmd == b'Q':
                    # Quit command
                    print("Quit command received")
                    break

                else:
                    print(f"Unknown command: {cmd}")

        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            conn.close()
            print("Agent disconnected")


def recv_exactly(conn, num_bytes):
    """Receive exact number of bytes."""
    data = b''
    while len(data) < num_bytes:
        chunk = conn.recv(num_bytes - len(data))
        if not chunk:
            raise ConnectionResetError("Connection closed")
        data += chunk
    return data


def send_observation(conn, ros_node):
    """Send observation in Unity protocol format."""
    # Get latest frame
    with ros_node.frame_lock:
        jpeg_bytes = ros_node.latest_frame

    if jpeg_bytes is None:
        # Black image if no camera data
        blank = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
        _, encoded = cv2.imencode('.jpg', blank)
        jpeg_bytes = encoded.tobytes()

    # Build telemetry (12 floats) - mostly zeros for real robot
    # [goal_cos, goal_sin, goal_dist, speed, yaw_rate,
    #  last_steer, last_thr, last_brk, lat_err, hdg_err, kappa, ds]
    telemetry = [
        1.0,  # goal_cos (straight ahead)
        0.0,  # goal_sin
        10.0,  # goal_dist (dummy)
        ros_node.current_speed,  # speed
        0.0,  # yaw_rate
        ros_node.current_steer,  # last_steer
        0.0,  # last_thr
        0.0,  # last_brk
        0.0,  # lat_err
        0.0,  # hdg_err
        0.0,  # kappa
        0.0,  # ds
    ]

    reward = 0.0
    done = 0
    truncated = 0

    # Pack and send (big-endian to match LiveUnityEnv receive)
    header = struct.pack('>I', len(jpeg_bytes))
    telemetry_bytes = struct.pack('>12f', *telemetry)
    metadata = struct.pack('>fBB', reward, done, truncated)

    conn.sendall(header + jpeg_bytes + telemetry_bytes + metadata)


def main(args=None):
    rclpy.init(args=args)
    node = RealRobotBridge()

    tcp_thread = threading.Thread(target=run_tcp_server, args=(node,), daemon=True)
    tcp_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
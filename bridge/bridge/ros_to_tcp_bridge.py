# Warning! Autogen code!
# This servers as the bridge between the Reinforcement Learning Model and its commands to the real robot
# For full documentation please see our docs https://airou-lab.github.io/arcpro_ros2_website/
#

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
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 5556
img_size = (84, 84) # double check again..
MAX_SPEED = 0.5  # We can change this later if it becomes weird
MAX_STEER = 0.4  # Radians (at steer=1.0) ~23 degrees

CAMERA_TOPIC = '/camera/camera/color/image_raw'  #confirm this later.. (realsense toipic
DRIVE_TOPIC = '/drive'

class RealRobotBridge(Node):
    def __init__(self):
        super().__init__('unity_impersonator_bridge')

        # ROS Setup
        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # Sub
        self.create_subscription(Image, CAMERA_TOPIC, self.image_callback, 1)

        # Pub
        self.drive_pub = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 1)

        self.get_logger().info(f"Bridge Node Started. Listening for Camera on: {CAMERA_TOPIC}")

    def image_callback(self, msg):
        """update frame to socket"""
        try:
            # Convert ROS Image -> OpenCV Image
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Resize to agent expected size
            resized = cv2.resize(cv_img, img_size)
            # port to  JPEG
            _, jpeg_data = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

            with self.frame_lock:
                self.latest_frame = jpeg_data.tobytes()

        except Exception as e:
            self.get_logger().error(f"CV Conversion Error: {e}")

    def publish_command(self, steer_norm, throttle_norm):
        """Translates normalized Agent actions to Real World units.
        This is where our actual drive commands happen!
        """
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        # Map [-1, 1] -> [-MAX_STEER, +MAX_STEER]
        msg.drive.steering_angle = steer_norm * MAX_STEER

        # Map [0, 1] -> [0, MAX_SPEED]
        msg.drive.speed = throttle_norm * MAX_SPEED

        self.drive_pub.publish(msg)


# --- TCP SERVER LOGIC ---

def run_tcp_server(ros_node):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    print(f"TCP Server listening on port {PORT}...")
    print("Waiting for Agent to connect...")

    conn, addr = server.accept()
    print(f"Agent connected from: {addr}")

    try:
        while True:
            # 1. Wait for command from Agent
            # Protocol: reset (1 byte) OR action (8 bytes)
            # We peek to see what's coming, but usually, we just read.
            # However, the agent loop is usually: Receive Frame -> Send Action.
            # BUT the very first thing is usually a Reset command.

            # Simplified logic based on your README flow:
            # The Loop is: Agent Sends Action -> Bridge Sends Obs

            # First, receive 8 bytes (2 floats)
            # If it's the start, the agent might send 'R'.
            # Let's try to read 1 byte first to check for 'R'

            data = conn.recv(8)
            if not data: break

            if len(data) == 1 and data == b'R':# Length 1 means RST/end episopde
                print("Received RST cmd (1 byte). Starting episode.")
                # Agent wants a frame immediately after reset
                pass
            elif len(data) == 8: #All actions are 8 long
                steer, throttle = struct.unpack('>ff', data)#first 4 bytes are the steer then second 4 are throttle
                #stored into above, then publsihed to commander
                ros_node.publish_command(steer, throttle)
            else:
                # If we got partial data (rare in local loop but possible), ignore or buffer
                pass

            # 2. Send Observation back to Agent
            # Protocol: len (u32) -> jpeg -> reward (f32) -> done (u8) -> truncated (u8)

            # Get latest frame (or blank if camera isn't ready yet)
            with ros_node.frame_lock:
                jpeg_bytes = ros_node.latest_frame

            if jpeg_bytes is None:
                # Create a black image if no camera data yet
                blank = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
                _, encoded = cv2.imencode('.jpg', blank)
                jpeg_bytes = encoded.tobytes()

            # Construct packet
            length = len(jpeg_bytes)
            reward = 0.0  # Real life gives no rewards
            done = 0  # We never finish unless you press Ctrl+C
            truncated = 0

            # Struct format: I (uint32), f (float), B (uchar), B (uchar) -> Big Endian (>)
            header = struct.pack('>I', length)
            metadata = struct.pack('>fBB', reward, done, truncated)

            # Send: Header + JPEG + Metadata
            conn.sendall(header + jpeg_bytes + metadata)

    except Exception as e:
        print(f"Connection Error: {e}")
    finally:
        conn.close()
        server.close()


def main(args=None):
    rclpy.init(args=args)
    node = RealRobotBridge()

    # Run TCP server in a separate thread so ROS callbacks keep firing
    tcp_thread = threading.Thread(target=run_tcp_server, args=(node,))
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
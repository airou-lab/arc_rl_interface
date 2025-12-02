import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import threading
import time


class Ros2UnityEnv(gym.Env, Node):
    """
    DDS-based Environment.
    Replaces TCP socket with high-throughput ROS 2 messaging.
    """

    def __init__(self, img_width=84, img_height=84):
        gym.Env.__init__(self)
        # Initialize ROS 2 context if not already active
        if not rclpy.ok():
            rclpy.init()
        Node.__init__(self, 'rl_agent_node')

        # QoS: Best Effort for images (drop old frames, prioritize new ones)
        # This reduces latency significantly compared to reliable TCP
        qos_vision = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.sub_image = self.create_subscription(
            Image, '/camera/image_raw', self._img_callback, qos_vision)

        self.sub_telemetry = self.create_subscription(
            Float32MultiArray, '/vehicle/telemetry', self._tel_callback, 10)

        # Publisher
        self.pub_control = self.create_publisher(AckermannDrive, '/cmd_vel', 10)

        # State buffers (Thread-safe)
        self.bridge = CvBridge()
        self.latest_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        self.latest_vec = np.zeros(12, dtype=np.float32)
        self.lock = threading.Lock()

        # Spin ROS in background thread
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self)
        self._thread = threading.Thread(target=self._executor.spin)
        self._thread.start()

    def _img_callback(self, msg):
        with self.lock:
            # Zero-copy conversion where possible
            try:
                self.latest_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            except Exception as e:
                self.get_logger().error(f"CV Bridge error: {e}")

    def _tel_callback(self, msg):
        with self.lock:
            self.latest_vec = np.array(msg.data, dtype=np.float32)

    def step(self, action):
        # 1. Publish Action
        msg = AckermannDrive()
        msg.steering_angle = float(action[0])
        msg.speed = float(action[1])
        msg.jerk = float(action[2])  # Mapping brake to jerk field
        self.pub_control.publish(msg)

        # 2. Wait for physics step (Sync with Unity)
        time.sleep(0.02)

        # 3. Get Observation
        with self.lock:
            obs = {
                "image": self.latest_img.copy(),
                "vec": self.latest_vec.copy()
            }

        # 4. Reward & Done (Placeholder - usually calculated here or in wrapper)
        reward = 0.0
        done = False

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        with self.lock:
            obs = {
                "image": self.latest_img.copy(),
                "vec": self.latest_vec.copy()
            }
        return obs, {}

    def close(self):
        self.destroy_node()
        rclpy.shutdown()
        self._thread.join()
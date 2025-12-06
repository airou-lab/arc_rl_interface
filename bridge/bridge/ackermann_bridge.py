# Warning! Autogen code!

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from builtin_interfaces.msg import Time

class AckermannBridge(Node):
    def __init__(self):
        super().__init__('ackermann_bridge')

        # Subscribe to old AckermannDrive topic
        self.sub = self.create_subscription(
            AckermannDrive,
            '/drive',  # topic currently publishing AckermannDrive
            self.callback,
            10
        )

        # Publish as AckermannDriveStamped
        self.pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive_stamped',  # new topic for mux
            10
        )

    def callback(self, msg: AckermannDrive):
        stamped_msg = AckermannDriveStamped()
        stamped_msg.header.stamp = self.get_clock().now().to_msg()
        stamped_msg.header.frame_id = 'base_link'
        stamped_msg.drive = msg
        self.pub.publish(stamped_msg)

def main(args=None):
    rclpy.init(args=args)
    node = AckermannBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

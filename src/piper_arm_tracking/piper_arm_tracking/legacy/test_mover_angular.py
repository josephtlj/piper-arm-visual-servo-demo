#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped

class TestMoverAngular(Node):
    def __init__(self):
        super().__init__("test_mover_angular")
        self.pub = self.create_publisher(TwistStamped, "/servo_server/delta_twist_cmds", 10)
        self.timer = self.create_timer(0.05, self.timer_cb)  # 20 Hz
        self.get_logger().info("TestMoverAngular: sending pure yaw commands...")

    def timer_cb(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        # No linear motion
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0

        # Slow yaw about Z (e.g. 0.15 rad/s)
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.15

        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TestMoverAngular()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()



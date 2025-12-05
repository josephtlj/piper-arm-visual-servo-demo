import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped

class TestMover(Node):
    def __init__(self):
        super().__init__('test_mover')
        self.publisher_ = self.create_publisher(
            TwistStamped,
            '/servo_server/delta_twist_cmds',
            10
        )
        self.timer = self.create_timer(0.033, self.timer_callback)
        self.get_logger().info('Test Mover Started: Sending UP command...')

    def timer_callback(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.05  # 5 cm/s up

        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0

        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TestMover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_msg = TwistStamped()
        stop_msg.header.stamp = node.get_clock().now().to_msg()
        stop_msg.header.frame_id = 'base_link'
        node.publisher_.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()



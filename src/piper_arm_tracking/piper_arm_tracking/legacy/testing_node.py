import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TestingNode(Node):
    def __init__(self):
        super().__init__('testing_node')
        self.publisher_ = self.create_publisher(String, 'test_topic', 10)
        self.subscription = self.create_subscription(
            String,
            'test_topic',
            self.listener_callback,
            10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.count = 0

    def timer_callback(self):
        msg = String()
        msg.data = f"Hello ROS2 World! Count: {self.count}"
        self.publisher_.publish(msg)
        self.get_logger().info(f"Publishing: '{msg.data}'")
        self.count += 1

    def listener_callback(self, msg):
        self.get_logger().info(f"Received: '{msg.data}'")

def main(args=None):
    rclpy.init(args=args)
    node = TestingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




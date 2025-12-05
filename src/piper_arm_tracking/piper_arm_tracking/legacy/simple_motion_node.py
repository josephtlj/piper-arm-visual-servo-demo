import rclpy
from rclpy.node import Node

from piper_msgs.srv import Enable          # service to enable/disable arm
from sensor_msgs.msg import JointState     # command message for /joint_ctrl


class SimpleMotionNode(Node):
    def __init__(self):
        super().__init__('simple_motion_node')

        # 1) Client for /enable_srv
        self.enable_client = self.create_client(Enable, '/enable_srv')
        self.get_logger().info("Waiting for /enable_srv...")
        self.enable_client.wait_for_service()
        self.get_logger().info("/enable_srv is available")

        # 2) Publisher for joint commands to /joint_ctrl
        self.cmd_pub = self.create_publisher(JointState, '/joint_states', 10)

        # 3) Enable the arm once at startup
        self.enable_arm(True)

        # 4) Timer to send commands periodically (every 3 seconds)
        self.state = 0
        self.timer = self.create_timer(3.0, self.timer_callback)
        self.joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'joint7', 'joint8',
        ]

    def enable_arm(self, enable: bool):
        req = Enable.Request()
        req.enable_request = enable

        self.get_logger().info(f"Calling /enable_srv with enable_request={enable}")
        future = self.enable_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info(
                f"/enable_srv response: enable_response={future.result().enable_response}"
            )
        else:
            self.get_logger().error("Failed to call /enable_srv")

    def timer_callback(self):
        msg = JointState()
        msg.name = self.joint_names

        if self.state == 0:
            self.get_logger().info("Moving to Pose A")
            # SAFE TEST POSE: small, near-current angles
            msg.position = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            self.get_logger().info("Moving to Pose B")
            msg.position = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # velocity and effort empty (driver will ignore them)
        msg.velocity = []
        msg.effort = []

        self.cmd_pub.publish(msg)
        self.state = 1 - self.state


def main(args=None):
    rclpy.init(args=args)
    node = SimpleMotionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()




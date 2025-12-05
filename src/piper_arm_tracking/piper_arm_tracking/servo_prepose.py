#!/usr/bin/env python3
# Sends a short burst of JointState commands to /piper_cmd to move the Piper arm
# into a non-zero "servo-ready" pose, then exits.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class ServoPrepose(Node):
    def __init__(self):
        super().__init__("servo_prepose")

        self.pub = self.create_publisher(JointState, "/piper_cmd", 10)
        self.tick = 0
        self.done = False

        # Timer at 50 Hz
        self.timer = self.create_timer(0.02, self.timer_cb)
        self.get_logger().info(
            "ServoPrepose: sending servo-ready pose for a short duration..."
        )

        # Fixed non-zero "servo home" joints (radians)
        self.target_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "gripper",
        ]
        self.target_positions = [
            -0.1,   # joint1
            0.8,   # joint2
            -1.0,  # joint3
            0.3,   # joint4
            0.4,   # joint5
            -0.3,   # joint6
            0.0,   # gripper
        ]

    def timer_cb(self):
        if self.done:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.target_names)
        msg.position = list(self.target_positions)
        self.pub.publish(msg)

        self.tick += 1

        # Publish for ~3 seconds (150 * 0.02s) then stop
        if self.tick > 150:
            self.get_logger().info("ServoPrepose: done, shutting down.")
            self.done = True


def main(args=None):
    rclpy.init(args=args)
    node = ServoPrepose()
    try:
        # Spin until done flag is set
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()



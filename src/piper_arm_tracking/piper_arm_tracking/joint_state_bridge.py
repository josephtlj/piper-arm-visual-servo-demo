import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStateBridge(Node):
    def __init__(self):
        super().__init__("joint_state_bridge")

        self.feedback_sub = self.create_subscription(
            JointState,
            "/joint_states_feedback",
            self.feedback_callback,
            10,
        )

        self.joint_states_pub = self.create_publisher(
            JointState,
            "/joint_states",
            10,
        )

        self.get_logger().info(
            "JointStateBridge: /joint_states_feedback → /joint_states "
            "(gripper → joint7, joint8)"
        )

    def feedback_callback(self, msg: JointState):
        # building a new JointState that:
        #  - copies joint1..6 as-is
        #  - converts 'gripper' into joint7 & joint8
        new_msg = JointState()
        new_msg.header.stamp = self.get_clock().now().to_msg()
        new_msg.header.frame_id = msg.header.frame_id

        names = []
        positions = []
        velocities = []
        efforts = []

        gripper_pos = 0.0
        gripper_vel = 0.0
        gripper_eff = 0.0
        has_gripper = False

        for i, name in enumerate(msg.name):
            if name == "gripper":
                has_gripper = True
                if i < len(msg.position):
                    gripper_pos = msg.position[i]
                if i < len(msg.velocity):
                    gripper_vel = msg.velocity[i]
                if i < len(msg.effort):
                    gripper_eff = msg.effort[i]
            else:
                # copy regular joints (joint1..6)
                names.append(name)
                if i < len(msg.position):
                    positions.append(msg.position[i])
                if i < len(msg.velocity):
                    velocities.append(msg.velocity[i])
                if i < len(msg.effort):
                    efforts.append(msg.effort[i])

        # get joint7 & joint8 from 'gripper'
        if has_gripper:
            names.extend(["joint7", "joint8"])
            positions.extend([gripper_pos, -gripper_pos])
            velocities.extend([gripper_vel, -gripper_vel])
            efforts.extend([gripper_eff, gripper_eff])

        new_msg.name = names
        new_msg.position = positions
        new_msg.velocity = velocities
        new_msg.effort = efforts

        self.joint_states_pub.publish(new_msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointStateBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()




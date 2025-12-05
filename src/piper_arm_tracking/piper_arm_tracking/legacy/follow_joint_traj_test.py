# Simple FollowJointTrajectory test client for Piper arm through the bridge

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class SimpleTrajClient(Node):
    def __init__(self):
        super().__init__('simple_traj_client')
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.last_joint_state = None
        self.action_client = ActionClient(
            self, FollowJointTrajectory, 'arm_controller/follow_joint_trajectory'
        )

    def joint_state_callback(self, msg: JointState):
        self.last_joint_state = msg

    def wait_for_joint_state(self, timeout_sec=5.0):
        end_time = self.get_clock().now().nanoseconds + int(timeout_sec * 1e9)
        while rclpy.ok() and self.last_joint_state is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.get_clock().now().nanoseconds > end_time:
                return False
        return self.last_joint_state is not None

    def send_small_motion(self):
        if not self.wait_for_joint_state(timeout_sec=5.0):
            self.get_logger().error('No /joint_states received within timeout')
            return

        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('FollowJointTrajectory action server not available')
            return

        js = self.last_joint_state
        name_to_index = {name: i for i, name in enumerate(js.name)}

        arm_joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        missing = [n for n in arm_joint_names if n not in name_to_index]
        if missing:
            self.get_logger().error(f'Missing joints in /joint_states: {missing}')
            return

        current = [js.position[name_to_index[n]] for n in arm_joint_names]

        target = list(current)
        target[0] = current[0] + 0.15

        point = JointTrajectoryPoint()
        point.positions = target
        point.time_from_start = Duration(sec=2, nanosec=0)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = arm_joint_names
        goal.trajectory.points.append(point)

        self.get_logger().info(
            f'Sending goal, joint1: {current[0]:.3f} -> {target[0]:.3f}'
        )

        send_future = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by action server')
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        self.get_logger().info(
            f'Goal finished with error_code={result.error_code}, error_string="{result.error_string}"'
        )


def main(args=None):
    rclpy.init(args=args)
    node = SimpleTrajClient()
    try:
        node.send_small_motion()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()



import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse

from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory


class FollowJointTrajectoryBridgeServo(Node):
    """
    Bridge between MoveIt (FollowJointTrajectory action OR Servo JointTrajectory topic)
    and Piper's topic-based command interface, publishing commands on /piper_cmd.

    - Action server (MoveIt planners):   arm_controller/follow_joint_trajectory
    - Servo topic (MoveIt Servo):        /servo_joint_trajectory  (JointTrajectory)
    - Sends commands on:                 /piper_cmd              (remapped into Piper driver)
    - Reads feedback from:               /joint_states_feedback  (from Piper driver)
    """

    def __init__(self):
        super().__init__('follow_joint_trajectory_bridge_servo')

        # Commands to Piper driver (driver subscribes to /joint_states remapped to /piper_cmd)
        self.cmd_pub = self.create_publisher(JointState, '/piper_cmd', 10)

        # Feedback from Piper driver
        self.feedback_sub = self.create_subscription(
            JointState,
            '/joint_states_feedback',
            self.feedback_callback,
            10,
        )

        # JointTrajectory stream from MoveIt Servo
        self.servo_sub = self.create_subscription(
            JointTrajectory,
            '/servo_joint_trajectory',
            self.servo_trajectory_callback,
            10,
        )

        # Action server for FollowJointTrajectory (kept for compatibility)
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'arm_controller/follow_joint_trajectory',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        # Internal state
        self.joint_names_full = None
        self.name_to_index = {}
        self.last_feedback_positions = []
        self.last_command_positions = []

        self.get_logger().info('FollowJointTrajectory Servo bridge node started')

    # ---------- Feedback from Piper driver ----------

    def feedback_callback(self, msg: JointState):
        if self.joint_names_full is None:
            self.joint_names_full = list(msg.name)
            self.name_to_index = {name: i for i, name in enumerate(self.joint_names_full)}
            self.get_logger().info(f'Got joint order from feedback: {self.joint_names_full}')

        self.last_feedback_positions = list(msg.position)
        if not self.last_command_positions:
            self.last_command_positions = list(msg.position)

    # ---------- Action path (MoveIt planners) ----------

    def goal_callback(self, goal_request):
        self.get_logger().info('Received FollowJointTrajectory goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received request to cancel goal')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing FollowJointTrajectory goal')

        goal = goal_handle.request
        traj = goal.trajectory

        if self.joint_names_full is None:
            self.get_logger().error('No joint feedback received yet on /joint_states_feedback')
            result = FollowJointTrajectory.Result()
            result.error_code = -1
            result.error_string = 'No joint feedback received yet'
            goal_handle.abort()
            return result

        try:
            joint_indices = [self.name_to_index[name] for name in traj.joint_names]
        except KeyError as e:
            bad_name = str(e)
            self.get_logger().error(f'Unknown joint in trajectory: {bad_name}')
            result = FollowJointTrajectory.Result()
            result.error_code = -2
            result.error_string = f'Unknown joint: {bad_name}'
            goal_handle.abort()
            return result

        if not traj.points:
            self.get_logger().warn('Trajectory has no points')
            result = FollowJointTrajectory.Result()
            result.error_code = -1
            result.error_string = 'Empty trajectory'
            goal_handle.abort()
            return result

        if not self.last_command_positions:
            self.last_command_positions = (
                list(self.last_feedback_positions)
                if self.last_feedback_positions
                else [0.0] * len(self.joint_names_full)
            )

        cmd_positions = list(self.last_command_positions)
        start_time = self.get_clock().now()

        for point_idx, point in enumerate(traj.points):
            if goal_handle.is_cancel_requested:
                self.get_logger().info('Goal canceled')
                goal_handle.canceled()
                result = FollowJointTrajectory.Result()
                result.error_code = -4
                result.error_string = 'Goal canceled'
                return result

            t_target = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9

            while True:
                now = self.get_clock().now()
                elapsed = (now - start_time).nanoseconds * 1e-9
                if elapsed >= t_target:
                    break
                time.sleep(0.01)

            if point.positions:
                for i, idx in enumerate(joint_indices):
                    if i < len(point.positions):
                        cmd_positions[idx] = point.positions[i]

            cmd_msg = JointState()
            cmd_msg.header.stamp = self.get_clock().now().to_msg()
            cmd_msg.name = list(self.joint_names_full)
            cmd_msg.position = list(cmd_positions)
            self.cmd_pub.publish(cmd_msg)

            self.last_command_positions = list(cmd_positions)

            feedback_msg = FollowJointTrajectory.Feedback()
            feedback_msg.joint_names = traj.joint_names
            feedback_msg.actual.positions = [
                self.last_feedback_positions[self.name_to_index[name]]
                if name in self.name_to_index and self.last_feedback_positions
                else 0.0
                for name in traj.joint_names
            ]
            feedback_msg.desired.positions = list(point.positions)
            goal_handle.publish_feedback(feedback_msg)

        self.get_logger().info('Trajectory execution completed (servo bridge level)')

        result = FollowJointTrajectory.Result()
        result.error_code = 0
        result.error_string = ''
        goal_handle.succeed()
        return result

    # ---------- Topic path (MoveIt Servo) ----------

    def servo_trajectory_callback(self, traj: JointTrajectory):
        if self.joint_names_full is None:
            self.get_logger().warn_once(
                'Servo trajectory received but no joint feedback yet on /joint_states_feedback'
            )
            return

        if not traj.joint_names or not traj.points:
            return

        try:
            joint_indices = [self.name_to_index[name] for name in traj.joint_names]
        except KeyError as e:
            bad_name = str(e)
            self.get_logger().warn(f'Servo trajectory has unknown joint name: {bad_name}')
            return

        last_point = traj.points[-1]
        if not last_point.positions:
            return

        if not self.last_command_positions:
            self.last_command_positions = (
                list(self.last_feedback_positions)
                if self.last_feedback_positions
                else [0.0] * len(self.joint_names_full)
            )

        cmd_positions = list(self.last_command_positions)

        for i, idx in enumerate(joint_indices):
            if i < len(last_point.positions):
                cmd_positions[idx] = last_point.positions[i]

        cmd_msg = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.name = list(self.joint_names_full)
        cmd_msg.position = list(cmd_positions)
        self.cmd_pub.publish(cmd_msg)

        self.last_command_positions = list(cmd_positions)


def main(args=None):
    rclpy.init(args=args)
    node = FollowJointTrajectoryBridgeServo()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()



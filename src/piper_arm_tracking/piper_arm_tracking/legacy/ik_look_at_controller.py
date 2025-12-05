import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.srv import GetPositionIK
from builtin_interfaces.msg import Duration


class IkLookAtController(Node):
    """
    Option B controller (baseline):
    - Subscribes to /roi_centroid_pose (PoseStamped, frame_id=base_link)
    - Subscribes to /joint_states (from joint_state_bridge)
    - Calls /compute_ik (MoveIt) for a 'safe' pose near the centroid
    - If IK succeeds, sends a FollowJointTrajectory goal to
      arm_controller/follow_joint_trajectory (your bridge -> Piper SDK).
    """

    def __init__(self):
        super().__init__('ik_look_at_controller')

        # Joints expected by your arm_controller (same as MoveIt config)
        self.joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'joint7', 'joint8',
        ]

        self.latest_joint_state: JointState | None = None
        self.latest_roi_pose: PoseStamped | None = None
        self.ik_pending = False

        # ---- Parameters (tunable) ----
        self.declare_parameter('loop_rate', 0.5)        # Hz (every 2 seconds)
        self.declare_parameter('safety_offset', 0.10)   # 10 cm away from centroid
        self.declare_parameter('min_radius', 0.20)      # min distance from base_link
        self.declare_parameter('motion_time', 2.0)      # seconds for each move

        loop_rate = self.get_parameter(
            'loop_rate').get_parameter_value().double_value
        if loop_rate <= 0.0:
            loop_rate = 0.5
        period = 1.0 / loop_rate

        self.safety_offset = self.get_parameter(
            'safety_offset').get_parameter_value().double_value
        self.min_radius = self.get_parameter(
            'min_radius').get_parameter_value().double_value
        self.motion_time = self.get_parameter(
            'motion_time').get_parameter_value().double_value

        # ---- Subscriptions ----
        self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_cb,
            10,
        )

        self.create_subscription(
            PoseStamped,
            'roi_centroid_pose',
            self.roi_pose_cb,
            10,
        )

        # ---- Clients ----
        self.ik_client = self.create_client(GetPositionIK, 'compute_ik')
        self.traj_client = ActionClient(
            self,
            FollowJointTrajectory,
            'arm_controller/follow_joint_trajectory',
        )

        # Wait for IK service
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for /compute_ik service...')

        self.get_logger().info('✅ Connected to /compute_ik')

        # Wait (briefly) for trajectory action server
        if not self.traj_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn(
                'FollowJointTrajectory action server not available yet '
                '(arm_controller/follow_joint_trajectory)'
            )
        else:
            self.get_logger().info(
                '✅ Connected to arm_controller/follow_joint_trajectory'
            )

        # ---- Main timer ----
        self.timer = self.create_timer(period, self.timer_cb)

    # ==================== Callbacks ====================

    def joint_state_cb(self, msg: JointState):
        # current real arm state (via joint_state_bridge)
        self.latest_joint_state = msg

    def roi_pose_cb(self, msg: PoseStamped):
        # Expect frame_id == "base_link" from pointcloud_roi_centroid
        if msg.header.frame_id != 'base_link':
            self.get_logger().warn_once(
                f'/roi_centroid_pose frame_id is {msg.header.frame_id}, '
                'expected base_link. Check pointcloud_roi_centroid TF.'
            )
        self.latest_roi_pose = msg

    def timer_cb(self):
        # Only one IK request at a time
        if self.ik_pending:
            return

        if self.latest_joint_state is None:
            self.get_logger().warn('No /joint_states yet, skipping this cycle')
            return

        if self.latest_roi_pose is None:
            self.get_logger().warn('No /roi_centroid_pose yet, skipping this cycle')
            return

        # Build a "safe" pose near ROI (same structure as ik_look_at_debug)
        safe_pose = self.make_safe_pose(self.latest_roi_pose)

        rp = self.latest_roi_pose.pose.position
        sp = safe_pose.pose.position
        self.get_logger().info(
            f'[controller] ROI in base_link: '
            f'({rp.x:.3f}, {rp.y:.3f}, {rp.z:.3f}) '
            f'-> safe: ({sp.x:.3f}, {sp.y:.3f}, {sp.z:.3f})'
        )

        # ---- Build IK request, mirroring ik_look_at_debug ----
        req = GetPositionIK.Request()
        req.ik_request.group_name = 'arm'             # MoveIt planning group
        req.ik_request.ik_link_name = 'gripper_base'  # end-effector link
        req.ik_request.pose_stamped = safe_pose

        # Let MoveIt use its internal "current state" (from /joint_states)
        # dont set req.ik_request.robot_state.joint_state here.
        req.ik_request.timeout = Duration(sec=0, nanosec=200_000_000)  # 0.2 s
        req.ik_request.avoid_collisions = False

        self.ik_pending = True
        future = self.ik_client.call_async(req)
        future.add_done_callback(self.ik_response_cb)

        p = safe_pose.pose.position
        self.get_logger().info(
            f'Trying IK for safe pose at ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) in base_link'
        )

    # ==================== Helpers ====================

    def make_safe_pose(self, roi_pose: PoseStamped) -> PoseStamped:
        """
        Take the raw ROI point in base_link and pull the target *towards the robot*
        so the EEF stops short of the object (safety_offset).
        Also clamp to a simple workspace box.
        """
        raw = roi_pose.pose.position
        dx, dy, dz = raw.x, raw.y, raw.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        safe_pose = PoseStamped()
        safe_pose.header.frame_id = 'base_link'
        safe_pose.header.stamp = self.get_clock().now().to_msg()

        if dist < 1e-3:
            # Degenerate case – just pick a default "look forward" point
            safe_pose.pose.position.x = 0.35
            safe_pose.pose.position.y = 0.0
            safe_pose.pose.position.z = 0.25
        else:
            # How far from the base the EEF should be
            desired_dist = max(self.min_radius, dist - self.safety_offset)
            scale = desired_dist / dist
            safe_pose.pose.position.x = dx * scale
            safe_pose.pose.position.y = dy * scale
            safe_pose.pose.position.z = dz * scale

        # rough workspace limits (same idea as ik_look_at_debug)
        safe_pose.pose.position.x = max(0.15, min(0.70, safe_pose.pose.position.x))
        safe_pose.pose.position.y = max(-0.30, min(0.30, safe_pose.pose.position.y))
        safe_pose.pose.position.z = max(0.05, min(0.60, safe_pose.pose.position.z))

        # Orientation: let MoveIt choose a valid IK solution
        safe_pose.pose.orientation.x = 0.0
        safe_pose.pose.orientation.y = 0.0
        safe_pose.pose.orientation.z = 0.0
        safe_pose.pose.orientation.w = 1.0

        return safe_pose

    def ik_response_cb(self, future):
        self.ik_pending = False

        if future.cancelled():
            self.get_logger().warn('IK service call was cancelled')
            return
        if future.exception() is not None:
            self.get_logger().error(f'IK service call failed: {future.exception()}')
            return

        resp = future.result()
        if resp.error_code.val != resp.error_code.SUCCESS:
            self.get_logger().warn(
                f'IK failed, error_code = {resp.error_code.val}')
            return

        js = resp.solution.joint_state
        # Log IK solution
        name_to_pos = {n: p for n, p in zip(js.name, js.position)}
        self.get_logger().info('✅ IK SUCCESS (controller):')
        for j in self.joint_names:
            self.get_logger().info(f'    {j}: {name_to_pos.get(j, 0.0):.3f}')

        # Build and send trajectory to arm_controller
        if self.latest_joint_state is None:
            self.get_logger().warn(
                'Latest /joint_states missing when building trajectory; skipping send'
            )
            return

        target_positions = self.extract_positions(js, self.joint_names)
        current_positions = self.extract_positions(
            self.latest_joint_state, self.joint_names)

        traj = JointTrajectory()
        traj.joint_names = list(self.joint_names)

        start = JointTrajectoryPoint()
        start.positions = current_positions
        start.time_from_start = Duration(sec=0, nanosec=0)

        end = JointTrajectoryPoint()
        end.positions = target_positions
        end.time_from_start = Duration(
            sec=int(self.motion_time),
            nanosec=int((self.motion_time % 1.0) * 1e9),
        )

        traj.points.append(start)
        traj.points.append(end)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        send_future = self.traj_client.send_goal_async(goal)
        send_future.add_done_callback(self.traj_goal_response_cb)

    @staticmethod
    def extract_positions(joint_state: JointState,
                          names: list[str]) -> list[float]:
        """
        Return a list of positions ordered according to `names`.
        If a joint is missing, default to 0.0.
        """
        out = []
        name_to_pos = {n: p for n, p in zip(joint_state.name, joint_state.position)}
        for n in names:
            out.append(name_to_pos.get(n, 0.0))
        return out

    def traj_goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Trajectory goal rejected by arm_controller')
            return

        self.get_logger().info('Trajectory goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.traj_result_cb)

    def traj_result_cb(self, future):
        result = future.result().result
        self.get_logger().info(
            f'Trajectory finished with error_code = {result.error_code}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = IkLookAtController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()



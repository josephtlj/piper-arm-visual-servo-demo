#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Pose
from trajectory_msgs.msg import JointTrajectory
from control_msgs.action import FollowJointTrajectory

from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import Constraints, PositionConstraint
from moveit_msgs.msg import RobotState, MotionPlanRequest
from shape_msgs.msg import SolidPrimitive


class WheelMoveitController(Node):
    """
    Wheel (YOLO target) -> MoveIt planner controller.

    - Subscribes to /wheel_target_pose (PoseStamped in base_link)
    - Subscribes to /joint_states (from joint_state_bridge)
    - Builds a "safe pose" with stand-off and workspace clamps
    - Smooths that safe pose over time (low-pass filter)
    - Only replans if the filtered safe pose moves more than a threshold
    - If the wheel is too close, first tries to move the arm BACK to a
      comfortable distance (close_range_threshold).
      If that "retreat" fails once, it stops trying to move back and will
      only move towards the wheel with the usual 20cm stand-off.
    - Calls MoveIt planner via /plan_kinematic_path (GetMotionPlan)
    - Sends the returned JointTrajectory to arm_controller/follow_joint_trajectory
    """

    def __init__(self):
        super().__init__('wheel_moveit_controller')

        # Joints expected by arm_controller / MoveIt config
        self.joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'joint7', 'joint8'
        ]

        self.latest_joint_state: JointState | None = None
        self.latest_target_pose: PoseStamped | None = None

        # Flags for planning/execution state
        self.plan_pending = False     # waiting for MoveIt plan response
        self.busy = False             # planning OR executing a trajectory

        # For jitter reduction / smoothing
        self.last_safe_pose_filtered: PoseStamped | None = None
        self.target_was_out_of_range = False

        # Retreat behaviour
        self.close_limit_reached = False     # once True, stop trying to move back
        self.last_goal_was_retreat = False   # remember the last plan

        # ---------------- Parameters ----------------
        # Main loop rate (how often it considers replanning)
        self.declare_parameter('loop_rate', 0.5)  # Hz (every 2 seconds)

        # Geometry / stand-off
        self.declare_parameter('safety_offset', 0.20)   # 20cm short of centroid (normal case)
        self.declare_parameter('min_radius', 0.25)      # min distance from base_link
        self.declare_parameter('pos_tolerance', 0.02)   # 2cm sphere around goal

        # Planning
        self.declare_parameter('allowed_planning_time', 1.0)  # seconds

        # Smoothing & replan threshold
        self.declare_parameter('safe_pose_alpha', 0.4)  # 0–1; higher = follow faster
        self.declare_parameter('min_goal_delta', 0.03)  # 3cm change before replanning

        # Tracking range (in base_link distance to target)
        self.declare_parameter('min_tracking_range', 0.25)  # m (not used for stopping yet)
        self.declare_parameter('max_tracking_range', 0.80)  # m (beyond this: no planning)

        # "Too close" range: if target is closer than this and has not hit
        # the back limit yet, try to move the arm BACK to this distance.
        self.declare_parameter('close_range_threshold', 0.40)  # m, must be > min_radius

        loop_rate = self.get_parameter('loop_rate').get_parameter_value().double_value
        if loop_rate <= 0.0:
            loop_rate = 0.5
        period = 1.0 / loop_rate

        self.safety_offset = self.get_parameter(
            'safety_offset').get_parameter_value().double_value
        self.min_radius = self.get_parameter(
            'min_radius').get_parameter_value().double_value
        self.pos_tolerance = self.get_parameter(
            'pos_tolerance').get_parameter_value().double_value
        self.allowed_planning_time = self.get_parameter(
            'allowed_planning_time').get_parameter_value().double_value
        self.safe_pose_alpha = self.get_parameter(
            'safe_pose_alpha').get_parameter_value().double_value
        self.min_goal_delta = self.get_parameter(
            'min_goal_delta').get_parameter_value().double_value
        self.min_tracking_range = self.get_parameter(
            'min_tracking_range').get_parameter_value().double_value
        self.max_tracking_range = self.get_parameter(
            'max_tracking_range').get_parameter_value().double_value
        self.close_range_threshold = self.get_parameter(
            'close_range_threshold').get_parameter_value().double_value

        # ---------------- Subscriptions ----------------
        self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_cb,
            10,
        )

        # Subscribe to /wheel_target_pose (PoseStamped in base_link)
        self.create_subscription(
            PoseStamped,
            'wheel_target_pose',
            self.target_pose_cb,
            10,
        )

        # ---------------- Clients ----------------
        # MoveIt planning service (exposed by move_group)
        self.plan_client = self.create_client(GetMotionPlan, 'plan_kinematic_path')

        # Action client to send trajectory to Piper bridge
        self.traj_client = ActionClient(
            self,
            FollowJointTrajectory,
            'arm_controller/follow_joint_trajectory',
        )

        # Wait for planner
        while not self.plan_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for /plan_kinematic_path (GetMotionPlan)...')

        self.get_logger().info('✅ Connected to /plan_kinematic_path')

        # Wait briefly for trajectory action server
        if not self.traj_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn(
                'FollowJointTrajectory action server not available yet '
                '(arm_controller/follow_joint_trajectory)'
            )
        else:
            self.get_logger().info(
                '✅ Connected to arm_controller/follow_joint_trajectory')

        # ---------------- Timer ----------------
        self.timer = self.create_timer(period, self.timer_cb)

    # ==================== Callbacks ====================

    def joint_state_cb(self, msg: JointState):
        self.latest_joint_state = msg

    def target_pose_cb(self, msg: PoseStamped):
        # Expect frame_id == "base_link"
        if msg.header.frame_id != 'base_link':
            self.get_logger().warn_once(
                f'/wheel_target_pose frame_id is {msg.header.frame_id}, '
                'expected base_link. Check wheel_yolo_detector TF transform.'
            )
        self.latest_target_pose = msg

    def timer_cb(self):
        # If already planning or executing a trajectory, skip this cycle
        if self.busy or self.plan_pending:
            return

        if self.latest_joint_state is None:
            self.get_logger().warn('No /joint_states yet, skipping this cycle')
            return

        if self.latest_target_pose is None:
            self.get_logger().warn('No /wheel_target_pose yet, skipping this cycle')
            return

        # ---------------- Range gating (too far = no planning) ----------------
        tgt_p = self.latest_target_pose.pose.position
        tgt_dist = math.sqrt(tgt_p.x * tgt_p.x + tgt_p.y * tgt_p.y + tgt_p.z * tgt_p.z)

        if tgt_dist > self.max_tracking_range:
            if not self.target_was_out_of_range:
                self.get_logger().warn(
                    f'[wheel_moveit] target at {tgt_dist:.3f} m > max_tracking_range '
                    f'{self.max_tracking_range:.3f} m → target too far, not planning'
                )
                self.target_was_out_of_range = True
            return
        else:
            if self.target_was_out_of_range:
                self.get_logger().info(
                    '[wheel_moveit] target back in range, resuming normal tracking'
                )
                self.target_was_out_of_range = False

        # Decide if in "retreat" mode (too close and still allowed to move back)
        retreat_mode = (tgt_dist < self.close_range_threshold
                        and not self.close_limit_reached)
        self.last_goal_was_retreat = retreat_mode

        # ---------------- Safe pose computation & smoothing ----------------
        # 1) Raw safe pose from target (stand-off or retreat)
        safe_pose_raw = self.make_safe_pose(self.latest_target_pose, retreat_mode)

        # 2) Smooth safe pose using low-pass filter
        prev_filtered = self.last_safe_pose_filtered
        safe_pose_filtered = self.compute_filtered_safe_pose(safe_pose_raw, prev_filtered)

        # 3) If filtered safe pose barely moved, skip replan (reduces jitter)
        if prev_filtered is not None:
            lp = prev_filtered.pose.position
            p = safe_pose_filtered.pose.position
            dx = p.x - lp.x
            dy = p.y - lp.y
            dz = p.z - lp.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            if dist < self.min_goal_delta:
                # Too small change → ignore this cycle
                return

        # Accept the new filtered pose as last
        self.last_safe_pose_filtered = safe_pose_filtered

        # Log the pose given to MoveIt
        p = safe_pose_filtered.pose.position
        if retreat_mode and not self.close_limit_reached:
            self.get_logger().info(
                f'[wheel_moveit] RETREAT: filtered safe pose in base_link: '
                f'({p.x:.3f}, {p.y:.3f}, {p.z:.3f})'
            )
        else:
            self.get_logger().info(
                f'[wheel_moveit] Filtered safe pose in base_link: '
                f'({p.x:.3f}, {p.y:.3f}, {p.z:.3f})'
            )

        # ---------------- Build MotionPlanRequest for MoveIt ----------------
        req = GetMotionPlan.Request()
        mpr = MotionPlanRequest()

        # Planning group name from MoveIt config
        mpr.group_name = 'arm'

        # Start state: actual robot joints from joint_state_bridge
        start_state = RobotState()
        start_state.joint_state = self.latest_joint_state
        mpr.start_state = start_state

        # Goal: position constraint around safe_pose_filtered
        goal_constraints = Constraints()

        pos_con = PositionConstraint()
        pos_con.header.frame_id = 'base_link'
        pos_con.link_name = 'gripper_base'
        pos_con.weight = 1.0

        # Small sphere around the safe pose as target region
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [self.pos_tolerance]  # radius

        region_pose = Pose()
        region_pose.position = safe_pose_filtered.pose.position
        region_pose.orientation.w = 1.0  # orientation of constraint region

        pos_con.constraint_region.primitives = [sphere]
        pos_con.constraint_region.primitive_poses = [region_pose]

        goal_constraints.position_constraints = [pos_con]

        # (No explicit orientation constraint – let MoveIt pick a valid one)
        mpr.goal_constraints = [goal_constraints]

        # Planning options
        mpr.allowed_planning_time = self.allowed_planning_time
        mpr.num_planning_attempts = 5

        req.motion_plan_request = mpr

        # Mark as busy: now planning (and later executing)
        self.busy = True
        self.plan_pending = True

        future = self.plan_client.call_async(req)
        future.add_done_callback(self.plan_response_cb)

    # ==================== Helpers ====================

    def compute_filtered_safe_pose(
        self,
        raw_pose: PoseStamped,
        prev_filtered: PoseStamped | None
    ) -> PoseStamped:
        """
        Low-pass filter on the safe pose to avoid sudden jumps.

        If there is no previous filtered pose, return the raw pose.
        Otherwise, blend positions using safe_pose_alpha.
        Orientation is kept from the raw pose (identity).
        """
        if prev_filtered is None:
            return raw_pose

        alpha = self.safe_pose_alpha
        alpha = max(0.0, min(1.0, alpha))

        sp = PoseStamped()
        sp.header = raw_pose.header

        rp = raw_pose.pose.position
        lp = prev_filtered.pose.position

        sp.pose.position.x = alpha * rp.x + (1.0 - alpha) * lp.x
        sp.pose.position.y = alpha * rp.y + (1.0 - alpha) * lp.y
        sp.pose.position.z = alpha * rp.z + (1.0 - alpha) * lp.z

        # Keep simple orientation (from raw)
        sp.pose.orientation = raw_pose.pose.orientation

        return sp

    def make_safe_pose(self, target_pose: PoseStamped, retreat_mode: bool) -> PoseStamped:
        """
        Take the raw target point in base_link and:

        - If retreat_mode=True and close_limit_reached=False:
            try to place the EEF at a fixed distance close_range_threshold
            from base_link (i.e. move BACK away from a too-close wheel).
        - Otherwise:
            use the usual stand-off: distance = dist - safety_offset,
            but never closer than min_radius.
        Also clamp to a simple workspace box.
        """
        raw = target_pose.pose.position
        dx, dy, dz = raw.x, raw.y, raw.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        safe_pose = PoseStamped()
        safe_pose.header.frame_id = 'base_link'
        safe_pose.header.stamp = self.get_clock().now().to_msg()

        if dist < 1e-3:
            # Degenerate case – a default "look forward" point
            safe_pose.pose.position.x = 0.35
            safe_pose.pose.position.y = 0.0
            safe_pose.pose.position.z = 0.25
        else:
            if retreat_mode and not self.close_limit_reached:
                # Too close: try to move the arm BACK to a comfortable distance.
                # This sets the EEF radius to close_range_threshold.
                desired_dist = max(self.close_range_threshold, self.min_radius)
            else:
                # Normal tracking: maintain a 20cm stand-off from the wheel,
                # but never closer than min_radius.
                desired_dist = max(self.min_radius, dist - self.safety_offset)

            # Project along the ray from base_link to target
            scale = desired_dist / dist
            safe_pose.pose.position.x = dx * scale
            safe_pose.pose.position.y = dy * scale
            safe_pose.pose.position.z = dz * scale

        # Very rough workspace limits (tweak as needed)
        safe_pose.pose.position.x = max(0.15, min(0.70, safe_pose.pose.position.x))
        safe_pose.pose.position.y = max(-0.30, min(0.30, safe_pose.pose.position.y))
        safe_pose.pose.position.z = max(0.05, min(0.60, safe_pose.pose.position.z))

        # Orientation: keep it simple, planner will find a valid IK
        safe_pose.pose.orientation.x = 0.0
        safe_pose.pose.orientation.y = 0.0
        safe_pose.pose.orientation.z = 0.0
        safe_pose.pose.orientation.w = 1.0

        return safe_pose

    # ==================== Planner response ====================

    def plan_response_cb(self, future):
        # no longer waiting for the planner response
        self.plan_pending = False

        if future.cancelled():
            self.get_logger().warn('[wheel_moveit] Motion plan request cancelled')
            self.busy = False
            return

        if future.exception() is not None:
            self.get_logger().error(
                f'[wheel_moveit] Motion plan service call failed: {future.exception()}'
            )
            self.busy = False
            return

        resp = future.result()
        mp_resp = resp.motion_plan_response

        # MoveIt error codes: SUCCESS = 1, others negative or non-1
        if mp_resp.error_code.val != 1:
            # If trying a "retreat" move and it failed, mark the retreat
            # limit as reached so it stops trying to move further back.
            if self.last_goal_was_retreat and not self.close_limit_reached:
                self.close_limit_reached = True
                self.get_logger().warn(
                    f'[wheel_moveit] Retreat planning failed (error_code={mp_resp.error_code.val}). '
                    'Marking close_limit_reached=True. Will keep current pose and from now on '
                    'only move towards the wheel with the normal 20cm stand-off.'
                )
            else:
                self.get_logger().warn(
                    f'[wheel_moveit] Planning failed, error_code = {mp_resp.error_code.val}'
                )

            self.busy = False
            return

        traj: JointTrajectory = mp_resp.trajectory.joint_trajectory
        if not traj.points:
            self.get_logger().warn('[wheel_moveit] Planning succeeded but trajectory is empty')
            self.busy = False
            return

        self.get_logger().info(
            f'[wheel_moveit] Planning SUCCESS, trajectory has {len(traj.points)} points. '
            'Sending to arm_controller...'
        )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        send_future = self.traj_client.send_goal_async(goal)
        send_future.add_done_callback(self.traj_goal_response_cb)

    def traj_goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('[wheel_moveit] Trajectory goal rejected by arm_controller')
            # Planning done, execution never started -> clear busy
            self.busy = False
            return

        self.get_logger().info('[wheel_moveit] Trajectory goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.traj_result_cb)

    def traj_result_cb(self, future):
        result = future.result().result
        self.get_logger().info(
            f'[wheel_moveit] Trajectory finished with error_code = {result.error_code}'
        )
        # Execution finished -> free to plan again
        self.busy = False


def main(args=None):
    rclpy.init(args=args)
    node = WheelMoveitController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()




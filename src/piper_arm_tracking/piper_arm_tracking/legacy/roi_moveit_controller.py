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


class RoiMoveitController(Node):
    """
    ROI → MoveIt planner controller.

    - Subscribes to /roi_centroid_pose (PoseStamped in base_link)
    - Subscribes to /joint_states (from joint_state_bridge)
    - Builds a "safe pose" with stand-off and workspace clamps
    - Smooths that safe pose over time (low-pass filter)
    - Only replans if the filtered safe pose moves more than a threshold
    - Calls MoveIt planner via /plan_kinematic_path (GetMotionPlan)
    - Sends the returned JointTrajectory to arm_controller/follow_joint_trajectory
    """

    def __init__(self):
        super().__init__('roi_moveit_controller')

        # Joints expected by arm_controller / MoveIt config
        self.joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'joint7', 'joint8'
        ]

        self.latest_joint_state: JointState | None = None
        self.latest_roi_pose: PoseStamped | None = None

        # Flags for planning/execution state
        self.plan_pending = False     # waiting for MoveIt plan response
        self.busy = False             # planning OR executing a trajectory

        # For jitter reduction / smoothing
        self.last_safe_pose_filtered: PoseStamped | None = None
        self.target_was_out_of_range = False

        # ---------------- Parameters ----------------
        # Main loop rate (how often we consider replanning)
        self.declare_parameter('loop_rate', 0.5)  # Hz (every 2 seconds)

        # Geometry / stand-off
        self.declare_parameter('safety_offset', 0.20)   # 20cm short of centroid
        self.declare_parameter('min_radius', 0.25)      # min distance from base_link
        self.declare_parameter('pos_tolerance', 0.02)   # 2cm sphere around goal

        # Planning
        self.declare_parameter('allowed_planning_time', 1.0)  # seconds

        # Smoothing & replan threshold
        self.declare_parameter('safe_pose_alpha', 0.4)  # 0–1; higher = follow faster
        self.declare_parameter('min_goal_delta', 0.03)  # 3cm change before replanning

        # Tracking range (in base_link distance to ROI)
        self.declare_parameter('min_tracking_range', 0.25)  # m (not used for stopping)
        self.declare_parameter('max_tracking_range', 0.80)  # m (beyond this = no planning)

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

        # ---------------- Subscriptions ----------------
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

        # Wait breifly for trajectory action server
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

    def roi_pose_cb(self, msg: PoseStamped):
        # Expect frame_id == "base_link"
        if msg.header.frame_id != 'base_link':
            self.get_logger().warn_once(
                f'/roi_centroid_pose frame_id is {msg.header.frame_id}, '
                'expected base_link. Check pointcloud_roi_centroid TF transform.'
            )
        self.latest_roi_pose = msg

    def timer_cb(self):
        # If already planning or executing a trajectory, skip this cycle
        if self.busy or self.plan_pending:
            return

        if self.latest_joint_state is None:
            self.get_logger().warn('No /joint_states yet, skipping this cycle')
            return

        if self.latest_roi_pose is None:
            self.get_logger().warn('No /roi_centroid_pose yet, skipping this cycle')
            return

        # ---------------- Range gating (too far → no planning) ----------------
        roi_p = self.latest_roi_pose.pose.position
        roi_dist = math.sqrt(roi_p.x * roi_p.x + roi_p.y * roi_p.y + roi_p.z * roi_p.z)

        if roi_dist > self.max_tracking_range:
            if not self.target_was_out_of_range:
                self.get_logger().warn(
                    f'[roi_moveit] ROI at {roi_dist:.3f} m > max_tracking_range '
                    f'{self.max_tracking_range:.3f} m → target too far, not planning'
                )
                self.target_was_out_of_range = True
            return
        else:
            if self.target_was_out_of_range:
                self.get_logger().info(
                    '[roi_moveit] ROI back in range, resuming normal tracking'
                )
                self.target_was_out_of_range = False

        # ---------------- Safe pose computation & smoothing ----------------
        # 1) Raw safe pose from ROI (20cm stand-off, workspace clamps)
        safe_pose_raw = self.make_safe_pose(self.latest_roi_pose)

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

        # Accept new filtered pose as last
        self.last_safe_pose_filtered = safe_pose_filtered

        # Log the pose given to MoveIt
        p = safe_pose_filtered.pose.position
        self.get_logger().info(
            f'[roi_moveit] Filtered safe pose in base_link: '
            f'({p.x:.3f}, {p.y:.3f}, {p.z:.3f})'
        )

        # ---------------- Build MotionPlanRequest for MoveIt ----------------
        req = GetMotionPlan.Request()
        mpr = MotionPlanRequest()

        # Planning group name from the MoveIt config
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
        if alpha < 0.0:
            alpha = 0.0
        if alpha > 1.0:
            alpha = 1.0

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
            # Degenerate case – a default "look forward" point
            safe_pose.pose.position.x = 0.35
            safe_pose.pose.position.y = 0.0
            safe_pose.pose.position.z = 0.25
        else:
            # Stand-off from the ROI point, but never closer than min_radius
            desired_dist = max(self.min_radius, dist - self.safety_offset)
            scale = desired_dist / dist
            safe_pose.pose.position.x = dx * scale
            safe_pose.pose.position.y = dy * scale
            safe_pose.pose.position.z = dz * scale

        # Very rough workspace limits (tweak when meeded)
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
            self.get_logger().warn('[roi_moveit] Motion plan request cancelled')
            self.busy = False
            return

        if future.exception() is not None:
            self.get_logger().error(
                f'[roi_moveit] Motion plan service call failed: {future.exception()}'
            )
            self.busy = False
            return

        resp = future.result()
        mp_resp = resp.motion_plan_response

        # MoveIt error codes: SUCCESS = 1, others negative or non-1
        if mp_resp.error_code.val != 1:
            self.get_logger().warn(
                f'[roi_moveit] Planning failed, error_code = {mp_resp.error_code.val}'
            )
            self.busy = False
            return

        traj: JointTrajectory = mp_resp.trajectory.joint_trajectory
        if not traj.points:
            self.get_logger().warn('[roi_moveit] Planning succeeded but trajectory is empty')
            self.busy = False
            return

        self.get_logger().info(
            f'[roi_moveit] Planning SUCCESS, trajectory has {len(traj.points)} points. '
            'Sending to arm_controller...'
        )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        send_future = self.traj_client.send_goal_async(goal)
        send_future.add_done_callback(self.traj_goal_response_cb)

    def traj_goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('[roi_moveit] Trajectory goal rejected by arm_controller')
            # Planning done, execution never started → clear busy
            self.busy = False
            return

        self.get_logger().info('[roi_moveit] Trajectory goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.traj_result_cb)

    def traj_result_cb(self, future):
        result = future.result().result
        self.get_logger().info(
            f'[roi_moveit] Trajectory finished with error_code = {result.error_code}'
        )
        # Execution finished -> free to plan again
        self.busy = False


def main(args=None):
    rclpy.init(args=args)
    node = RoiMoveitController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()



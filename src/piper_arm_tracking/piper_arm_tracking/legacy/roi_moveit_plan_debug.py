import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

from moveit_msgs.srv import GetPositionIK, GetMotionPlan
from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint
from builtin_interfaces.msg import Duration


class RoiMoveitPlanDebug(Node):
    """
    Option A (MoveIt planning) – DEBUG VERSION, **no hardware motion**.

    Pipeline:
      /roi_centroid_pose (PoseStamped in base_link)
        -> safe pose (pulled back + clamped)
        -> /compute_ik  (GetPositionIK)
        -> joint targets
        -> /plan_kinematic_path (GetMotionPlan)
        -> print if a trajectory plan exists (number of points)
    """

    def __init__(self):
        super().__init__('roi_moveit_plan_debug')

        # Joint order must match MoveIt group / URDF
        self.joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'joint7', 'joint8'
        ]

        self.latest_joint_state: JointState | None = None
        self.latest_roi_pose: PoseStamped | None = None
        self.planning_in_flight = False

        # -------- Parameters ( adjust later) --------
        self.declare_parameter('loop_rate', 0.5)        # Hz
        self.declare_parameter('safety_offset', 0.10)   # m
        self.declare_parameter('min_radius', 0.20)      # m
        self.declare_parameter('planning_time', 1.0)    # s

        loop_rate = self.get_parameter('loop_rate').get_parameter_value().double_value
        if loop_rate <= 0.0:
            loop_rate = 0.5
        period = 1.0 / loop_rate

        self.safety_offset = self.get_parameter(
            'safety_offset').get_parameter_value().double_value
        self.min_radius = self.get_parameter(
            'min_radius').get_parameter_value().double_value
        self.planning_time = self.get_parameter(
            'planning_time').get_parameter_value().double_value

        # -------- Subscriptions --------
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

        # -------- MoveIt service clients --------
        # 1) IK solver
        self.ik_client = self.create_client(GetPositionIK, 'compute_ik')

        # 2) Motion planner
        # NOTE: service name may be 'plan_kinematic_path' in setup.
        # Confirm with:  ros2 service list | grep plan
        self.plan_client = self.create_client(GetMotionPlan, 'plan_kinematic_path')

        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for /compute_ik service...')

        self.get_logger().info('✅ Connected to /compute_ik')

        while not self.plan_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for /plan_kinematic_path (GetMotionPlan)...')

        self.get_logger().info('✅ Connected to /plan_kinematic_path')

        # -------- Main timer --------
        self.timer = self.create_timer(period, self.timer_cb)

    # ==================== Callbacks ====================

    def joint_state_cb(self, msg: JointState):
        self.latest_joint_state = msg

    def roi_pose_cb(self, msg: PoseStamped):
        if msg.header.frame_id != 'base_link':
            self.get_logger().warn_once(
                f'/roi_centroid_pose frame_id is {msg.header.frame_id}, '
                'expected base_link. Check pointcloud_roi_centroid TF transform.'
            )
        self.latest_roi_pose = msg

    def timer_cb(self):
        if self.planning_in_flight:
            return

        if self.latest_joint_state is None:
            self.get_logger().warn('No /joint_states yet, skipping this cycle')
            return

        if self.latest_roi_pose is None:
            self.get_logger().warn('No /roi_centroid_pose yet, skipping this cycle')
            return

        # 1) Make a safe pose from the raw ROI centroid
        safe_pose = self.make_safe_pose(self.latest_roi_pose)

        p = safe_pose.pose.position
        self.get_logger().info(
            f'[plan-debug] safe pose in base_link: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})'
        )

        # 2) Build IK request (same style as ik_look_at_debug)
        ik_req = GetPositionIK.Request()
        ik_req.ik_request.group_name = 'arm'
        ik_req.ik_request.ik_link_name = 'gripper_base'
        ik_req.ik_request.pose_stamped = safe_pose

        # Let MoveIt pick a start state internally for IK
        ik_req.ik_request.timeout = Duration(sec=0, nanosec=200_000_000)  # 0.2s

        #  MoveIt versions have 'attempts', some don't. guard it.
        if hasattr(ik_req.ik_request, 'attempts'):
            ik_req.ik_request.attempts = 5



        self.planning_in_flight = True
        future = self.ik_client.call_async(ik_req)
        future.add_done_callback(self.ik_response_cb)

    # ==================== Helpers ====================

    def make_safe_pose(self, roi_pose: PoseStamped) -> PoseStamped:
        """
        Copy of the "safe target" logic:
        - take ROI point in base_link
        - pull the target towards the robot by safety_offset
        - enforce a min_radius
        - clamp to a simple box workspace
        """
        raw = roi_pose.pose.position
        dx, dy, dz = raw.x, raw.y, raw.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        safe_pose = PoseStamped()
        safe_pose.header.frame_id = 'base_link'
        safe_pose.header.stamp = self.get_clock().now().to_msg()

        if dist < 1e-3:
            # Degenerate case, default "look forward"
            safe_pose.pose.position.x = 0.35
            safe_pose.pose.position.y = 0.0
            safe_pose.pose.position.z = 0.25
        else:
            desired_dist = max(self.min_radius, dist - self.safety_offset)
            scale = desired_dist / dist
            safe_pose.pose.position.x = dx * scale
            safe_pose.pose.position.y = dy * scale
            safe_pose.pose.position.z = dz * scale

        # Clamp workspace
        safe_pose.pose.position.x = max(0.15, min(0.70, safe_pose.pose.position.x))
        safe_pose.pose.position.y = max(-0.30, min(0.30, safe_pose.pose.position.y))
        safe_pose.pose.position.z = max(0.05, min(0.60, safe_pose.pose.position.z))

        # Simple orientation – MoveIt chooses a valid one
        safe_pose.pose.orientation.x = 0.0
        safe_pose.pose.orientation.y = 0.0
        safe_pose.pose.orientation.z = 0.0
        safe_pose.pose.orientation.w = 1.0

        return safe_pose

    def ik_response_cb(self, future):
        # Called when /compute_ik returns
        if future.cancelled():
            self.get_logger().warn('[plan-debug] IK service call cancelled')
            self.planning_in_flight = False
            return
        if future.exception() is not None:
            self.get_logger().error(
                f'[plan-debug] IK service call failed: {future.exception()}')
            self.planning_in_flight = False
            return

        ik_res = future.result()
        if ik_res.error_code.val != ik_res.error_code.SUCCESS:
            self.get_logger().warn(
                f'[plan-debug] IK failed, error_code = {ik_res.error_code.val}')
            self.planning_in_flight = False
            return

        # Extract joint target for group
        target_state = ik_res.solution.joint_state
        joint_targets = self.extract_positions(target_state, self.joint_names)

        self.get_logger().info('[plan-debug] IK SUCCESS, calling GetMotionPlan...')

        # Build MotionPlanRequest
        plan_req = GetMotionPlan.Request()
        mpr = MotionPlanRequest()
        mpr.group_name = 'arm'
        mpr.num_planning_attempts = 1
        mpr.allowed_planning_time = self.planning_time

        # Start state = current joints from /joint_states
        mpr.start_state.joint_state = self.latest_joint_state

        # Goal constraints: one JointConstraint per joint
        goal_constraints = Constraints()
        for name, pos in zip(self.joint_names, joint_targets):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = 0.05  # ~3 degrees
            jc.tolerance_below = 0.05
            jc.weight = 1.0
            goal_constraints.joint_constraints.append(jc)

        mpr.goal_constraints.append(goal_constraints)
        plan_req.motion_plan_request = mpr

        plan_future = self.plan_client.call_async(plan_req)
        plan_future.add_done_callback(self.plan_response_cb)

    @staticmethod
    def extract_positions(joint_state: JointState, names: list[str]) -> list[float]:
        name_to_pos = {n: p for n, p in zip(joint_state.name, joint_state.position)}
        return [float(name_to_pos.get(n, 0.0)) for n in names]

    def plan_response_cb(self, future):
        # Called when GetMotionPlan returns
        self.planning_in_flight = False

        if future.cancelled():
            self.get_logger().warn('[plan-debug] Motion plan call cancelled')
            return
        if future.exception() is not None:
            self.get_logger().error(
                f'[plan-debug] Motion plan call failed: {future.exception()}')
            return

        res = future.result()

        # depending on MoveIt version, the field can be .motion_plan_response or .motion_plan
        try:
            mpr = res.motion_plan_response
        except AttributeError:
            mpr = res.motion_plan  # fallback if using newer naming

        if mpr.error_code.val != mpr.error_code.SUCCESS:
            self.get_logger().warn(
                f'[plan-debug] Motion plan FAILED, error_code = {mpr.error_code.val}')
            return

        traj = mpr.trajectory.joint_trajectory
        n_points = len(traj.points)
        self.get_logger().info(
            f'[plan-debug] Motion plan SUCCESS, trajectory has {n_points} points'
        )
        # DEBUG MODE: stop here. No trajectory is sent to the arm yet.


def main(args=None):
    rclpy.init(args=args)
    node = RoiMoveitPlanDebug()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()



import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from builtin_interfaces.msg import Duration


class IkLookAtDebugFixed(Node):
    """
    Debug node: call /compute_ik on a SINGLE fixed pose in base_link,
    using the current /joint_states as the seed.

    Goal: behave as close as possible to ik_look_at_debug, just without
    the ROI / pointcloud part, see if IK itself is consistent.
    """

    def __init__(self):
        super().__init__('ik_look_at_debug_fixed')

        # EXACTLY the same as ik_look_at_debug
        self.group_name = 'arm'
        self.ik_link_name = 'gripper_base'

        self.latest_joint_state: JointState | None = None
        self.ik_client = self.create_client(GetPositionIK, 'compute_ik')

        # Subscribe to /joint_states (from joint_state_bridge or GUI)
        self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_cb,
            10,
        )

        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for /compute_ik service...')

        self.get_logger().info('✅ ik_look_at_debug_fixed connected to /compute_ik')

        # Call IK every 2 seconds
        self.timer = self.create_timer(2.0, self.timer_cb)

    # ---------------- Callbacks ----------------

    def joint_state_cb(self, msg: JointState):
        self.latest_joint_state = msg

    def timer_cb(self):
        if self.latest_joint_state is None:
            self.get_logger().warn('No /joint_states yet, skipping this cycle')
            return

        # pose that previously worked in ik_look_at_debug
        fixed_pose = PoseStamped()
        fixed_pose.header.frame_id = 'base_link'
        fixed_pose.header.stamp = self.get_clock().now().to_msg()
        fixed_pose.pose.position.x = 0.307
        fixed_pose.pose.position.y = 0.001
        fixed_pose.pose.position.z = 0.266
        fixed_pose.pose.orientation.x = 0.0
        fixed_pose.pose.orientation.y = 0.0
        fixed_pose.pose.orientation.z = 0.0
        fixed_pose.pose.orientation.w = 1.0

        self.get_logger().info(
            f'[debug_fixed] Trying IK for fixed pose at '
            f'({fixed_pose.pose.position.x:.3f}, '
            f'{fixed_pose.pose.position.y:.3f}, '
            f'{fixed_pose.pose.position.z:.3f}) in base_link'
        )

        # ---- Build IK request IDENTICAL to ik_look_at_debug ----
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.ik_link_name
        req.ik_request.pose_stamped = fixed_pose

        # Seed with the current joint states
        req.ik_request.robot_state.joint_state = self.latest_joint_state

        # For debugging, turn off collision checking so we only test reachability
        req.ik_request.avoid_collisions = False

        # Same timeout as ik_look_at_debug
        req.ik_request.timeout = Duration(sec=0, nanosec=200_000_000)

        future = self.ik_client.call_async(req)
        future.add_done_callback(self.ik_response_cb)

    def ik_response_cb(self, future):
        if future.cancelled():
            self.get_logger().warn('[debug_fixed] IK call was cancelled')
            return
        if future.exception() is not None:
            self.get_logger().error(
                f'[debug_fixed] IK call failed: {future.exception()}')
            return

        resp = future.result()
        if resp.error_code.val != resp.error_code.SUCCESS:
            self.get_logger().warn(
                f'[debug_fixed] IK failed, error_code = {resp.error_code.val}')
            return

        js = resp.solution.joint_state
        name_to_pos = {n: p for n, p in zip(js.name, js.position)}
        self.get_logger().info('✅ [debug_fixed] IK SUCCESS:')
        for name in sorted(name_to_pos.keys()):
            self.get_logger().info(f'    {name}: {name_to_pos[name]:.3f}')


def main(args=None):
    rclpy.init(args=args)
    node = IkLookAtDebugFixed()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()



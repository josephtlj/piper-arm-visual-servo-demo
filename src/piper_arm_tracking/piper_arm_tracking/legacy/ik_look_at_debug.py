#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK

import tf2_ros


class IKLookAtDebug(Node):
    def __init__(self):
        super().__init__('ik_look_at_debug')
        self.get_logger().info("✅ ik_look_at_debug node started")

        # Frames
        self.base_frame = 'base_link'
        self.eef_frame = 'gripper_base'  # tip link used in MoveIt

        # TF buffer + listener (to get current EEF orientation)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Latest robot joint state (for seeding IK)
        self.current_joint_state = None

        # Latest ROI pose (in base_link)
        self.latest_roi_pose = None

        # Subscribe to ROI centroid in base_link
        self.create_subscription(
            PoseStamped,
            '/roi_centroid_pose',
            self.roi_pose_callback,
            10,
        )

        # Subscribe to /joint_states
        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10,
        )

        # IK service client
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.wait_for_ik_service()

        # Timer: try IK every 2 seconds (0.5 Hz)
        self.create_timer(2.0, self.timer_callback)

    def wait_for_ik_service(self):
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for /compute_ik service...')
        self.get_logger().info('✅ Connected to /compute_ik service')

    def joint_state_callback(self, msg: JointState):
        self.current_joint_state = msg

    def roi_pose_callback(self, msg: PoseStamped):
        #  cache latest ROI pose (already in base_link)
        self.latest_roi_pose = msg

    def timer_callback(self):
        if self.current_joint_state is None:
            self.get_logger().warn('No /joint_states yet, cannot seed IK')
            return

        if self.latest_roi_pose is None:
            self.get_logger().warn('No /roi_centroid_pose yet, cannot compute IK')
            return

        # Get current EEF orientation from TF: base_link -> gripper_base
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.eef_frame,
                rclpy.time.Time()
            )
            current_q = tf.transform.rotation
        except Exception as e:
            self.get_logger().warn(f"TF lookup for EEF orientation failed: {e}")
            return

        # Use latest ROI pose as target direction
        msg = self.latest_roi_pose
        target = msg.pose

        px = target.position.x
        py = target.position.y
        pz = target.position.z

        dist = math.sqrt(px * px + py * py + pz * pz)
        if dist < 1e-3:
            self.get_logger().warn('ROI too close to base_link origin, skipping')
            return

        # Keep 10 cm safety distance along the line base_link -> target
        safety_margin = 0.10
        if dist <= safety_margin:
            scale = safety_margin / dist
        else:
            scale = (dist - safety_margin) / dist

        safe_px = px * scale
        safe_py = py * scale
        safe_pz = pz * scale

        # Build safe pose: position near the ROI, orientation = current EEF
        safe_pose = PoseStamped()
        safe_pose.header.frame_id = self.base_frame
        safe_pose.header.stamp = self.get_clock().now().to_msg()
        safe_pose.pose.position.x = safe_px
        safe_pose.pose.position.y = safe_py
        safe_pose.pose.position.z = safe_pz
        safe_pose.pose.orientation = current_q

        self.get_logger().info(
            f"Trying IK for safe pose at "
            f"({safe_px:.3f}, {safe_py:.3f}, {safe_pz:.3f}) in {self.base_frame}"
        )

        req = GetPositionIK.Request()
        req.ik_request.group_name = 'arm'
        # Let MoveIt use the default tip, but asssume gripper_base is correct:
        req.ik_request.ik_link_name = self.eef_frame
        req.ik_request.pose_stamped = safe_pose
        req.ik_request.avoid_collisions = True
        req.ik_request.robot_state.joint_state = self.current_joint_state
        req.ik_request.timeout.sec = 0
        req.ik_request.timeout.nanosec = 200_000_000  # 0.2s

        future = self.ik_client.call_async(req)
        future.add_done_callback(self.ik_response_callback)

    def ik_response_callback(self, future):
        if future.result() is None:
            self.get_logger().error('IK service call failed (no result)')
            return

        res = future.result()
        ec = res.error_code.val

        if ec == 1:
            js = res.solution.joint_state
            self.get_logger().info("✅ IK SUCCESS:")
            for name, pos in zip(js.name, js.position):
                self.get_logger().info(f"    {name}: {pos:.3f}")
        else:
            self.get_logger().warn(f"IK failed, error_code = {ec}")


def main(args=None):
    rclpy.init(args=args)
    node = IKLookAtDebug()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()




#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, TwistStamped


class RoiServoController(Node):
    """
    ROI -> Twist node for MoveIt Servo.

    - Listens to /roi_centroid_pose (PoseStamped in base_link)
    - Computes a 20cm stand-off point along the ray from base_link to the centroid
    - Publishes a TwistStamped on /servo_twist_cmds that MoveIt Servo uses
      to drive the end-effector smoothly toward that stand-off point.
    """

    def __init__(self):
        super().__init__("roi_servo_controller")

        # Parameters you can tune
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("stand_off", 0.20)         # 20 cm from object
        self.declare_parameter("min_radius", 0.25)        # never closer than this to base
        self.declare_parameter("max_radius", 0.70)        # max reach used for scaling
        self.declare_parameter("kp_linear", 1.0)          # proportional gain for linear error
        self.declare_parameter("max_lin_speed", 0.2)      # m/s cap
        self.declare_parameter("loop_rate", 50.0)         # Hz

        self.target_frame = (
            self.get_parameter("target_frame").get_parameter_value().string_value
        )
        self.stand_off = (
            self.get_parameter("stand_off").get_parameter_value().double_value
        )
        self.min_radius = (
            self.get_parameter("min_radius").get_parameter_value().double_value
        )
        self.max_radius = (
            self.get_parameter("max_radius").get_parameter_value().double_value
        )
        self.kp_linear = (
            self.get_parameter("kp_linear").get_parameter_value().double_value
        )
        self.max_lin_speed = (
            self.get_parameter("max_lin_speed").get_parameter_value().double_value
        )
        loop_rate = (
            self.get_parameter("loop_rate").get_parameter_value().double_value
        )
        if loop_rate <= 0.0:
            loop_rate = 50.0
        self.dt = 1.0 / loop_rate

        self.latest_roi_pose: PoseStamped | None = None

        # Subscribed centroid in base_link
        self.create_subscription(
            PoseStamped,
            "/roi_centroid_pose",
            self.roi_pose_cb,
            10,
        )

        # Twist commands for Servo
        self.twist_pub = self.create_publisher(
            TwistStamped,
            "servo_twist_cmds",
            10,
        )

        # Timer loop that publishes twist commands at fixed rate
        self.timer = self.create_timer(self.dt, self.timer_cb)

        self.get_logger().info(
            "roi_servo_controller started (20cm stand-off tracking toward ROI)."
        )

    # ==================== Callbacks ====================

    def roi_pose_cb(self, msg: PoseStamped):
        # expect pose in base_link; if not, just warn once
        if msg.header.frame_id != self.target_frame:
            self.get_logger().warn_once(
                f"/roi_centroid_pose frame_id is {msg.header.frame_id}, "
                f"expected {self.target_frame}. Check TF pipeline."
            )
        self.latest_roi_pose = msg

    def timer_cb(self):
        """
        At each tick:
        - If there is a ROI pose, compute a stand-off point along the ray
        - Compute a linear error from some "look origin" to that stand-off point
        - Publish a capped TwistStamped for MoveIt Servo
        """
        if self.latest_roi_pose is None:
            # No target yet: send zero twist (Servo will hold / decay)
            self.publish_zero_twist()
            return

        p = self.latest_roi_pose.pose.position
        dx, dy, dz = p.x, p.y, p.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist < 1e-3:
            # Degenerate: don't move
            self.publish_zero_twist()
            return

        # --- Compute stand-off point along the ray base_link -> centroid ---
        # Desired radius from base to stand-off point
        desired_radius = max(self.min_radius, min(dist - self.stand_off, self.max_radius))

        # If the object is already closer than stand_off + min_radius,
        # just clamp at min_radius.
        scale = desired_radius / dist
        sx = dx * scale
        sy = dy * scale
        sz = dz * scale

        # --- Compute linear error (simple PD: here just P) ---
        # Interpret the "look origin" as the base_link origin (0,0,0)
        # so the error is just the stand-off coordinates.
        ex, ey, ez = sx, sy, sz

        # Proportional control to generate a desired velocity vector
        vx = self.kp_linear * ex
        vy = self.kp_linear * ey
        vz = self.kp_linear * ez

        # Cap linear speed to max_lin_speed
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        if speed > self.max_lin_speed:
            scale_v = self.max_lin_speed / speed
            vx *= scale_v
            vy *= scale_v
            vz *= scale_v

        # --- Build and publish TwistStamped (linear only for now) ---
        twist = TwistStamped()
        twist.header.frame_id = self.target_frame
        twist.header.stamp = self.get_clock().now().to_msg()

        twist.twist.linear.x = vx
        twist.twist.linear.y = vy
        twist.twist.linear.z = vz

        # For now, no rotational command; Servo can still use orientation constraints
        twist.twist.angular.x = 0.0
        twist.twist.angular.y = 0.0
        twist.twist.angular.z = 0.0

        self.twist_pub.publish(twist)

    def publish_zero_twist(self):
        twist = TwistStamped()
        twist.header.frame_id = self.target_frame
        twist.header.stamp = self.get_clock().now().to_msg()
        # all zeros
        self.twist_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = RoiServoController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()




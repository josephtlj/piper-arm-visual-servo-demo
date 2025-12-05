#!/usr/bin/env python3
import math
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, TwistStamped
import tf2_ros
from tf_transformations import euler_from_quaternion


class WheelMoveItServo(Node):
    """
    Visual-servo controller that has:
        - Linear motion: moves the EEF toward the wheel in base_link
        - Orientation: levels the gripper and yaws the arm to face the wheel
        - Distance gating: keeps distance within a band [min_tracking_dist, max_tracking_dist]
    """

    def __init__(self):
        super().__init__("wheel_moveit_servo")

        # Frames
        self.base_frame = self.declare_parameter(
            "base_frame", "base_link"
        ).get_parameter_value().string_value

        self.eef_frame = self.declare_parameter(
            "eef_frame", "gripper_base"
        ).get_parameter_value().string_value

        # Target and Servo topics
        self.target_topic = self.declare_parameter(
            "target_topic", "/wheel_target_pose"
        ).get_parameter_value().string_value

        self.servo_twist_topic = self.declare_parameter(
            "servo_twist_topic", "/servo_server/delta_twist_cmds"
        ).get_parameter_value().string_value

        # Distance band and standoff (all in meters)
        self.min_tracking_dist = self.declare_parameter(
            "min_tracking_dist", 0.35
        ).get_parameter_value().double_value
        self.max_tracking_dist = self.declare_parameter(
            "max_tracking_dist", 0.60
        ).get_parameter_value().double_value
        self.desired_standoff = self.declare_parameter(
            "desired_standoff", 0.45   # middle of the band
        ).get_parameter_value().double_value

        # Max linear speed (m/s)
        self.max_linear_speed = self.declare_parameter(
            "max_linear_speed", 0.50
        ).get_parameter_value().double_value

        # Orientation targets: "level" gripper in base_link
        # If a visually level pose has non-zero roll/pitch, roll_goal and pitch_goal can be adjusted.
        self.roll_goal = self.declare_parameter(
            "roll_goal", 0.0
        ).get_parameter_value().double_value
        self.pitch_goal = self.declare_parameter(
            "pitch_goal", 1.557
        ).get_parameter_value().double_value

        # Angular gains for leveling & yaw
        self.k_roll = self.declare_parameter(
            "k_roll", 0.3
        ).get_parameter_value().double_value
        self.k_pitch = self.declare_parameter(
            "k_pitch", 0.5
        ).get_parameter_value().double_value
        self.k_yaw = self.declare_parameter(
            "k_yaw", 0.3
        ).get_parameter_value().double_value

        # Detection timeout
        self.target_timeout = Duration(seconds=0.2)

        self.latest_target: Optional[PoseStamped] = None

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=2.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Sub / Pub
        self.target_sub = self.create_subscription(
            PoseStamped, self.target_topic, self.target_callback, 10
        )
        self.servo_pub = self.create_publisher(
            TwistStamped, self.servo_twist_topic, 10
        )

        # 50 Hz control loop
        self.timer = self.create_timer(0.02, self.control_loop)

        self.get_logger().info(
            f"WheelMoveItServo started. "
            f"Subscribing to {self.target_topic}, publishing Twist to {self.servo_twist_topic}."
        )

    # --------------------- Callbacks ---------------------

    def target_callback(self, msg: PoseStamped):
        self.latest_target = msg

    def control_loop(self):
        now = self.get_clock().now()

        if self.latest_target is None:
            return

        # Handle stale timestamps
        target_stamp = self.latest_target.header.stamp
        target_time = Time.from_msg(target_stamp)
        if target_stamp.sec == 0 and target_stamp.nanosec == 0:
            target_time = now
        if now - target_time > self.target_timeout:
            return

        # TF: base_link -> gripper_base
        try:
            tf_eef = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.eef_frame,
                Time(),
                timeout=Duration(seconds=0.1),
            )
        except Exception as ex:
            self.get_logger().warning(
                f"TF lookup failed for {self.base_frame}->{self.eef_frame}: {ex}"
            )
            return

        # Current EEF position
        eef_pos = np.array(
            [
                tf_eef.transform.translation.x,
                tf_eef.transform.translation.y,
                tf_eef.transform.translation.z,
            ]
        )

        # Target position in base_link
        target_pos = np.array(
            [
                self.latest_target.pose.position.x,
                self.latest_target.pose.position.y,
                self.latest_target.pose.position.z,
            ]
        )

        # ---------- Distance band + standoff logic ----------

        error_vec = target_pos - eef_pos
        dist_to_target = float(np.linalg.norm(error_vec))

        if dist_to_target < 1e-6:
            return

        # Direction from EEF to wheel
        z_axis = error_vec / dist_to_target

        # Linear command initialization
        v_cmd = np.zeros(3, dtype=float)

        if dist_to_target < self.min_tracking_dist:
            # Too close: back off directly along the line EEF <- wheel
            v_cmd = -self.max_linear_speed * z_axis
            mode = "back_off"
        elif dist_to_target > self.max_tracking_dist:
            # Too far: move forward along the line EEF -> wheel
            v_cmd = self.max_linear_speed * z_axis
            mode = "far_approach"
        else:
            # Within band: move toward desired standoff position
            desired_pos = target_pos - self.desired_standoff * z_axis
            pos_error = desired_pos - eef_pos
            err_norm = float(np.linalg.norm(pos_error))
            if err_norm > 1e-6:
                v_cmd = (pos_error / err_norm) * min(
                    self.max_linear_speed, err_norm / 0.1
                )
            mode = "standoff"

        # ---------- Orientation control (level + yaw) ----------

        rot = tf_eef.transform.rotation
        q = [rot.x, rot.y, rot.z, rot.w]
        roll, pitch, yaw = euler_from_quaternion(q)

        # Target orientation in base_link
        roll_err = self.roll_goal - roll
        pitch_err = self.pitch_goal - pitch

        # Yaw goal: face the wheel (project position error onto x-y plane)
        target_yaw = math.atan2(error_vec[1], error_vec[0])
        yaw_err = self._wrap_to_pi(target_yaw - yaw)

        # Angular velocities
        wx = self.k_pitch * pitch_err
        wy = self.k_roll * roll_err
        wz = self.k_yaw * yaw_err

        # Clamp angular speeds
        max_w = 1.0  # rad/s
        wx = float(np.clip(wx, -max_w, max_w))
        wy = float(np.clip(wy, -max_w, max_w))
        wz = float(np.clip(wz, -max_w, max_w))

        # ---------- Publish Twist ----------

        twist_msg = TwistStamped()
        twist_msg.header.stamp = now.to_msg()
        twist_msg.header.frame_id = self.base_frame

        twist_msg.twist.linear.x = float(v_cmd[0])
        twist_msg.twist.linear.y = float(v_cmd[1])
        twist_msg.twist.linear.z = float(v_cmd[2])

        twist_msg.twist.angular.x = wx
        twist_msg.twist.angular.y = wy
        twist_msg.twist.angular.z = wz

        self.servo_pub.publish(twist_msg)

        self.get_logger().info(
            f"Servo cmd [{mode}]: dist={dist_to_target:.3f} m, "
            f"v=({v_cmd[0]:.3f},{v_cmd[1]:.3f},{v_cmd[2]:.3f}) m/s, "
            f"rpy=({roll:.2f},{pitch:.2f},{yaw:.2f}), "
            f"w=({wx:.2f},{wy:.2f},{wz:.2f}) rad/s"
        )

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        a = (angle + math.pi) % (2.0 * math.pi) - math.pi
        return a


def main(args=None):
    rclpy.init(args=args)
    node = WheelMoveItServo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration

import tf2_ros
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point


class RoiCentroidNode(Node):
    def __init__(self):
        super().__init__("pointcloud_roi_centroid")

        # ------------ Parameters ------------
        # Depth / ROI filters (camera frame)
        self.declare_parameter("min_z", 0.10)          # m
        self.declare_parameter("max_z", 1.50)          # m
        self.declare_parameter("max_xy", 0.30)         # |x|, |y| window in meters
        self.declare_parameter("max_points", 5000)     # max ROI points to sample
        self.declare_parameter("min_points", 100)      # require at least this many ROI points

        # Smoothing / stability in camera frame
        self.declare_parameter("smoothing_alpha", 0.3)  # EMA factor (0–1); higher = follow faster
        self.declare_parameter("max_jump", 0.10)        # max jump allowed between centroids (m)

        # Extra smoothing / stability in base_link
        self.declare_parameter("base_smoothing_alpha", 0.5)  # EMA in base_link
        self.declare_parameter("base_max_jump", 0.08)        # max jump in base_link (m)

        # *** Table safety: minimum allowed z in base_link ***
        # a soft floor: clamp centroid z up to at least this height
        # so the arm will not be commanded too close to the table.
        self.declare_parameter("min_base_z", 0.08)  # m above base_link

        # Read parameters
        self.min_z = self.get_parameter("min_z").get_parameter_value().double_value
        self.max_z = self.get_parameter("max_z").get_parameter_value().double_value
        self.max_xy = self.get_parameter("max_xy").get_parameter_value().double_value
        self.max_points = self.get_parameter("max_points").get_parameter_value().integer_value
        self.min_points = self.get_parameter("min_points").get_parameter_value().integer_value

        self.smoothing_alpha = (
            self.get_parameter("smoothing_alpha").get_parameter_value().double_value
        )
        self.max_jump = self.get_parameter("max_jump").get_parameter_value().double_value

        self.base_smoothing_alpha = (
            self.get_parameter("base_smoothing_alpha").get_parameter_value().double_value
        )
        self.base_max_jump = (
            self.get_parameter("base_max_jump").get_parameter_value().double_value
        )

        self.min_base_z = (
            self.get_parameter("min_base_z").get_parameter_value().double_value
        )

        # Last filtered centroid in camera frame and base_link
        self.last_centroid_cam = None   # (x, y, z) in camera frame
        self.last_centroid_base = None  # (x, y, z) in base_link

        # ------------ Subscriptions ------------
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            "/camera/camera/depth/color/points",
            self.cloud_callback,
            10,
        )

        # ------------ Publishers ------------
        self.point_pub = self.create_publisher(PointStamped, "/roi_centroid_point", 10)
        self.marker_pub = self.create_publisher(Marker, "/roi_centroid_marker", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "/roi_centroid_pose", 10)

        # ------------ TF buffer / listener ------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info("ROI centroid node started (with narrow table safety clamp).")

    # =====================================================================
    # PointCloud callback
    # =====================================================================
    def cloud_callback(self, cloud: PointCloud2):
        """
        1. Sample a simple ROI from the pointcloud in camera frame.
        2. Compute centroid in camera frame.
        3. Stabilise in camera frame (EMA + jump limit).
        4. Publish:
           - /roi_centroid_point (camera frame)
           - /roi_centroid_marker (camera frame)
        5. Transform to base_link and stabilise again (EMA + jump limit + min_base_z clamp).
           - Publish /roi_centroid_pose (base_link)
        """
        points = []

        try:
            for x, y, z in point_cloud2.read_points(
                cloud, field_names=("x", "y", "z"), skip_nans=True
            ):
                # Simple ROI box in front of camera
                if (
                    self.min_z < z < self.max_z
                    and abs(x) < self.max_xy
                    and abs(y) < self.max_xy
                ):
                    points.append((x, y, z))
                    if len(points) >= self.max_points:
                        break
        except Exception as e:
            self.get_logger().warn(f"Error reading pointcloud: {e}")
            return

        if len(points) < self.min_points:
            self.get_logger().debug(
                f"ROI had only {len(points)} points (<{self.min_points}), skipping update."
            )
            return

        # 2) Raw centroid in camera frame
        n = len(points)
        cx_raw = sum(p[0] for p in points) / n
        cy_raw = sum(p[1] for p in points) / n
        cz_raw = sum(p[2] for p in points) / n

        # 3) Stabilise centroid in camera frame
        cx, cy, cz = cx_raw, cy_raw, cz_raw

        if self.last_centroid_cam is not None:
            lx, ly, lz = self.last_centroid_cam
            dx = cx_raw - lx
            dy = cy_raw - ly
            dz = cz_raw - lz
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            # Hard limit on jump size
            if dist > 1e-6 and dist > self.max_jump:
                scale = self.max_jump / dist
                cx = lx + dx * scale
                cy = ly + dy * scale
                cz = lz + dz * scale

            # Exponential moving average (smooth)
            alpha = self.smoothing_alpha
            cx = alpha * cx + (1.0 - alpha) * lx
            cy = alpha * cy + (1.0 - alpha) * ly
            cz = alpha * cz + (1.0 - alpha) * lz

        self.last_centroid_cam = (cx, cy, cz)

        # 4) Publish centroid in camera frame as PointStamped
        centroid_cam = PointStamped()
        centroid_cam.header = cloud.header
        centroid_cam.point.x = cx
        centroid_cam.point.y = cy
        centroid_cam.point.z = cz

        self.point_pub.publish(centroid_cam)

        # 5) Marker in camera frame
        marker = Marker()
        marker.header = cloud.header
        marker.ns = "roi_centroid"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = cx
        marker.pose.position.y = cy
        marker.pose.position.z = cz
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.lifetime = Duration(sec=0, nanosec=0)
        self.marker_pub.publish(marker)

        # 6) Transform centroid into base_link & apply base-frame smoothing + table safety
        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link",                   # target frame
                centroid_cam.header.frame_id,  # source frame
                rclpy.time.Time()              # latest
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        centroid_in_base = do_transform_point(centroid_cam, transform)

        bx_raw = centroid_in_base.point.x
        by_raw = centroid_in_base.point.y
        bz_raw = centroid_in_base.point.z

        # --- TABLE SAFETY CLAMP ---
        # Do NOT allow the target to go below min_base_z in base_link.
        # A "narrow" table filter: don't skip the centroid,
        # just floor the z so the arm stays slightly above the table.
        if bz_raw < self.min_base_z:
            bz_raw = self.min_base_z

        bx, by, bz = bx_raw, by_raw, bz_raw

        if self.last_centroid_base is not None:
            lbx, lby, lbz = self.last_centroid_base
            dx_b = bx_raw - lbx
            dy_b = by_raw - lby
            dz_b = bz_raw - lbz
            dist_b = math.sqrt(dx_b * dx_b + dy_b * dy_b + dz_b * dz_b)

            # Limit sudden huge jumps in base_link
            if dist_b > 1e-6 and dist_b > self.base_max_jump:
                scale_b = self.base_max_jump / dist_b
                bx = lbx + dx_b * scale_b
                by = lby + dy_b * scale_b
                bz = lbz + dz_b * scale_b

            # EMA in base_link
            alpha_b = self.base_smoothing_alpha
            bx = alpha_b * bx + (1.0 - alpha_b) * lbx
            by = alpha_b * by + (1.0 - alpha_b) * lby
            bz = alpha_b * bz + (1.0 - alpha_b) * lbz

        self.last_centroid_base = (bx, by, bz)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = centroid_cam.header.stamp
        pose_msg.header.frame_id = "base_link"
        pose_msg.pose.position.x = bx
        pose_msg.pose.position.y = by
        pose_msg.pose.position.z = bz
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0

        self.pose_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RoiCentroidNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()



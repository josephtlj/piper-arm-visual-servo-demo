#!/usr/bin/env python3
import time
from typing import Optional

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration

from cv_bridge import CvBridge
from ultralytics import YOLO

import numpy as np
import message_filters

import tf2_ros
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point


class WheelYoloDetector(Node):
    """
    YOLO-based detector for the 3D-printed aircraft wheel, using YOLO's 'clock' class.

    Pipeline:
      - Subscribe to:
          /camera/camera/color/image_raw
          /camera/camera/depth/image_rect_raw
          /camera/camera/color/camera_info
      - Run YOLO on the color image.
      - Find the best 'clock' detection.
      - Use the depth image + intrinsics to back-project the bbox center to 3D in the camera frame.
      - Transform that 3D point to base_link.
      - Publish:
          /wheel_target_point  (PointStamped in camera frame)
          /wheel_target_pose   (PoseStamped in base_link)
          /wheel_target_marker (Marker sphere in camera frame)
    """

    def __init__(self):
        super().__init__("wheel_yolo_detector")

        # ---- Parameters ----
        self.declare_parameter(
            "color_topic", "/camera/camera/color/image_raw"
        )
        self.declare_parameter(
            "depth_topic", "/camera/camera/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter(
            "camera_info_topic", "/camera/camera/color/camera_info"
        )
        self.declare_parameter("confidence_threshold", 0.4)
        self.declare_parameter("max_depth_m", 1.5)  # ignore deeper than this
        self.declare_parameter("debug_log_every", 1.0)  # seconds

        self.color_topic = (
            self.get_parameter("color_topic").get_parameter_value().string_value
        )
        self.depth_topic = (
            self.get_parameter("depth_topic").get_parameter_value().string_value
        )
        self.camera_info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        self.conf_thresh = (
            self.get_parameter("confidence_threshold")
            .get_parameter_value()
            .double_value
        )
        self.max_depth_m = (
            self.get_parameter("max_depth_m").get_parameter_value().double_value
        )
        self.debug_log_every = (
            self.get_parameter("debug_log_every").get_parameter_value().double_value
        )

        # ---- CV / YOLO ----
        self.bridge = CvBridge()
        self.get_logger().info(f"Subscribing to image topic: {self.color_topic}")
        self.get_logger().info("Loading YOLO model (yolov8n.pt)...")
        t0 = time.time()
        self.model = YOLO("yolov8n.pt")
        t1 = time.time()
        self.get_logger().info(
            f"YOLO model loaded in {t1 - t0:.2f} seconds. Classes: {self.model.names}"
        )

        # Treat YOLO class name "clock" as "wheel"
        self.target_class_name = "clock"

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- Publishers ----
        self.point_pub = self.create_publisher(
            PointStamped, "/wheel_target_point", 10
        )
        self.pose_pub = self.create_publisher(
            PoseStamped, "/wheel_target_pose", 10
        )
        self.marker_pub = self.create_publisher(
            Marker, "/wheel_target_marker", 10
        )

        # ---- Subscribers (with sync) ----
        color_sub = message_filters.Subscriber(self, Image, self.color_topic)
        depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)
        info_sub = message_filters.Subscriber(self, CameraInfo, self.camera_info_topic)

        # Approximate sync: color + depth + camera_info
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, info_sub],
            queue_size=10,
            slop=0.1,
        )
        self.sync.registerCallback(self.synced_callback)

        self.last_debug_log = 0.0

        self.get_logger().info("WheelYoloDetector node started.")

    # ======================================================================
    # Main synced callback: color + depth + camera_info
    # ======================================================================
    def synced_callback(
        self,
        color_msg: Image,
        depth_msg: Image,
        cam_info: CameraInfo,
    ):
        # 1) Run YOLO on color image
        try:
            cv_img = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")
            return

        results = self.model(cv_img, verbose=False)

        if not results or len(results) == 0:
            return

        det = results[0]
        if det.boxes is None or len(det.boxes) == 0:
            return

        # Find best 'clock' detection above confidence threshold
        best_conf = 0.0
        best_box = None

        for box in det.boxes:
            cls_idx = int(box.cls[0].item())
            cls_name = self.model.names.get(cls_idx, str(cls_idx))
            conf = float(box.conf[0].item())

            if cls_name != self.target_class_name:
                continue
            if conf < self.conf_thresh:
                continue

            if conf > best_conf:
                best_conf = conf
                best_box = box

        if best_box is None:
            # No suitable 'clock' (wheel) detection
            return

        xyxy = best_box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        u = 0.5 * (x1 + x2)
        v = 0.5 * (y1 + y2)

        now = time.time()
        if now - self.last_debug_log > self.debug_log_every:
            self.get_logger().info(
                f"Detected wheel (YOLO 'clock') conf={best_conf:.2f} "
                f"bbox=({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
            )
            self.last_debug_log = now

        # 2) Get depth at bbox center (small window, median)
        depth_m = self._depth_at_pixel(depth_msg, u, v)
        if depth_m is None:
            # Unable to get valid depth here
            return

        if depth_m > self.max_depth_m:
            # Too far, ignore
            return

        # 3) Back-project to 3D in camera frame
        X_cam, Y_cam, Z_cam = self._pixel_to_cam_3d(u, v, depth_m, cam_info)
        cam_frame = cam_info.header.frame_id or color_msg.header.frame_id

        point_cam = PointStamped()
        point_cam.header.stamp = color_msg.header.stamp
        point_cam.header.frame_id = cam_frame
        point_cam.point.x = X_cam
        point_cam.point.y = Y_cam
        point_cam.point.z = Z_cam

        self.point_pub.publish(point_cam)

        # Publish marker in camera frame
        marker = Marker()
        marker.header.stamp = point_cam.header.stamp
        marker.header.frame_id = cam_frame
        marker.ns = "wheel_target"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = X_cam
        marker.pose.position.y = Y_cam
        marker.pose.position.z = Z_cam
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.lifetime = Duration(sec=0, nanosec=0)
        self.marker_pub.publish(marker)

        # 4) Transform to base_link and publish PoseStamped
        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link",
                point_cam.header.frame_id,
                rclpy.time.Time(),
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        point_base = do_transform_point(point_cam, transform)

        pose = PoseStamped()
        pose.header.stamp = point_cam.header.stamp
        pose.header.frame_id = "base_link"
        pose.pose.position.x = point_base.point.x
        pose.pose.position.y = point_base.point.y
        pose.pose.position.z = point_base.point.z
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        self.pose_pub.publish(pose)

    # ======================================================================
    # Helpers
    # ======================================================================
    def _depth_at_pixel(self, depth_msg: Image, u: float, v: float) -> Optional[float]:
        """Return median depth (meters) in a small window around (u, v)."""
        try:
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge depth error: {e}")
            return None

        h, w = depth_img.shape[:2]
        u_i = int(round(u))
        v_i = int(round(v))

        if u_i < 0 or u_i >= w or v_i < 0 or v_i >= h:
            return None

        half_win = 2  # 5x5 window
        u_min = max(0, u_i - half_win)
        u_max = min(w - 1, u_i + half_win)
        v_min = max(0, v_i - half_win)
        v_max = min(h - 1, v_i + half_win)

        window = depth_img[v_min : v_max + 1, u_min : u_max + 1]

        # Handle 16UC1 (mm) or 32FC1 (m)
        if depth_msg.encoding == "16UC1":
            # zero or 0 means invalid
            valid = window.astype(np.float32)
            valid[valid <= 0.0] = np.nan
            depth_values_m = valid * 0.001  # mm → m
        else:
            # assume meters
            depth_values_m = window.astype(np.float32)
            depth_values_m[depth_values_m <= 0.0] = np.nan

        valid_vals = depth_values_m[~np.isnan(depth_values_m)]
        if valid_vals.size == 0:
            return None

        depth_m = float(np.median(valid_vals))
        return depth_m

    def _pixel_to_cam_3d(
        self,
        u: float,
        v: float,
        depth_m: float,
        cam_info: CameraInfo,
    ):
        """Back-project pixel (u, v, depth_m) to 3D in camera frame using intrinsics."""
        K = cam_info.k  # 3x3 row-major
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]

        X = (u - cx) * depth_m / fx
        Y = (v - cy) * depth_m / fy
        Z = depth_m

        return X, Y, Z


def main(args=None):
    rclpy.init(args=args)
    node = WheelYoloDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()



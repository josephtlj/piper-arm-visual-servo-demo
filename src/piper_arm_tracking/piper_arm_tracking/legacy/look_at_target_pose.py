import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped, PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point


class LookAtTargetPose(Node):
    def __init__(self):
        super().__init__("look_at_target_pose")

        # TF buffer + listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Parameters to  tune
        self.base_frame = "base_link"
        self.camera_frame = "camera_link"
        self.standoff_distance = 0.20  # metres away from target

        # Subscribe to centroid (from ROI node)
        self.centroid_sub = self.create_subscription(
            PointStamped,
            "/target_centroid",   # change for different topic nam
            self.centroid_callback,
            10,
        )

        # Publish desired camera pose in base_link
        self.pose_pub = self.create_publisher(
            PoseStamped,
            "/desired_camera_pose",
            10,
        )

        self.get_logger().info("look_at_target_pose node started")

    def centroid_callback(self, msg: PointStamped):
        # 1) Transform centroid to base_link
        try:
            tf_cam_to_base = self.tf_buffer.lookup_transform(
                self.base_frame,          # target frame
                msg.header.frame_id,      # source frame (e.g. camera_depth_optical_frame)
                rclpy.time.Time(),        # latest available
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed (centroid -> base): {ex}")
            return

        try:
            centroid_in_base = do_transform_point(msg, tf_cam_to_base)
        except Exception as ex:
            self.get_logger().warn(f"Failed to transform centroid: {ex}")
            return

        Px = centroid_in_base.point.x
        Py = centroid_in_base.point.y
        Pz = centroid_in_base.point.z

        # 2) Get camera origin in base_link
        try:
            tf_base_to_cam = self.tf_buffer.lookup_transform(
                self.base_frame,    # target
                self.camera_frame,  # source
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed (camera -> base): {ex}")
            return

        Cx = tf_base_to_cam.transform.translation.x
        Cy = tf_base_to_cam.transform.translation.y
        Cz = tf_base_to_cam.transform.translation.z

        # 3) Direction vector from camera -> target
        dx = Px - Cx
        dy = Py - Cy
        dz = Pz - Cz
        length = math.sqrt(dx*dx + dy*dy + dz*dz)

        if length < 1e-3:
            self.get_logger().warn("Centroid too close to camera; skipping")
            return

        dx /= length
        dy /= length
        dz /= length

        # 4) Desired camera position: standoff behind target along ray
        d = self.standoff_distance
        desired_x = Px - d * dx
        desired_y = Py - d * dy
        desired_z = Pz - d * dz

        # 5) Build PoseStamped in base_link
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.base_frame
        pose.pose.position.x = desired_x
        pose.pose.position.y = desired_y
        pose.pose.position.z = desired_z

        # Keep current camera orientation (no "look-at" rotation yet)
        pose.pose.orientation = tf_base_to_cam.transform.rotation

        self.pose_pub.publish(pose)

        self.get_logger().info(
            f"Published desired camera pose at "
            f"({desired_x:.3f}, {desired_y:.3f}, {desired_z:.3f}) in {self.base_frame}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = LookAtTargetPose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()




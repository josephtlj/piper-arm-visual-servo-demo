#!/usr/bin/env python3
import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import xacro

def load_yaml(package_name, relative_path):
    package_share = get_package_share_directory(package_name)
    file_path = os.path.join(package_share, relative_path)
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def generate_launch_description():
    # 1. Load Robot Description (URDF)
    piper_moveit_share = get_package_share_directory("piper_with_gripper_moveit")
    urdf_xacro_path = os.path.join(piper_moveit_share, "config", "piper.urdf.xacro")
    doc = xacro.process_file(urdf_xacro_path)
    robot_description = {"robot_description": doc.toxml()}

    # 2. Load SRDF
    srdf_path = os.path.join(piper_moveit_share, "config", "piper.srdf")
    with open(srdf_path, "r") as f:
        robot_description_semantic = {"robot_description_semantic": f.read()}

    # 3. Load Servo Parameters
    # DIRECT LOAD: We do not wrap this in another dictionary key.
    # The keys in the YAML already contain "moveit_servo." prefix.
    servo_params = load_yaml("piper_arm_tracking", "config/piper_servo.yaml")

    # 4. Robot State Publisher
    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    # 5. Servo Node
    servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="servo_server",
        output="screen",
        # Pass the flattened dictionary directly
        parameters=[
            servo_params, 
            robot_description, 
            robot_description_semantic
        ],
    )

    return LaunchDescription([rsp_node, servo_node])



# Launches the Piper single-arm controller for integration with MoveIt,
# remapping its command topic(s) to /piper_cmd instead of /joint_states.
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

os.environ["RCUTILS_COLORIZED_OUTPUT"] = "1"

def generate_launch_description():
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level (debug, info, warn, error, fatal).'
    )

    can_port_arg = DeclareLaunchArgument(
        'can_port',
        default_value='can0',
        description='CAN port to be used by the Piper node.'
    )

    auto_enable_arg = DeclareLaunchArgument(
        'auto_enable',
        default_value='true',
        description='Automatically enable the Piper node.'
    )

    rviz_ctrl_flag_arg = DeclareLaunchArgument(
        'rviz_ctrl_flag',
        default_value='false',
        description='Start rviz flag.'
    )

    gripper_exist_arg = DeclareLaunchArgument(
        'gripper_exist',
        default_value='true',
        description='gripper'
    )

    gripper_val_mutiple_arg = DeclareLaunchArgument(
        'gripper_val_mutiple',
        default_value='1',
        description='gripper'
    )

    piper_node = Node(
        package='piper',
        executable='piper_single_ctrl',
        name='piper_ctrl_single_node',
        output='screen',
        ros_arguments=['--log-level', LaunchConfiguration('log_level')],
        parameters=[{
            'can_port': LaunchConfiguration('can_port'),
            'auto_enable': LaunchConfiguration('auto_enable'),
            'gripper_val_mutiple': LaunchConfiguration('gripper_val_mutiple'),
            'gripper_exist': LaunchConfiguration('gripper_exist'),
        }],
        remappings=[
            # Whatever the SDK used to send/receive as "joint_ctrl_single"
            # is now treated as /piper_cmd
            ('joint_ctrl_single', '/piper_cmd'),
            # And anything that was on /joint_states is also treated as /piper_cmd
            ('/joint_states', '/piper_cmd'),
        ],
    )

    return LaunchDescription([
        log_level_arg,
        can_port_arg,
        auto_enable_arg,
        rviz_ctrl_flag_arg,
        gripper_exist_arg,
        gripper_val_mutiple_arg,
        piper_node,
    ])



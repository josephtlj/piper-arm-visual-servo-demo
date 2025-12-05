from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'piper_arm_tracking'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Index
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        # package.xml
        (
            'share/' + package_name,
            ['package.xml']
        ),
        # install all launch files
        (
            os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))
        ),
        # install all config files
        (
            os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='smart-mit',
    maintainer_email='josephtanlj@gmail.com',
    description='Visual servoing pipeline for the Piper arm: YOLOv8 detection bridged into MoveIt Servo over CAN.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # Core pipeline
            'joint_state_bridge = piper_arm_tracking.joint_state_bridge:main',
            'follow_joint_trajectory_bridge_servo = piper_arm_tracking.follow_joint_trajectory_bridge_servo:main',
            'servo_prepose = piper_arm_tracking.servo_prepose:main',
            'wheel_yolo_detector = piper_arm_tracking.wheel_yolo_detector:main',
            'wheel_moveit_servo = piper_arm_tracking.wheel_moveit_servo:main',
            # Legacy / experimental (see piper_arm_tracking/legacy/)
            'testing_node = piper_arm_tracking.legacy.testing_node:main',
            'simple_motion_node = piper_arm_tracking.legacy.simple_motion_node:main',
            'follow_joint_trajectory_bridge = piper_arm_tracking.legacy.follow_joint_trajectory_bridge:main',
            'pointcloud_roi_centroid = piper_arm_tracking.legacy.pointcloud_roi_centroid:main',
            'look_at_target_pose = piper_arm_tracking.legacy.look_at_target_pose:main',
            'ik_look_at_debug = piper_arm_tracking.legacy.ik_look_at_debug:main',
            'ik_look_at_controller = piper_arm_tracking.legacy.ik_look_at_controller:main',
            'roi_moveit_plan_debug = piper_arm_tracking.legacy.roi_moveit_plan_debug:main',
            'ik_look_at_debug_fixed = piper_arm_tracking.legacy.ik_look_at_debug_fixed:main',
            'roi_moveit_controller = piper_arm_tracking.legacy.roi_moveit_controller:main',
            'wheel_moveit_controller = piper_arm_tracking.legacy.wheel_moveit_controller:main',
        ],
    },
)



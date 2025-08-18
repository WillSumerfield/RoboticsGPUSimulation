# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg


HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.path.dirname(__file__), "Grasp3D-temp.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 3.0), joint_pos={"joint_right": 0.0, "joint_left": 0.0, "joint_back": 0.0, "joint_slide_x": 0.0, "joint_slide_y": 0.0, "joint_slide_z": 2.0}
    ),
    actuators={
        "planar_actuator": ImplicitActuatorCfg(
            joint_names_expr=["joint_slide_x", "joint_slide_y", "joint_slide_z"],
            effort_limit_sim=400.0,
            velocity_limit_sim=50.0,
            stiffness=200.0,
            damping=20.0,
        ),
        "digit_actuator": ImplicitActuatorCfg(
            joint_names_expr=["joint_left", "joint_right", "joint_back"], 
            effort_limit_sim=400.0,
            velocity_limit_sim=50.0, 
            stiffness=200.0, 
            damping=20.0
        ),
    },
)

SENSOR_CFG = ContactSensorCfg(track_air_time=True)

OBJ_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Robot/Object",
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    ),
)


@configclass
class GraspRotateEnvCfg(DirectRLEnvCfg):
    # env
    n_timesteps = 1000
    decimation = 2
    episode_length_s = 5.0
    normalize_input = True
    # - spaces definition
    action_space = 4
    observation_space = 12
    state_space = 12

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1/60, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # sensors
    sensor_left_cfg: ContactSensorCfg = SENSOR_CFG.replace(prim_path="/World/envs/env_.*/Robot/robot/left_lower_digit")
    sensor_right_cfg: ContactSensorCfg = SENSOR_CFG.replace(prim_path="/World/envs/env_.*/Robot/robot/right_lower_digit")
    sensor_back_cfg: ContactSensorCfg = SENSOR_CFG.replace(prim_path="/World/envs/env_.*/Robot/robot/back_lower_digit")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=10.0, replicate_physics=True)

    # objects
    object_cfg: RigidObjectCfg = OBJ_CFG

    # custom parameters/scales
    # - robot body and joint names
    hand_body_name = "palm"
    object_body_name = "Object"
    floor_body_name = "floor"
    left_finger_joint_name = "joint_left"
    right_finger_joint_name = "joint_right"
    back_finger_joint_name = "joint_back"
    planar_x_joint_name = "joint_slide_x"
    planar_y_joint_name = "joint_slide_y"
    planar_z_joint_name = "joint_slide_z"
    # - action scales
    action_scale_digits = np.pi/4 # [rad]
    action_scale_horizontal = 1.5 # [m]
    action_scale_vertical = 1.5 # [m]
    # - reward scales
    rew_scale_distance = 1.0
    rew_scale_lifted = 5.0
    rew_scale_terminated = 0.0
    # - reset states/conditions
    initial_hand_position_range_horizontal = [-1.5, 1.5] # [m]
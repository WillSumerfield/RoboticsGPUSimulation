# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from collections.abc import Sequence

import numpy as np
import torch
torch.set_printoptions(precision=1, sci_mode=False)

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .grasp_objects_cfg import GraspObjectsEnvCfg


class GraspObjectsEnv(DirectRLEnv):
    CONST_DIST_MAG = np.sqrt(300.0)

    cfg: GraspObjectsEnvCfg

    def __init__(self, cfg: GraspObjectsEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        if self.viewport_camera_controller is not None:
            self.viewport_camera_controller.update_view_location([-20, 0, 20], [10, 0, 0])

        self._hand_idx, _ = self.robot.find_bodies(self.cfg.hand_body_name)
        self._planar_x_dof_idx = self.robot.find_joints(self.cfg.planar_x_joint_name)[0][0]
        self._planar_y_dof_idx = self.robot.find_joints(self.cfg.planar_y_joint_name)[0][0]
        self._planar_z_dof_idx = self.robot.find_joints(self.cfg.planar_z_joint_name)[0][0]
        self._left_finger_dof_idx = self.robot.find_joints(self.cfg.left_finger_joint_name)[0][0]
        self._right_finger_dof_idx = self.robot.find_joints(self.cfg.right_finger_joint_name)[0][0]
        self._back_finger_dof_idx = self.robot.find_joints(self.cfg.back_finger_joint_name)[0][0]

        self.body_pos = self.robot.data.body_com_pos_w
        self.object1_pos = self.object1.data.body_pos_w
        self.object2_pos = self.object2.data.body_pos_w
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.sensor_left = ContactSensor(self.cfg.sensor_left_cfg)
        self.sensors_right = ContactSensor(self.cfg.sensor_right_cfg)
        self.sensors_back = ContactSensor(self.cfg.sensor_back_cfg)
        self.object1 = RigidObject(self.cfg.object1_cfg)
        self.object2 = RigidObject(self.cfg.object2_cfg)
        self.floor = spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["sensor_left"] = self.sensor_left
        self.scene.sensors["sensor_right"] = self.sensors_right
        self.scene.sensors["sensor_back"] = self.sensors_back
        self.scene.rigid_objects["object1"] = self.object1
        self.scene.rigid_objects["object2"] = self.object2
        self.scene.extras["floor"] = self.floor

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions[:, 0]*-self.cfg.action_scale_digits, joint_ids=self._left_finger_dof_idx)
        self.robot.set_joint_position_target(self.actions[:, 0]*-self.cfg.action_scale_digits, joint_ids=self._right_finger_dof_idx)
        self.robot.set_joint_position_target(self.actions[:, 0]*-self.cfg.action_scale_digits, joint_ids=self._back_finger_dof_idx)
        self.robot.set_joint_position_target(self.actions[:, 1]*self.cfg.action_scale_vertical, joint_ids=self._planar_z_dof_idx)

    def _get_observations(self) -> dict:
        obs = compute_observations(
            self.scene.env_origins, 
            self.body_pos[:, self._hand_idx[0]],
            self.object1_pos[:, 0],
            self.object2_pos[:, 0],
            self.joint_pos[:, self._left_finger_dof_idx],
            self.joint_pos[:, self._right_finger_dof_idx],
            self.joint_pos[:, self._back_finger_dof_idx],
            self.sensor_left.data.current_contact_time,
            self.sensors_right.data.current_contact_time,
            self.sensors_back.data.current_contact_time,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.CONST_DIST_MAG,
            self.cfg.rew_scale_distance,
            self.cfg.rew_scale_lifted_single,
            self.cfg.rew_scale_lifted_double,
            self.body_pos[:, self._hand_idx[0]],
            self.object1_pos,
            self.object2_pos,
            self.sensor_left.data.current_contact_time[:, 0] > 0,
            self.sensors_right.data.current_contact_time[:, 0] > 0,
            self.sensors_back.data.current_contact_time[:, 0] > 0,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.body_pos = self.robot.data.body_com_pos_w
        self.object1_pos = self.object1.data.body_pos_w
        self.object2_pos = self.object2.data.body_pos_w
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = compute_boundaries(self.object1_pos[:, 0, :], self.object2_pos[:, 0, :], self.scene.env_origins)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, self._planar_x_dof_idx] += sample_uniform(
            self.cfg.initial_hand_position_range_horizontal[0],
            self.cfg.initial_hand_position_range_horizontal[1],
            joint_pos[:, self._planar_x_dof_idx].shape,
            joint_pos.device,
        )
        joint_pos[:, self._planar_y_dof_idx] += sample_uniform(
            self.cfg.initial_hand_position_range_horizontal[0],
            self.cfg.initial_hand_position_range_horizontal[1],
            joint_pos[:, self._planar_y_dof_idx].shape, 
            joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        robot_root_state = self.robot.data.default_root_state[env_ids]
        robot_root_state[:, :3] += self.scene.env_origins[env_ids]
        object1_root_state = self.object1.data.default_root_state[env_ids]
        object1_root_state[:, :3] += self.scene.env_origins[env_ids]
        object2_root_state = self.object2.data.default_root_state[env_ids]
        object2_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Randomize object orientation for domain randomization
        random_z_rot = torch.rand((len(env_ids), 2), device=object1_root_state.device)
        quat_w = torch.sqrt(1 - random_z_rot**2)
        object1_root_state[:, 3:7] = torch.cat((quat_w[:, 0].unsqueeze(dim=1), torch.zeros((len(env_ids), 2), device=object1_root_state.device), random_z_rot[:, 0].unsqueeze(dim=1)), dim=1)
        object2_root_state[:, 3:7] = torch.cat((quat_w[:, 1].unsqueeze(dim=1), torch.zeros((len(env_ids), 2), device=object2_root_state.device), random_z_rot[:, 1].unsqueeze(dim=1)), dim=1)

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(robot_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(robot_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.object1.write_root_pose_to_sim(object1_root_state[:, :7], env_ids)
        self.object1.write_root_velocity_to_sim(object1_root_state[:, 7:], env_ids)
        self.object2.write_root_pose_to_sim(object2_root_state[:, :7], env_ids)
        self.object2.write_root_velocity_to_sim(object2_root_state[:, 7:], env_ids)


@torch.jit.script
def compute_observations(
    env_offsets: torch.Tensor, 
    body_pos: torch.Tensor, 
    object1_pos: torch.Tensor,
    object2_pos: torch.Tensor,
    joint_pos_l: torch.Tensor,
    joint_pos_r: torch.Tensor,
    joint_pos_b: torch.Tensor,
    contact_left: torch.Tensor,
    contact_right: torch.Tensor,
    contact_back: torch.Tensor,
) -> torch.Tensor:
    hand_pos = body_pos - env_offsets
    obj1_rel_pos = body_pos - object1_pos # relative to position of the hand
    obj2_rel_pos = body_pos - object2_pos # relative to position of the hand
    obs = torch.cat(
        (
            hand_pos,
            obj1_rel_pos,
            obj2_rel_pos,
            joint_pos_l.unsqueeze(dim=1),
            joint_pos_r.unsqueeze(dim=1),
            joint_pos_b.unsqueeze(dim=1),
            contact_left > 0,
            contact_right > 0,
            contact_back > 0,
        ),
        dim=-1,
    )
    return obs


#@torch.jit.script
def compute_rewards(
    CONST_DIST_MAG: float,
    rew_scale_distance: float,
    rew_scale_lifted_single: float,
    rew_scale_lifted_double: float,
    hand_pos: torch.Tensor,
    object1_pos: torch.Tensor,
    object2_pos: torch.Tensor,
    contact_left: torch.Tensor,
    contact_right: torch.Tensor,
    contact_back: torch.Tensor,
) -> torch.Tensor:
    obj1_lifted = object1_pos[:, 0, 2] > 1
    obj2_lifted = object2_pos[:, 0, 2] > 1
    object1_dist = torch.norm(hand_pos - object1_pos[:, 0], p=2, dim=-1)/CONST_DIST_MAG
    object2_dist = torch.norm(hand_pos - object2_pos[:, 0], p=2, dim=-1)/CONST_DIST_MAG
    sum_contact = contact_left.float() + contact_right.float() + contact_back.float()
    contact = sum_contact > 0.0
    double_contact = sum_contact > 1.0
    single_lifted = (obj1_lifted ^ obj2_lifted).float() * rew_scale_lifted_single
    double_lifted = (obj1_lifted.float() * obj2_lifted.float() * double_contact.float()) * rew_scale_lifted_double
    reward = (2-object1_dist-object2_dist)**rew_scale_distance + single_lifted + double_lifted
    return reward


@torch.jit.script
def compute_boundaries(
    object1_pos: torch.Tensor,
    object2_pos: torch.Tensor,
    start_pos: torch.Tensor
) -> torch.Tensor:
    horizontal1_bound = torch.any(torch.abs(object1_pos[:, :2] - start_pos[:, :2]) > 1.5, dim=1)
    horizontal2_bound = torch.any(torch.abs(object2_pos[:, :2] - start_pos[:, :2]) > 1.5, dim=1)
    vertical1_bound = torch.abs(object1_pos[:, 2] - start_pos[:, 2]) > 5.0
    vertical2_bound = torch.abs(object2_pos[:, 2] - start_pos[:, 2]) > 5.0
    return (horizontal1_bound | vertical1_bound) & (horizontal2_bound | vertical2_bound)
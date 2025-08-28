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
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .grasp_object_cfg import GraspObjectEnvCfg


class GraspObjectEnv(DirectRLEnv):
    cfg: GraspObjectEnvCfg

    def __init__(self, cfg: GraspObjectEnvCfg, render_mode: str | None = None, **kwargs):
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
        self.object_pos = self.object.data.body_pos_w
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self.wind_direction = torch.zeros((self.cfg.scene.num_envs, 3), device=self.device)
        self.wind_time = torch.zeros((self.cfg.scene.num_envs,), dtype=torch.bool, device=self.device)
        self.held_position = torch.zeros((self.cfg.scene.num_envs, 3), device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.sensor_left = ContactSensor(self.cfg.sensor_left_cfg)
        self.sensors_right = ContactSensor(self.cfg.sensor_right_cfg)
        self.sensors_back = ContactSensor(self.cfg.sensor_back_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
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
        self.scene.rigid_objects["object"] = self.object
        self.scene.extras["floor"] = self.floor

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

        # Enable wind forces if the wind time is reached
        new_wind_time = self.episode_length_buf > self.cfg.wind_time
        env_ids = self.wind_time.nonzero(as_tuple=False).squeeze(-1)
        new_env_ids = (~self.wind_time & new_wind_time).nonzero(as_tuple=False).squeeze(-1)
        self.wind_time = new_wind_time
        forces = self.wind_direction[env_ids]*500
        torques = torch.zeros(len(env_ids), 1, 3, device=self.device)
        world_forces = world_forces_to_local(forces, self.object.data.body_quat_w[env_ids, 0, :]).unsqueeze(1)
        self.object.set_external_force_and_torque(world_forces, torques, env_ids=env_ids)

        # Record how far the object is held
        self.held_position[new_env_ids] = self.object.data.body_pos_w[new_env_ids, 0]

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions[:, 0]*-self.cfg.action_scale_digits, joint_ids=self._left_finger_dof_idx)
        self.robot.set_joint_position_target(self.actions[:, 0]*-self.cfg.action_scale_digits, joint_ids=self._right_finger_dof_idx)
        self.robot.set_joint_position_target(self.actions[:, 0]*-self.cfg.action_scale_digits, joint_ids=self._back_finger_dof_idx)
        self.robot.set_joint_position_target(self.actions[:, 1]*self.cfg.action_scale_horizontal, joint_ids=self._planar_x_dof_idx)
        self.robot.set_joint_position_target(self.actions[:, 2]*self.cfg.action_scale_horizontal, joint_ids=self._planar_y_dof_idx)
        self.robot.set_joint_position_target(self.actions[:, 3]*self.cfg.action_scale_vertical, joint_ids=self._planar_z_dof_idx)

    def _get_observations(self) -> dict:
        obs = compute_observations(
            self.scene.env_origins, 
            self.robot.data.body_com_pos_w[:, self._hand_idx[0]],
            self.object.data.body_pos_w[:, 0],
            self.robot.data.joint_pos[:, self._left_finger_dof_idx],
            self.robot.data.joint_pos[:, self._right_finger_dof_idx],
            self.robot.data.joint_pos[:, self._back_finger_dof_idx],
            self.sensor_left.data.current_contact_time,
            self.sensors_right.data.current_contact_time,
            self.sensors_back.data.current_contact_time,
            self.wind_time,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_distance,
            self.cfg.rew_scale_lifted,
            self.body_pos[:, self._hand_idx[0]],
            self.object_pos[:, 0],
            self.sensor_left.data.current_contact_time[:, 0] > 0,
            self.sensors_right.data.current_contact_time[:, 0] > 0,
            self.sensors_back.data.current_contact_time[:, 0] > 0,
            self.wind_time,
            self.held_position,
            self.terminated
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.body_pos = self.robot.data.body_com_pos_w
        self.object_pos = self.object.data.body_pos_w
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = compute_boundaries(self.object_pos[:, 0, :], self.scene.env_origins)
        self.terminated = out_of_bounds
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Determine a random wind direction
        self.wind_direction[env_ids] = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        self.held_position[env_ids] = torch.zeros((len(env_ids), 3), device=self.device)
        self.wind_time[env_ids] = torch.zeros((len(env_ids),), dtype=torch.bool, device=self.device)

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
        object_root_state = self.object.data.default_root_state[env_ids]
        object_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Randomize object orientation for domain randomization
        random_z_rot = torch.rand((len(env_ids), 1), device=object_root_state.device)
        quat_w = torch.sqrt(1 - random_z_rot**2)
        object_root_state[:, 3:7] = torch.cat((quat_w, torch.zeros((len(env_ids), 2), device=object_root_state.device), random_z_rot), dim=1)

        # Write states to simulation - this will update the robot's internal data automatically
        self.robot.set_joint_position_target(torch.zeros(joint_pos.shape, device=joint_pos.device), env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.robot.write_root_pose_to_sim(robot_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(robot_root_state[:, 7:], env_ids)
        
        self.object.write_root_link_pose_to_sim(object_root_state[:, :7], env_ids)
        self.object.write_root_link_velocity_to_sim(object_root_state[:, 7:], env_ids)
        self.object.set_external_force_and_torque(torch.zeros(len(env_ids), 1, 3, device=self.device), torch.zeros(len(env_ids), 1, 3, device=self.device), env_ids=env_ids)


@torch.jit.script
def compute_observations(
    env_offsets: torch.Tensor, 
    body_pos: torch.Tensor, 
    object_pos: torch.Tensor,
    joint_pos_l: torch.Tensor,
    joint_pos_r: torch.Tensor,
    joint_pos_b: torch.Tensor,
    contact_left: torch.Tensor,
    contact_right: torch.Tensor,
    contact_back: torch.Tensor,
    wind_time: torch.Tensor,
) -> torch.Tensor:
    hand_pos = body_pos - env_offsets
    obj_rel_pos = body_pos - object_pos # relative to position of the hand
    obs = torch.cat(
        (
            hand_pos,
            obj_rel_pos,
            joint_pos_l.unsqueeze(dim=1),
            joint_pos_r.unsqueeze(dim=1),
            joint_pos_b.unsqueeze(dim=1),
            contact_left > 0,
            contact_right > 0,
            contact_back > 0,
            (~wind_time).unsqueeze(dim=1), # Manually do one hot observations for wind time
            wind_time.unsqueeze(dim=1),
        ),
        dim=-1,
    )
    return obs


@torch.jit.script
def compute_rewards(
    rew_scale_distance: float,
    rew_scale_lifted: float,
    hand_pos: torch.Tensor,
    object_pos: torch.Tensor,
    contact_left: torch.Tensor,
    contact_right: torch.Tensor,
    contact_back: torch.Tensor,
    wind_time: torch.Tensor,
    held_position: torch.Tensor,
    terminated: torch.Tensor,
) -> torch.Tensor:
    obj_lifted = object_pos[:, 2] > 1
    object_dist = torch.norm(hand_pos - object_pos, p=2, dim=-1)/torch.sqrt(300)
    sum_contact = contact_left.float() + contact_right.float() + contact_back.float()
    contact = sum_contact > 0.0
    double_contact = sum_contact > 1.0
    reward_initial = (1-object_dist)**rew_scale_distance + (obj_lifted.float()*double_contact.float())*rew_scale_lifted + contact.float()
    reward_wind = -10*(torch.norm(held_position - object_pos, p=2, dim=-1)/torch.sqrt(300)) + -50*terminated.float()
    reward = reward_initial#reward_initial*((~wind_time).float()) + reward_wind*(wind_time.float())
    return reward

@torch.jit.script
def compute_boundaries(
    object_pos: torch.Tensor,
    start_pos: torch.Tensor
) -> torch.Tensor:
    horizontal_bound = torch.any(torch.abs(object_pos[:, :2] - start_pos[:, :2]) > 1.5, dim=1)
    vertical_bound = torch.abs(object_pos[:, 2] - start_pos[:, 2]) > 5.0
    return horizontal_bound | vertical_bound


@torch.jit.script
def world_forces_to_local(forces_world, quats_wxyz):
    """
    forces_world: (N, 3) torch tensor, forces in world frame
    quats_wxyz:   (N, 4) torch tensor, quaternions (w, x, y, z) in world frame
    returns:      (N, 3) torch tensor, forces in local frame
    """
    w = quats_wxyz[:, 0]
    x = quats_wxyz[:, 1]
    y = quats_wxyz[:, 2]
    z = quats_wxyz[:, 3]

    # Rotation matrix from body→world
    R = torch.empty((quats_wxyz.shape[0], 3, 3), dtype=forces_world.dtype, device=forces_world.device)
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x*y - z*w)
    R[:, 0, 2] = 2 * (x*z + y*w)
    R[:, 1, 0] = 2 * (x*y + z*w)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y*z - x*w)
    R[:, 2, 0] = 2 * (x*z - y*w)
    R[:, 2, 1] = 2 * (y*z + x*w)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    # Inverse rotation matrix is its transpose
    R_inv = R.transpose(1, 2)

    # Multiply each inverse rotation by its force vector
    forces_local = torch.bmm(R_inv, forces_world.unsqueeze(-1)).squeeze(-1)
    return forces_local
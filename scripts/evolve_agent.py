# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with Stable Baselines3.

Since Stable-Baselines3 does not support buffers living on GPU directly,
we recommend using smaller number of environments. Otherwise,
there will be significant overhead in GPU->CPU transfer.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# Evolution-specific arguments
parser.add_argument("--evolution_cycles", type=int, default=5, help="Number of evolution cycles with different scales.")
parser.add_argument("--independent_axes", action="store_true", default=True, help="Scale X, Y, Z axes independently.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import random
from datetime import datetime
import gc
import threading
import time

import omni.timeline
import omni.usd
from pxr import Usd, UsdGeom, Gf

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import IsaacEnv.tasks  # noqa: F401


def find_and_scale_objects(env, scale_factor):
    """Alternative scaling method - find objects by searching the stage."""
    import omni.usd
    from pxr import Gf, UsdGeom, Usd
    
    try:
        stage = omni.usd.get_context().get_stage()
        num_envs = env.unwrapped.num_envs if hasattr(env, 'unwrapped') else env.num_envs
        
        
        scaled_count = 0
        for env_idx in range(num_envs):
            env_path = f"/World/envs/env_{env_idx}"
            
            # Search for objects in this environment
            object_path = f"{env_path}/Robot/object"
            prim = stage.GetPrimAtPath(object_path)
            
            if prim.IsValid():
                # Try to scale this object
                if scale_single_object(prim, scale_factor):
                    scaled_count += 1
                    break  # Found and scaled object for this env
        
        print(f"[INFO] Successfully scaled {scaled_count}/{num_envs} objects")
        return scaled_count > 0
        
    except Exception as e:
        print(f"[ERROR] Alternative scaling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def scale_single_object(prim, scale_factor):
    """Scale a single object prim."""
    from pxr import Gf, UsdGeom
    
    try:
        xform = UsdGeom.Xformable(prim)
        if not xform:
            return False
        
        # Look for existing scale operation instead of clearing all
        ops = xform.GetOrderedXformOps()
        scale_op = None
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op
                break
        
        # If no scale operation exists, create one
        if scale_op is None:
            scale_op = xform.AddScaleOp()
        
        # Set the scale
        scale_op.Set(Gf.Vec3f(scale_factor[0], scale_factor[1], scale_factor[2]))
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to scale single object {prim.GetPath()}: {e}")
        return False


def scale_objects_in_env(env, scale_factor):
    """Scale objects in all environments."""
    import omni.usd
    from pxr import Gf, UsdGeom
    
    try:
        stage = omni.usd.get_context().get_stage()
        num_envs = env.unwrapped.num_envs if hasattr(env, 'unwrapped') else env.num_envs
        
        success_count = 0
        for env_idx in range(num_envs):
            object_path = f"/World/envs/env_{env_idx}/object"
            prim = stage.GetPrimAtPath(object_path)
            
            if prim.IsValid():
                try:
                    xform = UsdGeom.Xformable(prim)
                    
                    # Get existing transform operations
                    ops = xform.GetOrderedXformOps()
                    
                    # Look for existing scale operation
                    scale_op = None
                    for op in ops:
                        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                            scale_op = op
                            break
                    
                    # If no scale operation exists, create one
                    if scale_op is None:
                        scale_op = xform.AddScaleOp()
                    
                    # Set the scale - this is safer than clearing all ops
                    scale_op.Set(Gf.Vec3f(scale_factor[0], scale_factor[1], scale_factor[2]))
                    
                    success_count += 1
                    print(f"[DEBUG] Successfully scaled object in env_{env_idx} to {scale_factor}")
                    
                except Exception as e:
                    print(f"[WARNING] Failed to scale object in env_{env_idx}: {e}")
                    continue
            else:
                print(f"[WARNING] Object prim {object_path} not found in env_{env_idx}")
        
        print(f"[INFO] Scaled {success_count}/{num_envs} objects to {scale_factor}")
        return success_count > 0
        
    except Exception as e:
        print(f"[ERROR] Failed to scale objects: {e}")
        import traceback
        traceback.print_exc()
        return False


def debug_environment_structure(env):
    """Debug function to print environment structure and find objects."""
    import omni.usd
    from pxr import Usd
    
    try:
        stage = omni.usd.get_context().get_stage()
        num_envs = min(2, env.unwrapped.num_envs if hasattr(env, 'unwrapped') else env.num_envs)
        
        print(f"\n[DEBUG] Environment structure analysis:")
        print(f"[DEBUG] Number of environments to check: {num_envs}")
        
        for env_idx in range(num_envs):
            env_path = f"/World/envs/env_{env_idx}"
            env_prim = stage.GetPrimAtPath(env_path)
            
            if env_prim.IsValid():
                print(f"\n[DEBUG] Environment {env_idx} ({env_path}):")
                children = list(env_prim.GetAllChildren())
                for child in children:
                    child_path = child.GetPath()
                    child_name = child.GetName()
                    print(f"  - {child_name} ({child_path})")
                    
                    # Check if this might be our object
                    if any(keyword in child_name.lower() for keyword in ['object', 'rigid', 'cube', 'sphere']):
                        print(f"    *** POTENTIAL OBJECT FOUND: {child_name} ***")
                        
                        # Check if it has children (like meshes)
                        grandchildren = list(child.GetAllChildren())
                        if grandchildren:
                            print(f"    Children of {child_name}:")
                            for grandchild in grandchildren:
                                print(f"      - {grandchild.GetName()} ({grandchild.GetPath()})")
            else:
                print(f"[DEBUG] Environment {env_idx} not found at {env_path}")
                
        # Also check the root structure
        print(f"\n[DEBUG] Root /World structure:")
        world_prim = stage.GetPrimAtPath("/World")
        if world_prim.IsValid():
            for child in world_prim.GetAllChildren():
                print(f"  - {child.GetName()} ({child.GetPath()})")
        
    except Exception as e:
        print(f"[ERROR] Debug failed: {e}")
        import traceback
        traceback.print_exc()


def verify_object_scales(env, expected_scale):
    """Verify that objects have been scaled correctly."""
    import omni.usd
    from pxr import UsdGeom
    
    try:
        stage = omni.usd.get_context().get_stage()
        num_envs = env.unwrapped.num_envs if hasattr(env, 'unwrapped') else env.num_envs
        
        # Check first few environments
        check_envs = min(3, num_envs)
        for env_idx in range(check_envs):
            object_path = f"/World/envs/env_{env_idx}/object"
            prim = stage.GetPrimAtPath(object_path)
            
            if prim.IsValid():
                xform = UsdGeom.Xformable(prim)
                ops = xform.GetOrderedXformOps()
                
                for op in ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                        current_scale = op.Get()
                        print(f"[VERIFY] Env {env_idx} object scale: {current_scale}, expected: {expected_scale}")
                        return True
        
        return False
    except Exception as e:
        print(f"[WARNING] Could not verify scales: {e}")
        return False


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with stable-baselines agent using evolving object scales."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    rng = np.random.default_rng(args_cli.seed)
    scale_objects(rng.random(3)*1.25 + 0.25)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    
    # Calculate timesteps per evolution cycle
    original_timesteps = 1000 #agent_cfg.get("n_timesteps", 10000)
    if args_cli.max_iterations is not None:
        original_timesteps = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs
    
    # Number of evolution cycles
    num_evolution_cycles = args_cli.evolution_cycles
    timesteps_per_cycle = original_timesteps // num_evolution_cycles

    # set the environment seed
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # directory for logging into
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"Exact experiment name requested from command line: {run_info}")
    print(f"[INFO] Evolution Training Configuration:")
    print(f"  - Cycles: {num_evolution_cycles}")
    print(f"  - Timesteps per cycle: {timesteps_per_cycle}")
    print(f"  - Independent axes: {args_cli.independent_axes}")
    log_dir = os.path.join(log_root_path, run_info)
    
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)
    policy_arch = agent_cfg.pop("policy")
    agent_cfg.pop("n_timesteps", None)  # Remove original timesteps

    # Initialize environment once
    print(f"[INFO] Creating environment with {env_cfg.scene.num_envs} parallel environments")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env)

    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # Create agent once
    agent = PPO(policy_arch, env, verbose=1, **agent_cfg)
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

    # Evolution training loop
    print(f"[INFO] Starting evolution training with {num_evolution_cycles} cycles")
    print(f"[INFO] {timesteps_per_cycle} timesteps per cycle")
    
    # Debug the initial environment structure
    debug_environment_structure(env)
    
    for cycle in range(num_evolution_cycles):
        print(f"\n{'='*60}")
        print(f"EVOLUTION CYCLE {cycle + 1}/{num_evolution_cycles}")
        print(f"{'='*60}")
        
        # For cycle 0, use existing environment. For others, recreate with new scale
        if cycle > 0:
            print(f"[INFO] Recreating environment with new object scale...")
                
            # Clean shutdown
            scene = env.unwrapped.scene
            env.unwrapped.sim.clear_instance()
            env.close()
            del env
            gc.collect()

            # Stop the timeline
            timeline = omni.timeline.get_timeline_interface()
            timeline.stop()
            timeline.commit()

            # Destroy all non-env objects
            if prim_utils.is_prim_path_valid("/World/ground"):
                prim_utils.delete_prim("/World/ground")
            if prim_utils.is_prim_path_valid("/World/Light"):
                prim_utils.delete_prim("/World/Light")

            torch.cuda.empty_cache()  # Clear GPU memory

            scale_factor = rng.random(3)*1.25 + 0.25
            print(f"[INFO] Cycle {cycle + 1}: Target scale {scale_factor}")
            scale_objects(scale_factor)
        
            # Recreate environment
            env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
            print(f"[INFO] Environment recreated with new scale factor {scale_factor}")
            
            # Re-apply all the wrappers
            if isinstance(env.unwrapped, DirectMARLEnv):
                env = multi_agent_to_single_agent(env)
            
            if args_cli.video:
                video_kwargs = {
                    "video_folder": os.path.join(log_dir, "videos", f"cycle_{cycle + 1}"),
                    "step_trigger": lambda step: step % args_cli.video_interval == 0,
                    "video_length": args_cli.video_length,
                    "disable_logger": True,
                }
                env = gym.wrappers.RecordVideo(env, **video_kwargs)
            
            env = Sb3VecEnvWrapper(env)
            
            if "normalize_input" in agent_cfg:
                env = VecNormalize(
                    env,
                    training=True,
                    norm_obs=True,
                    norm_reward=True,
                    clip_obs=10.0,
                    gamma=agent_cfg["gamma"],
                    clip_reward=np.inf,
                )
            
            # Update agent's environment
            agent.set_env(env)
            print(f"[INFO] Environment recreated and agent updated")
        
        # Reset environment to initialize with new scale (if recreated)
        env.reset()
        
        # Wait for environment to stabilize
        import time
        time.sleep(1.0)
        
        # Create checkpoint callback for this cycle
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1000, timesteps_per_cycle // 10), 
            save_path=os.path.join(log_dir, f"cycle_{cycle + 1}"),
            name_prefix=f"model_cycle_{cycle + 1}", 
            verbose=2
        )
        
        # Train for this cycle
        print(f"[INFO] Training cycle {cycle + 1} for {timesteps_per_cycle} timesteps...")
        agent.learn(total_timesteps=timesteps_per_cycle, callback=checkpoint_callback, reset_num_timesteps=False)
        
        # Save intermediate model
        cycle_model_path = os.path.join(log_dir, f"model_cycle_{cycle + 1}")
        agent.save(cycle_model_path)
        print(f"[INFO] Saved model for cycle {cycle + 1} to {cycle_model_path}")

    # Save final model
    final_model_path = os.path.join(log_dir, "model_final")
    agent.save(final_model_path)
    print(f"[INFO] Evolution training complete! Final model saved to {final_model_path}")

    # close the simulator
    env.close()


def scale_objects(scale_factor):
    stage = Usd.Stage.Open("source/IsaacEnv/IsaacEnv/tasks/direct/grasp_object/Grasp3D-v3.usd")
    prim = stage.GetPrimAtPath("/World/Object")
    xform = UsdGeom.Xformable(prim)
    scale_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break
    scale_op.Set(Gf.Vec3f(*scale_factor))
    stage.GetRootLayer().Export("source/IsaacEnv/IsaacEnv/tasks/direct/grasp_object/Grasp3D-temp.usd")

    # Reload the USD file using Omniverse context
    context = omni.usd.get_context()
    context.new_stage()
    context.get_stage().Reload()


if __name__ == "__main__":
    main()
    simulation_app.close()

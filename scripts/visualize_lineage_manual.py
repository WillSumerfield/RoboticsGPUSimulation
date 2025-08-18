"""Manual Lineage Visualization Script.

This script loads a population_results.yaml and an agent id, traces the evolutionary lineage,
recreates the environment for each ancestor, and waits for the user to press Enter before moving to the next.
Allows manual screenshot capture for each agent in the lineage.
"""

import argparse
import os
import yaml
import numpy as np
import torch
from isaaclab.app import AppLauncher
import gymnasium as gym
import time
import sys
import threading


def load_population_lineage(results_path, agent_id):
    with open(results_path, 'r') as f:
        results = yaml.unsafe_load(f)
    agents = {a['id']: a for a in results['agents']}
    lineage = []
    current = agents.get(agent_id)
    while current is not None:
        lineage.append(current)
        parent_id = current.get('parent_id')
        if parent_id is not None and parent_id in agents:
            current = agents[parent_id]
        else:
            current = None
    lineage.reverse()
    return lineage, results['config']


def main():
    parser = argparse.ArgumentParser(description="Manually visualize each agent in a lineage.")
    parser.add_argument('--results_file', type=str, required=True, help='Path to population_results.yaml')
    parser.add_argument('--agent_id', type=int, required=True, help='Target agent id')
    parser.add_argument('--task', type=str, default='GraspObject', help='Task name')
    parser.add_argument('--num_envs', type=int, default=1, help='Number of envs (default 1)')
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()

    # Launch Omniverse App
    args.headless = args.headless
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Load lineage
    lineage, config = load_population_lineage(args.results_file, args.agent_id)
    print(f"[INFO] Lineage for agent {args.agent_id}: {[a['id'] for a in lineage]}")

    # Import environment and parameter utilities
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.hydra import hydra_task_config
    import IsaacEnv.tasks  # noqa: F401
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
    from evolution_common import modify_params_usd


    sys.argv = [sys.argv[0]] + hydra_args

    def keychecker(stop_flag):
        input()
        stop_flag[0] = True

    @hydra_task_config(args.task, "sb3_cfg_entry_point")
    def run_env(env_cfg, agent_cfg: dict):
        env_cfg.scene.num_envs = args.num_envs
        if hasattr(env_cfg, 'sim'):
            env_cfg.sim.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        task_folder = args.task.replace("-", "_").lower()

        # Camera setup imports
        import omni.usd
        from pxr import UsdGeom, Gf

        def set_camera_pose(eye=(1.5, 0, 1.0), target=(0, 0, 0.5)):
            stage = omni.usd.get_context().get_stage()
            cam_prim = stage.GetPrimAtPath("/OmniverseKit_Persp")
            if not cam_prim.IsValid():
                print("[WARN] Camera prim not found at /OmniverseKit_Persp")
                return
            xform = UsdGeom.Xformable(cam_prim)
            # Compute look-at transform
            eye = Gf.Vec3d(*eye)
            target = Gf.Vec3d(*target)
            up = Gf.Vec3d(0, 0, 1)
            forward = (target - eye).GetNormalized()
            right = Gf.Cross(forward, up).GetNormalized()
            up = Gf.Cross(right, forward).GetNormalized()
            rot = Gf.Matrix3d(right[0], right[1], right[2], up[0], up[1], up[2], -forward[0], -forward[1], -forward[2])
            mat = Gf.Matrix4d().SetRotateOnly(rot)
            mat.SetTranslateOnly(eye)
            xform.ClearXformOpOrder()
            op = xform.AddTransformOp()
            op.Set(mat)

        for i, agent in enumerate(lineage):
            print(f"\n[INFO] Showing Agent {agent['id']} (step {i+1}/{len(lineage)})")
            # Set parameters
            modify_params_usd(task_folder, np.array(agent['params']))
            # Create environment
            env = gym.make(args.task, cfg=env_cfg)
            if isinstance(env.unwrapped, DirectMARLEnv):
                env = multi_agent_to_single_agent(env)
            env = Sb3VecEnvWrapper(env)
            # Reset and render (no agent actions)
            obs = env.reset()
            # Set camera pose after reset
            set_camera_pose()
            print("[INFO] Please take a screenshot now (or interact as needed). Press Enter to continue to next agent...")
            stop_flag = [False]
            t = threading.Thread(target=keychecker, args=(stop_flag,), daemon=True)
            t.start()
            while not stop_flag[0]:
                actions = torch.zeros((env.num_envs, env_cfg.action_space), device=env.unwrapped.device)
                actions[:, 3] = 1.0
                actions[:, 0] = 1.0
                env.step(actions)
                time.sleep(0.01)
            env.close()
            time.sleep(0.5)  # Give time for GUI to close before next

    env_cfg = run_env()

    print("[INFO] All agents in lineage visualized.")
    simulation_app.close()

if __name__ == "__main__":
    main()

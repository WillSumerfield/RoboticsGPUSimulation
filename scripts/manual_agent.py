import argparse
from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description="Manual agent for Isaac Lab environments.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch
    import pygame
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    import IsaacEnv.tasks  # noqa: F401


    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.robot_cfg.spawn.usd_path = env_cfg.robot_cfg.spawn.usd_path.replace("Grasp3D-temp.usd", "Grasp3D.usd")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    # Initialize pygame for keyboard input
    pygame.init()
    screen = pygame.display.set_mode((300, 100))
    pygame.display.set_caption("IsaacEnv Manual Agent Control")

    print("[INFO]: Use WASD for movement (actions 1-3), Up/Down arrows for gripper (action 0)")

    # Action mapping
    move_step = 0.1  # Change as needed
    grip_open = 1.0
    grip_closed = 0.0

    action = torch.zeros(env.action_space.shape[1], device=env.unwrapped.device)
    reward = 0.0

    while simulation_app.is_running():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                simulation_app.close()
                return

        keys = pygame.key.get_pressed()
        # Gripper control (action 0)
        if keys[pygame.K_UP]:
            action[0] = grip_open
        elif keys[pygame.K_DOWN]:
            action[0] = grip_closed

        # WASD movement (actions 1-3)
        if keys[pygame.K_w]:
            action[1] += move_step  # X+ (forward)
        if keys[pygame.K_s]:
            action[1] -= move_step  # X- (backward)
        if keys[pygame.K_a]:
            action[2] -= move_step  # Y- (left)
        if keys[pygame.K_d]:
            action[2] += move_step  # Y+ (right)

        # Y axis (up/down) can be mapped to other keys if needed
        if keys[pygame.K_q]:
            action[3] += move_step  # Z+ (up)
        if keys[pygame.K_e]:
            action[3] -= move_step  # Z- (down)

        action = torch.clamp(action, -1, 1)
        with torch.inference_mode():
            obs = env.step(action.clone()[None, :])

        # Reset actions on reset
        if torch.any(obs[2]) or torch.any(obs[3]):
            action = torch.zeros(env.action_space.shape[1], device=env.unwrapped.device)
            reward = 0.0

        reward += obs[1].mean().item()

        print(f"Obs: {[round(i, 2) for i in (obs[0]['policy'].tolist()[0])]}", end="\r")

        pygame.time.wait(int(1000/60))  # Control loop speed

    env.close()
    pygame.quit()
    simulation_app.close()

if __name__ == "__main__":
    main()
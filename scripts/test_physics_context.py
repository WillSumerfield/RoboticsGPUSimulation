#!/usr/bin/env python3

"""
Simple example showing how to create new physics contexts with different object scales.
This demonstrates the approach without full RL training.
"""

import torch
from isaaclab.app import AppLauncher

# local imports
from .isaacenv_env_cfg import IsaacenvEnvCfg
import IsaacEnv.tasks  # noqa: F401


def test_physics_context_recreation():
    """Test recreating physics contexts with different object scales."""
    
    print("Testing Physics Context Recreation with Object Scaling")
    print("=" * 60)
    
    # Create base configuration
    env_cfg = IsaacenvEnvCfg()
    env_cfg.scene.num_envs = 16  # Small number for testing
    env_cfg.sim.device = "cuda:0"
    
    # Test different scales
    test_scales = [
        (1.0, 1.0, 1.0),    # Original size
        (1.5, 1.5, 1.5),    # 1.5x larger
        (0.7, 0.7, 0.7),    # Smaller
        (2.0, 1.0, 1.0),    # Only X-axis scaled
    ]
    
    current_env = None
    
    for i, scale in enumerate(test_scales):
        print(f"\nCycle {i+1}: Creating Environment with scale {scale}")
        print("-" * 50)
        
        # Properly close previous environment
        if current_env is not None:
            print("Closing previous environment...")
            try:
                # Close the environment properly
                current_env.close()
                del current_env
                
                # Clear any remaining references
                torch.cuda.empty_cache()
                
                # Clear the simulation context completely
                from isaaclab.sim import SimulationContext
                if SimulationContext.instance() is not None:
                    print("Clearing simulation context...")
                    SimulationContext.clear_instance()
                
            except Exception as e:
                print(f"Warning during cleanup: {e}")
        
        print("Creating new environment...")
        
        # Create a new environment instance with fresh physics context
        # Pass the object scale during construction
        current_env = IsaacenvEnv(cfg=env_cfg, render_mode=None, object_scale=scale)
        
        # Verify the object size
        actual_size = current_env.cfg.object_cfg.spawn.size
        print(f"Object size set to: {actual_size}")
        
        # Test the environment
        print("Running simulation test...")
        current_env.reset()
        
        for step in range(5):
            actions = torch.randn(env_cfg.scene.num_envs, current_env.action_space, device=current_env.device)
            actions = torch.clamp(actions, -1.0, 1.0)
            
            obs, rewards, terminated, truncated, info = current_env.step(actions)
            
            if step == 0:
                print(f"Object positions: {current_env.object_pos[0]}")  # First env position
                print(f"Average reward: {rewards.mean().item():.3f}")
        
        print(f"Cycle {i+1} completed successfully!")
    
    # Final cleanup
    if current_env is not None:
        print("\nFinal cleanup...")
        current_env.close()
        del current_env
        
        # Clear simulation context
        from isaaclab.sim import SimulationContext
        if SimulationContext.instance() is not None:
            SimulationContext.clear_instance()
    
    print("\nPhysics context recreation test completed!")


def main():
    """Main function."""
    
    # Parse arguments and launch Isaac
    app_launcher = AppLauncher(args_cli=["--headless"])
    simulation_app = app_launcher.app
    
    try:
        test_physics_context_recreation()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

"""Example of how to add runtime object scaling to your IsaacenvEnv."""

import torch
import omni.usd
from pxr import Gf, UsdGeom

class IsaacenvEnvWithScaling(IsaacenvEnv):
    """Extended version of IsaacenvEnv with runtime object scaling capability."""
    
    def __init__(self, cfg: IsaacenvEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Store original scales for reset purposes
        self._original_object_scale = (1.0, 1.0, 1.0)
        self._current_object_scale = (1.0, 1.0, 1.0)
    
    def scale_object(self, env_idx: int, scale_factor: tuple[float, float, float]):
        """Scale the object in a specific environment during runtime.
        
        Args:
            env_idx: Environment index to scale object in
            scale_factor: (x, y, z) scale factors
        """
        stage = omni.usd.get_context().get_stage()
        object_prim_path = f"/World/envs/env_{env_idx}/object"
        
        prim = stage.GetPrimAtPath(object_prim_path)
        if prim.IsValid():
            xform = UsdGeom.Xformable(prim)
            
            # Clear existing scale operations
            xform.ClearXformOpOrder()
            
            # Add new scale operation
            scale_op = xform.AddScaleOp()
            scale_op.Set(Gf.Vec3f(scale_factor[0], scale_factor[1], scale_factor[2]))
            
            print(f"Scaled object in env_{env_idx} by {scale_factor}")
            return True
        else:
            print(f"Object prim {object_prim_path} not found.")
            return False
    
    def scale_all_objects(self, scale_factor: tuple[float, float, float]):
        """Scale objects in all environments.
        
        Args:
            scale_factor: (x, y, z) scale factors to apply to all objects
        """
        success_count = 0
        for env_idx in range(self.num_envs):
            if self.scale_object(env_idx, scale_factor):
                success_count += 1
        
        self._current_object_scale = scale_factor
        print(f"Successfully scaled objects in {success_count}/{self.num_envs} environments")
        return success_count
    
    def randomize_object_scales(self, min_scale: float = 0.5, max_scale: float = 2.0):
        """Randomize object scales across all environments.
        
        Args:
            min_scale: Minimum scale factor
            max_scale: Maximum scale factor
        """
        for env_idx in range(self.num_envs):
            # Generate random scale (uniform for all axes to maintain proportions)
            random_scale = torch.rand(1).item() * (max_scale - min_scale) + min_scale
            scale_factor = (random_scale, random_scale, random_scale)
            self.scale_object(env_idx, scale_factor)
    
    def reset_object_scales(self):
        """Reset all objects to their original scale."""
        self.scale_all_objects(self._original_object_scale)
        self._current_object_scale = self._original_object_scale
    
    def _reset_idx(self, env_ids):
        """Override reset to optionally reset scales."""
        super()._reset_idx(env_ids)
        
        # Uncomment if you want to reset scales on environment reset
        # for env_id in env_ids:
        #     self.scale_object(env_id, self._original_object_scale)


# Example usage in a training script:
def example_usage():
    """Example of how to use the scaling functionality."""
    
    # Create environment with scaling capability
    env = IsaacenvEnvWithScaling(cfg=your_config)
    
    # Scale all objects to 1.5x their original size
    env.scale_all_objects((1.5, 1.5, 1.5))
    
    # Scale object in environment 0 to be very small
    env.scale_object(0, (0.3, 0.3, 0.3))
    
    # Randomize scales across all environments
    env.randomize_object_scales(min_scale=0.7, max_scale=1.8)
    
    # Reset all scales back to original
    env.reset_object_scales()
    
    # You could also integrate this into your training loop:
    obs, _ = env.reset()
    for step in range(1000):
        actions = torch.rand(env.num_envs, env.action_space.shape[0])
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Change scales every 100 steps
        if step % 100 == 0:
            env.randomize_object_scales()

"""Safe object scaling for IsaacLab environments."""

import torch
from typing import Sequence
import omni.usd
from pxr import Gf, UsdGeom

# Add this to your IsaacenvEnv class
class ScalableMixin:
    """Mixin to add safe scaling capabilities to IsaacLab environments."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_scales = {}  # Track scales for each environment
    
    def scale_object_in_env(self, env_id: int, scale_factor: tuple[float, float, float]):
        """Scale object in a specific environment.
        
        IMPORTANT: Only call this during _reset_idx() or _setup_scene()
        """
        object_path = f"/World/envs/env_{env_id}/object"
        
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(object_path)
        
        if not prim.IsValid():
            print(f"Object prim {object_path} not found.")
            return False
        
        try:
            xform = UsdGeom.Xformable(prim)
            
            # Find or create scale operation
            ops = xform.GetOrderedXformOps()
            scale_op = None
            for op in ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale_op = op
                    break
            
            if scale_op is None:
                scale_op = xform.AddScaleOp()
            
            # Apply scale
            scale_op.Set(Gf.Vec3f(scale_factor[0], scale_factor[1], scale_factor[2]))
            
            # Store the scale for tracking
            self._object_scales[env_id] = scale_factor
            
            return True
            
        except Exception as e:
            print(f"Failed to scale object in env_{env_id}: {e}")
            return False
    
    def randomize_object_scales(self, env_ids: Sequence[int], 
                               min_scale: float = 0.8, max_scale: float = 1.2):
        """Randomize object scales for given environments."""
        for env_id in env_ids:
            # Generate random uniform scale to maintain proportions
            random_scale = torch.rand(1).item() * (max_scale - min_scale) + min_scale
            scale_factor = (random_scale, random_scale, random_scale)
            self.scale_object_in_env(env_id, scale_factor)
    
    def reset_object_scales(self, env_ids: Sequence[int]):
        """Reset object scales to original size."""
        for env_id in env_ids:
            self.scale_object_in_env(env_id, (1.0, 1.0, 1.0))
            if env_id in self._object_scales:
                del self._object_scales[env_id]


# Modified version of your IsaacenvEnv with safe scaling
class IsaacenvEnvWithSafeScaling(IsaacenvEnv, ScalableMixin):
    """Your environment with safe scaling capabilities."""
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Override to add safe scaling during reset."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        # Call parent reset first
        super()._reset_idx(env_ids)
        
        # Now it's safe to scale objects (simulation is paused for these envs)
        # Example: Randomize scales every reset
        if hasattr(self.cfg, 'randomize_object_scale') and self.cfg.randomize_object_scale:
            self.randomize_object_scales(env_ids, 
                                       min_scale=getattr(self.cfg, 'min_object_scale', 0.8),
                                       max_scale=getattr(self.cfg, 'max_object_scale', 1.2))
    
    def _setup_scene(self):
        """Override to set initial scales if needed."""
        super()._setup_scene()
        
        # You can set initial scales here if needed
        # Example: Scale object in env_0 to be larger for testing
        # self.scale_object_in_env(0, (1.5, 1.5, 1.5))


# Example configuration additions for your isaacenv_env_cfg.py
"""
Add these to your IsaacenvEnvCfg class:

@configclass
class IsaacenvEnvCfg(DirectRLEnvCfg):
    # ... existing config ...
    
    # Object scaling settings
    randomize_object_scale: bool = False  # Enable random scaling
    min_object_scale: float = 0.8         # Minimum scale factor
    max_object_scale: float = 1.2         # Maximum scale factor
"""

# Example usage in training script
def example_training_with_scaling():
    """Example of how to use the scaling environment in training."""
    
    # Create environment with scaling
    env = IsaacenvEnvWithSafeScaling(cfg=your_config)
    
    # Training loop
    obs, _ = env.reset()
    
    for episode in range(1000):
        for step in range(env.max_episode_length):
            actions = torch.rand(env.num_envs, env.action_space.shape[0])
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Environment will automatically randomize scales on reset if configured
        
        # Manual scaling every 10 episodes for specific environments
        if episode % 10 == 0:
            # Scale objects in first 4 environments
            env.randomize_object_scales(list(range(4)), min_scale=0.5, max_scale=2.0)

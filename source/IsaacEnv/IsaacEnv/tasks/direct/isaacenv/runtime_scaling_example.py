"""Example of scaling objects during runtime in IsaacLab."""

import omni.usd
import omni.physx
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
import carb

def scale_object_runtime(prim_path: str, scale_factor: tuple[float, float, float], update_physics: bool = True):
    """Scale an object during runtime in Isaac Sim with proper physics handling.
    
    Args:
        prim_path: USD path to the object to scale
        scale_factor: (x, y, z) scale factors
        update_physics: Whether to update physics properties after scaling
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    
    if not prim.IsValid():
        print(f"Prim {prim_path} not found.")
        return False
        
    try:
        # Apply the scale transform
        xform = UsdGeom.Xformable(prim)
        
        # Clear existing scale operations to avoid conflicts
        ops = xform.GetOrderedXformOps()
        scale_op = None
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op
                break
        
        # If no scale op exists, create one
        if scale_op is None:
            scale_op = xform.AddScaleOp()
        
        # Set the new scale
        scale_op.Set(Gf.Vec3f(scale_factor[0], scale_factor[1], scale_factor[2]))
        
        # Update physics properties if needed
        if update_physics:
            _update_physics_properties(prim, scale_factor)
        
        print(f"Scaled {prim_path} by {scale_factor}")
        
        # Restart physics simulation
        if update_physics and physx_interface:
            physx_interface.start_simulation()
            
        return True
        
    except Exception as e:
        carb.log_error(f"Failed to scale object {prim_path}: {e}")
        # Restart physics if we stopped it
        if update_physics:
            physx_interface = omni.physx.get_physx_interface()
            if physx_interface:
                physx_interface.start_simulation()
        return False

def _update_physics_properties(prim, scale_factor):
    """Update physics properties after scaling."""
    # Update mass properties if rigid body
    rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
    if rigid_body_api:
        # Scale mass proportionally (mass scales with volume, so use product of scale factors)
        mass_scale = scale_factor[0] * scale_factor[1] * scale_factor[2]
        if rigid_body_api.GetMassAttr().HasValue():
            current_mass = rigid_body_api.GetMassAttr().Get()
            if current_mass:
                rigid_body_api.GetMassAttr().Set(current_mass * mass_scale)
    
    # Update collision shapes
    collision_api = UsdPhysics.CollisionAPI(prim)
    if collision_api:
        # For mesh colliders, the scale is automatically handled by the transform
        # For primitive colliders (box, sphere, etc.), we might need to update dimensions
        pass

def safe_scale_object_runtime(prim_path: str, scale_factor: tuple[float, float, float]):
    """Safer version that pauses simulation during scaling."""
    import omni.timeline
    
    timeline = omni.timeline.get_timeline_interface()
    was_playing = timeline.is_playing()
    
    # Pause simulation
    if was_playing:
        timeline.pause()
    
    # Apply scaling
    success = scale_object_runtime(prim_path, scale_factor, update_physics=True)
    
    # Resume simulation if it was playing
    if was_playing and success:
        timeline.play()
    
    return success

# Example usage in your environment:
# IMPORTANT: Call these during environment reset or setup, not during simulation steps

def example_usage_in_env():
    """Example of how to use in your IsaacenvEnv - call during reset or setup."""
    
    # Method 1: Safe scaling with simulation pause
    safe_scale_object_runtime("/World/envs/env_0/object", (1.5, 1.5, 1.5))
    
    # Method 2: Scale during environment reset (recommended)
    # Add this to your _reset_idx() method in IsaacenvEnv:
    """
    def _reset_idx(self, env_ids):
        # Stop simulation updates for these environments
        super()._reset_idx(env_ids)
        
        # Now it's safe to scale objects
        for env_id in env_ids:
            object_path = f"/World/envs/env_{env_id}/object"
            scale_factor = (1.0 + torch.rand(1).item() * 0.5, 1.0, 1.0)  # Random x-scale
            scale_object_runtime(object_path, scale_factor, update_physics=False)
    """
    
    # Method 3: Scale multiple objects safely
    objects_to_scale = [
        ("/World/envs/env_0/object", (2.0, 2.0, 2.0)),
        ("/World/envs/env_1/object", (0.5, 0.5, 0.5)),
    ]
    
    for prim_path, scale in objects_to_scale:
        safe_scale_object_runtime(prim_path, scale)

def domain_randomization_example():
    """Example of using scaling for domain randomization."""
    import torch
    
    # Scale objects with random factors during training
    for env_idx in range(16):  # Assuming 16 environments
        # Random scale between 0.7 and 1.3
        random_scale = 0.7 + torch.rand(1).item() * 0.6
        scale_factor = (random_scale, random_scale, random_scale)
        
        object_path = f"/World/envs/env_{env_idx}/object"
        safe_scale_object_runtime(object_path, scale_factor)

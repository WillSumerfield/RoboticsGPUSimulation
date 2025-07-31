#!/usr/bin/env python3

"""Script to scale objects in a USD file."""

from pxr import Usd, UsdGeom, Gf


def scale_object_in_usd(usd_file_path: str, object_prim_path: str, scale_factor: tuple[float, float, float]):
    """
    Scale an object in a USD file.
    
    Args:
        usd_file_path: Path to the USD file
        object_prim_path: Path to the object prim in the USD (e.g., "/Robot/object")
        scale_factor: Scale factors for (X, Y, Z) axes
    """
    # Open the USD stage
    stage = Usd.Stage.Open(usd_file_path)
    
    # Get the prim
    prim = stage.GetPrimAtPath(object_prim_path)
    
    if not prim.IsValid():
        print(f"Error: Prim '{object_prim_path}' not found in USD file.")
        return False
    
    # Create or get the Xformable
    xformable = UsdGeom.Xformable(prim)
    
    # Add scale operation
    scale_op = xformable.AddScaleOp()
    scale_op.Set(Gf.Vec3f(scale_factor[0], scale_factor[1], scale_factor[2]))
    
    # Save the stage
    stage.Save()
    print(f"Successfully scaled '{object_prim_path}' by {scale_factor}")
    return True

def scale_multiple_objects(usd_file_path: str, objects_and_scales: dict):
    """
    Scale multiple objects in a USD file.
    
    Args:
        usd_file_path: Path to the USD file
        objects_and_scales: Dictionary mapping object paths to scale factors
    """
    stage = Usd.Stage.Open(usd_file_path)
    
    for object_path, scale_factor in objects_and_scales.items():
        prim = stage.GetPrimAtPath(object_path)
        
        if not prim.IsValid():
            print(f"Warning: Prim '{object_path}' not found, skipping.")
            continue
        
        xformable = UsdGeom.Xformable(prim)
        scale_op = xformable.AddScaleOp()
        scale_op.Set(Gf.Vec3f(scale_factor[0], scale_factor[1], scale_factor[2]))
        print(f"Scaled '{object_path}' by {scale_factor}")
    
    stage.Save()
    print(f"Saved changes to {usd_file_path}")

if __name__ == "__main__":
    # Example usage
    usd_file = "path/to/your/file.usd"  # Change this to your USD file path
    
    # Scale a single object
    scale_object_in_usd(
        usd_file_path=usd_file,
        object_prim_path="/Robot/object",  # Change to your object's path
        scale_factor=(2.0, 2.0, 2.0)  # 2x scale on all axes
    )

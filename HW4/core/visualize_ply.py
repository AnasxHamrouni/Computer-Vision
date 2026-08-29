#!/usr/bin/env python3
"""
Visualize PLY file to find correct coordinate system
"""
import numpy as np
import open3d as o3d
from plyfile import PlyData
import sys

def visualize_ply_with_axes(ply_path):
    """Load and visualize PLY with coordinate frame."""
    print(f"Loading {ply_path}...")
    
    # Load PLY
    try:
        ply = PlyData.read(ply_path)
        v = ply["vertex"].data
        
        if all(name in v.dtype.names for name in ("x", "y", "z")):
            xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
            print(f"Loaded {xyz.shape} points")
        else:
            print("No x,y,z in PLY. Trying with open3d...")
            pcd = o3d.io.read_point_cloud(ply_path)
            xyz = np.asarray(pcd.points, dtype=np.float32)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # Analyze bounds
    pmin = xyz.min(axis=0)
    pmax = xyz.max(axis=0)
    center = 0.5 * (pmin + pmax)
    
    print(f"\nScene Bounds:")
    print(f"  X: {pmin:.3f} to {pmax:.3f} (center: {center:.3f})")
    print(f"  Y: {pmin:.3f} to {pmax:.3f} (center: {center:.3f})")
    print(f"  Z: {pmin:.3f} to {pmax:.3f} (center: {center:.3f})")
    
    # Detect floor and ceiling (Y distribution)
    y_vals = xyz[:, 1]
    floor_y = np.percentile(y_vals, 2)
    ceiling_y = np.percentile(y_vals, 98)
    print(f"\nFloor (2nd percentile): {floor_y:.3f}")
    

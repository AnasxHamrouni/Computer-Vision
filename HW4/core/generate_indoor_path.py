#!/usr/bin/env python3
"""
Indoor Navigation Path Generator for 3D Gaussian Splatting Models

This script generates cinematic camera paths for indoor scenes (e.g., ConferenceHall.ply)
that:
- Avoid collisions with walls and obstacles
- Create smooth, cinematic camera movements
- Explore the interior space intelligently
- Support custom start positions

Usage:
    python generate_indoor_path.py --ply scenes/conference-hall/ConferenceHall.ply \
                                   --start-x 0.0 --start-y 1.5 --start-z 0.0 \
                                   --output scenes/conference-hall/trajectory_indoor.json
"""

import json
import math
import argparse
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import os
import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import splprep, splev
from sklearn.cluster import DBSCAN
from collections import deque

# Try to import matplotlib for visualization (optional)
HAS_MATPLOTLIB = False
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    pass

# Try to import object detection modules (optional)
HAS_OBJECT_DETECTION = False
try:
    from object_detection import (
        detect_objects_3d_geometric,
        detect_objects_3d_height_based,
        detect_objects_2d_yolo,
        detect_objects_2d_batch,
        project_2d_to_3d,
        visualize_detections_3d,
        visualize_detections_2d,
        export_detections_json,
        Detection2D,
        Detection3D
    )
    HAS_OBJECT_DETECTION = True
except ImportError:
    print("[WARN] Object detection modules not available. Install dependencies:")
    print("       pip install ultralytics scikit-learn")
    pass


# ===================== DATA STRUCTURES =====================

@dataclass
class Waypoint:
    position: np.ndarray
    quaternion: np.ndarray
    fov: float = 60.0


# ===================== PLY LOADING =====================

def unpack_position(packed: np.uint32) -> np.ndarray:
    """Unpack packed position from 3DGS PLY format."""
    # 3DGS uses packed format: 3x float16 packed into uint32
    # Each float16 is 16 bits, so we extract 3 values
    x_bits = (packed >> 0) & 0xFFFF
    y_bits = (packed >> 16) & 0xFFFF
    
    # For the third component, we need to check if it's in the same uint32 or next
    # Actually, packed_position is a single uint32 that contains 3 half-floats
    # Let's use a different approach - convert uint32 to bytes and interpret as half-floats
    import struct
    
    # Convert uint32 to bytes (little-endian)
    packed_bytes = struct.pack('<I', packed)
    # Unpack as 2 uint16s
    u16_0, u16_1 = struct.unpack('<HH', packed_bytes)
    
    # Convert uint16 to float16, then to float32
    def uint16_to_float16(u16):
        # IEEE 754 half-precision format
        sign = (u16 >> 15) & 0x1
        exponent = (u16 >> 10) & 0x1F
        mantissa = u16 & 0x3FF
        
        if exponent == 0:
            if mantissa == 0:
                return 0.0 if sign == 0 else -0.0
            # Denormalized
            value = mantissa / 1024.0 * (2.0 ** (-14))
        elif exponent == 31:
            if mantissa == 0:
                return float('inf') if sign == 0 else float('-inf')
            else:
                return float('nan')
        else:
            # Normalized
            value = (1.0 + mantissa / 1024.0) * (2.0 ** (exponent - 15))
        
        return -value if sign else value
    
    x = uint16_to_float16(u16_0)
    y = uint16_to_float16(u16_1)
    
    # For z, we might need to read from a different packed value or it's stored differently
    # Let's try a simpler approach: use numpy's view to interpret as half-floats
    try:
        # Try to interpret the packed value as 2 half-floats directly
        packed_array = np.array([packed], dtype=np.uint32)
        # View as uint16 array
        as_uint16 = packed_array.view(np.uint16)
        # Convert to float16, then to float32
        as_float16 = as_uint16.astype(np.float16)
        as_float32 = as_float16.astype(np.float32)
        
        if len(as_float32) >= 2:
            x, y = as_float32[0], as_float32[1]
        else:
            x, y = float(uint16_to_float16(as_uint16[0])), float(uint16_to_float16(as_uint16[1]))
    except:
        x, y = float(uint16_to_float16(u16_0)), float(uint16_to_float16(u16_1))
    
    # For z, we might need to check if there's another packed value or use a default
    # Actually, let's check the actual data structure - maybe z is in a different field
    z = 0.0  # Will be filled from actual data
    
    return np.array([x, y, z], dtype=np.float32)


def load_positions_from_npy(npy_path: str) -> np.ndarray:
    """Load positions from pre-extracted NPY file."""
    positions = np.load(npy_path)
    # Filter out invalid positions
    valid_mask = np.all(np.isfinite(positions), axis=1)
    # Also filter out positions that are way too large (likely unpacking errors)
    valid_mask = valid_mask & np.all(np.abs(positions) < 1e6, axis=1)
    positions = positions[valid_mask]
    print(f"[LOAD] Loaded {len(positions)} valid positions from {npy_path}")
    return positions

def load_ply_points(ply_path: str) -> np.ndarray:
    """Load point cloud from PLY file (supports both standard PLY and 3DGS format)."""
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    
    print(f"[LOAD] Reading PLY: {ply_path}")
    
    # Try 3DGS format first (using plyfile)
    try:
        ply = PlyData.read(ply_path)
        element_names = [e.name for e in ply.elements]
        
        if "vertex" not in element_names:
            raise ValueError("PLY file has no 'vertex' element; expected Gaussian splat centers.")
        
        v = ply["vertex"].data
        
        # Check for standard x, y, z format
        if all(name in v.dtype.names for name in ("x", "y", "z")):
            xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
            print(f"[LOAD] Loaded {xyz.shape[0]} points from 3DGS PLY format (x,y,z)")
            return xyz
        
        # Check for packed format (3DGS format)
        elif "packed_position" in v.dtype.names:
            print(f"[LOAD] Detected packed 3DGS format...")
            print(f"[LOAD] This format requires special unpacking. Using subsampled points for path planning...")
            
            # For 3DGS packed format, we'll use a workaround:
            # Since unpacking is complex and the file is large (6M+ points),
            # we'll create a simplified approach using the bounding box
            # and generate a path based on estimated room dimensions
            
            # Try to get a sense of scale from a small sample
            packed_positions = v["packed_position"]
            sample_size = min(1000, len(packed_positions))
            sample_indices = np.linspace(0, len(packed_positions) - 1, sample_size, dtype=int)
            
            # For now, we'll use a placeholder approach
            # The user can provide start coordinates, and we'll generate a path
            # based on estimated room size from the PLY metadata
            
            print(f"[LOAD] Note: Using estimated room dimensions for path planning.")
            print(f"[LOAD] Total vertices: {len(packed_positions)}")
            print(f"[LOAD] For accurate path generation, please provide start coordinates.")
            print(f"[LOAD] The path generator will create a safe exploration path.")
            
            # Check if we have a pre-extracted positions file
            npy_path = ply_path.replace('.ply', '_positions.npy')
            if os.path.exists(npy_path):
                print(f"[LOAD] Found pre-extracted positions file: {npy_path}")
                return load_positions_from_npy(npy_path)
            
            # Otherwise, raise error with instructions
            raise ValueError(
                f"Packed 3DGS format detected. Please extract positions first:\n"
                f"  python core/extract_positions_for_path.py {ply_path} {npy_path}\n"
                f"Then run the path generator again."
            )
        else:
            raise ValueError(f"PLY vertex element doesn't have x,y,z or packed_position. Available: {v.dtype.names}")
            
    except Exception as e:
        print(f"[LOAD] 3DGS format failed: {e}, trying standard PLY...")
        # Fallback to standard PLY format (using open3d)
        try:
            pcd = o3d.io.read_point_cloud(ply_path)
            if len(pcd.points) == 0:
                raise ValueError(f"Empty point cloud: {ply_path}")
            pts = np.asarray(pcd.points, dtype=np.float32)
            print(f"[LOAD] Loaded {len(pts)} points from standard PLY format")
            return pts
        except Exception as e2:
            raise ValueError(f"Failed to load PLY file with both methods. 3DGS error: {e}, Open3D error: {e2}")


def remove_outliers(pts: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    """Remove statistical outliers from point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    inlier_pts = np.asarray(cl.points, dtype=np.float32)
    print(f"[CLEAN] Removed outliers: {len(pts)} -> {len(inlier_pts)} points")
    return inlier_pts


# ===================== INDOOR SPACE ANALYSIS =====================

def visualize_room_detection(pts: np.ndarray, rooms: List[dict], start_pos: np.ndarray,
                            space_info: dict, output_path: str = "debug_room_detection.png"):
    """Visualize detected rooms and start position for debugging."""
    if not HAS_MATPLOTLIB:
        print("[DEBUG] Skipping visualization (matplotlib not available)")
        return
    
    try:
        fig = plt.figure(figsize=(16, 10))
        
        # 3D plot
        ax1 = fig.add_subplot(221, projection='3d')
        # Sample points for visualization (too many points slow down plotting)
        sample_idx = np.random.choice(len(pts), min(5000, len(pts)), replace=False)
        sample_pts = pts[sample_idx]
        ax1.scatter(sample_pts[:, 0], sample_pts[:, 2], sample_pts[:, 1], 
                   c='lightgray', s=1, alpha=0.3, label='Point cloud')
        
        # Draw rooms
        colors = plt.cm.tab20(np.linspace(0, 1, len(rooms)))
        for i, room in enumerate(rooms):
            center = room['center']
            bounds = room['bounds']
            width = bounds['x_max'] - bounds['x_min']
            depth = bounds['z_max'] - bounds['z_min']
            
            # Draw room as rectangle
            x_corners = [bounds['x_min'], bounds['x_max'], bounds['x_max'], bounds['x_min'], bounds['x_min']]
            z_corners = [bounds['z_min'], bounds['z_min'], bounds['z_max'], bounds['z_max'], bounds['z_min']]
            y_val = center[1]
            
            ax1.plot(x_corners, z_corners, [y_val]*5, 
                    color=colors[i], linewidth=2, label=f"Room {i+1} ({room['type']})")
            ax1.scatter([center[0]], [center[2]], [center[1]], 
                       color=colors[i], s=100, marker='o')
        
        # Draw start position
        ax1.scatter([start_pos[0]], [start_pos[2]], [start_pos[1]], 
                   color='red', s=200, marker='*', label='Start position', zorder=10)
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Z')
        ax1.set_zlabel('Y')
        ax1.set_title('3D View: Rooms and Start Position')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2D top-down view (XZ)
        ax2 = fig.add_subplot(222)
        ax2.scatter(sample_pts[:, 0], sample_pts[:, 2], c='lightgray', s=1, alpha=0.3)
        for i, room in enumerate(rooms):
            bounds = room['bounds']
            rect = plt.Rectangle((bounds['x_min'], bounds['z_min']),
                               bounds['x_max'] - bounds['x_min'],
                               bounds['z_max'] - bounds['z_min'],
                               fill=False, edgecolor=colors[i], linewidth=2)
            ax2.add_patch(rect)
            ax2.scatter(room['center'][0], room['center'][2], 
                       color=colors[i], s=100, marker='o')
        ax2.scatter(start_pos[0], start_pos[2], color='red', s=200, marker='*', zorder=10)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_title('Top-Down View (XZ plane)')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # Room sizes bar chart
        ax3 = fig.add_subplot(223)
        room_names = [f"Room {i+1}\n({r['type']})" for i, r in enumerate(rooms)]
        areas = [r['area'] for r in rooms]
        ax3.bar(room_names, areas, color=colors[:len(rooms)])
        ax3.set_ylabel('Area (m²)')
        ax3.set_title('Room Sizes')
        ax3.tick_params(axis='x', rotation=45)
        
        # Start position info
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        info_text = f"Start Position Debug Info\n"
        info_text += f"{'='*40}\n"
        info_text += f"Start: ({start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f})\n\n"
        
        # Check which room contains start position
        in_room = None
        for i, room in enumerate(rooms):
            bounds = room['bounds']
            if (bounds['x_min'] <= start_pos[0] <= bounds['x_max'] and
                bounds['z_min'] <= start_pos[2] <= bounds['z_max']):
                in_room = i
                info_text += f"✓ Inside Room {i+1} ({room['type']})\n"
                break
        
        if in_room is None:
            info_text += f"✗ NOT INSIDE ANY ROOM!\n"
            info_text += f"  This is the problem!\n\n"
            # Find nearest room
            min_dist = float('inf')
            nearest_room = None
            for i, room in enumerate(rooms):
                dist = np.linalg.norm(start_pos[[0, 2]] - room['center'][[0, 2]])
                if dist < min_dist:
                    min_dist = dist
                    nearest_room = i
            if nearest_room is not None:
                info_text += f"Nearest room: Room {nearest_room+1}\n"
                info_text += f"Distance: {min_dist:.2f}m\n"
        
        info_text += f"\nDetected {len(rooms)} rooms\n"
        info_text += f"Main room: Room 1 ({rooms[0]['area']:.1f}m²)\n"
        info_text += f"  Center: ({rooms[0]['center'][0]:.2f}, {rooms[0]['center'][1]:.2f}, {rooms[0]['center'][2]:.2f})\n"
        info_text += f"  Bounds: X[{rooms[0]['bounds']['x_min']:.2f}, {rooms[0]['bounds']['x_max']:.2f}]\n"
        info_text += f"           Z[{rooms[0]['bounds']['z_min']:.2f}, {rooms[0]['bounds']['z_max']:.2f}]"
        
        ax4.text(0.1, 0.5, info_text, fontfamily='monospace', fontsize=10,
                verticalalignment='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[DEBUG] Saved visualization to {output_path}")
        plt.close()
    except Exception as e:
        print(f"[DEBUG] Visualization failed: {e}")


def detect_interior_rooms(pts: np.ndarray, space_info: dict, 
                         voxel_size: float = 0.2, debug: bool = True) -> List[dict]:
    """
    Detect separate rooms/spaces in the building by analyzing point density
    and spatial clustering. Identifies corridors vs rooms.
    
    Returns list of room dictionaries with center, bounds, and type.
    """
    print("[ROOMS] Detecting interior rooms and spaces...")
    
    pmin = space_info['pmin']
    pmax = space_info['pmax']
    floor_y = space_info['floor_y']
    ceiling_y = space_info['ceiling_y']
    
    # Filter points to interior height range (exclude floor/ceiling)
    height_margin = (ceiling_y - floor_y) * 0.1
    interior_mask = (pts[:, 1] >= floor_y + height_margin) & (pts[:, 1] <= ceiling_y - height_margin)
    interior_pts = pts[interior_mask]
    
    if len(interior_pts) < 100:
        print("[ROOMS] Not enough interior points, using all points")
        interior_pts = pts
    
    print(f"[ROOMS] Analyzing {len(interior_pts)} interior points...")
    
    # Build 2D density map (XZ plane)
    size = pmax - pmin
    nx = int(np.ceil(size[0] / voxel_size))
    nz = int(np.ceil(size[2] / voxel_size))
    
    density = np.zeros((nx, nz), dtype=np.int32)
    ix = ((interior_pts[:, 0] - pmin[0]) / voxel_size).astype(int)
    iz = ((interior_pts[:, 2] - pmin[2]) / voxel_size).astype(int)
    ix = np.clip(ix, 0, nx - 1)
    iz = np.clip(iz, 0, nz - 1)
    
    for x, z in zip(ix, iz):
        density[x, z] += 1
    
    # Smooth density map
    density_smooth = gaussian_filter(density.astype(np.float32), sigma=1.5)
    
    # Find dense regions (rooms have high density)
    density_threshold = np.percentile(density_smooth[density_smooth > 0], 30)
    dense_mask = density_smooth >= density_threshold
    
    # Cluster dense regions to identify separate rooms
    # Use connected components on dense cells
    room_regions = []
    visited = np.zeros_like(dense_mask, dtype=bool)
    
    for x in range(nx):
        for z in range(nz):
            if dense_mask[x, z] and not visited[x, z]:
                # Flood fill to find connected region
                region_cells = []
                q = deque([(x, z)])
                visited[x, z] = True
                
                while q:
                    cx, cz = q.popleft()
                    region_cells.append((cx, cz))
                    
                    for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nx_, nz_ = cx + dx, cz + dz
                        if 0 <= nx_ < nx and 0 <= nz_ < nz:
                            if dense_mask[nx_, nz_] and not visited[nx_, nz_]:
                                visited[nx_, nz_] = True
                                q.append((nx_, nz_))
                
                if len(region_cells) >= 10:  # Minimum room size
                    # Convert cell indices to world coordinates
                    region_x = [pmin[0] + (cx + 0.5) * voxel_size for cx, _ in region_cells]
                    region_z = [pmin[2] + (cz + 0.5) * voxel_size for _, cz in region_cells]
                    
                    room_x_min, room_x_max = min(region_x), max(region_x)
                    room_z_min, room_z_max = min(region_z), max(region_z)
                    room_center_x = (room_x_min + room_x_max) * 0.5
                    room_center_z = (room_z_min + room_z_max) * 0.5
                    
                    # Classify as room or corridor based on aspect ratio
                    width = room_x_max - room_x_min
                    depth = room_z_max - room_z_min
                    aspect_ratio = max(width, depth) / (min(width, depth) + 1e-6)
                    
                    room_type = "corridor" if aspect_ratio > 3.0 else "room"
                    area = width * depth
                    
                    room_regions.append({
                        'center': np.array([room_center_x, (floor_y + ceiling_y) * 0.5, room_center_z], dtype=np.float32),
                        'bounds': {
                            'x_min': room_x_min,
                            'x_max': room_x_max,
                            'z_min': room_z_min,
                            'z_max': room_z_max
                        },
                        'area': area,
                        'type': room_type,
                        'width': width,
                        'depth': depth,
                        'aspect_ratio': aspect_ratio,
                        'cell_count': len(region_cells)
                    })
    
    # Sort by area (largest first)
    room_regions.sort(key=lambda r: r['area'], reverse=True)
    
    print(f"[ROOMS] Detected {len(room_regions)} rooms/spaces:")
    for i, room in enumerate(room_regions[:5]):  # Show top 5
        print(f"  Room {i+1}: {room['type']} at {room['center']}, area={room['area']:.2f}m², "
              f"size={room['width']:.2f}x{room['depth']:.2f}m")
    
    if debug:
        print(f"[DEBUG] Room detection details:")
        print(f"  Voxel size: {voxel_size}m")
        print(f"  Density threshold: {density_threshold:.2f}")
        print(f"  Total interior points analyzed: {len(interior_pts)}")
        print(f"  Dense cells found: {dense_mask.sum()}")
        print(f"  Room regions found: {len(room_regions)}")
        print(f"  Overall point cloud bounds: X[{pmin[0]:.2f}, {pmax[0]:.2f}], "
              f"Z[{pmin[2]:.2f}, {pmax[2]:.2f}]")
        if len(room_regions) > 0:
            print(f"  Largest room bounds: X[{room_regions[0]['bounds']['x_min']:.2f}, "
                  f"{room_regions[0]['bounds']['x_max']:.2f}], "
                  f"Z[{room_regions[0]['bounds']['z_min']:.2f}, "
                  f"{room_regions[0]['bounds']['z_max']:.2f}]")
            # Check if room is actually inside point cloud bounds
            room = room_regions[0]
            inside_x = pmin[0] <= room['bounds']['x_min'] and room['bounds']['x_max'] <= pmax[0]
            inside_z = pmin[2] <= room['bounds']['z_min'] and room['bounds']['z_max'] <= pmax[2]
            print(f"  Room inside point cloud bounds: X={inside_x}, Z={inside_z}")
            if not (inside_x and inside_z):
                print(f"  ⚠ WARNING: Detected room extends outside point cloud bounds!")
                print(f"     This might indicate the room detection is finding exterior areas.")
    
    return room_regions


def find_best_start_position(pts: np.ndarray, space_info: dict, kdtree: cKDTree, 
                             rooms: Optional[List[dict]] = None, 
                             debug: bool = True) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Automatically find the best starting position inside the building.
    - Uses provided rooms list or detects rooms
    - Selects the largest room (main conference room)
    - Finds a safe position near the center of that room
    
    Returns: (start_position, main_room_dict)
    """
    print("[AUTO] Automatically detecting best start position...")
    
    # Detect rooms if not provided
    if rooms is None:
        rooms = detect_interior_rooms(pts, space_info, debug=debug)
    
    if not rooms:
        print("[AUTO] No rooms detected, using scene center")
        center = space_info['center']
        floor_y = space_info['floor_y']
        ceiling_y = space_info['ceiling_y']
        start_pos = np.array([center[0], (floor_y + ceiling_y) * 0.5, center[2]], dtype=np.float32)
        start_pos_before = start_pos.copy()
        start_pos = find_nearest_free_position(start_pos, kdtree, space_info, preserve_y=True)
        
        if debug:
            print(f"[DEBUG] No rooms found, using scene center:")
            print(f"  Before collision check: {start_pos_before}")
            print(f"  After collision check: {start_pos}")
            print(f"  Moved by: {np.linalg.norm(start_pos - start_pos_before):.3f}m")
        
        return start_pos, None
    
    # Find the largest room (main conference room)
    main_room = rooms[0]  # Already sorted by area
    print(f"[AUTO] Selected main room: {main_room['type']} with area {main_room['area']:.2f}m²")
    
    if debug:
        print(f"[DEBUG] Main room details:")
        print(f"  Center: {main_room['center']}")
        print(f"  Bounds: X[{main_room['bounds']['x_min']:.2f}, {main_room['bounds']['x_max']:.2f}], "
              f"Z[{main_room['bounds']['z_min']:.2f}, {main_room['bounds']['z_max']:.2f}]")
        print(f"  Size: {main_room['width']:.2f}m x {main_room['depth']:.2f}m")
    
    # Start near the center of the main room
    start_pos = main_room['center'].copy()
    start_pos_before = start_pos.copy()
    
    if debug:
        print(f"[DEBUG] Initial start position (room center): {start_pos}")
    
    # Ensure it's collision-free
    start_pos = find_nearest_free_position(start_pos, kdtree, space_info, preserve_y=True)
    
    if debug:
        print(f"[DEBUG] After collision check: {start_pos}")
        movement = np.linalg.norm(start_pos - start_pos_before)
        print(f"[DEBUG] Moved by: {movement:.3f}m")
        
        # Verify it's still inside the room
        bounds = main_room['bounds']
        inside_x = bounds['x_min'] <= start_pos[0] <= bounds['x_max']
        inside_z = bounds['z_min'] <= start_pos[2] <= bounds['z_max']
        
        # Also check distance to nearest point in point cloud
        dist_to_nearest, nearest_idx = kdtree.query(start_pos, k=1)
        nearest_pt = kdtree.data[nearest_idx]
        
        print(f"[DEBUG] Position check:")
        print(f"  X in bounds [{bounds['x_min']:.2f}, {bounds['x_max']:.2f}]: {inside_x} "
              f"(value: {start_pos[0]:.2f})")
        print(f"  Z in bounds [{bounds['z_min']:.2f}, {bounds['z_max']:.2f}]: {inside_z} "
              f"(value: {start_pos[2]:.2f})")
        print(f"  Distance to nearest point in cloud: {dist_to_nearest:.3f}m")
        print(f"  Nearest point: {nearest_pt}")
        
        # Check if we're too far from any geometry (might be outside building)
        if dist_to_nearest > 2.0:
            print(f"[DEBUG] ⚠ WARNING: Start position is {dist_to_nearest:.2f}m from nearest geometry!")
            print(f"[DEBUG]   This suggests the position might be outside the building.")
        
        if not (inside_x and inside_z):
            print(f"[DEBUG] ⚠ WARNING: Start position moved OUTSIDE room bounds!")
            print(f"[DEBUG]   This might cause the camera to start outside the building.")
            # Try to push it back inside
            if not inside_x:
                start_pos[0] = np.clip(start_pos[0], bounds['x_min'] + 0.5, bounds['x_max'] - 0.5)
                print(f"[DEBUG]   Corrected X to: {start_pos[0]:.2f}")
            if not inside_z:
                start_pos[2] = np.clip(start_pos[2], bounds['z_min'] + 0.5, bounds['z_max'] - 0.5)
                print(f"[DEBUG]   Corrected Z to: {start_pos[2]:.2f}")
            print(f"[DEBUG]   Final corrected position: {start_pos}")
    
    print(f"[AUTO] Selected start position: {start_pos}")
    
    # Final summary for debugging
    if debug:
        print(f"\n[DEBUG] ========== START POSITION SUMMARY ==========")
        print(f"  Start Position: ({start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f})")
        if main_room:
            print(f"  Main Room Center: ({main_room['center'][0]:.3f}, {main_room['center'][1]:.3f}, {main_room['center'][2]:.3f})")
            print(f"  Main Room Bounds: X[{main_room['bounds']['x_min']:.2f}, {main_room['bounds']['x_max']:.2f}], "
                  f"Z[{main_room['bounds']['z_min']:.2f}, {main_room['bounds']['z_max']:.2f}]")
            print(f"  Distance from room center: {np.linalg.norm(start_pos - main_room['center']):.3f}m")
        print(f"  Distance to nearest geometry: {dist_to_nearest:.3f}m")
        print(f"  Point cloud bounds: X[{space_info['pmin'][0]:.2f}, {space_info['pmax'][0]:.2f}], "
              f"Z[{space_info['pmin'][2]:.2f}, {space_info['pmax'][2]:.2f}]")
        print(f"  ================================================\n")
    
    return start_pos, main_room


def analyze_indoor_space(pts: np.ndarray) -> dict:
    """Analyze indoor space to detect floor, ceiling, and free space."""
    pmin = pts.min(axis=0)
    pmax = pts.max(axis=0)
    center = 0.5 * (pmin + pmax)
    
    # Detect floor and ceiling from Y distribution
    y_values = pts[:, 1]
    floor_y = float(np.percentile(y_values, 2))  # 2nd percentile as floor
    ceiling_y = float(np.percentile(y_values, 98))  # 98th percentile as ceiling
    
    # Estimate room dimensions
    room_size = pmax - pmin
    room_height = ceiling_y - floor_y
    
    print(f"[SPACE] Floor: {floor_y:.3f}, Ceiling: {ceiling_y:.3f}, Height: {room_height:.3f}")
    print(f"[SPACE] Room size: X={room_size[0]:.3f}, Y={room_size[1]:.3f}, Z={room_size[2]:.3f}")
    print(f"[SPACE] Center: {center}")
    
    return {
        'pmin': pmin,
        'pmax': pmax,
        'center': center,
        'floor_y': floor_y,
        'ceiling_y': ceiling_y,
        'room_height': room_height,
        'room_size': room_size
    }


def build_free_space_map(pts: np.ndarray, space_info: dict, 
                        voxel_size: float = 0.15) -> Tuple[np.ndarray, np.ndarray, cKDTree]:
    """
    Build free space map that identifies void/empty areas inside the building.
    Returns: (free_space_map, void_centers, kdtree)
    """
    # Create KD-tree for collision detection
    kdtree = cKDTree(pts)
    
    pmin = space_info['pmin']
    pmax = space_info['pmax']
    floor_y = space_info['floor_y']
    ceiling_y = space_info['ceiling_y']
    
    nx = int(np.ceil((pmax[0] - pmin[0]) / voxel_size))
    nz = int(np.ceil((pmax[2] - pmin[2]) / voxel_size))
    
    # Build density map
    density_map = np.zeros((nx, nz), dtype=np.int32)
    height_map = np.full((nx, nz), -np.inf, dtype=np.float32)
    
    for pt in pts:
        ix = int((pt[0] - pmin[0]) / voxel_size)
        iz = int((pt[2] - pmin[2]) / voxel_size)
        ix = np.clip(ix, 0, nx - 1)
        iz = np.clip(iz, 0, nz - 1)
        
        density_map[ix, iz] += 1
        if pt[1] > height_map[ix, iz]:
            height_map[ix, iz] = pt[1]
    
    # Find free space: areas with low density (voids inside rooms)
    # Use a threshold - cells with very low density are likely free space
    density_threshold = np.percentile(density_map[density_map > 0], 10)  # Bottom 10% are voids
    free_space_map = density_map < density_threshold
    
    # Also check distance to nearest point - free space should be far from walls
    # Sample points in each cell and check distance
    void_centers = []
    for ix in range(nx):
        for iz in range(nz):
            if free_space_map[ix, iz]:
                # Check if this cell is actually free (far from geometry)
                cell_x = pmin[0] + (ix + 0.5) * voxel_size
                cell_z = pmin[2] + (iz + 0.5) * voxel_size
                cell_y = (floor_y + ceiling_y) * 0.5  # Middle height
                test_pos = np.array([cell_x, cell_y, cell_z], dtype=np.float32)
                
                dist, _ = kdtree.query(test_pos, k=1)
                # Free space: at least 1.0m from nearest geometry
                if dist >= 1.0:
                    void_centers.append(test_pos)
    
    void_centers = np.array(void_centers, dtype=np.float32) if void_centers else np.array([], dtype=np.float32).reshape(0, 3)
    
    print(f"[VOID] Grid size: {nx}x{nz}, Free space cells: {free_space_map.sum()}/{free_space_map.size}")
    print(f"[VOID] Found {len(void_centers)} void center points for path planning")
    
    return free_space_map, void_centers, kdtree


def build_occupancy_grid(pts: np.ndarray, space_info: dict, 
                        voxel_size: float = 0.1) -> Tuple[np.ndarray, cKDTree]:
    """Build 3D occupancy grid and KD-tree for collision detection."""
    # Create KD-tree for fast nearest neighbor queries
    kdtree = cKDTree(pts)
    
    # Build 2D occupancy map (XZ plane) for path planning
    pmin = space_info['pmin']
    pmax = space_info['pmax']
    floor_y = space_info['floor_y']
    ceiling_y = space_info['ceiling_y']
    
    nx = int(np.ceil((pmax[0] - pmin[0]) / voxel_size))
    nz = int(np.ceil((pmax[2] - pmin[2]) / voxel_size))
    
    # Height map: max Y per XZ cell
    height_map = np.full((nx, nz), -np.inf, dtype=np.float32)
    density_map = np.zeros((nx, nz), dtype=np.int32)
    
    for pt in pts:
        ix = int((pt[0] - pmin[0]) / voxel_size)
        iz = int((pt[2] - pmin[2]) / voxel_size)
        ix = np.clip(ix, 0, nx - 1)
        iz = np.clip(iz, 0, nz - 1)
        
        if pt[1] > height_map[ix, iz]:
            height_map[ix, iz] = pt[1]
        density_map[ix, iz] += 1
    
    # Mark occupied cells (where there's geometry)
    occupied = density_map > 0
    
    print(f"[OCCUPANCY] Grid size: {nx}x{nz}, Occupied cells: {occupied.sum()}/{occupied.size}")
    
    return height_map, kdtree


def is_collision_free(pos: np.ndarray, kdtree: cKDTree, 
                     cam_radius: float = 0.5, clearance: float = 0.5) -> bool:
    """Check if a position is collision-free."""
    dist, _ = kdtree.query(pos, k=1)
    return dist >= (cam_radius + clearance)


def find_nearest_free_position(pos: np.ndarray, kdtree: cKDTree,
                              space_info: dict, max_iterations: int = 30,
                              preserve_y: bool = False) -> np.ndarray:
    """Find nearest collision-free position by pushing away from obstacles."""
    cam_radius = 0.5  # Increased from 0.3
    clearance = 0.5   # Increased from 0.2 - more aggressive clearance
    min_dist = cam_radius + clearance
    
    original_y = pos[1] if preserve_y else None
    current_pos = pos.copy()
    
    for _ in range(max_iterations):
        dist, idx = kdtree.query(current_pos, k=1)
        
        if dist >= min_dist:
            # Restore original Y if preserving
            if preserve_y and original_y is not None:
                current_pos[1] = original_y
            return current_pos
        
        # Push away from nearest obstacle
        nearest_pt = kdtree.data[idx]
        direction = current_pos - nearest_pt
        norm = np.linalg.norm(direction)
        
        if norm < 1e-6:
            # Random direction if too close
            direction = np.random.randn(3)
            direction[1] = 0  # Keep Y component small
            norm = np.linalg.norm(direction)
        
        direction = direction / norm
        # More aggressive push - add extra margin
        push_distance = (min_dist - dist) * 1.5 + 0.2  # Increased multiplier and base distance
        current_pos = current_pos + direction * push_distance
        
        # Clamp to room bounds
        current_pos[0] = np.clip(current_pos[0], space_info['pmin'][0], space_info['pmax'][0])
        # Only clamp Y if not preserving it
        if not preserve_y:
            current_pos[1] = np.clip(current_pos[1], 
                                    space_info['floor_y'] + 0.5, 
                                    space_info['ceiling_y'] - 0.5)
        current_pos[2] = np.clip(current_pos[2], space_info['pmin'][2], space_info['pmax'][2])
    
    # Restore original Y if preserving
    if preserve_y and original_y is not None:
        current_pos[1] = original_y
    return current_pos


# ===================== PATH GENERATION =====================

def generate_indoor_exploration_path(start_pos: np.ndarray,
                                     space_info: dict,
                                     kdtree: cKDTree,
                                     n_frames: int = 600,
                                     exploration_radius: float = None,
                                     main_room: Optional[dict] = None,
                                     void_centers: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Generate an indoor exploration path that:
    - Starts from the given position
    - Explores the space in a cinematic way
    - Avoids walls and obstacles
    - Creates smooth, flowing movements
    """
    center = space_info['center']
    floor_y = space_info['floor_y']
    ceiling_y = space_info['ceiling_y']
    room_size = space_info['room_size']
    
    # Ensure start position is collision-free
    # Preserve Y coordinate if user provided it explicitly (might be in different coord system)
    preserve_y = True  # Always preserve user-provided Y for now
    start_pos = find_nearest_free_position(start_pos, kdtree, space_info, preserve_y=preserve_y)
    print(f"[PATH] Start position: {start_pos}")
    
    # Determine exploration radius - use full building space, not just main room
    # Start in main room but explore the entire building
    if exploration_radius is None:
        # Use larger exploration radius based on full building size
        exploration_radius = min(room_size[0], room_size[2]) * 0.6  # Explore 60% of building
        print(f"[PATH] Exploration radius: {exploration_radius:.2f}m (based on building size)")
    
    # Use building center for exploration, but start from main room
    if main_room is not None:
        print(f"[PATH] Starting in main room: {main_room['width']:.2f}x{main_room['depth']:.2f}m")
        print(f"[PATH] But exploring entire building: {room_size[0]:.2f}x{room_size[2]:.2f}m")
    else:
        print(f"[PATH] Exploring building: {room_size[0]:.2f}x{room_size[2]:.2f}m")
    
    # Use the start position's Y as the base height (user-provided coordinate)
    # This respects the coordinate system from SuperSplat
    base_y = start_pos[1]
    print(f"[PATH] Using start Y as base height: {base_y:.3f}")
    print(f"[PATH] Room floor: {floor_y:.3f}, ceiling: {ceiling_y:.3f}")
    
    # Generate path waypoints following void space
    waypoints = []
    
    # If we have void centers, use them to create a comprehensive exploration path
    if void_centers is not None and len(void_centers) > 0:
        print(f"[PATH] Using {len(void_centers)} void centers for comprehensive room exploration")
        
        # Filter void centers to be within main room if available
        if main_room is not None:
            bounds = main_room['bounds']
            room_mask = ((void_centers[:, 0] >= bounds['x_min']) & 
                        (void_centers[:, 0] <= bounds['x_max']) &
                        (void_centers[:, 2] >= bounds['z_min']) & 
                        (void_centers[:, 2] <= bounds['z_max']))
            room_voids = void_centers[room_mask]
            if len(room_voids) > 10:
                void_centers = room_voids
                print(f"[PATH] Filtered to {len(void_centers)} void centers within main room")
        
        # Create a comprehensive exploration path using grid-based approach
        # Build a grid of exploration points within the room
        if main_room is not None:
            bounds = main_room['bounds']
            room_width = bounds['x_max'] - bounds['x_min']
            room_depth = bounds['z_max'] - bounds['z_min']
            room_center_x = (bounds['x_min'] + bounds['x_max']) * 0.5
            room_center_z = (bounds['z_min'] + bounds['z_max']) * 0.5
            
            # Create a grid with spacing that ensures good coverage
            grid_spacing = min(room_width, room_depth) / 8.0  # 8x8 grid for good coverage
            grid_spacing = max(grid_spacing, 1.5)  # Minimum 1.5m spacing
            
            # Generate grid points
            grid_points = []
            num_x = int(room_width / grid_spacing) + 1
            num_z = int(room_depth / grid_spacing) + 1
            
            for ix in range(num_x):
                for iz in range(num_z):
                    x = bounds['x_min'] + ix * grid_spacing
                    z = bounds['z_min'] + iz * grid_spacing
                    # Only add if within bounds
                    if (bounds['x_min'] <= x <= bounds['x_max'] and 
                        bounds['z_min'] <= z <= bounds['z_max']):
                        grid_points.append([x, base_y, z])
            
            grid_points = np.array(grid_points, dtype=np.float32)
            print(f"[PATH] Generated {len(grid_points)} grid points for exploration")
            
            # Find void centers closest to each grid point
            start_tree = cKDTree(void_centers)
            exploration_points = []
            
            for grid_pt in grid_points:
                # Find nearest void center to this grid point
                dist, nearest_idx = start_tree.query(grid_pt, k=1)
                if isinstance(nearest_idx, np.ndarray):
                    nearest_idx = int(nearest_idx[0])
                else:
                    nearest_idx = int(nearest_idx)
                
                # Use void center position, but keep grid Y
                void_pt = void_centers[nearest_idx].copy()
                void_pt[1] = grid_pt[1]  # Use grid Y
                
                # Blend grid point with void center for better coverage
                blend = 0.6  # Prefer void center but keep some grid structure
                exploration_pt = (1 - blend) * grid_pt + blend * void_pt
                exploration_points.append(exploration_pt)
            
            exploration_points = np.array(exploration_points, dtype=np.float32)
            
            # Create a lawnmower/spiral path through exploration points
            # Sort points in a pattern that covers the room
            # Use a simple approach: sort by X, then Z (zigzag pattern)
            exploration_points = exploration_points[np.lexsort((exploration_points[:, 2], exploration_points[:, 0]))]
            
            # Add start position at beginning
            exploration_points = np.vstack([start_pos.reshape(1, 3), exploration_points])
            
            # Interpolate along exploration points to create smooth path
            num_exploration_steps = min(len(exploration_points), n_frames // 2)
            exploration_indices = np.linspace(0, len(exploration_points) - 1, num_exploration_steps, dtype=int)
            exploration_path = exploration_points[exploration_indices]
            
            print(f"[PATH] Created exploration path with {len(exploration_path)} waypoints")
            
            # Create smooth path through exploration points
            for i in range(n_frames):
                t = i / n_frames
                
                if len(exploration_path) > 1:
                    # Map t to exploration point index
                    path_idx_float = t * (len(exploration_path) - 1)
                    path_idx = int(path_idx_float)
                    path_idx_next = min(path_idx + 1, len(exploration_path) - 1)
                    alpha = path_idx_float - path_idx
                    
                    # Interpolate between exploration points
                    pos = (1 - alpha) * exploration_path[path_idx] + alpha * exploration_path[path_idx_next]
                    
                    # Add small variation for smoother movement
                    variation_radius = 0.2
                    variation_angle = t * math.pi * 8
                    pos[0] += variation_radius * math.cos(variation_angle)
                    pos[2] += variation_radius * math.sin(variation_angle)
                    
                    # Subtle height variation
                    pos[1] = base_y + 0.05 * math.sin(t * math.pi * 4)
                else:
                    pos = exploration_path[0].copy()
                    pos[1] = base_y
                
                # Ensure collision-free
                dist_to_wall, _ = kdtree.query(pos, k=1)
                if dist_to_wall < 0.8:
                    pos = find_nearest_free_position(pos, kdtree, space_info, preserve_y=True)
                
                waypoints.append(pos)
        else:
            # Fallback: use void centers directly with better sampling
            start_tree = cKDTree(void_centers)
            
            # Use more void centers for better coverage
            num_path_voids = min(80, len(void_centers))  # Increased from 30
            void_indices = np.linspace(0, len(void_centers) - 1, num_path_voids, dtype=int)
            path_voids = void_centers[void_indices]
            
            # Create smooth path through void centers
            for i in range(n_frames):
                t = i / n_frames
                
                if len(path_voids) > 1:
                    void_idx_float = t * (len(path_voids) - 1)
                    void_idx = int(void_idx_float)
                    void_idx_next = min(void_idx + 1, len(path_voids) - 1)
                    alpha = void_idx_float - void_idx
                    
                    void_pos = (1 - alpha) * path_voids[void_idx] + alpha * path_voids[void_idx_next]
                    
                    # Add variation for exploration
                    variation_radius = 0.4
                    variation_angle = t * math.pi * 6
                    void_pos[0] += variation_radius * math.cos(variation_angle)
                    void_pos[2] += variation_radius * math.sin(variation_angle)
                    void_pos[1] = base_y + 0.05 * math.sin(t * math.pi * 4)
                else:
                    void_pos = path_voids[0].copy()
                    void_pos[1] = base_y
                
                # Ensure collision-free
                dist_to_wall, _ = kdtree.query(void_pos, k=1)
                if dist_to_wall < 0.8:
                    pos = find_nearest_free_position(void_pos, kdtree, space_info, preserve_y=True)
                else:
                    pos = void_pos
                
                waypoints.append(pos)
    else:
        # Fallback: use grid-based exploration pattern for thorough room coverage
        print(f"[PATH] No void centers found, using grid-based exploration pattern")
        
        if main_room is not None:
            bounds = main_room['bounds']
            room_width = bounds['x_max'] - bounds['x_min']
            room_depth = bounds['z_max'] - bounds['z_min']
            room_center_x = (bounds['x_min'] + bounds['x_max']) * 0.5
            room_center_z = (bounds['z_min'] + bounds['z_max']) * 0.5
            
            # Create a comprehensive grid-based exploration
            grid_spacing = min(room_width, room_depth) / 6.0  # 6x6 grid
            grid_spacing = max(grid_spacing, 1.5)
            
            # Generate grid points in lawnmower pattern
            grid_points = []
            num_x = int(room_width / grid_spacing) + 1
            num_z = int(room_depth / grid_spacing) + 1
            
            # Create zigzag pattern for better coverage
            for iz in range(num_z):
                for ix in range(num_x):
                    # Zigzag: reverse direction on odd rows
                    if iz % 2 == 0:
                        x_idx = ix
                    else:
                        x_idx = num_x - 1 - ix
                    
                    x = bounds['x_min'] + x_idx * grid_spacing
                    z = bounds['z_min'] + iz * grid_spacing
                    
                    if (bounds['x_min'] <= x <= bounds['x_max'] and 
                        bounds['z_min'] <= z <= bounds['z_max']):
                        grid_points.append([x, base_y, z])
            
            grid_points = np.array(grid_points, dtype=np.float32)
            print(f"[PATH] Generated {len(grid_points)} grid points for exploration")
            
            # Interpolate along grid points
            num_path_points = min(len(grid_points), n_frames)
            path_indices = np.linspace(0, len(grid_points) - 1, num_path_points, dtype=int)
            exploration_path = grid_points[path_indices]
            
            # Add start position at beginning
            exploration_path = np.vstack([start_pos.reshape(1, 3), exploration_path])
            
            for i in range(n_frames):
                t = i / n_frames
                
                if len(exploration_path) > 1:
                    path_idx_float = t * (len(exploration_path) - 1)
                    path_idx = int(path_idx_float)
                    path_idx_next = min(path_idx + 1, len(exploration_path) - 1)
                    alpha = path_idx_float - path_idx
                    
                    pos = (1 - alpha) * exploration_path[path_idx] + alpha * exploration_path[path_idx_next]
                    
                    # Add smooth variation
                    variation_radius = 0.3
                    variation_angle = t * math.pi * 6
                    pos[0] += variation_radius * math.cos(variation_angle)
                    pos[2] += variation_radius * math.sin(variation_angle)
                    pos[1] = base_y + 0.05 * math.sin(t * math.pi * 4)
                else:
                    pos = exploration_path[0].copy()
                    pos[1] = base_y
                
                # Ensure collision-free
                pos = find_nearest_free_position(pos, kdtree, space_info, preserve_y=True)
                waypoints.append(pos)
        else:
            # No main room: use spiral exploration
            building_x_min = space_info['pmin'][0] + 1.5
            building_x_max = space_info['pmax'][0] - 1.5
            building_z_min = space_info['pmin'][2] + 1.5
            building_z_max = space_info['pmax'][2] - 1.5
            
            for i in range(n_frames):
                t = i / n_frames
                
                # Spiral exploration pattern
                angle = t * math.pi * 6  # Multiple rotations
                radius = exploration_radius * (0.2 + 0.6 * t)  # Expand over time
                x = start_pos[0] + radius * math.cos(angle)
                z = start_pos[2] + radius * math.sin(angle)
                
                x = np.clip(x, building_x_min, building_x_max)
                z = np.clip(z, building_z_min, building_z_max)
                y = base_y + 0.1 * math.sin(t * math.pi * 4)
                
                pos = np.array([x, y, z], dtype=np.float32)
                pos = find_nearest_free_position(pos, kdtree, space_info, preserve_y=True)
                waypoints.append(pos)
    
    waypoints = np.array(waypoints, dtype=np.float32)
    
    # Smooth the path (but less aggressively to avoid moving into walls)
    waypoints = smooth_path(waypoints, sigma=3.0)  # Reduced from 5.0
    
    # Aggressive collision check and adjustment - iterate multiple times
    print(f"[PATH] Applying collision avoidance to {len(waypoints)} waypoints...")
    max_collision_iterations = 5
    for iteration in range(max_collision_iterations):
        collision_count = 0
        collision_indices = []
        for i in range(len(waypoints)):
            if not is_collision_free(waypoints[i], kdtree):
                collision_count += 1
                collision_indices.append(i)
                # More aggressive push away from obstacles
                waypoints[i] = find_nearest_free_position(waypoints[i], kdtree, space_info, 
                                                        max_iterations=50, preserve_y=True)
        
        if collision_count == 0:
            print(f"[PATH] ✓ All waypoints are collision-free after {iteration + 1} iterations")
            break
        
        print(f"[PATH] Fixed {collision_count} collision points in iteration {iteration + 1}")
        
        # After fixing collisions, smooth again (but only non-collision points influence)
        if iteration < max_collision_iterations - 1:
            # Re-smooth, but preserve collision-free positions
            smoothed = smooth_path(waypoints, sigma=2.0)
            # Only update positions that were not in collision
            for i in range(len(waypoints)):
                if i not in collision_indices:
                    # Check if smoothed position is still collision-free
                    if is_collision_free(smoothed[i], kdtree):
                        waypoints[i] = smoothed[i]
    
    # Final verification and one more pass if needed
    final_collisions = []
    for i in range(len(waypoints)):
        if not is_collision_free(waypoints[i], kdtree):
            final_collisions.append(i)
            # Last resort: push even harder
            waypoints[i] = find_nearest_free_position(waypoints[i], kdtree, space_info, 
                                                    max_iterations=100, preserve_y=True)
    
    if len(final_collisions) > 0:
        print(f"[PATH] ⚠ Warning: {len(final_collisions)} waypoints still have collisions after all adjustments")
        print(f"[PATH]   These will be pushed away from walls, but may cause slight path deviation")
    else:
        print(f"[PATH] ✓ All waypoints are collision-free")
    
    return waypoints


def smooth_path(path: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Smooth the path using Gaussian filtering."""
    smoothed = path.copy()
    for dim in range(3):
        smoothed[:, dim] = gaussian_filter1d(path[:, dim], sigma=sigma, mode='nearest')
    return smoothed


# ===================== CAMERA ORIENTATION =====================

def compute_cinematic_orientation(positions: np.ndarray,
                                  space_info: dict,
                                  kdtree: cKDTree,
                                  look_ahead: int = 5) -> np.ndarray:
    """
    Compute cinematic camera orientations that:
    - Look ahead along the path
    - Dynamically pan left and right for natural head movements
    - Occasionally look at interesting features
    - Maintain smooth, natural head movements
    """
    n = len(positions)
    quaternions = np.zeros((n, 4), dtype=np.float32)
    center = space_info['center']
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    for i in range(n):
        pos = positions[i]
        t = i / n  # Normalized time for periodic movements
        
        # Primary look direction: along path
        if i + look_ahead < n:
            look_target = positions[i + look_ahead]
        elif i < n - 1:
            look_target = positions[i + 1]
        else:
            look_target = positions[i - 1]
        
        # Compute base forward direction along path
        forward_base = look_target - pos
        forward_norm = np.linalg.norm(forward_base)
        
        if forward_norm < 1e-6:
            forward_base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            forward_base = forward_base / forward_norm
        
        # Build right vector for horizontal panning
        right_base = np.cross(up, forward_base)
        right_norm = np.linalg.norm(right_base)
        if right_norm < 1e-6:
            right_base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right_base = right_base / right_norm
        
        # Dynamic horizontal panning (left/right)
        # Use multiple sine waves for natural, non-repetitive movement
        pan_angle = 0.0
        pan_angle += 0.4 * math.sin(t * math.pi * 2.3)  # Slow pan
        pan_angle += 0.25 * math.sin(t * math.pi * 5.7)  # Medium pan
        pan_angle += 0.15 * math.sin(t * math.pi * 11.3)  # Fast pan
        pan_angle = np.clip(pan_angle, -0.6, 0.6)  # Limit to ~35 degrees max
        
        # Rotate forward direction horizontally
        cos_pan = math.cos(pan_angle)
        sin_pan = math.sin(pan_angle)
        forward = forward_base * cos_pan + right_base * sin_pan
        
        # Blend with center of room for more cinematic feel (reduced blend)
        blend_factor = 0.15  # Reduced from 0.3 to allow more dynamic movement
        center_direction = center - pos
        center_norm = np.linalg.norm(center_direction)
        if center_norm > 1e-6:
            center_direction = center_direction / center_norm
            # Blend forward direction with center direction
            forward = (1 - blend_factor) * forward + blend_factor * center_direction
            forward = forward / np.linalg.norm(forward)
        
        # Occasional "look around" movements - add vertical variation
        vertical_tilt = 0.1 * math.sin(t * math.pi * 3.1)  # Subtle up/down
        vertical_tilt = np.clip(vertical_tilt, -0.3, 0.3)  # Limit vertical tilt
        
        # Apply vertical tilt
        forward[1] += vertical_tilt
        
        # Limit vertical tilt for cinematic feel (avoid looking straight up/down)
        forward[1] = np.clip(forward[1], -0.5, 0.5)
        forward = forward / np.linalg.norm(forward)
        
        # Build camera frame
        right = np.cross(up, forward)
        right_norm = np.linalg.norm(right)
        
        if right_norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right = right / right_norm
        
        true_up = np.cross(forward, right)
        true_up_norm = np.linalg.norm(true_up)
        if true_up_norm < 1e-6:
            true_up = up
        else:
            true_up = true_up / true_up_norm
        
        # Build rotation matrix (camera space: right, up, -forward)
        rot_mat = np.column_stack([right, true_up, -forward])
        
        # Ensure right-handed coordinate system (positive determinant)
        det = np.linalg.det(rot_mat)
        if det < 0:
            # Flip right vector to fix handedness
            right = -right
            rot_mat = np.column_stack([right, true_up, -forward])
        
        # Convert to quaternion
        r = R.from_matrix(rot_mat)
        quat = r.as_quat()  # [x, y, z, w]
        quaternions[i] = quat.astype(np.float32)
    
    # Smooth quaternions (reduced smoothing to preserve dynamic movement)
    quaternions = smooth_quaternions(quaternions, sigma=5.0)  # Reduced from 8.0
    
    return quaternions


def smooth_quaternions(quats: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Smooth quaternion sequence."""
    q = quats.copy().astype(np.float64)
    
    # Ensure shortest path (avoid sign flips)
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0.0:
            q[i] = -q[i]
    
    # Smooth each component
    for c in range(4):
        q[:, c] = gaussian_filter1d(q[:, c], sigma=sigma, mode='nearest')
    
    # Renormalize
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    q = q / norms
    
    return q.astype(np.float32)


# ===================== EXPORT =====================

def transform_coordinates_for_viewer(pos: np.ndarray, quat: np.ndarray, 
                                     transform_mode: str = "flip_yz") -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform coordinates to match the viewer's coordinate system.
    
    The viewer rotates the model by Math.PI around X-axis (mesh.rotation.x = Math.PI).
    
    transform_mode options:
    - "flip_yz": Flip both Y and Z: (x, y, z) -> (x, -y, -z) [default]
    - "flip_z_only": Only flip Z: (x, y, z) -> (x, y, -z) [for upside-down fix]
    - "none": No position transformation
    
    For quaternions, we rotate by PI around X to match model rotation.
    If rooms appear upside down, try setting transform_mode to "flip_z_only".
    """
    from scipy.spatial.transform import Rotation as R
    
    # Transform position based on mode
    if transform_mode == "flip_z_only":
        # Only flip Z - this should fix upside-down while keeping Y correct
        pos_transformed = np.array([pos[0], pos[1], -pos[2]], dtype=np.float32)
        # Don't transform quaternion - this prevents upside-down rendering
        quat_transformed = quat.copy()
    elif transform_mode == "none":
        # No transformation
        pos_transformed = pos.copy()
        quat_transformed = quat.copy()
    else:  # "flip_yz" (default)
        # Transform position: flip Y and Z to match model rotation
        # The model is rotated by PI around X, so we flip Y and Z
        pos_transformed = np.array([pos[0], -pos[1], -pos[2]], dtype=np.float32)
        
        # DON'T transform quaternion - this causes upside-down rendering
        # The model rotation is just for display, camera should use original orientation
        quat_transformed = quat.copy()
        
        # Alternative: if you need quaternion transform, uncomment below:
        # r_current = R.from_quat([quat[0], quat[1], quat[2], quat[3]])
        # r_flip = R.from_euler('x', np.pi)
        # r_final = r_flip * r_current
        # quat_transformed = r_final.as_quat()
    
    return pos_transformed, quat_transformed


def export_trajectory_json(positions: np.ndarray,
                           quaternions: np.ndarray,
                           output_path: str,
                           fov: float = 60.0,
                           transform_coords: bool = True,
                           transform_mode: str = "none",
                           y_offset: float = 0.0) -> None:
    """Export trajectory in format compatible with index.html."""
    assert len(positions) == len(quaternions)
    
    frames = []
    for i in range(len(positions)):
        pos = positions[i]
        quat = quaternions[i]
        
        # Transform coordinates to match viewer's coordinate system
        if transform_coords:
            pos_transformed, quat_transformed = transform_coordinates_for_viewer(pos, quat, transform_mode=transform_mode)
        else:
            pos_transformed = pos
            quat_transformed = quat
        
        # Apply Y offset if camera is under the map
        # This adjusts the height to account for coordinate system differences
        pos_transformed[1] += y_offset
        
        frames.append({
            "position": {
                "x": float(pos_transformed[0]),
                "y": float(pos_transformed[1]),
                "z": float(pos_transformed[2])
            },
            "quaternion": {
                "x": float(quat_transformed[0]),
                "y": float(quat_transformed[1]),
                "z": float(quat_transformed[2]),
                "w": float(quat_transformed[3])
            },
            "fov": fov
        })
    
    with open(output_path, 'w') as f:
        json.dump(frames, f, indent=2)
    
    print(f"[EXPORT] Saved {len(frames)} frames to {output_path}")


# ===================== OBJECT DETECTION =====================

def classify_furniture_3d(points: np.ndarray, size: Tuple[float, float, float], 
                         floor_y: float, ceiling_y: float) -> str:
    """
    Classify indoor furniture based on geometric properties.
    
    Args:
        points: (N, 3) points belonging to object
        size: (width, height, depth) tuple
        floor_y: Floor Y coordinate
        ceiling_y: Ceiling Y coordinate
    
    Returns:
        Furniture class name
    """
    width, height, depth = size
    room_height = ceiling_y - floor_y
    
    # Normalize height relative to room (for better classification)
    height_ratio = height / room_height if room_height > 0 else 0
    
    # Calculate aspect ratios
    aspect_ratio_hw = height / width if width > 0.01 else 0
    aspect_ratio_hd = height / depth if depth > 0.01 else 0
    aspect_ratio_wd = width / depth if depth > 0.01 else 0
    
    # Podium/Stand: Tall and narrow (height > 0.3m, narrow footprint)
    if height > 0.3 and (aspect_ratio_hw > 2.0 or aspect_ratio_hd > 2.0) and width < 1.5 and depth < 1.5:
        return "podium"
    
    # Screen/Display: Very tall and narrow
    if height > 0.5 and (aspect_ratio_hw > 3.0 or aspect_ratio_hd > 3.0):
        return "screen"
    
    # Table: Low height (0.5-1.0m), large horizontal footprint
    if 0.3 < height < 1.2 and width > 0.8 and depth > 0.8 and aspect_ratio_hw < 0.5:
        # Check if it's a large conference table
        if width > 2.0 or depth > 2.0:
            return "conference_table"
        return "table"
    
    # Chair: Low to medium height (0.3-1.2m), small footprint
    if 0.2 < height < 1.3 and width < 1.0 and depth < 1.0:
        # Chairs are typically taller than they are wide
        if aspect_ratio_hw > 0.8 or aspect_ratio_hd > 0.8:
            return "chair"
        # Low and wide might be a stool
        if height < 0.6:
            return "stool"
        return "chair"
    
    # Desk: Low height, wide and deep
    if 0.4 < height < 1.0 and width > 1.0 and depth > 0.6:
        return "desk"
    
    # Cabinet/Shelf: Medium height, narrow depth
    if 0.8 < height < 2.5 and depth < 0.8:
        return "cabinet"
    
    # Generic furniture: medium height objects
    if 0.3 < height < 2.0:
        return "furniture"
    
    # Low objects (might be floor items)
    if height < 0.3:
        return "floor_item"
    
    return "unknown"


def run_3d_object_detection(pts: np.ndarray,
                           space_info: dict,
                           output_dir: str = "detections") -> Optional[List[Detection3D]]:
    """
    Run 3D object detection on the indoor point cloud.
    
    Detects furniture items like chairs, tables, podiums, screens, etc.
    in the indoor scene using geometric clustering.
    """
    if not HAS_OBJECT_DETECTION:
        print("[WARN] Object detection not available. Skipping 3D detection.")
        return None
    
    print("[3D DETECT] Starting 3D furniture detection on indoor scene...")
    
    # Filter points to interior space (exclude floor/ceiling)
    floor_y = space_info['floor_y']
    ceiling_y = space_info['ceiling_y']
    height_margin = (ceiling_y - floor_y) * 0.05  # Smaller margin for furniture
    
    # Focus on interior objects (between floor and ceiling)
    interior_mask = (pts[:, 1] >= floor_y + height_margin) & (pts[:, 1] <= ceiling_y - height_margin)
    interior_pts = pts[interior_mask]
    
    if len(interior_pts) < 100:
        print("[3D DETECT] Not enough interior points, using all points")
        interior_pts = pts
    
    print(f"[3D DETECT] Analyzing {len(interior_pts)} interior points...")
    print(f"[3D DETECT] Room height: {ceiling_y - floor_y:.2f}m (floor: {floor_y:.2f}, ceiling: {ceiling_y:.2f})")
    
    # Use geometric clustering optimized for furniture
    # Furniture is typically smaller and more numerous than buildings
    detections_raw = detect_objects_3d_geometric(
        interior_pts,
        min_height=floor_y + height_margin,
        max_height=ceiling_y - height_margin,
        eps=0.1,  # Smaller for furniture clustering
        min_samples=20,  # Lower threshold for smaller furniture items
        min_size=0.15,  # Minimum 15cm (small furniture items)
        max_size=5.0  # Maximum 5m (large conference tables)
    )
    
    # Reclassify detections as furniture instead of buildings
    detections = []
    for det in detections_raw:
        # Reclassify using furniture-specific logic
        furniture_class = classify_furniture_3d(
            det.points, 
            det.size, 
            floor_y, 
            ceiling_y
        )
        
        # Create new detection with furniture classification
        from object_detection import Detection3D
        furniture_det = Detection3D(
            object_id=det.object_id,
            class_name=furniture_class,
            confidence=det.confidence,
            center_3d=det.center_3d,
            bbox_3d=det.bbox_3d,
            points=det.points,
            size=det.size
        )
        detections.append(furniture_det)
    
    # Filter out "unknown" and "terrain" classifications (not furniture)
    detections = [d for d in detections if d.class_name not in ["unknown", "terrain", "building"]]
    
    print(f"[3D DETECT] Detected {len(detections)} furniture items")
    
    # Count by type
    from collections import Counter
    class_counts = Counter(d.class_name for d in detections)
    print(f"[3D DETECT] Detection breakdown:")
    for class_name, count in class_counts.most_common():
        print(f"  - {class_name}: {count}")
    
    # Show details for first 15 detections
    for det in detections[:15]:
        print(f"  - {det.class_name} (ID: {det.object_id}, conf: {det.confidence:.2f}, "
              f"size: {det.size[0]:.2f}x{det.size[1]:.2f}x{det.size[2]:.2f}m, "
              f"center: [{det.center_3d[0]:.2f}, {det.center_3d[1]:.2f}, {det.center_3d[2]:.2f}])")
    
    # Visualize
    if len(detections) > 0:
        os.makedirs(output_dir, exist_ok=True)
        viz_path = os.path.join(output_dir, "detections_3d_indoor.png")
        try:
            visualize_detections_3d(interior_pts, detections, viz_path)
        except Exception as e:
            print(f"[3D DETECT] Visualization failed: {e}")
    
    return detections


def run_2d_object_detection(image_dir: Optional[str],
                            trajectory_path: str,
                            output_dir: str,
                            positions: np.ndarray,
                            quaternions: np.ndarray,
                            fov: float) -> Optional[Dict[int, List[Detection2D]]]:
    """
    Run 2D object detection on rendered video frames.
    
    Requires rendered frames from the trajectory to be saved in image_dir.
    """
    if not HAS_OBJECT_DETECTION:
        print("[WARN] Object detection not available. Skipping 2D detection.")
        return None
    
    if image_dir is None or not os.path.exists(image_dir):
        print("[2D DETECT] No image directory provided or directory doesn't exist.")
        print("[2D DETECT] To use 2D detection:")
        print("  1. Render frames from trajectory using your 3DGS renderer")
        print("  2. Save frames to a directory (e.g., 'frames/')")
        print("  3. Run with --detect-2d --image-dir frames/")
        return None
    
    print(f"[2D DETECT] Looking for images in: {image_dir}")
    
    # Find all images
    from pathlib import Path
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(image_dir).glob(f'*{ext}'))
        image_paths.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    
    image_paths = sorted([str(p) for p in image_paths])
    
    if len(image_paths) == 0:
        print(f"[2D DETECT] No images found in {image_dir}")
        return None
    
    print(f"[2D DETECT] Found {len(image_paths)} images")
    print(f"[2D DETECT] Running YOLO detection...")
    
    # Run batch detection
    detections_2d = detect_objects_2d_batch(
        image_paths,
        model_name="yolov8n.pt",
        confidence_threshold=0.25
    )
    
    total_detections = sum(len(dets) for dets in detections_2d.values())
    print(f"[2D DETECT] Total detections: {total_detections} across {len(detections_2d)} frames")
    
    # Show sample detections
    sample_count = 0
    for frame_id, dets in detections_2d.items():
        if len(dets) > 0 and sample_count < 5:
            print(f"  Frame {frame_id}: {len(dets)} objects")
            for det in dets[:3]:  # Show first 3 per frame
                print(f"    - {det.class_name} (conf: {det.confidence:.2f})")
            sample_count += 1
    
    # Visualize sample frames
    if len(detections_2d) > 0:
        os.makedirs(output_dir, exist_ok=True)
        viz_count = 0
        for frame_id in sorted(detections_2d.keys()):
            if len(detections_2d[frame_id]) > 0 and viz_count < 5:
                viz_path = os.path.join(output_dir, f"detections_2d_frame_{frame_id}.png")
                try:
                    visualize_detections_2d(image_paths[frame_id], 
                                           detections_2d[frame_id], 
                                           viz_path)
                    viz_count += 1
                except Exception as e:
                    print(f"[2D DETECT] Visualization failed for frame {frame_id}: {e}")
    
    return detections_2d


# ===================== MAIN PIPELINE =====================

def generate_indoor_trajectory(ply_file: str,
                              start_pos: Optional[np.ndarray],
                              output_json: str,
                              n_frames: int = 600,
                              fov: float = 60.0,
                              auto_detect_start: bool = True,
                              detect_objects_3d: bool = False,
                              detect_objects_2d: bool = False,
                              image_dir: Optional[str] = None,
                              output_dir: str = "detections") -> None:
    """Main pipeline for generating indoor navigation trajectory."""
    print("=" * 60)
    print("Indoor Navigation Path Generator")
    print("=" * 60)
    
    # 1. Load and clean point cloud
    print("\n[1/5] Loading point cloud...")
    pts = load_ply_points(ply_file)
    pts = remove_outliers(pts)
    
    # 2. Analyze indoor space
    print("\n[2/5] Analyzing indoor space...")
    space_info = analyze_indoor_space(pts)
    
    # 3. Build free space map and collision detection
    print("\n[3/5] Building free space map and collision detection...")
    free_space_map, void_centers, kdtree = build_free_space_map(pts, space_info, voxel_size=0.2)
    height_map, _ = build_occupancy_grid(pts, space_info)
    
    # 3.5. Auto-detect start position if not provided or if requested
    main_room = None
    debug_mode = True  # Enable debugging
    
    if start_pos is None or auto_detect_start:
        print("\n[3.5/5] Auto-detecting start position...")
        rooms = detect_interior_rooms(pts, space_info, debug=debug_mode)
        start_pos, main_room = find_best_start_position(pts, space_info, kdtree, rooms=rooms, debug=debug_mode)
        print(f"[AUTO] Using auto-detected start: {start_pos}")
        
        # Visualize for debugging
        if debug_mode and main_room is not None:
            try:
                visualize_room_detection(pts, rooms, start_pos, space_info, 
                                        output_path="debug_room_detection.png")
            except Exception as e:
                print(f"[DEBUG] Visualization failed: {e}")
    else:
        print(f"[PATH] Using provided start position: {start_pos}")
        
        # Still verify provided position
        if debug_mode:
            print(f"[DEBUG] Verifying provided start position: {start_pos}")
            # Check if it's collision-free
            is_free = is_collision_free(start_pos, kdtree)
            print(f"[DEBUG] Collision-free: {is_free}")
            if not is_free:
                print(f"[DEBUG] ⚠ Position is too close to obstacles, adjusting...")
                start_pos = find_nearest_free_position(start_pos, kdtree, space_info, preserve_y=True)
                print(f"[DEBUG] Adjusted position: {start_pos}")
    
    # 4. Generate path following void space
    print("\n[4/5] Generating exploration path through void space...")
    positions = generate_indoor_exploration_path(
        start_pos, space_info, kdtree, n_frames=n_frames, main_room=main_room,
        void_centers=void_centers
    )
    
    # 5. Compute camera orientations
    print("\n[5/5] Computing cinematic camera orientations...")
    quaternions = compute_cinematic_orientation(positions, space_info, kdtree)
    
    # 6. Export
    print("\n[EXPORT] Saving trajectory...")
    # Coordinate transformation mode:
    # - "flip_yz": Flip both Y and Z (matches model rotation by PI around X)
    # - "flip_z_only": Only flip Z (keeps Y as-is)
    # - "none": No transformation (camera coordinates match PLY coordinates directly)
    # 
    # Y offset: Add this to Y coordinate AFTER transformation
    # With "flip_yz" mode, Y gets flipped (-1.9 becomes +1.9), so we might not need offset
    # If camera is still under map, try small positive offset (0.5 to 1.0)
    # If camera is above map, try negative offset
    y_offset = 3.0  # Start with 0.0, adjust if needed
    
    # Use "flip_z_only" to fix upside-down issue - only flip Z, keep Y and quaternion unchanged
    # This should prevent upside-down rendering while keeping camera at correct height
    export_trajectory_json(positions, quaternions, output_json, fov=fov, 
                          transform_mode="flip_z_only", y_offset=y_offset)
    
    # 7. Object Detection (optional)
    detections_3d = None
    detections_2d = None
    
    if detect_objects_3d and HAS_OBJECT_DETECTION:
        print("\n" + "=" * 60)
        print("[7/7] Running 3D Object Detection...")
        print("=" * 60)
        detections_3d = run_3d_object_detection(pts, space_info, output_dir)
    
    if detect_objects_2d and HAS_OBJECT_DETECTION:
        print("\n" + "=" * 60)
        print("[7/7] Running 2D Object Detection (Rendered Video)...")
        print("=" * 60)
        detections_2d = run_2d_object_detection(
            image_dir, output_json, output_dir, 
            positions, quaternions, fov
        )
    
    # Export detection results
    if (detections_3d is not None or detections_2d is not None) and HAS_OBJECT_DETECTION:
        os.makedirs(output_dir, exist_ok=True)
        detection_json_path = os.path.join(output_dir, "detections_indoor.json")
        export_detections_json(
            detections_2d=detections_2d,
            detections_3d=detections_3d,
            output_path=detection_json_path
        )
        print(f"[DETECT] Detection results saved to {detection_json_path}")
    
    print("\n" + "=" * 60)
    print("✓ Trajectory generation complete!")
    print(f"  Output: {output_json}")
    print(f"  Frames: {len(positions)}")
    print(f"  Start: {start_pos}")
    if detections_3d is not None:
        print(f"  3D Detections: {len(detections_3d)} objects")
    if detections_2d is not None:
        total_2d = sum(len(dets) for dets in detections_2d.values())
        print(f"  2D Detections: {total_2d} detections across {len(detections_2d)} frames")
    print("=" * 60)


def load_config(config_path: str = "core/config.json") -> Optional[dict]:
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"[CONFIG] Loaded configuration from {config_path}")
        return config
    except Exception as e:
        print(f"[WARN] Failed to load config: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate indoor navigation path for 3DGS models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config file (easiest)
  python core/generate_indoor_path.py --config
  
  # Simple launcher (easiest)
  python core/run_path_generator.py
  
  # Command line
  python core/generate_indoor_path.py --ply scenes/conference-hall/ConferenceHall.ply \\
                                      --output test.json --frames 60
        """
    )
    parser.add_argument("--config", type=str, nargs='?', const="core/config.json", default=None,
                       help="Path to config JSON file (default: core/config.json if flag provided)")
    parser.add_argument("--scene", type=str, default=None,
                       help="Scene name from config file")
    parser.add_argument("--ply", type=str, default=None,
                       help="Path to PLY file (overrides config)")
    parser.add_argument("--start-x", type=float, default=None,
                       help="Start X coordinate (overrides config)")
    parser.add_argument("--start-y", type=float, default=None,
                       help="Start Y coordinate (overrides config)")
    parser.add_argument("--start-z", type=float, default=None,
                       help="Start Z coordinate (overrides config)")
    parser.add_argument("--auto-start", action="store_true", default=None,
                       help="Automatically detect best start position")
    parser.add_argument("--no-auto-start", action="store_false", dest="auto_start",
                       help="Disable automatic start position detection")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path (overrides config)")
    parser.add_argument("--frames", type=int, default=None,
                       help="Number of frames (overrides config)")
    parser.add_argument("--fov", type=float, default=None,
                       help="Field of view in degrees (overrides config)")
    parser.add_argument("--detect-3d", action="store_true",
                       help="Enable 3D object detection on point cloud")
    parser.add_argument("--detect-2d", action="store_true",
                       help="Enable 2D object detection on rendered video frames")
    parser.add_argument("--image-dir", type=str, default=None,
                       help="Directory containing rendered frames (required for 2D detection)")
    parser.add_argument("--detection-output", type=str, default="detections",
                       help="Output directory for detection results")
    
    args = parser.parse_args()
    
    # Try to load config
    config = None
    config_path = args.config if args.config else "core/config.json"
    if os.path.exists(config_path):
        config = load_config(config_path)
    
    # Determine scene configuration
    scene_config = None
    if config and "scenes" in config:
        scene_name = args.scene if args.scene else config.get("default_scene")
        if scene_name and scene_name in config["scenes"]:
            scene_config = config["scenes"][scene_name]
            print(f"[CONFIG] Using scene: {scene_name}")
    
    # Merge config with command line args (CLI args take precedence)
    ply_file = args.ply or (scene_config.get("ply") if scene_config else None)
    output_file = args.output or (scene_config.get("output") if scene_config else None)
    n_frames = args.frames or (scene_config.get("frames", 600) if scene_config else 600)
    fov = args.fov or (scene_config.get("fov", 60.0) if scene_config else 60.0)
    
    # Handle start position
    start_pos_config = scene_config.get("start_position", {}) if scene_config else {}
    auto_detect = args.auto_start
    if auto_detect is None:
        auto_detect = start_pos_config.get("auto_detect", True)
    
    # Get start coordinates (CLI args override config)
    start_x = args.start_x
    start_y = args.start_y
    start_z = args.start_z
    
    if start_x is None and start_pos_config:
        start_x = start_pos_config.get("x")
    if start_y is None and start_pos_config:
        start_y = start_pos_config.get("y")
    if start_z is None and start_pos_config:
        start_z = start_pos_config.get("z")
    
    # Validate required arguments
    if not ply_file:
        parser.error("--ply is required (or provide --config with scene configuration)")
    if not output_file:
        parser.error("--output is required (or provide --config with scene configuration)")
    
    # Determine start position
    # If any coordinate is provided, use it; otherwise auto-detect everything
    if start_x is not None or start_y is not None or start_z is not None:
        # User provided at least one coordinate
        start_x = start_x if start_x is not None else 0.0
        start_y = start_y if start_y is not None else None
        start_z = start_z if start_z is not None else 0.0
        
        # Auto-detect Y if not provided
        if start_y is None:
            try:
                pts = load_ply_points(ply_file)
                floor_y = float(np.percentile(pts[:, 1], 5))
                room_height = float(np.percentile(pts[:, 1], 95)) - floor_y
                if room_height > 0:
                    scale = room_height / 2.5
                    start_y = floor_y + 1.6 * scale
                else:
                    start_y = floor_y + room_height * 0.4
                print(f"[AUTO] Detected start Y: {start_y:.3f}")
            except Exception as e:
                print(f"[WARN] Could not auto-detect Y, using 0.0: {e}")
                start_y = 0.0
        
        start_pos = np.array([start_x, start_y, start_z], dtype=np.float32)
        auto_detect = False  # Don't override user coordinates
    else:
        # No coordinates provided - fully auto-detect
        start_pos = None
        if auto_detect is None:
            auto_detect = True
        if auto_detect:
            print("[AUTO] No start coordinates provided, will auto-detect best position")
    
    generate_indoor_trajectory(
        ply_file=ply_file,
        start_pos=start_pos,
        output_json=output_file,
        n_frames=n_frames,
        fov=fov,
        auto_detect_start=auto_detect,
        detect_objects_3d=args.detect_3d,
        detect_objects_2d=args.detect_2d,
        image_dir=args.image_dir,
        output_dir=args.detection_output
    )


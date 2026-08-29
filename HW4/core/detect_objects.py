"""
Main script for object detection in 3DGS scenes.

Usage:
    python core/detect_objects.py --ply outdoor-standard.ply --trajectory trajectory_autonomous.json
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from new_generator import load_ply_xyz, detect_island_roi
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


def load_trajectory(trajectory_path: str):
    """Load trajectory JSON"""
    with open(trajectory_path, 'r') as f:
        data = json.load(f)
    return data


def detect_objects_3d_pipeline(ply_file: str,
                                output_dir: str = "detections",
                                visualize: bool = True):
    """
    Complete 3D object detection pipeline.
    
    1. Load point cloud
    2. Detect ROI
    3. Run 3D object detection
    4. Visualize and export results
    """
    print("=" * 60)
    print("3D Object Detection Pipeline")
    print("=" * 60)
    
    # 1. Load point cloud
    print("\n[1/4] Loading point cloud...")
    all_points = load_ply_xyz(ply_file)
    
    # 2. Detect ROI
    print("\n[2/4] Detecting ROI...")
    roi, roi_metadata = detect_island_roi(
        all_points,
        eps=0.08,
        min_samples=100,
        extra_radius_scale=1.2,
        visualize=False
    )
    
    # 3. Run 3D detection methods (focus on buildings)
    print("\n[3/4] Running 3D building detection...")
    
    # Primary method: Height-based detection (best for buildings)
    print("  - Height-based building detection...")
    detections_height = detect_objects_3d_height_based(
        roi,
        height_threshold=None,  # Auto (70th percentile)
        n_clusters=25  # Try to discover many separate roofs
    )
    
    # Secondary method: Geometric clustering (for additional structures)
    print("  - Geometric clustering (additional structures)...")
    detections_geometric = detect_objects_3d_geometric(
        roi,
        eps=0.05,          # Smaller eps for normalized coordinates
        min_samples=15,    # Lower threshold to catch smaller buildings
        min_size=0.05,     # Much smaller for normalized coords
        max_size=5.0       # Adjusted for normalized coords
    )
    
    # Filter geometric detections to prioritize buildings
    building_detections = [d for d in detections_geometric if d.class_name in ["building", "structure"]]
    
    # Combine detections (remove duplicates with smaller threshold for normalized coords)
    # Use a small threshold (0.1) so nearby buildings aren't merged
    all_detections_3d = combine_3d_detections(detections_height, building_detections, distance_threshold=0.1)
    
    # Filter to only buildings/structures
    building_only = [d for d in all_detections_3d if d.class_name in ["building", "structure"]]
    if len(building_only) > 0:
        all_detections_3d = building_only
        print(f"  - Filtered to {len(all_detections_3d)} buildings/structures")
    else:
        # If combine removed everything, just use height-based detections
        print(f"  - [WARN] combine_3d_detections removed all buildings, using height-based only")
        all_detections_3d = detections_height
    
    print(f"\n[3D DETECT] Total objects detected: {len(all_detections_3d)}")
    for det in all_detections_3d:
        print(f"  - {det.class_name} (ID: {det.object_id}, conf: {det.confidence:.2f})")
    
    # 4. Visualize and export
    print("\n[4/4] Exporting results...")
    os.makedirs(output_dir, exist_ok=True)
    
    if visualize:
        viz_path = os.path.join(output_dir, "detections_3d.png")
        visualize_detections_3d(roi, all_detections_3d, viz_path)
    
    json_path = os.path.join(output_dir, "detections_3d.json")
    export_detections_json(detections_3d=all_detections_3d, output_path=json_path)
    
    return all_detections_3d, roi


def detect_objects_2d_pipeline(trajectory_path: str,
                               image_dir: str = None,
                               output_dir: str = "detections",
                               model_name: str = "yolov8n.pt",
                               visualize: bool = True):
    """
    Complete 2D object detection pipeline.
    
    Requires rendered frames from the trajectory.
    For now, this is a placeholder - you'll need to render frames first.
    """
    print("=" * 60)
    print("2D Object Detection Pipeline")
    print("=" * 60)
    
    if image_dir is None:
        print("[WARN] No image directory provided. 2D detection requires rendered frames.")
        print("       To use 2D detection:")
        print("       1. Render frames from trajectory using your 3DGS renderer")
        print("       2. Save frames to a directory")
        print("       3. Run with --image_dir <path>")
        return {}
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(image_dir).glob(f'*{ext}'))
        image_paths.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    
    image_paths = sorted([str(p) for p in image_paths])
    
    if len(image_paths) == 0:
        print(f"[ERROR] No images found in {image_dir}")
        return {}
    
    print(f"\n[1/3] Found {len(image_paths)} images")
    
    # Run detection
    print("\n[2/3] Running YOLO detection...")
    all_detections_2d = detect_objects_2d_batch(
        image_paths,
        model_name=model_name,
        confidence_threshold=0.25
    )
    
    total_detections = sum(len(dets) for dets in all_detections_2d.values())
    print(f"\n[2D DETECT] Total detections: {total_detections}")
    
    # Visualize sample frames
    if visualize:
        print("\n[3/3] Visualizing detections...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize first few frames
        for frame_id in list(all_detections_2d.keys())[:5]:
            if len(all_detections_2d[frame_id]) > 0:
                viz_path = os.path.join(output_dir, f"detections_2d_frame_{frame_id}.png")
                visualize_detections_2d(image_paths[frame_id], 
                                       all_detections_2d[frame_id], 
                                       viz_path)
    
    json_path = os.path.join(output_dir, "detections_2d.json")
    export_detections_json(detections_2d=all_detections_2d, output_path=json_path)
    
    return all_detections_2d


def combine_3d_detections(detections1: List[Detection3D],
                         detections2: List[Detection3D],
                         distance_threshold: float = 0.15) -> List[Detection3D]:  # Adjusted for normalized coords
    """
    Combine two sets of 3D detections, removing duplicates.
    
    Two detections are considered duplicates if their centers are within
    distance_threshold of each other.
    
    Note: distance_threshold is adjusted for normalized coordinates (default 0.15).
    """
    all_detections = detections1 + detections2
    if len(all_detections) == 0:
        return []
    
    # Build KD-tree for fast nearest neighbor search
    centers = np.array([d.center_3d for d in all_detections])
    from scipy.spatial import cKDTree
    tree = cKDTree(centers)
    
    # Find unique detections
    unique_detections = []
    used = set()
    
    for i, det in enumerate(all_detections):
        if i in used:
            continue
        
        # Find nearby detections
        neighbors = tree.query_ball_point(det.center_3d, r=distance_threshold)
        neighbors = [n for n in neighbors if n != i]
        
        # Keep the one with highest confidence
        candidates = [det] + [all_detections[n] for n in neighbors if n not in used]
        best = max(candidates, key=lambda d: d.confidence)
        
        unique_detections.append(best)
        used.add(i)
        used.update(neighbors)
    
    # Reassign object IDs
    for new_id, det in enumerate(unique_detections):
        det.object_id = new_id
    
    return unique_detections


def project_2d_to_3d_pipeline(detections_2d: dict,
                               trajectory_path: str,
                               ply_file: str,
                               output_dir: str = "detections"):
    """
    Project 2D detections to 3D space using camera poses and point cloud.
    """
    print("=" * 60)
    print("2D to 3D Projection Pipeline")
    print("=" * 60)
    
    # Load trajectory
    trajectory = load_trajectory(trajectory_path)
    
    # Load point cloud
    all_points = load_ply_xyz(ply_file)
    roi, _ = detect_island_roi(all_points, visualize=False)
    
    # Project each 2D detection
    detections_3d_from_2d = []
    
    for frame_id, dets_2d in detections_2d.items():
        if len(trajectory) <= frame_id:
            continue
        
        frame_data = trajectory[frame_id]
        cam_pos = np.array([
            frame_data['position']['x'],
            frame_data['position']['y'],
            frame_data['position']['z']
        ])
        cam_quat = np.array([
            frame_data['quaternion']['x'],
            frame_data['quaternion']['y'],
            frame_data['quaternion']['z'],
            frame_data['quaternion']['w']
        ])
        fov = frame_data.get('fov', 65.0)
        
        # Assume image dimensions (adjust if known)
        img_width, img_height = 1920, 1080
        
        for det_2d in dets_2d:
            det_3d = project_2d_to_3d(
                det_2d,
                cam_pos,
                cam_quat,
                fov,
                img_width,
                img_height,
                point_cloud=roi,
                max_depth=100.0
            )
            
            if det_3d is not None:
                detections_3d_from_2d.append(det_3d)
    
    print(f"\n[PROJECT] Projected {len(detections_3d_from_2d)} detections to 3D")
    
    return detections_3d_from_2d


def main():
    parser = argparse.ArgumentParser(description="Object detection for 3DGS scenes")
    parser.add_argument("--ply", type=str, required=True, help="PLY file path")
    parser.add_argument("--trajectory", type=str, help="Trajectory JSON file")
    parser.add_argument("--image_dir", type=str, help="Directory with rendered frames")
    parser.add_argument("--output_dir", type=str, default="detections", help="Output directory")
    parser.add_argument("--mode", type=str, choices=['2d', '3d', 'both'], default='3d',
                       help="Detection mode")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt",
                       help="YOLO model name")
    parser.add_argument("--no_viz", action="store_true", help="Skip visualization")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 3D detection
    if args.mode in ['3d', 'both']:
        detections_3d, roi = detect_objects_3d_pipeline(
            args.ply,
            output_dir=args.output_dir,
            visualize=not args.no_viz
        )
    else:
        detections_3d = []
        roi = None
    
    # 2D detection
    if args.mode in ['2d', 'both']:
        detections_2d = detect_objects_2d_pipeline(
            args.trajectory or "",
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            model_name=args.yolo_model,
            visualize=not args.no_viz
        )
        
        # Project 2D to 3D if trajectory available
        if args.trajectory and len(detections_2d) > 0:
            detections_3d_from_2d = project_2d_to_3d_pipeline(
                detections_2d,
                args.trajectory,
                args.ply,
                args.output_dir
            )
    else:
        detections_2d = {}
    
    print("\n" + "=" * 60)
    print("Detection Complete!")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()

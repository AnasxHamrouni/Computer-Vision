"""
Object Detection Module for 3D Gaussian Splatting Scenes

Provides both 2D (rendered frame) and 3D (point cloud) object detection.
"""

import json
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import cv2
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle as MplCircle

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] YOLO not available. Install with: pip install ultralytics")


@dataclass
class Detection2D:
    """2D detection from rendered frame"""
    frame_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    center_2d: Tuple[float, float]  # pixel coordinates


@dataclass
class Detection3D:
    """3D detection from point cloud"""
    object_id: int
    class_name: str
    confidence: float
    center_3d: np.ndarray  # (3,) xyz
    bbox_3d: Dict[str, np.ndarray]  # min_xyz, max_xyz
    points: np.ndarray  # (N, 3) points belonging to this object
    size: Tuple[float, float, float]  # width, height, depth


@dataclass
class Detection3DFrom2D:
    """3D detection projected from 2D detection"""
    frame_id: int
    detection_2d: Detection2D
    center_3d: np.ndarray
    depth: float
    points_3d: np.ndarray


# =========================
# 2D Object Detection (Rendered Frames)
# =========================

def detect_objects_2d_yolo(image_path: str, 
                          model_name: str = "yolov8n.pt",
                          confidence_threshold: float = 0.25) -> List[Detection2D]:
    """
    Detect objects in a 2D image using YOLO.
    
    Args:
        image_path: Path to image file
        model_name: YOLO model name (yolov8n.pt, yolov8s.pt, etc.)
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        List of Detection2D objects
    """
    if not YOLO_AVAILABLE:
        print("[ERROR] YOLO not available. Install with: pip install ultralytics")
        return []
    
    model = YOLO(model_name)
    results = model(image_path, conf=confidence_threshold)
    
    detections = []
    frame_id = 0  # Assuming single image
    
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            box = boxes[i]
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            detections.append(Detection2D(
                frame_id=frame_id,
                class_name=class_name,
                confidence=conf,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                center_2d=(float(center_x), float(center_y))
            ))
    
    return detections


def detect_objects_2d_batch(image_paths: List[str],
                           model_name: str = "yolov8n.pt",
                           confidence_threshold: float = 0.25) -> Dict[int, List[Detection2D]]:
    """
    Detect objects in multiple images.
    
    Returns:
        Dictionary mapping frame_id -> list of detections
    """
    all_detections = {}
    
    for frame_id, image_path in enumerate(image_paths):
        detections = detect_objects_2d_yolo(image_path, model_name, confidence_threshold)
        # Update frame_id to match index
        for det in detections:
            det.frame_id = frame_id
        all_detections[frame_id] = detections
    
    return all_detections


# =========================
# 3D Object Detection (Point Cloud)
# =========================

def detect_objects_3d_geometric(roi_points: np.ndarray,
                                min_height: float = None,
                                max_height: float = None,
                                eps: float = 0.1,
                                min_samples: int = 50,
                                min_size: float = 0.5,
                                max_size: float = 20.0) -> List[Detection3D]:
    """
    Detect 3D objects using geometric clustering on point cloud.
    
    This method:
    1. Filters points by height (optional)
    2. Clusters points using DBSCAN
    3. Filters clusters by size
    4. Classifies objects based on geometric properties
    
    Args:
        roi_points: (N, 3) point cloud
        min_height: Minimum Y coordinate to consider
        max_height: Maximum Y coordinate to consider
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        min_size: Minimum object size (meters)
        max_size: Maximum object size (meters)
    
    Returns:
        List of Detection3D objects
    """
    # Filter by height if specified
    if min_height is not None or max_height is not None:
        y_coords = roi_points[:, 1]
        mask = np.ones(len(roi_points), dtype=bool)
        if min_height is not None:
            mask &= (y_coords >= min_height)
        if max_height is not None:
            mask &= (y_coords <= max_height)
        filtered_points = roi_points[mask]
    else:
        filtered_points = roi_points
    
    if len(filtered_points) < min_samples:
        print(f"[3D DETECT] Not enough points after filtering: {len(filtered_points)}")
        return []
    
    print(f"[3D DETECT] Clustering {len(filtered_points)} points...")
    
    # Cluster points
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1).fit(filtered_points)
    labels = db.labels_
    
    unique_labels = np.unique(labels[labels >= 0])
    print(f"[3D DETECT] Found {len(unique_labels)} clusters")
    
    detections = []
    
    for obj_id, label in enumerate(unique_labels):
        mask = labels == label
        cluster_points = filtered_points[mask]
        
        # Compute bounding box
        min_xyz = cluster_points.min(axis=0)
        max_xyz = cluster_points.max(axis=0)
        center_3d = cluster_points.mean(axis=0)
        size = (max_xyz[0] - min_xyz[0], 
                max_xyz[1] - min_xyz[1], 
                max_xyz[2] - min_xyz[2])
        
        # Filter by size
        max_dimension = max(size)
        if max_dimension < min_size or max_dimension > max_size:
            continue
        
        # Classify object based on geometry
        class_name = classify_object_3d(cluster_points, size)
        
        detections.append(Detection3D(
            object_id=obj_id,
            class_name=class_name,
            confidence=0.8,  # Geometric detection confidence
            center_3d=center_3d,
            bbox_3d={'min': min_xyz, 'max': max_xyz},
            points=cluster_points,
            size=size
        ))
    
    print(f"[3D DETECT] Detected {len(detections)} objects after size filtering")
    return detections


def classify_object_3d(points: np.ndarray, size: Tuple[float, float, float]) -> str:
    """
    Classify a 3D object based on its geometric properties.
    
    Args:
        points: (N, 3) points belonging to object
        size: (width, height, depth) tuple
    
    Returns:
        Class name string
    """
    width, height, depth = size
    aspect_ratio_hw = height / width if width > 0 else 0
    aspect_ratio_hd = height / depth if depth > 0 else 0
    
    # Building: tall, rectangular (adjusted for normalized coordinates)
    # In normalized space, buildings are typically 0.1-0.5 units tall
    if height > 0.05 and aspect_ratio_hw > 0.3 and aspect_ratio_hw < 3.0:
        return "building"
    
    # Tree: tall, narrow
    if height > 0.1 and (aspect_ratio_hw > 2.0 or aspect_ratio_hd > 2.0):
        return "tree"
    
    # Vehicle: low, wide
    if height < 0.15 and width > 0.1 and depth > 0.1:
        return "vehicle"
    
    # Structure: medium height, regular shape
    if height > 0.03 and height < 0.5:
        return "structure"
    
    # Terrain feature: low, irregular
    if height < 0.1:
        return "terrain"
    
    return "unknown"


def detect_objects_3d_height_based(roi_points: np.ndarray,
                                  height_threshold: float = None,
                                  n_clusters: int = 25,
                                  min_building_height: float = 0.05) -> List[Detection3D]:  # Normalized coords
    """
    Detect buildings by finding high-elevation clusters.
    Optimized for building detection on islands.
    
    Args:
        roi_points: (N, 3) point cloud
        height_threshold: Y coordinate threshold (auto if None, uses 80th percentile)
        n_clusters: Number of top clusters to return
        min_building_height: Minimum height to classify as building (meters)
    
    Returns:
        List of Detection3D objects (buildings/structures)
    """
    # Debug: Print height statistics
    y_coords = roi_points[:, 1]
    y_min, y_max = y_coords.min(), y_coords.max()
    y_mean, y_median = y_coords.mean(), np.median(y_coords)
    y_std = y_coords.std()
    print(f"    [DEBUG] Height stats: min={y_min:.3f}, max={y_max:.3f}, mean={y_mean:.3f}, median={y_median:.3f}, std={y_std:.3f}")
    
    if height_threshold is None:
        # Use 70th percentile to include more mid-height roofs
        height_threshold = np.percentile(roi_points[:, 1], 70)
        print(f"    [DEBUG] Computed 70th percentile threshold: {height_threshold:.3f}")
    
    print(f"    Using height threshold: {height_threshold:.3f} (70th percentile)")
    high_points = roi_points[roi_points[:, 1] >= height_threshold]
    
    if len(high_points) < 10:
        print(f"    [WARN] Not enough high points: {len(high_points)}")
        print(f"    [DEBUG] Trying lower threshold (60th percentile)...")
        height_threshold = np.percentile(roi_points[:, 1], 60)
        high_points = roi_points[roi_points[:, 1] >= height_threshold]
        print(f"    [DEBUG] New threshold {height_threshold:.3f} gives {len(high_points)} points")
        if len(high_points) < 10:
            return []
    
    print(f"    Found {len(high_points)} high-elevation points ({100*len(high_points)/len(roi_points):.1f}% of ROI)")
    
    # Use a much more aggressive clustering approach to separate individual buildings
    # Method: Compute nearest neighbor distances and use a small fraction of that
    xz_high = high_points[:, [0, 2]]
    
    # Build KD-tree to find nearest neighbors
    from scipy.spatial import cKDTree
    tree = cKDTree(xz_high)
    
    # Find nearest neighbor distance for each point (skip self)
    if len(xz_high) > 1:
        distances, _ = tree.query(xz_high, k=min(6, len(xz_high)))  # Get more neighbors for better stats
        nn_dists = distances[:, 1] if distances.shape[1] > 1 else distances[:, 0]  # Nearest neighbor
        # Use median of nearest neighbor distances as base, then scale down
        median_nn_dist = np.median(nn_dists)
        p25_nn_dist = np.percentile(nn_dists, 25)
        p75_nn_dist = np.percentile(nn_dists, 75)
        print(f"    [DEBUG] Nearest neighbor distances: median={median_nn_dist:.4f}, p25={p25_nn_dist:.4f}, p75={p75_nn_dist:.4f}")
        
        # Try multiple eps values
        eps_candidates = [
            median_nn_dist * 1.5,
            median_nn_dist * 2.0,
            p25_nn_dist * 2.0,
            p75_nn_dist * 1.0,
        ]
        eps = eps_candidates[0]  # Start with first
    else:
        # Fallback
        center_xz = xz_high.mean(axis=0)
        dists = np.linalg.norm(xz_high - center_xz[None, :], axis=1)
        eps = np.percentile(dists, 5) * 0.15  # Very small
        print(f"    [DEBUG] Using fallback eps calculation: {eps:.4f}")
    
    print(f"    Clustering with eps={eps:.4f} (based on nearest neighbor distances)...")
    db = DBSCAN(eps=eps, min_samples=5, n_jobs=1).fit(xz_high)  # Lower min_samples for smaller clusters
    labels = db.labels_
    
    unique_labels = np.unique(labels[labels >= 0])
    noise_count = np.sum(labels == -1)
    print(f"    Found {len(unique_labels)} clusters, {noise_count} noise points")
    
    # If still too few clusters, try progressively smaller eps
    if len(unique_labels) < 3 and len(high_points) > 100:
        print(f"    [DEBUG] Too few clusters ({len(unique_labels)}), trying smaller eps values...")
        for factor in [0.7, 0.5, 0.3, 0.2]:
            test_eps = eps * factor
            test_db = DBSCAN(eps=test_eps, min_samples=3, n_jobs=1).fit(xz_high)
            test_labels = test_db.labels_
            test_unique = np.unique(test_labels[test_labels >= 0])
            print(f"      eps={test_eps:.4f} (factor {factor}): {len(test_unique)} clusters")
            if len(test_unique) >= 3:
                eps = test_eps
                db = test_db
                labels = test_labels
                unique_labels = test_unique
                print(f"    [DEBUG] Using eps={eps:.4f} with {len(unique_labels)} clusters")
                break
    
    detections = []
    # Sort clusters by size (largest first) to prioritize significant buildings
    cluster_sizes = [(label, np.sum(labels == label)) for label in unique_labels]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    
    for obj_id, (label, cluster_size) in enumerate(cluster_sizes[:n_clusters]):
        mask = labels == label
        cluster_points = high_points[mask]
        
        if len(cluster_points) < 5:  # Lower threshold to catch smaller buildings
            continue
        
        min_xyz = cluster_points.min(axis=0)
        max_xyz = cluster_points.max(axis=0)
        center_3d = cluster_points.mean(axis=0)
        size = (max_xyz[0] - min_xyz[0], 
                max_xyz[1] - min_xyz[1], 
                max_xyz[2] - min_xyz[2])
        
        # Classify based on height (adjusted for normalized coordinates)
        building_height = max_xyz[1] - min_xyz[1]
        # For normalized coords, use much smaller height threshold
        normalized_min_height = 0.05  # Equivalent to ~0.5m in normalized space
        
        if building_height >= normalized_min_height:
            class_name = "building"
            # Scale confidence based on normalized height (max ~2.0 in this scene)
            confidence = min(0.9, 0.6 + (building_height / 2.0) * 0.3)
        else:
            class_name = "structure"
            confidence = 0.6
        
        # Filter out very small detections (but be more lenient)
        # Note: coordinate system appears to be normalized, not meters
        max_dimension = max(size)
        min_dimension = min(size)
        
        # For normalized coordinates, use much smaller thresholds
        # Buildings should have at least some minimum extent
        if max_dimension < 0.05:  # Very small threshold for normalized coords
            print(f"      [DEBUG] Skipping cluster {obj_id}: too small (max_dim={max_dimension:.4f})")
            continue
        
        # Also check if it's too flat (might be noise)
        if min_dimension < 0.01 and max_dimension < 0.1:
            print(f"      [DEBUG] Skipping cluster {obj_id}: too flat (min={min_dimension:.4f}, max={max_dimension:.4f})")
            continue
        
        print(f"      [DEBUG] Cluster {obj_id}: {len(cluster_points)} points, size={size}, height={building_height:.3f}, class={class_name}")
        
        detections.append(Detection3D(
            object_id=obj_id,
            class_name=class_name,
            confidence=confidence,
            center_3d=center_3d,
            bbox_3d={'min': min_xyz, 'max': max_xyz},
            points=cluster_points,
            size=size
        ))
    
    print(f"    Detected {len(detections)} buildings/structures after filtering")
    if len(detections) == 0:
        print(f"    [WARN] No detections! Check:")
        print(f"      - Height threshold might be too high")
        print(f"      - eps might be too small (all points as noise)")
        print(f"      - min_samples might be too high")
    return detections


# =========================
# 2D to 3D Projection
# =========================

def project_2d_to_3d(detection_2d: Detection2D,
                    camera_position: np.ndarray,
                    camera_quaternion: np.ndarray,
                    fov_deg: float,
                    image_width: int,
                    image_height: int,
                    depth_map: np.ndarray = None,
                    point_cloud: np.ndarray = None,
                    max_depth: float = 100.0) -> Optional[Detection3DFrom2D]:
    """
    Project a 2D detection to 3D space.
    
    Args:
        detection_2d: 2D detection
        camera_position: (3,) camera position
        camera_quaternion: (4,) camera quaternion [x, y, z, w]
        fov_deg: Field of view in degrees
        image_width: Image width in pixels
        image_height: Image height in pixels
        depth_map: Optional depth map (H, W)
        point_cloud: Optional point cloud for ray casting
        max_depth: Maximum depth to consider
    
    Returns:
        Detection3DFrom2D or None if projection fails
    """
    from scipy.spatial.transform import Rotation as R
    
    # Convert quaternion to rotation matrix
    r = R.from_quat(camera_quaternion)
    rot_mat = r.as_matrix()
    
    # Camera forward, right, up vectors
    forward = rot_mat[:, 2]  # -Z in camera space
    right = rot_mat[:, 0]
    up = rot_mat[:, 1]
    
    # Pixel to normalized device coordinates
    cx, cy = detection_2d.center_2d
    nx = (cx / image_width - 0.5) * 2.0
    ny = (0.5 - cy / image_height) * 2.0  # Flip Y
    
    # FOV to focal length
    fov_rad = np.radians(fov_deg)
    focal_length = 0.5 / np.tan(fov_rad / 2.0)
    
    # Ray direction in camera space
    ray_cam = np.array([nx / focal_length, ny / focal_length, -1.0])
    ray_cam = ray_cam / np.linalg.norm(ray_cam)
    
    # Transform to world space
    ray_world = rot_mat @ ray_cam
    
    # If depth map available, use it
    if depth_map is not None:
        x, y = int(cx), int(cy)
        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
            depth = depth_map[y, x]
            if 0 < depth < max_depth:
                point_3d = camera_position + ray_world * depth
                return Detection3DFrom2D(
                    frame_id=detection_2d.frame_id,
                    detection_2d=detection_2d,
                    center_3d=point_3d,
                    depth=depth,
                    points_3d=np.array([point_3d])
                )
    
    # Otherwise, ray cast to point cloud
    if point_cloud is not None:
        # Find nearest point along ray
        # Simplified: find point in point cloud closest to ray
        distances = np.linalg.norm(
            np.cross(point_cloud - camera_position[None, :], ray_world[None, :]),
            axis=1
        )
        nearest_idx = np.argmin(distances)
        nearest_point = point_cloud[nearest_idx]
        
        # Check if point is in front of camera
        to_point = nearest_point - camera_position
        depth = np.dot(to_point, ray_world)
        
        if 0 < depth < max_depth:
            return Detection3DFrom2D(
                frame_id=detection_2d.frame_id,
                detection_2d=detection_2d,
                center_3d=nearest_point,
                depth=depth,
                points_3d=np.array([nearest_point])
            )
    
    return None


# =========================
# Visualization
# =========================

def visualize_detections_2d(image_path: str,
                           detections: List[Detection2D],
                           output_path: str = "detections_2d.png"):
    """Visualize 2D detections on image"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[VIZ] Could not load image: {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_rgb)
    
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        width = x2 - x1
        height = y2 - y1
        
        rect = Rectangle((x1, y1), width, height, 
                        linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        label = f"{det.class_name} {det.confidence:.2f}"
        ax.text(x1, y1 - 5, label, color='red', fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_title(f"2D Object Detections ({len(detections)} objects)")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[VIZ] Saved 2D detections to {output_path}")
    plt.close()


def visualize_detections_3d(roi_points: np.ndarray,
                            detections: List[Detection3D],
                            output_path: str = "detections_3d.png"):
    """Visualize 3D detections in point cloud (optimized for buildings)"""
    fig = plt.figure(figsize=(16, 6))
    
    # Top-down view (fix orientation - try different flips)
    ax1 = fig.add_subplot(131)
    # Try flipping both X and Z to match typical coordinate systems
    ax1.scatter(roi_points[:, 0], roi_points[:, 2],  # Try without flip first
               c='lightgray', s=0.1, alpha=0.3, label='Point cloud')
    
    # Color buildings differently
    building_count = sum(1 for d in detections if d.class_name == "building")
    structure_count = len(detections) - building_count
    
    for det in detections:
        if det.class_name == "building":
            color = 'red'
            marker = 's'  # Square for buildings
            size = 150
        else:
            color = 'orange'
            marker = '^'  # Triangle for structures
            size = 100
        
        ax1.scatter(det.center_3d[0], det.center_3d[2],  # No flip
                   c=color, s=size, marker=marker, 
                   edgecolors='black', linewidths=1.5,
                   label=f"{det.class_name} {det.object_id}" if det.object_id < 5 else "")
        
        # Draw bounding box (2D projection)
        min_xyz = det.bbox_3d['min']
        max_xyz = det.bbox_3d['max']
        width = max_xyz[0] - min_xyz[0]
        depth = max_xyz[2] - min_xyz[2]
        rect = Rectangle((min_xyz[0], min_xyz[2]), width, depth,
                         linewidth=2, edgecolor=color, facecolor='none', alpha=0.7, linestyle='--')
        ax1.add_patch(rect)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_title(f'Top-down View: {building_count} Buildings, {structure_count} Structures')
    if len(detections) <= 10:
        ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Side view (flip Y axis to fix orientation)
    ax2 = fig.add_subplot(132)
    ax2.scatter(roi_points[:, 0], -roi_points[:, 1],  # Flip Y to fix upside-down
               c=roi_points[:, 1], s=0.1, alpha=0.3, cmap='terrain', label='Point cloud')
    
    for det in detections:
        color = 'red' if det.class_name == "building" else 'orange'
        marker = 's' if det.class_name == "building" else '^'
        size = 150 if det.class_name == "building" else 100
        
        ax2.scatter(det.center_3d[0], -det.center_3d[1],  # Flip Y
                   c=color, s=size, marker=marker,
                   edgecolors='black', linewidths=1.5,
                   label=f"{det.class_name} {det.object_id}" if det.object_id < 5 else "")
        
        # Draw height bar (flip Y)
        min_xyz = det.bbox_3d['min']
        max_xyz = det.bbox_3d['max']
        ax2.plot([det.center_3d[0], det.center_3d[0]], 
                [-max_xyz[1], -min_xyz[1]],  # Flip and swap min/max for correct direction
                color=color, linewidth=3, alpha=0.6)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y (height, flipped)')
    ax2.set_title('Side View: Building Heights')
    if len(detections) <= 10:
        ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3D view
    ax3 = fig.add_subplot(133, projection='3d')
    if len(roi_points) > 10000:
        idx = np.random.choice(len(roi_points), size=10000, replace=False)
        viz_points = roi_points[idx]
    else:
        viz_points = roi_points
    
    ax3.scatter(viz_points[:, 0], viz_points[:, 1], viz_points[:, 2],
               c=viz_points[:, 1], s=1, alpha=0.2, cmap='terrain')
    
    for det in detections:
        color = 'red' if det.class_name == "building" else 'orange'
        marker = 's' if det.class_name == "building" else '^'
        size = 200 if det.class_name == "building" else 150
        
        ax3.scatter(det.center_3d[0], det.center_3d[1], det.center_3d[2],
                   c=color, s=size, marker=marker,
                   edgecolors='black', linewidths=1.5,
                   label=f"{det.class_name} {det.object_id}" if det.object_id < 5 else "")
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y (height)')
    ax3.set_zlabel('Z')
    ax3.set_title('3D View: Building Detections')
    if len(detections) <= 10:
        ax3.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[VIZ] Saved 3D detections to {output_path}")
    plt.close()


# =========================
# Utility Functions
# =========================

def combine_3d_detections(detections1: List[Detection3D],
                         detections2: List[Detection3D],
                         distance_threshold: float = 2.0) -> List[Detection3D]:
    """
    Combine two sets of 3D detections, removing duplicates.
    
    Two detections are considered duplicates if their centers are within
    distance_threshold of each other.
    """
    all_detections = detections1 + detections2
    if len(all_detections) == 0:
        return []
    
    # Build KD-tree for fast nearest neighbor search
    centers = np.array([d.center_3d for d in all_detections])
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


# =========================
# Export Results
# =========================

def export_detections_json(detections_2d: Dict[int, List[Detection2D]] = None,
                          detections_3d: List[Detection3D] = None,
                          output_path: str = "detections.json"):
    """Export detections to JSON format (cast NumPy types to native Python)."""
    result = {
        "detections_2d": {},
        "detections_3d": []
    }
    
    if detections_2d:
        for frame_id, dets in detections_2d.items():
            result["detections_2d"][str(frame_id)] = [
                {
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "bbox": d.bbox,
                    "center_2d": d.center_2d
                }
                for d in dets
            ]
    
    if detections_3d:
        result["detections_3d"] = [
            {
                "object_id": d.object_id,
                "class_name": d.class_name,
                "confidence": float(d.confidence),
                "center_3d": [float(v) for v in d.center_3d.tolist()],
                "bbox_3d": {
                    "min": [float(v) for v in d.bbox_3d["min"].tolist()],
                    "max": [float(v) for v in d.bbox_3d["max"].tolist()]
                },
                "size": [float(s) for s in d.size]
            }
            for d in detections_3d
        ]
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[EXPORT] Saved detections to {output_path}")

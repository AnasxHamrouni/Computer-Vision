import json
import math
import os
from dataclasses import dataclass

import numpy as np
from plyfile import PlyData
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# =========================
# Utility data structures
# =========================

@dataclass
class SceneStats:
    min_xyz: np.ndarray
    max_xyz: np.ndarray
    center_xyz: np.ndarray
    y_range: float
    y_min: float
    y_max: float


# =========================
# PLY loading
# =========================

def load_ply_xyz(ply_path: str) -> np.ndarray:
    """
    Load XYZ points from a PLY file with standard 'x','y','z' vertex properties.
    Returns: (N,3) float32 array.
    """
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    print(f"[LOAD] Reading PLY: {ply_path}")
    ply = PlyData.read(ply_path)

    # `ply.elements` is a list of PlyElement objects; check by element name.
    element_names = [e.name for e in ply.elements]
    if "vertex" not in element_names:
        raise ValueError("PLY file has no 'vertex' element; expected Gaussian splat centers.")

    v = ply["vertex"].data
    if not all(name in v.dtype.names for name in ("x", "y", "z")):
        raise ValueError("PLY 'vertex' element must contain x, y, z properties.")

    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
    print(f"[LOAD] Loaded {xyz.shape[0]} points")
    return xyz


# =========================
# ROI detection (island vs noise)
# =========================

def detect_island_roi(points_xyz: np.ndarray,
                      eps: float = 0.1,
                      min_samples: int = 50,
                      extra_radius_scale: float = 1.3,
                      visualize: bool = True):
    """
    Detect the main island as the largest dense cluster in XZ using DBSCAN,
    then keep points within a radius around that cluster center.
    Returns (filtered points (N_roi, 3), metadata dict for visualization).
    """
    if points_xyz.shape[0] < min_samples:
        raise ValueError("Not enough points to run DBSCAN; check your PLY conversion.")

    # To avoid exhausting RAM and long runtimes on very large splat clouds,
    # aggressively subsample before DBSCAN. We only need a rough island contour.
    max_dbscan_points = 80_000
    if points_xyz.shape[0] > max_dbscan_points:
        print(f"[ROI] Subsampling {points_xyz.shape[0]} -> {max_dbscan_points} points for DBSCAN...")
        idx = np.random.choice(points_xyz.shape[0], size=max_dbscan_points, replace=False)
        dbscan_points = points_xyz[idx]
        subsample_idx = idx
    else:
        dbscan_points = points_xyz
        subsample_idx = np.arange(points_xyz.shape[0])

    xz = dbscan_points[:, [0, 2]]
    print("[ROI] Running DBSCAN on XZ...")
    # Use a single job to reduce multiprocessing / semaphore pressure on macOS.
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1).fit(xz)
    labels = db.labels_

    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique) == 0:
        raise RuntimeError("DBSCAN found no clusters; try increasing eps or check scene scale.")

    # Pick largest cluster as island
    largest_idx = np.argmax(counts)
    island_label = unique[largest_idx]
    island_mask = labels == island_label
    # Map mask back to the subset of points used for DBSCAN.
    island_pts = dbscan_points[island_mask]

    print(f"[ROI] Largest cluster label={island_label}, size={island_pts.shape[0]} / {points_xyz.shape[0]}")

    # Compute XZ center and max distance to define radius
    xz_island = island_pts[:, [0, 2]]
    center_xz = xz_island.mean(axis=0)
    dists = np.linalg.norm(xz_island - center_xz[None, :], axis=1)
    base_radius = dists.max()
    radius = base_radius * extra_radius_scale

    keep_mask = dists <= radius
    roi_pts = island_pts[keep_mask]

    print(f"[ROI] Pruned far noise: {island_pts.shape[0]} -> {roi_pts.shape[0]} points")
    print(f"[ROI] Center XZ: {center_xz}, base_radius={base_radius:.3f}, used_radius={radius:.3f}")
    
    metadata = {
        'all_points': points_xyz,
        'dbscan_points': dbscan_points,
        'labels': labels,
        'island_label': island_label,
        'center_xz': center_xz,
        'base_radius': base_radius,
        'used_radius': radius,
        'roi_pts': roi_pts,
    }
    return roi_pts, metadata


def compute_scene_stats(points_xyz: np.ndarray) -> SceneStats:
    pmin = points_xyz.min(axis=0)
    pmax = points_xyz.max(axis=0)
    center = 0.5 * (pmin + pmax)
    y_min, y_max = pmin[1], pmax[1]
    return SceneStats(
        min_xyz=pmin,
        max_xyz=pmax,
        center_xyz=center,
        y_range=float(y_max - y_min),
        y_min=float(y_min),
        y_max=float(y_max),
    )


# =========================
# Path generation: outdoor orbit
# =========================

def generate_orbit_positions(roi_points: np.ndarray,
                             n_frames: int = 600,
                             radius_scale: float = 1.2,
                             base_clearance_rel: float = 0.15,
                             vertical_amplitude_rel: float = 0.05) -> np.ndarray:
    """
    Generate a more exploratory FPV-like path around the island:
    
    - Mixes wide orbits, inward dives, and direction changes.
    - Keeps a safety clearance above local geometry using a KD-tree.
    """
    stats = compute_scene_stats(roi_points)
    print(f"[STATS] ROI bounds X:[{stats.min_xyz[0]:.3f}, {stats.max_xyz[0]:.3f}] "
          f"Y:[{stats.min_xyz[1]:.3f}, {stats.max_xyz[1]:.3f}] "
          f"Z:[{stats.min_xyz[2]:.3f}, {stats.max_xyz[2]:.3f}]")

    # Island center in XZ
    center_x = stats.center_xyz[0]
    center_z = stats.center_xyz[2]

    # Estimate island "radius" in XZ
    xz = roi_points[:, [0, 2]]
    center_xz = np.array([center_x, center_z], dtype=np.float32)
    dists = np.linalg.norm(xz - center_xz[None, :], axis=1)
    island_radius = float(np.percentile(dists, 95))  # ignore few outliers

    outer_radius = island_radius * radius_scale
    inner_radius = outer_radius * 0.55
    print(f"[ORBIT] Island_radius≈{island_radius:.3f}, "
          f"outer_radius={outer_radius:.3f}, inner_radius={inner_radius:.3f}")

    # Estimate ground level and vertical scales
    ground_y = float(np.percentile(roi_points[:, 1], 5))
    y_span = stats.y_range if stats.y_range > 1e-6 else 1.0

    base_clearance = y_span * base_clearance_rel
    vert_amp = y_span * vertical_amplitude_rel
    base_y = ground_y + base_clearance

    print(f"[ALT] ground_y≈{ground_y:.3f}, y_span={y_span:.3f}, "
          f"base_clearance={base_clearance:.3f}, base_y={base_y:.3f}, vert_amp={vert_amp:.3f}")

    # Build a KD-tree on ROI points (full 3D) to keep distance from obstacles.
    tree = cKDTree(roi_points[:, :3])
    obstacle_radius = island_radius * 0.08  # local neighborhood for "walls"
    height_clearance = max(0.1 * y_span, 0.2)  # absolute min clearance above local geometry

    # Normalized time [0,1) for path design
    u = np.linspace(0.0, 1.0, n_frames, endpoint=False)

    xs = np.zeros_like(u, dtype=np.float32)
    zs = np.zeros_like(u, dtype=np.float32)
    ys = np.zeros_like(u, dtype=np.float32)

    # Design several phases:
    #  0.00–0.25: outer orbit clockwise
    #  0.25–0.50: spiral inward, direction change
    #  0.50–0.75: inner orbit counter‑clockwise
    #  0.75–1.00: weaving exit back to mid radius
    for i, ui in enumerate(u):
        if ui < 0.25:
            # Wide establishing shot
            t = ui / 0.25 * 2.0 * math.pi
            radius = outer_radius
            angle = -t
        elif ui < 0.50:
            # Spiral inward, reverse direction
            v = (ui - 0.25) / 0.25  # 0..1
            radius = outer_radius * (1.0 - 0.6 * v)  # go from outer to ~0.4*outer
            t = v * 2.0 * math.pi
            angle = t + math.pi * 0.25  # offset so we don't retrace exactly
        elif ui < 0.75:
            # Tight inner orbit the other way
            v = (ui - 0.50) / 0.25
            radius = inner_radius
            t = v * 2.0 * math.pi
            angle = t
        else:
            # Weaving exit back outward with radius modulation
            v = (ui - 0.75) / 0.25
            radius = inner_radius + (outer_radius - inner_radius) * 0.6 * v
            t = v * 2.0 * math.pi
            # Add a small lateral weave
            weave = 0.12 * math.sin(4.0 * t)
            angle = -t + weave

        x = center_x + radius * math.cos(angle)
        z = center_z + radius * math.sin(angle)

        # Base vertical profile: gentle waves along the whole path
        y_profile = base_y + vert_amp * math.sin(3.0 * ui * 2.0 * math.pi)

        # Query local geometry around (x, z) and enforce clearance above it.
        # Use base_y as approximate vertical for the query point.
        nearby_idx = tree.query_ball_point([x, base_y, z], r=obstacle_radius)
        if nearby_idx:
            local_max_y = float(roi_points[nearby_idx, 1].max())
            safe_y = local_max_y + height_clearance
            y = max(y_profile, safe_y)
        else:
            y = y_profile

        xs[i] = x
        ys[i] = y
        zs[i] = z

    # Smooth altitude for cinematic feel while preserving safety (we only smooth once)
    ys = gaussian_filter1d(ys, sigma=8)

    positions = np.stack([xs, ys, zs], axis=1).astype(np.float32)
    return positions


# =========================
# Camera orientation
# =========================

def look_at_quaternions(positions: np.ndarray,
                        target: np.ndarray,
                        important_regions: np.ndarray = None,
                        up: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float32)) -> np.ndarray:
    """
    Compute camera quaternions with smart, smooth orientation:

    - Primarily look along the direction of motion.
    - Dynamically looks at nearby important regions (peaks, structures).
    - Adjusts vertical angle based on distance to interesting features.
    - Enforce a right-handed frame suitable for three.js (Y‑up, -Z forward).
    """
    N = positions.shape[0]
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    quats = np.zeros((N, 4), dtype=np.float32)

    for i in range(N):
        eye = positions[i]

        # Motion direction (forward along path)
        if i < N - 1:
            dir_vec = positions[i + 1] - eye
        else:
            dir_vec = eye - positions[i - 1]

        # Find nearest important region
        if important_regions is not None and len(important_regions) > 0:
            dists_to_regions = np.linalg.norm(important_regions - eye[None, :], axis=1)
            nearest_idx = np.argmin(dists_to_regions)
            nearest_region = important_regions[nearest_idx]
            dist_to_region = dists_to_regions[nearest_idx]
            
            # Direction toward nearest important region
            region_vec = nearest_region - eye
            
            # Adaptive vertical angle: closer = look more downward, farther = more horizontal
            # Normalize distance (assume max distance is ~2x island radius)
            max_dist = np.linalg.norm(important_regions - target, axis=1).max() * 2.0
            dist_factor = min(dist_to_region / max_dist, 1.0)
            
            # Closer regions get more downward bias, farther get less
            downward_bias = 0.25 * (1.0 - dist_factor * 0.5)  # 0.25 to 0.125
            
            # Blend: motion (60%), nearest region (30%), center (10%)
            motion_weight = 0.6
            region_weight = 0.3
            center_weight = 0.1
            center_vec = target - eye
            
            look_vec = (motion_weight * dir_vec + 
                       region_weight * region_vec + 
                       center_weight * center_vec - 
                       downward_bias * up)
        else:
            # Fallback: blend motion and center view
            center_vec = target - eye
            motion_weight = 0.7
            center_weight = 0.3
            look_vec = motion_weight * dir_vec + center_weight * center_vec - 0.15 * up

        norm_f = np.linalg.norm(look_vec)
        if norm_f < 1e-6:
            # Degenerate, fall back to center-only view
            look_vec = center_vec if np.linalg.norm(center_vec) > 1e-6 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
            norm_f = np.linalg.norm(look_vec)

        forward = look_vec / norm_f

        # Right = up x forward
        right = np.cross(up, forward)
        norm_r = np.linalg.norm(right)
        if norm_r < 1e-6:
            # forward parallel to up; pick arbitrary right
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right /= norm_r

        # Recompute orthonormal up
        true_up = np.cross(forward, right)

        # Build rotation matrix (camera space: columns = right, up, -forward)
        rot_mat = np.stack([right, true_up, -forward], axis=1)

        # Ensure we have a proper right-handed rotation with positive determinant.
        det = np.linalg.det(rot_mat)
        if det < 0.0:
            # Flip the right vector to fix handedness.
            right = -right
            rot_mat = np.stack([right, true_up, -forward], axis=1)

        r = R.from_matrix(rot_mat)
        q = r.as_quat()  # [x, y, z, w]
        quats[i] = q.astype(np.float32)

    # Smooth quaternions a bit for gimbal-like motion
    quats = smooth_quaternions_gaussian(quats, sigma=6)
    return quats


def detect_important_regions(roi_points: np.ndarray, n_regions: int = 5) -> np.ndarray:
    """
    Detect important/interesting regions in the island:
    - High elevation points (peaks, buildings)
    - Dense clusters (interesting structures)
    Returns (n_regions, 3) array of important point positions.
    """
    # Method 1: Find highest points (top 10% by Y)
    y_threshold = np.percentile(roi_points[:, 1], 90)
    high_points = roi_points[roi_points[:, 1] >= y_threshold]
    
    if len(high_points) < n_regions:
        # Fallback: use all points
        candidates = roi_points
    else:
        candidates = high_points
    
    # Method 2: Cluster these high points to find distinct peaks
    if len(candidates) > n_regions * 10:
        # Subsample for clustering
        idx = np.random.choice(len(candidates), size=min(5000, len(candidates)), replace=False)
        cluster_candidates = candidates[idx]
    else:
        cluster_candidates = candidates
    
    if len(cluster_candidates) >= n_regions:
        # Use DBSCAN to find distinct clusters
        xz_candidates = cluster_candidates[:, [0, 2]]
        eps_local = np.percentile(np.linalg.norm(xz_candidates - xz_candidates.mean(axis=0), axis=1), 20) * 0.3
        db_local = DBSCAN(eps=eps_local, min_samples=5, n_jobs=1).fit(xz_candidates)
        labels_local = db_local.labels_
        
        # Get cluster centers
        important_regions = []
        for label in np.unique(labels_local[labels_local >= 0])[:n_regions]:
            mask = labels_local == label
            cluster_pts = cluster_candidates[mask]
            # Use the highest point in each cluster
            highest_idx = np.argmax(cluster_pts[:, 1])
            important_regions.append(cluster_pts[highest_idx])
        
        # If we don't have enough, add some random high points
        while len(important_regions) < n_regions:
            idx = np.random.choice(len(candidates))
            important_regions.append(candidates[idx])
        
        important_regions = np.array(important_regions[:n_regions])
    else:
        # Fallback: just pick highest points
        sorted_by_y = candidates[np.argsort(candidates[:, 1])[::-1]]
        important_regions = sorted_by_y[:n_regions]
        if len(important_regions) < n_regions:
            # Pad with center if needed
            center = roi_points.mean(axis=0)
            while len(important_regions) < n_regions:
                important_regions = np.vstack([important_regions, center])
    
    print(f"[IMPORTANT] Detected {len(important_regions)} important regions (peaks/structures)")
    return important_regions


def smooth_quaternions_gaussian(quats: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """
    Smooth quaternion sequence with Gaussian filter in a way that respects q and -q equivalence.
    """
    q = quats.copy().astype(np.float64)

    # Ensure shortest path (avoid jumps due to sign flips)
    for i in range(1, q.shape[0]):
        if np.dot(q[i - 1], q[i]) < 0.0:
            q[i] = -q[i]

    # Smooth each component separately
    for c in range(4):
        q[:, c] = gaussian_filter1d(q[:, c], sigma=sigma, mode="nearest")

    # Renormalize
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    q /= norms
    return q.astype(np.float32)


# =========================
# Visualization
# =========================

def visualize_roi_and_path(roi_metadata: dict,
                            positions: np.ndarray,
                            important_regions: np.ndarray = None,
                            output_path: str = "path_visualization.png") -> None:
    """
    Create visualization plots showing:
    1. ROI detection (DBSCAN clusters)
    2. Planned path
    3. Important regions
    """
    fig = plt.figure(figsize=(16, 6))
    
    # Plot 1: Top-down view (XZ plane)
    ax1 = fig.add_subplot(131)
    
    # Show all points (subsampled for speed)
    all_pts = roi_metadata['all_points']
    if len(all_pts) > 50000:
        idx_viz = np.random.choice(len(all_pts), size=50000, replace=False)
        viz_pts = all_pts[idx_viz]
    else:
        viz_pts = all_pts
    
    ax1.scatter(viz_pts[:, 0], viz_pts[:, 2], c='lightgray', s=0.1, alpha=0.3, label='All points')
    
    # Show DBSCAN clusters
    dbscan_pts = roi_metadata['dbscan_points']
    labels = roi_metadata['labels']
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Noise points
            mask = labels == label
            if np.sum(mask) > 0:
                ax1.scatter(dbscan_pts[mask, 0], dbscan_pts[mask, 2], 
                           c='black', s=0.5, alpha=0.1, label='Noise' if label == unique_labels[0] else '')
        else:
            mask = labels == label
            if np.sum(mask) > 0:
                label_str = f'Cluster {label}' if label != roi_metadata['island_label'] else f'Island (Cluster {label})'
                ax1.scatter(dbscan_pts[mask, 0], dbscan_pts[mask, 2], 
                           c=[color], s=1, alpha=0.5, label=label_str if label == unique_labels[0] else '')
    
    # Highlight ROI
    roi_pts = roi_metadata['roi_pts']
    ax1.scatter(roi_pts[:, 0], roi_pts[:, 2], c='red', s=2, alpha=0.6, label='ROI (filtered)')
    
    # Show ROI boundary circle
    center_xz = roi_metadata['center_xz']
    radius = roi_metadata['used_radius']
    circle = Circle(center_xz, radius, fill=False, edgecolor='red', linewidth=2, linestyle='--', label='ROI boundary')
    ax1.add_patch(circle)
    
    # Show path
    ax1.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2, alpha=0.7, label='Camera path')
    ax1.scatter(positions[::50, 0], positions[::50, 2], c='blue', s=20, marker='o', alpha=0.8, label='Path waypoints')
    
    # Show important regions
    if important_regions is not None and len(important_regions) > 0:
        ax1.scatter(important_regions[:, 0], important_regions[:, 2], 
                   c='yellow', s=100, marker='*', edgecolors='orange', linewidths=2, 
                   label='Important regions', zorder=10)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_title('Top-down View: ROI Detection & Path Planning')
    ax1.legend(loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Plot 2: Side view (XZ with Y as color)
    ax2 = fig.add_subplot(132)
    
    # Color by height
    scatter = ax2.scatter(roi_pts[:, 0], roi_pts[:, 2], c=roi_pts[:, 1], 
                         s=1, alpha=0.4, cmap='terrain', label='ROI (colored by height)')
    plt.colorbar(scatter, ax=ax2, label='Y (height)')
    
    # Path with height coloring
    path_colors = positions[:, 1]
    ax2.scatter(positions[:, 0], positions[:, 2], c=path_colors, 
               s=10, cmap='coolwarm', alpha=0.8, edgecolors='black', linewidths=0.5, label='Camera path')
    
    if important_regions is not None and len(important_regions) > 0:
        ax2.scatter(important_regions[:, 0], important_regions[:, 2], 
                   c='yellow', s=150, marker='*', edgecolors='orange', linewidths=2, 
                   label='Important regions', zorder=10)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title('Side View: Height Map & Path')
    ax2.legend(loc='upper right', fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # Plot 3: 3D path visualization
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Show ROI as point cloud (subsampled)
    if len(roi_pts) > 10000:
        idx_3d = np.random.choice(len(roi_pts), size=10000, replace=False)
        roi_3d = roi_pts[idx_3d]
    else:
        roi_3d = roi_pts
    
    ax3.scatter(roi_3d[:, 0], roi_3d[:, 1], roi_3d[:, 2], 
               c=roi_3d[:, 1], s=1, alpha=0.2, cmap='terrain')
    
    # Path
    ax3.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            'b-', linewidth=2, alpha=0.8, label='Camera path')
    ax3.scatter(positions[::30, 0], positions[::30, 1], positions[::30, 2], 
               c='blue', s=30, marker='o', alpha=0.9, edgecolors='darkblue', linewidths=1)
    
    if important_regions is not None and len(important_regions) > 0:
        ax3.scatter(important_regions[:, 0], important_regions[:, 1], important_regions[:, 2], 
                   c='yellow', s=200, marker='*', edgecolors='orange', linewidths=2, 
                   label='Important regions', zorder=10)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y (height)')
    ax3.set_zlabel('Z')
    ax3.set_title('3D View: Path & Important Regions')
    ax3.legend(loc='upper left', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[VIZ] Saved visualization to {output_path}")
    plt.close()


# =========================
# JSON export
# =========================

def export_trajectory_json(positions: np.ndarray,
                           quats: np.ndarray,
                           output_path: str,
                           fps: int = 30,
                           fov_deg: float = 65.0) -> None:
    """
    Export trajectory as a JSON file compatible with index.html.
    The viewer expects a flat array of frame objects:
        [
          {
            "frame": i,
            "position": { "x": ..., "y": ..., "z": ... },
            "quaternion": { "x": ..., "y": ..., "z": ..., "w": ... },
            "fov": fov_deg
          },
          ...
        ]
    """
    assert positions.shape[0] == quats.shape[0]
    frames = []
    for i, (p, q) in enumerate(zip(positions, quats)):
        frames.append({
            "frame": int(i),
            "position": {
                "x": float(p[0]),
                "y": float(p[1]),
                "z": float(p[2]),
            },
            "quaternion": {
                "x": float(q[0]),
                "y": float(q[1]),
                "z": float(q[2]),
                "w": float(q[3]),
            },
            "fov": float(fov_deg),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(frames, f, indent=2)
    print(f"[SAVE] Wrote {len(frames)} frames to {output_path}")


# =========================
# Main pipeline
# =========================

def generate_autonomous_outdoor_trajectory(
    ply_file: str = "outdoor-standard.ply",
    output_json: str = "trajectory_autonomous.json",
    n_frames: int = 600,
    visualize: bool = True,
    detect_objects: bool = False,
) -> None:
    # 1) Load points
    pts = load_ply_xyz(ply_file)

    # 2) Detect island ROI and remove far-away noise (e.g., distant city)
    roi, roi_metadata = detect_island_roi(
        pts,
        eps=0.08,          # adjust if clustering fails: smaller if model in [-2,2], larger if in meters
        min_samples=100,   # more robust for dense splats
        extra_radius_scale=1.2,
        visualize=visualize
    )

    # 3) Detect important regions (peaks, structures)
    important_regions = detect_important_regions(roi, n_regions=6)
    print(f"[IMPORTANT] Important regions:\n{important_regions}")

    # 4) Generate orbit positions
    positions = generate_orbit_positions(
        roi,
        n_frames=n_frames,
        radius_scale=1.2,
        base_clearance_rel=0.12,   # relative to height range; tweak for closer/farther fly
        vertical_amplitude_rel=0.04
    )

    # 5) Target: island center (from ROI)
    stats = compute_scene_stats(roi)
    look_target = stats.center_xyz
    print(f"[CAM] Look target (island center): {look_target}")

    # 6) Compute camera quaternions (look-at with important regions)
    quats = look_at_quaternions(positions, target=look_target, important_regions=important_regions)

    # 7) Visualization
    if visualize:
        viz_path = output_json.replace('.json', '_visualization.png')
        visualize_roi_and_path(roi_metadata, positions, important_regions, output_path=viz_path)

    # 8) Export JSON
    export_trajectory_json(positions, quats, output_json)
    
    # 9) Optional: Object detection
    if detect_objects:
        try:
            from object_detection import detect_objects_3d_geometric, detect_objects_3d_height_based, combine_3d_detections, visualize_detections_3d, export_detections_json
            
            print("\n" + "=" * 60)
            print("Running 3D Object Detection...")
            print("=" * 60)
            
            # Run 3D detection
            detections_geometric = detect_objects_3d_geometric(
                roi, eps=0.15, min_samples=30, min_size=0.5, max_size=25.0
            )
            detections_height = detect_objects_3d_height_based(roi, n_clusters=15)
            all_detections = combine_3d_detections(detections_geometric, detections_height)
            
            print(f"[DETECT] Found {len(all_detections)} objects")
            for det in all_detections:
                print(f"  - {det.class_name} at {det.center_3d}")
            
            # Visualize and export
            detections_path = output_json.replace('.json', '_detections_3d.png')
            visualize_detections_3d(roi, all_detections, detections_path)
            
            detections_json = output_json.replace('.json', '_detections_3d.json')
            export_detections_json(detections_3d=all_detections, output_path=detections_json)
            
        except ImportError as e:
            print(f"[WARN] Object detection not available: {e}")
            print("       Install dependencies or run separately with detect_objects.py")


if __name__ == "__main__":
    generate_autonomous_outdoor_trajectory()

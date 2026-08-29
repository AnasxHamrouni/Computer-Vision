#!/usr/bin/env python3
"""
Autonomous camera-path generator for 3D point clouds (e.g., island scan).

Pipeline:

1. Load PLY, remove outliers, and extract a dense ROI.
2. Build height / density / interest maps over XZ.
3. Build 2D occupancy + 3D KD-tree.
4. Sample collision-free candidate viewpoints above the surface
   (interest-aware + preferred distance to shoreline).
5. Build a k-NN graph with 3D collision-checked edges.
6. Run greedy orienteering with a smooth-turn bias; if the path is too short,
   fall back to a collision-safe chain.
7. Smooth the path, clamp it to a height band above terrain,
   push it out of geometry, and apply easing.
8. Compute cinematic FPV orientation guided by interest_map.
9. Save poses as JSON.

Assumes Y is the up axis (height).
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import open3d as o3d
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter, gaussian_filter1d, distance_transform_edt
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


# ===================== DATA CLASSES =====================

@dataclass
class Viewpoint:
    pos: np.ndarray      # (3,)
    forward: np.ndarray  # unit vector
    up: np.ndarray       # unit vector
    score: float         # interest


@dataclass
class GraphEdge:
    u: int
    v: int
    cost: float


# ===================== BASIC IO / UTILS =====================

def load_point_cloud(ply_path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        raise ValueError("Empty point cloud")
    return pts


def compute_bounds(pts: np.ndarray, margin: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    pmin = pts.min(axis=0)
    pmax = pts.max(axis=0)
    extent = pmax - pmin
    pmin -= margin * extent
    pmax += margin * extent
    return pmin, pmax


def remove_statistical_outliers(
    pts: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    inlier_pts = np.asarray(cl.points)
    print(f"Outlier removal: {pts.shape[0]} -> {inlier_pts.shape[0]} points")
    return inlier_pts


# ===================== ROI SELECTION =====================

def extract_dense_roi(
    pts: np.ndarray,
    voxel_size_xy: float = 0.05,
    min_rel_density: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Grow a dense region in XZ from the global density maximum.

    Returns ROI points and ROI bounds (pmin_roi, pmax_roi).
    """
    pmin = pts.min(axis=0)
    pmax = pts.max(axis=0)
    size = pmax - pmin

    nx = int(math.ceil(size[0] / voxel_size_xy))
    nz = int(math.ceil(size[2] / voxel_size_xy))

    density = np.zeros((nx, nz), dtype=np.int32)
    ix = ((pts[:, 0] - pmin[0]) / voxel_size_xy).astype(int)
    iz = ((pts[:, 2] - pmin[2]) / voxel_size_xy).astype(int)
    ix = np.clip(ix, 0, nx - 1)
    iz = np.clip(iz, 0, nz - 1)

    for x, z in zip(ix, iz):
        density[x, z] += 1

    # Smooth and find max
    density_smooth = gaussian_filter(density.astype(np.float32), sigma=2.0)
    max_val = float(density_smooth.max())

    if max_val <= 0:
        print("[WARN] Density map empty in extract_dense_roi; returning original points.")
        return pts, pmin, pmax

    center_idx = np.argmax(density_smooth)
    cx, cz = np.unravel_index(center_idx, density_smooth.shape)
    threshold = min_rel_density * max_val

    print(f"ROI: max density={max_val:.1f}, threshold={threshold:.1f}")

    roi_mask = np.zeros_like(density_smooth, dtype=bool)
    visited = np.zeros_like(density_smooth, dtype=bool)
    q = deque()
    q.append((cx, cz))
    visited[cx, cz] = True

    while q:
        x, z = q.popleft()
        if density_smooth[x, z] < threshold:
            continue
        roi_mask[x, z] = True
        for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx_, nz_ = x + dx, z + dz
            if 0 <= nx_ < nx and 0 <= nz_ < nz and not visited[nx_, nz_]:
                visited[nx_, nz_] = True
                q.append((nx_, nz_))

    if not roi_mask.any():
        print("[WARN] ROI mask is empty; returning original points.")
        return pts, pmin, pmax

    pts_mask = roi_mask[ix, iz]
    pts_roi = pts[pts_mask]

    pmin_roi = pts_roi.min(axis=0)
    pmax_roi = pts_roi.max(axis=0)

    print(f"ROI points: {pts.shape[0]} -> {pts_roi.shape[0]}")
    print(f"ROI bounds Y: [{pmin_roi[1]:.3f}, {pmax_roi[1]:.3f}]")
    print(f"[ROI] X:[{pmin_roi[0]:.2f},{pmax_roi[0]:.2f}] "
          f"Y:[{pmin_roi[1]:.2f},{pmax_roi[1]:.2f}] "
          f"Z:[{pmin_roi[2]:.2f},{pmax_roi[2]:.2f}]")

    return pts_roi, pmin_roi, pmax_roi


# ===================== HEIGHT & INTEREST MAPS =====================

def build_height_and_interest_maps(
    pts: np.ndarray,
    pmin: np.ndarray,
    pmax: np.ndarray,
    voxel_size_xy: float = 0.05,
):
    size = pmax - pmin
    nx = int(math.ceil(size[0] / voxel_size_xy))
    nz = int(math.ceil(size[2] / voxel_size_xy))

    height_map = np.full((nx, nz), -np.inf, dtype=np.float32)
    density_map = np.zeros((nx, nz), dtype=np.int32)
    y_min_map = np.full((nx, nz), np.inf, dtype=np.float32)

    idx_x = ((pts[:, 0] - pmin[0]) / voxel_size_xy).astype(int)
    idx_z = ((pts[:, 2] - pmin[2]) / voxel_size_xy).astype(int)
    idx_x = np.clip(idx_x, 0, nx - 1)
    idx_z = np.clip(idx_z, 0, nz - 1)

    for (ix, iz, y) in zip(idx_x, idx_z, pts[:, 1]):
        if y > height_map[ix, iz]:
            height_map[ix, iz] = y
        if y < y_min_map[ix, iz]:
            y_min_map[ix, iz] = y
        density_map[ix, iz] += 1

    mask_zero = (density_map == 0)
    height_map[mask_zero] = pmin[1]
    y_min_map[mask_zero] = pmin[1]

    height_range = height_map - y_min_map

    # Density-based component
    density_float = density_map.astype(np.float32)
    density_norm = np.log1p(density_float)
    density_norm /= (density_norm.max() + 1e-6)

    # Vertical variation component
    height_norm = height_range.copy()
    height_norm -= height_norm.min()
    if height_norm.max() > 1e-6:
        height_norm /= height_norm.max()

    # Interest = mix of density & vertical structure
    #interest_map = 0.5 * density_norm + 0.5 * height_norm
    interest_map = density_norm 
    interest_map = gaussian_filter(interest_map, sigma=1.0)

    # Diagnostics
    print("height_map range:", float(height_map.min()), float(height_map.max()))
    print("y_min_map range:", float(y_min_map.min()), float(y_min_map.max()))
    print("vertical_span range:", float(height_range.min()), float(height_range.max()))

    nonzero = density_map[density_map > 0]
    if nonzero.size > 0:
        print(f"[MAP] density nonzero min/med/max: "
              f"{nonzero.min()} / {np.median(nonzero):.1f} / {nonzero.max()}")
    print(f"[INTEREST] min/mean/max: "
          f"{interest_map.min():.3f}/{interest_map.mean():.3f}/{interest_map.max():.3f}")

    return height_map, y_min_map, density_map, interest_map, (nx, nz), voxel_size_xy


def detect_floor_and_ceiling(
    pts: np.ndarray,
    y_bins: int = 64,
    indoor_range_threshold: float = 5.0,
) -> Tuple[float, float, bool]:
    ys = pts[:, 1]
    y_min, y_max = float(ys.min()), float(ys.max())
    global_range = y_max - y_min

    if global_range < indoor_range_threshold:
        # Likely indoor or confined volume
        floor_y = float(np.percentile(ys, 5))
        ceiling_y = float(np.percentile(ys, 95))
        is_indoor = True
    else:
        floor_y = y_min
        ceiling_y = y_max
        is_indoor = False

    return floor_y, ceiling_y, is_indoor


# ===================== OCCUPANCY & CANDIDATE POSITIONS =====================

def build_2d_occupancy_from_height(
    y_min_map: np.ndarray,
    height_map: np.ndarray,
    floor_y: float,
    ceiling_y: Optional[float],
    density_map: np.ndarray,
    clearance: float = 0.1,
    min_free_fraction: float = 0.1,
) -> np.ndarray:
    vertical_span = height_map - y_min_map
    span_med = float(np.median(vertical_span))
    span_thr = max(0.5 * span_med, 0.2)

    nonzero = density_map[density_map > 0]
    dens_med = float(np.median(nonzero)) if nonzero.size > 0 else 0.0
    dens_thr = max(2.0, dens_med)

    occ_2d = (vertical_span > span_thr) & (density_map >= dens_thr)
    occ_frac = float(occ_2d.mean())

    if occ_frac > 1.0 - min_free_fraction:
        print(f"[WARN] Occupancy very dense (frac={occ_frac:.3f}). Relaxing thresholds slightly.")
        span_thr *= 1.5
        dens_thr *= 0.5
        occ_2d = (vertical_span > span_thr) & (density_map >= dens_thr)
        occ_frac = float(occ_2d.mean())
        print(f"[INFO] After relaxation, occ_frac={occ_frac:.3f}")

    print(f"[OCC] occupied cells: {int(occ_2d.sum())} / total: {int(occ_2d.size)} "
          f"=> frac: {occ_frac:.3f}")
    return occ_2d


def sample_candidate_positions(
    occ_2d: np.ndarray,
    height_map: np.ndarray,
    y_min_map: np.ndarray,
    pmin: np.ndarray,
    voxel_size_xy: float,
    floor_y: float,
    ceiling_y: Optional[float],
    is_indoor: bool,
    interest_map: Optional[np.ndarray] = None,
    dist_to_occ: Optional[np.ndarray] = None,
    n_samples: int = 400,
    clearance: float = 0.5,
    preferred_dist_band: Tuple[float, float] = (3.0, 15.0),
    interest_weight: float = 0.7,
    shoreline_weight: float = 0.3,
) -> List[np.ndarray]:
    """
    Sample collision-free camera bases from free cells.

    - Prefer higher-interest cells (interest_map).
    - Prefer cells whose distance to occupied region (shoreline) lies in a band.
    """
    nx, nz = occ_2d.shape
    free_cells = np.argwhere(~occ_2d)

    if free_cells.shape[0] == 0:
        raise ValueError("No free cells found in occupancy map")

    n_free = free_cells.shape[0]

    if interest_map is None and dist_to_occ is None:
        # Fall back to uniform random sampling
        if n_samples < n_free:
            idx = np.random.choice(n_free, size=n_samples, replace=False)
            free_cells = free_cells[idx]
    else:
        # Build a sampling distribution over free cells
        scores = np.zeros(n_free, dtype=np.float32)

        # Interest term
        if interest_map is not None:
            ivals = interest_map[free_cells[:, 0], free_cells[:, 1]].astype(np.float32)
            if ivals.max() > 0:
                ivals /= ivals.max()
            scores += interest_weight * ivals

        # Shoreline distance term (prefer a band near the island)
        if dist_to_occ is not None:
            dvals = dist_to_occ[free_cells[:, 0], free_cells[:, 1]].astype(np.float32)
            dmin, dmax = preferred_dist_band
            if dmax > dmin:
                # "Tent" profile: 0 near dmin/dmax, 1 in the middle
                mid = 0.5 * (dmin + dmax)
                half = 0.5 * (dmax - dmin)
                t = np.clip((dvals - mid) / (half + 1e-6), -1.0, 1.0)
                shoreline_score = 1.0 - np.abs(t)  # 1 at mid, 0 at edges
                shoreline_score = np.clip(shoreline_score, 0.0, 1.0)
                scores += shoreline_weight * shoreline_score

        scores = np.clip(scores, 1e-4, None)
        probs = scores / scores.sum()

        m = min(n_samples, n_free)
        chosen_idx = np.random.choice(n_free, size=m, replace=False, p=probs)
        free_cells = free_cells[chosen_idx]

    positions: List[np.ndarray] = []
    for ix, iz in free_cells:
        x = pmin[0] + (ix + 0.5) * voxel_size_xy
        z = pmin[2] + (iz + 0.5) * voxel_size_xy

        ground_y = y_min_map[ix, iz]
        top_y = height_map[ix, iz]

        if is_indoor:
            if ceiling_y is None:
                ceiling_local = top_y + 2.0
            else:
                ceiling_local = ceiling_y
            low = max(floor_y + clearance, ground_y + clearance)
            high = min(ceiling_local - clearance, top_y + 0.5 * (ceiling_local - top_y))
            y = 0.5 * (low + high)
        else:
            # Outdoor: stay just above the terrain and floor
            y = max(ground_y + clearance, floor_y + clearance)

        positions.append(np.array([x, y, z], dtype=np.float32))

    print(f"[CANDIDATES] free_cells={n_free}, sampled={len(positions)}")
    return positions


def build_viewpoints(
    positions: List[np.ndarray],
    interest_map: np.ndarray,
    pmin: np.ndarray,
    voxel_size_xy: float,
    up_global: np.ndarray = np.array([0.0, 1.0, 0.0]),
    interest_radius: float = 1.0,
) -> List[Viewpoint]:
    nx, nz = interest_map.shape
    viewpoints: List[Viewpoint] = []
    r_cells = max(1, int(math.ceil(interest_radius / voxel_size_xy)))

    for pos in positions:
        ix = int((pos[0] - pmin[0]) / voxel_size_xy)
        iz = int((pos[2] - pmin[2]) / voxel_size_xy)
        ix = np.clip(ix, 0, nx - 1)
        iz = np.clip(iz, 0, nz - 1)

        ix0 = max(0, ix - r_cells)
        ix1 = min(nx - 1, ix + r_cells)
        iz0 = max(0, iz - r_cells)
        iz1 = min(nz - 1, iz + r_cells)

        patch = interest_map[ix0:ix1 + 1, iz0:iz1 + 1]
        flat_idx = np.argmax(patch)
        dx, dz = np.unravel_index(flat_idx, patch.shape)
        tx = ix0 + dx
        tz = iz0 + dz

        target_x = pmin[0] + (tx + 0.5) * voxel_size_xy
        target_z = pmin[2] + (tz + 0.5) * voxel_size_xy
        target = np.array([target_x, pos[1], target_z], dtype=np.float32)

        forward = target - pos
        if np.linalg.norm(forward) < 1e-6:
            forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        forward = forward / (np.linalg.norm(forward) + 1e-6)

        # Limit vertical tilt for cinematic feel
        forward[1] = np.clip(forward[1], -0.3, 0.3)
        forward = forward / (np.linalg.norm(forward) + 1e-6)

        up = up_global.copy()
        if abs(np.dot(up, forward)) > 0.95:
            up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            up /= (np.linalg.norm(up) + 1e-6)

        score = interest_map[ix, iz]
        viewpoints.append(Viewpoint(pos=pos, forward=forward, up=up, score=float(score)))

    return viewpoints


# ===================== KD-TREE & COLLISION =====================

def build_scene_kdtree(pts: np.ndarray) -> cKDTree:
    return cKDTree(pts.astype(np.float32))


def is_segment_collision_free_3d(
    p0: np.ndarray,
    p1: np.ndarray,
    kdtree: cKDTree,
    cam_radius: float = 0.25,
    n_samples: int = 24,
) -> bool:
    for t in np.linspace(0.0, 1.0, n_samples):
        p = (1.0 - t) * p0 + t * p1
        dist, _ = kdtree.query(p, k=1)
        if dist < cam_radius:
            return False
    return True


def push_out_of_geometry(
    path: np.ndarray,
    kdtree: cKDTree,
    cam_radius: float = 0.25,
    max_push_iters: int = 3,
) -> np.ndarray:
    for i, p in enumerate(path):
        for _ in range(max_push_iters):
            dist, idx = kdtree.query(p, k=1)
            if dist >= cam_radius or dist == 0.0:
                break
            closest = kdtree.data[idx]
            dir_vec = p - closest
            norm = np.linalg.norm(dir_vec)
            if norm < 1e-6:
                dir_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                norm = 1.0
            dir_vec /= norm
            step = (cam_radius - dist) + 1e-3
            p = p + dir_vec * step
        path[i] = p
    return path


def clamp_to_ground_band(
    path: np.ndarray,
    pmin: np.ndarray,
    voxel_size_xy: float,
    y_min_map: np.ndarray,
    height_map: np.ndarray,
    floor_y: float,
    ceiling_y: Optional[float],
    ground_clearance: float = 1.0,
    max_above_surface: float = 3.0,
) -> np.ndarray:
    nx, nz = y_min_map.shape
    for i, p in enumerate(path):
        ix = int((p[0] - pmin[0]) / voxel_size_xy)
        iz = int((p[2] - pmin[2]) / voxel_size_xy)
        ix = np.clip(ix, 0, nx - 1)
        iz = np.clip(iz, 0, nz - 1)

        terrain_top = height_map[ix, iz]
        if not np.isfinite(terrain_top):
            terrain_top = floor_y

        y_low = terrain_top + ground_clearance
        if ceiling_y is None:
            y_high = terrain_top + max_above_surface
        else:
            y_high = min(terrain_top + max_above_surface, ceiling_y)

        if y_low > y_high:
            y_low = y_high

        p[1] = np.clip(p[1], y_low, y_high)
        path[i] = p

    return path


# ===================== GRAPH BUILDING & PATH PLANNING =====================

def build_viewpoint_graph_knn(
    viewpoints: List[Viewpoint],
    kdtree: cKDTree,
    k: int = 30,                # much larger
    max_edge_length: float | None = 0.6,  # limit jumps a bit
    cam_radius: float = 0.3,    # unused when we skip collision here
) -> Tuple[List[GraphEdge], List[List[GraphEdge]]]:
    n = len(viewpoints)
    positions = np.stack([vp.pos for vp in viewpoints], axis=0)

    edges: List[GraphEdge] = []
    adj: List[List[GraphEdge]] = [[] for _ in range(n)]

    for i in range(n):
        dists = np.linalg.norm(positions - positions[i], axis=1)
        order = np.argsort(dists)
        neighbors = order[1: 1 + min(k, n - 1)]

        for j in neighbors:
            dist = float(dists[j])
            if max_edge_length is not None and dist > max_edge_length:
                continue

            # TEMP: do NOT call is_segment_collision_free_3d here.
            e1 = GraphEdge(u=i, v=j, cost=dist)
            e2 = GraphEdge(u=j, v=i, cost=dist)
            edges.append(e1)
            edges.append(e2)
            adj[i].append(e1)
            adj[j].append(e2)

    return edges, adj


def greedy_orienteering_path(
    viewpoints: List[Viewpoint],
    adj: List[List[GraphEdge]],
    start_idx: int,
    max_path_length: float,
    smooth_turn_weight: float = 0.2,
) -> List[int]:
    """
    Greedy orienteering with score/length ratio and a soft penalty for sharp turns.
    """
    n = len(viewpoints)
    visited = np.zeros(n, dtype=bool)
    path = [start_idx]
    visited[start_idx] = True
    length_used = 0.0

    while True:
        cur = path[-1]

        # Previous direction for turn penalty
        dir_prev = None
        if len(path) >= 2:
            prev = path[-2]
            dir_prev = viewpoints[cur].pos - viewpoints[prev].pos
            nrm = np.linalg.norm(dir_prev)
            if nrm > 1e-6:
                dir_prev = dir_prev / nrm
            else:
                dir_prev = None

        best_j = None
        best_ratio = 0.0
        best_cost = None

        for e in adj[cur]:
            j = e.v
            cost = e.cost
            if length_used + cost > max_path_length:
                continue

            base_gain = 0.2  # tune this
            interest_gain = viewpoints[j].score
            gain = (base_gain + interest_gain) if not visited[j] else 0.0
            if cost <= 1e-6:
                continue

            # Turn smoothness penalty
            turn_penalty = 1.0
            if smooth_turn_weight > 0.0 and dir_prev is not None:
                dir_cur = viewpoints[j].pos - viewpoints[cur].pos
                nrm2 = np.linalg.norm(dir_cur)
                if nrm2 > 1e-6:
                    dir_cur /= nrm2
                    cosang = np.clip(np.dot(dir_prev, dir_cur), -1.0, 1.0)
                    angle = math.acos(cosang)  # 0..pi
                    turn_penalty = 1.0 + smooth_turn_weight * (angle / math.pi)

            ratio = gain / (cost * turn_penalty)

            if ratio > best_ratio:
                best_ratio = ratio
                best_j = j
                best_cost = cost

        if best_j is None or best_ratio <= 0.0:
            break

        path.append(best_j)
        length_used += best_cost
        visited[best_j] = True

    print(f"Final path length (nodes): {len(path)}, length used: {length_used:.2f}")
    return path


def build_collision_safe_chain(
    viewpoints: List[Viewpoint],
    adj: List[List[GraphEdge]],
    start_idx: int,
    max_nodes: int = 300,
    max_path_length: float = 25.0,
) -> List[int]:
    n = len(viewpoints)
    path_idx = [start_idx]
    visited = {start_idx}
    current = start_idx
    length_used = 0.0

    for _ in range(max_nodes - 1):
        if length_used >= max_path_length:
            break

        neighbors = adj[current]
        if not neighbors:
            # No neighbors left from here: stop
            break

        neighbors_sorted = sorted(neighbors, key=lambda e: e.cost)
        found_next = False
        for e in neighbors_sorted:
            nxt = e.v
            if nxt in visited:
                continue
            if length_used + e.cost > max_path_length:
                continue
            path_idx.append(nxt)
            visited.add(nxt)
            current = nxt
            length_used += e.cost
            found_next = True
            break

        if not found_next:
            break

    print(f"Final chain length (nodes): {len(path_idx)}, total length: {length_used:.2f}")
    return path_idx


# ===================== TRAJECTORY SMOOTHING & ORIENTATION =====================

def smooth_positions(
    positions: np.ndarray,
    smoothing: float = 0.1,
    n_interp: int = 400,
) -> np.ndarray:
    if positions.shape[0] < 4:
        return positions

    t = np.linspace(0.0, 1.0, positions.shape[0])
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    tck, _ = splprep([x, y, z], s=smoothing)
    t_new = np.linspace(0.0, 1.0, n_interp)
    x_new, y_new, z_new = splev(t_new, tck)
    return np.vstack([x_new, y_new, z_new]).T.astype(np.float32)


def resample_with_easing(path: np.ndarray, n_points: int) -> np.ndarray:
    if path.shape[0] < 2:
        return path

    diffs = np.linalg.norm(path[1:] - path[:-1], axis=1)
    dist_steps = np.concatenate([[0.0], np.cumsum(diffs)])
    total_len = dist_steps[-1]
    if total_len < 1e-6:
        return path

    t = np.linspace(0, 1, n_points)
    t_eased = (np.cos((t - 1) * math.pi) + 1) / 2.0
    s_new = t_eased * total_len

    x = np.interp(s_new, dist_steps, path[:, 0])
    y = np.interp(s_new, dist_steps, path[:, 1])
    z = np.interp(s_new, dist_steps, path[:, 2])

    return np.vstack([x, y, z]).T.astype(np.float32)


def smooth_quaternions(quats: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    q_smooth = quats.copy()
    for i in range(1, len(q_smooth)):
        dot = np.dot(q_smooth[i], q_smooth[i - 1])
        if dot < 0:
            q_smooth[i] = -q_smooth[i]
    q_filtered = gaussian_filter1d(q_smooth, sigma=sigma, axis=0)
    norms = np.linalg.norm(q_filtered, axis=1, keepdims=True)
    return q_filtered / (norms + 1e-8)


def compute_exploration_targets(
    path_points: np.ndarray,
    interest_map: np.ndarray,
    height_map: np.ndarray,
    pmin: np.ndarray,
    voxel_size: float,
    window_radius_m: float = 30.0,
) -> np.ndarray:
    targets = []
    n_pts = len(path_points)
    nx, nz = interest_map.shape
    r_px = int(window_radius_m / voxel_size)

    for i in range(n_pts):
        pos = path_points[i]
        cx = int((pos[0] - pmin[0]) / voxel_size)
        cz = int((pos[2] - pmin[2]) / voxel_size)
        cx = np.clip(cx, 0, nx - 1)
        cz = np.clip(cz, 0, nz - 1)

        x0, x1 = max(0, cx - r_px), min(nx, cx + r_px)
        z0, z1 = max(0, cz - r_px), min(nz, cz + r_px)

        patch_int = interest_map[x0:x1, z0:z1]
        if patch_int.size == 0 or patch_int.max() < 0.1:
            next_i = min(i + 10, n_pts - 1)
            tangent_target = path_points[next_i]
            targets.append(tangent_target)
            continue

        local_max = patch_int.max()
        candidates = np.argwhere(patch_int > 0.7 * local_max)

        if len(candidates) == 0:
            targets.append(path_points[min(i + 1, n_pts - 1)])
            continue

        best_cand = None
        best_score = -1.0

        for c in candidates:
            gx, gz = x0 + c[0], z0 + c[1]
            wx = pmin[0] + (gx + 0.5) * voxel_size
            wz = pmin[2] + (gz + 0.5) * voxel_size
            wy = height_map[gx, gz]
            if not np.isfinite(wy):
                wy = pos[1]

            dist_h = np.sqrt((wx - pos[0]) ** 2 + (wz - pos[2]) ** 2)
            angle_vert = abs(math.atan2(wy - pos[1], dist_h + 1e-6))
            score = patch_int[c[0], c[1]] - (angle_vert * 2.0)

            if score > best_score:
                best_score = score
                best_cand = np.array([wx, wy, wz], dtype=np.float32)

        targets.append(best_cand)

    return np.array(targets)


def compute_fpv_orientation(
    path_points: np.ndarray,
    interest_map: np.ndarray,
    height_map: np.ndarray,
    pmin: np.ndarray,
    voxel_size: float,
    look_ahead_frames: int = 10,
    blend_strength: float = 0.4,
    smooth_sigma: float = 10.0,
) -> List[dict]:
    raw_targets = compute_exploration_targets(
    path_points, interest_map, height_map, pmin, voxel_size,
    window_radius_m=5.0,   # was 30.0, focus more locally
)

    smooth_targets = np.zeros_like(raw_targets)
    smooth_targets[:, 0] = gaussian_filter1d(raw_targets[:, 0], sigma=3.0)
    smooth_targets[:, 1] = gaussian_filter1d(raw_targets[:, 1], sigma=3.0)
    smooth_targets[:, 2] = gaussian_filter1d(raw_targets[:, 2], sigma=3.0)

    n_points = len(path_points)
    raw_quats = []

    for i in range(n_points):
        pos = path_points[i]
        next_idx = min(i + look_ahead_frames, n_points - 1)
        v_forward = path_points[next_idx] - pos
        v_forward /= (np.linalg.norm(v_forward) + 1e-6)

        target = smooth_targets[i]
        v_interest = target - pos
        dist_interest = np.linalg.norm(v_interest)

        if dist_interest > 1e-3:
            v_interest /= dist_interest
        else:
            v_interest = v_forward

        final_forward = (1.0 - 0.2) * v_forward + 0.8 * v_interest
        final_forward /= (np.linalg.norm(final_forward) + 1e-6)

        right = np.cross(final_forward, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(right) < 0.1:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)

        up = np.cross(right, final_forward)
        up /= np.linalg.norm(up)

        rot_mat = np.column_stack([right, up, -final_forward])
        q = R.from_matrix(rot_mat).as_quat()
        raw_quats.append(q)

    raw_quats = np.array(raw_quats)
    traj_smooth = smooth_quaternions(raw_quats, sigma=5.0)

    output = []
    for i in range(n_points):
        p = path_points[i]
        q = traj_smooth[i]
        output.append(
            {
                "position": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
                "quaternion": {
                    "x": float(q[0]),
                    "y": float(q[1]),
                    "z": float(q[2]),
                    "w": float(q[3]),
                },
            }
        )

    return output

def build_fpv_sweep_path(
    viewpoints: List[Viewpoint],
    scene_kdtree: cKDTree,
    start_idx: int,
    max_step: float | None = None,   # allow unlimited step if None
    max_path_length: float = 25.0,
) -> List[int]:
    """
    FPV-style sweep:
    - Order viewpoints along X (island long axis).
    - Move monotonically in that order from the start.
    - Respect only the global length budget and collision checks.
    """
    n = len(viewpoints)
    positions = np.stack([vp.pos for vp in viewpoints], axis=0)
    x_coords = positions[:, 0]

    x_min, x_max = x_coords.min(), x_coords.max()
    x_start = x_coords[start_idx]

    # Choose direction that covers the larger remaining span
    if abs(x_start - x_min) < abs(x_start - x_max):
        order = np.argsort(x_coords)      # sweep towards +X
    else:
        order = np.argsort(-x_coords)     # sweep towards -X

    used = np.zeros(n, dtype=bool)
    used[start_idx] = True
    path = [start_idx]
    length_used = 0.0

    for idx in order:
        if used[idx]:
            continue

        last = path[-1]
        p_last = positions[last]
        p_next = positions[idx]
        step = np.linalg.norm(p_next - p_last)

        # Optional local step limit; keep it loose or disable via None
        if max_step is not None and step > max_step:
            continue

        # Relaxed collision check (smaller radius, fewer samples)
        if not is_segment_collision_free_3d(
            p_last, p_next, scene_kdtree, cam_radius=0.18, n_samples=10
        ):
            continue

        if length_used + step > max_path_length:
            break

        path.append(idx)
        used[idx] = True
        length_used += step

    print(f"[FPV] sweep path nodes={len(path)}, length={length_used:.2f}")
    return path


def build_global_axis_sweep(
    viewpoints: List[Viewpoint],
    path_length_budget: float = 25.0,
) -> List[int]:
    """
    Global 1D sweep over the island along X:
    - Sort all viewpoints by X (min -> max).
    - Walk through that order, accumulating length until the budget is used.
    - Ignores collisions here; push_out_of_geometry will fix them later.
    """
    n = len(viewpoints)
    positions = np.stack([vp.pos for vp in viewpoints], axis=0)
    x_coords = positions[:, 0]

    order = np.argsort(x_coords)  # from min X to max X
    path_idx = [int(order[0])]
    length_used = 0.0

    for idx in order[1:]:
        idx = int(idx)
        prev = path_idx[-1]
        p_prev = positions[prev]
        p_next = positions[idx]
        step = float(np.linalg.norm(p_next - p_prev))

        if length_used + step > path_length_budget:
            # cannot take this step, but maybe some shorter later steps;
            # skip instead of breaking to allow more coverage.
            continue

        path_idx.append(idx)
        length_used += step

    print(f"[SWEEP] global axis path nodes={len(path_idx)}, length={length_used:.2f}")
    return path_idx

def build_boustrophedon_path(
    viewpoints: List[Viewpoint],
    start_idx: int,
    path_length_budget: float = 25.0,
    strip_width: float = 2.0, 
) -> List[int]:
    """
    Stricter Boustrophedon sweep.
    1. Assign points to X-strips.
    2. WITHIN each strip, sort strictly by Z (ignoring X variations).
    3. Connect strips end-to-end.
    """
    n = len(viewpoints)
    positions = np.stack([vp.pos for vp in viewpoints], axis=0)
    
    # 1. Determine Strips
    x_coords = positions[:, 0]
    x_min, x_max = x_coords.min(), x_coords.max()
    
    # Calculate number of strips based on width
    # Ensure at least 1 strip
    n_strips = max(1, int(math.ceil((x_max - x_min) / strip_width)))
    
    # Create buckets for each strip
    strips = [[] for _ in range(n_strips)]
    
    for i in range(n):
        strip_id = int((x_coords[i] - x_min) / strip_width)
        strip_id = min(strip_id, n_strips - 1) # Clamp to last bucket
        strips[strip_id].append(i)

    # 2. Build the "Perfect" Geometric Path Order
    full_order = []
    
    for s_id in range(n_strips):
        # Get indices in this strip
        indices = strips[s_id]
        if not indices:
            continue
            
        # Sort them by Z coordinate
        # If strip_id is even (0, 2, 4...), go South -> North (Ascending Z)
        # If strip_id is odd (1, 3, 5...), go North -> South (Descending Z)
        if s_id % 2 == 0:
            indices.sort(key=lambda idx: positions[idx, 2])
        else:
            indices.sort(key=lambda idx: positions[idx, 2], reverse=True)
            
        full_order.extend(indices)

    # 3. Find Start and Traverse
    try:
        start_ptr = full_order.index(start_idx)
    except ValueError:
        start_ptr = 0
        
    # Decide direction: traverse towards the longer remaining tail
    if start_ptr < len(full_order) / 2:
        final_order = full_order[start_ptr:]
    else:
        final_order = full_order[start_ptr::-1]

    # 4. Construct Path with Budget
    path = [final_order[0]]
    length_used = 0.0
    
    for i in range(1, len(final_order)):
        curr = final_order[i]
        prev = path[-1]
        
        dist = np.linalg.norm(positions[curr] - positions[prev])
        
        # Skip huge jumps (e.g. between disjoint islands of points)
        if dist > 5.0 * strip_width:
             continue
             
        if length_used + dist > path_length_budget:
            break
            
        path.append(curr)
        length_used += dist

    print(f"[SWEEP] Snake path nodes={len(path)}, length={length_used:.2f}")
    return path


# ===================== DEBUG PLOTTING =====================

def debug_plot_maps(
    density_map: np.ndarray,
    interest_map: np.ndarray,
    occ_2d: np.ndarray,
    positions: Optional[List[np.ndarray]] = None,
    path_positions: Optional[np.ndarray] = None,
    title_prefix: str = "",
    save_path: Optional[str] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skipping debug plots.")
        return

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].set_title(f"{title_prefix} density")
    im0 = axs[0].imshow(density_map.T, origin="lower", cmap="viridis")
    plt.colorbar(im0, ax=axs[0], fraction=0.046)

    axs[1].set_title(f"{title_prefix} interest + occupancy")
    im1 = axs[1].imshow(interest_map.T, origin="lower", cmap="magma")
    axs[1].contour(occ_2d.T, levels=[0.5], colors="cyan", linewidths=0.8)
    plt.colorbar(im1, ax=axs[1], fraction=0.046)

    axs[2].set_title(f"{title_prefix} candidates + path (XZ)")
    if positions is not None and len(positions) > 0:
        xs = [p[0] for p in positions]
        zs = [p[2] for p in positions]
        axs[2].scatter(xs, zs, s=3, c="white", alpha=0.4, label="candidates")
    if path_positions is not None and path_positions.shape[0] > 0:
        axs[2].plot(path_positions[:, 0], path_positions[:, 2],
                    "-r", linewidth=2, label="path")
        axs[2].scatter(path_positions[0, 0], path_positions[0, 2],
                       c="green", s=30, label="start")
    axs[2].legend(loc="upper right")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[DEBUG] Saved plot to {save_path}")
    else:
        plt.show()


# ===================== MAIN PIPELINE =====================

def generate_autonomous_trajectory(
    ply_file: str,
    start_pos: np.ndarray,
    output_json: str = "trajectory_autonomous.json",
    voxel_size_xy: float = 0.08,
    path_length_budget: float = 25.0,
    n_samples: int = 600,
    spline_points: int = 600,
    min_path_nodes: int = 10,
    debug_plots: bool = False,
    debug_plot_path: Optional[str] = None,
) -> None:
    # 1) Load & clean
    raw_pts = load_point_cloud(ply_file)
    pts_clean = remove_statistical_outliers(raw_pts, nb_neighbors=20, std_ratio=2.0)

    # 2) ROI
    pts, pmin, pmax = extract_dense_roi(
        pts_clean,
        voxel_size_xy=voxel_size_xy,
        min_rel_density=0.3,
    )

    scene_kdtree = build_scene_kdtree(pts)

    # 3) Height & interest
    height_map, y_min_map, density_map, interest_map, (nx, nz), vox_xy = \
        build_height_and_interest_maps(pts, pmin, pmax, voxel_size_xy=voxel_size_xy)

    # 4) Floor / ceiling detection (then override to outdoor)
    floor_y, ceiling_y, is_indoor = detect_floor_and_ceiling(pts)
    is_indoor = False
    ceiling_y = None
    print(f"ROI floor_y, ceiling_y, is_indoor: {floor_y}, {ceiling_y}, {is_indoor}")

    # 5) 2D occupancy
    occ_2d = build_2d_occupancy_from_height(
        y_min_map,
        height_map,
        floor_y,
        ceiling_y,
        density_map=density_map,
        clearance=0.15,
        min_free_fraction=0.1,
    )
    print("floor_y, ceiling_y, is_indoor:", floor_y, ceiling_y, is_indoor)

    # Distance to occupied cells (for shoreline-like band)
    free_mask = ~occ_2d
    dist_to_occ = distance_transform_edt(free_mask) * voxel_size_xy

    # 6) Candidate positions & viewpoints
    positions = sample_candidate_positions(
        occ_2d,
        height_map,
        y_min_map,
        pmin,
        voxel_size_xy,
        floor_y,
        ceiling_y,
        is_indoor=is_indoor,
        interest_map=interest_map,
        dist_to_occ=dist_to_occ,
        n_samples=n_samples,
        clearance=0.05,              # keep bases low
        preferred_dist_band=(1.0, 6.0),  # closer to walls/shoreline
        interest_weight=0.8,
        shoreline_weight=0.2,
    )

    viewpoints = build_viewpoints(
        positions,
        interest_map,
        pmin,
        voxel_size_xy,
        up_global=np.array([0.0, 1.0, 0.0]),
        interest_radius=1.0,
    )

    print(f"Viewpoints: {len(viewpoints)}")

    # 7) Select start node (must be clear of geometry)
    pos_arr = np.stack([vp.pos for vp in viewpoints], axis=0)
    dists = np.linalg.norm(pos_arr - start_pos[None, :], axis=1)
    sorted_indices = np.argsort(dists)

    start_idx = -1
    for idx in sorted_indices:
        dist_to_obs, _ = scene_kdtree.query(viewpoints[idx].pos, k=1)
        if dist_to_obs > 0.3:
            start_idx = int(idx)
            break

    if start_idx == -1:
        print("[WARN] Could not find any collision-free start node! Using closest.")
        start_idx = int(sorted_indices[0])

    print(f"Selected start_idx: {start_idx} at {viewpoints[start_idx].pos}")

    # 8) Graph & paths (3D collision-safe)
    edges, adj = build_viewpoint_graph_knn(
        viewpoints,
        kdtree=scene_kdtree,
        k=20,
        max_edge_length=0.5,
        cam_radius=0.25,
    )

    print(f"Edges: {len(edges)}")
    n = len(viewpoints)
    adj_counts = [0] * n
    for e in edges:
        adj_counts[e.u] += 1
    print(f"Neighbors of start_idx {start_idx}: {adj_counts[start_idx]}")
    deg = np.array(adj_counts)
    print(f"[GRAPH] degree min/mean/max: {deg.min()} / {deg.mean():.2f} / {deg.max()}")

    if deg[start_idx] == 0:
        print("[WARN] Selected start node has no neighbors; choosing alternative start.")
        valid = np.where(deg > 0)[0]
        if valid.size > 0:
            d_valid = np.linalg.norm(pos_arr[valid] - start_pos[None, :], axis=1)
            new_idx = valid[np.argmin(d_valid)]
            print(f"[INFO] Switching start_idx {start_idx} -> {new_idx}")
            start_idx = int(new_idx)
        else:
            print("[ERROR] All viewpoints are isolated; try reducing cam_radius or "
                  "max_edge_length constraints.")

    # Graph diagnostics
    if len(edges) > 0:
        edge_lengths = np.array([e.cost for e in edges])
        print(f"[GRAPH] N_viewpoints={len(viewpoints)}, N_edges={len(edges)}")
        print(f"[GRAPH] edge length mean/std/min/max: "
              f"{edge_lengths.mean():.3f}/{edge_lengths.std():.3f}/"
              f"{edge_lengths.min():.3f}/{edge_lengths.max():.3f}")

    # Option A: Coverage Sweep (Snake Pattern)
    # Use this if you want to see the whole island smoothly
    path_idx = build_boustrophedon_path(
        viewpoints,
        start_idx=start_idx, # Now we respect the start node!
        path_length_budget=path_length_budget,
       strip_width=1.5 # Tune: roughly 10-20% of island width
    )

    # Option B: Interest-Based Graph Search (The "Smart" Way)
    # Use this if you specifically want the "dense/interesting" parts
    # Uncomment below to use instead of sweep:
    # path_idx = greedy_orienteering_path(
       #   viewpoints,
      #    adj,
      #    start_idx,
       #   max_path_length=path_length_budget,
       #   smooth_turn_weight=0.5
     # )


    # 9) Extract polyline positions
    path_positions = np.stack([viewpoints[i].pos for i in path_idx], axis=0)

    # Path interest diagnostics
    path_interest = np.array([viewpoints[i].score for i in path_idx])
    print(f"[PATH] N_nodes={len(path_idx)}, score mean/med/max: "
          f"{path_interest.mean():.3f}/{np.median(path_interest):.3f}/{path_interest.max():.3f}")

    scene_top = float(height_map.max())
    if ceiling_y is None or ceiling_y < scene_top - 0.2:
        print("Ignoring ceiling constraints (Outdoor mode or unreliable ceiling).")
        effective_ceiling = None
    else:
        effective_ceiling = ceiling_y

    # 10) Smooth, clamp, push out, easing
    path_smooth = smooth_positions(path_positions, smoothing=1.0, n_interp=spline_points)

    # tight band above terrain
    path_smooth = clamp_to_ground_band(
        path_smooth,
        pmin,
        voxel_size_xy,
        y_min_map,
        height_map,
        floor_y,
        effective_ceiling,
        ground_clearance=0.05,   # very low over surface
        max_above_surface=0.20,  # tight FPV band
    )

    path_smooth = push_out_of_geometry(path_smooth, scene_kdtree, cam_radius=0.25, max_push_iters=4)
    path_smooth = resample_with_easing(path_smooth, n_points=spline_points)
    
    # Apply Y-smoothing FIRST (so we smooth the general trend)
    path_smooth[:, 1] = gaussian_filter1d(path_smooth[:, 1], sigma=5.0) 

    # THEN Clamp strictly to the ground band LAST
    # This forces the smoothed line back into the safe corridor
    path_smooth = clamp_to_ground_band(
        path_smooth,
        pmin,
        voxel_size_xy,
        y_min_map,
        height_map,
        floor_y,
        effective_ceiling,
        ground_clearance=0.05,
        max_above_surface=0.25,
    )

    print(f"[ALTITUDE] y min/mean/max: "
          f"{path_smooth[:,1].min():.3f}/"
          f"{path_smooth[:,1].mean():.3f}/"
          f"{path_smooth[:,1].max():.3f}")

    # Optional debug plots
    if debug_plots:
        debug_plot_maps(
            density_map=density_map,
            interest_map=interest_map,
            occ_2d=occ_2d,
            positions=positions,
            path_positions=path_smooth,
            title_prefix="island",
            save_path=debug_plot_path,
        )

    # 11) Orientation
    traj = compute_fpv_orientation(
        path_smooth,
        interest_map=interest_map,
        height_map=height_map,
        pmin=pmin,
        voxel_size=voxel_size_xy,
        blend_strength=0.8,
        smooth_sigma=3.0,
    )

    # 12) Save
    with open(output_json, "w") as f:
        json.dump(traj, f, indent=2)

    c_val = effective_ceiling if effective_ceiling is not None else 0.0
    print(f"Indoor: {is_indoor}, floor_y={floor_y:.3f}, ceiling_y={c_val:.3f}")
    print(f"✓ Generated {len(traj)} poses")
    print(f"Saved to: {output_json}")
    print(f"Bounds X:[{pmin[0]:.2f}, {pmax[0]:.2f}] Z:[{pmin[2]:.2f}, {pmax[2]:.2f}]")


if __name__ == "__main__":
    # Example: set your own start coordinate in model space
    start = np.array([0.048, 0.075, -0.603], dtype=np.float32)
    ply_file = "outdoor-standard.ply"

    generate_autonomous_trajectory(
        ply_file=ply_file,
        start_pos=start,
        output_json="trajectory_autonomous.json",
        voxel_size_xy=0.08,
        path_length_budget=40.0,
        n_samples=100,
        spline_points=400,
        min_path_nodes=80,
        debug_plots=True,            # set False if you don't want plots / need headless
        debug_plot_path="debug_island.png",
    )

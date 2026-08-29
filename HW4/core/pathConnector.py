import json
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

def convert_coordinate_system(pos, target):
    """
    Applies a 180-degree rotation around the X-axis to match
    the mesh.rotation.x = Math.PI in your Three.js script.
    
    Rule: (x, y, z) -> (x, -y, -z)
    """
    # Flip Y and Z
    pos_new = np.array([pos[0], -pos[1], -pos[2]])
    target_new = np.array([target[0], -target[1], -target[2]])
    return pos_new, target_new

def quaternion_from_lookat(position, target, up=np.array([0, 1, 0])):
    """
    Computes the Three.js compatible quaternion.
    Three.js Camera looks down -Z.
    """
    # Forward vector (Camera to Target)
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    
    # Right vector
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    
    # Recompute Up to ensure orthogonality
    new_up = np.cross(forward, right)
    
    # Create Matrix: [Right, Up, -Forward]
    # (Negative forward because Three.js cameras look down -Z)
    rot_mat = np.column_stack([right, new_up, -forward])
    
    # Fix determinant if necessary (check for reflection)
    if np.linalg.det(rot_mat) < 0:
        rot_mat[:, 0] *= -1  # Flip X axis
        
    r = Rotation.from_matrix(rot_mat)
    return r.as_quat()

def process_trajectory(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    track = data['animTracks'][0]
    times = np.array(track['keyframes']['times'])
    
    # Extract raw values
    raw_pos = np.array(track['keyframes']['values']['position']).reshape(-1, 3)
    raw_tgt = np.array(track['keyframes']['values']['target']).reshape(-1, 3)
    
    # --- TRANSFORM COORDINATES ---
    # We transform the raw keyframes BEFORE interpolation to match the model's rotation
    fixed_pos = []
    fixed_tgt = []
    for p, t in zip(raw_pos, raw_tgt):
        p_new, t_new = convert_coordinate_system(p, t)
        fixed_pos.append(p_new)
        fixed_tgt.append(t_new)
        
    fixed_pos = np.array(fixed_pos)
    fixed_tgt = np.array(fixed_tgt)

    # --- INTERPOLATION ---
    # Create splines
    cs_pos = CubicSpline(times, fixed_pos)
    cs_tgt = CubicSpline(times, fixed_tgt)
    
    # Generate frames at 30 FPS
    fps = 30
    total_time = times[-1]
    frame_count = int(total_time * fps)
    
    out_data = []
    
    for i in range(frame_count):
        t = i / fps
        
        # Get interpolated position and target
        curr_pos = cs_pos(t)
        curr_tgt = cs_tgt(t)
        
        # Calculate rotation
        # We use [0, -1, 0] as up because we flipped the world upside down
        # transforming the original Up [0, 1, 0] -> [0, -1, 0]
        quat = quaternion_from_lookat(curr_pos, curr_tgt, up=np.array([0, -1, 0]))
        
        out_data.append({
            "frame": i,
            "position": {"x": curr_pos[0], "y": curr_pos[1], "z": curr_pos[2]},
            "quaternion": {"x": quat[0], "y": quat[1], "z": quat[2], "w": quat[3]},
            "fov": data['camera']['fov']
        })

    with open(output_file, 'w') as f:
        json.dump(out_data, f, indent=2)
    
    print(f"Converted {len(out_data)} frames.")
    print(f"Sample POS (Frame 0): {out_data[0]['position']}")

if __name__ == "__main__":
    process_trajectory('settings.json', 'trajectory_autonomous.json')

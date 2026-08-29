#!/usr/bin/env python3
"""
Simple launcher script for the indoor path generator.
This makes it easier to run without typing long terminal commands.

Usage:
    python core/run_path_generator.py
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.generate_indoor_path import generate_indoor_trajectory, load_ply_points

def main():
    print("=" * 60)
    print("Indoor Path Generator - Simple Launcher")
    print("=" * 60)
    print()
    
    # Try to load config
    config_path = "core/config.json"
    config = None
    scene_config = None
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✓ Found config file: {config_path}")
            
            scene_name = config.get("default_scene", "conference-hall")
            if "scenes" in config and scene_name in config["scenes"]:
                scene_config = config["scenes"][scene_name]
                print(f"✓ Using scene: {scene_name}")
            else:
                print(f"⚠ Scene '{scene_name}' not found in config")
        except Exception as e:
            print(f"⚠ Could not load config: {e}")
    else:
        print(f"⚠ Config file not found: {config_path}")
        print("  Creating example config...")
        create_example_config(config_path)
        print(f"  ✓ Created {config_path}")
        print("  Please edit it with your settings and run again.")
        return
    
    if not scene_config:
        print("\n❌ No scene configuration found. Please check config.json")
        return
    
    # Get settings from config
    ply_file = scene_config.get("ply")
    output_file = scene_config.get("output")
    n_frames = scene_config.get("frames", 600)
    fov = scene_config.get("fov", 60.0)
    
    start_pos_config = scene_config.get("start_position", {})
    auto_detect = start_pos_config.get("auto_detect", True)
    
    start_x = start_pos_config.get("x")
    start_y = start_pos_config.get("y")
    start_z = start_pos_config.get("z")
    
    # Validate
    if not ply_file or not output_file:
        print("❌ Missing required settings in config (ply or output)")
        return
    
    print(f"\nSettings:")
    print(f"  PLY file: {ply_file}")
    print(f"  Output: {output_file}")
    print(f"  Frames: {n_frames}")
    print(f"  FOV: {fov}°")
    
    if auto_detect:
        print(f"  Start position: Auto-detect (smart room detection)")
    elif start_x is not None or start_y is not None or start_z is not None:
        print(f"  Start position: ({start_x}, {start_y}, {start_z})")
    else:
        print(f"  Start position: Auto-detect (no coordinates provided)")
    
    print("\n" + "=" * 60)
    print("Generating trajectory...")
    print("=" * 60)
    print()
    
    # Determine start position
    if start_x is not None or start_y is not None or start_z is not None:
        # User provided coordinates
        if start_x is None:
            start_x = 0.0
        if start_y is None:
            # Auto-detect Y
            try:
                pts = load_ply_points(ply_file)
                floor_y = float(np.percentile(pts[:, 1], 5))
                room_height = float(np.percentile(pts[:, 1], 95)) - floor_y
                if room_height > 0:
                    scale = room_height / 2.5
                    start_y = floor_y + 1.6 * scale
                else:
                    start_y = floor_y + room_height * 0.4
            except:
                start_y = 0.0
        if start_z is None:
            start_z = 0.0
        
        import numpy as np
        start_pos = np.array([start_x, start_y, start_z], dtype=np.float32)
        auto_detect = False
    else:
        start_pos = None
        auto_detect = True
    
    # Generate trajectory
    try:
        generate_indoor_trajectory(
            ply_file=ply_file,
            start_pos=start_pos,
            output_json=output_file,
            n_frames=n_frames,
            fov=fov,
            auto_detect_start=auto_detect
        )
        print("\n" + "=" * 60)
        print("✓ Success! Trajectory generated.")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def create_example_config(config_path):
    """Create an example config file."""
    example_config = {
        "scenes": {
            "conference-hall": {
                "ply": "scenes/conference-hall/ConferenceHall.ply",
                "output": "scenes/conference-hall/trajectory_indoor.json",
                "frames": 600,
                "fov": 60.0,
                "start_position": {
                    "x": None,
                    "y": None,
                    "z": None,
                    "auto_detect": True,
                    "comment": "Set auto_detect to false and provide x, y, z to use specific coordinates"
                }
            }
        },
        "default_scene": "conference-hall"
    }
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(example_config, f, indent=2)


if __name__ == "__main__":
    import numpy as np
    sys.exit(main())


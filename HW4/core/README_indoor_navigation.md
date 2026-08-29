# Indoor Navigation Path Generator

A smart program for generating cinematic camera paths in 3D Gaussian Splatting indoor scenes (e.g., ConferenceHall.ply). The generated paths avoid walls, create smooth cinematic movements, and explore the interior space intelligently.

## Features

- **Collision Avoidance**: Automatically detects and avoids walls and obstacles
- **Cinematic Framing**: Creates smooth, natural camera movements with proper orientation
- **Intelligent Exploration**: Explores the space in multiple phases (forward movement, circular exploration, zigzag patterns)
- **Custom Start Position**: Supports user-specified starting coordinates
- **Compatible Output**: Generates JSON trajectories compatible with `index.html` viewer

## Requirements

```bash
pip install numpy open3d scipy
```

## Usage

### Basic Usage

Generate a trajectory with auto-detected start position:

```bash
python core/generate_indoor_path.py \
    --ply scenes/conference-hall/ConferenceHall.ply \
    --output scenes/conference-hall/trajectory_indoor.json
```

### With Custom Start Position

Specify exact starting coordinates:

```bash
python core/generate_indoor_path.py \
    --ply scenes/conference-hall/ConferenceHall.ply \
    --start-x 0.0 \
    --start-y 1.5 \
    --start-z 0.0 \
    --output scenes/conference-hall/trajectory_indoor.json
```

### Advanced Options

```bash
python core/generate_indoor_path.py \
    --ply scenes/conference-hall/ConferenceHall.ply \
    --start-x 0.0 \
    --start-y 1.5 \
    --start-z 0.0 \
    --output scenes/conference-hall/trajectory_indoor.json \
    --frames 800 \
    --fov 65.0
```

## Parameters

- `--ply`: Path to the PLY file (required)
- `--start-x`: Starting X coordinate (default: 0.0)
- `--start-y`: Starting Y coordinate (auto-detected if not provided)
- `--start-z`: Starting Z coordinate (default: 0.0)
- `--output`: Output JSON file path (required)
- `--frames`: Number of frames to generate (default: 600)
- `--fov`: Field of view in degrees (default: 60.0)

## How It Works

1. **Point Cloud Loading**: Loads and cleans the PLY file, removing outliers
2. **Space Analysis**: Detects floor, ceiling, and room dimensions
3. **Collision Detection**: Builds a KD-tree for fast collision checking
4. **Path Generation**: Creates a multi-phase exploration path:
   - Phase 1 (0-25%): Gentle forward movement from start
   - Phase 2 (25-50%): Circular exploration around center
   - Phase 3 (50-75%): Zigzag pattern for coverage
   - Phase 4 (75-100%): Return journey with exploration
5. **Camera Orientation**: Computes cinematic camera orientations that look ahead along the path
6. **Smoothing**: Applies Gaussian smoothing for natural motion
7. **Export**: Saves trajectory in JSON format compatible with `index.html`

## Viewing the Trajectory

1. Open `index.html` in a web browser
2. Select "Conference Hall (Indoor)" from the scene dropdown
3. The trajectory will automatically load and play

## Algorithm Details

### Collision Avoidance

- Uses KD-tree for fast nearest neighbor queries
- Maintains minimum clearance (0.3m camera radius + 0.2m safety margin)
- Automatically pushes positions away from obstacles if too close

### Cinematic Framing

- Camera looks ahead along the path (5 frames look-ahead)
- Blends path direction with room center for more cinematic feel
- Limits vertical tilt to avoid extreme angles
- Smooth quaternion interpolation for natural head movements

### Path Smoothing

- Gaussian filtering on position (σ=5.0)
- Quaternion smoothing with shortest-path correction (σ=8.0)
- Preserves collision-free guarantees after smoothing

## Example Output

The generated JSON file contains an array of frames:

```json
[
  {
    "position": {"x": 0.0, "y": 1.5, "z": 0.0},
    "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
  },
  ...
]
```

## Tips

1. **Finding Start Coordinates**: 
   - Load the PLY in a 3D viewer first to identify good starting positions
   - Or use the auto-detection feature (omit `--start-y`)

2. **Adjusting Path Length**:
   - Increase `--frames` for longer, more detailed exploration
   - Decrease for faster, more focused paths

3. **Customizing Exploration**:
   - Edit `generate_indoor_exploration_path()` in the script to modify path patterns
   - Adjust `exploration_radius` parameter for different room sizes

## Troubleshooting

**Issue**: Path goes through walls
- **Solution**: The collision detection should prevent this, but if it happens, try:
  - Increasing the `cam_radius` parameter in `is_collision_free()`
  - Adjusting the `clearance` parameter

**Issue**: Camera height is wrong
- **Solution**: The script auto-detects floor/ceiling. If incorrect:
  - Manually specify `--start-y` with the desired height
  - Check the PLY file coordinate system

**Issue**: Path is too jerky
- **Solution**: Increase smoothing parameters:
  - `sigma` in `smooth_path()` (default: 5.0)
  - `sigma` in `smooth_quaternions()` (default: 8.0)

## License

Part of the 3DGS exploration project.


# Configuration Guide

## Easy Way: Using Config File

### 1. Edit `core/config.json`

Open the config file and modify the settings:

```json
{
  "scenes": {
    "conference-hall": {
      "ply": "scenes/conference-hall/ConferenceHall.ply",
      "output": "scenes/conference-hall/trajectory_indoor.json",
      "frames": 600,
      "fov": 60.0,
      "start_position": {
        "x": null,
        "y": null,
        "z": null,
        "auto_detect": true
      }
    }
  }
}
```

### 2. Set Custom Start Coordinates

To use specific coordinates, set `auto_detect` to `false` and provide coordinates:

```json
"start_position": {
  "x": 12.847,
  "y": -1.500,
  "z": 16.425,
  "auto_detect": false
}
```

### 3. Run the Generator

**Option A: Simple launcher (easiest)**
```bash
python core/run_path_generator.py
```

**Option B: Using config with command**
```bash
python core/generate_indoor_path.py --config
```

**Option C: Command line (overrides config)**
```bash
python core/generate_indoor_path.py \
    --ply scenes/conference-hall/ConferenceHall.ply \
    --output test.json \
    --frames 60 \
    --start-x 12.847 --start-y -1.5 --start-z 16.425
```

## Auto-Detection (Recommended)

If you set `auto_detect: true` (or leave coordinates as `null`), the algorithm will:
- Analyze the point cloud distribution
- Detect all rooms and spaces
- Select the largest room (main conference room)
- Start inside that room automatically
- Generate a path that stays within the room

This is the smartest option and works best for most cases!

## Examples

### Example 1: Auto-detect (recommended)
```json
"start_position": {
  "x": null,
  "y": null,
  "z": null,
  "auto_detect": true
}
```

### Example 2: Custom coordinates
```json
"start_position": {
  "x": -19.966,
  "y": 0.629,
  "z": 26.053,
  "auto_detect": false
}
```

### Example 3: Partial coordinates (X and Z specified, Y auto-detected)
```json
"start_position": {
  "x": 12.847,
  "y": null,
  "z": 16.425,
  "auto_detect": false
}
```


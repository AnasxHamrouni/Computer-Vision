# Object Detection for 3D Gaussian Splatting Scenes

This module provides both **2D** (rendered frames) and **3D** (point cloud) object detection capabilities.

## Features

### 3D Object Detection
- **Geometric clustering**: Detects objects by clustering point cloud based on spatial proximity
- **Height-based detection**: Finds buildings/structures by detecting high-elevation clusters
- **Automatic classification**: Classifies objects as buildings, trees, vehicles, structures, or terrain based on geometric properties
- **Visualization**: Creates 3D visualizations showing detected objects

### 2D Object Detection
- **YOLO integration**: Uses YOLOv8 for detecting objects in rendered frames
- **Batch processing**: Processes multiple frames efficiently
- **2D to 3D projection**: Projects 2D detections to 3D space using camera poses

## Installation

```bash
# Install YOLO (for 2D detection)
pip install ultralytics

# Other dependencies should already be installed
pip install numpy scipy scikit-learn matplotlib opencv-python plyfile
```

## Usage

### 3D Object Detection (Point Cloud)

```bash
# Basic usage
python core/detect_objects.py --ply outdoor-standard.ply --mode 3d

# With custom output directory
python core/detect_objects.py --ply outdoor-standard.ply --mode 3d --output_dir my_detections

# Skip visualization
python core/detect_objects.py --ply outdoor-standard.ply --mode 3d --no_viz
```

### 2D Object Detection (Rendered Frames)

First, you need to render frames from your trajectory. Then:

```bash
# With rendered frames
python core/detect_objects.py --ply outdoor-standard.ply \
    --trajectory trajectory_autonomous.json \
    --image_dir path/to/rendered/frames \
    --mode 2d
```

### Combined (2D + 3D)

```bash
python core/detect_objects.py --ply outdoor-standard.ply \
    --trajectory trajectory_autonomous.json \
    --image_dir path/to/rendered/frames \
    --mode both
```

### Integration with Trajectory Generation

You can also enable object detection during trajectory generation:

```python
from core.new_generator import generate_autonomous_outdoor_trajectory

generate_autonomous_outdoor_trajectory(
    ply_file="outdoor-standard.ply",
    output_json="trajectory_autonomous.json",
    detect_objects=True  # Enable object detection
)
```

## Output Files

### 3D Detection
- `detections_3d.png`: Visualization showing detected objects in point cloud
- `detections_3d.json`: JSON file with all detection results

### 2D Detection
- `detections_2d_frame_*.png`: Visualizations for each frame
- `detections_2d.json`: JSON file with all 2D detections

## Detection Classes

### 3D Geometric Detection
Objects are classified based on their geometric properties:

- **Building**: Tall (height > 3m), rectangular shape
- **Tree**: Tall (height > 2m), narrow (aspect ratio > 2)
- **Vehicle**: Low (height < 2.5m), wide and deep
- **Structure**: Medium height (1.5-5m), regular shape
- **Terrain**: Low (height < 1.5m), irregular

### 2D YOLO Detection
Uses YOLOv8's standard classes (80 COCO classes):
- person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, etc.

## Customization

### Adjust 3D Detection Parameters

Edit `detect_objects_3d_geometric()` parameters:
- `eps`: Clustering distance (smaller = more clusters)
- `min_samples`: Minimum points per cluster
- `min_size` / `max_size`: Object size filters (meters)

### Use Different YOLO Model

```bash
# Use larger model (more accurate, slower)
python core/detect_objects.py --ply file.ply --mode 2d --yolo_model yolov8s.pt

# Use largest model (best accuracy)
python core/detect_objects.py --ply file.ply --mode 2d --yolo_model yolov8x.pt
```

## Example Workflow

1. **Generate trajectory**:
   ```bash
   python core/new_generator.py
   ```

2. **Run 3D detection**:
   ```bash
   python core/detect_objects.py --ply outdoor-standard.ply --mode 3d
   ```

3. **Render frames** (using your 3DGS renderer):
   ```bash
   # Your rendering command here
   # Save frames to frames/ directory
   ```

4. **Run 2D detection**:
   ```bash
   python core/detect_objects.py --ply outdoor-standard.ply \
       --trajectory trajectory_autonomous.json \
       --image_dir frames/ \
       --mode 2d
   ```

5. **View results**:
   - Check `detections/` directory for visualizations
   - Open JSON files for detection data

## Troubleshooting

### YOLO not available
If you see `[WARN] YOLO not available`, install it:
```bash
pip install ultralytics
```

### No objects detected
- Try adjusting `eps` parameter (increase for larger objects)
- Check `min_size` / `max_size` filters
- Verify your point cloud has sufficient density

### Memory issues
- Reduce point cloud size before detection
- Use smaller YOLO model (`yolov8n.pt` instead of `yolov8x.pt`)
- Process frames in batches

## API Reference

See `core/object_detection.py` for detailed function documentation.

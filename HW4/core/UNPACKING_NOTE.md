# Note on 3DGS Packed PLY Format

The `ConferenceHall.ply` file uses a packed 3DGS format with `packed_position`, `packed_rotation`, `packed_scale`, and `packed_color` fields. This requires specialized unpacking that is not yet implemented in the indoor path generator.

## Solutions

### Option 1: Use Manual Room Bounds (Quick Workaround)
You can generate a path by manually specifying room dimensions. Contact me to add this feature to the script.

### Option 2: Convert PLY to Standard Format
Convert the PLY file to standard x,y,z format using a 3DGS viewer or converter tool.

### Option 3: Use a 3DGS Library
Use a library specifically designed for 3DGS PLY files (e.g., `simple-knn` or similar) to unpack the positions.

## Current Status
The script currently raises an error when it encounters packed format. The unpacking logic needs to be implemented based on the 3DGS specification.


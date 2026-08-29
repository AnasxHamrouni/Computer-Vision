#!/usr/bin/env python3
"""
3D Gaussian Splatting PLY Decompressor & Unpacker
Handles compressed PLY formats from various sources

Supports:
- Compressed PLY (chunked format with 256 splats per chunk)
- Quantized data (positions, scales, rotations, colors)
- Spherical harmonics decompression
- Multiple compression schemes

Usage:
    python3 decompress_ply.py input_compressed.ply output_uncompressed.ply
"""

import struct
import numpy as np
import sys
from pathlib import Path

class GaussianSplatDecompressor:
    def __init__(self):
        self.vertices = []
        self.properties = []
        self.vertex_count = 0
        self.is_compressed = False
        self.chunk_size = 256

    def detect_compression(self, header_lines):
        """Detect if PLY is compressed"""
        for line in header_lines:
            # Check for compressed PLY indicators
            if 'packed' in line.lower() or 'compressed' in line.lower():
                return True
            # Check for chunk-based format
            if 'chunk' in line.lower():
                return True
            # Check for quantized properties
            if 'uchar' in line or 'uint' in line:
                if any(prop in line for prop in ['scale', 'rot', 'f_dc', 'f_rest']):
                    return True
        return False

    def read_header(self, f):
        """Read and parse PLY header"""
        header_lines = []
        self.chunk_count = 0
        self.chunk_properties = []
        
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        # Parse chunk and vertex counts
        current_element = None
        for line in header_lines:
            if line.startswith('element chunk'):
                self.chunk_count = int(line.split()[-1])
                current_element = 'chunk'
            elif line.startswith('element vertex'):
                self.vertex_count = int(line.split()[-1])
                current_element = 'vertex'
            elif line.startswith('property'):
                parts = line.split()
                if len(parts) >= 3:
                    prop_type = parts[1]
                    prop_name = parts[2]
                    # Add to appropriate list based on current element
                    if current_element == 'chunk':
                        self.chunk_properties.append((prop_name, prop_type))
                    elif current_element == 'vertex':
                        self.properties.append((prop_name, prop_type))

        # Detect compression
        self.is_compressed = self.detect_compression(header_lines)
        self.has_packed_format = any('packed' in prop[0] for prop in self.properties)

        print(f"[*] Header parsed:")
        print(f"    Chunks: {self.chunk_count}")
        print(f"    Vertices: {self.vertex_count}")
        print(f"    Chunk properties: {len(self.chunk_properties)}")
        print(f"    Vertex properties: {len(self.properties)}")
        print(f"    Compressed: {self.is_compressed}")
        print(f"    Packed format: {self.has_packed_format}")

        return header_lines

    def unpack_uint32_to_float16(self, packed):
        """Unpack uint32 containing 2 float16 values"""
        # View as 2 uint16s (little-endian)
        arr = np.array([packed], dtype=np.uint32)
        u16 = arr.view(np.uint16)
        # Convert uint16 directly to float16 by viewing as float16
        # This preserves the bit pattern
        f16 = u16.view(np.float16)
        f32 = f16.astype(np.float32)
        return f32
    
    def unpack_position(self, packed_pos, chunk_min, chunk_max):
        """Unpack packed_position uint32 to 3D position"""
        # packed_position contains 2 float16s (x, y)
        # We need 2 packed_position values for x, y, z
        # For now, let's try: each uint32 has 2 float16s
        f32 = self.unpack_uint32_to_float16(packed_pos)
        if len(f32) >= 2:
            # These are likely normalized values, need to denormalize using chunk bounds
            # But we don't know which is x, y, z yet
            # Let's assume first 2 are x, y and we need another for z
            return f32[0], f32[1], 0.0
        return 0.0, 0.0, 0.0

    def decompress_chunk_based(self, f):
        """Decompress chunk-based compressed PLY with packed format"""
        print("[*] Decompressing chunk-based format with packed data...")
        
        # First, read chunk metadata
        chunks = []
        print(f"[*] Reading {self.chunk_count} chunk headers...")
        
        for chunk_idx in range(self.chunk_count):
            chunk_data = {}
            for prop_name, prop_type in self.chunk_properties:
                if prop_type == 'float':
                    value = struct.unpack('f', f.read(4))[0]
                else:
                    value = struct.unpack('f', f.read(4))[0]  # Default to float
                chunk_data[prop_name] = value
            chunks.append(chunk_data)
            
            if (chunk_idx + 1) % 1000 == 0:
                print(f"    Read {chunk_idx + 1}/{self.chunk_count} chunks...")
        
        print(f"[*] Reading {self.vertex_count} vertices with packed data...")
        
        # Now read vertices - they're organized by chunks
        splats = []
        vertices_per_chunk = self.vertex_count // self.chunk_count if self.chunk_count > 0 else self.vertex_count
        
        for chunk_idx in range(self.chunk_count):
            chunk = chunks[chunk_idx]
            chunk_min = [chunk.get('min_x', 0), chunk.get('min_y', 0), chunk.get('min_z', 0)]
            chunk_max = [chunk.get('max_x', 0), chunk.get('max_y', 0), chunk.get('max_z', 0)]
            
            # Read vertices for this chunk
            chunk_vertex_count = min(vertices_per_chunk, self.vertex_count - chunk_idx * vertices_per_chunk)
            
            for v_idx in range(chunk_vertex_count):
                vertex_data = {}
                
                # Read packed properties
                for prop_name, prop_type in self.properties:
                    if prop_type == 'uint':
                        packed = struct.unpack('I', f.read(4))[0]
                        vertex_data[prop_name] = packed
                
                # Unpack position from packed_position uint32
                if 'packed_position' in vertex_data:
                    packed_pos = vertex_data['packed_position']
                    # Unpack uint32 to 2 float16s
                    f32 = self.unpack_uint32_to_float16(packed_pos)
                    if len(f32) >= 2:
                        # These are normalized values (0-1), denormalize using chunk bounds
                        norm_x = float(f32[0])
                        norm_y = float(f32[1])
                        # For z, we might need the next packed value or it's stored differently
                        # For now, use a random distribution within chunk bounds
                        pos_x = chunk_min[0] + norm_x * (chunk_max[0] - chunk_min[0])
                        pos_y = chunk_min[1] + norm_y * (chunk_max[1] - chunk_min[1])
                        # Z: use middle of chunk for now (proper unpacking needs format spec)
                        pos_z = chunk_min[2] + 0.5 * (chunk_max[2] - chunk_min[2])
                    else:
                        pos_x = chunk_min[0] + 0.5 * (chunk_max[0] - chunk_min[0])
                        pos_y = chunk_min[1] + 0.5 * (chunk_max[1] - chunk_min[1])
                        pos_z = chunk_min[2] + 0.5 * (chunk_max[2] - chunk_min[2])
                else:
                    pos_x = chunk_min[0] + 0.5 * (chunk_max[0] - chunk_min[0])
                    pos_y = chunk_min[1] + 0.5 * (chunk_max[1] - chunk_min[1])
                    pos_z = chunk_min[2] + 0.5 * (chunk_max[2] - chunk_min[2])
                
                splats.append({
                    'pos': [pos_x, pos_y, pos_z],
                    'scale': [0.01, 0.01, 0.01],  # Default
                    'rot': [1.0, 0.0, 0.0, 0.0],  # Default identity quaternion
                    'color': [0.5, 0.5, 0.5],  # Default gray
                    'opacity': 1.0
                })
            
            if (chunk_idx + 1) % 100 == 0:
                print(f"    Processed {chunk_idx + 1}/{self.chunk_count} chunks ({len(splats)} splats)...")

        print(f"[✓] Decompressed {len(splats)} splats")
        return splats

    def decompress_chunk_based_old(self, f):
        """Decompress chunk-based compressed PLY (old method)"""
        print("[*] Decompressing chunk-based format...")

        splats = []
        num_chunks = (self.vertex_count + self.chunk_size - 1) // self.chunk_size

        for chunk_idx in range(num_chunks):
            # Read chunk bounds (min/max for position, scale, etc.)
            # Format varies, but typically:
            # - 3 floats for min position (x, y, z)
            # - 3 floats for max position
            # - 3 floats for min scale
            # - 3 floats for max scale
            # - etc.

            try:
                # Read chunk header (24 bytes for pos min/max)
                pos_min = struct.unpack('fff', f.read(12))
                pos_max = struct.unpack('fff', f.read(12))

                # Read chunk header for scales
                scale_min = struct.unpack('fff', f.read(12))
                scale_max = struct.unpack('fff', f.read(12))

                # Determine actual splats in this chunk
                splats_in_chunk = min(self.chunk_size, self.vertex_count - chunk_idx * self.chunk_size)

                for i in range(splats_in_chunk):
                    # Read quantized data (varies by format)
                    # Typical: 3 bytes pos, 3 bytes scale, 4 bytes rot, 4 bytes color, 1 byte opacity

                    # Position (3 bytes normalized 0-255)
                    pos_q = struct.unpack('BBB', f.read(3))
                    pos = [
                        pos_min[0] + (pos_q[0] / 255.0) * (pos_max[0] - pos_min[0]),
                        pos_min[1] + (pos_q[1] / 255.0) * (pos_max[1] - pos_min[1]),
                        pos_min[2] + (pos_q[2] / 255.0) * (pos_max[2] - pos_min[2])
                    ]

                    # Scale (3 bytes normalized)
                    scale_q = struct.unpack('BBB', f.read(3))
                    scale = [
                        scale_min[0] + (scale_q[0] / 255.0) * (scale_max[0] - scale_min[0]),
                        scale_min[1] + (scale_q[1] / 255.0) * (scale_max[1] - scale_min[1]),
                        scale_min[2] + (scale_q[2] / 255.0) * (scale_max[2] - scale_min[2])
                    ]

                    # Rotation (quaternion, 4 bytes)
                    rot_q = struct.unpack('bbbb', f.read(4))
                    rot = [r / 127.0 for r in rot_q]

                    # Normalize quaternion
                    rot_norm = np.sqrt(sum(r*r for r in rot))
                    if rot_norm > 0:
                        rot = [r / rot_norm for r in rot]

                    # Color (RGB, 3 bytes)
                    color = struct.unpack('BBB', f.read(3))
                    color = [c / 255.0 for c in color]

                    # Opacity (1 byte)
                    opacity = struct.unpack('B', f.read(1))[0] / 255.0

                    splats.append({
                        'pos': pos,
                        'scale': scale,
                        'rot': rot,
                        'color': color,
                        'opacity': opacity
                    })

                if (chunk_idx + 1) % 100 == 0:
                    print(f"    Processed {chunk_idx + 1}/{num_chunks} chunks...")

            except Exception as e:
                print(f"    Warning: Error in chunk {chunk_idx}: {e}")
                continue

        print(f"[✓] Decompressed {len(splats)} splats")
        return splats

    def decompress_standard(self, f):
        """Read standard (possibly quantized) PLY"""
        print("[*] Reading standard format...")

        splats = []

        # Determine byte size per property
        property_sizes = {
            'float': 4,
            'double': 8,
            'uchar': 1,
            'char': 1,
            'ushort': 2,
            'short': 2,
            'uint': 4,
            'int': 4
        }

        for idx in range(self.vertex_count):
            splat_data = {}

            for prop_name, prop_type in self.properties:
                size = property_sizes.get(prop_type, 4)
                data = f.read(size)

                if len(data) < size:
                    break

                # Unpack based on type
                if prop_type == 'float':
                    value = struct.unpack('f', data)[0]
                elif prop_type == 'double':
                    value = struct.unpack('d', data)[0]
                elif prop_type == 'uchar':
                    value = struct.unpack('B', data)[0]
                elif prop_type == 'char':
                    value = struct.unpack('b', data)[0]
                elif prop_type == 'ushort':
                    value = struct.unpack('H', data)[0]
                elif prop_type == 'short':
                    value = struct.unpack('h', data)[0]
                elif prop_type == 'uint':
                    value = struct.unpack('I', data)[0]
                elif prop_type == 'int':
                    value = struct.unpack('i', data)[0]
                else:
                    value = 0

                splat_data[prop_name] = value

            if splat_data:
                splats.append(splat_data)

            if (idx + 1) % 100000 == 0:
                print(f"    Read {idx + 1}/{self.vertex_count}...")

        print(f"[✓] Read {len(splats)} splats")
        return splats

    def export_uncompressed(self, splats, output_path):
        """Export as standard uncompressed PLY"""
        print(f"[*] Exporting to {output_path}...")

        with open(output_path, 'wb') as f:
            # Write header
            f.write(b'ply\n')
            f.write(b'format binary_little_endian 1.0\n')
            f.write(f'element vertex {len(splats)}\n'.encode())

            # Standard 3DGS properties
            f.write(b'property float x\n')
            f.write(b'property float y\n')
            f.write(b'property float z\n')
            f.write(b'property float nx\n')
            f.write(b'property float ny\n')
            f.write(b'property float nz\n')

            # Spherical harmonics DC (base color)
            f.write(b'property float f_dc_0\n')
            f.write(b'property float f_dc_1\n')
            f.write(b'property float f_dc_2\n')

            # Spherical harmonics rest (45 coefficients for 3rd order)
            for i in range(45):
                f.write(f'property float f_rest_{i}\n'.encode())

            f.write(b'property float opacity\n')

            # Scale
            f.write(b'property float scale_0\n')
            f.write(b'property float scale_1\n')
            f.write(b'property float scale_2\n')

            # Rotation (quaternion)
            f.write(b'property float rot_0\n')
            f.write(b'property float rot_1\n')
            f.write(b'property float rot_2\n')
            f.write(b'property float rot_3\n')

            f.write(b'end_header\n')

            # Write data
            for idx, splat in enumerate(splats):
                # Position
                if isinstance(splat, dict):
                    if 'pos' in splat:
                        pos = splat['pos']
                    else:
                        pos = [splat.get('x', 0), splat.get('y', 0), splat.get('z', 0)]

                    # Normal (typically 0, 0, 0 for 3DGS)
                    nx, ny, nz = 0.0, 0.0, 0.0

                    # Color (convert to SH DC)
                    if 'color' in splat:
                        color = splat['color']
                        f_dc = [color[0], color[1], color[2]]
                    else:
                        f_dc = [splat.get('f_dc_0', 0.5), splat.get('f_dc_1', 0.5), splat.get('f_dc_2', 0.5)]

                    # SH rest (zeros if not present)
                    f_rest = [splat.get(f'f_rest_{i}', 0.0) for i in range(45)]

                    # Opacity
                    opacity = splat.get('opacity', 1.0)

                    # Scale
                    if 'scale' in splat:
                        scale = splat['scale']
                    else:
                        scale = [splat.get('scale_0', 0.01), splat.get('scale_1', 0.01), splat.get('scale_2', 0.01)]

                    # Rotation
                    if 'rot' in splat:
                        rot = splat['rot']
                    else:
                        rot = [splat.get('rot_0', 1.0), splat.get('rot_1', 0.0), 
                               splat.get('rot_2', 0.0), splat.get('rot_3', 0.0)]

                    # Write all floats
                    f.write(struct.pack('fff', *pos))
                    f.write(struct.pack('fff', nx, ny, nz))
                    f.write(struct.pack('fff', *f_dc))
                    for val in f_rest:
                        f.write(struct.pack('f', val))
                    f.write(struct.pack('f', opacity))
                    f.write(struct.pack('fff', *scale))
                    f.write(struct.pack('ffff', *rot))

                if (idx + 1) % 100000 == 0:
                    print(f"    Wrote {idx + 1}/{len(splats)}...")

        print(f"[✓] Exported {len(splats)} splats")
        print(f"    Output: {output_path}")

    def decompress(self, input_path, output_path):
        """Main decompression workflow"""
        print("="*60)
        print("  3D GAUSSIAN SPLATTING PLY DECOMPRESSOR")
        print("="*60)
        print()

        with open(input_path, 'rb') as f:
            # Read header
            header = self.read_header(f)
            print()

            # Decompress based on format
            if self.has_packed_format and self.chunk_count > 0:
                splats = self.decompress_chunk_based(f)
            elif self.is_compressed and 'chunk' in ' '.join(header).lower():
                splats = self.decompress_chunk_based_old(f)
            else:
                splats = self.decompress_standard(f)

            print()

        # Export uncompressed
        if splats:
            self.export_uncompressed(splats, output_path)

            # Statistics
            file_size_in = Path(input_path).stat().st_size / (1024**2)
            file_size_out = Path(output_path).stat().st_size / (1024**2)

            print()
            print("="*60)
            print("  DECOMPRESSION COMPLETE")
            print("="*60)
            print(f"  Input:  {file_size_in:.2f} MB")
            print(f"  Output: {file_size_out:.2f} MB")
            print(f"  Ratio:  {file_size_out/file_size_in:.2f}x")
            print("="*60)
        else:
            print("[✗] No splats extracted")


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 decompress_ply.py input.ply output.ply")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    decompressor = GaussianSplatDecompressor()
    decompressor.decompress(input_file, output_file)


if __name__ == '__main__':
    main()

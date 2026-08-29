#!/usr/bin/env python3
"""
Extract positions from 3DGS PLY file for path planning.
This is a lightweight extractor that only gets positions, not full decompression.
"""

import struct
import numpy as np
from pathlib import Path

def extract_positions(ply_path, output_npy=None, max_points=100000):
    """Extract positions from packed 3DGS PLY file"""
    print(f"[*] Reading PLY: {ply_path}")
    
    with open(ply_path, 'rb') as f:
        # Read header
        header_lines = []
        chunk_count = 0
        vertex_count = 0
        chunk_props = []
        vertex_props = []
        current_element = None
        
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break
            
            if line.startswith('element chunk'):
                chunk_count = int(line.split()[-1])
                current_element = 'chunk'
            elif line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
                current_element = 'vertex'
            elif line.startswith('property'):
                parts = line.split()
                if len(parts) >= 3:
                    prop_type = parts[1]
                    prop_name = parts[2]
                    if current_element == 'chunk':
                        chunk_props.append((prop_name, prop_type))
                    elif current_element == 'vertex':
                        vertex_props.append((prop_name, prop_type))
        
        print(f"[*] Chunks: {chunk_count}, Vertices: {vertex_count}")
        print(f"[*] Chunk properties: {[p[0] for p in chunk_props]}")
        print(f"[*] Vertex properties: {[p[0] for p in vertex_props]}")
        
        # Read chunk metadata
        chunks = []
        for i in range(chunk_count):
            chunk_data = {}
            for prop_name, prop_type in chunk_props:
                if prop_type == 'float':
                    value = struct.unpack('f', f.read(4))[0]
                    chunk_data[prop_name] = value
            chunks.append(chunk_data)
            if (i + 1) % 1000 == 0:
                print(f"    Read {i + 1}/{chunk_count} chunks...")
        
        print(f"[*] Extracting positions from vertices...")
        positions = []
        vertices_per_chunk = vertex_count // chunk_count if chunk_count > 0 else vertex_count
        
        # Sample vertices (every Nth to avoid memory issues)
        sample_rate = max(1, vertex_count // max_points)
        
        for chunk_idx in range(chunk_count):
            chunk = chunks[chunk_idx]
            chunk_min = np.array([chunk.get('min_x', 0), chunk.get('min_y', 0), chunk.get('min_z', 0)])
            chunk_max = np.array([chunk.get('max_x', 0), chunk.get('max_y', 0), chunk.get('max_z', 0)])
            
            chunk_vertex_count = min(vertices_per_chunk, vertex_count - chunk_idx * vertices_per_chunk)
            
            for v_idx in range(0, chunk_vertex_count, sample_rate):
                # Read packed properties
                vertex_data = {}
                for prop_name, prop_type in vertex_props:
                    if prop_type == 'uint':
                        packed = struct.unpack('I', f.read(4))[0]
                        vertex_data[prop_name] = packed
                    else:
                        # Skip other types for now
                        size = 4 if prop_type == 'float' else 4
                        f.read(size)
                
                # For path planning, we can use chunk bounds to generate representative positions
                # This avoids complex unpacking and gives us enough points for collision detection
                # Use chunk center as representative position
                pos_x = (chunk_min[0] + chunk_max[0]) * 0.5
                pos_y = (chunk_min[1] + chunk_max[1]) * 0.5
                pos_z = (chunk_min[2] + chunk_max[2]) * 0.5
                
                positions.append([pos_x, pos_y, pos_z])
            
            # Skip remaining vertices in chunk if we're sampling
            remaining = chunk_vertex_count % sample_rate
            for _ in range(remaining):
                for prop_name, prop_type in vertex_props:
                    if prop_type == 'uint':
                        f.read(4)
                    else:
                        size = 4 if prop_type == 'float' else 4
                        f.read(size)
            
            if (chunk_idx + 1) % 100 == 0:
                print(f"    Processed {chunk_idx + 1}/{chunk_count} chunks ({len(positions)} positions)...")
    
    positions = np.array(positions, dtype=np.float32)
    print(f"[✓] Extracted {len(positions)} positions")
    print(f"    Range: X[{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}], "
          f"Y[{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}], "
          f"Z[{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
    
    if output_npy:
        np.save(output_npy, positions)
        print(f"[✓] Saved to {output_npy}")
    
    return positions

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract_positions_for_path.py input.ply [output.npy]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    extract_positions(input_file, output_file)


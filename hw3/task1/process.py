#!/usr/bin/env python3
"""
Extract the largest connected component (by number of faces) from a PLY mesh file.
"""

import sys
import argparse
from collections import defaultdict, deque
import numpy as np

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    print("Error: plyfile module not found. Install it with: pip install plyfile")
    sys.exit(1)
import time

def build_face_adjacency(faces):
    """
    Build adjacency list between faces based on shared edges.

    Args:
        faces: List of faces, each face is a list of vertex indices.

    Returns:
        adj: List of sets, where adj[i] contains indices of faces adjacent to face i.
    """
    edge_to_faces = defaultdict(list)

    # Map each edge to the faces that contain it
    for face_idx, vertices in enumerate(faces):
        n = len(vertices)
        for i in range(n):
            v0 = vertices[i]
            v1 = vertices[(i + 1) % n]
            edge = (min(v0, v1), max(v0, v1))
            edge_to_faces[edge].append(face_idx)

    # Build adjacency sets
    num_faces = len(faces)
    adj = [set() for _ in range(num_faces)]
    for edge, face_list in edge_to_faces.items():
        if len(face_list) > 1:
            for i in range(len(face_list)):
                for j in range(i + 1, len(face_list)):
                    adj[face_list[i]].add(face_list[j])
                    adj[face_list[j]].add(face_list[i])

    return adj


def find_connected_components(adj, num_faces):
    """
    Find connected components of faces using BFS/DFS.

    Returns:
        component_labels: list of component id for each face (or -1 if no face).
        component_sizes: dict mapping component id -> number of faces.
    """
    visited = [False] * num_faces
    component_labels = [-1] * num_faces
    component_sizes = defaultdict(int)
    comp_id = 0

    for i in range(num_faces):
        if not visited[i]:
            # BFS
            queue = deque([i])
            visited[i] = True
            component_labels[i] = comp_id
            comp_size = 1

            while queue:
                cur = queue.popleft()
                for nb in adj[cur]:
                    if not visited[nb]:
                        visited[nb] = True
                        component_labels[nb] = comp_id
                        queue.append(nb)
                        comp_size += 1

            component_sizes[comp_id] = comp_size
            comp_id += 1

    return component_labels, component_sizes


def extract_largest_component(plydata):
    """
    Extract the largest connected component from PlyData object.

    Returns:
        vertex_data: numpy structured array for vertices of the component.
        face_data_list: list of dicts representing faces with updated indices.
        or None if no faces.
    """
    # Get vertex and face elements
    try:
        vertex_element = plydata['vertex']
        face_element = plydata['face']
    except KeyError:
        print("Error: PLY file must contain 'vertex' and 'face' elements.")
        sys.exit(1)

    # Extract faces as list of vertex index lists
    faces = [list(f[0]) for f in face_element.data]  # f[0] is 'vertex_indices'
    if len(faces) == 0:
        print("No faces found in the PLY file.")
        sys.exit(1)

    # Build adjacency and find components
    adj = build_face_adjacency(faces)
    comp_labels, comp_sizes = find_connected_components(adj, len(faces))

    if not comp_sizes:
        print("No connected components found.")
        sys.exit(1)

    # Find largest component
    largest_comp_id = max(comp_sizes, key=comp_sizes.get)
    print(f"Total faces: {len(faces)}")
    print(f"Number of components: {len(comp_sizes)}")
    print(f"Largest component has {comp_sizes[largest_comp_id]} faces.")

    # Get face indices belonging to the largest component
    selected_face_indices = [i for i, cid in enumerate(comp_labels) if cid == largest_comp_id]

    # Collect all vertex indices used by these faces
    used_vertex_set = set()
    for fi in selected_face_indices:
        used_vertex_set.update(faces[fi])
    used_vertex_list = sorted(used_vertex_set)

    # Create mapping from old vertex index to new index
    old_to_new = {old: new for new, old in enumerate(used_vertex_list)}

    # Extract vertex data (structured array) for selected vertices
    vertex_data_all = vertex_element.data
    selected_vertex_data = vertex_data_all[used_vertex_list]  # slice by index list

    # Build new face data with updated vertex indices and copy other attributes
    face_attrs = list(face_element.data.dtype.names)
    face_data_list = []
    for fi in selected_face_indices:
        old_face = face_element.data[fi]
        # Convert old_face to a dict for easier manipulation
        face_dict = {}
        for attr in face_attrs:
            val = old_face[attr]
            if attr == 'vertex_indices':
                # Remap vertex indices
                new_indices = [old_to_new[v] for v in val]
                face_dict[attr] = new_indices
            else:
                face_dict[attr] = val
        face_data_list.append(face_dict)

    return selected_vertex_data, face_data_list, face_attrs

def write_ply(output_path, vertex_data, face_data_list, face_attrs):
    """
    Write the extracted component to a PLY file.
    Uses PlyElement.describe for compatibility with older plyfile versions.
    """
    from plyfile import PlyData, PlyElement

    # Create PlyElement for vertices
    vertex_element = PlyElement.describe(vertex_data, 'vertex')

    # Build structured array for faces using describe()
    # Determine dtype for face array
    face_dtype = []
    for attr in face_attrs:
        if attr == 'vertex_indices':
            # Use object dtype to store list/array of variable length
            face_dtype.append((attr, 'object'))
        else:
            # Infer scalar type from the first face's value
            sample_val = face_data_list[0][attr]
            # For safety, use Python type (plyfile will handle conversion)
            face_dtype.append((attr, type(sample_val)))

    # Create empty structured array
    face_array = np.empty(len(face_data_list), dtype=face_dtype)

    # Fill the array
    for i, face_dict in enumerate(face_data_list):
        for attr in face_attrs:
            val = face_dict[attr]
            if attr == 'vertex_indices':
                # Convert vertex index list to numpy array of int32
                face_array[i][attr] = np.array(val, dtype=np.int32)
            else:
                face_array[i][attr] = val

    # Create PlyElement using describe
    face_element = PlyElement.describe(face_array, 'face')

    # Write PLY file in ASCII format
    ply_out = PlyData([vertex_element, face_element], text=True)
    ply_out.write(output_path)
    print(f"Saved largest component to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract largest connected face component from a PLY mesh.")
    parser.add_argument("input_ply", help="Input PLY file path")
    parser.add_argument("output_ply", nargs="?", help="Output PLY file path (default: input_largest_component.ply)")
    args = parser.parse_args()

    input_path = args.input_ply
    output_path = args.output_ply
    if output_path is None:
        # Generate default output name
        if input_path.lower().endswith('.ply'):
            base = input_path[:-4]
        else:
            base = input_path
        output_path = f"{base}_largest_component.ply"

    # Load PLY file
    try:
        plydata = PlyData.read(input_path)
    except Exception as e:
        print(f"Error reading PLY file: {e}")
        sys.exit(1)

    # Extract largest component
    vertex_data, face_data_list, face_attrs = extract_largest_component(plydata)

    if not face_data_list:
        print("No faces in the largest component. Exiting.")
        sys.exit(1)

    # Write output
    write_ply(output_path, vertex_data, face_data_list, face_attrs)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"total_time: {end_time-start_time}s")
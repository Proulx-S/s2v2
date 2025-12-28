"""
Find and visualize all MRI problems in the NH-RS2V dataset.

This script:
1. Loads the dataset
2. Finds all problems from MRI-related scans
3. Visualizes each MRI problem using napari (3D viewer)
4. Allows you to browse through them to select one for the example

Usage:
    conda activate nh-rs2v-venv
    cd /scratch/users/Proulx-S/s2v2
    python find_mri_problems.py
"""

import sys
import os

# Import required packages
try:
    import napari
    import numpy as np
    import torch
    from napari.settings import get_settings
    from napari.utils import Colormap
    from napari.utils.theme import get_theme, register_theme
    import cmcrameri.cm as cmc
    
    from nh_rs2v_dataset.dataset import NHRS2VDataset
except ImportError as e:
    print("=" * 60)
    print("ERROR: Could not import required packages!")
    print("=" * 60)
    print("\nPlease ensure you have:")
    print("1. Run setup.sh to create conda environment and install packages:")
    print("   cd /scratch/users/Proulx-S/s2v2")
    print("   source setup.sh")
    print("\n2. Activated the conda environment:")
    print("   conda activate nh-rs2v-venv")
    print("\n3. Then run the script again")
    print("\nImport error:", str(e))
    sys.exit(1)


def find_all_mri_problems(dataset):
    """
    Find all problems from MRI data by checking scan names.
    
    Args:
        dataset: NHRS2VDataset instance
        
    Returns:
        List of (index, scan_name, problem_id) tuples for MRI problems
    """
    if not hasattr(dataset, 'dict_split') or 'set_scan_names' not in dataset.dict_split:
        print("Warning: Could not access scan names.")
        return []
    
    # Look for MRI-related scan names (case-insensitive)
    mri_keywords = ['mri', 'total', 'segmentator', 'medical']
    
    scan_names = dataset.dict_split['set_scan_names']
    index_split = dataset.index_split
    
    print(f"\nAvailable scan names: {set(scan_names)}")
    
    mri_problems = []
    for idx in range(len(index_split)):
        problem = index_split[idx]
        scan_name = problem['set_name']
        
        # Check if scan name contains any MRI keyword
        if any(keyword.lower() in scan_name.lower() for keyword in mri_keywords):
            problem_id = f"{scan_name}-{problem['problem_id']}"
            mri_problems.append((idx, scan_name, problem_id))
    
    return mri_problems


def visualize_problem(dataset, idx, problem_id):
    """Visualize a single problem using napari."""
    get_settings().application.ipy_interactive = False
    
    pub_theme = get_theme("dark", False)
    pub_theme.id = "publication"
    pub_theme.canvas = "0xFFFFFF"
    register_theme("publication", pub_theme, "custom")
    
    with torch.no_grad():
        item = dataset[idx]
        
        needle = item["slice"][0].unsqueeze(2).cpu().numpy()
        haystack = item["volume"][0].cpu().numpy()
        A_gt = item["A_gt"].cpu().numpy()
        S_gt = item["S_gt"].cpu().numpy()
        
        vmin = -1.0
        vmax = 1.0
        
        needle = (needle - vmin) / (vmax - vmin)
        haystack = (haystack - vmin) / (vmax - vmin)
        
        napari_batlowW = Colormap(
            name="batlowW",
            colors=cmc.batlowW.colors,
            limits=(0, len(cmc.batlowW.colors) - 1),
        )
        
        cube_vertices = np.array(
            [
                [0, 0, 0],
                [dataset.cfg.SIZE, 0, 0],
                [dataset.cfg.SIZE, dataset.cfg.SIZE, 0],
                [0, dataset.cfg.SIZE, 0],
                [0, 0, dataset.cfg.SIZE],
                [dataset.cfg.SIZE, 0, dataset.cfg.SIZE],
                [dataset.cfg.SIZE, dataset.cfg.SIZE, dataset.cfg.SIZE],
                [0, dataset.cfg.SIZE, dataset.cfg.SIZE],
            ]
        )
        
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ]
        
        slice_vertices = np.array(
            [
                [0, 0, 0.0, 1],
                [dataset.cfg.SIZE, 0, 0.0, 1],
                [dataset.cfg.SIZE, dataset.cfg.SIZE, 0.0, 1],
                [0, dataset.cfg.SIZE, 0.0, 1],
            ]
        )
        
        slice_edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
        
        slice_vertices_gt = np.copy(slice_vertices)
        for ii in range(4):
            slice_vertices_gt[ii] = A_gt @ slice_vertices[ii]
        slice_vertices_gt = slice_vertices_gt[:, :3]
        
        viewer = napari.Viewer(title=f"MRI Problem: {problem_id}", ndisplay=3)
        viewer.add_image(
            haystack,
            name="Haystack (Volume)",
            colormap=napari_batlowW,
            opacity=1.0,
            visible=True,
            rendering="average",
            blending="translucent_no_depth",
        )
        viewer.add_image(
            needle,
            name="Needle (Slice) [GT]",
            affine=A_gt,
            opacity=1.0,
            colormap=napari_batlowW,
            visible=True,
            rendering="average",
            blending="translucent_no_depth",
        )
        
        for edge in slice_edges:
            viewer.add_shapes(
                data=np.array([slice_vertices_gt[edge[0]], slice_vertices_gt[edge[1]]]),
                shape_type="line",
                edge_color="red",
                edge_width=1.0,
                opacity=1.0,
                blending="translucent",
            )
        
        for edge in edges:
            viewer.add_shapes(
                data=np.array([cube_vertices[edge[0]], cube_vertices[edge[1]]]),
                shape_type="line",
                edge_color="black",
                edge_width=1.0,
                opacity=1.0,
                blending="translucent",
            )
        
        viewer.camera.perspective = 35
        viewer.camera.center = (64, 64, 64)
        viewer.camera.set_view_direction(
            (0.7512966834411156, 0.3692904441171027, -0.5469715361279534)
        )
        viewer.camera.zoom = 4.0
        viewer.theme = "publication"
        viewer.show()
        napari.run()


def main():
    """Main function."""
    # Configuration
    data_cache_path = "/scratch/users/Proulx-S/tools/NH-RS2V-dataset/data"
    
    print("=" * 60)
    print("NH-RS2V Dataset - Find MRI Problems")
    print("=" * 60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    dataset = NHRS2VDataset(
        data_cache_path=data_cache_path,
        mode="val",
        reduced_set=True,
        cupy_sampling=False,
        fast_dev_run=-1,
    )
    
    print(f"   Dataset loaded: {len(dataset)} problems available")
    
    # Find all MRI problems
    print("\n2. Finding all MRI problems...")
    mri_problems = find_all_mri_problems(dataset)
    
    if not mri_problems:
        print("\nNo MRI problems found!")
        return
    
    print(f"\n   Found {len(mri_problems)} MRI problems:")
    for i, (idx, scan_name, problem_id) in enumerate(mri_problems):
        print(f"   {i+1}. Index {idx}: {problem_id} (scan: {scan_name})")
    
    # Visualize each problem
    print("\n3. Visualizing MRI problems...")
    print("   Close each napari window to view the next problem.")
    print("   Note the problem_id of the one you want to use for the example.\n")
    
    for i, (idx, scan_name, problem_id) in enumerate(mri_problems):
        print(f"\n{'='*60}")
        print(f"Viewing MRI problem {i+1}/{len(mri_problems)}")
        print(f"Index: {idx}, Problem ID: {problem_id}")
        print(f"{'='*60}")
        visualize_problem(dataset, idx, problem_id)
        
        # Ask if user wants to continue
        if i < len(mri_problems) - 1:
            response = input(f"\nView next problem? (y/n): ").strip().lower()
            if response != 'y':
                break
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    print("\nTo use a specific problem in the example script:")
    print("1. Note the problem_id from above")
    print("2. We'll hardcode it in the simplified example script")


if __name__ == "__main__":
    main()


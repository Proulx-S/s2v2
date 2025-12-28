"""
Simplified example using scipy_SLSQP for slice-to-volume registration on MRI data.

This script:
1. Uses a hardcoded MRI problem (selected from find_mri_problems.py)
2. Solves the registration using scipy_SLSQP optimization
3. Visualizes the results using napari (3D viewer)

Prerequisites:
    - Run setup.sh first: source setup.sh
    - Activate conda environment: conda activate nh-rs2v-venv
    - Dataset must be downloaded

Usage:
    conda activate nh-rs2v-venv
    cd /scratch/users/Proulx-S/s2v2
    python simple_slsqp_example_visualized.py
"""

import sys
import os

# Import required packages
try:
    import napari
    import numpy as np
    import torch
    from joblib import Parallel
    from napari.settings import get_settings
    from napari.utils import Colormap
    from napari.utils.theme import get_theme, register_theme
    import cmcrameri.cm as cmc
    
    from nh_rs2v_dataset.dataset import NHRS2VDataset
    from nh_rs2v_dataset.utils.metrics import compute_metrics
    from nh_rs2v_dataset.utils.sampling import sample_slice
    from nh_rs2v_dataset.utils.transforms import S2VTransform, CanonicalTransform
    from nh_rs2v_baseline.optim.solve import solve_for_RT
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


# ============================================================================
# HARDCODED MRI PROBLEM - Update this after running find_mri_problems.py
# ============================================================================
# Replace these values with the problem you selected from find_mri_problems.py
HARDCODED_PROBLEM_ID = None  # e.g., "scan_name-0"
HARDCODED_PROBLEM_INDEX = None  # e.g., 42
# ============================================================================


IDX_SLICE_EDGES = [[0, 1], [1, 2], [2, 3], [3, 0]]


def transform_slice_vertices(A, size):
    """Transform slice vertices using transformation matrix A."""
    PX_SLICE_VERTICES = np.array(
        [
            [0, 0, 0.0, 1],
            [size, 0, 0.0, 1],
            [size, size, 0.0, 1],
            [0, size, 0.0, 1],
        ]
    )
    slice_vertices = np.copy(PX_SLICE_VERTICES)
    
    for ii in range(4):
        slice_vertices[ii] = A @ slice_vertices[ii]
    
    return slice_vertices[:, :3]


def setup_napari():
    """Setup napari theme."""
    get_settings().application.ipy_interactive = False
    pub_theme = get_theme("dark", False)
    pub_theme.id = "paper"
    pub_theme.canvas = "0xFFFFFF"
    register_theme("paper", pub_theme, "custom")


def visualize_results(dataset, item, R_pred, T_pred, S_gt, A_gt, problem_id, opt_name):
    """Visualize registration results using napari."""
    setup_napari()
    
    s2v = S2VTransform.from_RTS(dataset.cfg.SIZE, R_pred, T_pred, S_gt)
    
    napari_batlowW = Colormap(
        name="batlowW",
        colors=cmc.batlowW.colors,
        limits=(0, len(cmc.batlowW.colors) - 1),
    )
    
    slice_vertices = transform_slice_vertices(s2v.get_A(), dataset.cfg.SIZE)
    slice_vertices_gt = transform_slice_vertices(A_gt, dataset.cfg.SIZE)
    
    viewer = napari.Viewer(
        title=f"Registration Results: {problem_id}", ndisplay=3
    )
    
    # Add volume
    viewer.add_image(
        item["volume"][0].cpu().numpy(),
        name="Volume",
        colormap=napari_batlowW,
        opacity=1.0,
        visible=True,
        rendering="average",
        blending="translucent_no_depth",
    )
    
    # Add predicted slice
    viewer.add_image(
        sample_slice(
            item["volume"][0, 0].cpu().numpy(),
            s2v.get_A(),
            dataset.cfg.SIZE,
            inverted_A=True,
        )[:, :, np.newaxis],
        name=f"Slice [{opt_name}]",
        affine=s2v.get_A(),
        opacity=1.0,
        colormap=napari_batlowW,
        visible=True,
        rendering="average",
        blending="translucent_no_depth",
    )
    
    # Add ground truth slice
    viewer.add_image(
        item["slice"][0, 0].unsqueeze(2).cpu().numpy(),
        name="Slice [GT, red border]",
        affine=A_gt,
        opacity=1.0,
        colormap=napari_batlowW,
        visible=True,
        rendering="average",
        blending="translucent_no_depth",
    )
    
    # Add canonical slice
    viewer.add_image(
        item["slice"][0, 0].unsqueeze(2).cpu().numpy(),
        name="Slice [Canonical]",
        affine=CanonicalTransform(dataset.cfg.SIZE, S_gt).get_ST_c_inv(),
        opacity=1.0,
        colormap=napari_batlowW,
        visible=False,
        rendering="average",
        blending="translucent_no_depth",
    )
    
    # Add ground truth slice border (red)
    for edge in IDX_SLICE_EDGES:
        viewer.add_shapes(
            data=np.array([slice_vertices_gt[edge[0]], slice_vertices_gt[edge[1]]]),
            shape_type="line",
            edge_color="red",
            edge_width=2.0,
            opacity=1.0,
            blending="translucent",
        )
    
    # Add predicted slice border (blue)
    for edge in IDX_SLICE_EDGES:
        viewer.add_shapes(
            data=np.array([slice_vertices[edge[0]], slice_vertices[edge[1]]]),
            shape_type="line",
            edge_color="#0072b2",  # Blue
            edge_width=2.0,
            opacity=1.0,
            blending="translucent",
        )
    
    # Add volume bounding box
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
    
    for edge in edges:
        viewer.add_shapes(
            data=np.array([cube_vertices[edge[0]], cube_vertices[edge[1]]]),
            shape_type="line",
            edge_color="black",
            edge_width=1.0,
            opacity=1.0,
            blending="translucent",
        )
    
    # Set camera view
    viewer.camera.perspective = 35
    viewer.camera.center = (64, 64, 64)
    viewer.camera.set_view_direction(
        (0.7512966834411156, 0.3692904441171027, -0.5469715361279534)
    )
    viewer.camera.zoom = 3.0
    viewer.theme = "paper"
    viewer.show()
    
    napari.run()


def main():
    """Main function."""
    # Configuration
    data_cache_path = "/scratch/users/Proulx-S/tools/NH-RS2V-dataset/data"
    population_size = 64  # Reduced for faster execution
    loss_name = "MAE"
    opt_name = "scipy_SLSQP"
    backend = "numpy"
    
    print("=" * 60)
    print("NH-RS2V Dataset - scipy_SLSQP Example (Visualized)")
    print("=" * 60)
    
    # Check if problem is hardcoded
    if HARDCODED_PROBLEM_INDEX is None or HARDCODED_PROBLEM_ID is None:
        print("\nERROR: Please set HARDCODED_PROBLEM_INDEX and HARDCODED_PROBLEM_ID")
        print("Run find_mri_problems.py first to select a problem, then update")
        print("the hardcoded values at the top of this script.")
        return
    
    # Load dataset
    print("\n1. Loading dataset...")
    dataset = NHRS2VDataset(
        data_cache_path=data_cache_path,
        mode="val",
        reduced_set=True,
        cupy_sampling=(backend == "cupy"),
        fast_dev_run=-1,
    )
    
    print(f"   Dataset loaded: {len(dataset)} problems available")
    
    # Load the hardcoded problem
    print(f"\n2. Loading problem: {HARDCODED_PROBLEM_ID} (index {HARDCODED_PROBLEM_INDEX})...")
    if HARDCODED_PROBLEM_INDEX >= len(dataset):
        print(f"ERROR: Problem index {HARDCODED_PROBLEM_INDEX} is out of range!")
        return
    
    item = dataset[HARDCODED_PROBLEM_INDEX]
    
    # Verify problem ID matches
    if item["problem_id"] != HARDCODED_PROBLEM_ID:
        print(f"WARNING: Problem ID mismatch!")
        print(f"  Expected: {HARDCODED_PROBLEM_ID}")
        print(f"  Got: {item['problem_id']}")
        print("  Continuing anyway...")
    
    # Extract data
    slice_img = item["slice"][0].numpy()
    volume = item["volume"][0].numpy()
    S_gt = item["S_gt"].numpy()
    R_gt = item["R_gt"].numpy()
    T_gt = item["T_gt"].numpy()
    A_gt = item["A_gt"].numpy()
    problem_id = item["problem_id"]
    
    print(f"   Problem ID: {problem_id}")
    print(f"   Slice shape: {slice_img.shape}")
    print(f"   Volume shape: {volume.shape}")
    
    # Define optimization bounds
    bounds = np.zeros((7, 2), dtype=np.float64)
    bounds[:, 0] = np.array([-1, -1, -1, -1, 0, 0, 0])
    bounds[:, 1] = np.array([1, 1, 1, 1, 1, 1, 1])
    
    # Set up parallel processing
    parallel = Parallel(n_jobs=-1, verbose=1, backend="loky")
    
    # Solve the registration problem
    print(f"\n3. Solving registration with {opt_name}...")
    print(f"   Population size: {population_size}")
    print(f"   Loss function: {loss_name}")
    print("   (This may take a few minutes...)")
    
    R_pred, T_pred = solve_for_RT(
        opt_name=opt_name,
        loss_name=loss_name,
        bounds=bounds,
        in_slice=slice_img,
        in_volume=volume,
        S_gt=S_gt,
        size=dataset.cfg.SIZE,
        population_size=population_size,
        parallel=parallel,
        backend=backend,
    )
    
    # Evaluate results
    print("\n4. Evaluating results...")
    metrics = {}
    metrics["problem_id"] = problem_id
    compute_metrics(
        metrics,
        R_gt,
        T_gt,
        R_pred,
        T_pred,
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        A_gt,
        inlier_threshold=3 * np.sqrt(3),
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Problem ID: {metrics['problem_id']}")
    print(f"\nAngular Pose Error: {metrics['ang_pose_err']:.2f} degrees")
    
    if metrics['ang_pose_err'] < 5.0:
        print(f"\n✓ Registration EXCELLENT (error < 5°)")
    elif metrics['ang_pose_err'] < 10.0:
        print(f"\n✓ Registration VERY GOOD (error < 10°)")
    elif metrics['ang_pose_err'] < 20.0:
        print(f"\n✓ Registration SUCCESSFUL (error < 20°)")
    else:
        print(f"\n✗ Registration needs improvement (error >= 20°)")
    
    # Visualize results
    print("\n5. Visualizing results...")
    print("   Opening napari viewer...")
    print("   - Red border: Ground truth slice position")
    print("   - Blue border: Predicted slice position")
    print("   - Close the viewer window when done")
    
    visualize_results(
        dataset, item, R_pred, T_pred, S_gt, A_gt, problem_id, opt_name
    )
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()


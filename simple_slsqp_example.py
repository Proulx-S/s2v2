"""
Simple example using scipy_SLSQP for slice-to-volume registration on MRI data.

This script demonstrates how to:
1. Load data from the NH-RS2V dataset
2. Filter for MRI data (or use any available data)
3. Solve a registration problem using scipy_SLSQP optimization
4. Evaluate the results

Prerequisites:
    - Run setup.sh first: source setup.sh
      This creates a conda environment (nh-rs2v-venv) and installs all packages
    - After setup.sh completes, the conda environment should be activated
    - Dataset will auto-download on first run

Usage:
    # First time setup (creates conda env and installs packages):
    cd /scratch/users/Proulx-S/s2v2
    source setup.sh
    
    # After setup, if conda environment is not active:
    conda activate nh-rs2v-venv
    
    # Then run the script:
    cd /scratch/users/Proulx-S/s2v2
    python simple_slsqp_example.py
"""

import sys
import os

# Import required packages (must be installed via poetry)
try:
    from nh_rs2v_dataset.dataset import NHRS2VDataset
    from nh_rs2v_dataset.utils.metrics import compute_metrics
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

import numpy as np
from joblib import Parallel


def find_mri_problem(dataset, max_search=100):
    """
    Find a problem from MRI data by checking scan names.
    
    Args:
        dataset: NHRS2VDataset instance
        max_search: Maximum number of problems to check
        
    Returns:
        Index of first MRI problem found, or 0 if none found
    """
    if not hasattr(dataset, 'dict_split') or 'set_scan_names' not in dataset.dict_split:
        print("Warning: Could not access scan names. Using first problem.")
        return 0
    
    # Look for MRI-related scan names (case-insensitive)
    mri_keywords = ['mri', 'total', 'segmentator', 'medical']
    
    scan_names = dataset.dict_split['set_scan_names']
    index_split = dataset.index_split
    
    print(f"Available scan names: {set(scan_names)}")
    
    # Find first problem from an MRI-related scan
    for idx in range(min(len(index_split), max_search)):
        problem = index_split[idx]
        scan_name = problem['set_name']
        
        # Check if scan name contains any MRI keyword
        if any(keyword.lower() in scan_name.lower() for keyword in mri_keywords):
            print(f"Found MRI-related problem at index {idx}: scan_name='{scan_name}'")
            return idx
    
    print("No MRI-specific scan found. Using first available problem.")
    return 0


def main():
    """Main function to run the example."""
    
    # Configuration
    # Dataset path - matches setup.sh structure
    # The dataset will auto-download here on first run
    data_cache_path = "/scratch/users/Proulx-S/tools/NH-RS2V-dataset/data"
    population_size = 64  # Reduced for faster execution (original uses 2048)
    loss_name = "MAE"  # Mean Absolute Error
    opt_name = "scipy_SLSQP"
    backend = "numpy"  # Use "cupy" if you have GPU and CuPy installed
    
    print("=" * 60)
    print("NH-RS2V Dataset - scipy_SLSQP Example")
    print("=" * 60)
    
    # Load dataset in validation mode
    print("\n1. Loading dataset...")
    dataset = NHRS2VDataset(
        data_cache_path=data_cache_path,
        mode="val",
        reduced_set=True,  # Use reduced set for faster loading
        cupy_sampling=(backend == "cupy"),
        fast_dev_run=-1,  # Load all problems in reduced set
    )
    
    print(f"   Dataset loaded: {len(dataset)} problems available")
    
    # Find an MRI problem (or use first available)
    print("\n2. Finding MRI problem...")
    problem_idx = find_mri_problem(dataset)
    
    # Get the problem
    print(f"\n3. Loading problem at index {problem_idx}...")
    item = dataset[problem_idx]
    
    # Extract data
    slice_img = item["slice"][0].numpy()  # Remove channel dimension: (128, 128)
    volume = item["volume"][0].numpy()  # Remove channel dimension: (128, 128, 128)
    S_gt = item["S_gt"].numpy()
    R_gt = item["R_gt"].numpy()
    T_gt = item["T_gt"].numpy()
    A_gt = item["A_gt"].numpy()
    problem_id = item["problem_id"]
    
    print(f"   Problem ID: {problem_id}")
    print(f"   Slice shape: {slice_img.shape}")
    print(f"   Volume shape: {volume.shape}")
    print(f"   Size: {dataset.cfg.SIZE}")
    
    # Define optimization bounds (7 parameters: 3 rotation, 3 translation, 1 scale)
    bounds = np.zeros((7, 2), dtype=np.float64)
    bounds[:, 0] = np.array([-1, -1, -1, -1, 0, 0, 0])
    bounds[:, 1] = np.array([1, 1, 1, 1, 1, 1, 1])
    
    # Set up parallel processing
    parallel = Parallel(n_jobs=-1, verbose=1, backend="loky")
    
    # Solve the registration problem
    print(f"\n4. Solving registration with {opt_name}...")
    print(f"   Population size: {population_size}")
    print(f"   Loss function: {loss_name}")
    print(f"   Backend: {backend}")
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
    
    print("\n5. Evaluating results...")
    
    # Compute metrics
    metrics = {}
    metrics["problem_id"] = problem_id
    compute_metrics(
        metrics,
        R_gt,
        T_gt,
        R_pred,
        T_pred,
        np.zeros((0, 3)),  # No feature matches for optimization-based method
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
    print(f"  (Rotation + Translation combined)")
    print(f"\nGround Truth:")
    print(f"  R_gt shape: {R_gt.shape}")
    print(f"  T_gt: {T_gt}")
    print(f"\nPredicted:")
    print(f"  R_pred shape: {R_pred.shape}")
    print(f"  T_pred: {T_pred}")
    
    # Check if registration was successful (error < 20 degrees)
    if metrics['ang_pose_err'] < 5.0:
        print(f"\n✓ Registration EXCELLENT (error < 5°)")
    elif metrics['ang_pose_err'] < 10.0:
        print(f"\n✓ Registration VERY GOOD (error < 10°)")
    elif metrics['ang_pose_err'] < 20.0:
        print(f"\n✓ Registration SUCCESSFUL (error < 20°)")
    else:
        print(f"\n✗ Registration needs improvement (error >= 20°)")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()


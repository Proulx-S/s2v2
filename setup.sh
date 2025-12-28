#! /bin/bash
WORKDIR=/scratch/users/Proulx-S/s2v2
TOOLDIR=/scratch/users/Proulx-S/tools

# Create a conda environment with Python 3.11 (required for numpy 1.23.0 compatibility)
cd $WORKDIR
conda create -y -n nh-rs2v-venv python=3.11
conda activate nh-rs2v-venv

# Install poetry if not already available
if ! command -v poetry &> /dev/null; then
    pip install poetry
fi

# Install the dataset
cd $TOOLDIR
if [ ! -d "NH-RS2V-dataset" ]; then
    git clone https://github.com/xaf-cv/NH-RS2V-dataset.git
fi
cd NH-RS2V-dataset
poetry install

# Install the algorithm
cd $TOOLDIR
if [ ! -d "NH-RS2V-baselines" ]; then
    git clone https://github.com/xaf-cv/NH-RS2V-baselines.git
fi
cd NH-RS2V-baselines
poetry install

# Trigger dataset download by initializing the dataset
cd $WORKDIR
echo ""
echo "============================================================"
echo "Triggering dataset download..."
echo "============================================================"
echo "This will download ~35GB compressed (~87GB uncompressed)"
echo "The download will start automatically..."
echo ""

python -c "
import sys
import os
sys.path.insert(0, '$TOOLDIR/NH-RS2V-dataset')
sys.path.insert(0, '$TOOLDIR/NH-RS2V-baselines')

try:
    from nh_rs2v_dataset.dataset import NHRS2VDataset
    print('Initializing dataset to trigger download...')
    dataset = NHRS2VDataset(
        data_cache_path='$TOOLDIR/NH-RS2V-dataset/data',
        mode='val',
        reduced_set=True,
        cupy_sampling=False,
        fast_dev_run=1  # Just load 1 problem to trigger download
    )
    print(f'Dataset initialized successfully! ({len(dataset)} problems available)')
    print('Download complete!')
except Exception as e:
    print(f'Error during dataset initialization: {e}')
    print('You may need to download the dataset manually when running the scripts.')
    sys.exit(1)
"

echo ""
echo "Setup complete!"
cd $WORKDIR
#!/bin/bash

# Install LightGlue from GitHub
echo "Installing LightGlue from GitHub..."
pip install git+https://github.com/cvg/LightGlue.git

# Create necessary directory for model checkpoints
echo "Creating necessary directory for model checkpoints..."
mkdir -p ~/.cache/torch/hub/checkpoints

# Download and extract LightGlue model checkpoints
echo "Downloading and extracting LightGlue model checkpoints..."
kaggle models instances versions download oldufo/lightglue/pyTorch/aliked/1
tar -xzf lightglue.tar.gz
cp aliked_lightglue.pth ~/.cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv.pth

# Download and extract ALIKED model checkpoints
echo "Downloading and extracting ALIKED model checkpoints..."
kaggle models instances versions download oldufo/aliked/pyTorch/aliked-n16/1
tar -xzf aliked.tar.gz
cp aliked-n16.pth ~/.cache/torch/hub/checkpoints/aliked-n16.pth

# Download and extract DinoV2 model checkpoints
echo "Downloading and extracting DinoV2 model checkpoints..."
kaggle models instances versions download metaresearch/dinov2/pyTorch/base/1
mkdir -p dinov2/pytorch/base/1
tar -xzf dinov2.tar.gz -C dinov2/pytorch/base/1

#Setting up 3d-viz library hloc
echo "Setting up hloc..."
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/
python3 -m pip install -e .

echo "All installations and setups are completed!"


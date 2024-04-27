#!/bin/bash

# Install LightGlue from GitHub
pip install git+https://github.com/cvg/LightGlue.git

# Create necessary directory for model checkpoints
mkdir -p ~/.cache/torch/hub/checkpoints

# Download and extract LightGlue model checkpoints
kaggle models instances versions download oldufo/lightglue/pyTorch/aliked/1
tar -xzf lightglue.tar.gz
cp aliked_lightglue.pth ~/.cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv.pth

# Download and extract another version of LightGlue model checkpoints
kaggle models instances versions download oldufo/aliked/pyTorch/aliked-n16/1
tar -xzf aliked.tar.gz
cp aliked-n16.pth ~/.cache/torch/hub/checkpoints/aliked-n16.pth

# Download and extract DinoV2 model checkpoints
kaggle models instances versions download metaresearch/dinov2/pyTorch/base/1
mkdir -p dinov2/pytorch/base/1
tar -xzf base.tar.gz -C dinov2/pytorch/base/1

echo "All installations and setups are completed!"

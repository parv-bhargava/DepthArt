# General utilities
import matplotlib.pyplot as plt

import os
# Set environment variables
os.environ['PATH'] += ':/usr/local/cuda/bin'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64'
import subprocess
from tqdm import tqdm
from pathlib import Path
from time import time, sleep
from fastprogress import progress_bar
import gc
import numpy as np
import h5py
from IPython.display import clear_output
from collections import defaultdict
from copy import deepcopy
from typing import Any
import itertools
import pandas as pd

# CV/MLe
import cv2
import torch
from torch import Tensor as T
import torch.nn.functional as F
import kornia as K
import kornia.feature as KF
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from hloc.utils import viz_3d

import torch
from lightglue import match_pair
from lightglue import LightGlue, ALIKED
from lightglue.utils import load_image, rbd

# 3D reconstruction
import pycolmap

# Provided by organizers
from scripts.database import *
from scripts.h5_to_db import *

def arr_to_str(a):
    """Returns ;-separated string representing the input"""
    return ";".join([str(x) for x in a.reshape(-1)])

def load_torch_image(file_name: Path | str, device=torch.device("cpu")):
    """Loads an image and adds batch dimension"""
    img = K.io.load_image(file_name, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img

def plot_reconstruction(rec_gt, name='Reconstruction.html'):
    """Plots the 3D reconstruction"""
    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(fig, rec_gt, cameras=False, color='rgba(255,50,255, 0.5)', name="Ground Truth", cs=5)
    fig.write_html(name)

def run_colmap_command(command):
    """ Utility function to run a COLMAP command. """
    try:
        output = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print(output.stdout)
    except subprocess.CalledProcessError as e:
        print("Error running command:", command)
        print(e.output)

def colmap_dense_reconstruction(image_path, sparse_model_path, output_path):
    # Image Undistortion
    dense_path = os.path.join(output_path, 'dense')
    os.makedirs(dense_path, exist_ok=True)
    run_colmap_command(f"colmap image_undistorter --image_path '{image_path}' --input_path '{sparse_model_path}' --output_path '{dense_path}' --output_type COLMAP --max_image_size 2000")

    # Dense Reconstruction
    run_colmap_command(f"colmap patch_match_stereo --workspace_path '{dense_path}' --workspace_format COLMAP --PatchMatchStereo.geom_consistency true")

    # Stereo Fusion
    fused_ply_path = os.path.join(dense_path, 'fused.ply')
    run_colmap_command(f"colmap stereo_fusion --workspace_path '{dense_path}' --workspace_format COLMAP --input_type geometric --output_path '{fused_ply_path}'")

    print("Dense reconstruction completed. Output stored at:", fused_ply_path)


def save_rot_tra_info(maps, filename):
    data = []
    for k, im in maps[0].images.items():
        rotation_matrix = im.cam_from_world.rotation.matrix()
        translation = im.cam_from_world.translation
        data.append({
            "Image ID": k,
            "Rotation Matrix": rotation_matrix,
            "Translation": translation
        })
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

device = K.utils.get_cuda_device_if_available(0)
print(device)

# DEBUG = len([p for p in Path("/kaggle/input/image-matching-challenge-2024/test/").iterdir() if p.is_dir()]) == 2
# print("DEBUG:", DEBUG)
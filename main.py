from pathlib import Path
import pycolmap
import pandas as pd
import torch
from scripts.extract_keypoint import detect_keypoints
from scripts.image_pair import get_image_pairs
from scripts.keypoint_distance import keypoint_distances
from scripts.match import visualize_matches
from scripts.ransac import import_into_colmap
from scripts.utils import plot_reconstruction,colmap_dense_reconstruction,save_rot_tra_info

PATH = '/home/ubuntu/DepthArt/saj'
EXT = 'jpg'
PATH_FEATURES = '/home/ubuntu/DepthArt/features'
DINO_PATH = '/home/ubuntu/DepthArt/dinov2/pytorch/base/1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get Image Pairs for Matching using DINO
images_list = list(Path(PATH).glob(f'*.{EXT}'))
index_pairs = get_image_pairs(images_list, DINO_PATH)

# Extract keypoints using ALIKED
feature_dir = Path(PATH_FEATURES)
detect_keypoints(images_list, feature_dir, device=device)

# Compute Keypoint Distances  using LightGlue
keypoint_distances(images_list, index_pairs, feature_dir, verbose=False, device=device)

# Visualise Image matching
idx1, idx2 = index_pairs[2]
visualize_matches(images_list, idx1, idx2, feature_dir)



# Import into Colmap
database_path = "colmap_saj.db"
images_dir = images_list[0].parent
import_into_colmap(
    images_dir,
    feature_dir,
    database_path,
)

# This does RANSAC
pycolmap.match_exhaustive(database_path)

# This does the reconstruction
mapper_options = pycolmap.IncrementalPipelineOptions()
mapper_options.min_model_size = 3
mapper_options.max_num_models = 2

maps = pycolmap.incremental_mapping(
    database_path=database_path,
    image_path=images_dir,
    output_path=Path.cwd() / "incremental_pipeline_outputs",
    options=mapper_options,
)

# Save rotational matrix and translation info into a csv file
save_rot_tra_info(maps, 'rot_tra_info.csv')

# Visualize the 3D reconstruction
# plot_reconstruction(maps[0], 'Reconstructionsaj.html')


# Dense Reconstruction
# colmap_dense_reconstruction(images_dir, database_path, Path.cwd())


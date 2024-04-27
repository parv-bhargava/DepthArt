from scripts.image_pair import get_image_pairs
from scripts.extract_keypoint import detect_keypoints
from scripts.keypoint_distance import keypoint_distances
from scripts.ransac import import_into_colmap
from scripts.match import visualize_matches
from pathlib import Path
import pycolmap
import torch
import pycolmap
from hloc.utils import viz_3d


PATH = '/home/ubuntu/DepthArt/train/haiper/bike/images'
EXT = 'jpeg'
PATH_FEATURES = '/home/ubuntu/DepthArt/features'
DINO_PATH = '/home/ubuntu/DepthArt/dinov2/pytorch/base/1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get Image Pairs
images_list = list(Path(PATH).glob(f'*.{EXT}'))[:10]
index_pairs = get_image_pairs(images_list, DINO_PATH)

# Extract keypoints
feature_dir = Path(PATH_FEATURES)
detect_keypoints(images_list, feature_dir, device=device)

# Compute Keypoint Distances
keypoint_distances(images_list, index_pairs, feature_dir, verbose=False, device=device)

# Image matching
idx1,idx2= index_pairs[2]
visualize_matches(images_list, idx1, idx2, feature_dir)

# Import into Colmap
database_path = "colmap.db"
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


unidstorted_outputs = Path.cwd() / "unidstorted_outputs"
fusion_outputs = Path.cwd() / "stereofusion_outputs"
sparse_outputs = Path.cwd() / 'sparse_outputs'
workspace_path = Path.cwd() / 'workspace'

# Run sparse reconstruction
rec_gt = pycolmap.Reconstruction(f'{Path.cwd()}/incremental_pipeline_outputs/0')
fig = viz_3d.init_figure()
viz_3d.plot_cameras(fig, rec_gt, color='rgba(50,255,50, 0.5)', name="Ground Truth", size=10)
viz_3d.plot_reconstruction(fig, rec_gt, cameras = False, color='rgba(255,50,255, 0.5)', name="Ground Truth", cs=5)
fig.write_html('Sparse_Reconstruction.html')
# fig.show()
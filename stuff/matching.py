import cv2
import torch
from lightglue import match_pair
from lightglue import ALIKED, LightGlue
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import pycolmap
from pathlib import Path
import os
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")
extractor = ALIKED(max_num_keypoints=4096, detection_threshold=0.01).eval().to(device)
matcher = LightGlue(features='aliked').eval().to(device) # load the matcher

image0 = load_image('kn_church-2.jpg').to(device)
image1 = load_image('kn_church-8.jpg').to(device)



with torch.inference_mode():
    feats0, feats1, matches01 = match_pair(extractor, matcher, image0, image1)




kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
viz2d.save_plot('kn_church.png')



# Setup paths and database
project_path = Path(os.getcwd())
project_path.mkdir(parents=True, exist_ok=True)
database_path = project_path / 'database.db'

# Initialize the database
db = pycolmap.ReconstructionDatabase(str(database_path))
db.create_tables()

# Assuming you have image paths and their IDs
image_id1 = db.add_image('kn_church-2.jpg', 1)
image_id2 = db.add_image('kn_church-8.jpg', 2)

# Add keypoints (convert your keypoints to the required format)
db.add_keypoints(image_id1, m_kpts0.cpu().numpy().astype(np.float32))
db.add_keypoints(image_id2, m_kpts1.cpu().numpy().astype(np.float32))

# Add matches
matches = np.stack((matches01['matches'][..., 0], matches01['matches'][..., 1]), axis=1)
db.add_matches(image_id1, image_id2, matches)

# Run the mapper
mapper_options = pycolmap.MapperOptions()
# Set any specific options you need
pycolmap.run_mapper(database_path=str(database_path), image_path=str(project_path), output_path=str(project_path / 'sparse'), options=mapper_options)

# Load the reconstruction
rec = pycolmap.Reconstruction(str(project_path / 'sparse/0'))

# image_path = 'path/to/your/images'
# sparse_path = 'path/to/your/sparse'
# dense_path = 'path/to/your/dense'
#
# # Undistort images
# pycolmap.undistort_images(image_path=image_path, sparse_model_path=sparse_path, dense_workspace_path=dense_path)
#
# # Optionally, run MVS using PyCOLMAP or COLMAP's CLI
# pycolmap.patch_match_stereo(dense_workspace_path=dense_path)
# pycolmap.stereo_fusion(dense_workspace_path=dense_path, output_model_path=dense_path + '/fused.ply')
#

# kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
# fig = viz2d.plot_images([image0, image1])
# fig1 = viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)

# viz2d.plot_images([image0, image1])
# viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
# viz2d.save_plot('kn_church_keypoints.png')
# viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)

import torch
# from lightglue import match_pair, ALIKED, LightGlue, viz2d
# from lightglue.utils import load_image
#
# def process_and_visualize_matches(image_path0, image_path1, output_path):
#     # Setup device and models
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     print(f"Using device: {device}")
#     extractor = ALIKED(max_num_keypoints=4096, detection_threshold=0.01).eval().to(device)
#     matcher = LightGlue(features='aliked').eval().to(device)
#
#     # Load images
#     image0 = load_image(image_path0).to(device)
#     image1 = load_image(image_path1).to(device)
#
#     # Process images
#     with torch.inference_mode():
#         feats0, feats1, matches01 = match_pair(extractor, matcher, image0, image1)
#
#     # Extract keypoints and matches
#     kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
#     m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
#
#     # Visualization
#     axes = viz2d.plot_images([image0, image1])
#     viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
#     viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
#     viz2d.save_plot(output_path)
#
# # Example usage
# process_and_visualize_matches('path_to_image0.jpg', 'path_to_image1.jpg', 'output_plot.png')

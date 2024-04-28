# # %%
# # Read and viuzulize images from the dataset
# import os
#
# import matplotlib.pyplot as plt
# from PIL import Image
#
#
# def read_images(path):
#     images = []
#     for filename in os.listdir(path):
#         img = Image.open(os.path.join(path, filename))
#         images.append(img)
#     return images
#
#
# def plot_images(images):
#     # Plot all images
#     fig, axes = plt.subplots(1, len(images), figsize=(20, 10))
#     for ax, img in zip(axes, images):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.show()
#
#
# category = "heritage"
# object_name = 'wall'
# path = f'/home/ubuntu/DepthArt/train/{category}/{object_name}/images'
# images = read_images(path)
# plot_images(images)
from scripts.utils import *
# Dense Reconstruction
database_path = '/home/ubuntu/DepthArt/incremental_pipeline_outputs/0'
images_dir = '/home/ubuntu/DepthArt/train/phototourism/colosseum_exterior/images'
colmap_dense_reconstruction(images_dir, database_path, Path.cwd())

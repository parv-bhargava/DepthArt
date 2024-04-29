# from image_pair import get_image_pairs
from scripts.utils import *
import streamlit as st
# images_list = list(Path("/home/ubuntu/DepthArt/train/haiper/bike/images").glob("*.jpeg"))[:10]
# index_pairs = get_image_pairs(images_list, "/home/ubuntu/DepthArt/dinov2/pytorch/base/1")
# print(index_pairs)
def points(images_list):
    if True:
        dtype = torch.float32  # ALIKED has issues with float16

        extractor = ALIKED(
            max_num_keypoints=4096,
            detection_threshold=0.01,
            resize=1024
        ).eval().to(device, dtype)

        path = images_list[0]
        image = load_torch_image(path, device=device).to(dtype)
        features = extractor.extract(image)

        fig, ax = plt.subplots(1, 2, figsize=(10, 20))
        ax[0].imshow(image[0, ...].permute(1, 2, 0).cpu())
        ax[1].imshow(image[0, ...].permute(1, 2, 0).cpu())
        ax[1].scatter(features["keypoints"][0, :, 0].cpu(), features["keypoints"][0, :, 1].cpu(), s=0.5, c="red")
        plt.show()
        st.pyplot(fig)
        del extractor
# Extract keypoints

# # Compute Keypoint Distances
# feature_dir = Path("/home/ubuntu/DepthArt/features")
# # keypoint_distances(images_list, index_pairs, feature_dir, verbose=False)
#
# if True:
#     matcher_params = {
#         "width_confidence": -1,
#         "depth_confidence": -1,
#         "mp": True if 'cuda' in str(device) else False,
#     }
#     matcher = KF.LightGlueMatcher("aliked", matcher_params).eval().to(device)
#
#     with h5py.File(feature_dir / "keypoints.h5", mode="r") as f_keypoints, \
#             h5py.File(feature_dir / "descriptors.h5", mode="r") as f_descriptors:
#         idx1, idx2 = index_pairs[0]
#         key1, key2 = images_list[idx1].name, images_list[idx2].name
#
#         keypoints1 = torch.from_numpy(f_keypoints[key1][...]).to(device)
#         keypoints2 = torch.from_numpy(f_keypoints[key2][...]).to(device)
#         print("Keypoints:", keypoints1.shape, keypoints2.shape)
#         descriptors1 = torch.from_numpy(f_descriptors[key1][...]).to(device)
#         descriptors2 = torch.from_numpy(f_descriptors[key2][...]).to(device)
#         print("Descriptors:", descriptors1.shape, descriptors2.shape)
#
#         with torch.inference_mode():
#             distances, indices = matcher(
#                 descriptors1,
#                 descriptors2,
#                 KF.laf_from_center_scale_ori(keypoints1[None]),
#                 KF.laf_from_center_scale_ori(keypoints2[None]),
#             )
#     print(distances, indices)

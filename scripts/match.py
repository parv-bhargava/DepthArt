from matplotlib.patches import Circle, ConnectionPatch
import matplotlib as mpl
from scripts.image_pair import get_image_pairs
from scripts.utils import *
import streamlit as st
# Constants and Configuration
DEBUG = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Path and image settings
# feature_dir = Path("/home/ubuntu/DepthArt/features/")
# images_list = list(Path("/home/ubuntu/DepthArt/train/haiper/bike/images").glob("*.jpeg"))[:10]
# index_pairs = get_image_pairs(images_list, "/home/ubuntu/DepthArt/dinov2/pytorch/base/1")

def visualize_matches(paths, idx1, idx2, feature_dir):
    # Load images
    print(idx1,idx2)
    try:
        img1 = Image.open(paths[idx1])
        img2 = Image.open(paths[idx2])
        img1 = np.array(img1)
        img2 = np.array(img2)
    except IOError as e:
        print(f"Error loading images: {e}")
        return

    # Make sure the images are the same height
    if img1.shape[0]==img2.shape[0]:
        if img1.shape[0] > img2.shape[0]:
            add = (img1.shape[0] - img2.shape[0])
            pad = np.zeros((add, img2.shape[1], 3))
            img2 = np.append(img2, pad, axis=0)
        elif img2.shape[0] > img1.shape[0]:
            add = (img2.shape[0] - img1.shape[0])
            pad = np.zeros((add, img1.shape[1], 3))
            img1 = np.append(img1, pad, axis=0)
    # Check the heights of the images
    # Determine the maximum height among both images
    else:
        max_height = max(img1.shape[0], img2.shape[0])

        # Resize both images to have the same height
        img1 = cv2.resize(img1, (int(img1.shape[1] * max_height / img1.shape[0]), max_height))
        img2 = cv2.resize(img2, (int(img2.shape[1] * max_height / img2.shape[0]), max_height))

    combined_image = np.hstack((img1, img2))

    # Open the feature files
    try:
        with h5py.File(feature_dir / "keypoints.h5", "r") as f_keypoints, \
                h5py.File(feature_dir / "matches.h5", "r") as f_matches:
            # Check if keypoints and matches data exist for the images
            if paths[idx1].name in f_keypoints and paths[idx2].name in f_keypoints:
                keypoints1 = np.array(f_keypoints[paths[idx1].name])
                keypoints2 = np.array(f_keypoints[paths[idx2].name])
                width = img1.shape[1]
            else:
                print("Keypoint data not found for one or both images.")
                return

            # Check if there are matches recorded
            if paths[idx1].name in f_matches and paths[idx2].name in f_matches[paths[idx1].name]:
                matches = np.array(f_matches[paths[idx1].name][paths[idx2].name])
            else:
                st.write("No matches found.")
                print("No matches found.")
                return
    except IOError as e:
        st.write(f"Error opening feature files: {e}")
        print(f"Error opening feature files: {e}")
        return

    # Visualization
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_aspect('equal')
    ax.imshow(combined_image)
    ax.axis('off')

    radius = 5  # Radius of circle
    thickness = 2  # Line thickness

    colormaps = mpl.colormaps['tab20c']
    norm = mpl.colors.Normalize(vmin=0, vmax=(len(matches) - 1))

    for i, match in enumerate(matches[:20]):
        point1 = keypoints1[match[0]]
        point2 = keypoints2[match[1]] + np.array([width, 0])  # Adjust point2's x-coordinate

        color = colormaps(norm(i))
        circle1 = Circle((point1[0], point1[1]), radius, color=color, fill=True, linewidth=thickness)
        circle2 = Circle((point2[0], point2[1]), radius, color=color, fill=True, linewidth=thickness)
        line = ConnectionPatch(point1, point2, 'data', 'data', arrowstyle='-', color=color, linewidth=thickness)

        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.add_artist(line)

    st.pyplot(fig)
    plt.show()


# Example usage within your debug condition
# if DEBUG:
#     idx1,idx2= index_pairs[2]
#
#     visualize_matches(images_list, idx1,idx2, feature_dir)


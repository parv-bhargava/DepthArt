import streamlit as st
from pathlib import Path
import os
from PIL import Image
import torch
from scripts.extract_keypoint import detect_keypoints
from scripts.match import visualize_matches
from scripts.keypoint_distance import keypoint_distances
from scripts.image_pair import get_image_pairs
from scripts.ransac import import_into_colmap
import pycolmap
from scripts.utils import plot_reconstruction
from scripts.test import points
import io
PATH_FEATURES = '/home/ubuntu/DepthArt/features'
DINO_PATH = '/home/ubuntu/DepthArt/dinov2/pytorch/base/1'
feature_dir = Path(PATH_FEATURES)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EXT = ['jpeg', 'jpg', 'png']
images_list = []


def save_uploaded_file(uploaded_file, base_path, dataset, scene):
    # Construct the directory path
    directory_path = f'{base_path}/{dataset}/{scene}/images'
    # Create the directory if it does not exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
    # Construct the full file path
    file_path = os.path.join(directory_path, uploaded_file.name)
    # Save the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def handle_dataset_choice():
    st.title("Choose Dataset")
    select_action = st.selectbox("Select function", ["Choose", "Custom Dataset", "Upload Dataset"])
    base_path = '/home/ubuntu/DepthArt/train'
    images_list = []
    # Local scope for image list handling

    if select_action == "Custom Dataset":
        dataset_option = st.selectbox("Select a dataset",
                                      ["British Museum", "Colosseum", "Lincoln Memorial", "Taj Mahal", "Nara Temple","Fountains","Kyiv Theater"])
        dataset_path_info = {
            "British Museum": ("phototourism", "british_museum"),
            "Colosseum": ("phototourism", "colosseum_exterior"),
            "Lincoln Memorial": ("phototourism", "lincoln_memorial_statue"),
            "Taj Mahal": ("phototourism", "taj_mahal"),
            "Nara Temple": ("phototourism", "temple_nara_japan"),
            "Fountains": ("haiper","fountain"),
            "Kyiv Theater":("urban","kyiv-puppet-theater")

        }
        dataset, scene = dataset_path_info[dataset_option]
        path = Path(f'{base_path}/{dataset}/{scene}/images')
        # images_list = sorted(list(path.glob('*.jpg')) + list(path.glob('*.jpeg')) + list(path.glob('*.png')))
        images_list = [Path(image) for image in
                       sorted(list(path.glob('*.jpg'))) + sorted(list(path.glob('*.jpeg'))) + sorted(
                           list(path.glob('*.png')))]
        # st.write(images_list)
        # st.write(path)
        st.success(f"Files loaded successfully")
        st.session_state['images_list'] = images_list
        st.session_state['scene'] = scene
        st.session_state['dataset'] = dataset
        st.session_state['path'] = path

    elif select_action == "Upload Dataset":
        dataset = "upload"
        scene = st.text_input("Enter scene name:")
        uploaded_files = st.file_uploader("Upload your dataset images", type=['png', 'jpg', 'jpeg'],
                                          accept_multiple_files=True)
        if uploaded_files:
            directory_path = Path(f'{base_path}/{dataset}/{scene}/images')  # Use Path for directory_path
            if not os.path.exists(directory_path):
                os.makedirs(directory_path, exist_ok=True)

            # Save each uploaded file and collect their paths
            image_paths = []
            for uploaded_file in uploaded_files:
                file_path = save_uploaded_file(uploaded_file, base_path, dataset, scene)
                image_paths.append(Path(file_path))  # Ensure file_path is converted to Path object here

            # After all files are saved, then perform the glob to list all images
            images_list = [image for ext in ['*.jpg', '*.jpeg', '*.png'] for image in
                           sorted(directory_path.glob(ext))]
            st.success(f"Files uploaded and saved successfully")

            # st.write(directory_path)
            # st.write(images_list)  # Display paths of all images in the directory
            st.session_state['images_list'] = images_list
            st.session_state['scene'] = scene
            st.session_state['dataset'] = dataset
            st.session_state['path'] = directory_path  # store directory path


def visualize_images():
    st.title("Let's Visualize")
    if 'scene' in st.session_state:
        st.subheader(f"Selected Scene: {st.session_state['scene']}")

    if 'images_list' in st.session_state and st.session_state['images_list']:
        image_paths = st.session_state['images_list']
        # Display only the first 12 images
        max_images = min(12, len(image_paths))
        cols = st.columns(3)  # Create 3 columns for images
        for index, image_path in enumerate(image_paths[:max_images]):
            with cols[index % 3]:  # Cycle through columns
                img = Image.open(image_path)
                st.image(img, caption=f"Image {index + 1}", use_column_width=True)
    else:
        st.error("No images to display.")


def extract():
    st.title("Extracting Keypoints ...")
    if 'scene' in st.session_state:
        st.subheader(f"Selected Scene: {st.session_state['scene']}")

    if 'images_list' in st.session_state and st.session_state['images_list']:
        images_list = st.session_state['images_list']

        # Assuming detect_keypoints function is defined to handle the list of images and save keypoints
        detect_keypoints(images_list, feature_dir, device=device)
        st.success("Keypoints detected and saved.")
        points(images_list)
    else:
        st.error("No images to display")


def image_match():
    st.title("Matching Images...")

    if 'scene' in st.session_state:
        st.subheader(f"Selected Scene: {st.session_state['scene']}")

    if 'images_list' in st.session_state and st.session_state['images_list']:
        images_list = st.session_state['images_list'][:10]
        feature_dir = Path(PATH_FEATURES)

        # Get image pairs
        index_pairs = get_image_pairs(images_list, DINO_PATH)

        # Compute Keypoint Distances
        keypoint_distances(images_list, index_pairs, feature_dir, verbose=False, device=device)

        # Visualize the first pair for example
        if index_pairs:

            # Let user select which pair to visualize
            try:
                selected_pair_index = st.selectbox("Select Image Pair to Visualize:", range(len(index_pairs)),
                                                   format_func=lambda x: index_pairs[x])
                idx1, idx2 = index_pairs[selected_pair_index]
                visualize_matches(images_list, idx1, idx2, feature_dir)
            except IndexError as e:
                st.error(f"Selected index out of range: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("No pairs to display. Ensure there are at least two images with detectable features.")





def perform_reconstruction(scene):
    if 'images_list' not in st.session_state or not st.session_state['images_list']:
        st.error("Image list is not available.")
        return
    images_list = st.session_state['images_list']
    database_path = f"colmap_{scene}.db"
    images_dir = images_list[0].parent
    import_into_colmap(
        images_dir,
        feature_dir,
        database_path,
    )
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
    # Visualize the 3D reconstruction
    plot_reconstruction(maps[0], f'Reconstruction_{scene}.html')
    path_to_html = Path(f'/home/ubuntu/DepthArt/Reconstruction_{scene}.html')
    with open(path_to_html, 'r') as f:
        html_data = f.read()

    st.components.v1.html(html_data, height=800)
    # Download

    st.download_button(
        label="Download HTML",
        data=html_data,
        file_name=f"Reconstruction_{scene}.html",
        mime="text/html"
    )


def handle_reconstruction():
    st.title("Reconstructing...")

    if 'scene' in st.session_state and 'dataset' in st.session_state:
        if st.session_state['dataset'] == 'upload':
            # Call the function to perform reconstruction
            perform_reconstruction(st.session_state['scene'])
        elif st.session_state['dataset'] != 'upload':
            # Show the HTML file related to the custom dataset

            html_file_path = f"/home/ubuntu/DepthArt/3D_sparse_reconstructions/Reconstruction_{st.session_state['scene']}.html"
            html_file = Path(html_file_path)
            if html_file.is_file():
                html_data = html_file.read_text(encoding='utf-8')
                st.components.v1.html(html_data, height=800, scrolling=True)
                # Download

                st.download_button(
                    label="Download HTML",
                    data=html_data,
                    file_name=f"Reconstruction_{st.session_state['scene']}.html",
                    mime="text/html"

                )
            else:
                st.error("HTML file does not exist for the selected scene.")
    else:
        st.error("No dataset selected or scene information is missing.")


def main():

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Introduction", "Literature Review", "Choose Dataset", "Visualize Images",
                                          "Extract Keypoints", "Match Images", "Sparse Reconstruction"])

    if app_mode == "Introduction":
        st.title("Image Matching and 3D Reconstruction")
    elif app_mode == "Literature Review":
        st.title("Literature Review")
    elif app_mode == "Choose Dataset":
        handle_dataset_choice()
    elif app_mode == "Visualize Images":
        visualize_images()
    elif app_mode == "Extract Keypoints":
        extract()
    elif app_mode == "Match Images":
        image_match()
    elif app_mode == "Sparse Reconstruction":
        handle_reconstruction()

    # Implement other app modes as needed


if __name__ == "__main__":
    main()


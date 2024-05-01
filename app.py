import os
from pathlib import Path

import pycolmap
import streamlit as st
import torch
from PIL import Image

from scripts.content import *
from scripts.extract_keypoint import detect_keypoints
from scripts.image_pair import get_image_pairs
from scripts.keypoint_distance import keypoint_distances
from scripts.match import visualize_matches
from scripts.ransac import import_into_colmap

from scripts.utils import plot_reconstruction
from scripts.test import points

# Current working directory
base_dir = os.getcwd()

# Join the current directory with the relative paths
PATH_FEATURES = os.path.join(base_dir, 'features')
DINO_PATH = os.path.join(base_dir, 'Dinov2', 'pytorch', 'base', '1')

feature_dir = Path(PATH_FEATURES)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EXT = ['jpeg', 'jpg', 'png']
images_list = []

def model_explanation():
    st.header("Model Explanation")
    select_action = st.selectbox("Understanding the Models", ["Choose", "Dinov2", "ALIKED", "LightGLUE"])
    if select_action == "Dinov2":

        dinotext= DINO
        st.markdown(dinotext)
        # dino_pca = os.path.join(base_dir, 'Architecture',
        #                               'Dino_pca.png')
        # # Display an image
        # st.image(dino_pca, caption="PCA Color Coded Patches")
        st.markdown(dinobody)
    elif select_action == "ALIKED":
        alikedtext1=alikedheader
        st.markdown(alikedtext1)
        deform_cnn = os.path.join(base_dir, 'Architecture',
                                      'deformable_convolution.png')
        # Display an image
        st.image(deform_cnn, caption="Deformable Convolution")
        alikedtext2=alikedintro
        st.markdown(alikedtext2)
        arch= os.path.join(base_dir, 'Architecture',
                                  'ALIKED_architecture.png')
        st.image(arch, caption="ALIKED Architecture")
        alikedtext3 = alikedbody
        st.markdown(alikedtext3)
        sddh= os.path.join(base_dir, 'Architecture',
                                  'SDDH.png')
        st.image(sddh, caption="SDDH")
        alikedtext4 = alikedconclusion
        st.markdown(alikedtext4)

    elif select_action == "LightGLUE":
        st.subheader("LightGLUE")
        lightglue = os.path.join(base_dir, 'Architecture',
                                 'LightGlue.png')
        # Display an image
        st.image(lightglue, caption="LightGLUE Architecture")
        light=light_glue
        st.markdown(light)
        prune = os.path.join(base_dir, 'Architecture',
                                 'pruning.png')
        # Display an image
        st.image(prune, caption="Pruning")
        st.markdown(lightbody)


def show_introduction():
    st.header(HEADER)
    introduction_markdown = INTRODUCTION
    st.markdown(introduction_markdown)
def process():
    st.header("Process Flow")
    flow = os.path.join(base_dir, 'Architecture',
                              'flow.png')
    # Display an image
    st.image(flow, caption="Process Flow")

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
    st.header("Choose Dataset")
    select_action = st.selectbox("Select function", ["Choose", "Custom Dataset", "Upload Dataset"])
    base_path = os.path.join(base_dir, 'train')
    images_list = []
    # Local scope for image list handling

    if select_action == "Custom Dataset":
        dataset_option = st.selectbox("Select a dataset",LIST_DATASETS)
        dataset_path_info = DATASET_PATH_INFO
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
            st.session_state['path'] = directory_path
            st.session_state['extract']='extract'# store directory path


def visualize_images():
    st.header("Let's Visualize")
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
    st.header("Extracting Keypoints ...")
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
    st.session_state['match'] = 'match'

def image_match():
    st.header("Matching Keypoints...")
    if 'match' not in st.session_state:
        st.error("Extract Keypoints first")

    elif 'scene' in st.session_state:
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
                    st.session_state['sparse'] = 'sparse'
                except IndexError as e:
                    st.error(f"Selected index out of range: {e}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("No pairs to display. Ensure there are at least two images with detectable features.")
    else:
        st.error("No images to display")

def to_3d():
    st.header("Going from 2D to 3D")
    tri = os.path.join(base_dir, 'Architecture',
                             'TriangulationIdeal.svg.png')
    # Display an image
    st.image(tri, caption="Triangulation")
    st.markdown(triangulation)
def perform_reconstruction(scene):
    if 'images_list' not in st.session_state or not st.session_state['images_list']:
        st.error("Image list is not available.")
        return

    images_list = st.session_state['images_list']
    # Assuming base_dir is defined somewhere above this function or you should define it here
    base_dir = Path.cwd()

    database_path = base_dir / f'colmap_{scene}.db'

    images_dir = images_list[0].parent if images_list else None
    if not images_list:
        st.error("No images available for processing.")
        return

    # Ensure database path does not exist before continuing
    if database_path.exists():
        database_path.unlink()
        st.success(f"Existing database {database_path} removed successfully.")

    import_into_colmap(
        images_dir,
        feature_dir,
        database_path,
    )
    pycolmap.match_exhaustive(database_path)

    # Reconstruction process
    mapper_options = pycolmap.IncrementalPipelineOptions()
    mapper_options.min_model_size = 3
    mapper_options.max_num_models = 2

    maps = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=images_dir,
        output_path=base_dir / "incremental_pipeline_outputs",
        options=mapper_options,
    )

    # Check if any maps were created
    if not maps:
        st.error("No reconstruction maps were generated. Check input data and parameters.")
        return

    # If maps are available, proceed with visualization and further processing
    plot_reconstruction(maps[0], f'Reconstruction_{scene}.html')
    path_to_html = base_dir / f'Reconstruction_{scene}.html'

    if path_to_html.exists():
        with open(path_to_html, 'r') as f:
            html_data = f.read()

        st.components.v1.html(html_data, height=800)
        st.download_button(
            label="Download HTML",
            data=html_data,
            file_name=f"Reconstruction_{scene}.html",
            mime="text/html"
        )



def handle_reconstruction():
    st.header("Reconstructing...")

    if 'match' not in st.session_state or 'sparse' not in st.session_state:
        st.error("Extract Keypoints and Match Images first")


    elif 'scene' in st.session_state and 'dataset' in st.session_state:
        st.subheader(f"Selected Scene: {st.session_state['scene']}")
        scene = st.session_state['scene']
        if st.session_state['dataset'] == 'upload':
            # Call the function to perform reconstruction
            perform_reconstruction(st.session_state['scene'])
        elif st.session_state['dataset'] != 'upload':
            # Show the HTML file related to the custom dataset

            html_file_path = os.path.join(base_dir, 'assets',
                                          f'Reconstruction_{scene}.html')
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


def further_scope():
    st.header("Further Scope")
    dense_reconstruction_text = DENSE_RECONSTRUCTION

    st.markdown(dense_reconstruction_text)


def references():
    st.header("References")
    st.markdown(reference)


def main():

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", GOTO)
    if app_mode == "Introduction":
        show_introduction()
    elif app_mode=="Process Flow":
        process()
    elif app_mode == "Model Explanation":
        model_explanation()
    elif app_mode == "Choose Dataset":
        handle_dataset_choice()
    elif app_mode == "Visualize Images":
        visualize_images()
    elif app_mode == "Extract Keypoints":
        extract()
    elif app_mode == "Match Keypoints":
        image_match()
    elif app_mode =="From 2D to 3D":
        to_3d()
    elif app_mode == "Sparse Reconstruction":
        handle_reconstruction()
    elif app_mode == "Further Scope":
        further_scope()
    elif app_mode == "References":
        references()

    # Implement other app modes as needed


if __name__ == "__main__":
    main()


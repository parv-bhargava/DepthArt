HEADER = "Image Matching and 3D Reconstruction"

INTRODUCTION = """
    #### Context
    The most accessible camera today is often the smartphone. When individuals capture and share photos of landmarks, these images represent limited two-dimensional perspectives. Structure from Motion (SfM) leverages these diverse snapshots to reconstruct three-dimensional models, enhancing our visual interaction with the world through advanced machine learning techniques.

    #### Technical Background
    Structure from Motion reconstructs 3D structures from sequences of 2D images, typically enhanced by motion sensors. While high-quality datasets are usually obtained under controlled conditions, creating 3D models from varied unstructured images introduces complexities due to inconsistent lighting, weather, and differing viewpoints.

    #### Problem Selection and Justification
    We focus on image matching to reconstruct 3D images from 2D photos, driven by its relevance to augmented reality, urban planning, and navigation systems. This challenge is central to computer vision, enabling machines to interpret visual environments in a human-like manner.

    #### Dataset Overview
    Our project employs a robust dataset designed for image matching technology benchmarking. This dataset features a wide range of images from various environments, ideal for developing and refining deep learning models to efficiently perform image matching.
        """

DENSE_RECONSTRUCTION = """
     #### Dense Reconstruction
    Dense reconstruction refers to the process of creating detailed, high-resolution 3D models from sets of images. Unlike Structure from Motion (SfM) that primarily produces sparse point clouds by identifying and matching keypoints across images, dense reconstruction techniques fill in the gaps, offering a comprehensive 3D representation of the photographed scene.

    This technique often utilizes methods such as Multi-View Stereo (MVS) to analyze the multiple images of a scene and reconstruct a dense mesh by considering a wider array of visible points. This not only improves the visual quality and usability of the reconstructed models for applications in virtual reality and augmented reality but also enhances the accuracy of environmental simulations and urban planning tools. The advent of deep learning has further pushed the boundaries of what's achievable with dense reconstruction, enabling more nuanced understanding and interaction with the physical world through digital twins.
        """
DATASET_PATH_INFO = {
    "British Museum": ("phototourism", "british_museum"),
    "Colosseum": ("phototourism", "colosseum_exterior"),
    "Lincoln Memorial": ("phototourism", "lincoln_memorial_statue"),
    "Taj Mahal": ("phototourism", "taj_mahal"),
    "Nara Temple": ("phototourism", "temple_nara_japan"),
    "Fountains": ("haiper", "fountain"),
    "Kyiv Theater": ("urban", "kyiv-puppet-theater")

}
LIST_DATASETS = ["British Museum", "Colosseum", "Lincoln Memorial", "Taj Mahal", "Nara Temple",
                 "Fountains", "Kyiv Theater"]

GOTO = ["Introduction", "Literature Review", "Choose Dataset", "Visualize Images",
        "Extract Keypoints", "Match Images", "Sparse Reconstruction", "Further Scope",
        "References"]

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
       
    For more details on the dataset, visit the [Kaggle competition page](https://www.kaggle.com/competitions/image-matching-challenge-2023/data).
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

GOTO = ["Introduction","Process Flow", "Model Explanation", "Choose Dataset", "Visualize Images",
        "Extract Keypoints", "Match Keypoints","From 2D to 3D", "Sparse Reconstruction", "Further Scope",
        "References"]

DINO="""
#### DINOv2: Learning Robust Visual Features through Vision Transformers
It is based on a self-supervised learning approach which means it needs no labels. It learns directly from the image. Thus, can capture better features.

**DINOv2’s Vision Transformer Architecture**

1.	The input image is first divided into fixed-size patches, which are then linearly embedded. Positional encodings are added to these embeddings to preserve information about the relative positions of the patches.

2.	The embeddings pass through multiple layers of the transformer encoder, each consisting of multi-headed self-attention and position-wise fully connected layers.

3.	DINO uses a teacher-student setup where both models are identical in architecture but differ in their parameter update dynamics. The student model tries to predict the output of the teacher. 
"""
dinobody="""
#### **How Image Matching Works?**

1. Extract dense embeddings from images using the pretrained Vision Transformer.

2. Feature Normalization: To ensure embeddings have consistent scale, crucial for reliable similarity measurements.

3. Efficient Pairwise Distances: Calculating distances between embeddings, facilitating the identification of similar images.

4. Select image pairs based on a similarity threshold.

"""

alikedheader="""
#### ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation

**Deformable Convolution** – A modified convolution that adds 2D offsets to the regular grid sampling locations of a standard convolution.
"""
alikedintro="""
The 2D convolution consists of two steps: 

1.	sampling using a regular grid over the input feature map
2.	Summation of sampled values weighted by W.

In deformable convolution the regular grid is augmented with offsets.

**Why deformable convolutions?**
"""
alikedbody="""
Traditionally, DNNs were used to extract descriptors of image patches at predefined keypoints. The conventional convolution operations lack to provide the geometric invariance required for the descriptor. The deformable convolutional network can model any geometric transformation by adjusting the offset for each pixel in the convolution.
ALIKED used a Sparse Deformable Descriptor Head (SDDH) to learn deformable positions of supporting features for each keypoint and constructs deformable descriptors.

"""
alikedconclusion="""
The input image I is initially encoded into multi-scale features {F1,F2,F3,F4} with encoding block1 to block4, and the number of channels of Fi is ci. Then, the multi-scale features are aggregated with upsample blocks (ublock4 to ublock1), and the output features Fui are concatenated to obtain the final image feature F. The Score Map Head (SMH) extracts the score map S with F followed by a Differentiable Keypoint Detection (DKD) module [10] to detect the keypoints p1 p2. The SDDH then efficiently extracts deformable invariant descriptors at the detected keypoints. “BN”, “poolingN”, and “DCN3x3” denote batch normalization, NxN average pooling, and 3x3 deformable convolution, respectively.
The SDDH estimates M deformable sample positions on KxK keypoint feature patches (K=5 in this example), samples M supporting features on the feature map based on the deformable sample positions, encodes the supporting features, and aggregates them with convM for descriptor extraction.

"""
light_glue="""
### 1. Input

The model begins by receiving inputs consisting of two sets of local features from two images (referred to as image A and image B). Each local feature in these sets comprises:
- **2D point location** (`p_i`): The coordinates (x, y) of the feature in its respective image, normalized by the image dimensions to fall between 0 and 1. (The keypoints)
- **Visual descriptor** (`d_i`): A high-dimensional vector extracted using a feature detector and descriptor (ALIKED in our case), which encodes the appearance around the feature point, allowing for a robust comparison across different views.(The descriptors)

### 2. Initial State Setting

Each feature's visual descriptor initializes its corresponding state vector in the model. This state vector (`x_i`) is what the Transformer architecture will manipulate through its layers to refine and compare feature information between the two images.

### 3. Transformer Backbone

This is where the primary computation of LightGlue occurs, involving layers of self-attention and cross-attention:

#### **Self-Attention Mechanism**

- **Goal**: To refine each feature's representation by aggregating contextual information from all other features within the same image.
- **Operation**:
  - Each feature generates **query (q)** and **key (k)** vectors through trainable transformations of its state vector.
  - The attention scores between all pairs of features within the image are calculated based on the dot product of queries and keys.
  - The feature states are updated by aggregating (via a weighted sum) the states of all other features in the image, weighted by the softmax-normalized attention scores.

#### **Cross-Attention Mechanism**

- **Goal**: To enhance the feature states by incorporating relevant information from corresponding features in the opposite image, essentially aligning features across images.
- **Operation**:
  - Each feature in image A generates a key, and each feature in image B uses these keys to calculate cross-image attention scores, and vice versa.
  - Similar to self-attention, the states are updated based on these cross-attention scores, allowing features in one image to pull information from the other.

### 4. Adaptive Mechanisms

To enhance efficiency, LightGlue employs adaptive depth and point pruning strategies:

#### **Adaptive Depth**
- **Concept**: The model assesses at each layer if further processing is necessary by evaluating a confidence measure.
- **Implementation**: A classifier predicts the confidence level of the current matches, and if it exceeds a predefined threshold, the model terminates early.

#### **Point Pruning**
- **Concept**: During processing, points determined as confidently unmatched or unmatchable are removed from subsequent calculations.
- **Implementation**: This reduces the computational load, especially in deeper layers, by focusing only on features still under consideration.
"""
lightbody=""""
### 5. Correspondence Prediction and Output

After the Transformer layers process the features:

#### **Assignment Scores**
- **Calculation**: The model computes a pairwise score for each pair of features across the two images, based on the similarity of their updated state vectors.

#### **Matchability Scores**
- **Calculation**: Each feature receives a score indicating its likelihood of having a match in the opposite image, which is used to weigh the pairwise scores.

### 6. Output - Partial Assignment Matrix

The final output is a soft partial assignment matrix, where each element represents the probability of a feature in image A matching with a feature in image B. High values in this matrix indicate likely matches, from which correspondences are selected.

### 7. Supervision and Training

- **Training**: The model is optimized using a loss function that combines the likelihoods of correct and incorrect matches, adjusting the model parameters to reduce mismatches.
- **Data**: Trained on a mix of synthetic data for robustness and real-world images for practical applicability.

This comprehensive structure allows LightGlue not just to match features but to do so in a way that is both computationally efficient and highly adaptive to the complexity of the matching scenario.
"""
triangulation="""
**How do we go from 2D to 3D?**

1.	An image is a projection of a 3D space onto a 2D plane.

2.	Each point in an image, hence, corresponds to a line in 3D space.

3.	All points on the line in 3D are projected to that singular point in the image.

4.	If a pair of corresponding points in two, or more images, can be found it must be the case that they are the projection of a common 3D point x.

5.	The set of lines generated by the image points must intersect at x.

**This process is called triangulation.**

"""
reference="""
1. P. Lindenberger, P. -E. Sarlin and M. Pollefeys, "LightGlue: Local Feature Matching at Light Speed," 2023 IEEE/CVF International Conference on Computer Vision (ICCV), Paris, France, 2023, pp. 17581-17592, doi: 10.1109/ICCV51070.2023.01616.
keywords: {Adaptation models;Visualization;Technological innovation;Codes;Three-dimensional displays;Computational modeling;Memory management}
(https://ieeexplore.ieee.org/document/10377620)

2. Oquab, Maxime & Darcet, Timothée & Moutakanni, Théo & Vo, Huy & Szafraniec, Marc & Khalidov, Vasil & Fernandez, Pierre & Haziza, Daniel & Massa, Francisco & El-Nouby, Alaaeldin & Assran, Mahmoud & Ballas, Nicolas & Galuba, Wojciech & Howes, Russell & Huang, Po-Yao & Li, Shang-Wen & Misra, Ishan & Rabbat, Michael & Sharma, Vasu & Bojanowski, Piotr. (2023). DINOv2: Learning Robust Visual Features without Supervision.
(https://www.researchgate.net/publication/370058767_DINOv2_Learning_Robust_Visual_Features_without_Supervision)

3. J. Dai et al., "Deformable Convolutional Networks," 2017 IEEE International Conference on Computer Vision (ICCV), Venice, Italy, 2017, pp. 764-773, doi: 10.1109/ICCV.2017.89. keywords: {Convolution;Kernel;Object detection;Standards;Feature extraction;Two dimensional displays},
 (https://ieeexplore.ieee.org/document/8237351)
 
4. Zhao, Xiaoming & Wu, Xingming & Chen, Weihai & Chen, Peter C. Y. & Xu, Qingsong & Li, Z.G.. (2023). ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation.
 (https://arxiv.org/abs/2304.03608)
 
5. (https://www.kaggle.com/code/asarvazyan/imc-understanding-the-baseline#Sparse-Reconstruction)
"""
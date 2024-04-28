# Setup

## Download Image Matching Dataset

**Install Kaggle**

```bash
pip install --user kaggle
```

**Download Kaggle API key**
**Create the dir using this command and move the key to the dir**

```bash
mkdir ~/.kaggle
```

**Move the key to the dir**

```bash
mv kaggle.json ~/.kaggle
```

**CLI command to download kaggle dataset**

```bash 
kaggle competitions download -c 'image-matching-challenge-2023'
```

**Unzip the dataset**

```bash
unzip image-matching-challenge-2023.zip
```

Refrences: [Kaggle Setup](https://www.endtoend.ai/tutorial/how-to-download-kaggle-datasets-on-ubuntu/)

## Install Models LightGlue, ALiKeD, and DINOv2

**Install LightGlue**

```bash
pip install git+https://github.com/cvg/LightGlue.git
```

**Create the dir for the checkpoints**

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
```

**Download the LightGlue model**

```bash
kaggle models instances versions download oldufo/lightglue/pyTorch/aliked/1
```

**Unzip the model**

```bash
tar -xzf lightglue.tar.gz
```

**Move the model to the checkpoints dir**

```bash
cp aliked_lightglue.pth ~/.cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv.pth
```

**Download the ALiKeD model**

```bash
kaggle models instances versions download oldufo/aliked/pyTorch/aliked-n16/1
```

**Unzip the model**

```bash
tar -xzf aliked.tar.gz
```

**Move the model to the checkpoints dir**

```bash
cp aliked-n16.pth ~/.cache/torch/hub/checkpoints/aliked-n16.pth
```

**Download the DINOv2 model**

```bash
kaggle models instances versions download metaresearch/dinov2/pyTorch/base/1
```

**Create the dir for the model**

```bash
mkdir -p dinov2/pytorch/base/1
````
``
**Unzip the model**

```bash
tar -xzf dinov2.tar.gz -C dinov2/pytorch/base/1
```

## Setup Script

Use get_data.sh to download the dataset

```bash
bash get_data.sh
```

Use setup.sh to install the models

```bash
bash setup.sh
```


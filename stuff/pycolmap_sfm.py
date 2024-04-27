from pathlib import Path
import os
import pandas as pd
from torchvision import transforms
from data_loader import ImageDataset
import cv2
from glob import glob
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d
from torch.utils.data import DataLoader
# git clone --recursive https://github.com/cvg/Hierarchical-Localization/
# cd Hierarchical-Localization/
# python3 -m pip install -e .
# !pip install pycolmap
import pycolmap
import plotly.express as px


main_path = Path(os.getcwd())
train_path = main_path / 'train'
train_labels_path = train_path / 'train_labels.csv'
train_labels = pd.read_csv(train_labels_path)
train_labels.head(4)

df_train_labels = train_labels
# df_train_labels = train_labels.sample(frac=0.7, random_state=33)
# df_test_labels = train_labels.drop(df_train_labels.index)

image_transform = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor()])

train_dataset = ImageDataset(df_train_labels, image_transform)
# test_dataset = ImageDataset(df_test_labels, image_transform)

# train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
# test_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)
dataset = 'phototourism'
scene = 'lincoln_memorial_statue'
# print(os.getcwd())
src = f'{os.getcwd()}/train/{dataset}/{scene}'

images = [cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB) for im in glob(f'{src}/images/*')]

rec_gt = pycolmap.Reconstruction(f'{src}/sfm')

fig = viz_3d.init_figure()
viz_3d.plot_cameras(fig, rec_gt, color='rgba(50,255,50, 0.5)', name="Ground Truth", size=10)
viz_3d.plot_reconstruction(fig, rec_gt, cameras = False, color='rgba(255,50,255, 0.5)', name="Ground Truth", cs=5)
fig.show()

fig.write_html('file.html')
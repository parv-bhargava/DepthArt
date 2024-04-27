import pandas as pd
import numpy as np
from pathlib import Path
# import wandb
import os
from tqdm.autonotebook import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


main_path=Path(os.getcwd())
train_path= main_path / 'train'
train_labels_path= train_path / 'train_labels.csv'
train_labels=pd.read_csv(train_labels_path)
train_labels.head(4)

df_train_labels=train_labels.sample(frac=0.7, random_state=33)
df_test_labels=train_labels.drop(df_train_labels.index)

image_transform=transforms.Compose([transforms.Resize(size=(224,224)), transforms.ToTensor()])

class ImageDataset(Dataset):
    def __init__(self,
                 df_labels: pd.DataFrame,
                 image_transform: transforms = None
                 ):
        super().__init__()
        self.image_paths = df_labels['image_path'].tolist()
        self.rotation_matrices = df_labels['rotation_matrix'].tolist()
        self.translation_vectors = df_labels['translation_vector'].tolist()
        self.image_transform = image_transform

    def __getitem__(self, index):
        actual_image_path = 'train/' + self.image_paths[index]
        image = Image.open(actual_image_path)

        if self.image_transform is not None:
            image = self.image_transform(image)

        rotation_matrix = self.rotation_matrices[index].split(';')
        rotation_matrix = np.array(rotation_matrix, dtype=float)
        rotation_matrix = torch.from_numpy(rotation_matrix).type(torch.float32)

        translation_vector = self.translation_vectors[index].split(';')
        translation_vector = np.array(translation_vector, dtype=float)
        translation_vector = torch.from_numpy(translation_vector).type(torch.float32)

        return image, rotation_matrix, translation_vector

    def __len__(self):
        return len(self.image_paths)

train_dataset = ImageDataset(df_train_labels, image_transform)
test_dataset = ImageDataset(df_test_labels, image_transform)

train_dataloader=DataLoader(dataset=train_dataset, batch_size=8, shuffle= True)
test_dataloader=DataLoader(dataset=test_dataset, batch_size=8, shuffle= False)

sample_img,sample_rotmat,sample_transvect =next(iter(train_dataloader))
print(sample_img.shape)
print(sample_rotmat)
print(sample_transvect)

# # Code to use the train_dataset(i.e. trainng images
# sample_img,sample_rotmat,sample_transvect=train_dataset[0]
# print(sample_img.shape)
# print(sample_rotmat)
# print(sample_transvect)
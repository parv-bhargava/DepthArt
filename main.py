from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import ImageDataset
import pandas as pd
from pathlib import Path
import os

main_path=Path(os.getcwd())
train_path= main_path / 'train'
train_labels_path= train_path / 'train_labels.csv'
train_labels=pd.read_csv(train_labels_path)
train_labels.head(4)

df_train_labels=train_labels.sample(frac=0.9, random_state=42)
df_val_labels=train_labels.drop(df_train_labels.index)

image_transform=transforms.Compose([transforms.Resize(size=(224,224)), transforms.ToTensor()])
train_dataset = ImageDataset(df_train_labels, image_transform)
val_dataset = ImageDataset(df_val_labels, image_transform)

train_dataloader=DataLoader(dataset=train_dataset, batch_size=8, shuffle= True)
val_dataloader=DataLoader(dataset=val_dataset, batch_size=8, shuffle= False)


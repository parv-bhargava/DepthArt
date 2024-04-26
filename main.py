from pathlib import Path
import os
import pandas as pd
from torchvision import transforms
from data_loader import ImageDataset
from torch.utils.data import DataLoader

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


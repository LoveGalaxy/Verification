import os
import torch
import pandas as pd
import random

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class PlayDataset(Dataset):
    char_table = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self, is_train=True, train_val=0.9, transform=None):
        self.label_file = "./data/train/train_label.csv"
        self.data = pd.read_csv(self.label_file)
        self.image_dir = "./data/train/"
        self.is_train = is_train
        self.train_val = train_val
        self.transform = transform
        self.char_to_id = {ch:i+1 for i, ch in enumerate(self.char_table)}
        self.id_to_char = {i+1:ch for i, ch in enumerate(self.char_table)}

    def __len__(self):
        if self.is_train:
            return int(len(self.data) * self.train_val)
        else:
            return len(self.data) - int(len(self.data) * self.train_val)
    
    def __getitem__(self, idx):
        if not self.is_train:
            idx += int(len(self.data) * self.train_val)
        file_path = self.image_dir + self.data.loc[idx, 'ID']
        label = np.zeros((4))

        for idx, ch in enumerate(self.data.loc[idx, 'label']):
            label[idx] = self.char_to_id[ch]

        image = Image.open(file_path)

        if self.transform:
            image = self.transform(image)

        sample = {
            'image': image,
            'label': label,
        }

        return sample

    
if __name__ == "__main__":
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = PlayDataset(is_train=True, train_val=1, transform=transform_test)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=8)
    for i, sample_batch in enumerate(dataloader):
        print(i, sample_batch['image'].shape)
        print(i, sample_batch['label'].shape)
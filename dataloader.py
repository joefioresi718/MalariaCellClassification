from __future__ import print_function, division
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch


class CellClassificationDataset(Dataset):
    def __init__(self, transform, set):
        self.set = set
        self.transform = transform
        if set == 'train':
            self.root_dir = 'data/cells/'
            self.image_frame = pd.read_csv('files/train.csv')
        elif set == 'val':
            self.root_dir = 'data/cells/'
            self.image_frame = pd.read_csv('files/val.csv')
        else:
            self.root_dir = 'data/cells/'
            train_list = pd.read_csv('files/train.csv')
            val_list = pd.read_csv('files/val.csv')
            self.image_frame = pd.concat([train_list, val_list], ignore_index=True)

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        image_p = os.path.join(self.root_dir, self.image_frame['image_name'][idx])
        label = torch.tensor(self.image_frame['cell_type'][idx])
        image = Image.open(image_p)
        image = image.convert('RGB')

        image = self.transform(image)

        return image, label

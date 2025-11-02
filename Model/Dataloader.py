import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class ASLImageDataset(Dataset):
    """ASL Image Dataset"""

    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.data_frame.iloc[idx, 0]
        image = self.data_frame.iloc[idx, 1:]
        image = np.array(image).reshape(28, 28)
        sample = {'label': label, 'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        label, image = sample['label'], sample['image']

        return {'label': label,
                'image': torch.from_numpy(image).unsqueeze(dim=0)}

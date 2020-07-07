import os
from os.path import join
import torch
import numpy as np
from skimage import io
from torch.utils.data import Dataset, DataLoader

class ImageDataSets(Dataset):
    def __init__(self, root_dir):
      self.root_dir = root_dir
      self.fileList = sorted([join(root,f) for root, dirs, files in os.walk(root_dir) for f in files if f.endswith('.jpg')])

    def __len__(self):
        return len(self.fileList)

    def __getitem__(self, idx):
        imagepath = self.fileList[idx]
        return io.imread(imagepath)



    
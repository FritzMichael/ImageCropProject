import os
from os.path import join
import torch
import numpy as np
from skimage import io
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

class ImageDataSet(Dataset):
    def __init__(self, root_dir):
      self.root_dir = root_dir
      self.fileList = sorted([join(root,f) for root, dirs, files in os.walk(root_dir) for f in files if f.endswith('.jpg')])

    def __len__(self):
        return len(self.fileList)

    def __getitem__(self, idx):
        imagepath = self.fileList[idx]
        return io.imread(imagepath)

class CroppedOutImageDataSet(Dataset):
    def __init__(self, Dataset, maxCropPercentage: float = 30.):
      self.dataset = Dataset
      self.maxCropPercentage = maxCropPercentage

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, id):
        image_data = np.squeeze(self.dataset.__getitem__(id))
        height, width = image_data.shape

        image_data = np.asarray(image_data, dtype=np.float32)
        mean = image_data.mean()
        std = image_data.std()
        image_data[:] -= mean
        image_data[:] /= std

        # Setting some random crop_center
        top, bottom = np.sort([int(np.random.uniform(height*0.2, height*0.8)), int(np.random.uniform(height*0.2, height*0.8))])
        left, right = np.sort([int(np.random.uniform(width*0.2, width*0.8)), int(np.random.uniform(width*0.2, width*0.8))])

        cropped_image, crop_array, target_array = cropImage(image_data, top, bottom, left, right)

        # constructing the image we want to input to the model
        inputs = np.zeros(shape = (*cropped_image.shape, 4), dtype=cropped_image.dtype)
        inputs[..., 0] = cropped_image
        inputs[..., 1] = crop_array
        inputs[..., 2] = np.tile(np.linspace(start=-1, stop=1, num=width),reps=[height,1])
        inputs[..., 3] = np.tile(np.expand_dims(np.linspace(start=-1, stop=1, num=height),1),reps=[1,width])

        return TF.to_tensor(inputs), TF.to_tensor(target_array), id



def cropImage(image_array, top, bottom, left, right):
    cropped_image = np.copy(image_array)
    crop_array = np.zeros(image_array.shape)

    if ((left < 20 or top < 20) or ((-right + image_array.shape[1] -1) < 20 or
        (-bottom + image_array.shape[0] -1 ) < 20)):
        raise ValueError('Too close to border')

    target_array = image_array[top:bottom+1,left:right+1]
    crop_array[top:bottom+1,left:right+1] = 1
    cropped_image[top:bottom+1,left:right+1] = 0
    
    return (cropped_image, crop_array.astype(image_array.dtype), target_array)



    
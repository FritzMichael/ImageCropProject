import os
from os.path import join
import torch
import numpy as np
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import PIL
from PIL import Image

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(70, PIL.Image.LANCZOS),
    transforms.ToTensor()
])


class ImageDataSet(Dataset):
    def __init__(self, root_dir):
      self.root_dir = root_dir
      self.fileList = sorted([join(root,f) for root, dirs, files in os.walk(root_dir) for f in files if f.endswith('.jpg')])

    def __len__(self):
        return len(self.fileList)

    def __getitem__(self, idx):
        imagepath = self.fileList[idx]
        image = io.imread(imagepath)
        image = transform(image)
        return image 

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

        # setting some random crops
        top = int(np.random.uniform(20, height-40))
        bottom = top + int(np.random.uniform(5,21))
        left = int(np.random.uniform(20, width-40))
        right = left + int(np.random.uniform(5, 21))

        #top, bottom = np.sort([int(np.random.uniform(height*0.2, height*0.8)), int(np.random.uniform(height*0.2, height*0.8))])
        #left, right = np.sort([int(np.random.uniform(width*0.2, width*0.8)), int(np.random.uniform(width*0.2, width*0.8))])

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

    #if ((left < 20 or top < 20) or ((-right + image_array.shape[1] -1) < 20 or
    #    (-bottom + image_array.shape[0] -1 ) < 20)):
    #    raise ValueError('Too close to border')

    target_array = image_array[top:bottom+1,left:right+1]
    crop_array[top:bottom+1,left:right+1] = 1
    cropped_image[top:bottom+1,left:right+1] = 0
    
    return (cropped_image, crop_array.astype(image_array.dtype), target_array)

def collate_Images(batch):
    #maxHeight = np.max([sample[0].size()[1] for sample in batch])
    #maxWidth = np.max([sample[0].size()[2] for sample in batch])
    #inputs = torch.zeros((len(batch), 4,maxHeight, maxWidth))
    #for singleInput,sample in zip(inputs,batch):
    #    for layer, clayer in zip(sample[0],singleInput):
    #        clayer[0:layer.size()[0],0:layer.size()[1]] = layer


    inputs = [sample[0] for sample in batch]
    targets = [sample[1] for sample in batch]
    ids = [sample[2] for sample in batch]
    return [inputs, targets, ids]

    
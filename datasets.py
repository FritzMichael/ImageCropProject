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
import dill as pkl

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(70, PIL.Image.LANCZOS),
    transforms.ToTensor()
])

scaledtransform = transforms.Compose([
    transforms.ToTensor()
])


class scaledImageDataSet(Dataset):
    def __init__(self, pkl_file):
        with open(pkl_file,'rb') as tsp:
            self.imagefile = pkl.load(tsp)
        print('... images loaded')
        print(f'memory used by images: {self.imagefile.nbytes}')

    def __len__(self):
        return len(self.imagefile)

    def __getitem__(self, idx):
        return torch.tensor(self.imagefile[idx])

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

        image_data = image_data.to(dtype=torch.float32)
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

        cropdict = {'top': top, 'bottom': bottom, 'left': left, 'right': right}

        return cropped_image, cropdict, target_array, id

def cropImage(image_array, top, bottom, left, right):
    cropped_image = image_array.detach().clone()
    crop_array = torch.zeros_like(image_array)

    #if ((left < 20 or top < 20) or ((-right + image_array.shape[1] -1) < 20 or
    #    (-bottom + image_array.shape[0] -1 ) < 20)):
    #    raise ValueError('Too close to border')

    target_array = image_array[top:bottom+1,left:right+1]
    crop_array[top:bottom+1,left:right+1] = 1
    cropped_image[top:bottom+1,left:right+1] = 0
    
    return (cropped_image, crop_array, target_array)

def collate_Images(batch):
    #collating inputs
    maxHeight = np.max([sample[0].size()[0] for sample in batch])
    maxWidth = np.max([sample[0].size()[1] for sample in batch])
    inputs = torch.zeros((len(batch), 1,maxHeight, maxWidth))
    for singleInput,sample in zip(inputs,batch):
        singleInput[0,0:sample[0].size()[0],0:sample[0].size()[1]] = sample[0]

        #for layer, clayer in zip(sample[0],singleInput):
        #    clayer[0:layer.size()[0],0:layer.size()[1]] = layer


    #inputs = [sample[0] for sample in batch]

    # collating targets
    #maxHeight = np.max([sample[2].size()[0] for sample in batch])
    #maxWidth = np.max([sample[2].size()[1] for sample in batch])
    targets = torch.zeros((len(batch), 1, maxHeight, maxWidth))

    for tlayer, sample in zip(targets,batch):
        target = sample[2]
        top = sample[1]['top']
        bottom = sample[1]['bottom']
        left = sample[1]['left']
        right = sample[1]['right']
        tlayer[0,top:bottom+1,left:right+1] = target


    #targets = [sample[1] for sample in batch]
    ids = [sample[3] for sample in batch]
    crop_dicts = [sample[1] for sample in batch]
    return [inputs, crop_dicts, targets, ids]

    
import os
import numpy as np
from PIL import Image
import timeit


def rescale(root_dir, target_dir):
    fileList = sorted([os.path.join(root,f) for root, dirs, files in os.walk(root_dir) for f in files if f.endswith('.jpg')])

    for i,file in enumerate(fileList):
        img = Image.open(file, mode='r')
        width, height = (img.width, img.height)
        factor = np.min([width, height])/100
        newdims = (int(width/factor),int(height/factor))
        img = img.resize(newdims, resample=Image.LANCZOS)
        img.save(os.path.join(target_dir, f'{i}.jpg'))

if __name__ == "__main__":
    rescale('data', 'downscaled_data')
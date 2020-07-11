import os
import numpy as np
from PIL import Image
import timeit
import dill as pkl


def rescale(root_dir, targetfile):
    fileList = sorted([os.path.join(root,f) for root, dirs, files in os.walk(root_dir) for f in files if f.endswith('.jpg')])

    imgList = []
    for i,file in enumerate(fileList):
        img = Image.open(file, mode='r')
        width, height = (img.width, img.height)
        factor = np.min([width, height])/100
        newdims = (int(width/factor),int(height/factor))
        img = img.resize(newdims, resample=Image.LANCZOS)
        imgList.append(np.array(img))

        if i%10 == 0:
            print(i)
    pkl_out = open(targetfile,'wb')
    pkl.dump(imgList, pkl_out)
    pkl_out.close()

if __name__ == "__main__":
    rescale('data', 'downscaled_data.pkl')
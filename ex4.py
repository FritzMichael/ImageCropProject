# ex_4.py
# Author -- Michael Fritzenwallner
# Matrikelnumber -- k11702640
# Date -- 14.05.2020
###############################################################################

import numpy as np

def ex4(image_array, crop_size, crop_center):
    if (not isinstance(image_array,np.ndarray) or 
    len(image_array.shape) != 2  or 
    len(crop_size) != 2 or 
    len(crop_center) != 2 or 
    len([x for x in crop_size if x%2 == 0]) != 0):
        raise ValueError

    cropped_image = np.copy(image_array)
    crop_array = np.zeros(image_array.shape)

    top = crop_center[0]-(np.asarray(crop_size,dtype = np.int)[0]/2).astype(int)
    bottom = crop_center[0]+(np.asarray(crop_size,dtype = np.int)[0]/2).astype(int)
    left = crop_center[1]-(np.asarray(crop_size,dtype = np.int)[1]/2).astype(int)
    right = crop_center[1]+(np.asarray(crop_size,dtype = np.int)[1]/2).astype(int)

    if ((left < 20 or top < 20) or ((-right + image_array.shape[1] -1) < 20 or
        (-bottom + image_array.shape[0] -1 ) < 20)):
        raise ValueError('Too close to border')

    target_array = image_array[top:bottom+1,left:right+1]
    crop_array[top:bottom+1,left:right+1] = 1
    cropped_image[top:bottom+1,left:right+1] = 0
    
    return (cropped_image, crop_array.astype(image_array.dtype), target_array)
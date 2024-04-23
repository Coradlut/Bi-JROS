import numpy as np
import nibabel as nib
import os
import glob
import cv2


labels = [0, 1, 11, 22, 33, 44, 46, 47, 48, 49, 50, 51, 52, 55] 


good_label = [0, 1, 11, 22, 33, 44, 46, 47, 48, 49, 50, 51, 52, 55] 


extra_label = list(set(labels).difference(set(good_label)))
index = list(range(len(good_label)))


def convert(seg):
    output = np.copy(seg)
    for i in extra_label:
        output[seg == i] = 0
    for k, v in zip(good_label, index):
        output[seg == k ] = v
    return output

def inverse_convert(seg):
    output = np.copy(seg)
    for k, v in zip(index, good_label):
        output[seg == k] = v
    return output
    

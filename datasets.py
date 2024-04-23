import os, glob
import torch, sys
from torch.utils.data import Dataset
import numpy as np
from scipy import ndimage
from convert_labels import convert
import nibabel as nib

def n_score_normalize(data):
    """
    Normalize as n-score and minus minimum
    """
    mean = data.mean()
    std = np.std(data)
    data = (data - mean) / std
    return data - data.min()

def normalize_image_intensity(image):
    """
    对输入图像进行强度归一化处理。
    
    :param image: 待处理的图像数组。
    :return: 强度归一化后的图像数组。
    """
    # 将图像数据转换为浮点数，以便进行除法运算
    image = image.astype(np.float32)
    
    # 计算图像的最小和最大强度值
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    
    # 执行归一化：(image - min) / (max - min)
    # 结果中的所有值都将位于[0, 1]区间
    normalized_image = (image - min_intensity) / (max_intensity - min_intensity)
    
    return normalized_image

def load_volfile(datafile):
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % datafile

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
        if 'nib' not in sys.modules:
            try:
                import nibabel as nib
            except:
                print('Failed to import nibabel. need nibabel library for these data file types.')

        X = nib.load(datafile).get_fdata()

    else:  # npz
        X = np.load(datafile)['vol_data']

    return X

class TrainOneShot(Dataset):
    def __init__(self, data_path):
        self.train_vol_names = data_path

    def __getitem__(self, index):
        src_path, tgt_path, sseg_path = self.train_vol_names[index]

        tgt, src = load_volfile(tgt_path), load_volfile(src_path)

        tgt, src = normalize_image_intensity(tgt), normalize_image_intensity(src)

        src_seg = load_volfile(sseg_path)  
        src_seg = convert(src_seg)

        tgt = ndimage.zoom(tgt, (128 / 160, 128 / 192, 128 / 224), order=1)
        tgt = torch.Tensor(tgt).float()
        src = ndimage.zoom(src, (128 / 160, 128 / 192, 128 / 224), order=1)
        src = torch.Tensor(src).float()
        tgt, src = tgt.unsqueeze(0), src.unsqueeze(0)

        src_seg = ndimage.zoom(src_seg, (128 / 160, 128 / 192, 128 / 224), order=0)
        src_seg = torch.Tensor(src_seg).float()
        src_seg = src_seg.unsqueeze(0)

        return src, tgt, src_seg

    def __len__(self):
        return len(self.train_vol_names)

class InferDataset(Dataset):
    def __init__(self, data_path):
        self.paths = data_path

    def __getitem__(self, index):
        src_path, tgt_path, sseg_path, tseg_path = self.paths[index]

        tgt, src = load_volfile(tgt_path), load_volfile(src_path)
        tgt, src = normalize_image_intensity(tgt), normalize_image_intensity(src)

        tgt_seg, src_seg = load_volfile(tseg_path), load_volfile(sseg_path)  
        tgt_seg = convert(tgt_seg)
        src_seg = convert(src_seg)

        tgt = ndimage.zoom(tgt, (128 / 160, 128 / 192, 128 / 224), order=1)
        src = ndimage.zoom(src, (128 / 160, 128 / 192, 128 / 224), order=1)
        tgt = torch.Tensor(tgt).float()
        src = torch.Tensor(src).float()
        tgt, src = tgt.unsqueeze(0), src.unsqueeze(0)

        tgt_seg = ndimage.zoom(tgt_seg, (128 / 160, 128 / 192, 128 / 224), order=0)
        src_seg = ndimage.zoom(src_seg, (128 / 160, 128 / 192, 128 / 224), order=0)
        tgt_seg = torch.Tensor(tgt_seg).float()
        src_seg = torch.Tensor(src_seg).float()
        tgt_seg, src_seg = tgt_seg.unsqueeze(0), src_seg.unsqueeze(0)

        return src, tgt, src_seg, tgt_seg, tgt_path

    def __len__(self):
        return len(self.paths)
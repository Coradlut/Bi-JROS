import sys
import torch.nn.functional as F
import torch.nn as nn
from layers import SpatialTransformer, ResizeTransform 
import numpy as np
import torch
from torch import Tensor
import math
import nibabel as nib
from scipy import ndimage

class LossFunction_dice(nn.Module):
    def __init__(self):
        super(LossFunction_dice, self).__init__()
        # GT在前
        self.dice_loss = Dice()
        self.spatial_transform = SpatialTransformer()

    def forward(self, mask_0, mask_1, flow):
        mask_1 = F.one_hot(mask_1.squeeze(1).to(torch.int64), num_classes=14).permute(0, 4, 1, 2, 3).float()
        mask = self.spatial_transform(mask_1, flow)
        loss = self.dice_loss(mask, mask_0)
        return loss

class gradient_loss(nn.Module):
    def __init__(self):
        super(gradient_loss, self).__init__()

    def forward(self, s, penalty='l2'):
        dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
        dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
        dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])
        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz
        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        return d / 3.0

class Dice(nn.Module):
    """
    N-D dice for segmentation
    """
    def __init__(self):
        super(Dice, self).__init__()

    def forward(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


    
def nas_ncc(I, J):
    
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    win = [9] * ndims
    sum_filt = torch.ones([1, 1, *win]).to("cuda")
    pad_no = math.floor(win[0] / 2)
    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    cc = cross * cross / (I_var * J_var + 1e-5)
    return -1 * torch.mean(cc)
    # return -1 * torch.mean(cc) + 1



def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross

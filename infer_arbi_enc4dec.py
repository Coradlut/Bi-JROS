import glob
import os, losses 
import utils.utils as utils
from torch.utils.data import DataLoader
import datasets
import numpy as np
import torch
from models import *
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from layers import VecInt
import torch.nn.functional as F
import imageio
import time
import nibabel as nib
import shutil


def main(data_dir, model_dir):
    model = Architecture(inshape=(128, 128, 128), n_channels=1, n_classes=14)
    model.load_state_dict(torch.load(model_dir, map_location={'cuda:1':'cuda:0'}))
    model.cuda()
    reg_model = utils.register_model('nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model('bilinear')
    reg_model_bilin.cuda()
    for param in reg_model.parameters():
        param.requires_grad = False
        param.volatile = True

    test_path_table = np.load(data_dir)['data']
    test_set = datasets.InferDataset(test_path_table)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    dice_list0, dice_list1, dice_list2, ncc_list0, ncc_list1 = [], [], [], [], []
    hd95_list, assd_list = [], []
    reg_list, seg_list = [], []
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()

            x = data[0].cuda()
            y = data[1].cuda()
            y = torch.tensor(y, dtype=torch.float32)
            x_seg = data[2].cuda()
            y_seg = data[3].cuda()
            path = data[4][0]

            x_def, flow, x_logits, y_logits = model(x, y)

            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            
            full_mask = F.softmax(y_logits, dim=1).float()
            mask = full_mask.argmax(dim=1)

            vals0, _ = utils.dice(x_seg.cpu().numpy().squeeze(), y_seg.cpu().numpy().squeeze(), nargout=2)
            dice_list0.append(np.average(vals0))
            vals1, _ = utils.dice(def_out.cpu().numpy().squeeze(), y_seg.cpu().numpy().squeeze(), nargout=2)
            dice_list1.append(np.average(vals1))
            vals2, _ = utils.dice(mask.cpu().numpy().squeeze(), y_seg.cpu().numpy().squeeze(), nargout=2)
            dice_list2.append(np.average(vals2))


            ncc0 = -losses.nas_ncc(y, x)
            ncc1 = -losses.nas_ncc(y, x_def)
            ncc_list0.append(ncc0.item())
            ncc_list1.append(ncc1.item())

        print(model_dir)
        print('------------------------Dice------------------------')
        print(' initial :', np.mean(dice_list0), np.std(dice_list0))
        print('   Reg   :', np.mean(dice_list1), np.std(dice_list1))
        print('   Seg   :', np.mean(dice_list2), np.std(dice_list2))

        print('------------------------NCC------------------------')
        print(' initial :', np.mean(ncc_list0), np.std(ncc_list0))
        print('  warped :', np.mean(ncc_list1), np.std(ncc_list1))


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main('/camera_ready/Data/test.npz', '/camera_ready/experiment/test.ckpt')
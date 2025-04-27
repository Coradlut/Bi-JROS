import os
import sys
import glob
import random
import logging
import argparse
import datasets
import losses
import torch
import torch.utils.data as da
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from models import *
import utils.utils as utils
import torch.nn.functional as F
import utils.augmentation as aug
import numpy as np

parser = argparse.ArgumentParser("bilevel")

parser.add_argument('--data', type=str, default='/camera_ready/train.npz', help='location of the data corpus')

# The selectable parameters are "sam", "synthseg", "bi-jros", and "RRL SAM".
parser.add_argument('--enc', type=str, default='bi-jros',required = True, help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--reg_lr', type=float, default=1e-4, help='init learning rate') # 4e-4
parser.add_argument('--seg_lr', type=float, default=1e-4, help='init learning rate')

parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=1500, help='num of training epochs')  

# Set an appropriate range for random weight generation.
parser.add_argument('--seg_start', type=float, default=0.6, help='random weight') 
parser.add_argument('--seg_end', type=float, default=0.95, help='random weight')
parser.add_argument('--reg_start', type=float, default=0.6, help='random weight')
parser.add_argument('--reg_end', type=float, default=0.95, help='random weight')

parser.add_argument('--feature', type=str, default='/home/jiawang/data1/Bi-JROS/weights/encoder.ckpt', 
                        help='path to load the pre-trained encoder')
parser.add_argument('--keep_training', type=str, default=None, 
                        help='continue training')
parser.add_argument('--save', type=str, default='/camera_ready/experiment', help='experiment name')
args = parser.parse_args()

# prepare model folder
model_dir = args.save
os.makedirs(model_dir, exist_ok=True)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    '''
    Initialize model
    '''
    img_size = (128, 128, 128)
    if args.enc == 'bi-jros' or args.enc == 'RRL-SAM':
        model = Architecture(inshape=img_size, n_channels=1, n_classes=14)
        model = model.cuda()

        '''
        Load Feature Extractor 
        '''
        if args.feature is not None:
            pretrained_dict = torch.load(args.feature)
            model_dict = model.state_dict()
            pretrained_dict_updated = {key: value for key, value in pretrained_dict.items() if (key in model_dict and 'fea_extractor' in key)}
            model_dict.update(pretrained_dict_updated)
            model.load_state_dict(model_dict)
    elif args.enc == 'sam':
        model = 
    # elif args.enc == 'synthseg'
    #     model = 

    '''
    Load Model Weight
    '''
    if args.keep_training is not None:
        model.load_state_dict(torch.load(args.keep_training))

    '''
    Data Augmentation
    '''
    spatial_aug = aug.SpatialTransform(do_rotation=True,
                                        angle_x=(-np.pi / 18, np.pi / 18),
                                        angle_y=(-np.pi / 18, np.pi / 18),
                                        angle_z=(-np.pi / 18, np.pi / 18),
                                        do_scale=True,
                                        scale=(0.9, 1.1))

    '''
    Data Generator
    '''
    train_path_table = np.load(args.data)['data']
    train_set = datasets.TrainOneShot(train_path_table)
    train_queue = da.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    '''
    Spatial Transformation
    '''
    transform1 = utils.register_model('nearest')
    transform1.cuda()
    transform2 = utils.register_model('bilinear')
    transform2.cuda()

    '''
    Optimization
    '''
    r_optimizer = Adam(model.Reg_Decoder.parameters(), lr=args.reg_lr)
    s_optimizer = Adam(model.Seg_Decoder.parameters(), lr=args.seg_lr)

    '''
    Loss
    '''   
    grad_loss = losses.gradient_loss()
    cycle_reg = torch.nn.L1Loss()
    s_criterion_dice1 = losses.Dice().cuda()
    s_criterion_dice2 = losses.LossFunction_dice().cuda()
    criterions = [grad_loss, cycle_reg, s_criterion_dice1, s_criterion_dice2]

    for epoch in range(args.epochs):
        # training
        train(train_queue, model, args, r_optimizer, s_optimizer, criterions, transform1, transform2, spatial_aug, epoch)
        save_file_name = os.path.join(args.save, '%d.ckpt' % epoch)
        torch.save(model.state_dict(), save_file_name)


def train(train_queue, model, args, r_optimizer, s_optimizer, criterions, transform1, transform2, spatial_aug, epoch):
    for step, (x, y, x_seg) in enumerate(train_queue):
        x = Variable(x, requires_grad=False).cuda()
        y = Variable(y, requires_grad=False).cuda()
        x_seg = Variable(x_seg, requires_grad=False).cuda()

        '''
        Training Seg_Decoder
        '''
        model.Reg_Decoder.eval()
        model.Seg_Decoder.train()
        encoded_x = model.fea_extractor(x)
        encoded_y = model.fea_extractor(y)
        x_y, flow_x_y = model.Reg_Decoder(x, encoded_x, encoded_y)
        y_x, flow_y_x = model.Reg_Decoder(y, encoded_y, encoded_x)


        def_out = transform1([y_x, flow_x_y.cuda()])
        weight = random.uniform(args.seg_start, args.seg_end)
        y_feature = weight*(def_out - y)      
        refine_reverse_fake = def_out - y_feature 
        new_lab = transform2([F.one_hot(x_seg.squeeze(1).to(torch.int64)).permute(0, 4, 1, 2, 3).float(), flow_x_y])         

        s_gen = model.Seg_Decoder(model.fea_extractor(refine_reverse_fake))

        # Reg Seg loss
        dice_RegSeg = criterions[2](F.softmax(s_gen, dim=1).float(), new_lab)

        # compute gradient and do SGD step
        s_optimizer.zero_grad()
        dice_RegSeg.backward()
        s_optimizer.step()
        if step % args.report_freq == 0:
            logging.info('train seg epoch %03d step %03d %f', epoch, step, dice_RegSeg)

        '''
        Training Reg_Decoder
        '''
        model.Seg_Decoder.eval()
        model.Reg_Decoder.train()
        r_optimizer.zero_grad()

        code_spa = spatial_aug.rand_coords(x.shape[2:])
        x = spatial_aug.augment_spatial(x, code_spa)
        x_seg = spatial_aug.augment_spatial(x_seg, code_spa, mode='nearest')
        y = spatial_aug.augment_spatial(y, code_spa)


        encoded_x = model.fea_extractor(x)
        encoded_y = model.fea_extractor(y)
        x_y, flow_x_y = model.Reg_Decoder(x, encoded_x, encoded_y)
        y_x, flow_y_x = model.Reg_Decoder(y, encoded_y, encoded_x)

        xlogits = model.Seg_Decoder(encoded_x)
        ylogits = model.Seg_Decoder(encoded_y)

        # Reg loss
        reg_x_y, grad_x_y = losses.nas_ncc(y, x_y), criterions[0](flow_x_y)
        reg_y_x, grad_y_x = losses.nas_ncc(x, y_x), criterions[0](flow_y_x)

        reg = reg_x_y + reg_y_x
        grad = grad_x_y + grad_y_x

        #cycle loss
        fake_x = transform2([x_y, flow_y_x])
        fake_y = transform2([y_x, flow_x_y])

        cycle_x = criterions[1](x, fake_x)
        cycle_y = criterions[1](y, fake_y)

        cycle = cycle_x + cycle_y

        # Reg Seg loss
        dice_RegSeg = criterions[3](F.softmax(ylogits, dim=1).float(), x_seg, flow_x_y)

        # Upper Loss
        Reg_loss = reg + grad + 10*cycle + dice_RegSeg

        def_out = transform1([y_x, flow_x_y.cuda()])
        weight = random.uniform(args.reg_start, args.reg_end)
        y_feature = weight*(def_out - y)      
        refine_reverse_fake = def_out - y_feature 
        new_lab = transform2([F.one_hot(x_seg.squeeze(1).to(torch.int64)).permute(0, 4, 1, 2, 3).float(), flow_x_y])         

        s_gen = model.Seg_Decoder(model.fea_extractor(refine_reverse_fake))

        # Lower loss
        Reg_Seg_loss = criterions[2](F.softmax(s_gen, dim=1).float(), new_lab)

        Grad_Reg = torch.autograd.grad(Reg_loss, model.Seg_Decoder.parameters(), retain_graph=True)
        Grad_Seg = torch.autograd.grad(Reg_Seg_loss, model.Seg_Decoder.parameters(), retain_graph=True)
        gReSe = 0
        gSeSe = 0
        for rs, ss in zip(Grad_Reg, Grad_Seg):
            gReSe = gReSe + torch.sum(rs * ss)
            gSeSe = gSeSe + torch.sum(ss * ss)
        GRe_loss = - gReSe.detach() / gSeSe.detach() * Reg_Seg_loss

        reg_grad = torch.autograd.grad(Reg_loss + GRe_loss, model.Reg_Decoder.parameters())
        for p, t in zip(reg_grad, model.Reg_Decoder.parameters()):
            if t.grad is not None:
                t.grad += p
            else:
                t.grad = p
        r_optimizer.step()

        if step % args.report_freq == 0:
            logging.info('train reg epoch %03d step %03d  %f, %f', epoch, step, Reg_loss, GRe_loss)


if __name__ == '__main__':
    main()
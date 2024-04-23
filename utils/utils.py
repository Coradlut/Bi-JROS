# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Email:  alisonzli@tencent.com or alisonbrielee@gmail.com
""" Utils """
# **********************************

import os
import cv2
import numpy as np
import torch
import shutil
import pandas as pd
import logging
import SimpleITK as sitk
from scipy.spatial import distance
import torch.nn as nn
from layers import SpatialTransformer, Re_SpatialTransformer

LANDMARK_COORDS = ['X', 'Y']

"""  *********************** Registration Related *************************** """

class register_model(nn.Module):
    def __init__(self, mode='bilinear'):
        super(register_model, self).__init__()
        self.spatial_trans = SpatialTransformer(mode)

    def forward(self, x):
        img = x[0].cuda()
        flow = x[1].cuda()
        out = self.spatial_trans(img, flow)
        return out

class re_register_model(nn.Module):
    def __init__(self, mode='bilinear'):
        super(re_register_model, self).__init__()
        self.spatial_trans = Re_SpatialTransformer(mode)

    def forward(self, x):
        img = x[0].cuda()
        flow = x[1].cuda()
        out = self.spatial_trans(img, flow)
        return out
        
"""  *********************** Landmark Related *************************** """

def compute_tre(points_1, points_2):
    """ computing Target Registration Error for each landmark pair

    :param ndarray points_1: set of points
    :param ndarray points_2: set of points
    :return ndarray: list of errors of size min nb of points
    array([ 0.21...,  0.70...,  0.44...,  0.34...,  0.41...,  0.41...])
    """
    nb_common = min([len(pts) for pts in [points_1, points_2]
                     if pts is not None])
    assert nb_common > 0, 'no common landmarks for metric'
    points_1 = np.asarray(points_1)[:nb_common]
    points_2 = np.asarray(points_2)[:nb_common]
    diffs = np.sqrt(np.sum(np.power(points_1 - points_2, 2), axis=1))
    return diffs


def compute_target_regist_error_statistic(points_ref, points_est):
    """ compute distance as between related points in two sets
    and make a statistic on those distances - mean, std, median, min, max

    :param ndarray points_ref: final landmarks in target image of  np.array<nb_points, dim>
    :param ndarray points_est: warped landmarks from source to target of np.array<nb_points, dim>
    :return tuple(ndarray,dict): (np.array<nb_points, 1>, dict)
    ([], {'overlap points': 0})
    """
    if not all(pts is not None and list(pts) for pts in [points_ref, points_est]):
        return [], {'overlap points': 0}

    lnd_sizes = [len(points_ref), len(points_est)]
    assert min(lnd_sizes) > 0, 'no common landmarks for metric'
    diffs = compute_tre(points_ref, points_est)

    inter_dist = distance.cdist(points_ref[:len(diffs)], points_ref[:len(diffs)])
    # inter_dist[range(len(points_ref)), range(len(points_ref))] = np.inf
    dist = np.mean(inter_dist, axis=0)
    weights = dist / np.sum(dist)

    dict_stat = {
        'Mean': np.mean(diffs),
        'Mean_weighted': np.sum(diffs * weights),
        'STD': np.std(diffs),
        'Median': np.median(diffs),
        'Min': np.min(diffs),
        'Max': np.max(diffs),
        'overlap points': min(lnd_sizes) / float(max(lnd_sizes))
    }
    return diffs, dict_stat


def load_landmarks(path_file):
    """ load landmarks in csv and txt format

    :param str path_file: path to the input file
    :return ndarray: np.array<np_points, dim> of landmarks points

    """
    if not os.path.isfile(path_file):
        logging.warning('missing landmarks "%s"', path_file)
        return None
    _, ext = os.path.splitext(path_file)
    if ext == '.csv':
        return load_landmarks_csv(path_file)
    elif ext == '.pts':
        return load_landmarks_pts(path_file)


def load_landmarks_pts(path_file):
    """ load file with landmarks in txt format

    :param str path_file: path to the input file
    :return ndarray: np.array<np_points, dim> of landmarks points

    """
    assert os.path.isfile(path_file), 'missing file "%s"' % path_file
    with open(path_file, 'r') as fp:
        data = fp.read()
        lines = data.split('\n')
        # lines = [re.sub("(\\r|)\\n$", '', line) for line in lines]
    if len(lines) < 2:
        return np.zeros((0, 2))
    nb_points = int(lines[1])
    points = [[float(n) for n in line.split()]
              for line in lines[2:] if line]
    assert nb_points == len(points), 'number of declared (%i) and found (%i) ' \
                                     'does not match' % (nb_points, len(points))
    return np.array(points, dtype=np.float)


def load_landmarks_csv(path_file):
    """ load file with landmarks in cdv format

    :param str path_file: path to the input file
    :return ndarray: np.array<np_points, dim> of landmarks points
    """
    assert os.path.isfile(path_file), 'missing file "%s"' % path_file
    df = pd.read_csv(path_file, index_col=0)
    points = df[LANDMARK_COORDS].values
    return points


def save_landmarks(path_file, landmarks):
    """ save landmarks into a specific file

    both used formats csv/pts is using the same coordinate frame,
    the origin (0, 0) is located in top left corner of the image

    :param str path_file: path to the output file
    :param landmarks: np.array<np_points, dim>
    """
    assert os.path.isdir(os.path.dirname(path_file)), \
        'missing folder "%s"' % os.path.dirname(path_file)
    path_file, _ = os.path.splitext(path_file)
    landmarks = landmarks.values if isinstance(landmarks, pd.DataFrame) else landmarks
    save_landmarks_csv(path_file + '.csv', landmarks)
    save_landmarks_pts(path_file + '.pts', landmarks)


def save_landmarks_pts(path_file, landmarks):
    """ save landmarks into a txt file

    we are using VTK pointdata legacy format, ITK compatible::

        <index, point>
        <number of points>
        point1-x point1-y [point1-z]
        point2-x point2-y [point2-z]

    .. seealso:: https://simpleelastix.readthedocs.io/PointBasedRegistration.html

    :param str path_file: path to the output file
    :param landmarks: np.array<np_points, dim>
    :return str: file path
    """
    assert os.path.isdir(os.path.dirname(path_file)), \
        'missing folder "%s"' % os.path.dirname(path_file)
    path_file = os.path.splitext(path_file)[0] + '.pts'
    lines = ['point', str(len(landmarks))]
    lines += [' '.join(str(i) for i in point) for point in landmarks]
    with open(path_file, 'w') as fp:
        fp.write('\n'.join(lines))
    return path_file


def save_landmarks_csv(path_file, landmarks):
    """ save landmarks into a csv file

    we are using simple format::

        ,X,Y
        0,point1-x,point1-y
        1,point2-x,point2-y

    :param str path_file: path to the output file
    :param landmarks: np.array<np_points, dim>
    :return str: file path
    """
    assert os.path.isdir(os.path.dirname(path_file)), \
        'missing folder "%s"' % os.path.dirname(path_file)
    path_file = os.path.splitext(path_file)[0] + '.csv'
    df = pd.DataFrame(landmarks, columns=LANDMARK_COORDS)
    df.index = np.arange(0, len(df))
    df.to_csv(path_file)
    return path_file


"""  *********************** Warp Related *************************** """

def transform_point(transform, point):
    transformed_point = transform.TransformPoint(point)
    return transformed_point


def warp_img(src_img, tar_img, src_transform):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(tar_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(src_transform)
    src_warp = resampler.Execute(src_img)
    return src_warp

def warp_seg(src_img, tar_img, src_transform):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(tar_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(src_transform)
    src_warp = resampler.Execute(src_img)
    return src_warp


"""  *********************** Segmentation Related *************************** """

def dice(vol1, vol2, labels=None, nargout=1):
    '''
    Dice [1] volume overlap metric

    The default is to *not* return a measure for the background layer (label = 0)

    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)



"""  *********************** Checkpoint Related *************************** """

def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


"""  *********************** Others *************************** """


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def print_network(net):
  num_params = 0
  for param in net.parameters():
    num_params += param.numel()
  print('Total number of parameters: %d' % num_params)


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


def findall(s, string):
    ret = []
    index = 0
    while True:
        index = string.find(s, index)
        if index != -1:
            ret.append(index)
            index += len(s)
        else:
            break
    return tuple(ret)

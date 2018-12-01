import __init_paths

import os
import cv2
import epdb
import datetime
import numpy as np
from easydict import EasyDict as edict

import torch
from torch.autograd import Variable

from experiments.siamese_fc.Config import Config
from experiments.siamese_fc.network import SiamNet
from lib.Tracking_Utils import *


print(torch.__version__)


def tracker_init(im, target_position, target_size, net):
    '''
    im: cv2.imread
    target_position: [cy, cx]
    target_sz: [h, w]
    net: loaded net.cuda()
    '''
    # get the default parameters
    p = Config()

    # first frame
    img_uint8 = im
    # img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)
    img_uint8 = img_uint8[:, :, ::-1]
    img_double = np.double(img_uint8)    # uint8 to float

    # compute avg for padding
    avg_chans = np.mean(img_double, axis=(0, 1))

    wc_z = target_size[1] + p.context_amount * sum(target_size)
    hc_z = target_size[0] + p.context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.examplar_size / s_z

    # crop examplar z in the first frame
    z_crop = get_subwindow_tracking(img_double, target_position, p.examplar_size, round(s_z), avg_chans)

    # z_crop = np.uint8(z_crop)  # you need to convert it to uint8
    # convert image to tensor
    # z_crop_tensor = 255.0 * _to_tensor(z_crop).unsqueeze(0)
    z_crop_tensor = _to_tensor(z_crop).unsqueeze(0)

    d_search = (p.instance_size - p.examplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad
    # arbitrary scale saturation
    min_s_x = p.scale_min * s_x
    max_s_x = p.scale_max * s_x

    # generate cosine window
    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size * p.response_UP), np.hanning(p.score_size * p.response_UP))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size * p.response_UP, p.score_size * p.response_UP))
    window = window / sum(sum(window))

    # pyramid scale search
    scales = p.scale_step**np.linspace(-np.ceil(p.num_scale/2), np.ceil(p.num_scale/2), p.num_scale)

    # extract feature for examplar z
    z_features = net.feat_extraction(Variable(z_crop_tensor).cuda())
    z_features = z_features.repeat(p.num_scale, 1, 1, 1)

    state = edict()
    state.p = Config()
    state.window = window
    state.avg_chans = avg_chans

    state.scales = scales
    state.s_x = s_x
    state.min_s_x = min_s_x
    state.max_s_x = max_s_x

    state.target_size = target_size
    state.target_position = target_position
    state.z_features = z_features

    state.net = net

    return state


def tracker_track(state, img):
    '''
    img: cv2.imread
    state
    '''
    s_x = state.s_x
    min_s_x = state.min_s_x
    max_s_x = state.max_s_x
    scales = state.scales
    target_size = state.target_size
    target_position = state.target_position
    p = state.p
    avg_chans = state.avg_chans
    z_features = state.z_features
    window = state.window
    net = state.net

    # do detection
    # currently, we only consider RGB images for tracking
    img_uint8 = img
    # img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)
    img_uint8 = img_uint8[:, :, ::-1]
    img_double = np.double(img_uint8)  # uint8 to float

    scaled_instance = s_x * scales
    scaled_target = np.zeros((2, scales.size), dtype = np.double)
    scaled_target[0, :] = target_size[0] * scales
    scaled_target[1, :] = target_size[1] * scales

    # extract scaled crops for search region x at previous target position
    x_crops = make_scale_pyramid(img_double, target_position, scaled_instance, p.instance_size, avg_chans, p)

    # get features of search regions
    x_crops_tensor = torch.FloatTensor(x_crops.shape[3], x_crops.shape[2], x_crops.shape[1], x_crops.shape[0])
    # response_map = SiameseNet.get_response_map(z_features, x_crops)
    for k in range(x_crops.shape[3]):
        tmp_x_crop = x_crops[:, :, :, k]
        # tmp_x_crop = np.uint8(tmp_x_crop)
        # numpy array to tensor
        # x_crops_tensor[k, :, :, :] = 255.0 * _to_tensor(tmp_x_crop).unsqueeze(0)
        x_crops_tensor[k, :, :, :] = _to_tensor(tmp_x_crop).unsqueeze(0)

    # get features of search regions
    x_features = net.feat_extraction(Variable(x_crops_tensor).cuda())

    # evaluate the offline-trained network for exemplar x features
    target_position, new_scale = tracker_eval(net, round(s_x), z_features, x_features, target_position, window, p)

    # scale damping and saturation
    s_x = max(min_s_x, min(max_s_x, (1 - p.scale_LR) * s_x + p.scale_LR * scaled_instance[int(new_scale)]))
    target_size = (1 - p.scale_LR) * target_size + p.scale_LR * np.array([scaled_target[0, int(new_scale)], scaled_target[1, int(new_scale)]])

    rect_position = np.array([target_position[1]-target_size[1]/2, target_position[0]-target_size[0]/2, target_size[1], target_size[0]])

    if p.visualization:
        visualize_tracking_result(img_uint8, rect_position, 1)

    # output bbox in the original frame coordinates
    state.target_position = target_position
    state.target_size = target_size
    return state


def _to_tensor(img):
    im_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    return im_tensor


if __name__ == "__main__":

    # get the default parameters
    p = Config()

    # load model
    net = SiamNet()
    net.load_state_dict(torch.load(os.path.join(p.net_base_path, p.net)))
    net = net.cuda()

    # evaluation mode
    net.eval()

    # load sequence
    img_list, target_position, target_size = load_sequence(p.seq_base_path, p.video)

    state = tracker_init(cv2.imread(img_list[0]), target_position, target_size, net)

    for idx, img in enumerate(img_list[1:], start=1):
        state = tracker_track(state, cv2.imread(img))
        print('processed {}th frame'.format(idx))
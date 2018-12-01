#!/usr/bin/python
import __init_paths

import vot
from vot import Rectangle
import sys
import cv2  # imread
import torch
import numpy as np
import os
from os.path import realpath, dirname, join

from experiments.siamese_fc.network import SiamNet
from experiments.siamese_fc.Config import Config
from run_SiamFC_VOT import tracker_track, tracker_init


print('calling again')


def cxy_wh_2_rect(pos, sz):
    pos = pos[::-1]
    sz = sz[::-1]
    return np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])  # 0-index


def get_axis_aligned_bbox(region):
    region = np.array([region[0][0][0], region[0][0][1], region[0][1][0], region[0][1][1],
                       region[0][2][0], region[0][2][1], region[0][3][0], region[0][3][1]])
    cx = np.mean(region[0::2])
    cy = np.mean(region[1::2])
    x1 = min(region[0::2])
    x2 = max(region[0::2])
    y1 = min(region[1::2])
    y2 = max(region[1::2])
    A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1
    return cx, cy, w, h


# get the default parameters
p = Config()

# load model
net = SiamNet()
net.load_state_dict(torch.load(os.path.join(p.net_base_path, p.net)))
net = net.cuda()

# evaluation mode
net.eval().cuda()

# warm up
for i in range(10):
    net(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda(),
        torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())

# start to track
handle = vot.VOT("polygon")
Polygon = handle.region()
cx, cy, w, h = get_axis_aligned_bbox(Polygon)

image_file = handle.frame()
if not image_file:
    sys.exit(0)

target_pos, target_sz = np.array([cy, cx]), np.array([h, w])
im = cv2.imread(image_file)  # HxWxC

state = tracker_init(im, target_pos, target_sz, net)  # init tracker

while True:
    image_file = handle.frame()
    if not image_file:
        break
    im = cv2.imread(image_file)  # HxWxC
    state = tracker_track(state, im)  # track
    res = cxy_wh_2_rect(state['target_position'], state['target_size'])

    handle.report(Rectangle(res[0], res[1], res[2], res[3]))


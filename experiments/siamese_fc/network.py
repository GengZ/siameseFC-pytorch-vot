from __future__ import absolute_import

import epdb
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Config import Config


class SiamNet(nn.Module):

    def __init__(self):
        super(SiamNet, self).__init__()
        self.config = Config()

        # architecture (AlexNet like)
        self.feat_extraction = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 96, 11, 2)),             # conv1
                ('bn1', nn.BatchNorm2d(96)),
                ('relu1', nn.ReLU(inplace=True)),
                ('pool1', nn.MaxPool2d(3, 2)),
                ('conv2', nn.Conv2d(96, 256, 5, 1, groups=2)),  # conv2
                ('bn2', nn.BatchNorm2d(256)),
                ('relu2', nn.ReLU(inplace=True)),
                ('pool2', nn.MaxPool2d(3, 2)),
                ('conv3', nn.Conv2d(256, 384, 3, 1)),           # conv3
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),
                ('conv4', nn.Conv2d(384, 384, 3, 1, groups=2)),  # conv4
                ('bn4', nn.BatchNorm2d(384)),
                ('relu4', nn.ReLU(inplace=True)),
                ('conv5', nn.Conv2d(384, 256, 3, 1, groups=2))  # conv5
            ])
        )

        # adjust layer as in the original SiamFC in matconvnet
        self.adjust = nn.Conv2d(1, 1, 1, 1)

        # initialize weights
        self._initialize_weight()

    def forward(self, z, x):
        """
        forward pass
        z: examplare, BxCxHxW
        x: search region, BxCxH1xW1
        """
        # get features for z and x
        z_feat = self.feat_extraction(z)
        x_feat = self.feat_extraction(x)

        # correlation of z and z
        xcorr_out = self.xcorr_my(z_feat, x_feat)

        score = self.adjust(xcorr_out)

        return score

    def xcorr_my(self, z, x):
        """
        correlation layer as in the original SiamFC,
        (convolution process in fact)
        """
        batch_size, kernel_size = z.size(0), z.size(3)

        # ----------------------------------------------
        # NOTE:
        # this could be replaced with group convolution

        cls_out = None
        for i in range(batch_size):
            iim_feat = x[i:i+1].contiguous()
            iref_im_feat = z[i:i+1].contiguous()
            # conv
            ikernel = iref_im_feat.view(1, -1, kernel_size, kernel_size)
            icls_out = F.conv2d(iim_feat, ikernel, groups=1)
            # concat
            if type(cls_out).__name__ == 'NoneType':
                cls_out = icls_out
            else:
                cls_out = torch.cat((cls_out, icls_out))
        # ----------------------------------------------

        return cls_out

    def xcorr(self, z, x):
        """
        correlation layer as in the original SiamFC,
        (convolution process in fact)
        """
        batch_size_x, channel_x, w_x, h_x = x.shape
        x = x.view(1, batch_size_x * channel_x, w_x, h_x).contiguous()

        # group convolution
        out = F.conv2d(x, z, groups=batch_size_x)

        batch_size_out, channel_out, w_out, h_out = out.shape
        xcorr_out = out.view(channel_out, batch_size_out, w_out, h_out)

        return xcorr_out

    def _initialize_weight(self):
        """
        initialize network parameters
        """
        tmp_layer_idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                tmp_layer_idx = tmp_layer_idx + 1

                if tmp_layer_idx < 6:
                    # kaiming initialization
                    if self.config.init == 'kaiming':
                        nn.init.kaiming_normal(m.weight.data, mode='fan_out')

                    # xavier initialization
                    elif self.config.init == 'xavier':
                        nn.init.xavier_normal(m.weight.data)
                        m.bias.data.fill_(.1)

                    elif self.config.init == 'truncated':
                        def truncated_norm_init(data, stddev=.01):
                            weight = np.random.normal(size=data.shape)
                            weight = np.clip(weight,
                                             a_min=-2*stddev, a_max=2*stddev)
                            weight = torch.from_numpy(weight).float()
                            return weight
                        m.weight.data = truncated_norm_init(m.weight.data)
                        m.bias.data.fill_(.1)

                    else:
                        raise NotImplementedError
                else:
                    # initialization for adjust layer as in the original paper
                    m.weight.data.fill_(1e-3)
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.momentum = self.config.bn_momentum

    def weight_loss(self, prediction, label, weight):
        """
        weighted cross entropy loss
        """
        return F.binary_cross_entropy_with_logits(prediction,
                                                  label,
                                                  weight,
                                                  size_average=True)

    def customize_loss(self, prediction, label, weight):
        score, y, weights = prediction, label, weight

        a = -(score * y)
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b) + torch.exp(a-b))
        loss = torch.mean(weights * loss)
        return loss

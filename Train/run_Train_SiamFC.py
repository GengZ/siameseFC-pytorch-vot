import __init_paths

import os
import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from lib.VIDDataset import VIDDataset
from lib.DataAugmentation import RandomStretch, CenterCrop, RandomCrop, ToTensor
from lib.Utils import create_label
from lib.eval_utils import centerThrErr

from experiments.siamese_fc.Config import Config
from experiments.siamese_fc.network import SiamNet


# fix random seed
np.random.seed(1357)
torch.manual_seed(1234)


def train(data_dir, train_imdb, val_imdb,
          model_save_path="./model/",
          config=None):

    assert config is not None
    use_gpu = config.use_gpu

    # set gpu ID
    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

    # do data augmentation in PyTorch;
    # you can also do complex data augmentation as in the original paper
    center_crop_size = config.instance_size - config.stride
    random_crop_size = config.instance_size - 2 * config.stride

    train_z_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((config.examplar_size, config.examplar_size)),
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((center_crop_size, center_crop_size)),
        RandomCrop((random_crop_size, random_crop_size)),
        ToTensor()
    ])
    valid_z_transforms = transforms.Compose([
        CenterCrop((config.examplar_size, config.examplar_size)),
        ToTensor(),
    ])
    valid_x_transforms = transforms.Compose([
        ToTensor()
    ])

    # load data (see details in VIDDataset.py)
    train_dataset = VIDDataset(train_imdb, data_dir, config,
                               train_z_transforms, train_x_transforms)
    val_dataset = VIDDataset(val_imdb, data_dir, config, valid_z_transforms,
                             valid_x_transforms, "Validation")

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.train_num_workers,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=True,
                            num_workers=config.val_num_workers,
                            drop_last=True)

    # create SiamFC network architecture (see details in SiamNet.py)
    net = SiamNet()
    # move network to GPU if using GPU
    if use_gpu:
        net.cuda()

    # define training strategy;
    # ========================================
    # customize parameters attributes
    # 1. adjust layer weight learnable
    # 2. bias in feat_extraction exempt from weight_decay
    params = []
    # feature extract
    for key, value in dict(net.feat_extraction.named_parameters()).items():
        if 'conv' in key:
            if 'bias' in key:
                params += [{'params': [value],
                            'weight_decay': 0}]
            else:   # weight
                params += [{'params': [value],
                            'weight_decay': config.weight_decay}]
        if 'bn' in key:
            params += [{'params': [value],
                        'weight_decay': 0}]
    # adjust layer
    params += [
                {'params': [net.adjust.bias]},
                # {'params': [net.adjust.weight], 'lr': not config.fix_adjust_layer},
    ]
    if config.fix_adjust_layer:
        params += [
                {'params': [net.adjust.weight], 'lr': 0},
        ]
    else:
        params += [
                {'params': [net.adjust.weight]},
        ]
    # ========================================
    optimizer = torch.optim.SGD(params,
                                config.lr,
                                config.momentum,
                                config.weight_decay)
    # adjusting learning in each epoch
    if not config.resume:
        train_lrs = np.logspace(-2, -5, config.num_epoch)
        scheduler = LambdaLR(optimizer, lambda epoch: train_lrs[epoch])
    else:
        train_lrs = np.logspace(-2, -5, config.num_epoch)
        train_lrs = train_lrs[config.start_epoch:]

        net.load_state_dict(torch.load(os.path.join(model_save_path,
                                                    'SiamFC_' +
                                                    str(config.start_epoch) +
                                                    '_model.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(model_save_path,
                                                          'optimizer.pth')))
        scheduler = LambdaLR(optimizer, lambda epoch: train_lrs[epoch])
        print('resume training from epoch {} checkpoint'.format(config.start_epoch))

    # used to control generating label for training;
    # once generated, they are fixed since the labels for each
    # pair of images (examplar z and search region x) are the same
    train_response_flag = False
    valid_response_flag = False

    # -------------------- training & validation process --------------------
    for i in range(config.start_epoch, config.num_epoch):

        # adjusting learning rate
        scheduler.step()

        # -------------------------- training --------------------------
        # indicating training (very important for batch normalization)
        net.train()

        # used to collect loss
        train_loss = []

        # used as eval metric
        err_disp = 0
        sample_num = 0

        for j, data in enumerate(train_loader):

            # fetch data,
            # i.e., B x C x W x H (batchsize x channel x wdith x heigh)
            exemplar_imgs, instance_imgs = data

            # forward pass
            if use_gpu:
                exemplar_imgs = exemplar_imgs.cuda()
                instance_imgs = instance_imgs.cuda()
            output = net.forward(Variable(exemplar_imgs),
                                 Variable(instance_imgs))

            # create label for training (only do it one time)
            if not train_response_flag:
                # change control flag
                train_response_flag = True
                # get shape of output (i.e., response map)
                response_size = output.shape[2:4]
                # generate label and weight
                train_eltwise_label, train_instance_weight = \
                    create_label(response_size, config, use_gpu)

            # clear the gradient
            optimizer.zero_grad()

            # loss
            if config.loss == "logistic":
                loss = net.weight_loss(output,
                                       Variable(train_eltwise_label),
                                       Variable(train_instance_weight))
            elif config.loss == 'customize':
                loss = net.customize_loss(output,
                                          Variable(train_eltwise_label),
                                          Variable(train_instance_weight))

            # backward
            loss.backward()

            # update parameter
            optimizer.step()

            # collect training loss
            train_loss.append(loss.data)

            # collect additional data for metric
            err_disp = centerThrErr(output.data.cpu().numpy(),
                                    train_eltwise_label.cpu().numpy(),
                                    err_disp, sample_num)
            sample_num += config.batch_size

            # stdout
            if (j + 1) % config.log_freq == 0:
                print ("Epoch %d, Iter %06d, loss: %f, error disp: %f"
                       % (i+1, (j+1), np.mean(train_loss), err_disp))

        # ------------------------- saving model ---------------------------
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        torch.save(net.state_dict(),
                   os.path.join(model_save_path,
                                "SiamFC_" + str(i + 1) + "_model.pth"))
        torch.save(optimizer.state_dict(),
                   os.path.join(model_save_path,
                                'optimizer.pth'))

        # --------------------------- validation ---------------------------
        # indicate validation
        net.eval()

        # used to collect validation loss
        val_loss = []

        for j, data in enumerate(tqdm(val_loader)):

            exemplar_imgs, instance_imgs = data

            # forward pass
            if use_gpu:
                exemplar_imgs = exemplar_imgs.cuda()
                instance_imgs = instance_imgs.cuda()
            output = net.forward(Variable(exemplar_imgs),
                                 Variable(instance_imgs))

            # create label for validation (only do it one time)
            if not valid_response_flag:
                valid_response_flag = True
                response_size = output.shape[2:4]
                valid_eltwise_label, valid_instance_weight = \
                    create_label(response_size, config, use_gpu)

            # loss
            if config.loss == "logistic":
                loss = net.weight_loss(output,
                                       Variable(valid_eltwise_label),
                                       Variable(valid_instance_weight))
            elif config.loss == 'customize':
                loss = net.customize_loss(output,
                                          Variable(valid_eltwise_label),
                                          Variable(valid_instance_weight))

            # collect validation loss
            val_loss.append(loss.data)

        print ("Epoch %d   training loss: %f, validation loss: %f"
               % (i+1, np.mean(train_loss), np.mean(val_loss)))


if __name__ == "__main__":

    # initialize training configuration
    config = Config()

    data_dir = config.data_dir
    train_imdb = config.train_imdb
    val_imdb = config.val_imdb
    model_save_path = os.path.join(config.save_base_path, config.save_sub_path)

    # training SiamFC network, using GPU by default
    train(data_dir, train_imdb, val_imdb,
          model_save_path=model_save_path,
          config=config)

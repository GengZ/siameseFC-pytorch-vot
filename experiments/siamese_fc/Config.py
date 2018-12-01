"""
Configuration for training SiamFC and tracking evaluation
Written by Heng Fan
"""


class Config:
    def __init__(self):
        #
        self.use_gpu = True
        self.gpu_id = 0
        self.log_freq = 50
        self.save_base_path = '../models'
        self.save_sub_path = 'logistic_all_data'
        self.resume = False
        self.start_epoch = 0
        if not self.resume:
            assert self.start_epoch == 0

        # data path
        self.data_dir = "/home/gzhan/data/ILSVRC/ILSVRC2015_curation/Data/VID/train"
        self.train_imdb = "../ILSVRC15-curation/imdb_video_train.json"
        self.val_imdb = "../ILSVRC15-curation/imdb_video_val.json"

        # parameters for training
        self.pos_pair_range = 100
        self.num_pairs = 5.32e4
        self.val_ratio = 0.
        self.num_epoch = 50
        self.batch_size = 8
        self.examplar_size = 127
        self.instance_size = 255
        self.sub_mean = 0
        self.train_num_workers = 8  # number of threads to load data
        self.val_num_workers = 8
        self.stride = 8
        self.rPos = 16
        self.rNeg = 0
        self.label_weight_method = "balanced"

        self.lr = 1                  # learning rate of SGD
        self.momentum = 0.9          # momentum of SGD
        self.weight_decay = 5e-4     # weight decay of optimizator
        self.bn_momentum = .0003
        self.init = 'truncated'
        self.loss = 'logistic'
        self.neg_label = 0
        self.fix_adjust_layer = False

        # parameters for tracking (SiamFC-3s by default)
        self.num_scale = 3
        self.scale_step = 1.0375
        self.scale_penalty = 0.9745
        self.scale_LR = 0.59
        self.response_UP = 16
        self.windowing = "cosine"
        self.w_influence = 0.176

        self.video = "Lemming"
        self.visualization = 0
        self.bbox_output = True
        self.bbox_output_path = "./tracking_result/"

        self.context_amount = 0.5
        self.scale_min = 0.2
        self.scale_max = 5
        self.score_size = 17

        # path to your trained model
        self.net_base_path = "/home/gzhan/baseline/track.baseline/Train/model"
        # path to your sequences (sequence should be in OTB format)
        self.seq_base_path = "/home/gzhan/data/OTB/data"
        # which model to use
        self.net = "SiamFC_50_model.pth"

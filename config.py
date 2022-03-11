import torch


class Config:
    def __init__(self):
        self.num_layers = 3
        self.in_points_dim = 3
        self.first_feats_dim = 512
        self.final_feats_dim = 96
        self.first_subsampling_dl = 0.06
        self.in_feats_dim = 1
        self.conv_radius = 2.75
        self.deform_radius = 5.0
        self.num_kernel_points = 15
        self.KP_extent = 2.0
        self.KP_influence = "linear"
        self.aggregation_mode = "sum"
        self.fixed_kernel_points = "center"
        self.use_batch_norm = True
        self.batch_norm_momentum = 0.02
        self.deformable = False
        self.modulated = False

        self.batch_size = 1
        self.num_workers = 12

        self.max_epoch = 250
        self.save_dir = "./params"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_points = 256

        # self.model = args.model.to(self.device)
        self.lr = 0.01
        self.momentum = 0.98
        self.weight_decay = 0.000001

        self.architecture = [
            'simple',
            'resnetb',
            'resnetb',
            'resnetb_strided',
            'resnetb',
            'resnetb',
            'resnetb_strided',
            'resnetb',
            'resnetb',
            'nearest_upsample',
            'unary',
            'unary',
            'nearest_upsample',
            'unary',
            'last_unary'
        ]

        self.root = "C:/Users/sdnyz/PycharmProjects/dataset/modelnet40_ply_hdf5_2048"
        self.train_cls = "modelnet40_half1"
        self.val_cls = "modelnet40_half1"
        self.test_cls = "modelnet40_half2"

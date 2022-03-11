import numpy as np
import torch
import open3d as o3d
from torch.utils import data
from common.utils import get_correspondences, to_o3d_pcd, to_tsfm, square_distance
from datasets.modelnet40.datasets import get_train_datasets, get_test_datasets
import json
from descriptor.lrf import LRF
from descriptor import ball_surface


class ModelNet40(data.Dataset):
    # cls_ = ["modelnet40_all", "modelnet40_half1", "modelnet40_half2"]
    def __init__(self, root, mode="train", cls_=("modelnet40_all", "modelnet40_half2"), use_B=False):
        super(ModelNet40, self).__init__()
        # noise_type = ['clean', 'jitter', 'crop']
        noise_type, rot_mag, trans_mag, num_points, partial = "crop", 45.0, 0.5, 1024, [0.7, 0.7]
        train_set, val_set, test_set = None, None, None
        if mode == "train" or mode == "val":
            train_categoryfile, val_categoryfile = "./datasets/modelnet40/%s.txt" % cls_[0], "./datasets/modelnet40/%s.txt" % cls_[1]
            train_set, val_set = get_train_datasets(
                train_categoryfile, val_categoryfile,
                noise_type, rot_mag, trans_mag, num_points, partial,
                root
            )
        else:
            test_category_file = "./datasets/modelnet40/%s.txt" % cls_
            test_set = get_test_datasets(test_category_file, noise_type, rot_mag, trans_mag, num_points, partial, root)
        self.data = None
        self.mode = mode
        if mode == "train":
            self.data = train_set
        elif mode == "val":
            self.data = val_set
        elif mode == "test":
            self.data = test_set
        self.overlap_radius = 0.04
        self.config = None
        self.use_B = use_B

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        src_pcd = sample['points_src'][:, :3]
        tgt_pcd = sample['points_ref'][:, :3]
        rot = sample['transform_gt'][:, :3]
        trans = sample['transform_gt'][:, 3][:, None]
        matching_inds = get_correspondences(to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd), to_tsfm(rot, trans), self.overlap_radius)

        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)

        for k, v in sample.items():
            if k not in ['deterministic', 'label', 'idx']:
                sample[k] = torch.from_numpy(v).unsqueeze(0)

        if self.use_B:
            # ball surface
            raw_pc = to_o3d_pcd(sample["points_raw"][0][:, :3])
            features = np.zeros(shape=(sample["points_raw"][0][:, :3].shape[0], 72))
            raw_lrf = LRF(raw_pc, o3d.geometry.KDTreeFlann(raw_pc), 0.3)
            raw_lrfs = np.zeros(shape=(sample["points_raw"][0][:, :3].shape[0], 3, 3))
            for j in range(sample["points_raw"][0][:, :3].shape[0]):
                pt = sample["points_raw"][0][:, :3][j].numpy()
                patch = raw_lrf.get(pt)
                feat = ball_surface.compute(patch)
                features[j, :] = feat
                raw_lrfs[j, :, :] = raw_lrf.get_lrf(pt).T
            src_idx = square_distance(
                torch.from_numpy(np.asarray(to_o3d_pcd(src_pcd)
                                            .transform(np.concatenate([
                    np.concatenate([rot, trans], axis=1),
                    np.array([[0, 0, 0, 1]])
                ], axis=0)).points)).float().unsqueeze(0), sample["points_raw"][:, :, :3])[0].min(1)[1]
            tgt_idx = square_distance(torch.from_numpy(tgt_pcd).float().unsqueeze(0), sample["points_raw"][:, :, :3])[0].min(1)[1]
            # src_lrfs = torch.matmul(torch.from_numpy(raw_lrfs).float(), torch.from_numpy(rot).float().unsqueeze(0).repeat(
            #     [sample["points_raw"][0][:, :3].shape[0], 1, 1])
            # )[src_idx]
            # tgt_lrfs = torch.from_numpy(raw_lrfs).float()[tgt_idx]
            src_feats, tgt_feats = features[src_idx], features[tgt_idx]
            # return src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans, matching_inds, src_pcd, tgt_pcd, sample, src_lrfs, tgt_lrfs

        return src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans, matching_inds, src_pcd, tgt_pcd, sample


class ModelLoNet40:
    def __init__(self, modelnet_test_set):
        self.dataset = modelnet_test_set
        with open("ModelLoNet.json", "r") as f:
            self.idx = json.load(f)
        self.config = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        return self.dataset[self.idx[item]]


if __name__ == '__main__':
    import open3d as o3d
    root = "E:/modelnet40_ply_hdf5_2048"
    train_set = ModelNet40(root=root, mode="train", cls_=["modelnet40_all", "modelnet40_half2"])
    val_set = ModelNet40(root=root, mode="val", cls_=["modelnet40_all", "modelnet40_half1"])
    print(len(train_set), len(val_set))
    for i in range(len(train_set)):
        src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans, matching_inds, _, _, sample = train_set[i]
        print(src_pcd.shape)
        src_pc = to_o3d_pcd(src_pcd)
        tgt_pc = to_o3d_pcd(tgt_pcd)
        src_pc.colors = o3d.Vector3dVector(np.array([[1, 0.706, 0]]*src_pcd.shape[0]))
        tgt_pc.colors = o3d.Vector3dVector(np.array([[0, 0.651, 0.929]]*tgt_pcd.shape[0]))
        o3d.draw_geometries([src_pc, tgt_pc], window_name="modelnet40", width=1000, height=800)
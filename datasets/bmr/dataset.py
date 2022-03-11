import numpy as np
import torch
import open3d as o3d
from torch.utils import data
from io import StringIO
from common.utils import to_o3d_pcd, get_correspondences, to_tsfm
from dataloader import batch_grid_subsampling_kpconv


class BMR(data.Dataset):
    def __init__(self):
        super(BMR, self).__init__()
        with open("./BMR/dataload.txt", "r") as f:
            self.pairs = [line.replace("\n", "").split(" ") for line in f.readlines()[1:]]
        self.overlap_radius = 0.03

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        T = np.array([float(x) for x in self.pairs[index][2:]]).reshape(4, 4)
        if self.pairs[index][0].startswith("M"):
            self.pairs[index][0] = self.pairs[index][0].replace("M", "m")
        if self.pairs[index][1].startswith("M"):
            self.pairs[index][1] = self.pairs[index][1].replace("M", "m")

        with open("./BMR/%s" % self.pairs[index][0], "r") as f:
            src_pcd = np.loadtxt(StringIO("\n".join(f.readlines()[1:])))
        with open("./BMR/%s" % self.pairs[index][1], "r") as f:
            tgt_pcd = np.loadtxt(StringIO("\n".join(f.readlines()[1:])))
        rot, trans = T[:3, :3], T[:3, 3:]

        src_center = np.mean(src_pcd, axis=0, keepdims=True)
        src_pcd_centered = src_pcd - src_center
        src_max_norm = np.max(np.linalg.norm(src_pcd_centered, axis=1))
        tgt_center = np.mean(tgt_pcd, axis=0, keepdims=True)
        tgt_pcd_centered = tgt_pcd - tgt_center
        tgt_max_norm = np.max(np.linalg.norm(tgt_pcd_centered, axis=1))
        scale = max(src_max_norm, tgt_max_norm)

        src_pcd = np.asarray(to_o3d_pcd(src_pcd).transform(T).points)
        src_pcd, tgt_pcd = src_pcd / scale, tgt_pcd / scale
        T_rev = np.eye(4)
        T_rev[:3, :3] = T[:3, :3].T
        src_pcd = np.asarray(to_o3d_pcd(src_pcd-T[:3, 3].reshape(1, 3)).transform(T_rev).points)

        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)

        src_pcd, _ = batch_grid_subsampling_kpconv(torch.from_numpy(src_pcd), torch.from_numpy(np.array([src_pcd.shape[0]])).int(), sampleDl=0.03)
        tgt_pcd, _ = batch_grid_subsampling_kpconv(torch.from_numpy(tgt_pcd), torch.from_numpy(np.array([tgt_pcd.shape[0]])).int(), sampleDl=0.03)
        src_pcd, tgt_pcd = src_pcd.numpy(), tgt_pcd.numpy()
        matching_inds = get_correspondences(to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd), to_tsfm(rot, trans), self.overlap_radius)

        return src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans, matching_inds, src_pcd, tgt_pcd, T


if __name__ == '__main__':
    bmr_set = BMR()
    for i in range(300, len(bmr_set)):
        src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans, matching_inds, _, _, T = bmr_set[i]
        src_center = np.mean(src_pcd, axis=0, keepdims=True)
        src_pcd_centered = src_pcd - src_center
        print(src_pcd_centered[:, 0].max(), src_pcd_centered[:, 1].max(), src_pcd_centered[:, 2].max())

        src_pc = o3d.geometry.PointCloud()
        src_pc.points = o3d.utility.Vector3dVector(src_pcd)
        src_pc.colors = o3d.utility.Vector3dVector(np.array([[1, 0.706, 0]]*src_pcd.shape[0]))
        src_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(0.12))

        tgt_pc = o3d.geometry.PointCloud()
        tgt_pc.points = o3d.utility.Vector3dVector(tgt_pcd)
        tgt_pc.colors = o3d.utility.Vector3dVector(np.array([[0, 0.651, 0.929]]*tgt_pcd.shape[0]))
        tgt_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(0.12))

        print(matching_inds)
        overlap_idx = np.array(list(set(matching_inds[:, 0].tolist())))
        np.asarray(src_pc.colors)[overlap_idx] = np.array([[1, 0, 0]]*overlap_idx.shape[0])
        print("overlap ratio: %.3f" % (overlap_idx.shape[0] / src_pcd.shape[0]))

        o3d.visualization.draw_geometries([src_pc, tgt_pc])
        o3d.visualization.draw_geometries([src_pc.transform(T), tgt_pc])
import numpy as np
import torch
import open3d as o3d
from torch import optim
from dataset import ModelNet40, ModelLoNet40
from dataloader import get_dataloader
from config import Config
from models import SAO
from common.utils import processbar, get_inputs
from loss import transform_loss
from common.utils import compute_metrics, summarize_metrics
from common.utils import to_o3d_pcd
from common.math_torch import se3
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


root = "C:/Users/sdnyz/PycharmProjects/dataset/modelnet40_ply_hdf5_2048"
# train_cls = "modelnet40_half1"
train_cls = "draw_pic"
val_cls = "modelnet40_half1"
test_cls = "modelnet40_half2"
train_set = ModelNet40(root=root, mode="train", cls_=[train_cls, val_cls])
val_set = ModelNet40(root=root, mode="val", cls_=[train_cls, val_cls])
test_set = ModelNet40(root=root, mode="test", cls_=test_cls)
test_lo_set = ModelLoNet40(test_set)

config = Config()
train_set.config, val_set.config, test_set.config, test_lo_set.config = config, config, config, config
train_loader, neighborhood_limits = get_dataloader(train_set, batch_size=config.batch_size, num_workers=0, shuffle=True)
val_loader, _ = get_dataloader(val_set, batch_size=config.batch_size, num_workers=0, shuffle=False, neighborhood_limits=neighborhood_limits)
test_loader, _ = get_dataloader(test_set, batch_size=config.batch_size, num_workers=0, shuffle=False, neighborhood_limits=neighborhood_limits)
test_lo_loader, _ = get_dataloader(test_lo_set, batch_size=config.batch_size, num_workers=0, shuffle=True, neighborhood_limits=neighborhood_limits)


def look(use_A=False, use_B=False, look_set="test"):
    params_path = {
        "0_0": "./params/sao-no-A-no-B.pth",
        "0_1": "./params/sao-no-A-only-B.pth",
        "1_0": "./params/sao-no-B-only-A.pth",
        "1_1": "./params/sao-A-and-B.pth"
    }
    model = SAO(config, use_A=use_A, use_B=use_B)
    model.to(config.device)
    model.load_state_dict(torch.load(params_path[str(int(use_A))+"_"+str(int(use_B))]))
    loader = None
    if look_set == "train":
        loader = train_loader
    elif look_set == "val":
        loader = val_loader
    elif look_set == "test":
        loader = test_loader
    else:
        loader = test_lo_loader
    test_loss = 0
    all_val_losses = {
        # mae_i: []
        # outlier_i: []
        # total: []
    }
    all_val_metrics_np = {
        'r_mse': [],
        'r_mae': [],
        't_mse': [],
        't_mae': [],
        'err_r_deg': [],
        'err_t': [],
        'chamfer_dist': []
    }
    model.eval()
    with torch.no_grad():
        num_iter = int(len(loader.dataset) // loader.batch_size)
        c_loader_iter = loader.__iter__()
        for c_iter in range(num_iter):
            inputs = get_inputs(c_loader_iter.next(), config)
            # forward pass
            trans_pred, endpoints = model(inputs, num_iter=5)
            # print(endpoints)
            # loss
            rot_gt, trans_gt = inputs['rot'], inputs['trans']
            src_pcd, tgt_pcd = inputs['src_pcd_raw'], inputs['tgt_pcd_raw']
            overlap_ratio = len(set(inputs['correspondences'][:, 0].cpu().numpy().tolist())) / src_pcd.shape[0]

            if overlap_ratio > 0.35:
                continue
            print(overlap_ratio)
            transform_gt = torch.cat((rot_gt, trans_gt), dim=1).unsqueeze(0)
            data = {
                'transform_gt': transform_gt,
                'points_src': src_pcd.unsqueeze(0),
                'points_ref': tgt_pcd.unsqueeze(0),
                'points_raw': inputs["sample"]["points_raw"].to(src_pcd.device)
            }

            val_loss = transform_loss(trans_pred, data, endpoints)
            val_metrics = compute_metrics(data, trans_pred[-1])
            for k in val_loss:
                if all_val_losses.get(k) is None:
                    all_val_losses[k] = []
                all_val_losses[k].append(val_loss[k].view(1, ))
            for k in val_metrics:
                all_val_metrics_np[k].append(val_metrics[k])

            test_loss += val_loss["total"].item()
            print(
                "\rtest process: %s  total loss: %.5f   cd: %.5f  mae 0: %.5f   mae 1: %.5f  outlier 0: "
                "%.5f  outlier 1: %.5f" % (processbar(c_iter + 1, num_iter), val_loss["total"],
                                           val_metrics["chamfer_dist"], val_loss["mae_0"],
                                           val_loss["mae_1"], val_loss["outlier_0"], val_loss["outlier_1"]), end="")
            # 画图
            src_pc, tgt_pc = to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd)
            src_pc.colors = o3d.utility.Vector3dVector(np.array([[1, 0.706, 0]]*src_pcd.shape[0]))
            tgt_pc.colors = o3d.utility.Vector3dVector(np.array([[0, 0.651, 0.929]]*tgt_pcd.shape[0]))
            # registration
            src_pc_trans = to_o3d_pcd(se3.transform(trans_pred[-1], src_pcd.unsqueeze(0))[0])
            src_pc_trans.colors = o3d.utility.Vector3dVector(np.array([[1, 0.706, 0]]*src_pcd.shape[0]))
            # weight ref
            w_pc = to_o3d_pcd(endpoints["weighted_ref"][-1][0].cpu().numpy())
            w_pc.colors = o3d.utility.Vector3dVector(np.array([[0, 0.651, 0.929]]*src_pcd.shape[0]))
            # src, tgt
            o3d.visualization.draw_geometries([src_pc, tgt_pc], window_name="registration", width=1000, height=800)
            # src, w_ref
            # o3d.visualization.draw_geometries([src_pc, w_pc], window_name="registration", width=1000, height=800)
            # src-trans, tgt
            o3d.visualization.draw_geometries([src_pc_trans, tgt_pc], window_name="registration", width=1000, height=800)

        test_loss /= len(val_loader.dataset)
        print("\ntest loss: %.5f" % test_loss)
    # 总指标
    all_val_losses = {k: torch.cat(all_val_losses[k]) for k in all_val_losses}
    all_val_metrics_np = {k: np.concatenate(all_val_metrics_np[k]) for k in all_val_metrics_np}
    mean_val_losses = {k: torch.mean(all_val_losses[k]) for k in all_val_losses}
    summary_metrics = summarize_metrics(all_val_metrics_np)

    print('DeepCP metrics:{:.4f}(rot-rmse) | {:.4f}(rot-mae) | {:.4g}(trans-rmse) | {:.4g}(trans-mae)'.format(
        summary_metrics['r_rmse'], summary_metrics['r_mae'],
        summary_metrics['t_rmse'], summary_metrics['t_mae'],
    ))
    print('Rotation error {:.4f}(deg, mean) | {:.4f}(deg, rmse)'.format(
        summary_metrics['err_r_deg_mean'], summary_metrics['err_r_deg_rmse']))
    print('Translation error {:.4g}(mean) | {:.4g}(rmse)'.format(
        summary_metrics['err_t_mean'], summary_metrics['err_t_rmse']))
    print('Chamfer error: {:.7f}(mean-sq)'.format(
        summary_metrics['chamfer_dist']
    ))

    return test_loss, summary_metrics['chamfer_dist']


if __name__ == '__main__':
    # look_set = ["train", "val", "test", "test_lo"]
    look(use_A=False, use_B=False, look_set="train")
import numpy as np
import torch
from torch import optim
from dataset import ModelNet40, ModelLoNet40
from dataloader import get_dataloader, get_dataloader_lrf
from config import Config
from models import SAO
from common.utils import processbar, get_inputs
from loss import transform_loss
from common.utils import compute_metrics, summarize_metrics


root = "/home/zhang/dataset/modelnet40_ply_hdf5_2048"
train_cls = "modelnet40_all"
val_cls = "modelnet40_half2"
test_cls = "modelnet40_half2"


def evaluate(model, loader):
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
            inputs = get_inputs(c_loader_iter.next(), loader.dataset.config)
            # forward pass
            trans_pred, endpoints = model(inputs, num_iter=5)
            # print(endpoints)
            # loss
            rot_gt, trans_gt = inputs['rot'], inputs['trans']
            src_pcd, tgt_pcd = inputs['src_pcd_raw'], inputs['tgt_pcd_raw']
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
        test_loss /= len(loader.dataset)
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


def train_SAO():
    train_set = ModelNet40(root=root, mode="train", cls_=[train_cls, val_cls])
    val_set = ModelNet40(root=root, mode="val", cls_=[train_cls, val_cls])
    test_set = ModelNet40(root=root, mode="test", cls_=test_cls)
    test_lo_set = ModelLoNet40(test_set)

    config = Config()
    train_set.config, val_set.config, test_set.config, test_lo_set.config = config, config, config, config
    train_loader, neighborhood_limits = get_dataloader(train_set, batch_size=config.batch_size, num_workers=12, shuffle=True)
    val_loader, _ = get_dataloader(val_set, batch_size=config.batch_size, num_workers=12, shuffle=False, neighborhood_limits=neighborhood_limits)
    test_loader, _ = get_dataloader(test_set, batch_size=config.batch_size, num_workers=12, shuffle=False, neighborhood_limits=neighborhood_limits)
    test_lo_loader, _ = get_dataloader(test_lo_set, batch_size=config.batch_size, num_workers=12, shuffle=False, neighborhood_limits=neighborhood_limits)

    save_path = "./params/sao-no-A-no-B.pth"
    model = SAO(config, use_B=False, use_A=False)
    model.to(config.device)
    model.load_state_dict(torch.load(save_path))
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.95,
    )

    min_val_loss, min_val_cd = 1e8, 1e8
    for epoch in range(config.max_epoch):
        num_iter = int(len(train_loader.dataset) // train_loader.batch_size)
        c_loader_iter = train_loader.__iter__()
        train_loss = 0
        model.train()
        for c_iter in range(num_iter):
            ##################################
            # load inputs to device.
            inputs = c_loader_iter.next()
            inputs = get_inputs(inputs, config)
            ##################################
            # forward pass
            trans_pred, endpoints = model(inputs, num_iter=2)
            # print(endpoints)
            # loss
            rot_gt, trans_gt = inputs['rot'], inputs['trans']
            src_pcd, tgt_pcd = inputs['src_pcd_raw'], inputs['tgt_pcd_raw']
            transform_gt = torch.cat((rot_gt, trans_gt), dim=1).unsqueeze(0)
            data = {
                'transform_gt': transform_gt,
                'points_src': src_pcd.unsqueeze(0)
            }
            loss = transform_loss(trans_pred, data, endpoints)
            train_loss += loss['total'].item()
            # backward pass
            optimizer.zero_grad()
            loss['total'].backward()
            optimizer.step()

            print(
                "\rprocess: %s  total loss: %.5f   mae 0: %.5f   mae 1: %.5f  outlier 0: "
                "%.5f  outlier 1: %.5f" % (processbar(c_iter + 1, num_iter), loss["total"], loss["mae_0"],
                          loss["mae_1"], loss["outlier_0"], loss["outlier_1"]), end="")
        train_loss /= len(train_loader.dataset)
        print("\nepoch: %d  train loss: %.5f" % (epoch+1, train_loss))
        scheduler.step()
        val_loss, val_cd = evaluate(model, val_loader)
        # save
        if min_val_cd > val_cd:
            min_val_cd = val_cd
            print("Saving ...")
            torch.save(model.state_dict(), save_path)
            print("Save finish !!!")


def train_SAO_use_B():
    train_set = ModelNet40(root=root, mode="train", cls_=[train_cls, val_cls], use_B=True)
    val_set = ModelNet40(root=root, mode="val", cls_=[train_cls, val_cls], use_B=True)
    test_set = ModelNet40(root=root, mode="test", cls_=test_cls, use_B=True)
    test_lo_set = ModelLoNet40(test_set)

    config = Config()
    train_set.config, val_set.config, test_set.config, test_lo_set.config = config, config, config, config
    train_loader, neighborhood_limits = get_dataloader(train_set, batch_size=config.batch_size, num_workers=12, shuffle=True)
    val_loader, _ = get_dataloader(val_set, batch_size=config.batch_size, num_workers=12, shuffle=False, neighborhood_limits=neighborhood_limits)
    test_loader, _ = get_dataloader(test_set, batch_size=config.batch_size, num_workers=12, shuffle=False, neighborhood_limits=neighborhood_limits)
    test_lo_loader, _ = get_dataloader(test_lo_set, batch_size=config.batch_size, num_workers=12, shuffle=False, neighborhood_limits=neighborhood_limits)
    # train_loader, neighborhood_limits = get_dataloader(train_set, batch_size=config.batch_size, num_workers=12,
    #                                                        shuffle=True)
    # val_loader, _ = get_dataloader(val_set, batch_size=config.batch_size, num_workers=12, shuffle=False,
    #                                    neighborhood_limits=neighborhood_limits)
    # test_loader, _ = get_dataloader(test_set, batch_size=config.batch_size, num_workers=12, shuffle=False,
    #                                     neighborhood_limits=neighborhood_limits)
    # test_lo_loader, _ = get_dataloader(test_lo_set, batch_size=config.batch_size, num_workers=12, shuffle=False,
    #                                        neighborhood_limits=neighborhood_limits)

    save_path = "./params/sao-no-A-only-B.pth"
    model = SAO(config, use_B=True, use_A=False)
    model.to(config.device)
    # model.load_state_dict(torch.load(save_path))
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.95,
    )

    min_val_loss, min_val_cd = 1e8, 1e8
    for epoch in range(config.max_epoch):
        num_iter = int(len(train_loader.dataset) // train_loader.batch_size)
        c_loader_iter = train_loader.__iter__()
        train_loss = 0
        model.train()
        for c_iter in range(num_iter):
            ##################################
            # load inputs to device.
            inputs = c_loader_iter.next()
            inputs = get_inputs(inputs, config)
            ##################################
            # forward pass
            trans_pred, endpoints = model(inputs, num_iter=2)
            # print(endpoints)
            # loss
            rot_gt, trans_gt = inputs['rot'], inputs['trans']
            src_pcd, tgt_pcd = inputs['src_pcd_raw'], inputs['tgt_pcd_raw']
            transform_gt = torch.cat((rot_gt, trans_gt), dim=1).unsqueeze(0)
            data = {
                'transform_gt': transform_gt,
                'points_src': src_pcd.unsqueeze(0)
            }
            loss = transform_loss(trans_pred, data, endpoints)
            train_loss += loss['total'].item()
            # backward pass
            optimizer.zero_grad()
            loss['total'].backward()
            optimizer.step()

            print(
                "\rprocess: %s  total loss: %.5f   mae 0: %.5f   mae 1: %.5f  outlier 0: "
                "%.5f  outlier 1: %.5f" % (processbar(c_iter + 1, num_iter), loss["total"], loss["mae_0"],
                          loss["mae_1"], loss["outlier_0"], loss["outlier_1"]), end="")
        train_loss /= len(train_loader.dataset)
        print("\nepoch: %d  train loss: %.5f" % (epoch+1, train_loss))
        scheduler.step()
        val_loss, val_cd = evaluate(model, val_loader)
        # save
        if min_val_cd > val_cd:
            min_val_cd = val_cd
            print("Saving ...")
            torch.save(model.state_dict(), save_path)
            print("Save finish !!!")


if __name__ == '__main__':
    # train_SAO()
    # train_SAO_use_B()
    pass
    # save_path = "./params/sao-no-A-no-B.pth"
    # config = Config
    # model = SAO(config, use_B=False, use_A=False)
    # model.to(config.device)
    # model.load_state_dict(torch.load(save_path))
    # evaluate(model, test_lo_loader)

    # ModelLoNet no remove no B
    # DeepCP
    # metrics: 5.5118(rot - rmse) | 1.8975(rot - mae) | 0.03987(trans - rmse) | 0.01608(trans - mae)
    # Rotation
    # error
    # 3.6729(deg, mean) | 9.3653(deg, rmse)
    # Translation
    # error
    # 0.03398(mean) | 0.06907(rmse)
    # Chamfer
    # error: 0.0018740(mean - sq)
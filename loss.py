import torch
from torch import nn
from common.math_torch import se3


def transform_loss(pred_transforms, data, endpoints, reduction="mean", wt_inliers=0.01):
    losses = {}
    num_iter = len(pred_transforms)

    # Compute losses
    gt_src_transformed = se3.transform(data['transform_gt'], data['points_src'][..., :3])
    # MSE loss to the groundtruth (does not take into account possible symmetries)
    criterion = nn.L1Loss(reduction=reduction)
    for i in range(num_iter):
        pred_src_transformed = se3.transform(pred_transforms[i], data['points_src'][..., :3])
        if reduction.lower() == 'mean':
            losses['mae_{}'.format(i)] = criterion(pred_src_transformed, gt_src_transformed)
        elif reduction.lower() == 'none':
            losses['mae_{}'.format(i)] = torch.mean(criterion(pred_src_transformed, gt_src_transformed),
                                                    dim=[-1, -2])
    # Penalize outliers
    for i in range(num_iter):
        ref_outliers_strength = (1.0 - torch.sum(endpoints['perm_matrices'][i], dim=1)) * wt_inliers
        src_outliers_strength = (1.0 - torch.sum(endpoints['perm_matrices'][i], dim=2)) * wt_inliers
        if reduction.lower() == 'mean':
            losses['outlier_{}'.format(i)] = torch.mean(ref_outliers_strength) + torch.mean(src_outliers_strength)
        elif reduction.lower() == 'none':
            losses['outlier_{}'.format(i)] = torch.mean(ref_outliers_strength, dim=1) + \
                                             torch.mean(src_outliers_strength, dim=1)

    discount_factor = 0.5  # Early iterations will be discounted
    total_losses = []
    for k in losses:
        discount = discount_factor ** (num_iter - int(k[k.rfind('_') + 1:]) - 1)
        total_losses.append(losses[k] * discount)
    losses['total'] = torch.sum(torch.stack(total_losses), dim=0)

    return losses
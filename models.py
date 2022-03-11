import numpy as np
import torch
from torch import nn
from kpconv.blocks import block_decider
from torch.nn import functional as F
from common.utils import _EPS
from common.utils import match_features, sinkhorn, compute_rigid_transform
from common.math_torch import se3

from common.utils import to_o3d_pcd
import open3d as o3d


class ParameterPredictionNet(nn.Module):
    def __init__(self, weights_dim):
        super().__init__()

        self.weights_dim = weights_dim

        # Pointnet
        self.prepool = nn.Sequential(
            nn.Conv1d(4, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),

            nn.Conv1d(128, 1024, 1),
            nn.GroupNorm(16, 1024),
            nn.ReLU(),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.postpool = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(),

            nn.Linear(256, 2 + np.prod(weights_dim)),
        )

    def forward(self, x):

        src_padded = F.pad(x[0], (0, 1), mode='constant', value=0)
        ref_padded = F.pad(x[1], (0, 1), mode='constant', value=1)
        concatenated = torch.cat([src_padded, ref_padded], dim=1)

        prepool_feat = self.prepool(concatenated.permute(0, 2, 1))
        pooled = torch.flatten(self.pooling(prepool_feat), start_dim=-2)
        raw_weights = self.postpool(pooled)

        beta = F.softplus(raw_weights[:, 0])
        alpha = F.softplus(raw_weights[:, 1])

        return beta, alpha


class SAOB(nn.Module):

    def __init__(self, config, use_B=False):
        super(SAOB, self).__init__()

        ############
        # Parameters
        ############
        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_feats_dim
        out_dim = config.first_feats_dim
        self.K = config.num_kernel_points
        self.epsilon = torch.nn.Parameter(torch.tensor(-5.0))
        self.final_feats_dim = config.final_feats_dim

        #####################
        # List Encoder blocks
        #####################
        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(
                block, r,
                in_dim, out_dim,
                layer, config
            ))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(
                block, r,
                in_dim, out_dim,
                layer, config
            ))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        self.use_B = use_B
        if use_B:
            self.first_layer = nn.Sequential(
                nn.Conv1d(72, 7, kernel_size=1, stride=1, bias=False),
                nn.InstanceNorm1d(7),
                nn.ReLU()
            )

        return

    def regular_score(self, score):
        score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
        score = torch.where(torch.isinf(score), torch.zeros_like(score), score)
        return score

    def forward(self, batch):
        # Get input features
        x = batch['features'].clone().detach()
        len_src_c = batch['stack_lengths'][-1][0]
        len_src_f = batch['stack_lengths'][0][0]
        pcd_c = batch['points'][-1]
        pcd_f = batch['points'][0]
        src_pcd_c, tgt_pcd_c = pcd_c[:len_src_c], pcd_c[len_src_c:]

        if self.use_B:
            x = self.first_layer(x.t().unsqueeze(0))[0].t()
            x = torch.cat([x, torch.ones(x.shape[0], 1).to(x.device)], dim=1)
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):

            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)
        for block_i, block_op in enumerate(self.decoder_blocks):

            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)
        feats_f = x
        # normalise point-wise features
        feats_f = F.normalize(feats_f, p=2, dim=1)
        # print(feats_f.shape) 1434 96

        return feats_f


## Fake !!!  only for test
class SAOA(nn.Module):
    def __init__(self, config):
        super(SAOA, self).__init__()
        self.config = config

    def forward(self, x):
        src_overlap_idx = torch.from_numpy(np.array(list(set(x['correspondences'][:, 0].cpu().numpy().tolist())))).long().to(self.config.device)
        tgt_overlap_idx = torch.from_numpy(np.array(list(set(x['correspondences'][:, 1].cpu().numpy().tolist())))).long().to(self.config.device)
        return src_overlap_idx, tgt_overlap_idx


class SAO(nn.Module):
    def __init__(self, config, use_B, use_A):
        super(SAO, self).__init__()
        if use_B:
            config.in_feats_dim = 8
            # config.in_feats_dim = 1
        else:
            config.in_feats_dim = 1
        self.sao_b = SAOB(config, use_B=use_B)
        if use_A:
            self.sao_a = SAOA(config)
        self.beta_alpha = ParameterPredictionNet(weights_dim=[0])
        self.num_sk_iter = 5
        self.add_slack = True
        self.use_B, self.use_A = use_B, use_A

    def compute_affinity(self, beta, feat_distance, alpha=0.5):
        if isinstance(alpha, float):
            hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha)
        else:
            hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha[:, None, None])
        return hybrid_affinity

    def forward(self, x, num_iter):
        len_src = x['stack_lengths'][0][0]
        # correspondence = x['correspondences']
        point_inp = [inp.detach().clone() for inp in x["points"]]
        src_pcd, tgt_pcd = x['src_pcd_raw'], x['tgt_pcd_raw']
        src_pcd_t = src_pcd.unsqueeze(0)
        src_pcd, tgt_pcd = src_pcd.unsqueeze(0), tgt_pcd.unsqueeze(0)

        endpoints = {}
        transforms = []
        all_gamma, all_perm_matrices, all_weighted_ref = [], [], []
        all_beta, all_alpha = [], []

        # if self.use_B:
        #     lrf_inp = [lrfs.detach().clone() for lrfs in x["lrfs"]]

        # print(len(x["points"]))
        # for i in range(len(x["points"])):
        #     print(x["points"][i].shape, type(x["points"][i]))
        # for i in range(len(x["stack_lengths"])):
        #     print(x["stack_lengths"][i][0], x["stack_lengths"][i][1])

        for k in range(num_iter):
            feats = self.sao_b(x)
            # False print(feats.shape)  #1434 96
            # True print(feats.shape) #1434 96
            beta, alpha = self.beta_alpha([src_pcd_t, tgt_pcd])
            src_feats, tgt_feats = feats[:len_src], feats[len_src:]
            feat_distance = match_features(src_feats.unsqueeze(0), tgt_feats.unsqueeze(0))
            affinity = self.compute_affinity(beta, feat_distance, alpha=alpha)

            # Compute weighted coordinates
            log_perm_matrix = sinkhorn(affinity, n_iters=self.num_sk_iter, slack=self.add_slack)
            perm_matrix = torch.exp(log_perm_matrix)
            if not self.use_A:
                weighted_ref = perm_matrix @ tgt_pcd / (torch.sum(perm_matrix, dim=2, keepdim=True) + _EPS)
                # Compute transform and transform points

                # a_ = weighted_ref.cpu().squeeze(0).numpy()
                # b_ = tgt_pcd.cpu().squeeze(0).numpy()
                # a, b = to_o3d_pcd(a_), to_o3d_pcd(b_)
                # a.colors = o3d.utility.Vector3dVector(np.array([[1, 0.706, 0]] * a_.shape[0]))
                # b.colors = o3d.utility.Vector3dVector(np.array([[0, 0.651, 0.929]] * b_.shape[0]))
                # o3d.visualization.draw_geometries([a, b], window_name="registration", width=1000, height=800)

                transform = compute_rigid_transform(src_pcd, weighted_ref, weights=torch.sum(perm_matrix, dim=2))
            else:
                ## remove no overlap
                src_overlap_idx, tgt_overlap_idx = self.sao_a(x)
                weighted_ref = perm_matrix[:, src_overlap_idx][:, :, tgt_overlap_idx] @ tgt_pcd[:, tgt_overlap_idx] / (torch.sum(perm_matrix[:, src_overlap_idx][:, :, tgt_overlap_idx], dim=2, keepdim=True) + _EPS)
                transform = compute_rigid_transform(src_pcd[:, src_overlap_idx], weighted_ref, weights=torch.sum(perm_matrix[:, src_overlap_idx][:, :, tgt_overlap_idx], dim=2))


            src_pcd_t = se3.transform(transform.detach(), src_pcd)
            # 要把x里的src部分的点用transform变换，从原位置变换
            for i in range(len(x["points"])):
                x["points"][i] = torch.cat([
                    se3.transform(transform.detach(), point_inp[i][:x["stack_lengths"][i][0].item()].unsqueeze(0))[0],
                    point_inp[i][x["stack_lengths"][i][0].item():]
                ], dim=0)
                # if self.use_B:
                #     x["lrfs"][i] = torch.cat([
                #         torch.matmul(
                #             lrf_inp[i][:x["stack_lengths"][i][0].item()],
                #             transform.detach()[:, :, :3].permute([0, 2, 1]).repeat([x["stack_lengths"][i][0].item(), 1, 1])
                #         ),
                #         lrf_inp[i][x["stack_lengths"][i][0].item():]
                #     ], dim=0)

            transforms.append(transform)
            all_gamma.append(torch.exp(affinity))
            all_perm_matrices.append(perm_matrix)
            all_weighted_ref.append(weighted_ref)
            all_beta.append(beta.detach().cpu().numpy())
            all_alpha.append(alpha.detach().cpu().numpy())

        endpoints['perm_matrices_init'] = all_gamma
        endpoints['perm_matrices'] = all_perm_matrices
        endpoints['weighted_ref'] = all_weighted_ref
        endpoints['beta'] = np.stack(all_beta, axis=0)
        endpoints['alpha'] = np.stack(all_alpha, axis=0)

        return transforms, endpoints


if __name__ == '__main__':
    pass
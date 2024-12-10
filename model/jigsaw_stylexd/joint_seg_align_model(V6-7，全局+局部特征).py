"""
V6-7，全局+局部特征
"""


import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn
from torch.nn.quantized.functional import threshold

from model import MatchingBaseModel, build_encoder
from utils import Sinkhorn, hungarian, stitch_indices2mat, stitch_mat2indices
from utils import get_batch_length_from_part_points, square_distance
from utils import permutation_loss
from .affinity_layer import build_affinity
from .pc_classifier_layer import build_pc_classifier
from .attention_layer import PointTransformerLayer, CrossAttentionLayer
from torch.nn import CrossEntropyLoss,MSELoss,BCELoss,BCEWithLogitsLoss
from utils import pointcloud_visualize, pointcloud_and_stitch_visualize

import numpy as np
import os

import shutil

class JointSegmentationAlignmentModel(MatchingBaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.N_point = cfg.DATA.NUM_PC_POINTS

        self.num_classes = 2

        self.aff_feat_dim = self.cfg.MODEL.AFF_FEAT_DIM
        assert self.aff_feat_dim % 2 == 0, "The affinity feature dimension must be even!"
        self.half_aff_feat_dim = self.aff_feat_dim // 2

        self.w_cls_loss = self.cfg.MODEL.LOSS.w_cls_loss
        self.w_mat_loss = self.cfg.MODEL.LOSS.w_mat_loss

        # 二分类结果的阈值
        self.pc_cls_threshold = self.cfg.MODEL.PC_CLS_THRESHOLD

        self.num_classes = self.cfg.MODEL.PC_NUM_CLS

        self.encoder = self._init_encoder()
        self.affinity_layer = self._init_affinity_layer()
        self.pc_classifier_layer = self._init_pc_classifier_layer()
        self.sinkhorn = self._init_sinkhorn()
        self.affinity_extractor = self._init_affinity_extractor()
        self.tf_self1 = PointTransformerLayer(
            in_feat=self.pc_feat_dim*2, out_feat=self.pc_feat_dim*2,
            n_heads=self.cfg.MODEL.TF_NUM_HEADS, nsampmle=self.cfg.MODEL.TF_NUM_SAMPLE,
        )
        self.tf_cross1 = CrossAttentionLayer(d_in=self.pc_feat_dim*2,
                                             n_head=self.cfg.MODEL.TF_NUM_HEADS,)
        # jigsaw中作者使用了 自注意力+交叉注意力 的PointTransformer层
        self.tf_layers = [("self", self.tf_self1), ("cross", self.tf_cross1)]

        if not self.cfg.MODEL.TEST_S_MASK:
            # default: True. The mask is not needed based on the design of the primal-dual descriptor.
            print("No mask for s in test.")

        # 分阶段学习 -----------------------------------------------------------------------------------------------------
        self.is_train_in_stages = self.cfg.MODEL.get("IS_TRAIN_IN_STAGE", False)  # 是否分阶段学习
        self.init_dynamic_adjustment()  # 分阶段学习的初始化





    def _init_encoder(self):
        in_feat_dim = 3
        encoder = build_encoder(
            self.cfg.MODEL.ENCODER,
            feat_dim=self.pc_feat_dim,
            global_feat=False,
            in_feat_dim=in_feat_dim,
        )
        return encoder


    def _init_affinity_extractor(self):
        affinity_extractor = nn.Sequential(
            nn.BatchNorm1d(self.pc_feat_dim*2),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.pc_feat_dim*2, self.aff_feat_dim, 1),
        )
        return affinity_extractor

    def _init_affinity_layer(self):
        affinity_layer = build_affinity(
            self.cfg.MODEL.AFFINITY.lower(), self.aff_feat_dim
        )
        return affinity_layer

    def _init_pc_classifier_layer(self):
        pc_classifier_layer = build_pc_classifier(
            self.pc_feat_dim*2 # [modified]
        )
        return pc_classifier_layer

    def _init_sinkhorn(self):
        return Sinkhorn(
            max_iter=self.cfg.MODEL.SINKHORN_MAXITER, tau=self.cfg.MODEL.SINKHORN_TAU
        )

    def _extract_part_feats(self, part_pcs, batch_length):
        B, N_sum, _ = part_pcs.shape  # [B, N_sum, 3]
        # shared-weight encoder
        valid_pcs = part_pcs.reshape(B * N_sum, -1)
        valid_feats = self.encoder(valid_pcs, batch_length)  # [B * N_sum, F]
        pc_feats = valid_feats.reshape(B, N_sum, -1)  # [B, N_sum, F]
        return pc_feats

    def _get_stitch_pcs_feats(self, feat, n_stitch_pcs_sum, pc_cls_mask, B_size, N_point, F_dim):
        critical_feats = torch.zeros(B_size, N_point, F_dim, device=self.device, dtype=feat.dtype)
        for b in range(B_size):
            critical_feats[b, : n_stitch_pcs_sum[b]] = feat[b, pc_cls_mask[b] == 1]
        return critical_feats

    def _get_stitch_pcs_gt_mat(self, mat_gt, pc_cls_mask, B_size,N_point,n_stitch_pcs_sum):
        stitch_pcs_gt_mat = torch.zeros((B_size,N_point,N_point),device=mat_gt.device)
        for B in range(B_size):
            stitch_pcs_gt_mat[B][:n_stitch_pcs_sum[B], :n_stitch_pcs_sum[B]] = mat_gt[B][pc_cls_mask[B] == 1][:,pc_cls_mask[B] == 1]
        return stitch_pcs_gt_mat

    def forward(self, data_dict):
        out_dict = dict()
        pcs = data_dict["pcs"]  # [B_size, N_point, 3]
        B_size, N_point, _ = pcs.shape

        n_pcs = data_dict["n_pcs"]  # [B, P]
        part_valids = data_dict["part_valids"]
        n_valid = torch.sum(part_valids, dim=1).to(torch.long)  # [B]

        piece_id = data_dict["piece_id"]
        batch_length = get_batch_length_from_part_points(n_pcs, n_valids=n_valid).to(self.device)

        # PART1:front-end Feature Extractor
        # PointNet++提取出每个顶点的特征
        pcs_feats_local = self._extract_part_feats(pcs, batch_length)
        pcs_feats_global = self._extract_part_feats(pcs, torch.tensor([N_point]*B_size))
        pcs_feats = torch.concat([pcs_feats_local,pcs_feats_global],dim=-1)
        pcs_flatten = pcs.reshape(-1, 3).contiguous()
        # 顶点特征输入到PointTransformer层中，获取点与点之间的关系
        for name, layer in self.tf_layers:
            if name == "self":
                pcs_feats = (
                    layer(
                        pcs_flatten,
                        pcs_feats.view(-1, self.pc_feat_dim*2),
                        batch_length,
                    )
                    .view(B_size, N_point, -1)
                    .contiguous()
                )
            else:
                pcs_feats = layer(pcs_feats)
        data_dict.update({"part_feats": pcs_feats})

        pc_cls = self.pc_classifier_layer(pcs_feats.transpose(1, 2)).transpose(1, 2).squeeze(-1)
        pc_cls = torch.sigmoid(pc_cls)
        pc_cls_mask = ((pc_cls>self.pc_cls_threshold) * 1)

        out_dict.update({"pc_cls": pc_cls,
                         "pc_cls_mask": pc_cls_mask,})

        # pointcloud_visualize(pcs[0][pc_cls_mask[0]==1])
        n_stitch_pcs_sum = torch.sum(pc_cls_mask, dim=-1)
        stitch_pcs_feats = self._get_stitch_pcs_feats(pcs_feats, n_stitch_pcs_sum, pc_cls_mask, B_size, N_point, self.pc_feat_dim*2)
        out_dict.update({"n_stitch_pcs_sum": n_stitch_pcs_sum,})

        affinity_feat = self.affinity_extractor(stitch_pcs_feats.permute(0, 2, 1))
        affinity_feat = affinity_feat.permute(0, 2, 1)
        affinity_feat = torch.cat(
            [
                F.normalize(
                    affinity_feat[:, :, : self.half_aff_feat_dim], p=2, dim=-1
                ),
                F.normalize(
                    affinity_feat[:, :, self.half_aff_feat_dim:], p=2, dim=-1
                ),
            ],
            dim=-1,
        )
        # 预测 Affinity matrix: s
        affinity = self.affinity_layer(affinity_feat, affinity_feat)
        B_point_num = torch.tensor([N_point]*B_size)
        out_dict.update({"B_point_num": B_point_num})
        # if (not self.training) and self.cfg.MODEL.TEST_S_MASK:
        #     mask = self.diagonal_square_mask(
        #         _affinity.shape, B_point_num, n_part=n_valid, pos_msk=1, neg_msk=0
        #     ).detach()
        #     neg_mask = self.diagonal_square_mask(
        #         _affinity.shape, B_point_num, n_part=n_valid, pos_msk=0, neg_msk=-1e6
        #     ).detach()
        #     affinity = _affinity * mask + neg_mask
        #     out_dict.update(
        #         {
        #             "s_mask": mask,
        #             "s_neg_mask": neg_mask,
        #         }
        #     )
        # else:
        #     affinity = _affinity
        # alienation_ = self.alienation_layer(torch.matmul(F.normalize(affinity_feat[:, :, : self.half_aff_feat_dim], p=2, dim=-1), self.affinity_layer.A).transpose(1,2)).transpose(1,2)

        # a = torch.max(torch.softmax(torch.pow(affinity_, 1),dim=-1),dim=-1)[0].detach().numpy()
        # b = alienation.squeeze(-1).detach().numpy()
        # c = np.concatenate([a,b],axis=0)

        # [todo] 二选一，感觉sigmoid和sinkhorn都是合理的
        # mat = torch.sigmoid(affinity)


        mat = self.sinkhorn(affinity, n_stitch_pcs_sum, n_stitch_pcs_sum)
        out_dict.update({"ds_mat": mat, })  # [B, N_, N_]

        # [todo] 计算点点匹配的，记得训练完改回去
        # perm_mat = hungarian(mat, B_point_num, B_point_num+1)
        # out_dict.update({"perm_mat": perm_mat})

        return out_dict


    def _loss_function(self, data_dict, out_dict={}, optimizer_idx=-1):
        pcs = data_dict["pcs"]
        pcs_gt = data_dict["pcs_gt"]
        B_size, N_point, _ = pcs_gt.shape
        n_stitch_pcs_sum = out_dict["n_stitch_pcs_sum"]
        part_valids = data_dict["part_valids"]
        n_pcs = data_dict["n_pcs"]

        loss_dict = {
            "batch_size": B_size,
        }

        ds_mat = out_dict["ds_mat"]  # 预测的点点匹配概率
        # 【下三角为0】所有采样点的gt缝合关系
        gt_mat = data_dict.get("mat_gt", None)
        if self.training:
            # calculate cls loss ---------------------------------------------------------------------------------------
            # 【概率】预测的点的二分类概率
            pc_cls = (out_dict.get("pc_cls", None)).squeeze(-1)
            # 【1为是缝合点】pc_cls>pc_cls_threshold得到的分类结果
            pc_cls_mask = (out_dict.get("pc_cls_mask", None)).squeeze(-1)
            pc_cls_gt = (torch.sum(gt_mat + gt_mat.transpose(-1, -2),
                                   dim=-1) == 1) * 1.0
            cls_loss = BCELoss()(pc_cls, pc_cls_gt)
            loss_dict.update({"cls_loss": cls_loss,})

            # calculate matching loss ----------------------------------------------------------------------------------
            """
               之所以让模型与预测一个尽量对称的ds_mat，而不是仅取ds_mat的上半部分与gt_mat计算损失，是因为我希望模型能真正学到通过几何关系来计算
            缝合关系，即：如果点A被预测和点B缝合，那么对B进行预测的结果也应当是A
                因此，我选择让模型去和双向的gt_mat（上三角被复制到下三角）来计算损失，从而让affinity layer学到从几何角度推出缝合关系，从而
            解决那些错误缝合。在使用模型的inference结果时，可以先将行中最大值小于阈值的行（以及对应的列）全部剔除，然后进行匈牙利算法
            """
            n_stitch_pcs_sum = n_stitch_pcs_sum.reshape(-1)
            # 【下三角为0】具有缝合关系的点，它们之间的gt缝合关系（单向缝合）
            stitch_pcs_gt_mat_half = self._get_stitch_pcs_gt_mat(gt_mat,pc_cls_mask,B_size,N_point,n_stitch_pcs_sum)
            # stitch_pcs=pcs[0][pc_cls_mask[0] == 1]
            # pointcloud_visualize(stitch_pcs)
            # pointcloud_and_stitch_visualize(stitch_pcs, stitch_mat2indices(stitch_pcs_gt_mat[0].cpu().detach().numpy()))
            # 【上下三角sum相等】双向的的缝合关系
            stitch_pcs_gt_mat = stitch_pcs_gt_mat_half + stitch_pcs_gt_mat_half.transpose(-1, -2)
            mat_loss = permutation_loss(
                ds_mat, stitch_pcs_gt_mat.float(), n_stitch_pcs_sum, n_stitch_pcs_sum
            )
            loss_dict.update(
                {
                    "mat_loss": mat_loss,
                }
            )

            # ------------------------- Following Only For Evaluation --------------------------------------------------

            # calculate stitch_dis_loss --------------------------------------------------------------------------------
            # mean distance between stitched points
            with torch.no_grad():
                # calculate mean dist between stitched points
                Dis = torch.sqrt(((pcs[:, :, None, :] - pcs[:, None, :, :]) ** 2).sum(dim=-1)) + (
                            torch.eye(pcs.shape[1])).to(pcs.device)

                for B in range(B_size):
                    Dis[B][:n_stitch_pcs_sum[B], :n_stitch_pcs_sum[B]] = Dis[B][pc_cls_mask[B] == 1][:,pc_cls_mask[B] == 1]

                stitch_dis_loss = torch.sum(torch.mul(Dis, ds_mat)) / torch.sum(n_stitch_pcs_sum)
                loss_dict.update(
                    {
                        "stitch_dis_loss": stitch_dis_loss,
                    }
                )
                # [todo] stitch_dis_loss目前只能作为一种评估的方法，不能直接放入loss
                # loss += stitch_dis_loss

            # calculate TP FP TN FN ACC TPR TNR ------------------------------------------------------------------------
            with torch.no_grad():
                B_size, N_point, _ = data_dict["pcs"].shape

                stitch_mat_ = out_dict["ds_mat"]
                mat_gt = data_dict["mat_gt"]
                pc_cls = out_dict["pc_cls"]
                threshold = self.pc_cls_threshold

                pc_cls_ = pc_cls.squeeze(-1)
                pc_cls_gt = (torch.sum(mat_gt[:, :N_point] + mat_gt[:, :N_point].transpose(-1, -2),
                                       dim=-1) == 1) * 1.0
                indices = pc_cls_ > threshold
                TP = torch.sum(torch.sum(indices[pc_cls_gt == 1] * 1))
                # print(f"TP={TP}")
                indices = pc_cls_ > threshold
                FP = torch.sum(torch.sum((indices[pc_cls_gt == 1] == False) * 1))
                # print(f"TN={FP}")
                indices = pc_cls_ < threshold
                TN = torch.sum(torch.sum(indices[pc_cls_gt == 0] * 1))
                # print(f"FP={TN}")
                indices = pc_cls_ < threshold
                FN = torch.sum(torch.sum((indices[pc_cls_gt == 0] == False) * 1))
                # print(f"FN={FN}")

                ACC = (TP + TN) / (TP + FP + TN + FN)
                # print(f"ACC={ACC:.4f}")
                TPR = TP / (TP + FN)
                # print(f"TPR={TPR:.4f}")
                TNR = TN / (FP + TN)
                # print(f"FPR={TNR:.4f}")
                PRECISION = TP / (TP + FP + 1e-5)
                loss_dict.update(
                    {
                        "pcs_1_ACC": ACC,
                        "pcs_2_TPR": TPR,
                        "pcs_3_TNR": TNR,
                        "pcs_4_PRECISION": PRECISION,
                    }
                )

            if self.is_train_in_stages:
                with torch.no_grad():
                    self.cls_loss_list.append(float(cls_loss))
                    self.mat_loss_list.append(float(mat_loss))

            # # 动态调整，阶段学习
            # if cls_loss < 0.15:
            #     self.pc_cls_threshold = 0.9
            #     self.w_mat_loss = self.cfg.MODEL.LOSS.w_mat_loss
            # elif cls_loss < 0.2:
            #     self.pc_cls_threshold = 0.8
            #     self.w_mat_loss = self.cfg.MODEL.LOSS.w_mat_loss / 2
            # else:
            #     self.pc_cls_threshold = self.cfg.MODEL.PC_CLS_THRESHOLD
            #     self.w_mat_loss = 0

            loss = (cls_loss * self.w_cls_loss+
                    mat_loss * self.w_mat_loss)
            loss_dict.update({"loss": loss,})
            return loss_dict
        else:
            raise NotImplementedError
            return loss_dict

    # 在训练开始时执行的
    def init_dynamic_adjustment(self):
        if not self.is_train_in_stages:
            return

        # self.training_stage = 0
        self.cls_loss_list = []
        self.mat_loss_list = []
        self.w_mat_loss = 0
        self.pc_cls_threshold = 0.5

    # 在一个epoce结束时动态调整超参数，来实现分阶段学习
    def training_epoch_end(self, outputs):
        self.dynamic_adjustment_epoch_end()
    def dynamic_adjustment_epoch_end(self):
        if not self.is_train_in_stages:
            return

        # 是否不可恢复：恢复指的是一个超参数可以回到上一阶段的状态
        is_unrecoverable = True

        with torch.no_grad():
            cls_loss_mean = torch.mean(torch.tensor(self.cls_loss_list))
            if cls_loss_mean < 0.08:
                new_pc_cls_threshold = self.cfg.MODEL.PC_CLS_THRESHOLD * 1.0
                new_w_mat_loss = self.cfg.MODEL.LOSS.w_mat_loss * 1.0
            elif cls_loss_mean < 0.12:
                new_pc_cls_threshold = self.cfg.MODEL.PC_CLS_THRESHOLD * 1.0
                new_w_mat_loss = self.cfg.MODEL.LOSS.w_mat_loss * 0.4
            elif cls_loss_mean < 0.16:
                new_pc_cls_threshold = self.cfg.MODEL.PC_CLS_THRESHOLD * 1.0
                new_w_mat_loss = self.cfg.MODEL.LOSS.w_mat_loss * 0.2
            else:
                new_pc_cls_threshold = 0.5
                new_w_mat_loss = self.cfg.MODEL.LOSS.w_mat_loss * 0.0

            if is_unrecoverable:
                new_w_mat_loss = max(self.w_mat_loss, new_w_mat_loss)
                new_pc_cls_threshold = max(self.pc_cls_threshold, new_pc_cls_threshold)

            self.w_mat_loss = new_w_mat_loss
            self.pc_cls_threshold = new_pc_cls_threshold

            self.cls_loss_list = []
            self.mat_loss_list = []

            self.log_dict({
                # "Charts/training_stage": self.training_stage,
                "Charts/w_cls_loss": torch.tensor(self.w_cls_loss, dtype=torch.float32),
                "Charts/w_mat_loss": torch.tensor(self.w_mat_loss, dtype=torch.float32),
            },logger=True, sync_dist=False, rank_zero_only=True)

    @torch.no_grad()
    def compute_label(self, part_pcs, nps, n_valid, label_thresholds):
        """
        Compute ground truth label of fracture points.
        :param part_pcs: all points from all pieces, [B, N_sum, 3]
        :param nps: number of points for each piece, [B, N]
        :param n_valid: number of valid parts in each object, [B]
        :param label_thresholds: threshold for ground truth label, [B, N_sum]
        :return: labels: 1 if point is a fracture point and 0 otherwise [B, N_sum]
        """
        B, N_, _ = part_pcs.shape
        dists = torch.sqrt(square_distance(part_pcs, part_pcs))
        neg_mask = self.diagonal_square_mask(
            shape=(B, N_, N_), n_pcs=nps, n_part=n_valid, pos_msk=0, neg_msk=1e6
        )
        dists = dists + neg_mask
        dists_min, _ = torch.min(dists, dim=-1)
        dists_min = dists_min.reshape(B, N_)
        labels = (dists_min < label_thresholds).to(torch.int64)
        return labels

    @torch.no_grad()
    def diagonal_square_mask(
            self, shape, n_pcs, n_part=None, pos_msk=0.0, neg_msk=1000.0
    ):
        """
        generate a mask which diagonal matrices are neg_msk and others pos_mask
        :param shape: list like [B, N_, N_], the shape of wanted mask
        :param n_pcs: [B, P] points of each part
        :param n_part: [B] number of parts of each object
        :param pos_msk: positive mask
        :param neg_msk: negative mask
        :return: msk: a matrix mask out diagonal squares with neg_msk.
        """
        # shape [B, N_, N_]
        B = n_pcs.shape[0]
        n_pcs_cumsum = torch.cumsum(n_pcs, dim=-1)  # [B, P]
        if n_part is None:
            P = n_pcs_cumsum.shape[-1]
            n_part = torch.tensor([P for _ in range(B)], dtype=torch.long)
        msk = torch.ones(shape).to(self.device) * neg_msk
        for b in range(B):
            n_p = n_part[b]
            msk[b, : n_pcs_cumsum[b, n_p - 1], : n_pcs_cumsum[b, n_p - 1]] = pos_msk
            for p in range(n_part[b]):
                st = 0 if p == 0 else n_pcs_cumsum[b, p - 1]
                ed = n_pcs_cumsum[b, p]
                msk[b, st:ed, st:ed] = neg_msk
        return msk

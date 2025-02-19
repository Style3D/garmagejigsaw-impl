from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BCELoss

from model import MatchingBaseModel, build_encoder
from .affinity_layer import build_affinity
from .pc_classifier_layer import build_pc_classifier
from .attention_layer import PointTransformerLayer, CrossAttentionLayer, PointTransformerBlock
from utils import permutation_loss
from utils import get_batch_length_from_part_points, square_distance, is_contour_OutLine, merge_c2p_byPanelIns
from utils import pointcloud_visualize, pointcloud_and_stitch_visualize
from utils import Sinkhorn, hungarian, stitch_indices2mat, stitch_mat2indices



class JointSegmentationAlignmentModel(MatchingBaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.N_point = cfg.DATA.NUM_PC_POINTS               # 点数量

        self.mat_loss_type = cfg.MODEL.LOSS.get("MAT_LOSS_TYPE", "local")
        if self.mat_loss_type not in ["local", "global"]:
            raise ValueError(f"self.mat_loss_type = {self.mat_loss_type} is wrong")
        self.w_cls_loss = self.cfg.MODEL.LOSS.w_cls_loss    # 点分类损失
        self.w_mat_loss = self.cfg.MODEL.LOSS.w_mat_loss    # 缝合损失
        self.pc_cls_threshold = self.cfg.MODEL.PC_CLS_THRESHOLD  # 二分类结果的阈值

        self.use_point_feature = cfg.MODEL.get("USE_POINT_FEATURE", True)                   # 是否提取点的特征
        self.use_local_point_feature = cfg.MODEL.get("USE_LOCAL_POINT_FEATURE", True)       # 是否提取点的局部特征
        self.use_global_point_feature = cfg.MODEL.get("USE_GLOBAL_POINT_FEATURE", True)     # 是否提取点的局部特征

        self.use_uv_feature = cfg.MODEL.get("USE_UV_FEATURE", False)                                    # 是否使用UV特征
        self.use_local_uv_feature = cfg.MODEL.get("USE_LOCAL_UV_FEATURE", self.use_uv_feature)                     # 是否提取UV的局部特征（为了兼容）
        self.use_global_uv_feature = cfg.MODEL.get("USE_GLOBAL_UV_FEATURE", False)     # 是否提取UV的局部特征

        self.pc_feat_dim = self.cfg.MODEL.get("PC_FEAT_DIM", 128)
        self.uv_feat_dim = self.cfg.MODEL.get("UV_FEAT_DIM", 128)

        # === 计算backbone提取的特征维度 ===
        self.backbone_feat_dim = 0
        if self.use_point_feature:
            if self.use_local_point_feature:
                self.backbone_feat_dim += self.pc_feat_dim
            if self.use_global_point_feature:
                self.backbone_feat_dim += self.pc_feat_dim
        if self.use_uv_feature:
            if self.use_local_uv_feature:
                self.backbone_feat_dim += self.uv_feat_dim
            if self.use_global_uv_feature:
                self.backbone_feat_dim += self.uv_feat_dim

        assert self.backbone_feat_dim!=0, "No feature will be extracted"

        if self.use_point_feature:
            self.pc_encoder = self._init_pc_encoder()
        if self.use_uv_feature:
            self.uv_encoder = self._init_uv_encoder()

        self.aff_feat_dim = self.cfg.MODEL.AFF_FEAT_DIM
        assert self.aff_feat_dim % 2 == 0, "The affinity feature dimension must be even!"
        self.half_aff_feat_dim = self.aff_feat_dim // 2

        # self.encoder = self._init_encoder()
        self.pccls_feat_dim = self.backbone_feat_dim
        self.pc_classifier_layer = self._init_pc_classifier_layer()
        self.affinity_extractor = self._init_affinity_extractor()
        self.affinity_layer = self._init_affinity_layer()
        self.sinkhorn = self._init_sinkhorn()

        self.tf_layer_num = cfg.MODEL.get("TF_LAYER_NUM", 1)
        assert self.tf_layer_num >= 0, "tf_layer_num too small"
        self.use_tf_block = cfg.MODEL.get("USE_TF_BLOCK", False)
        # === 如果不使用 PointTransformer Block === (这种方法仅能够支持 self.tf_layer_num <= 2，在层数过多的情况下会出现训练过程中的梯度骤增)
        if not self.use_tf_block:
            # [todo] 分成以下三种情况是为了兼容过去的checkpoints，将来可以考虑将这三个if整合到一起，重新训练一遍
            if self.tf_layer_num == 1:
                self.tf_self1 = PointTransformerLayer(
                    in_feat=self.backbone_feat_dim, out_feat=self.backbone_feat_dim,
                    n_heads=self.cfg.MODEL.TF_NUM_HEADS, nsampmle=self.cfg.MODEL.TF_NUM_SAMPLE,
                )
                self.tf_cross1 = CrossAttentionLayer(d_in=self.backbone_feat_dim,
                                                     n_head=self.cfg.MODEL.TF_NUM_HEADS, )
                self.tf_layers = [("self", self.tf_self1), ("cross", self.tf_cross1)]
            elif self.tf_layer_num > 1:
                # 加入ModuleList是为了让这些模型在训练开始时自动装入GPU
                self.tf_layers_ml = nn.ModuleList()
                self.tf_layers = []
                for i in range(self.tf_layer_num):
                    self_tf_layer = PointTransformerLayer(
                        in_feat=self.backbone_feat_dim,
                        out_feat=self.backbone_feat_dim,
                        n_heads=self.cfg.MODEL.TF_NUM_HEADS,
                        nsampmle=self.cfg.MODEL.TF_NUM_SAMPLE, )
                    cross_tf_layer = CrossAttentionLayer(
                        d_in=self.backbone_feat_dim,
                        n_head=self.cfg.MODEL.TF_NUM_HEADS, )
                    self.tf_layers_ml.append(self_tf_layer)
                    self.tf_layers_ml.append(cross_tf_layer)
                    self.tf_layers.append(("self", self_tf_layer))
                    self.tf_layers.append(("cross", cross_tf_layer))
            elif self.tf_layer_num == 0:
                self.tf_layers = []
        # === 如果使用 PointTransformer Block ===
        else:
            layers_name = ["self", "cross"] * self.tf_layer_num
            self.tf_layer_num = len(layers_name)
            self.tf_layers_ml = nn.ModuleList()
            self.tf_layers = []
            for i in range(self.tf_layer_num):
                tf_block = PointTransformerBlock(
                    name=layers_name[i],
                    backbone_feat_dim=self.backbone_feat_dim,
                    num_points=self.N_point,
                    n_heads=self.cfg.MODEL.TF_NUM_HEADS,
                    nsampmle=self.cfg.MODEL.TF_NUM_SAMPLE,
                )
                self.tf_layers_ml.append(tf_block)
                self.tf_layers.append(("block", tf_block))


        # 分阶段学习 -----------------------------------------------------------------------------------------------------
        self.is_train_in_stages = self.cfg.MODEL.get("IS_TRAIN_IN_STAGE", False)  # 是否分阶段学习
        self.init_dynamic_adjustment()  # 分阶段学习的初始化

    # 提取点云特征的pointnet2
    def _init_pc_encoder(self):
        in_feat_dim = 3
        encoder = build_encoder(
            self.cfg.MODEL.ENCODER,
            feat_dim=self.pc_feat_dim,
            global_feat=False,
            in_feat_dim=in_feat_dim,
        )
        return encoder
    # 提取UV特征的pointnet2
    def _init_uv_encoder(self):
        in_feat_dim = 3
        encoder = build_encoder(
            self.cfg.MODEL.ENCODER,
            feat_dim=self.uv_feat_dim,
            global_feat=False,
            in_feat_dim=in_feat_dim,
        )
        return encoder

    def _init_affinity_extractor(self):
        affinity_extractor = nn.Sequential(
            nn.BatchNorm1d(self.backbone_feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.backbone_feat_dim, self.aff_feat_dim, 1),
        )
        return affinity_extractor

    def _init_affinity_layer(self):
        affinity_layer = build_affinity(
            self.cfg.MODEL.AFFINITY.lower(), self.aff_feat_dim
        )
        return affinity_layer

    def _init_pc_classifier_layer(self):
        pc_classifier_layer = build_pc_classifier(
            self.pccls_feat_dim
        )
        return pc_classifier_layer

    def _init_sinkhorn(self):
        return Sinkhorn(
            max_iter=self.cfg.MODEL.SINKHORN_MAXITER, tau=self.cfg.MODEL.SINKHORN_TAU
        )

    def _extract_pointcloud_feats(self, part_pcs, batch_length):
        B, N_sum, _ = part_pcs.shape  # [B, N_sum, 3]
        valid_pcs = part_pcs.reshape(B * N_sum, -1)
        valid_feats = self.pc_encoder(valid_pcs, batch_length)  # [B * N_sum, F]
        pc_feats = valid_feats.reshape(B, N_sum, -1)  # [B, N_sum, F]
        return pc_feats

    def _extract_uv_feats(self, uv, batch_length):
        B, N_sum, _ = uv.shape  # [B, N_sum, 3]
        valid_uv = uv.reshape(B * N_sum, -1)
        valid_feats = self.uv_encoder(valid_uv, batch_length)  # [B * N_sum, F]
        uv_feats = valid_feats.reshape(B, N_sum, -1)  # [B, N_sum, F]
        return uv_feats

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
        # 根据 panel_instance_seg，将contours合并为panels
        data_dict = merge_c2p_byPanelIns(deepcopy(data_dict))

        pcs = data_dict.get("pcs", None)  # [B_size, N_point, 3]
        uv = data_dict.get("uv", None)
        B_size, N_point, _ = pcs.shape
        piece_id = data_dict["piece_id"]
        part_valids = data_dict["part_valids"]
        n_valid = torch.sum(part_valids, dim=1).to(torch.long)  # [B]
        panel_instance_seg = data_dict["panel_instance_seg"]
        # n_pcs_ [B, C] 当前为“每个contour有多少点，需要进行合并”
        n_pcs = data_dict["n_pcs"]

        # # n_pcs [B, P] 每个panel有多少点
        # n_pcs = torch.zeros_like(n_pcs_)
        # for B in range(B_size):
        #     for contour_idx, num in enumerate(n_pcs_[B]):
        #         panel_idx = panel_instance_seg[B][contour_idx]
        #         # if not is_contour_OutLine(contour_idx, panel_instance_seg[B])[0]:
        #         # n_pcs[B][panel_idx] += num

        batch_length = get_batch_length_from_part_points(n_pcs, n_valids=n_valid).to(self.device)
        # === 用PointNet从点云或UV中提取特征，并拼接 ===
        features = []
        # [todo] 根据panel_instance_seg将属于同一个panel的点云（在 n_pcs 中）进行合并
        if self.use_point_feature:
            if self.use_local_point_feature:
                local_pcs_feats  = self._extract_pointcloud_feats(pcs, batch_length)
                features.append(local_pcs_feats)
            if self.use_global_point_feature:
                pcs_feats_global = self._extract_pointcloud_feats(pcs, torch.tensor([N_point]*B_size))
                features.append(pcs_feats_global)
        if self.use_uv_feature:
            if self.use_local_uv_feature:
                uv_feats = self._extract_uv_feats(uv.to(torch.float32), batch_length)
                features.append(uv_feats)
            if self.use_global_uv_feature:
                uv_feats = self._extract_uv_feats(uv.to(torch.float32), torch.tensor([N_point]*B_size))
                features.append(uv_feats)
        assert len(features)>0, "None feature extracted!"
        features = torch.concat(features,dim=-1)

        # === 提取出的特征输入到PointTransformer Layers\Blocks ===
        pcs_flatten = pcs.reshape(-1, 3).contiguous()
        # 顶点特征输入到PointTransformer层中，获取点与点之间的关系
        for name, layer in self.tf_layers:
            # 如果是自注意力层
            if name == "self":
                features = (
                    layer(
                        pcs_flatten,
                        features.view(-1, self.backbone_feat_dim),
                        batch_length,
                    ).view(B_size, N_point, -1).contiguous()
                )
            # 如果是交叉注意力层
            elif name == "cross":
                features = layer(features)
            # 如果是被封装成块了
            elif name == "block" and self.use_tf_block:
                features = (
                    layer(
                        pcs_flatten,
                        features,
                        batch_length,
                        B_size,
                        N_point
                    )
                )

        # 预测点分类
        pc_cls = self.pc_classifier_layer(features.transpose(1, 2)).transpose(1, 2).squeeze(-1)
        pc_cls = torch.sigmoid(pc_cls)
        pc_cls_mask = ((pc_cls>self.pc_cls_threshold) * 1)
        out_dict.update({"pc_cls": pc_cls,
                         "pc_cls_mask": pc_cls_mask,})

        # === 预测点点缝合关系 ===
        # pointcloud_visualize(pcs[0][pc_cls_mask[0]==1])
        n_stitch_pcs_sum = torch.sum(pc_cls_mask, dim=-1)

        stitch_pcs_feats = self._get_stitch_pcs_feats(features, n_stitch_pcs_sum, pc_cls_mask, B_size, N_point, self.backbone_feat_dim)
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

        mat = self.sinkhorn(affinity, n_stitch_pcs_sum, n_stitch_pcs_sum)
        out_dict.update({"ds_mat": mat, })  # [B, N_, N_]

        return out_dict


    def _loss_function(self, data_dict, out_dict={}, optimizer_idx=-1):
        pcs = data_dict["pcs"]
        # pcs_gt = data_dict["pcs_gt"]
        B_size, N_point, _ = pcs.shape
        n_stitch_pcs_sum = out_dict["n_stitch_pcs_sum"]

        loss_dict = {
            "batch_size": B_size,
        }

        ds_mat = out_dict["ds_mat"]  # 预测的点点匹配概率
        # 【下三角为0】所有采样点的gt缝合关系
        gt_mat = data_dict.get("mat_gt", None)
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
        【gt_mat是单向的】
        
        之所以让模型与预测一个尽量对称的ds_mat，而不是仅取ds_mat的上半部分与gt_mat计算损失，是因为我希望模型能真正学到通过几何关系来计算
        缝合关系，即：如果点A被预测和点B缝合，那么对B进行预测的结果也应当是A
        
        因此，我选择让模型去和双向的gt_mat（上三角被复制到下三角）来计算损失，从而让affinity layer学到从几何角度推出缝合关系，从而
        解决那些错误缝合。在使用模型的inference结果时，可以先将行中最大值小于阈值的行（以及对应的列）全部剔除，然后进行匈牙利算法
        """

        if self.mat_loss_type=="local":
            n_stitch_pcs_sum = n_stitch_pcs_sum.reshape(-1)

            # 获取 对称的 gt_mat
            stitch_pcs_gt_mat_half = self._get_stitch_pcs_gt_mat(gt_mat, pc_cls_mask, B_size, N_point, n_stitch_pcs_sum)
            stitch_pcs_gt_mat = stitch_pcs_gt_mat_half + stitch_pcs_gt_mat_half.transpose(-1, -2)
            mat_loss = permutation_loss(
                ds_mat, stitch_pcs_gt_mat.float(), n_stitch_pcs_sum, n_stitch_pcs_sum
            )
        elif self.mat_loss_type=="global":
            n_stitch_pcs_sum = n_stitch_pcs_sum.reshape(-1)

            # 获取 global的 ds_mat
            ds_mat_global = torch.zeros((B_size,N_point,N_point), device=ds_mat.device)
            for B in range(B_size):
                mask = pc_cls_mask[B] == 1
                indices = torch.where(mask)[0]
                ds_mat_global[B].index_put_((indices[:, None], indices), ds_mat[B][:n_stitch_pcs_sum[B], :n_stitch_pcs_sum[B]])

            # 获取 对称的 gt_mat
            stitch_pcs_gt_mat_half = gt_mat
            stitch_pcs_gt_mat = stitch_pcs_gt_mat_half + stitch_pcs_gt_mat_half.transpose(-1, -2)

            n_pcs_sum = torch.ones((B_size), device=pcs.device, dtype=torch.int64) * N_point
            mat_loss = permutation_loss(
                ds_mat_global, stitch_pcs_gt_mat.float(), n_pcs_sum, n_pcs_sum
            )
        else:
            raise  NotImplementedError(f"self.mat_loss_type={self.mat_loss_type}")
        # stitch_pcs=pcs[0][pc_cls_mask[0] == 1]
        # pointcloud_visualize(stitch_pcs)
        # pointcloud_and_stitch_visualize(stitch_pcs, stitch_mat2indices(stitch_pcs_gt_mat[0].cpu().detach().numpy()))
        # 【上下三角sum相等】双向的的缝合关系

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

        # if self.is_train_in_stages and self.training:
        #     with torch.no_grad():
        #         self.cls_loss_list.append(float(cls_loss))
        #         self.mat_loss_list.append(float(mat_loss))

        if self.is_train_in_stages and self.trainer.validating:
            with torch.no_grad():
                self.val_ACC_list.append(float(ACC))


        loss = (cls_loss * self.w_cls_loss+
                mat_loss * self.w_mat_loss)
        loss_dict.update({"loss": loss,})
        return loss_dict

    # 在训练开始时执行的
    def init_dynamic_adjustment(self):
        if not self.is_train_in_stages:
            return

        # self.training_stage = 0
        self.cls_loss_list = []
        self.mat_loss_list = []
        self.val_ACC_list = []
        self.w_mat_loss = 0
        self.pc_cls_threshold = 0.5

    # # 在一个epoce结束时动态调整超参数，来实现分阶段学习
    # def training_epoch_end(self, outputs):
    #     super().training_epoch_end(outputs)
    #     self.dynamic_adjustment_epoch_end()

    def validation_epoch_end(self, outputs):
        super().training_epoch_end(outputs)
        self.dynamic_adjustment_epoch_end()

    def dynamic_adjustment_epoch_end(self):
        if not self.is_train_in_stages:
            return

        # 是否不可恢复：恢复指的是一个超参数可以回到上一阶段的状态
        is_unrecoverable = True

        with torch.no_grad():
            # cls_loss_mean = torch.mean(torch.tensor(self.cls_loss_list))
            ACC_mean = torch.mean(torch.tensor(self.val_ACC_list))
            print(ACC_mean)
            print(self.cfg.MODEL.TRAIN_IN_STAGE.VAL_ACC)
            # if cls_loss_mean < self.cfg.MODEL.TRAIN_IN_STAGE.W_CLS:
            if ACC_mean > self.cfg.MODEL.TRAIN_IN_STAGE.VAL_ACC:
                new_pc_cls_threshold = self.cfg.MODEL.PC_CLS_THRESHOLD * 1.0
                new_w_mat_loss = self.cfg.MODEL.LOSS.w_mat_loss * 1.0
                # 将之冻结
                for param in self.pc_classifier_layer.parameters():
                    param.requires_grad = False
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
            self.val_ACC_list = []

            self.log_dict({
                # "Charts/training_stage": self.training_stage,
                "Charts/w_cls_loss": torch.tensor(self.w_cls_loss, dtype=torch.float32),
                "Charts/w_mat_loss": torch.tensor(self.w_mat_loss, dtype=torch.float32),
                # "train/cls_loss_mean": cls_loss_mean,
                "VAL/ACC_mean": ACC_mean,
            },logger=True, sync_dist=False, rank_zero_only=True)

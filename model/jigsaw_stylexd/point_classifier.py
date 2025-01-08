
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BCELoss

from model import MatchingBaseModel, build_encoder
from .affinity_layer import build_affinity
from .pc_classifier_layer import build_pc_classifier
from .attention_layer import PointTransformerLayer, CrossAttentionLayer, PointTransformerBlock
from utils import permutation_loss
from utils import get_batch_length_from_part_points, square_distance
from utils import pointcloud_visualize, pointcloud_and_stitch_visualize
from utils import Sinkhorn, hungarian, stitch_indices2mat, stitch_mat2indices



class PointClassifier(MatchingBaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.N_point = cfg.DATA.NUM_PC_POINTS               # 点数量
        self.w_cls_loss = self.cfg.MODEL.LOSS.w_cls_loss    # 点分类损失
        self.pc_cls_threshold = self.cfg.MODEL.PC_CLS_THRESHOLD  # 二分类结果的阈值

        self.use_point_feature = cfg.MODEL.get("USE_POINT_FEATURE", True)                   # 是否提取点的特征
        self.use_local_point_feature = cfg.MODEL.get("USE_LOCAL_POINT_FEATURE", True)       # 是否提取点的局部特征
        self.use_global_point_feature = cfg.MODEL.get("USE_GLOBAL_POINT_FEATURE", True)     # 是否提取点的局部特征

        self.use_uv_feature = cfg.MODEL.get("USE_UV_FEATURE", False)    # 是否使用UV特征

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
            self.backbone_feat_dim += self.uv_feat_dim
        assert self.backbone_feat_dim!=0, "No feature will be extracted"

        if self.use_point_feature:
            self.pc_encoder = self._init_pc_encoder()
        if self.use_uv_feature:
            self.uv_encoder = self._init_uv_encoder()

        # self.encoder = self._init_encoder()
        self.pccls_feat_dim = self.backbone_feat_dim
        self.pc_classifier_layer = self._init_pc_classifier_layer()
        self.tf_layer_num = cfg.MODEL.POINTCLASSIFIER.get("TF_LAYER_NUM", 1)
        assert self.tf_layer_num >= 0, "tf_layer_num too small"
        self.use_tf_block = cfg.MODEL.POINTCLASSIFIER.get("USE_TF_BLOCK", False)
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

    def _init_pc_classifier_layer(self):
        pc_classifier_layer = build_pc_classifier(
            self.pccls_feat_dim
        )
        return pc_classifier_layer

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

    def forward(self, data_dict):
        out_dict = dict()

        pcs = data_dict.get("pcs", None)  # [B_size, N_point, 3]
        uv = data_dict.get("uv", None)
        piece_id = data_dict["piece_id"]
        n_pcs = data_dict["n_pcs"]  # [B, P]
        part_valids = data_dict["part_valids"]
        n_valid = torch.sum(part_valids, dim=1).to(torch.long)  # [B]

        B_size, N_point, _ = pcs.shape

        batch_length = get_batch_length_from_part_points(n_pcs, n_valids=n_valid).to(self.device)
        # === 用PointNet从点云或UV中提取特征，并拼接 ===
        features = []
        if self.use_point_feature:
            if self.use_local_point_feature:
                local_pcs_feats  = self._extract_pointcloud_feats(pcs, batch_length)
                features.append(local_pcs_feats)
            if self.use_global_point_feature:
                pcs_feats_global = self._extract_pointcloud_feats(pcs, torch.tensor([N_point]*B_size))
                features.append(pcs_feats_global)
        if self.use_uv_feature:
            uv_feats = self._extract_uv_feats(uv.to(torch.float32), batch_length)
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
                features = layer(pcs_flatten, features, batch_length, B_size, N_point)

        # 预测点分类
        pc_cls = self.pc_classifier_layer(features.transpose(1, 2)).transpose(1, 2).squeeze(-1)
        pc_cls = torch.sigmoid(pc_cls)
        pc_cls_mask = ((pc_cls>self.pc_cls_threshold) * 1)
        out_dict.update({"pc_cls": pc_cls,
                         "pc_cls_mask": pc_cls_mask,
                         "features": features})

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

        # ------------------------- Following Only For Evaluation --------------------------------------------------
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

        loss = cls_loss * self.w_cls_loss
        loss_dict.update({"loss": loss,})
        return loss_dict
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BCELoss

from model import BaseModel, build_encoder
from .affinity_layer import build_affinity
from .pc_classifier_layer import build_pc_classifier
from .attention_layer import PointTransformerLayer, CrossAttentionLayer, PointTransformerBlock
from .feature_conv_layer import feature_conv_layer_contourwise
from utils import Sinkhorn
from utils import permutation_loss
from utils import get_batch_length_from_part_points, merge_c2p_byPanelIns


class GarmageJigsawModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Number of points
        self.N_point = cfg.DATA.NUM_PC_POINTS

        self.mat_loss_type = cfg.MODEL.LOSS.get("MAT_LOSS_TYPE", "local")
        if self.mat_loss_type not in ["local", "global"]:
            raise ValueError(f"self.mat_loss_type = {self.mat_loss_type} is wrong")
        self.w_cls_loss = self.cfg.MODEL.LOSS.w_cls_loss    # Point classification loss
        self.w_mat_loss = self.cfg.MODEL.LOSS.w_mat_loss    # Stitching loss
        self.cal_mat_loss_sym = self.cfg.MODEL.LOSS.get("MAT_LOSS_SYM", True)   # Calculate loss using skew-symmetric GT stitching matrix
        self.pc_cls_threshold = self.cfg.MODEL.PC_CLS_THRESHOLD  # Threshold for binary classification results

        self.use_point_feature = cfg.MODEL.get("USE_POINT_FEATURE", True)                   # Whether to extract point features
        self.use_local_point_feature = cfg.MODEL.get("USE_LOCAL_POINT_FEATURE", True)       # Whether to extract local point features
        self.use_global_point_feature = cfg.MODEL.get("USE_GLOBAL_POINT_FEATURE", True)     # Whether to extract global point features

        self.use_uv_feature = cfg.MODEL.get("USE_UV_FEATURE", False)                                     # Whether to use UV features
        self.use_local_uv_feature = cfg.MODEL.get("USE_LOCAL_UV_FEATURE", self.use_uv_feature)           # Whether to extract local UV features (for compatibility)
        self.use_global_uv_feature = cfg.MODEL.get("USE_GLOBAL_UV_FEATURE", False)                       # Whether to extract global UV features

        self.pc_feat_dim = self.cfg.MODEL.get("PC_FEAT_DIM", 128)
        self.uv_feat_dim = self.cfg.MODEL.get("UV_FEAT_DIM", 128)

        # === cal feature dim ===
        self.backbone_feat_dim = 0
        self.in_dim_sum = 0
        if self.use_point_feature:
            if self.use_local_point_feature:
                self.backbone_feat_dim += self.pc_feat_dim
                self.in_dim_sum+=3
            if self.use_global_point_feature:
                self.backbone_feat_dim += self.pc_feat_dim
        if self.use_uv_feature:
            if self.use_local_uv_feature:
                self.backbone_feat_dim += self.uv_feat_dim
                self.in_dim_sum+=2
            if self.use_global_uv_feature:
                self.backbone_feat_dim += self.uv_feat_dim
        assert self.backbone_feat_dim!=0, "No feature will be extracted"
        # Whether to use a shared encoder to extract all features
        # (assuming the input includes point cloud, UV, and normals concatenated together)
        self.encode_all_once = self.cfg.MODEL.get("ENCODE_ALL_ONCE", False)
        if self.encode_all_once:
            assert not self.use_global_point_feature
            assert not self.use_global_uv_feature
        if not self.encode_all_once:
            if self.use_point_feature:
                self.pc_encoder = self._init_pc_encoder()
            if self.use_uv_feature:
                self.uv_encoder = self._init_uv_encoder()
        else:
            self.all_encoder = self._init_all_encoder(in_feat_dim=self.in_dim_sum, feat_dim=self.backbone_feat_dim)

        # === feature conv ===
        """
        feature conv before PointTransformer
        """
        self.use_feature_conv = self.cfg.MODEL.get("FEATURE_CONV", {}).get("USE_FEATURE_CONV", False)
        if self.use_feature_conv:
            feature_conv_ks = self.cfg.MODEL.get("FEATURE_CONV", {}).get("KERNEL_SIZE", 3)
            feature_conv_dilation = self.cfg.MODEL.get("FEATURE_CONV", {}).get("DILATION", 1)
            feature_conv_type = self.cfg.MODEL.get("FEATURE_CONV", {}).get("TYPE", "default")
            self.feature_conv = feature_conv_layer_contourwise(
                in_channels=self.backbone_feat_dim,
                out_channels=self.backbone_feat_dim,
                type = feature_conv_type,
                kernel_size=feature_conv_ks,
                dilation=feature_conv_dilation,
            )
        """
        feature conv after PointTransformer
        """
        self.use_feature_conv_2 = self.cfg.MODEL.get("FEATURE_CONV_2", {}).get("USE_FEATURE_CONV", False)
        if self.use_feature_conv_2:
            feature_conv_ks = self.cfg.MODEL.get("FEATURE_CONV_2", {}).get("KERNEL_SIZE", 3)
            feature_conv_dilation = self.cfg.MODEL.get("FEATURE_CONV_2", {}).get("DILATION", 1)
            feature_conv_type = self.cfg.MODEL.get("FEATURE_CONV_2", {}).get("TYPE", "default")
            self.feature_conv_2 = feature_conv_layer_contourwise(
                in_channels=self.backbone_feat_dim,
                out_channels=self.backbone_feat_dim,
                type = feature_conv_type,
                kernel_size=feature_conv_ks,
                dilation=feature_conv_dilation,
            )

        self.aff_feat_dim = self.cfg.MODEL.AFF_FEAT_DIM
        assert self.aff_feat_dim % 2 == 0, "The affinity feature dimension must be even!"
        self.half_aff_feat_dim = self.aff_feat_dim // 2

        self.pccls_feat_dim = self.backbone_feat_dim
        self.pc_classifier_layer = self._init_pc_classifier_layer()
        self.affinity_extractor = self._init_affinity_extractor()
        self.affinity_layer = self._init_affinity_layer()
        self.sinkhorn = self._init_sinkhorn()

        # === PointTransformer ===
        self.tf_layer_num = cfg.MODEL.get("TF_LAYER_NUM", 1)
        assert self.tf_layer_num >= 0, "tf_layer_num too small"
        self.use_tf_block = cfg.MODEL.get("USE_TF_BLOCK", False)
        if not self.use_tf_block:
            if self.tf_layer_num == 1:
                self.tf_self1 = PointTransformerLayer(
                    in_feat=self.backbone_feat_dim, out_feat=self.backbone_feat_dim,
                    n_heads=self.cfg.MODEL.TF_NUM_HEADS, nsampmle=self.cfg.MODEL.TF_NUM_SAMPLE,
                )
                self.tf_cross1 = CrossAttentionLayer(d_in=self.backbone_feat_dim,
                                                     n_head=self.cfg.MODEL.TF_NUM_HEADS, )
                self.tf_layers = [("self", self.tf_self1), ("cross", self.tf_cross1)]
            elif self.tf_layer_num > 1:
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
        # === Use PointTransformer Block ===
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


    def _init_pc_encoder(self):
        # PointNet2 for point cloud feature extraction
        in_feat_dim = 3
        encoder = build_encoder(
            self.cfg.MODEL.ENCODER,
            feat_dim=self.pc_feat_dim,
            global_feat=False,
            in_feat_dim=in_feat_dim,
        )
        return encoder

    def _init_uv_encoder(self):
        # PointNet2 for point cloud feature extraction
        in_feat_dim = 3
        encoder = build_encoder(
            self.cfg.MODEL.ENCODER,
            feat_dim=self.uv_feat_dim,
            global_feat=False,
            in_feat_dim=in_feat_dim,
        )
        return encoder

    def _init_all_encoder(self, in_feat_dim, feat_dim):
        # PointNet2 for point cloud feature extraction
        encoder = build_encoder(
            self.cfg.MODEL.ENCODER,
            feat_dim=feat_dim,
            global_feat=False,
            in_feat_dim=in_feat_dim,
        )
        return encoder

    def _init_affinity_extractor(self):
        norm = self.cfg.MODEL.STITCHPREDICTOR.get("NORM", "batch")

        assert norm in ["batch", "instance"]

        def get_norm(norm_type, dim):
            if norm_type == "batch":
                return nn.BatchNorm1d(dim)
            elif norm_type == "instance":
                return nn.InstanceNorm1d(dim)
            else:
                raise ValueError(f"Unsupported norm type: {norm_type}")

        affinity_extractor = nn.Sequential(
            get_norm(norm, self.backbone_feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.backbone_feat_dim, self.aff_feat_dim, 1),
        )
        return affinity_extractor

    def _init_affinity_layer(self):
        affinity_layer = build_affinity(
            self.cfg.MODEL.AFFINITY.lower(),
            self.aff_feat_dim,
            norm=self.cfg.MODEL.POINTCLASSIFIER.get("NORM", "batch"),
        )
        return affinity_layer

    def _init_pc_classifier_layer(self):
        pc_classifier_layer = build_pc_classifier(
            self.pccls_feat_dim,
            norm=self.cfg.MODEL.POINTCLASSIFIER.get("NORM", "batch"),
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

    def _extract_all_feats(self, pcs_list:list, batch_length):
        pcs_all = torch.cat(pcs_list, dim=-1)
        B, N_sum, _ = pcs_all.shape  # [B, N_sum, 3]
        valid_pcs = pcs_all.reshape(B * N_sum, -1)
        valid_feats = self.all_encoder(valid_pcs.to(torch.float32), batch_length)  # [B * N_sum, F]
        feats = valid_feats.reshape(B, N_sum, -1)  # [B, N_sum, F]
        return feats

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

    def forward(self, data_dict, pc_cls_mask_predefine=None):
        out_dict = dict()

        # Merge contours into panels according to panel_instance_seg (to extract features more effectively)
        data_dict = merge_c2p_byPanelIns(deepcopy(data_dict))

        pcs = data_dict.get("pcs", None)  # [B_size, N_point, 3]
        uv = data_dict.get("uv", None)
        B_size, N_point, _ = pcs.shape
        piece_id = data_dict["piece_id"]
        part_valids = data_dict["part_valids"]
        n_valid = torch.sum(part_valids, dim=1).to(torch.long)  # [B]
        # panel_instance_seg = data_dict["panel_instance_seg"]
        n_pcs = data_dict["n_pcs"]
        num_parts = data_dict["num_parts"]
        # contour_n_pcs = data_dict["contour_n_pcs"]
        # num_contours = data_dict["num_contours"]

        batch_length = get_batch_length_from_part_points(n_pcs, n_valids=n_valid).to(self.device)

        # === Extract features from point cloud or UV using PointNet and concatenate them ===
        if not self.encode_all_once:
            features = []
            if self.use_point_feature:
                if self.use_local_point_feature:
                    local_pcs_feats = self._extract_pointcloud_feats(pcs, batch_length)
                    features.append(local_pcs_feats)
                if self.use_global_point_feature:
                    pcs_feats_global = self._extract_pointcloud_feats(pcs, torch.tensor([N_point] * B_size))
                    features.append(pcs_feats_global)
            if self.use_uv_feature:
                if self.use_local_uv_feature:
                    uv_feats = self._extract_uv_feats(uv.to(torch.float32), batch_length)
                    features.append(uv_feats)
                if self.use_global_uv_feature:
                    uv_feats = self._extract_uv_feats(uv.to(torch.float32), torch.tensor([N_point] * B_size))
                    features.append(uv_feats)
            assert len(features) > 0, "None feature extracted!"
            features = torch.concat(features, dim=-1)
        else:
            # Use same encoder to extract all features
            pcs_list = []
            if self.use_point_feature:
                if self.use_local_point_feature:
                    pcs_list.append(pcs)
            if self.use_uv_feature:
                if self.use_local_uv_feature:
                    pcs_list.append(uv[..., :-1])
            features = self._extract_all_feats(pcs_list, batch_length)

        # === Apply 1D convolution for each panel (before TF) ===
        if self.use_feature_conv:
            features_list = []

            for b in range(B_size):
                n_parts = num_parts[b]
                n_pcs_part = n_pcs[b][:n_parts]
                n_pcs_part_cumsum = torch.cumsum(n_pcs_part, dim=-1)

                features_b = []
                for i in range(len(n_pcs_part_cumsum)):
                    st = 0 if i == 0 else n_pcs_part_cumsum[i - 1]
                    ed = n_pcs_part_cumsum[i]
                    part_feature = self.feature_conv(features[b][st:ed])  # shape: (N_i, D)
                    features_b.append(part_feature)
                features_list.append(torch.cat(features_b, dim=0))  # shape: (N_b, D)
            features = torch.stack(features_list, dim=0)

        # === PointTransformer Layers\Blocks ===
        pcs_flatten = pcs.reshape(-1, 3).contiguous()
        for name, layer in self.tf_layers:
            if name == "self":
                features = (
                    layer(
                        pcs_flatten,
                        features.view(-1, self.backbone_feat_dim),
                        batch_length,
                    ).view(B_size, N_point, -1).contiguous()
                )
            elif name == "cross":
                features = layer(features)
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

        # === Apply 1D convolution for each panel (after TF) ===
        if self.use_feature_conv_2:
            features_list = []

            for b in range(B_size):
                n_parts = num_parts[b]
                n_pcs_part = n_pcs[b][:n_parts]
                n_pcs_part_cumsum = torch.cumsum(n_pcs_part, dim=-1)

                features_b = []
                for i in range(len(n_pcs_part_cumsum)):
                    st = 0 if i == 0 else n_pcs_part_cumsum[i - 1]
                    ed = n_pcs_part_cumsum[i]
                    part_feature = self.feature_conv_2(features[b][st:ed])  # shape: (N_i, D)
                    features_b.append(part_feature)
                features_list.append(torch.cat(features_b, dim=0))  # shape: (N_b, D)
            features = torch.stack(features_list, dim=0)
        out_dict.update({"features": features})

        # === pointcloud classification prediction ===
        pc_cls = self.pc_classifier_layer(features.transpose(1, 2)).transpose(1, 2).squeeze(-1)
        pc_cls = torch.sigmoid(pc_cls)
        pc_cls_mask = ((pc_cls>self.pc_cls_threshold) * 1)

        if pc_cls_mask_predefine is not None:
            pc_cls_mask = pc_cls_mask_predefine

        out_dict.update({"pc_cls": pc_cls,
                         "pc_cls_mask": pc_cls_mask,})

        # === point-stitch prediction ===
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

        # Predicted point-to-point matching probability
        ds_mat = out_dict["ds_mat"]
        # Ground truth stitching relationships of all sampled points (Lower triangular part is zero)
        gt_mat = data_dict.get("mat_gt", None)

        # === calculate cls loss ===
        # Predicted binary classification probability of points.
        pc_cls = (out_dict.get("pc_cls", None)).squeeze(-1)
        # Classification result obtained by pc_cls > pc_cls_threshold.
        pc_cls_mask = (out_dict.get("pc_cls_mask", None)).squeeze(-1)
        pc_cls_gt = (torch.sum(gt_mat + gt_mat.transpose(-1, -2),  dim=-1) == 1) * 1.0
        cls_loss = BCELoss()(pc_cls, pc_cls_gt)
        loss_dict.update({"cls_loss": cls_loss,})

        # === calculate matching loss ===
        # Calculate loss only on the matrix composed of predicted stitching points.
        if self.mat_loss_type=="local":
            n_stitch_pcs_sum = n_stitch_pcs_sum.reshape(-1)
            stitch_pcs_gt_mat_half = self._get_stitch_pcs_gt_mat(gt_mat, pc_cls_mask, B_size, N_point, n_stitch_pcs_sum)
            if self.cal_mat_loss_sym:
                stitch_pcs_gt_mat = stitch_pcs_gt_mat_half + stitch_pcs_gt_mat_half.transpose(-1, -2)
            else:
                stitch_pcs_gt_mat = stitch_pcs_gt_mat_half
            mat_loss = permutation_loss(
                ds_mat, stitch_pcs_gt_mat.float(), n_stitch_pcs_sum, n_stitch_pcs_sum
            )
        # Calculate loss on the matrix composed of all points.
        elif self.mat_loss_type=="global":
            n_stitch_pcs_sum = n_stitch_pcs_sum.reshape(-1)
            ds_mat_global = torch.zeros((B_size,N_point,N_point), device=ds_mat.device)
            for B in range(B_size):
                mask = pc_cls_mask[B] == 1
                indices = torch.where(mask)[0]
                ds_mat_global[B].index_put_((indices[:, None], indices), ds_mat[B][:n_stitch_pcs_sum[B], :n_stitch_pcs_sum[B]])
            stitch_pcs_gt_mat_half = gt_mat

            if self.cal_mat_loss_sym:
                stitch_pcs_gt_mat = stitch_pcs_gt_mat_half + stitch_pcs_gt_mat_half.transpose(-1, -2)
            else:
                stitch_pcs_gt_mat = stitch_pcs_gt_mat_half
            n_pcs_sum = torch.ones((B_size), device=pcs.device, dtype=torch.int64) * N_point
            mat_loss = permutation_loss(
                ds_mat_global, stitch_pcs_gt_mat.float(), n_pcs_sum, n_pcs_sum
            )
        else:
            raise  NotImplementedError(f"self.mat_loss_type={self.mat_loss_type}")
        loss_dict.update({"mat_loss": mat_loss,})

        # === total Loss ===
        loss = (cls_loss * self.w_cls_loss+
                mat_loss * self.w_mat_loss)
        loss_dict.update({"loss": loss,})

        # === Evaluation metrics ===
        # mean stitching distance
        with torch.no_grad():
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

        # calculate ACC TPR TNR PRECISION
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
            indices = pc_cls_ > threshold
            FP = torch.sum(torch.sum((indices[pc_cls_gt == 1] == False) * 1))
            indices = pc_cls_ < threshold
            TN = torch.sum(torch.sum(indices[pc_cls_gt == 0] * 1))
            indices = pc_cls_ < threshold
            FN = torch.sum(torch.sum((indices[pc_cls_gt == 0] == False) * 1))

            ACC = (TP + TN) / (TP + FP + TN + FN)
            TPR = TP / (TP + FN)
            TNR = TN / (FP + TN)
            PRECISION = TP / (TP + FP + 1e-5)
            loss_dict.update(
                {
                    "pcs_1_ACC": ACC,
                    "pcs_2_TPR": TPR,
                    "pcs_3_TNR": TNR,
                    "pcs_4_PRECISION": PRECISION,
                }
            )

        return loss_dict

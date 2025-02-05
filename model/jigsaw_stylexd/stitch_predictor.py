
import torch
from torch import nn
import torch.nn.functional as F

from model import MatchingBaseModel
from .affinity_layer import build_affinity
from .attention_layer import PointTransformerLayer, CrossAttentionLayer, PointTransformerBlock
from utils import permutation_loss
from utils import get_batch_length_from_part_points
from utils import Sinkhorn

from utils import pointcloud_visualize, pointcloud_and_stitch_visualize


class StitchPredictor(MatchingBaseModel):
    def __init__(self, cfg, point_classifier):
        super().__init__(cfg)
        self.N_point = cfg.DATA.NUM_PC_POINTS               # 点数量
        self.w_mat_loss = self.cfg.MODEL.LOSS.w_mat_loss    # 缝合损失
        self.pc_cls_threshold = self.cfg.MODEL.PC_CLS_THRESHOLD  # 二分类结果的阈值

        self.point_classifier = point_classifier

        self.backbone_feat_dim = point_classifier.backbone_feat_dim

        self.aff_feat_dim = self.cfg.MODEL.AFF_FEAT_DIM
        assert self.aff_feat_dim % 2 == 0, "The affinity feature dimension must be even!"
        self.half_aff_feat_dim = self.aff_feat_dim // 2

        self._init_pointtransformer(cfg)
        self.affinity_extractor = self._init_affinity_extractor()
        self.affinity_layer = self._init_affinity_layer()
        self.sinkhorn = self._init_sinkhorn()


    def _init_pointtransformer(self, cfg):
        self.tf_layer_num = cfg.MODEL.POINTCLASSIFIER.get("TF_LAYER_NUM", 1)
        assert self.tf_layer_num >= 0, "tf_layer_num too small"
        self.use_tf_block = cfg.MODEL.POINTCLASSIFIER.get("USE_TF_BLOCK", False)
        # === 如果不使用 PointTransformer Block === (这种方法仅能够支持 self.tf_layer_num <= 2，在层数过多的情况下会出现训练过程中的梯度骤增)
        if not self.use_tf_block:
            if self.tf_layer_num >= 1:
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

    def _init_sinkhorn(self):
        return Sinkhorn(
            max_iter=self.cfg.MODEL.SINKHORN_MAXITER, tau=self.cfg.MODEL.SINKHORN_TAU
        )

    def _get_stitch_pcs_gt_mat(self, mat_gt, pc_cls_mask, B_size,N_point,n_stitch_pcs_sum):
        stitch_pcs_gt_mat = torch.zeros((B_size,N_point,N_point),device=mat_gt.device)
        for B in range(B_size):
            stitch_pcs_gt_mat[B][:n_stitch_pcs_sum[B], :n_stitch_pcs_sum[B]] = mat_gt[B][pc_cls_mask[B] == 1][:,pc_cls_mask[B] == 1]
        return stitch_pcs_gt_mat

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

        pc_cls_rst = self.point_classifier(data_dict)
        out_dict.update(pc_cls_rst)
        pc_cls_mask = pc_cls_rst["pc_cls_mask"]
        features = pc_cls_rst["features"]
        n_stitch_pcs_sum = torch.sum(pc_cls_mask, dim=-1)

        stitch_pcs = self._get_stitch_pcs(pcs, n_stitch_pcs_sum, pc_cls_mask, B_size, N_point)
        stitch_pcs_feats = self._get_stitch_pcs_feats(features, n_stitch_pcs_sum, pc_cls_mask, B_size, N_point, self.backbone_feat_dim)
        out_dict.update({"n_stitch_pcs_sum": n_stitch_pcs_sum,})

        # === 提取出的特征输入到PointTransformer Layers\Blocks ===
        pcs_flatten = torch.concat([stitch_pcs[i][:n_stitch_pcs_sum[i]] for i in range(B_size)])
        # 顶点特征输入到PointTransformer层中，获取点与点之间的关系
        for name, layer in self.tf_layers:
            # 如果是自注意力层
            if name == "self":
                features = (
                    layer(
                        pcs_flatten,
                        stitch_pcs_feats.view(-1, self.backbone_feat_dim),
                        batch_length,
                    ).view(B_size, N_point, -1).contiguous()
                )
            # 如果是交叉注意力层
            elif name == "cross":
                features = layer(features)
            # 如果是被封装成块了
            elif name == "block" and self.use_tf_block:
                features = layer(pcs_flatten, features, n_stitch_pcs_sum, B_size, N_point, unmean=True)

        # === 预测点点缝合关系 ===
        # pointcloud_visualize(pcs[0][pc_cls_mask[0]==1])
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
        pc_cls_mask = (out_dict.get("pc_cls_mask", None)).squeeze(-1)
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

        loss = mat_loss * self.w_mat_loss
        loss_dict.update({"loss": loss,})
        return loss_dict

    def _get_stitch_pcs(self, pcs, n_stitch_pcs_sum, pc_cls_mask, B_size, N_point):
        critical_pcs = torch.zeros(B_size, N_point, 3, device=self.device, dtype=pcs.dtype)
        for b in range(B_size):
            critical_pcs[b, : n_stitch_pcs_sum[b]] = pcs[b, pc_cls_mask[b] == 1]
        return critical_pcs

    def _get_stitch_pcs_feats(self, feat, n_stitch_pcs_sum, pc_cls_mask, B_size, N_point, F_dim):
        critical_feats = torch.zeros(B_size, N_point, F_dim, device=self.device, dtype=feat.dtype)
        for b in range(B_size):
            critical_feats[b, : n_stitch_pcs_sum[b]] = feat[b, pc_cls_mask[b] == 1]
        return critical_feats
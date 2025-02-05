# 用训练数据进行inference

import torch
import numpy as np
# from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# from pytorch_lightning.loggers import WandbLogger
from utils import hungarian,stitch_mat2indices,pointcloud_and_stitch_visualize,pointcloud_and_stitch_logits_visualize
from dataset import build_stylexd_dataloader_train_val
from model import build_model

if __name__ == "__main__":
    from utils.config import cfg
    from utils.parse_args import parse_args

    args = parse_args("Jigsaw")

    model_save_path = cfg.MODEL_SAVE_PATH
    output_path = cfg.OUTPUT_PATH
    point_num = cfg.DATA.NUM_PC_POINTS

    model = build_model(cfg).load_from_checkpoint(cfg.WEIGHT_FILE)
    train_loader = build_stylexd_dataloader_train_val(cfg)[0]
    for batch in train_loader:
        inf_rst=model(batch)
        n_stitch_pcs_sum = inf_rst['n_stitch_pcs_sum']
        stitch_mat_pred_ = inf_rst["ds_mat"][:,:n_stitch_pcs_sum,:n_stitch_pcs_sum]
        # stitch_mat_pred_ = stitch_mat_pred_ + stitch_mat_pred_.transpose(1,2)
        # stitch_mat_pred = torch.triu(stitch_mat_pred_)

        stitch_mat_pred = torch.zeros_like(stitch_mat_pred_)
        # 对称选项
        sym_choice = "sym_min"
        if sym_choice == "sym_max":
            stitch_mat_pred_mask = stitch_mat_pred_ > stitch_mat_pred_.transpose(1, 2)
            stitch_mat_pred[stitch_mat_pred_mask] = stitch_mat_pred_[stitch_mat_pred_mask]
            stitch_mat_pred[~stitch_mat_pred_mask] = stitch_mat_pred_.transpose(1, 2)[~stitch_mat_pred_mask]
        elif sym_choice == "sym_avg":
            stitch_mat_pred = (stitch_mat_pred_.transpose(1, 2)+stitch_mat_pred_)/2
        elif sym_choice == "sym_min":
            stitch_mat_pred_mask = stitch_mat_pred_ < stitch_mat_pred_.transpose(1, 2)
            stitch_mat_pred[stitch_mat_pred_mask] = stitch_mat_pred_[stitch_mat_pred_mask]
            stitch_mat_pred[~stitch_mat_pred_mask] = stitch_mat_pred_.transpose(1, 2)[~stitch_mat_pred_mask]
        else:
            stitch_mat_pred[:] = stitch_mat_pred_[:]

        # 两种办法：1，简单取行最大值；2，匈牙利
        mat_choice = "col_max"
        if mat_choice == "col_max":  # 简单取行最大值
            stitch_mat = torch.zeros_like(stitch_mat_pred)
            max_values, max_indices = stitch_mat_pred.max(dim=-1)
            stitch_mat.scatter_(-1, max_indices.unsqueeze(-1), 1)

            # stitch_mat = (torch.zeros_like(stitch_mat_pred)
            #               .scatter_(1, torch.max(stitch_mat_pred, dim=1)[1].unsqueeze(1),1))
        elif mat_choice == "hun":  # 匈牙利
            stitch_mat = hungarian(stitch_mat_pred)
        else:
            raise ValueError(f"错误的mat_choice：{mat_choice}")

        # stitch_mat = hungarian(stitch_mat_pred)

        # 是否去除不对称
        is_sym_equal = False
        if is_sym_equal:
            stitch_mat_mask = stitch_mat != stitch_mat.transpose(1, 2)
            stitch_mat[stitch_mat_mask] = 0

        # visualize ground-true stitch -----------------------------------------------------------------------------
        pcs = batch["pcs"].squeeze(0)
        pc_cls_mask = inf_rst["pc_cls_mask"].squeeze(0)
        stitch_pcs = pcs[pc_cls_mask == 1]
        mat_gt = batch["mat_gt"].squeeze(0)
        stitch_indices_gt = stitch_mat2indices(mat_gt)
        pointcloud_and_stitch_visualize(pcs, stitch_indices_gt, title="ground-true stitch")
        print(f"s/p={len(stitch_indices_gt) / point_num}")

        # visualize predict stitch ---------------------------------------------------------------------------------
        # 是否删除概率太小的
        is_filter_too_small = True
        if is_filter_too_small:
            pc_stitch_threshold = 0.02
        else:
            pc_stitch_threshold = 1

        pc_stitch_mask = torch.sum(stitch_mat_pred * stitch_mat, dim=-1) < pc_stitch_threshold
        pc_stitch_mask = pc_stitch_mask.squeeze(0)
        stitch_mat[0][pc_stitch_mask] = 0
        stitch_indices = stitch_mat2indices(np.array(stitch_mat[0, :, :-1].detach().cpu()))

        logits = np.zeros(stitch_indices.shape[0])
        logits_ = torch.sum(stitch_mat_pred * stitch_mat, dim=-1)  # 缝合置信度
        logits[:] = (logits_.detach().cpu().numpy())[0][~pc_stitch_mask.detach().cpu().numpy()]
        pointcloud_and_stitch_logits_visualize(stitch_pcs, stitch_indices, logits,
                                        title=f"predict stitch(threshold={pc_stitch_threshold})")
        input("Press ENTER to continue")
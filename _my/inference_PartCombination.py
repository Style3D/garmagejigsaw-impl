import torch
import numpy as np
from model import build_model
from dataset import build_stylexd_dataloader_test, build_stylexd_dataloader_train_val

from utils import hungarian, to_device
from utils import (stitch_mat2indices, pointcloud_visualize,
                   pointcloud_and_stitch_visualize, pointcloud_and_stitch_logits_visualize)

if __name__ == "__main__":
    from utils.config import cfg
    from utils.parse_args import parse_args

    args = parse_args("Jigsaw")

    model = build_model(cfg).load_from_checkpoint(cfg.WEIGHT_FILE).cuda()
    test_loader = build_stylexd_dataloader_test(cfg)

    for batch in test_loader:
        batch = to_device(batch, model.device)
        inf_rst = model(batch)
        pcs = batch["pcs"].squeeze(0)
        pc_cls_mask = inf_rst["pc_cls_mask"].squeeze(0)

        # VISUALIZE pc classify ----------------------------------------------------------------------------------------
        stitch_pcs = pcs[pc_cls_mask == 1]
        unstitch_pcs = pcs[pc_cls_mask == 0]
        pointcloud_visualize([stitch_pcs, unstitch_pcs],
                             title=f"predict pcs classify(threshold={cfg})",
                             colormap='cool',colornum=10)

        # get stitch ---------------------------------------------------------------------------------------------------
        n_stitch_pcs_sum = inf_rst['n_stitch_pcs_sum']
        stitch_mat_pred_ = inf_rst["ds_mat"][:, :n_stitch_pcs_sum, :n_stitch_pcs_sum]

        # 对称选项
        stitch_mat_pred = torch.zeros_like(stitch_mat_pred_)
        sym_choice = "sym_max"
        if sym_choice == "sym_max":
            stitch_mat_pred_mask = stitch_mat_pred_ > stitch_mat_pred_.transpose(1, 2)
            stitch_mat_pred[stitch_mat_pred_mask] = stitch_mat_pred_[stitch_mat_pred_mask]
            stitch_mat_pred[~stitch_mat_pred_mask] = stitch_mat_pred_.transpose(1, 2)[~stitch_mat_pred_mask]
        elif sym_choice == "sym_avg":
            stitch_mat_pred = (stitch_mat_pred_.transpose(1, 2) + stitch_mat_pred_) / 2
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

        # VISUALIZE predict stitch -------------------------------------------------------------------------------------
        pc_cls_mask = inf_rst["pc_cls_mask"].squeeze(0)
        stitch_pcs = pcs[pc_cls_mask == 1]

        # 是否删除概率太小的
        is_filter_too_small = True
        if is_filter_too_small:
            # col_max0.2 hun0.15
            pc_stitch_threshold = 0.15
        else:
            pc_stitch_threshold = 0

        logits_ = torch.sum(stitch_mat_pred * stitch_mat, dim=-1)  # 缝合置信度
        pc_stitch_mask = logits_ < pc_stitch_threshold
        pc_stitch_mask = pc_stitch_mask.squeeze(0)
        stitch_mat[0][pc_stitch_mask] = 0
        stitch_indices = stitch_mat2indices(stitch_mat[:, :].detach().cpu().numpy().squeeze(0))
        logits = np.zeros(stitch_indices.shape[0])
        logits[:] = (logits_.detach().cpu().numpy())[0][~pc_stitch_mask.detach().cpu().numpy()]
        # pointcloud_and_stitch_visualize(stitch_pcs, stitch_indices,
        #                                 title=f"predict stitch(threshold={pc_stitch_threshold})")
        pointcloud_and_stitch_logits_visualize(stitch_pcs, stitch_indices, logits,
                                        title=f"predict stitch(threshold={pc_stitch_threshold})",)
        input("Press ENTER to continue")
# 获取点点缝合关系

import torch
import numpy as np
from utils import (pointcloud_visualize, pointcloud_and_stitch_visualize, pointcloud_and_stitch_logits_visualize,
                   hungarian, stitch_mat2indices, stitch_indices2mat, get_export_config)

def get_pointstitch(batch, inf_rst,
                    sym_choice = "sym_max", mat_choice = "hun",
                    filter_neighbor_stitch = True,  filter_neighbor=7,
                    filter_too_long = True, filter_length = 0.2,
                    filter_too_small = True, filter_logits = 0.2,
                    only_triu = False, filter_uncontinue = False,
                    show_pc_cls = False, show_stitch = False,
                    export_vis_result = False):
    """
    :param batch:           # garment_jigsaw model input
    :param inf_rst:         # garment_jigsaw model output
    :param sym_choice:      # sym_max;  sym_avg;  sym_min;
    :param mat_choice:      # hun:hungarian algorithm;  col_max:choose max num of each column;
    :param filter_neighbor_stitch:  # filter stitches between filter_neighbor step neighbor
    :param filter_neighbor
    :param filter_too_long:         # filter stitches whose distance longer than filter_length
    :param filter_length:
    :param filter_too_small:        # filter stitches whose logits smaller than filter_logits
    :param filter_logits:
    :param only_triu:       # if true, the result map will only include triu part of stitches
    :param show_pc_cls:     # visualize pointcloud classify result in 3D space
    :param show_stitch:     # visualize stitch predict result in 3D space
    :return:
    """

    pcs = batch["pcs"].squeeze(0)
    pc_cls_mask = inf_rst["pc_cls_mask"].squeeze(0)

    # VISUALIZE pc classify ----------------------------------------------------------------------------------------
    stitch_pcs = pcs[pc_cls_mask == 1]
    unstitch_pcs = pcs[pc_cls_mask == 0]
    if show_pc_cls:
        pointcloud_visualize([stitch_pcs, unstitch_pcs],
                             title=f"predict pcs classify",
                             colormap='cool', colornum=20, color_norm=[0,1])

    # get stitch ---------------------------------------------------------------------------------------------------
    n_stitch_pcs_sum = inf_rst['n_stitch_pcs_sum']
    stitch_mat_pred_ = inf_rst["ds_mat"][:, :n_stitch_pcs_sum, :n_stitch_pcs_sum]

    # 禁止相邻点相交 ------------------------------------------------------------------------------------------------------
    # 禁止点自交
    stitch_mat_pred_[0][torch.eye(stitch_mat_pred_.shape[-1], stitch_mat_pred_.shape[-1]) == 1] = 0
    # 禁止相邻点相交
    # [todo] 收束类型的相邻点相交不应该被筛除，这个可以通过缝合距离来判断
    #  [todo] 这一步放到结尾比较合理
    if filter_neighbor_stitch:
        i_indices = torch.arange(stitch_mat_pred_.shape[-1]).view(-1, 1).repeat(1, stitch_mat_pred_.shape[-1])
        j_indices = torch.arange(stitch_mat_pred_.shape[-1]).view(1, -1).repeat(stitch_mat_pred_.shape[-1], 1)
        mask_neighbor = torch.abs(i_indices - j_indices) < filter_neighbor
        stitch_mat_pred_[0][mask_neighbor] = 0

    # 对称选项 ================================================
    stitch_mat_pred = torch.zeros_like(stitch_mat_pred_)
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



    # 两种办法：1，简单取行最大值；2，匈牙利 =======================
    if mat_choice == "col_max":  # 简单取行最大值
        stitch_mat = torch.zeros_like(stitch_mat_pred)
        max_values, max_indices = stitch_mat_pred.max(dim=-1)
        stitch_mat.scatter_(-1, max_indices.unsqueeze(-1), 1)
        # stitch_mat = (torch.zeros_like(stitch_mat_pred)
        #               .scatter_(1, torch.max(stitch_mat_pred, dim=1)[1].unsqueeze(1),1))
    elif mat_choice == "col_max2":
        msnpp = 2 # max stitch num per point
        # 获取每行最大的2个值和对应的索引
        topk_values, topk_indices = stitch_mat_pred.topk(msnpp, dim=-1)
        # 将stitch_mat初始化为0
        stitch_mat = torch.zeros_like(stitch_mat_pred)
        # 对于每行，将最大两个值的位置设置为1
        stitch_mat.scatter_(dim=-1, index=topk_indices, value=1)
        a=1
    elif mat_choice == "hun":  # 匈牙利
        stitch_mat = hungarian(stitch_mat_pred)
    else:
        raise ValueError(f"错误的mat_choice：{mat_choice}")
    stitch_mat = stitch_mat.int()  # 转换为int类型

    # VISUALIZE predict stitch -------------------------------------------------------------------------------------
    pc_cls_mask = inf_rst["pc_cls_mask"].squeeze(0)
    stitch_pcs = pcs[pc_cls_mask == 1]

    # 去除明显太长的错误结果 ====================================
    if filter_too_long:
        stitch_indices = stitch_mat2indices(stitch_mat[:, :].detach().cpu().numpy().squeeze(0))
        stitch_dis = torch.sqrt(
            torch.sum((stitch_pcs[stitch_indices[:, 0]] - stitch_pcs[stitch_indices[:, 1]]) ** 2, dim=-1))
        stitch_dis = stitch_dis.detach().cpu().numpy()
        stitch_indices = stitch_indices[stitch_dis < filter_length]
        stitch_mat = torch.tensor(stitch_indices2mat(stitch_pcs.shape[-2], stitch_indices),
                                  device=pcs.device, dtype=torch.int64).unsqueeze(0)

    # 删除概率太小的 ===========================================
    if filter_too_small:
        # col_max0.2 hun0.15
        pc_stitch_threshold = filter_logits
    else:
        pc_stitch_threshold = 0


    # 过滤掉概率太小的缝合关系 =========================
    # [modified] logits_ = torch.sum(stitch_mat_pred * stitch_mat, dim=-1)  # 缝合置信度
    logits_ = stitch_mat_pred[stitch_mat == 1]  # 缝合置信度
    pc_stitch_mask = logits_ < pc_stitch_threshold
    logits_ = logits_[~pc_stitch_mask]
    # pc_stitch_mask = pc_stitch_mask.squeeze(0)
    stitch_mat[stitch_mat==1] = ~pc_stitch_mask*1
    stitch_indices = stitch_mat2indices(stitch_mat[:, :].detach().cpu().numpy().squeeze(0))
    # [可视化] 缝合点上，带有概率的缝合关系 =========================
    logits = np.zeros(stitch_indices.shape[0])
    logits[:] = (logits_.detach().cpu().numpy())
    # pointcloud_and_stitch_visualize(stitch_pcs, stitch_indices,
    #                                 title=f"predict stitch(threshold={pc_stitch_threshold})")
    if show_stitch:
        pointcloud_and_stitch_logits_visualize(stitch_pcs, stitch_indices, logits,
                                               title=f"predict stitch(threshold={pc_stitch_threshold}) in stitch points", )

    # [可视化]所有点上，带有概率的缝合关系 =========================
    # 获取完整mat上的缝合关系
    stitch_mat_full = torch.zeros((1, pcs.shape[0], pcs.shape[0]), dtype=torch.int64, device=pcs.device)
    mask1 = torch.zeros((1, pcs.shape[0], pcs.shape[0]), dtype=torch.bool, device=pcs.device)
    mask1[:, pc_cls_mask == 1, :] = True
    mask2 = torch.zeros((1, pcs.shape[0], pcs.shape[0]), dtype=torch.bool, device=pcs.device)
    mask2[:, :, pc_cls_mask == 1] = True
    stitch_mat_full_mask = torch.bitwise_and(mask1, mask2)
    stitch_mat_full[stitch_mat_full_mask] = stitch_mat.reshape(-1)

    # 仅取上三角部分 =========================
    if only_triu:
        stitch_mat_full = torch.triu(stitch_mat_full)

    # 转换成缝在一起的两个点的indices
    stitch_indices_full = stitch_mat2indices(stitch_mat_full)

    # 筛除不连续的缝合（缝合的对象panel突然变了）=============================
    if filter_uncontinue:
        # pointcloud_and_stitch_visualize(pcs, stitch_indices_full.detach().cpu().numpy(),
        #             title=f"predict stitch(threshold={pc_stitch_threshold}) in all points")
        piece_id = batch["piece_id"][0]
        piece_id_cor = piece_id[stitch_indices_full[:,1]]
        mask_r = torch.concat([piece_id_cor[:-1] == piece_id_cor[1:],torch.tensor([False],device=piece_id.device)])
        mask_l = torch.concat([torch.tensor([False],device=piece_id.device),piece_id_cor[1:] == piece_id_cor[:-1]])
        mask = torch.logical_or(mask_r, mask_l)
        stitch_indices_full = stitch_indices_full[mask]
        stitch_mat_full = stitch_indices2mat(pcs.shape[0], stitch_indices_full)
        # pointcloud_and_stitch_visualize(pcs, stitch_indices_full.detach().cpu().numpy(),
        #           title=f"predict stitch(threshold={pc_stitch_threshold}) in all points")

    # 如果缝合关系可传递，则

    if show_stitch:
        pointcloud_and_stitch_logits_visualize(pcs, stitch_indices_full.detach().cpu().numpy(), logits,
                                               title=f"predict stitch(threshold={pc_stitch_threshold}) in all points", )

    # export data
    if export_vis_result:
        pointcloud_and_stitch_logits_visualize(pcs,
                            stitch_indices_full.detach().cpu().numpy(), logits,
                            title=f"predict pcs classify",
                            export_data_config=get_export_config("_tmp/PC_and_stitch_vis", pic_num=120))

    stitch_mat_full.to(pcs.device)
    stitch_indices_full = torch.tensor(stitch_indices_full, device=pcs.device, dtype=torch.int64)

    return stitch_mat_full, stitch_indices_full, logits
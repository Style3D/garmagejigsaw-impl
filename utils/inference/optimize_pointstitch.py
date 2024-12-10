# 优化缝合关系【beta，目前效果一般般】

import torch
from utils import pointcloud_and_stitch_visualize

def optimize_pointstitch(batch, inf_rst, stitch_mat, stitch_indices,
                         show_stitch = False):

    device_ = stitch_indices.device
    pcs = batch["pcs"][0]
    n_pcs = batch["n_pcs"][0]
    pcs_idx = torch.arange(pcs.shape[-2])
    piece_id = batch["piece_id"][0]
    panel_num = torch.sum(n_pcs!=0)

    # 缝合关系的映射
    stitch_mat = torch.zeros((len(pcs)), dtype=torch.int64, device=device_)-1
    stitch_mat[stitch_indices[:, 0]] = stitch_indices[:, 1]
    stitch_mat[stitch_indices[:, 1]] = stitch_indices[:, 0]

    # 根据多大范围进行优化
    opt_range = 3
    result_stitch_indices = []
    n_pcs_cumsum = torch.cumsum(n_pcs, dim=-1)
    for panel_idx in range(len(n_pcs_cumsum)):
        if panel_idx==0: point_start_idx = 0
        else:  point_start_idx = n_pcs_cumsum[panel_idx-1]
        point_end_idx = n_pcs_cumsum[panel_idx]

        points = pcs[point_start_idx:point_end_idx]
        points_idx = pcs_idx[point_start_idx:point_end_idx]
        num_points = n_pcs[panel_idx]

        for point_idx in points_idx:
            if stitch_mat[point_idx]==-1:
                continue

            neighbor_idx = torch.arange(point_idx-opt_range,point_idx+opt_range+1, device=device_)
            # neighbor_idx[neighbor_idx <]
            neighbor_idx[neighbor_idx < point_start_idx] += num_points
            neighbor_idx[neighbor_idx >= point_end_idx] -= num_points

            neighbor_piece_idx = piece_id[neighbor_idx]
            neighbor_piece_idx[neighbor_idx==-1] = -1

            neighbor_idx_cor = stitch_mat[neighbor_idx]
            neighbor_piece_idx_cor = piece_id[neighbor_idx_cor]
            neighbor_piece_idx_cor[neighbor_idx_cor==-1] = -1

            # 获取出现最多的panel的idx
            piece_idx_counts = torch.bincount(neighbor_piece_idx_cor[neighbor_piece_idx_cor!=-1])
            most_frequent_piece_idx = torch.argmax(piece_idx_counts).item()
            most_frequent_mask = neighbor_piece_idx_cor==most_frequent_piece_idx

            dis_from_center = torch.abs(torch.arange(-opt_range,opt_range+1, device=device_))
            target_pts = neighbor_idx_cor[most_frequent_mask][torch.argmin(dis_from_center[most_frequent_mask])]

            result_stitch_indices.append([point_idx, target_pts])

    result_stitch_indices = torch.tensor(result_stitch_indices, dtype=torch.int64, device=device_)
    if show_stitch:
        pointcloud_and_stitch_visualize(pcs, result_stitch_indices.detach().cpu().numpy(), title=f"", )
    return



# 本代码用于计算论文中需要的metric

import os
import json
import torch

from model import build_model
from dataset import build_stylexd_dataloader_train_val
from utils import  to_device, get_pointstitch, stitch_mat2indices
# from utils import pointcloud_visualize, pointcloud_and_stitch_visualize, pointcloud_and_stitch_logits_visualize, composite_visualize
# from utils.inference.save_result import save_result

if __name__ == "__main__":
    from utils.config import cfg
    from utils.parse_args import parse_args

    args = parse_args("Jigsaw")

    model_save_path = cfg.MODEL_SAVE_PATH
    output_path = cfg.OUTPUT_PATH
    point_num = cfg.DATA.NUM_PC_POINTS
    model = build_model(cfg).load_from_checkpoint(cfg.WEIGHT_FILE)
    _, val_loader = build_stylexd_dataloader_train_val(cfg)

    metric_dict = {"PRECISION_CLS":[], "RECALL_CLS":[], "PRECISION_STITCH":[], "RECALL_STITCH":[], "AMD":[]}

    for batch in val_loader:
        batch = to_device(batch, model.device)

        try:
            inf_rst = model(batch)
        except Exception as e:
            continue

        B_size, N_point, _ = batch["pcs"].shape
        stitch_mat_ = inf_rst["ds_mat"]
        mat_gt = batch["mat_gt"]
        pc_cls = inf_rst["pc_cls"]
        threshold = cfg.MODEL.PC_CLS_THRESHOLD

        # 计算点分类的metrics ------------------------------------------------------------------------------------
        pc_cls_ = pc_cls.squeeze(-1)
        pc_cls_gt = (torch.sum(mat_gt[:, :N_point] + mat_gt[:, :N_point].transpose(-1, -2),
                               dim=-1) == 1) * 1.0

        indices = pc_cls_ > threshold
        TP_CLS = torch.sum(torch.sum(indices[pc_cls_gt == 1] * 1))
        indices = pc_cls_ > threshold
        FP_CLS = torch.sum(torch.sum((indices[pc_cls_gt == 1] == False) * 1))
        indices = pc_cls_ < threshold
        TN_CLS = torch.sum(torch.sum(indices[pc_cls_gt == 0] * 1))
        indices = pc_cls_ < threshold
        FN_CLS = torch.sum(torch.sum((indices[pc_cls_gt == 0] == False) * 1))
        # === Precision和Recall ===
        PRECISION_CLS = TP_CLS / (TP_CLS + FP_CLS + 1e-13)
        RECALL_CLS = TP_CLS / (TP_CLS + FN_CLS + 1e-13)

        metric_dict["PRECISION_CLS"].append(PRECISION_CLS)
        metric_dict["RECALL_CLS"].append(RECALL_CLS)

        # 计算点缝合的metric ------------------------------------------------------------------------------------
        stitch_mat_full, stitch_indices_full, logits = (
            get_pointstitch(batch, inf_rst,
                            sym_choice="", mat_choice="col_max",
                            filter_neighbor_stitch=False, filter_neighbor=3,
                            filter_too_long=False, filter_length=0.08,
                            filter_too_small=True, filter_logits=0.2,
                            only_triu=False, filter_uncontinue=False,
                            show_pc_cls=False, show_stitch=False, export_vis_result=False))

        mat_gt = mat_gt==1


        pcs = batch["pcs"].squeeze(0)
        mat_gt = mat_gt + mat_gt.transpose(-1, -2)
        stitch_mat_full = stitch_mat_full>=0.9
        # === 缝合Recall ===
        RECALL_STITCH = torch.sum(torch.bitwise_and(stitch_mat_full==mat_gt, mat_gt))/torch.sum(mat_gt)
        metric_dict["RECALL_STITCH"].append(RECALL_STITCH)

        # === 缝合 AMD (average mean distance) ===
        stitch_indices = stitch_mat2indices(stitch_mat_full)
        stitch_indices_gt = stitch_mat2indices(mat_gt)
        mask_cls_pred = (pc_cls_ > threshold).squeeze(0)
        idx_range = torch.range(0,point_num-1,dtype=torch.int64,device=mat_gt.device)
        stitch_map = torch.zeros((point_num), dtype=torch.int64) - 1
        stitch_map[stitch_indices[:, 0]] = stitch_indices[:, 1]
        stitch_map[stitch_indices[:, 1]] = stitch_indices[:, 0]

        stitch_map_gt = torch.zeros((point_num), dtype=torch.int64) - 1
        stitch_map_gt[stitch_indices_gt[:, 0]] = stitch_indices_gt[:, 1]
        stitch_map_gt[stitch_indices_gt[:, 1]] = stitch_indices_gt[:, 0]

        stitch_point_idx = idx_range[mask_cls_pred]

        stitch_cor_pred = stitch_map[stitch_point_idx]
        stitch_cor_gt = stitch_map_gt[stitch_point_idx]

        stitch_pair_valid_mask = torch.bitwise_and(stitch_cor_pred!=-1, stitch_cor_gt!=-1)
        stitch_point_idx_valid = stitch_point_idx[stitch_pair_valid_mask]

        stitch_cor_pred = stitch_map[stitch_point_idx_valid]
        stitch_cor_gt = stitch_map_gt[stitch_point_idx_valid]

        stitch_cor_position_pred = pcs[stitch_cor_pred]
        stitch_cor_position_gt = pcs[stitch_cor_gt]

        normalize_range = batch['normalize_range'].squeeze(0)
        AMD = torch.sum(torch.norm(stitch_cor_position_pred - stitch_cor_position_gt, dim=1))/len(stitch_point_idx_valid)
        AMD *= normalize_range
        metric_dict["AMD"].append(AMD)

        # === 用于保存结果的dict ===
        out_dict = {
            "PRECISION_CLS": torch.mean(torch.Tensor(metric_dict['PRECISION_CLS'])).float().item(),
            "RECALL_CLS": torch.mean(torch.Tensor(metric_dict['RECALL_CLS'])).float().item(),
            "RECALL_STITCH": torch.mean(torch.Tensor(metric_dict["RECALL_STITCH"])).item(),
            "AMD": torch.mean(torch.Tensor(metric_dict["AMD"])).item(),
        }
        with open(os.path.join("/home/Ex1/ProjectFiles/Pycharm_MyPaperWork/Jigsaw_matching/_tmp/metric", 'metric.json'), 'w') as f:
            json.dump(out_dict, f)

        print(f"\nPRECISION_CLS: {out_dict['PRECISION_CLS']}\n"
              f"RECALL_CLS: {out_dict['RECALL_CLS']}\n"
              f"RECALL_STITCH: {out_dict['RECALL_STITCH']}\n"
              f"AMD: {out_dict['AMD']}\n"
              )
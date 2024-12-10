# 用test(无stitches)数据进行inference

import torch
from model import build_model
from dataset import build_stylexd_dataloader_test, build_stylexd_dataloader_train_val
from utils import stitch_mat2indices, pointcloud_and_stitch_visualize, pointcloud_visualize


if __name__ == "__main__":
    from utils.config import cfg
    from utils.parse_args import parse_args

    args = parse_args("Jigsaw")

    model_save_path = cfg.MODEL_SAVE_PATH
    output_path = cfg.OUTPUT_PATH
    point_num = cfg.DATA.NUM_PC_POINTS

    model = build_model(cfg).load_from_checkpoint(cfg.WEIGHT_FILE)
    train_loader = build_stylexd_dataloader_train_val(cfg)[0]
    test_loader = build_stylexd_dataloader_test(cfg)
    for batch in train_loader:
        inf_rst=model(batch)
        B_size,N_point,_ = batch["pcs"].shape

        pcs = batch["pcs"].squeeze(0)
        mat_gt = batch["mat_gt"].squeeze(0)
        pc_cls = inf_rst["pc_cls"].squeeze(0)
        pc_cls_gt = (torch.sum(mat_gt+ mat_gt.transpose(-1, -2), dim=-1) == 1) * 1.0

        pointcloud_visualize([pcs[pc_cls_gt == 1], pcs[pc_cls_gt == 0]],
                             title="gt pcs",
                             colormap='cool',)
        threshold = 0.95

        # indices = pc_cls_ > threshold
        # pointcloud_visualize((batch["pcs"][B])[indices],title="stitch predict pcs")
        # indices = pc_cls_ < threshold
        # pointcloud_visualize((batch["pcs"][B])[indices],title="unstitch predict pcs")

        # 顶点的二分类结果
        indices = pc_cls > threshold
        pointcloud_visualize([pcs[indices], pcs[~indices]],
                             title=f"predict pcs classify(threshold={threshold})",
                             colormap='cool',)

        # 被正确分类的点和错误分类的点
        indices_ = indices == pc_cls_gt
        pointcloud_visualize([pcs[~indices_], pcs[indices_]],
                             title=f"正确分类的点和错误分类的点",
                             colormap='brg',
                             colornum=3)

        indices = pc_cls > threshold
        CLS_ACC = torch.sum((indices == pc_cls_gt) * 1) / N_point
        print(f"ACC={CLS_ACC}")
        indices = pc_cls > threshold
        TP = torch.sum(torch.sum(indices[pc_cls_gt == 1] * 1))
        print(f"TP={TP}")
        indices = pc_cls > threshold
        FP = torch.sum(torch.sum((indices[pc_cls_gt == 1] == False) * 1))
        print(f"TN={FP}")
        indices = pc_cls < threshold
        TN = torch.sum(torch.sum(indices[pc_cls_gt == 0] * 1))
        print(f"FP={TN}")
        indices = pc_cls < threshold
        FN = torch.sum(torch.sum((indices[pc_cls_gt == 0] == False) * 1))
        print(f"FN={FN}")

        ACC = (TP + TN) / (TP + FP + TN + FN)
        print(f"ACC={ACC:.4f}")
        TPR = TP / (TP + FN)
        print(f"TPR={TPR:.4f}")
        TNR = TN / (FP + TN)
        print(f"FPR={TNR:.4f}")

        input("Press ENTER to continue")

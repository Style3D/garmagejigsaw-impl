
from model import build_point_classifier
from dataset import build_stylexd_dataloader_inference # , build_stylexd_dataloader_train_val

from utils import (to_device, get_pointstitch,
                   stitch_mat2indices, pointcloud_visualize,
                   pointcloud_and_stitch_visualize, pointcloud_and_stitch_logits_visualize,
                   composite_visualize)

if __name__ == "__main__":
    from utils.config import cfg
    from utils.parse_args import parse_args

    args = parse_args("Jigsaw")

    model = build_point_classifier(cfg).load_from_checkpoint(cfg.WEIGHT_FILE).cuda()

    # 一些超参数从cfg文件中获得，不用ckpt中的
    model.pc_cls_threshold = cfg.MODEL.PC_CLS_THRESHOLD

    test_loader = build_stylexd_dataloader_inference(cfg)

    for g_idx, batch in enumerate(test_loader):
        batch = to_device(batch, model.device)
        inf_rst = model(batch)

        pcs = batch["pcs"].squeeze(0)
        pc_cls_mask = inf_rst["pc_cls_mask"].squeeze(0)

        # VISUALIZE pc classify ----------------------------------------------------------------------------------------
        stitch_pcs = pcs[pc_cls_mask == 1]
        unstitch_pcs = pcs[pc_cls_mask == 0]
        pointcloud_visualize([stitch_pcs, unstitch_pcs],
                             title=f"predict pcs classify",
                             colormap='cool', colornum=20, color_norm=[0, 1])
        input("Press ENTER to continue")
from model import build_model
from dataset import build_stylexd_dataloader_test, build_stylexd_dataloader_train_val

from utils import (to_device, get_pointstitch,
                   stitch_mat2indices, pointcloud_visualize,
                   pointcloud_and_stitch_visualize, pointcloud_and_stitch_logits_visualize)

if __name__ == "__main__":
    from utils.config import cfg
    from utils.parse_args import parse_args

    args = parse_args("Jigsaw")

    model = build_model(cfg).load_from_checkpoint(cfg.WEIGHT_FILE).cuda()
    # 一些超参数从cfg文件中获得，不用ckpt中的
    model.pc_cls_threshold = cfg.MODEL.PC_CLS_THRESHOLD
    test_loader = build_stylexd_dataloader_test(cfg)

    for batch in test_loader:
        batch = to_device(batch, model.device)
        inf_rst = model(batch)

        stitch_mat_full, stitch_indices_full = get_pointstitch(batch, inf_rst,
                         sym_choice = "sym_max", mat_choice = "hun",
                         filter_too_long = True, filter_length = 0.2,
                         filter_too_small = True, filter_logits = 0.15,
                         show_pc_cls = True, show_stitch = False)

        input("Press ENTER to continue")
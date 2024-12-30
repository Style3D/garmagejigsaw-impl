from dataset import build_stylexd_dataloader_train_val
from dataset import build_stylexd_dataloader_inference

from utils import panel_optimize
from utils import pointcloud_visualize

if __name__ == "__main__":
    from utils.config import cfg
    from utils.parse_args import parse_args

    args = parse_args("Jigsaw")

    train_loader, val_loader = build_stylexd_dataloader_train_val(cfg)
    test_loader = build_stylexd_dataloader_inference(cfg)
    for batch in test_loader:

        pcs = batch["pcs"].squeeze(0).cuda()
        if batch["pcs"].shape[0] != 1: raise ValueError("Batch_size 只能为1")
        n_pcs = batch["n_pcs"].squeeze(0)
        nps = batch["num_parts"].squeeze(0)

        pcs_list = []
        start = 0
        for idx in range(nps):
            pcs_list.append(pcs[start:start + n_pcs[idx]])
            start += n_pcs[idx]

        panel_optimize(pcs_list, max_iter_t=100, max_iter_s=120, target_dis=0., filter_distance=0.12)
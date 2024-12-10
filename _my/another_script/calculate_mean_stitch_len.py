# 用于计算某一噪声下的平均缝合边长
# 然后放到stitch_dis_loss的计算过程中
import numpy as np
import torch
from tqdm import tqdm
from dataset import build_stylexd_dataloader_train_val
from utils import pointcloud_visualize

def calculate_mean_stitch_len(cfg):
    train_loader = build_stylexd_dataloader_train_val(cfg)[0]

    mean_stitch_dis_list = []

    for batch in tqdm(train_loader):
        pcs = batch["pcs"]
        Dis = torch.sqrt(((pcs[:, :, None, :] - pcs[:, None, :, :]) ** 2).sum(dim=-1)) + (
            torch.eye(pcs.shape[1])).to(pcs.device)
        mat_gt = batch["mat_gt"]
        mean_stitch_dis = torch.mean(Dis[mat_gt==1])
        mean_stitch_dis_list.append(mean_stitch_dis)
    mean_stitch_dis = torch.mean(torch.tensor(mean_stitch_dis_list))
    print(mean_stitch_dis)



if __name__ == "__main__":
    from utils.config import cfg
    from utils.parse_args import parse_args
    args = parse_args("Jigsaw")
    calculate_mean_stitch_len(cfg)
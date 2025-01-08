# this code is used to clip the checkpoint of all model to pointclassifier model

import torch
import os
from glob import glob
from tqdm import tqdm

if __name__ == "__main__":
    from utils.config import cfg
    from utils.parse_args import parse_args

    args = parse_args("Jigsaw")

    change_cfg = True  # 是否修改超参数里的cfg文件

    ckpt_dir = "_my/another_script/ckpt_transfer/data"
    out_dir = "_my/another_script/ckpt_transfer/results"
    ckpt_list = sorted(glob(os.path.join(ckpt_dir,"*.ckpt")))

    for ckpt_path in tqdm(ckpt_list[::-1]):
        save_path = os.path.join(out_dir, os.path.basename(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location="cuda:0")
        if "state_dict" in ckpt: state_dict = ckpt["state_dict"]
        else: state_dict = ckpt

        ckpt['hyper_parameters']["cfg"] = cfg

        torch.save(ckpt, save_path)
        print(f"Filtered checkpoint saved to {save_path}")
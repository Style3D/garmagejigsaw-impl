# 对StyleXD带l的数据集的进行预处理
# 得到训练数据

import os
from glob import glob

import numpy as np
from tqdm import tqdm

import igl
import trimesh

from _my.preprocess.stylexd.utils_data_process import *
from utils import stitch_visualize

if __name__ == "__main__":

    # stylexd_dir = "/home/Ex1/Datasets/S3D/StyleXD/StyleXD_with_stitch"
    # out_dir = "/home/Ex1/Datasets/S3D/StyleXD/preprocessed_jigsaw_train"
    # file_list = glob(os.path.join(stylexd_dir,"*.obj"))

    stylexd_dir = "/home/Ex1/Download/Termius/objs_with_stitch"
    # out_dir = "/home/Ex1/Datasets/S3D/StyleXD/preprocessed_jigsaw_train"
    out_dir = "data/stylexd_jigsaw/train"
    file_list = glob(os.path.join(stylexd_dir,"Q*","*","*","*","*","*.obj"))
    for idx, file_path in tqdm(list(enumerate(file_list))):
        base_name = os.path.basename(file_path)

        garment_save_dir = os.path.join(out_dir,"garment_"+f"{idx}".zfill(5))

        obj_dict = parse_obj_file(file_path)
        # stitch_visualize(np.array(obj_dict["vertices"]),np.array(obj_dict["stitch"]))

        meshes = split_mesh_into_parts(obj_dict)


        save_results(obj_dict, meshes, garment_save_dir, file_path)

"""
对整个数据集：
    筛除Panel数量他太多的Garment；
    筛除存在太小的Panel的Garment；
    生成筛除完毕的数据集的训练、验证、测试划分；
    保存数据集的各类数据。
"""



import os
from glob import glob
import json
from tqdm import tqdm

import igl
import trimesh

from torch.utils.data import random_split

# 一个衣服上的Panel如果过多
def filter_toomuchpanel(garment_list, max_panel_num):
    filtered_list = []
    for garment_dir in tqdm(garment_list):
        panel_num = len(glob(os.path.join(garment_dir, "piece_*")))
        if panel_num<=max_panel_num:
            filtered_list.append(garment_dir)
    return filtered_list

def filter_toosmallpanel(garment_list, min_panel_boundary_len):
    filtered_list = []
    for garment_dir in tqdm(garment_list):
        mesh_files = sorted(glob(os.path.join(garment_dir, "piece_*.obj")))
        valid = True
        for idx, mesh_file in enumerate(mesh_files):
            mesh = trimesh.load(mesh_file, force = "mesh", process = False)
            sum_boundary_point=0
            for loop in igl.all_boundary_loop(mesh.faces):
                sum_boundary_point+=len(loop)
            if sum_boundary_point<min_panel_boundary_len:
                valid=False
                break
        if valid:
            filtered_list.append(garment_dir)
    return filtered_list

def filter_by_list(garment_list, list_file_path):
    if list_file_path is None or not os.path.exists(list_file_path):
        print("list_file not exist")
        return garment_list
    filtered_list = []
    with open(list_file_path, "r", encoding="utf-8") as f:
        list_f = json.load(f)
    for garment_dir in garment_list:
        if garment_dir in list_f:
            continue
        filtered_list.append(garment_dir)
    return filtered_list


if __name__ == "__main__":
    max_panel_num = 64
    min_panel_boundary_len = 16

    dataset_dir = "data/stylexd_jigsaw/train"
    output_dir = "_my/preprocess/stylexd/results/dataset_split"

    all_garment_dir = sorted(glob(os.path.join(dataset_dir, "garment_*")))

    # 过滤掉Panel数量过多的Garment
    filtered_garments_dir = filter_toomuchpanel(all_garment_dir, max_panel_num)

    # 过滤掉存在特别小的Panel的Garment
    filtered_garments_dir = filter_toosmallpanel(filtered_garments_dir, min_panel_boundary_len)

    # 仅保留文件名
    filtered_garments_dir = [os.path.basename(dir_) for dir_ in filtered_garments_dir]

    # 删除自定义列表中有的
    filtered_garments_dir = filter_by_list(filtered_garments_dir, list_file_path = "_my/preprocess/stylexd/filter_list.json")

    # 计算每一批数据在筛选完后还有多少
    Q1_num, Q2_num, Q4_num = 0,0,0
    for garments_dir in filtered_garments_dir:
        garment_idx = int(garments_dir.split("_")[-1])
        if 0 <= garment_idx < 890: Q1_num += 1
        if 890 <= garment_idx < 11077: Q2_num += 1
        if 11077 <= garment_idx < 12275: Q4_num += 1

    split = [8, 1, 1]
    garment_num = len(filtered_garments_dir)
    train_size = int(garment_num * split[0]/sum(split))
    val_size = int(garment_num * split[1]/sum(split))
    test_size = garment_num-train_size-val_size

    # 按比例随机划分
    train_dataset, val_dataset, test_dataset = random_split(filtered_garments_dir, [train_size, val_size, test_size])
    train_split = sorted(list(train_dataset))
    val_split = sorted(list(val_dataset))
    test_split = sorted(list(test_dataset))

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_split, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val_split, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_split, f, ensure_ascii=False, indent=2)

    another_info = {
        "total_num": garment_num,
        "max_panel_num":max_panel_num,
        # 训练集、验证集、测试集 的长度
        "size_train":train_size,
        "size_val":val_size,
        "size_test":test_size,
        # 筛选后的数据中，Q1、Q2、Q4分别占了多少
        "Q1_orig_num":890,
        "Q2_orig_num":10187,
        "Q4_orig_num":1198,
        # 筛选后的数据中，Q1、Q2、Q4分别占了多少
        "Q1_filtered_num":Q1_num,
        "Q2_filtered_num":Q2_num,
        "Q4_filtered_num":Q4_num,
    }

    with open(os.path.join(output_dir, "another_info.json"), "w", encoding="utf-8") as f:
        json.dump(another_info, f, ensure_ascii=False, indent=2)

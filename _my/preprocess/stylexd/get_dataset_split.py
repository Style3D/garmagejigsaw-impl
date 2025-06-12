"""
对整个数据集：
    筛除Panel数量他太多的Garment；
    筛除存在太小的Panel的Garment；
    生成筛除完毕的数据集的训练、验证、测试划分；
    保存数据集的各类数据。
"""


import os
import json
import pickle
import random
from glob import glob
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


def keep_percentage(lst, r):
    n = max(1, int(len(lst) * r))  # 至少保留1个元素，防止为空
    return random.sample(lst, n)


if __name__ == "__main__":
    max_panel_num = 64
    min_panel_boundary_len = 16

    dataset_dir = "data/stylexd_jigsaw/train"
    output_dir = "_my/preprocess/stylexd/results/dataset_split"


    filtered_garments_dirs_path = "/home/Ex1/ProjectFiles/Pycharm_MyPaperWork/Jigsaw_matching/_my/preprocess/stylexd/filtered_garments_dirs.json"
    if os.path.exists(filtered_garments_dirs_path):
        with open(filtered_garments_dirs_path, "r", encoding="utf-8") as f:
            filtered_garments_dir = json.load(f)
    else:
        all_garment_dir = sorted(glob(os.path.join(dataset_dir, "garment_*")))

        # 过滤掉Panel数量过多的Garment
        filtered_garments_dir = filter_toomuchpanel(all_garment_dir, max_panel_num)

        # 过滤掉存在特别小的Panel的Garment
        filtered_garments_dir = filter_toosmallpanel(filtered_garments_dir, min_panel_boundary_len)

        # 仅保留文件名
        filtered_garments_dir = [os.path.basename(dir_) for dir_ in filtered_garments_dir]

        # 删除自定义列表中有的
        filtered_garments_dir = filter_by_list(filtered_garments_dir, list_file_path = "_my/preprocess/stylexd/filter_list.json")

        with open("_my/preprocess/stylexd/filtered_garments_dirs.json", "w", encoding="utf-8") as f:
            json.dump(filtered_garments_dir, f)


    # 计算每一批数据在筛选完后还有多少
    Q1_num, Q2_num, Q4_num = 0,0,0
    for garments_dir in filtered_garments_dir:
        garment_idx = int(garments_dir.split("_")[-1])
        if 0 <= garment_idx < 890: Q1_num += 1
        if 890 <= garment_idx < 11077: Q2_num += 1
        if 11077 <= garment_idx: Q4_num += 1

    # 按批次分别存放
    Q_type = ["Q1", "Q2", "Q4"]  # 批次
    Q_range = [1, 0.1, 1]  # 每批数据采样的比例
    Q_list = {k:[] for k in Q_type}

    for garments_dir in tqdm(filtered_garments_dir):
        garment_idx = int(garments_dir.split("_")[-1])
        if 0 <= garment_idx < 890:
            Q_list["Q1"].append(garments_dir)
        elif 890 <= garment_idx < 11077:
            Q_list["Q2"].append(garments_dir)
        else:
            Q_list["Q4"].append(garments_dir)

    # 每个批次仅取一定百分比数量
    for i, Q in enumerate(Q_type):
        if len(Q_list[Q])>0:
            Q_list[Q] = keep_percentage(Q_list[Q], Q_range[i])

    garment_list = []
    for Q in Q_type:
        garment_list.extend(Q_list[Q])

    # # 数据集划分
    # split = [9., 1.]
    # data_list = {"train": [], "val": []}
    # split[0] = int(len(garment_list) * split[0]/sum(split))
    # split[1] = len(garment_list) - split[0]
    #
    # idx_list = range(len(garment_list))
    # train_dataset, val_dataset = random_split(idx_list, split)
    # train_list, val_list = list(train_dataset), list(val_dataset)
    # train_list = [garment_list[idx] for idx in train_list]
    # val_list = [garment_list[idx] for idx in val_list]
    #
    # data_list["train"] = train_list
    # data_list["val"] = val_list

    # with open(os.path.join("_LSR/gen_data_list/output", "stylexd_data_split_reso_256_Q1Q2Q4.pkl"), "wb") as f:
    #     pickle.dump(data_list, f)
    # ===END



    split = [8, 1, 1]
    garment_num = len(garment_list)
    train_size = int(garment_num * split[0]/sum(split))
    val_size = int(garment_num * split[1]/sum(split))
    test_size = garment_num-train_size-val_size

    # 按比例随机划分
    train_dataset, val_dataset, test_dataset = random_split(garment_list, [train_size, val_size, test_size])
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
        # 筛选前的数据中，Q1、Q2、Q4分别占了多少
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

"""
Process raw obj with stitches ("l idx1 idx2") annotation.
"""

import os
import argparse
from glob import glob
from tqdm import tqdm

from data_process.garmageset.utils_data_process import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objs_dir", type=str, default=None, required=True)
    parser.add_argument("--output_root", type=str, default=None, required=True)
    args = parser.parse_args()

    objs_dir = args.objs_dir
    output_root = args.output_root

    file_list = glob(os.path.join(objs_dir,"**","*.obj"), recursive=True)
    for idx, file_path in tqdm(list(enumerate(file_list))):

        uuid = os.path.splitext(os.path.basename(file_path))[0]

        garment_save_dir = os.path.join(output_root, f"{uuid}")

        # load data
        obj_dict = parse_obj_file(file_path)

        # spilt panel-wise
        meshes = split_mesh_into_parts(obj_dict)

        save_results(obj_dict, meshes, garment_save_dir, file_path)

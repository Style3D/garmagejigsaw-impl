# 用于解析 StyleXD中的单个obj文件
import os.path
import numpy as np
import json

def save_results(obj_dict:dict, meshes:list, garment_save_dir:str, file_path):
    os.makedirs(garment_save_dir, exist_ok=True)

    # save pieces mesh
    for idx, mesh in enumerate(meshes):
        piece_save_path = os.path.join(garment_save_dir,"piece_"+f"{idx}".zfill(2)+".obj")
        mesh.export(piece_save_path)

    # === save annotations ===
    annotations_dir = os.path.join(garment_save_dir,"annotations")
    os.makedirs(annotations_dir, exist_ok=True)

    # save stitch
    stitch_save_path = os.path.join(annotations_dir,"stitch.npy")
    np.save(stitch_save_path, obj_dict["stitch"])
    # save uv
    uv_save_path = os.path.join(annotations_dir,"uv.npy")
    np.save(uv_save_path, obj_dict["uv"])
    # save original_path
    additional_info = {"mesh_file_path":file_path}
    with open(os.path.join(annotations_dir,"additional_info.json"),"w",encoding="utf-8") as f:
        json.dump(additional_info, f,ensure_ascii=False)
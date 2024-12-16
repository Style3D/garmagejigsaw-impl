# 本文件在inference_DYdata的基础上修改，用于获得边边缝合关系
import os.path

from model import build_model
from dataset import build_stylexd_dataloader_test

from utils import  to_device, get_pointstitch, pointstitch_2_edgestitch
from utils import pointcloud_visualize, pointcloud_and_stitch_visualize, pointcloud_and_stitch_logits_visualize
from utils.inference.save_result import save_result

if __name__ == "__main__":
    data_type = "brep_reso_128"
    if not data_type in ["DY_0.023",    # [todo] 后面记得删
                         "DY_ML",       # [todo] 后面记得删
                         "StyleGen",
                         "StyleGen_multilayer",   # multi-layer of StyleGen data
                         "brep_reso_128",
                         "brep_reso_256",
                         "brep_reso_512",
                         "brep_reso_1024"]:
        raise ValueError(f"data_type{data_type} is not valid")

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

        # 获取点点缝合关系 -------------------------------------------------------------------------------------------------
        if data_type == "DY_0.023":
            stitch_mat_full, stitch_indices_full = (
                get_pointstitch(batch, inf_rst,
                                sym_choice="sym_max", mat_choice="col_max",
                                filter_neighbor_stitch=True, filter_neighbor = 7,
                                filter_too_long=True, filter_length=0.2,
                                filter_too_small=True, filter_logits=0.18,
                                only_triu=True, filter_uncontinue=False,
                                show_pc_cls=False, show_stitch=True))
        elif data_type == "DY_ML":
            stitch_mat_full, stitch_indices_full = (
                get_pointstitch(batch, inf_rst,
                                sym_choice="sym_max", mat_choice="col_max",
                                filter_neighbor_stitch=True, filter_neighbor = 5,
                                filter_too_long=True, filter_length=0.2,
                                filter_too_small=True, filter_logits=0.05,
                                only_triu=True, filter_uncontinue=False,
                                show_pc_cls=False, show_stitch=False))
        elif data_type == "brep_reso_128":
            stitch_mat_full, stitch_indices_full = (
                get_pointstitch(batch, inf_rst,
                                sym_choice="sym_max", mat_choice="col_max",
                                filter_neighbor_stitch=True, filter_neighbor = 3,
                                filter_too_long=True, filter_length=0.1,
                                filter_too_small=True, filter_logits=0.2,
                                only_triu=True, filter_uncontinue=False,
                                show_pc_cls=False, show_stitch=False))
        # # 优化缝合关系【OPTIONAL】
        # optimized_stitch_indices = optimize_pointstitch(batch, inf_rst, stitch_mat_full, stitch_indices_full, show_stitch = True)

        # 从点点缝合关系获取边边缝合关系 -------------------------------------------------------------------------------------
        garment_json = pointstitch_2_edgestitch(batch, inf_rst,
                                                stitch_mat_full, stitch_indices_full,
                                                unstitch_thresh=15, fliter_len=3,
                                                param_dis_optimize_thresh=0.9)

        # 保存结果 -------------------------------------------------------------------------------------------------------
        save_dir = "_tmp/garment_json_output"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"garment_"+f"{int(batch['data_id'])}".zfill(5)+".json")
        save_result(garment_json, save_path)
        a=1
        # input("Press ENTER to continue")
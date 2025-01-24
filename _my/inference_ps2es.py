# first inference point stitch，then obtain edge stitch

from tqdm import tqdm

import os.path

from model import build_model
from dataset import build_stylexd_dataloader_inference

from utils import  to_device, get_pointstitch, pointstitch_2_edgestitch
from utils import pointcloud_visualize, pointcloud_and_stitch_visualize, pointcloud_and_stitch_logits_visualize, composite_visualize
from utils.inference.save_result import save_result

if __name__ == "__main__":
    data_type = "StyleGen_256"
    if not data_type in [
        "StyleGen",
        "StyleGen_multilayer",   # multi-layer of StyleGen data
        "StyleGen_256",
        "brep_reso_128",
        "brep_reso_256",
        "brep_reso_512",
        "brep_reso_1024"
        ]: raise ValueError(f"data_type{data_type} is not valid")

    from utils.config import cfg
    from utils.parse_args import parse_args

    args = parse_args("Jigsaw")

    model = build_model(cfg).load_from_checkpoint(cfg.WEIGHT_FILE).cuda()
    # 一些超参数从cfg文件中获得，不用ckpt中的
    model.pc_cls_threshold = cfg.MODEL.PC_CLS_THRESHOLD
    test_loader = build_stylexd_dataloader_inference(cfg)

    for batch in tqdm(test_loader):
        batch = to_device(batch, model.device)

        inf_rst = model(batch)

        # 获取点点缝合关系 -------------------------------------------------------------------------------------------------
        if data_type == "StyleGen":
            stitch_mat_full, stitch_indices_full, logits = (
                get_pointstitch(batch, inf_rst,
                                sym_choice="sym_max", mat_choice="col_max",
                                filter_neighbor_stitch=True, filter_neighbor = 7,
                                filter_too_long=True, filter_length=0.2,
                                filter_too_small=True, filter_logits=0.18,
                                only_triu=True, filter_uncontinue=False,
                                show_pc_cls=False, show_stitch=False, export_vis_result = False))
        elif data_type == "StyleGen_multilayer":
            stitch_mat_full, stitch_indices_full, logits = (
                get_pointstitch(batch, inf_rst,
                                sym_choice="sym_max", mat_choice="col_max",
                                filter_neighbor_stitch=True, filter_neighbor = 5,
                                filter_too_long=True, filter_length=0.2,
                                filter_too_small=True, filter_logits=0.05,
                                only_triu=True, filter_uncontinue=False,
                                show_pc_cls=False, show_stitch=False, export_vis_result = False))
        elif data_type == "StyleGen_256":
            stitch_mat_full, stitch_indices_full, logits = (
                get_pointstitch(batch, inf_rst,
                                sym_choice="", mat_choice="col_max",
                                filter_neighbor_stitch=True, filter_neighbor = 3,
                                filter_too_long=True, filter_length=0.08,
                                filter_too_small=True, filter_logits=0.2,
                                only_triu=True, filter_uncontinue=False,
                                show_pc_cls=False, show_stitch=False, export_vis_result = False))
        elif data_type == "brep_reso_128":
            stitch_mat_full, stitch_indices_full, logits = (
                get_pointstitch(batch, inf_rst,
                                sym_choice="sym_max", mat_choice="col_max",
                                filter_neighbor_stitch=True, filter_neighbor = 3,
                                filter_too_long=True, filter_length=0.1,
                                filter_too_small=True, filter_logits=0.2,
                                only_triu=True, filter_uncontinue=False,
                                show_pc_cls=False, show_stitch=True, export_vis_result = False))

        # # 优化缝合关系【OPTIONAL】[有问题，而且不是很必要。。。]
        # optimized_stitch_indices = optimize_pointstitch(batch, inf_rst, stitch_mat_full, stitch_indices_full, show_stitch = True)

        # 从点点缝合关系获取边边缝合关系 -------------------------------------------------------------------------------------
        edgestitch_results = pointstitch_2_edgestitch(batch, inf_rst,
                                                stitch_mat_full, stitch_indices_full,
                                                unstitch_thresh=5, fliter_len=2,
                                                param_dis_optimize_thresh=0.9)
        garment_json = edgestitch_results["garment_json"]

        # 保存可视化结果 ---------------------------------------------------------------------------------------------------
        fig_comp = composite_visualize(batch, inf_rst,
                                       stitch_indices_full=stitch_indices_full, logits=logits)

        # 保存结果 -------------------------------------------------------------------------------------------------------
        save_dir = "_tmp/inference_ps2es_output"
        save_dir  = os.path.join(save_dir, data_type)
        if data_type == "StyleGen_256": data_id = int(batch['mesh_file_path'][0].split("_")[-1])
        else: data_id=int(batch['data_id'])
        save_result(save_dir, data_id=data_id, garment_json=garment_json, fig=fig_comp)
        # input("Press ENTER to continue")


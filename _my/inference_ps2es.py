# first inference point stitch，then obtain edge stitch
import torch
from tqdm import tqdm
import time
import os.path
import numpy as np
from model import build_model
from dataset import build_stylexd_dataloader_inference

from utils import  to_device, get_pointstitch, pointstitch_2_edgestitch, pointstitch_2_edgestitch2, export_video_results
from utils import pointcloud_visualize, pointcloud_and_stitch_visualize, pointcloud_and_stitch_logits_visualize, composite_visualize
from utils.inference.save_result import save_result

if __name__ == "__main__":
    data_type = "Garmage256"
    if not data_type in [
        "Garmage64",
        "Garmage64_ML",   # multi-layer of Garmage data
        "Garmage256",
        "brep_reso_128",
        "brep_reso_256",
        "brep_reso_512",
        "brep_reso_1024"
        ]: raise ValueError(f"data_type{data_type} is not valid")
    lst_tmp = []
    from utils.config import cfg
    from utils.parse_args import parse_args

    args = parse_args("Jigsaw")

    model = build_model(cfg).load_from_checkpoint(cfg.WEIGHT_FILE).cuda()
    # 一些超参数从cfg文件中获得，不用ckpt中的
    model.pc_cls_threshold = cfg.MODEL.PC_CLS_THRESHOLD
    inference_loader = build_stylexd_dataloader_inference(cfg)
    model.to("cpu")
    #是否导出为视频（分图片）
    export_vis_result = False
    export_vis_source = True

    for idx, batch in tqdm(enumerate(inference_loader)):
        try:
            time1 = time.time()
            batch = to_device(batch, model.device)
            inf_rst = model(batch)
            if data_type == "Garmage256":
                try:
                    data_id = int(os.path.basename(batch['mesh_file_path'][0]).split("_")[1])
                except Exception:
                    data_id = idx
                g_basename = os.path.basename(batch['mesh_file_path'][0])
            else: data_id=int(batch['data_id'])

            # 获取点点缝合关系 -------------------------------------------------------------------------------------------------
            # if data_type == "Garmage64":
            #     stitch_mat_full, stitch_indices_full, logits = (
            #         get_pointstitch(batch, inf_rst,
            #                         sym_choice="sym_max", mat_choice="col_max",
            #                         filter_neighbor_stitch=True, filter_neighbor = 7,
            #                         filter_too_long=True, filter_length=0.2,
            #                         filter_too_small=True, filter_logits=0.18,
            #                         only_triu=True, filter_uncontinue=False,
            #                         show_pc_cls=False, show_stitch=False, export_vis_result = False))
            # elif data_type == "Garmage64_ML":
            #     stitch_mat_full, stitch_indices_full, logits = (
            #         get_pointstitch(batch, inf_rst,
            #                         sym_choice="sym_max", mat_choice="col_max",
            #                         filter_neighbor_stitch=True, filter_neighbor = 5,
            #                         filter_too_long=True, filter_length=0.2,
            #                         filter_too_small=True, filter_logits=0.05,
            #                         only_triu=True, filter_uncontinue=False,
            #                         show_pc_cls=False, show_stitch=False, export_vis_result = False))
            if data_type == "Garmage256":
                stitch_mat_full, stitch_pcs, unstitch_pcs, stitch_indices, stitch_indices_full, logits = (
                    get_pointstitch(batch, inf_rst,
                                    sym_choice="sym_max", mat_choice="col_max",
                                    filter_neighbor_stitch=True, filter_neighbor = 1,
                                    filter_too_long=True, filter_length=0.1,
                                    filter_too_small=True, filter_logits=0.16,
                                    only_triu=True, filter_uncontinue=False,
                                    show_pc_cls=False, show_stitch=False))
            # elif data_type == "brep_reso_128":
            #     stitch_mat_full, stitch_indices_full, logits = (
            #         get_pointstitch(batch, inf_rst,
            #                         sym_choice="sym_max", mat_choice="col_max",
            #                         filter_neighbor_stitch=True, filter_neighbor = 3,
            #                         filter_too_long=True, filter_length=0.1,
            #                         filter_too_small=True, filter_logits=0.2,
            #                         only_triu=True, filter_uncontinue=False,
            #                         show_pc_cls=False, show_stitch=False, export_vis_result = False))
            else:
                raise NotImplementedError
            if export_vis_result:
                export_video_results(batch, inf_rst, stitch_pcs, unstitch_pcs, stitch_indices, logits, data_id,
                                     vid_len=240, output_dir=os.path.join("_tmp","video_rotate"))
            if export_vis_source:
                batch_np = {}
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch_np[k] = batch[k].detach().cpu()
                    else:
                        batch_np[k] = batch[k]
                inf_rst_np = {}
                for k in inf_rst:
                    if isinstance(inf_rst[k], torch.Tensor):
                        inf_rst_np[k] = inf_rst[k].detach().cpu()
                    else:
                        inf_rst_np[k] = inf_rst[k]
                vis_resource = {
                    "batch":batch_np,
                    "inf_rst":inf_rst_np,
                    "stitch_pcs": stitch_pcs.detach().cpu(),
                    "unstitch_pcs": unstitch_pcs.detach().cpu(),
                    "stitch_indices": stitch_indices,
                    "logits": logits,
                }
            else:
                vis_resource = None

            # # 优化缝合关系【OPTIONAL】[有问题，而且不是很必要。。。]
            # optimized_stitch_indices = optimize_pointstitch(batch, inf_rst, stitch_mat_full, stitch_indices_full, show_stitch = True)

            # 从点点缝合关系获取边边缝合关系 -------------------------------------------------------------------------------------

            # edgestitch_results = pointstitch_2_edgestitch2(batch, inf_rst,
            #                                                stitch_mat_full, stitch_indices_full,
            #                                                unstitch_thresh=3, fliter_len=2,
            #                                                optimize_thresh_neighbor_index_dis=12,
            #                                                optimize_thresh_side_index_dis=6)

            edgestitch_results = pointstitch_2_edgestitch2(batch, inf_rst,
                                                           stitch_mat_full, stitch_indices_full,
                                                           unstitch_thresh=3, fliter_len=2,
                                                           optimize_thresh_neighbor_index_dis=12,
                                                           optimize_thresh_side_index_dis=3)
            garment_json = edgestitch_results["garment_json"]

            # 保存可视化结果 ---------------------------------------------------------------------------------------------------
            fig_comp = composite_visualize(batch, inf_rst,
                                           stitch_indices_full=stitch_indices_full, logits=logits)

            # 保存结果 -------------------------------------------------------------------------------------------------------
            save_dir = "_tmp/inference_ps2es_output"
            data_dir_list = cfg["DATA"]["DATA_TYPES"]["INFERENCE"]
            data_dir = "+".join(data_dir_list)
            save_dir  = os.path.join(save_dir, data_dir)
            save_result(save_dir, data_id=data_id, garment_json=garment_json, fig=fig_comp, g_basename=g_basename
                        , vis_resource=vis_resource, mesh_file_path=batch['mesh_file_path'][0])
            # input("Press ENTER to continue")
            torch.cuda.empty_cache()

            time2 = time.time()
            lst_tmp.append([time2-time1])
            print(np.mean(np.array(lst_tmp)[:,0]))
        except Exception as e:
            print(e)
            print("Failed during processing, continue...")
            continue
"""
predict point stitches
extract seg2seg stitches
save results

Best configuration:
    Update the model for one round using BatchNorm.
    pretrained/train_on_Q4_feature_conv/PClocal+UVlocal_FconvK55D11_tfB3_sn12_bn4_globalSL_STloss0.1_Sample2500/_inference.yaml
      + UPDATE_DIS_ITER = 1
        ADD_NOISE_INFERENCE = True
        NOISE_STRENGTH = 6
"""

import os.path
import argparse
from tqdm import tqdm
from glob import glob

import torch
import numpy as np

from model import build_model
from utils import composite_visualize
from utils.inference.save_result import save_result
from dataset import build_stylexd_dataloader_inference
from utils import  (
    set_seed,
    to_device,
    get_pointstitch,
    pointstitch_2_edgestitch,
    export_video_results)


# === VIS ===
export_vis_result = False
export_vis_source = True


def save_bn_stats(model):
    """
    Iterate through all BatchNorm layers in the model and save
    their running_mean and running_var.
    """
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            # Set momentum
            module.momentum = 0.9

            # Copy tensors to CPU as a precaution and ensure they are detached from the autograd graph
            stats[name] = {
                'running_mean': module.running_mean.clone().detach(),
                'running_var': module.running_var.clone().detach()
            }
    return stats


def restore_bn_stats(model, stats_checkpoint):
    """
    Iterate through all BatchNorm layers in the model and restore
    their state using the saved statistical data.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and name in stats_checkpoint:
            module.running_mean.data.copy_(stats_checkpoint[name]['running_mean'])
            module.running_var.data.copy_(stats_checkpoint[name]['running_var'])

            # Ensure the BN layer is set to track running statistics
            module.track_running_stats = True


def add_noise_on_garmage(batch, inf_noise_strength=3):
    """
    Add noise on generated Garmage`s boundary pcs (same way as training)
    """
    if inf_noise_strength>0:
        stitch_noise_strength3 = inf_noise_strength
        noise3 = (np.random.rand(*(batch["pcs"].shape)) * 2. - 1.)
        noise3 = noise3 / (np.linalg.norm(noise3, axis=1, keepdims=True) + 1e-6)
        noise3 = noise3 * stitch_noise_strength3 * 0.0072
        noise3 = torch.from_numpy(noise3).float().to(batch["pcs"].device)
        batch["pcs_before_add_noise_inference"] = batch["pcs"].clone()
        batch["pcs"] += noise3


def remove_noise_on_garmage(batch):
    if "pcs_before_add_noise_inference" in batch:
        batch["pcs"] = batch["pcs_before_add_noise_inference"]
        del batch["pcs_before_add_noise_inference"]


def get_inference_args(parser:argparse.ArgumentParser):
    assert isinstance(parser, argparse.ArgumentParser)
    parser.add_argument("--weight_file", default=None, type=str)
    parser.add_argument("--data_dir", default=None, type=str)

    # inference noise
    parser.add_argument("--update_dis_iter", default=1, type=int)
    parser.add_argument("--inf_noise_strength", default=6, type=int)


def check_inference_cfg(cfg, args):
    # base setting of inference
    cfg.DATA.SHUFFLE = False
    cfg.BATCH_SIZE = 1
    cfg.NUM_WORKERS = 1

    # set pretrained model
    if args.weight_file is not None:
        if not os.path.isfile(args.weight_file):
            raise FileNotFoundError(args.weight_file)
        cfg.WEIGHT_FILE = args.weight_file

    # Ensure no panel noise.
    cfg.DATA.SCALE_RANGE = 0
    cfg.DATA.ROT_RANGE = 0
    cfg.DATA.TRANS_RANGE = 0
    cfg.DATA.BBOX_NOISE_STRENGTH = 0

    args.save_dir = os.path.join(os.path.dirname(args.data_dir), "garmagejigsaw_output")

    return cfg


if __name__ == "__main__":
    set_seed(seed = 42)

    from utils.config import cfg
    from utils.parse_args import parse_args

    args = parse_args(
        "GarmageJigsaw Train",
        get_inference_args
    )

    cfg = check_inference_cfg(cfg, args)
    update_dis_iter = args.update_dis_iter
    inf_noise_strength = args.inf_noise_strength

    model = build_model(cfg).load_from_checkpoint(cfg.WEIGHT_FILE).cuda()
    model.pc_cls_threshold = 0.5
    model.eval()

    bn_stats_checkpoint = save_bn_stats(model)

    # build inference dataloader
    inference_data_list = {os.path.dirname(p) for p in glob(os.path.join(args.data_dir,"**","*.obj"), recursive=True)}
    inference_data_list = sorted(list(inference_data_list),key=lambda x: os.path.basename(x))
    inference_loader = build_stylexd_dataloader_inference(cfg, inference_data_list=inference_data_list)
    for idx, batch in tqdm(enumerate(inference_loader)):
        try:
            batch = to_device(batch, model.device)

            # === Avoid excessive number of vertices ===
            if batch["pcs"].shape[-2]>5000:
                print("num point too mach, continue...")
                continue

            try: data_id = int(os.path.basename(batch['mesh_file_path'][0]).split("_")[1])
            except Exception: data_id = idx
            g_basename = os.path.basename(batch['mesh_file_path'][0])

            # === model inference ===
            add_noise_on_garmage(batch, inf_noise_strength=inf_noise_strength)

            # Warm up stage (BatchNorm Adaption)
            if update_dis_iter>0:
                with torch.no_grad():
                    restore_bn_stats(model, bn_stats_checkpoint)
                    model.train()
                    for i in range(update_dis_iter):
                        inf_rst = model(batch)
                    model.eval()

            # inference stage
            with torch.no_grad():
                inf_rst = model(batch)
            remove_noise_on_garmage(batch)

            # === optimize point-point stitch ===
            stitch_mat_full, stitch_pcs, unstitch_pcs, stitch_indices, stitch_indices_full, logits = (
                get_pointstitch(batch, inf_rst,
                                sym_choice="", mat_choice="col_max",
                                filter_neighbor_stitch=True, filter_neighbor = 1,
                                filter_too_long=True, filter_length=0.12,
                                filter_too_small=True, filter_prob=0.11,
                                only_triu=True))
            batch = to_device(batch, "cpu")

            # === get seg2seg stitches ===
            edgestitch_results = pointstitch_2_edgestitch(batch, inf_rst,
                                                           stitch_mat_full, stitch_indices_full,
                                                           unstitch_thresh=6, fliter_len=2, division_thresh = 5,
                                                           optimize_thresh_neighbor_index_dis=6,
                                                           optimize_thresh_side_index_dis=8)
            garment_json = edgestitch_results["garment_json"]

            # === export visualization data ===
            # export page for visualize (vis_comp.html)
            fig_comp = composite_visualize(batch, inf_rst,
                                           stitch_indices_full=stitch_indices_full, logits=logits)
            # export video (very slow)
            if export_vis_result:
                export_video_results(batch, inf_rst, stitch_pcs, unstitch_pcs, stitch_indices, logits, data_id,
                                     vid_len=240, output_dir=os.path.join("_tmp","video_rotate"))

            # export visualization source data
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

            # === save results ===
            save_reldir = os.path.relpath(batch["mesh_file_path"][0], args.data_dir)
            save_dir = os.path.join(args.save_dir, save_reldir)
            save_result(
                save_dir,
                garment_json=garment_json,
                fig=fig_comp, vis_resource=vis_resource,
                mesh_file_path=batch['mesh_file_path'][0]
            )

            torch.cuda.empty_cache()

        except Exception as e:
            print(e)
            continue
"""
predict point stitches
extract seg2seg stitches
save results
"""

import os.path
import argparse
from tqdm import tqdm

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
    pointstitch_2_edgestitch4,
    pointstitch_2_edgestitch5,
    export_video_results)

# === Inference ===
UPDATE_DIS_ITER = 1
ADD_NOISE_INFERENCE = True
NOISE_STRENGTH = 6

# === VIS ===
export_vis_result = False
export_vis_source = True


def add_noise_on_garmage(batch, noise_strength=3):
    """
    Add noise on generated Garmage`s boundary pcs (same way as training)
    """
    if ADD_NOISE_INFERENCE:
        stitch_noise_strength3 = noise_strength
        noise3 = (np.random.rand(*(batch["pcs"].shape)) * 2. - 1.)
        noise3 = noise3 / (np.linalg.norm(noise3, axis=1, keepdims=True) + 1e-6)
        noise3 = noise3 * stitch_noise_strength3 * 0.0072
        noise3 = torch.from_numpy(noise3).float().to(batch["pcs"].device)
        batch["pcs_before_add_noise_inference"] = batch["pcs"].clone()
        batch["pcs"] += noise3


def remove_noise_on_garmage(batch):
    if ADD_NOISE_INFERENCE:
        batch["pcs"] = batch["pcs_before_add_noise_inference"]
        del batch["pcs_before_add_noise_inference"]


def get_inference_args(parser:argparse.ArgumentParser):
    assert isinstance(parser, argparse.ArgumentParser)
    parser.add_argument("--weight_file", default=None, type=str)
    parser.add_argument("--data_dir", default=None, type=str)


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

    # set inference data loading file
    if args.data_dir is not None:
        cfg.DATA.DATA_DIR = args.data_dir

    return cfg


if __name__ == "__main__":
    set_seed(seed = 42)

    data_type = "Garmage256"
    from utils.config import cfg
    from utils.parse_args import parse_args

    args = parse_args(
        "Jigsaw",
        get_inference_args
    )

    cfg = check_inference_cfg(cfg, args)

    # model = build_model(cfg).load_from_checkpoint(cfg.WEIGHT_FILE).cuda()
    ModelClass = type(build_model(cfg))
    model = ModelClass.load_from_checkpoint(cfg.WEIGHT_FILE, cfg=cfg, weights_only=False).cuda()
    model.pc_cls_threshold = 0.5

    model.eval()

    inference_loader = build_stylexd_dataloader_inference(cfg)
    for idx, batch in tqdm(enumerate(inference_loader)):
            batch = to_device(batch, model.device)

            # === Avoid excessive number of vertices ===
            if batch["pcs"].shape[-2]>5000:
                print("num point too mach, continue...")
                continue

            try: data_id = int(os.path.basename(batch['mesh_file_path'][0]).split("_")[1])
            except Exception: data_id = idx

            g_basename = os.path.basename(batch['mesh_file_path'][0])

            # === model inference ===
            add_noise_on_garmage(batch, noise_strength=NOISE_STRENGTH)
            # Warm up stage
            if UPDATE_DIS_ITER>0:
                with torch.no_grad():
                    model.train()
                    for i in range(UPDATE_DIS_ITER):
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
                                only_triu=True, filter_uncontinue=True))
            batch = to_device(batch, "cpu")

            # === get seg2seg stitches ===
            # edgestitch_results = pointstitch_2_edgestitch4(batch, inf_rst,
            #                                                stitch_mat_full, stitch_indices_full,
            #                                                unstitch_thresh=12, fliter_len=3, division_thresh = 3,
            #                                                optimize_thresh_neighbor_index_dis=6,
            #                                                optimize_thresh_side_index_dis=3,
            #                                                auto_adjust=False)
            edgestitch_results = pointstitch_2_edgestitch5(
                batch, inf_rst,
                stitch_mat_full,
                stitch_indices_full
            )
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
            save_dir = "_tmp/inference_ps2es_output"
            data_dir_list = cfg["DATA"]["DATA_TYPES"]["INFERENCE"]
            data_dir = "+".join(data_dir_list)
            save_dir  = os.path.join(save_dir, data_dir)
            save_result(save_dir, data_id=data_id, garment_json=garment_json, fig=fig_comp, g_basename=g_basename
                        , vis_resource=vis_resource, mesh_file_path=batch['mesh_file_path'][0])

            torch.cuda.empty_cache()

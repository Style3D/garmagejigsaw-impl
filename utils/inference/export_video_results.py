import os
import pickle
from glob import glob

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from utils.utils import _denormalize_pts
from utils import (
    pointcloud_visualize,
    pointcloud_and_stitch_logits_visualize,
    draw_bbox_geometry,
    get_export_config
)


def export_video_results(batch, inf_rst,
                       stitch_pcs, unstitch_pcs,
                        stitch_indices, logits,
                       data_id, vid_len, output_dir=""):
    # Rotating video of point clouds grouped by panels
    pcs = batch["pcs"].squeeze(0)
    num_parts = batch["num_parts"].item()
    piece_id = batch["piece_id"].squeeze(0).detach().cpu().numpy()
    part_masks = [piece_id == part_idx for part_idx in range(num_parts)]
    pcs_parts = [pcs[msk] for msk in part_masks]

    # === Geometry Image ===

    # Garment visualization
    with open(glob(os.path.join(batch["mesh_file_path"][0], "original_data","*.pkl"))[0], 'rb') as f:
        orig_data = pickle.load(f, fix_imports=True)
    n_surfs = orig_data["surf_bbox"].shape[-2]
    surf_bbox = orig_data["surf_bbox"]
    surf_ncs = orig_data["surf_ncs"]
    surf_mask = orig_data["surf_mask"]

    surf_mask = surf_mask.reshape(n_surfs, 256, 256, 1)
    surf_ncs = surf_ncs.reshape(n_surfs, 256, 256, 3)

    surf_ncs_ = surf_ncs[...,: 3].reshape(n_surfs, -1, 3)
    surf_mask_ = surf_mask[..., -1:].reshape(n_surfs, -1) > 0.0
    surf_wcs_ = _denormalize_pts(surf_ncs_,surf_bbox)
    colormap = plt.cm.coolwarm
    colors = [to_hex(colormap(i)) for i in np.linspace(0, 1, n_surfs)]

    draw_bbox_geometry(
        bboxes=surf_bbox,
        bbox_colors=colors,
        points=surf_wcs_,
        boundary_points = [p.detach().cpu().numpy() for p in pcs_parts],
        point_colors=colors,
        point_masks=surf_mask_,
        num_point_samples=1500,
        show_bbox=False,
        export_data_config=get_export_config(os.path.join(output_dir, "geometry_bbox_vis", f"{data_id}".zfill(5)), pic_num=vid_len)
    )

    # Rotating video of extracted boundary points
    pointcloud_visualize(pcs_parts, colormap='coolwarm', colornum=len(pcs_parts), color_norm=[0, len(pcs_parts)],
                         export_data_config=get_export_config(os.path.join(output_dir, "point_cloud_vis", f"{data_id}".zfill(5)), pic_num=vid_len))

    # Rotating video of point cloud classification
    pointcloud_visualize([stitch_pcs, unstitch_pcs], colormap='bwr', colornum=2, color_norm=[0, 1],
                         export_data_config=get_export_config(os.path.join(output_dir, "point_cloud_cls_vis", f"{data_id}".zfill(5)), pic_num=vid_len))

    # Rotating video of stitching/seaming results
    pointcloud_and_stitch_logits_visualize([stitch_pcs, unstitch_pcs],
                                           stitch_indices, logits, colormap='bwr', colornum=2, color_norm=[0, 1],
                                           title=f"predict pcs classify",
                                           export_data_config=get_export_config(os.path.join(output_dir, "PC_and_stitch_vis", f"{data_id}".zfill(5)), pic_num=vid_len))

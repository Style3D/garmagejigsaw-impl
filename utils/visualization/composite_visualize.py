# 新增功能：composite_visualize 在一个页面中显示多个内容，并能够保存为html文件
import math
import os
import pickle

import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def composite_visualize(batch, inf_rst, stitch_indices_full=None, logits=None, choice=(tuple([True]*3),tuple([True]*3))):
    """
    :param batch:       dataset output with 1 batch_size
    :param inf_rst:     inference result of model
    :return:
    """
    # === 数据 ===
    uv = batch["uv"].squeeze(0).detach().cpu().numpy()
    pcs = batch["pcs"].squeeze(0).detach().cpu().numpy()
    num_parts = batch["num_parts"].item()
    piece_id = batch["piece_id"].squeeze(0).detach().cpu().numpy()
    part_masks = [piece_id == part_idx for part_idx in range(num_parts)]
    pc_cls_mask = inf_rst["pc_cls_mask"].squeeze(0).detach().cpu().numpy()
    if stitch_indices_full is not None:
        stitch_indices_full = stitch_indices_full.detach().cpu().numpy()

    # === fig的布局设置 ===
    fig = make_subplots(
        rows=2, cols=3, subplot_titles=("Original 3D Boundary", "Classified 3D Boundary", "Point Stitch",
                                        "Geometry Image",       "Original UV",            "Classified UV"),
        specs=[
            [{"type": "scene", "colspan": 1, "rowspan": 1}, {"type": "scene", "colspan": 1, "rowspan": 1}, {"type": "scene", "colspan": 1, "rowspan": 1}],
            [{"type": "xy", "colspan": 1, "rowspan": 1},    {"type": "xy", "colspan": 1, "rowspan": 1},    {"type": "xy", "colspan": 1, "rowspan": 1}]
        ],
        vertical_spacing=0.02,  # 设置垂直间距
        horizontal_spacing=0.02  # 设置水平间距
    )

    # === 各种cmap ===
    part_colors = plt.get_cmap('rainbow', num_parts)
    cls_colors = plt.get_cmap('bwr', 2)
    cls_color_norm = mcolors.Normalize(vmin=-0.3, vmax=1.3)
    stitch_logits_colors = plt.get_cmap('coolwarm_r', 10)

    # 往fig的不同subplot里添加各种go ------------------------------------------------------------------
    # === Original Boundary Points ===
    if choice[0][0]:
        for part_idx in range(num_parts):
            part_color = mcolors.to_hex(part_colors(part_idx))
            pts = pcs[part_masks[part_idx]]
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode='markers',
                name='s%02d_xyz' % part_idx,
                marker=dict(
                    size=2,
                    color=part_color,
                    opacity=0.8
                )
            ), row=1, col=1)

    # === Classified Boundary Pointd ===
    if choice[0][1]:
        for cls_idx in range(2):
            part_color = mcolors.to_hex(cls_colors(cls_color_norm(cls_idx)))
            cls_mask = pc_cls_mask==1 if cls_idx==0 else pc_cls_mask==0
            pts = pcs[cls_mask]
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode='markers',
                name='s%02d_xyz' % part_idx,
                marker=dict(
                    size=2,
                    color=part_color,
                    opacity=0.8
                )
            ), row=1, col=2)

    # === Point Stitch ===
    if choice[0][2] and stitch_indices_full is not None:
        # Point
        for cls_idx in range(2):
            part_color = mcolors.to_hex(cls_colors(cls_color_norm(cls_idx)))
            cls_mask = pc_cls_mask==1 if cls_idx==0 else pc_cls_mask==0
            pts = pcs[cls_mask]
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode='markers',
                name='s%02d_xyz' % part_idx,
                marker=dict(
                    size=2,
                    color=part_color,
                    opacity=0.8
                )
            ), row=1, col=3)
        # Stitch
        color_norm = mcolors.Normalize(vmin=0, vmax=1)
        for i, pair in enumerate(stitch_indices_full):
            color = mcolors.to_hex(stitch_logits_colors(color_norm(logits[i])))
            pts = pcs[np.array(pair)]
            x,y,z = pts[:, 0], pts[:, 1], pts[:, 2]
            fig.add_trace(go.Scatter3d(
                x=[x[0], x[1]],
                y=[y[0], y[1]],
                z=[z[0], z[1]],
                mode='lines',
                line=dict(
                    color=color,
                    width=4
                ),
                showlegend=False
            ), row=1, col=3)

    # === Geometry Image ===
    if choice[1][0]:
        geo_fp = os.path.join(batch["mesh_file_path"][0], "original_data", "xyz.pkl")
        mask_fp = os.path.join(batch["mesh_file_path"][0], "original_data", "mask.pkl")
        with open(geo_fp, 'rb') as geo_f, open(mask_fp, 'rb') as mask_f:
            geo = pickle.load(geo_f).transpose(0,3,1,2)
            mask = pickle.load(mask_f).transpose(0,3,1,2)

        grid_imgs = make_grid(torch.cat([torch.FloatTensor((geo + 1.0) * 0.5), torch.FloatTensor(mask)], dim=1),
                              nrow=math.ceil(math.sqrt(num_parts)), ncol=math.ceil(math.sqrt(num_parts)), padding=5)
        grid_imgs = grid_imgs.permute(1, 2, 0).cpu().numpy()
        grid_imgs = np.concatenate([grid_imgs[:, :, :3], np.repeat(grid_imgs[:, :, -1:], 3, axis=-1)], axis=0)
        fig.add_trace(px.imshow(grid_imgs).data[0], row=2, col=1)

    # === Original UV ===
    if choice[1][1]:
        for part_idx in range(num_parts):
            part_color = mcolors.to_hex(part_colors(part_idx))
            pts_uv = uv[part_masks[part_idx]]
            fig.add_trace(go.Scatter(
                x=pts_uv[:, 0],
                y=pts_uv[:, 1],
                mode='markers',
                name='s%02d_uv' % part_idx,
                marker=dict(
                    size=3,
                    color=part_color,
                    opacity=0.8
                )
            ), row=2, col=2)

    # === Classified UV ===
    if choice[1][2]:
        for cls_idx in range(2):
            part_color = mcolors.to_hex(cls_colors(cls_color_norm(cls_idx)))
            cls_mask = pc_cls_mask==1 if cls_idx==0 else pc_cls_mask==0
            pts_uv = uv[cls_mask]
            fig.add_trace(go.Scatter(
                x=pts_uv[:, 0],
                y=pts_uv[:, 1],
                mode='markers',
                name='s%02d_uv' % part_idx,
                marker=dict(
                    size=3,
                    color=part_color,
                    opacity=0.8
                )
            ), row=2, col=3)

    # fig的设置 ------------------------------------------------------------------------------------
    scene_dict = dict(
            xaxis=dict(range=[-1.5, 1.5]),
            yaxis=dict(range=[-1.5, 1.5]),
            zaxis=dict(range=[-1.5, 1.5]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1)
        )
    fig.update_layout(
        height=1400, width=2400, title_text="Garment",
        scene_camera =  dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=0.1, z=0.5)),
        scene2_camera = dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=0.1, z=0.5)),
        scene3_camera = dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=0.1, z=0.5)),
        scene=scene_dict,
        scene2=scene_dict,
        scene3=scene_dict,
    )
    return fig
    # n_surfs = surfPos.shape[0]
    # colors = plt.get_cmap('rainbow', n_surfs)
    #
    # fig = make_subplots(
    #     rows=2, cols=2, subplot_titles=("Original Boundary", "Classify Result", "Point Stitch"),
    #     specs=[
    #         [{"type": "scene", "colspan": 1, "rowspan": 1}, {"type": "scene", "colspan": 1, "rowspan": 1}],
    #         [{"type": "scene", "colspan": 1, "rowspan": 1}, {"colspan": 5, "type": "xy"}]
    #     ])
    #
    # # geometry images and mask
    # grid_imgs = make_grid(
    #     torch.cat([torch.FloatTensor((surf_ncs + 1.0) * 0.5), torch.FloatTensor(point_mask)], dim=1), nrow=n_surfs, padding=5)
    # grid_imgs = grid_imgs.permute(1, 2, 0).cpu().numpy()
    # grid_imgs = np.concatenate([grid_imgs[:, :, :3], np.repeat(grid_imgs[:, :, -1:], 3, axis=-1)], axis=0)
    #
    # for s_idx in range(n_surfs):
    #     surf_color = mcolors.to_hex(colors(s_idx))
    #
    #     valid_pts = surf_pnts_wcs[s_idx].reshape(3, -1)[:, point_mask[s_idx].reshape(-1) > 0.5]
    #     fig.add_trace(go.Scatter3d(
    #         x=valid_pts[0, :],
    #         y=valid_pts[1, :],
    #         z=valid_pts[2, :],
    #         mode='markers',
    #         name='s%02d_xyz' % s_idx,
    #         marker=dict(
    #             size=2,
    #             color=surf_color,
    #             opacity=0.8
    #         )
    #     ), row=1, col=1)
    #
    #     valid_pts_uv = surf_pnts_wcs_uv[s_idx].reshape(2, -1)[:, point_mask[s_idx].reshape(-1) > 0.5]
    #     fig.add_trace(go.Scatter(
    #         x=valid_pts_uv[0, :],
    #         y=valid_pts_uv[1, :],
    #         mode='markers',
    #         name='s%02d_uv' % s_idx,
    #         marker=dict(
    #             size=2,
    #             color=surf_color,
    #             opacity=0.8
    #         )
    #     ), row=1, col=3)
    #
    #     fig.add_trace(px.imshow(grid_imgs).data[0], row=3, col=1)
    #
    # fig.update_layout(
    #     height=1200, width=2400, title_text="Generated Garment",
    #     scene_camera=dict(
    #         up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=0, z=1.5)
    #     ),
    #     scene=dict(
    #         xaxis=dict(range=[-1.5, 1.5]),
    #         yaxis=dict(range=[-1.5, 1.5]),
    #         zaxis=dict(range=[-1.5, 1.5]),
    #         aspectmode='manual',
    #         aspectratio=dict(x=1, y=1, z=1)
    #     )
    # )
    # fig.update_xaxes(title_text="u", range=[-1.75, 1.0], row=1, col=2)
    # fig.update_yaxes(title_text="v", range=[-1.5, 1.25], row=1, col=2)
    #
    # fig.write_html(f"{out_dir}/vis.html")







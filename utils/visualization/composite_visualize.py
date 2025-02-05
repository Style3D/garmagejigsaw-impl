# 新增功能：composite_visualize 在一个页面中显示多个内容，并能够保存为html文件
import math
import os
import pickle
import json
from copy import deepcopy

import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def composite_visualize(batch, inf_rst, stitch_indices_full=None, logits=None):
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
        rows=2, cols=3, subplot_titles=("Original 3D Boundary", "Classified 3D Boundary",   "Point Stitch",
                                        "Geometry Image",       "Original UV",              "Approx Edge",
                                        ),
        specs=[
            [{"type": "scene", "colspan": 1, "rowspan": 1}, {"type": "scene", "colspan": 1, "rowspan": 1}, {"type": "scene", "colspan": 1, "rowspan": 1}],
            [{"type": "xy", "colspan": 1, "rowspan": 1},    {"type": "xy", "colspan": 1, "rowspan": 1},    {"type": "scene", "colspan": 1, "rowspan": 1}],
        ],
        vertical_spacing=0.02,  # 设置垂直间距
        horizontal_spacing=0.02  # 设置水平间距
    )

    # === 各种cmap ===
    part_colors = plt.get_cmap('coolwarm', num_parts)
    cls_colors = plt.get_cmap('bwr', 2)
    cls_color_norm = mcolors.Normalize(vmin=-0.3, vmax=1.3)
    stitch_logits_colors = plt.get_cmap('coolwarm_r', 10)

    # 往fig的不同subplot里添加各种go ------------------------------------------------------------------
    # === Original Boundary Points ===
    for part_idx in range(num_parts):
        part_color = mcolors.to_hex(part_colors(part_idx))
        pts = pcs[part_masks[part_idx]]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode='markers',
            name='s%02d_123123' % part_idx,
            marker=dict(
                size=3,
                color=part_color,
                opacity=0.8
            )
        ), row=1, col=1)

    # === Classified Boundary Pointd ===
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
                size=3,
                color=part_color,
                opacity=0.8
            )
        ), row=1, col=2)

    # === Point Stitch ===
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
                width=8
            ),
            showlegend=False
        ), row=1, col=3)

    # === Geometry Image ===
    geo_fp = os.path.join(batch["mesh_file_path"][0], "original_data", "xyz.pkl")
    mask_fp = os.path.join(batch["mesh_file_path"][0], "original_data", "mask.pkl")
    with open(geo_fp, 'rb') as geo_f, open(mask_fp, 'rb') as mask_f:
        geo = pickle.load(geo_f).transpose(0,3,1,2)
        mask = pickle.load(mask_f).transpose(0,3,1,2)

    geo = (geo+1.)*0.5
    geo[~np.repeat(mask, 3, axis=1)] = 1
    grid_imgs = make_grid(torch.cat([torch.FloatTensor(geo), torch.FloatTensor(mask)], dim=1),
                          nrow=math.ceil(math.sqrt(num_parts)), ncol=math.ceil(math.sqrt(num_parts)), padding=5, pad_value=1.,normalize=False, value_range=(-1,1))
    grid_imgs = grid_imgs.permute(1, 2, 0).cpu().numpy()
    grid_imgs = np.concatenate([grid_imgs[:, :, :3], np.repeat(grid_imgs[:, :, -1:], 3, axis=-1)], axis=0)
    fig.add_trace(px.imshow(grid_imgs).data[0], row=2, col=1)

    # === Original UV ===
    for part_idx in range(num_parts):
        part_color = mcolors.to_hex(part_colors(part_idx))
        pts_uv = uv[part_masks[part_idx]]
        fig.add_trace(go.Scatter(
            x=pts_uv[:, 0],
            y=pts_uv[:, 1],
            mode='markers',
            name='s%02d_uv' % part_idx,
            marker=dict(
                size=7,
                color=part_color,
                opacity=0.8
            )
        ), row=2, col=2)
    #
    # # === Classified UV ===
    # for cls_idx in range(2):
    #     part_color = mcolors.to_hex(cls_colors(cls_color_norm(cls_idx)))
    #     cls_mask = pc_cls_mask==1 if cls_idx==0 else pc_cls_mask==0
    #     pts_uv = uv[cls_mask]
    #     fig.add_trace(go.Scatter(
    #         x=pts_uv[:, 0],
    #         y=pts_uv[:, 1],
    #         mode='markers',
    #         name='s%02d_uv' % part_idx,
    #         marker=dict(
    #             size=3,
    #             color=part_color,
    #             opacity=0.8
    #         )
    #     ), row=2, col=3)
    #
    # color_norm = mcolors.Normalize(vmin=0, vmax=1)
    # for i, pair in enumerate(stitch_indices_full):
    #     color = mcolors.to_hex(stitch_logits_colors(color_norm(logits[i])))
    #     pts = pcs[np.array(pair)]
    #     x,y,z = pts[:, 0], pts[:, 1], pts[:, 2]
    #     fig.add_trace(go.Scatter3d(
    #         x=[x[0], x[1]],
    #         y=[y[0], y[1]],
    #         z=[z[0], z[1]],
    #         mode='lines',
    #         line=dict(
    #             color=color,
    #             width=8
    #         ),
    #         showlegend=False
    #     ), row=1, col=3)

    # === Approx Edge ===
    garment_json_path = batch["garment_json_path"][0]
    annotations_json_path = batch["annotations_json_path"][0]

    with open(garment_json_path, "r") as gf, open(annotations_json_path,"r") as af:
        garment_json = json.load(gf)
        annotations_json = json.load(af)

    panel_nes = np.array(annotations_json["panel_nes"], dtype=np.int64)
    edge_approx = np.array(annotations_json["edge_approx"], dtype=np.int64)


    n_pcs = batch["n_pcs"][0].detach().cpu().numpy()
    panel_num = np.sum(n_pcs!=0)
    n_pcs = n_pcs[:panel_num]
    n_pcs_cumsum = np.cumsum(n_pcs, axis=-1)
    panel_nes_cumsum = np.cumsum(panel_nes, axis=-1)
    edge_approx_global = deepcopy(edge_approx)
    for panel_idx in range(len(n_pcs_cumsum)):
        if panel_idx == 0: edge_start_idx = 0
        else: edge_start_idx = panel_nes_cumsum[panel_idx - 1]
        edge_end_idx = panel_nes_cumsum[panel_idx]
        if panel_idx!=0: edge_approx_global[edge_start_idx:edge_end_idx] += n_pcs_cumsum[panel_idx-1]

    boundary_color = '#169df7'
    fig.add_trace(go.Scatter3d(
        x=pcs[:, 0],
        y=pcs[:, 1],
        z=pcs[:, 2],
        mode='markers',
        name='pcs' ,
        marker=dict(
            size=2,
            color=boundary_color,
            opacity=0.5
        )
    ), row=2, col=3)
    for part_idx in range(num_parts):
        if part_idx == 0: edge_start_idx = 0
        else: edge_start_idx = panel_nes_cumsum[part_idx - 1]
        edge_end_idx = panel_nes_cumsum[part_idx]

        part_color = mcolors.to_hex(part_colors(part_idx))
        part_Approxs = edge_approx_global[edge_start_idx:edge_end_idx]
        for pair  in part_Approxs:
            pts = pcs[pair]
            x,y,z = pts[:, 0], pts[:, 1], pts[:, 2]
            fig.add_trace(go.Scatter3d(
                x=[x[0], x[1]],
                y=[y[0], y[1]],
                z=[z[0], z[1]],
                mode='lines',
                line=dict(
                    color=part_color,
                    width=6
                ),
                showlegend=False
            ), row=2, col=3)


    # fig的设置 ------------------------------------------------------------------------------------
    scene_dict_3d = dict(
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(range=[-1.5, 1.5], visible=False),  # 隐藏x轴
            yaxis=dict(range=[-1.5, 1.5], visible=False),  # 隐藏y轴
            zaxis=dict(range=[-1.5, 1.5], visible=False),  # 隐藏z轴
            xaxis_title=None,  # 移除x轴标题
            yaxis_title=None,  # 移除y轴标题
            zaxis_title=None,  # 移除z轴标题
            bgcolor='rgba(0,0,0,0)',  # 设置背景为透明
        )
    scene_dict_2d = dict(
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(range=[-1.5, 1.5], visible=False, showgrid = False, ),  # 隐藏x轴
            yaxis=dict(range=[-1.5, 1.5], visible=False, showgrid = False, ),  # 隐藏y轴
            xaxis_title=None,  # 移除x轴标题
            yaxis_title=None,  # 移除y轴标题
            bgcolor='rgba(0,0,0,0)',  # 设置背景为透明
        )
    fig.update_layout(
        height=1400, width=2400, title_text="Garment",
        scene_camera  = dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=0.1, z=0.5)),
        scene2_camera = dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=0.1, z=0.5)),
        scene3_camera = dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=0.1, z=0.5)),
        scene4_camera = dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=0.1, z=0.5)),  # 给approx用的
        scene=scene_dict_3d,
        scene2=scene_dict_3d,
        scene3=scene_dict_3d,
        scene4=scene_dict_3d,
        scene5=scene_dict_3d,
        scene6=scene_dict_3d,

        xaxis4=dict(range=[-1.5, 1.5], showgrid=False, visible=False),  # scene4 x轴
        yaxis4=dict(range=[-1.5, 1.5], showgrid=False, visible=False),  # scene4 y轴
        xaxis5=dict(range=[-1.5, 1.5], showgrid=False, visible=False),  # scene5 x轴
        yaxis5=dict(range=[-1.5, 1.5], showgrid=False, visible=False),  # scene5 y轴
        # xaxis6=dict(range=[-1.5, 1.5], showgrid=False, visible=False),  # scene6 x轴
        # yaxis6=dict(range=[-1.5, 1.5], showgrid=False, visible=False),  # scene6 y轴
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


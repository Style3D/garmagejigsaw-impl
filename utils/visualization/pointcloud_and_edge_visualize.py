# 用于将首位相连的（与另一个pointcloud_and_edge_visualize.py的唯一区别）拟合边进行可视化

import math
import os
import os.path
from copy import deepcopy
from typing import List

import numpy as np
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import torch
from matplotlib import pyplot as plt


def pointcloud_and_edge_visualize(vertices:np.array, edge_approx, contour_nes, title="",
                                   export_data_config=None):

    vertices = deepcopy(vertices)

    if isinstance(vertices,torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(vertices, list):
        if isinstance(vertices[0], torch.Tensor):
            for idx, _ in enumerate(vertices):
                vertices[idx] = vertices[idx].detach().cpu().numpy()
        elif isinstance(vertices[0], np.ndarray):
            pass
        else:
            vertices = np.array(vertices)
    if not isinstance(vertices,List) and vertices.ndim==2:
        vertices = vertices.reshape(1,-1,3)


    if not export_data_config:
        point_size = 6
        # if show_edge_start: start_point_size = point_size * 2
        line_width = 32
    else:
        point_size = 4
        # if show_edge_start: start_point_size = point_size * 2
        line_width = 10


    # 场景
    fig = go.Figure()

    all_coords = np.concatenate(vertices, axis=0)

    min_val = np.min(all_coords)
    max_val = np.max(all_coords)
    # # 定义颜色映射
    # colors = cm.get_cmap('tab20', 32)
    # color_norm = mcolors.Normalize(vmin=0, vmax=10)

    boundary_color = '#169df7'
    part_colors = plt.get_cmap('coolwarm', len(vertices))
    for contour_idx in range(len(vertices)):
        contour_pts = vertices[contour_idx]
        contour_edge_approx = edge_approx[contour_idx]

        x = contour_pts[:, 0]
        y = contour_pts[:, 1]
        z = contour_pts[:, 2]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=point_size,
                color=boundary_color,
                opacity=1
            )
            , showlegend=False
        ))

        for e_approx in contour_edge_approx:
            color = mcolors.to_hex(part_colors(contour_idx))
            e_p = contour_pts[e_approx]
            x_line = np.array([e_p[0][0], e_p[1][0]])
            y_line = np.array([e_p[0][1], e_p[1][1]])
            z_line = np.array([e_p[0][2], e_p[1][2]])
            # 添加一条线段，连接点A和点B
            fig.add_trace(go.Scatter3d(
                x=x_line, y=y_line, z=z_line,
                mode='lines',
                line=dict(
                    color=color,  # 线段颜色
                    width=line_width  # 线段宽度
                )
                , showlegend=False
            ))

    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(
            x=(np.max(all_coords[:, 0]) + np.min(all_coords[:, 0])) / 2,
            y=0,
            z=(np.max(all_coords[:, 2]) + np.min(all_coords[:, 2])) / 2),
        eye=dict(x=0, y=0, z=1.5)
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',  # 整个图表背景透明
        plot_bgcolor='rgba(0,0,0,0)'  # 绘图区域背景透明
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='X Axis',
                range=[min_val, max_val],
                showgrid=False,  # 隐藏网格线
                showticklabels=False,  # 隐藏刻度标签
                zeroline=False,
                visible=False  # 隐藏以上所有，和其它的
            ),
            yaxis=dict(
                title='Y Axis',
                range=[min_val, max_val],
                showgrid=False,  # 隐藏网格线
                showticklabels=False,  # 隐藏刻度标签
                zeroline=False,
                visible=False  # 隐藏以上所有，和其它的
            ),
            zaxis=dict(
                title='Z Axis',
                range=[min_val, max_val],
                showgrid=False,  # 隐藏网格线
                showticklabels=False,  # 隐藏刻度标签
                zeroline=False,
                visible=False  # 隐藏以上所有，和其它的
            ),
            aspectmode='cube'  # 确保各个轴的比例相同
        ),
        title=title,
        scene_camera=camera
    )

    # 绕着人旋转一圈，并保存图片 --------------------------------------------------------------------------------------------
    if export_data_config:
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title='X Axis',
                    range=[min_val, max_val],
                    showgrid = False,  # 隐藏网格线
                    showticklabels = False,  # 隐藏刻度标签
                    zeroline = False,
                    visible=False  # 隐藏以上所有，和其它的
                ),
                yaxis=dict(
                    title='Y Axis',
                    range=[min_val, max_val],
                    showgrid = False,  # 隐藏网格线
                    showticklabels = False,  # 隐藏刻度标签
                    zeroline = False,
                    visible=False  # 隐藏以上所有，和其它的
                ),
                zaxis=dict(
                    title='Z Axis',
                    range=[min_val, max_val],
                    showgrid = False,  # 隐藏网格线
                    showticklabels = False,  # 隐藏刻度标签
                    zeroline=False,
                    visible=False  # 隐藏以上所有，和其它的
                ),
                aspectmode='cube'  # 确保各个轴的比例相同
            ),
            title=""
        )
        camera = dict(
            up=dict(x=0, y=1, z=0),
            center=dict(
                x=(np.max(all_coords[:,0])+np.min(all_coords[:,0]))/2,
                y=0,
                z=(np.max(all_coords[:,2])+np.min(all_coords[:,2]))/2),
            eye=dict(x=0, y=0, z=1.5)
        )
        fig.update_layout(scene_camera=camera)

        pic_num = export_data_config["pic_num"]
        os.makedirs(export_data_config["export_path"], exist_ok=True)
        for i in range(export_data_config["pic_num"]+1):
            img_path = os.path.join(export_data_config["export_path"],f"{i}".zfill(3)+".png")

            math.sin(pic_num)
            eye_z = math.cos(2 * math.pi * i / pic_num) * 1.5
            eye_x = math.sin(2 * math.pi * i / pic_num) * 1.5
            camera = dict(
                up=dict(x=0, y=1, z=0),
                center=dict(
                    x=0,
                    y=0,
                    z=0),
                eye=dict(x=eye_x, y=0, z=eye_z)
            )
            # Update layout
            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        visible=False,
                        showbackground=False,
                        showgrid=False,
                        showline=False,
                        showticklabels=False,
                        title=''
                    ),
                    yaxis=dict(
                        visible=False,
                        showbackground=False,
                        showgrid=False,
                        showline=False,
                        showticklabels=False,
                        title=''
                    ),
                    zaxis=dict(
                        visible=False,
                        showbackground=False,
                        showgrid=False,
                        showline=False,
                        showticklabels=False,
                        title=''
                    ),
                    aspectmode='cube'  # Keep the aspect ratio of data
                ),
                width=800,
                height=800,
                margin=dict(r=0, l=0, b=0, t=0),
                showlegend=False,
                title=dict(text=title, automargin=True),
                scene_camera=camera,
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
                paper_bgcolor='rgba(0,0,0,0)'  # Transparent paper background
            )
            fig.write_image(img_path, width=1920, height=1920, scale=2)
            # fig.show()

    # 显示图形
    if export_data_config is None:
        fig.show("browser")
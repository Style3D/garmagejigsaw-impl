
import os
import os.path
from copy import deepcopy

import math
import torch
import numpy as np
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def pointcloud_visualize(vertices_:np.array, title="", colormap="tab20", colornum=32, color_norm = [-10,10], export_data_config=None):
    # 拷贝
    if  isinstance(vertices_, torch.Tensor):
        vertices = deepcopy(vertices_.detach())
    elif isinstance(vertices_, list) and isinstance(vertices_[0], torch.Tensor):
        vertices = [deepcopy(v.detach()) for v in vertices_]
    else:
        vertices = deepcopy(vertices_)

    # 转换ndarray
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(vertices, list):
        if isinstance(vertices[0], torch.Tensor):
            for idx,_ in enumerate(vertices):
                vertices[idx] = vertices[idx].detach().cpu().numpy()
        elif isinstance(vertices[0], np.ndarray):
            pass
        else:
            vertices = np.array(vertices)
    if isinstance(vertices, np.ndarray) and vertices.ndim==2:
        vertices = vertices.reshape(1,-1,3)

    if not export_data_config:
        point_size = 4
    else:
        point_size = 5

    # 场景
    fig = go.Figure()

    all_coords = np.concatenate(vertices, axis=0)

    min_val = np.min(all_coords)
    max_val = np.max(all_coords)
    # 定义颜色映射
    colors = cm.get_cmap(colormap, colornum)
    color_norm = mcolors.Normalize(vmin=color_norm[0], vmax=color_norm[1])

    for i, vertex in enumerate(vertices):
        # 获取颜色
        color = mcolors.to_hex(colors(color_norm(i)))

        # 一个piece上的点
        surf_pnts = vertex

        # x, y, z坐标
        x = surf_pnts[:, 0]
        y = surf_pnts[:, 1]
        z = surf_pnts[:, 2]

        # 在场景中添加点云
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=point_size,
                color=color,
                opacity=0.8
            )
        ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='X Axis',
                range=[min_val, max_val]
            ),
            yaxis=dict(
                title='Y Axis',
                range=[min_val, max_val]
            ),
            zaxis=dict(
                title='Z Axis',
                range=[min_val, max_val]
            ),
            aspectmode='cube'  # 确保各个轴的比例相同
        ),
        title=title
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
            title=title
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
        for i in range(export_data_config["pic_num"]):
            img_path = os.path.join(export_data_config["export_path"],f"{i}".zfill(3)+".jpg")

            math.sin(pic_num)
            eye_z = math.cos(2 * math.pi * i / pic_num) * export_data_config["cam_dis"]
            eye_x = math.sin(2 * math.pi * i / pic_num) * export_data_config["cam_dis"]
            camera = dict(
                up=dict(x=0, y=1, z=0),
                center=dict(
                    x=np.mean(all_coords[:, 0]),
                    y=0,
                    z=np.mean(all_coords[:, 2])),
                eye=dict(x=eye_x, y=0, z=eye_z)
            )
            fig.update_layout(scene_camera=camera)
            fig.write_image(img_path, width=1920, height=1920, scale=2)
            # fig.show()

    # 显示图形
    fig.show()
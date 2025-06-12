import os
import pickle
from glob import glob
import math
import torch
import numpy as np
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os
import os.path
from copy import deepcopy
import torch
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from tqdm import tqdm

from utils.utils import _denormalize_pts
from utils import  get_export_config

def min_max_normalize_pc(points):
    # 计算每一列的最小值和最大值
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # 移动到中心

    points = points - (max_vals + min_vals)/2
    a=1
    # 计算每一列的最小值和最大值
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # 防止除零
    ranges = np.array([1]*points.shape[-1]) * np.max(max_vals - min_vals)
    ranges[ranges == 0] = 1

    # 归一化到 [-0.5, 0.5]
    normalized_points = points/ranges

    return normalized_points

def min_max_normalize_pc_bbox(points, bbox):
    # 计算每一列的最小值和最大值
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # 移动到中心

    bbox_normalized = bbox.reshape(-1, 3)
    points = points - (max_vals + min_vals)/2
    bbox_normalized = bbox_normalized - (max_vals + min_vals)/2
    a=1
    # 计算每一列的最小值和最大值
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # 防止除零
    ranges = np.array([1]*points.shape[-1]) * np.max(max_vals - min_vals)
    ranges[ranges == 0] = 1

    # 归一化到 [-0.5, 0.5]
    normalized_points = points/ranges
    bbox_normalized = bbox_normalized/ranges
    bbox_normalized = bbox_normalized.reshape(-1, 6)

    return normalized_points, bbox_normalized

_CMAP = {
    "帽": {"alias": "帽", "color": "#F7815D"},
    "领": {"alias": "领", "color": "#F9D26D"},
    "肩": {"alias": "肩", "color": "#F23434"},
    "袖片": {"alias": "袖片", "color": "#C4DBBE"},
    "袖口": {"alias": "袖口", "color": "#F0EDA8"},
    "衣身前中": {"alias": "衣身前中", "color": "#8CA740"},
    "衣身后中": {"alias": "衣身后中", "color": "#4087A7"},
    "衣身侧": {"alias": "衣身侧", "color": "#DF7D7E"},
    "底摆": {"alias": "底摆", "color": "#DACBBD"},
    "腰头": {"alias": "腰头", "color": "#DABDD1"},
    "裙前中": {"alias": "裙前中", "color": "#46B974"},
    "裙后中": {"alias": "裙后中", "color": "#6B68F5"},
    "裙侧": {"alias": "裙侧", "color": "#D37F50"},

    "橡筋": {"alias": "橡筋", "color": "#696969"},
    "木耳边": {"alias": "木耳边", "color": "#A8D4D2"},
    "袖笼拼条": {"alias": "袖笼拼条", "color": "#696969"},
    "荷叶边": {"alias": "荷叶边", "color": "#A8D4D2"},
    "绑带": {"alias": "绑带", "color": "#696969"}
}

_PANEL_CLS = [
    '帽', '领', '肩', '袖片', '袖口', '衣身前中', '衣身后中', '衣身侧', '底摆', '腰头', '裙前中', '裙后中', '裙侧', '橡筋', '木耳边', '袖笼拼条', '荷叶边', '绑带']


def _create_bounding_box_lines(min_point, max_point, color):
    # Create the 12 lines of the bounding box
    x_lines = []
    y_lines = []
    z_lines = []

    # List of all 8 corners of the box
    x0, y0, z0 = min_point
    x1, y1, z1 = max_point

    corners = np.array([
        [x0, y0, z0],  # 0
        [x1, y0, z0],  # 1
        [x1, y1, z0],  # 2
        [x0, y1, z0],  # 3
        [x0, y0, z1],  # 4
        [x1, y0, z1],  # 5
        [x1, y1, z1],  # 6
        [x0, y1, z1]  # 7
    ])

    # Pairs of corners between which to draw lines
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)  # Vertical edges
    ]

    for edge in edges:
        start = corners[edge[0]]
        end = corners[edge[1]]
        x_lines.extend([start[0], end[0], None])  # None to break the line
        y_lines.extend([start[1], end[1], None])
        z_lines.extend([start[2], end[2], None])

    line_trace = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode='lines',
        line=dict(color=color, width=2),
        showlegend=False
    )
    return line_trace


def _create_bounding_box_mesh(min_point, max_point, color, opacity=0.2):
    # List of all 8 corners of the box
    x0, y0, z0 = min_point
    x1, y1, z1 = max_point

    corners = np.array([
        [x0, y0, z0],  # 0
        [x1, y0, z0],  # 1
        [x1, y1, z0],  # 2
        [x0, y1, z0],  # 3
        [x0, y0, z1],  # 4
        [x1, y0, z1],  # 5
        [x1, y1, z1],  # 6
        [x0, y1, z1]  # 7
    ])

    # Define the triangles composing the surfaces of the box
    # Each face is composed of two triangles
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 1, 5], [0, 5, 4],  # Side face
        [1, 2, 6], [1, 6, 5],  # Side face
        [2, 3, 7], [2, 7, 6],  # Side face
        [3, 0, 4], [3, 4, 7]  # Side face
    ])

    x = corners[:, 0]
    y = corners[:, 1]
    z = corners[:, 2]

    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]

    mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=opacity,
        name='Bounding Box',
        showlegend=False,
        flatshading=True
    )

    return mesh


def draw_bbox_geometry(
    bboxes,
    bbox_colors,
    points=None,
    boundary_points=None,
    point_masks=None,
    point_colors=None,
    num_point_samples=1000,
    show_bbox=True,
    show_num=False,
    export_data_config=None
):
    annotations = []

    fig = go.Figure()

    if points is not None:
        cur_points_list = []
        for idx in range(len(bboxes)):
        # visuzlize point clouds if given
            cur_points, cur_points_mask = points[idx].reshape(-1, 3), point_masks[idx].reshape(-1)
            cur_points = cur_points[cur_points_mask, :]
            if cur_points.shape[0] > num_point_samples:
                rand_idx = np.random.choice(cur_points.shape[0], num_point_samples, replace=False)
                cur_points = cur_points[rand_idx, :]
                cur_points_list.append(cur_points)

                if boundary_points is not None:
                    cur_points_list[-1] = np.concatenate([cur_points_list[-1],boundary_points[idx]], axis=0)

        pcs_num_cumsum = np.cumsum([len(n) for n in cur_points_list])
        cur_points_normalized, bboxes = min_max_normalize_pc_bbox(np.concatenate(cur_points_list), bboxes)
        for i in range(len(pcs_num_cumsum)):
            if i==0:
                st = 0
            else:
                st = pcs_num_cumsum[i-1]
            ed = pcs_num_cumsum[i]
            cur_points_list[i] = cur_points_normalized[st:ed]

        min_val = np.min(cur_points_normalized)
        max_val = np.max(cur_points_normalized)

        for idx in range(len(bboxes)):
            cur_points = cur_points_list[idx]
            fig.add_trace(go.Scatter3d(
                x=cur_points[:, 0],
                y=cur_points[:, 1],
                z=cur_points[:, 2],
                mode='markers',
                marker=dict(size=5, color=point_colors[idx]),
                name=f'Point Cloud {idx+1}'
            ))

    if show_bbox:
        for idx in range(len(bboxes)):
            # Add the bounding box lines
            min_point, max_point = bboxes[idx, :3], bboxes[idx, 3:]
            bbox_lines = _create_bounding_box_lines(min_point, max_point, color=bbox_colors[idx])
            fig.add_trace(bbox_lines)
            # Add the bounding box surfaces with transparency
            bbox_mesh = _create_bounding_box_mesh(min_point, max_point, color=bbox_colors[idx], opacity=0.05)
            fig.add_trace(bbox_mesh)

            if show_num:
                # Add annotation (always on top)
                center = (min_point + max_point) / 2
                annotations.append(dict(
                    showarrow=False,
                    x=center[0],
                    y=center[1],
                    z=center[2],
                    text=f'<b>{idx}</b>',
                    font=dict(color='black', size=14),
                    xanchor='left',
                    yanchor='bottom',
                    bgcolor='rgba(255,255,255,0.7)',  # Optional: white semi-transparent background
                    bordercolor='black',
                    borderwidth=1,
                    opacity=1
                ))


    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-1.0, y=0, z=2.5)
    )

    # Update layout
    fig.update_layout(
        scene=dict(
            annotations=annotations,
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
            aspectmode='data'  # Keep the aspect ratio of data
        ),
        width=800,
        height=800,
        margin=dict(r=0, l=0, b=0, t=0),
        showlegend=False,
        title=dict(text="", automargin=True),
        scene_camera=camera,
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
        paper_bgcolor='rgba(0,0,0,0)'  # Transparent paper background
    )

    # 显示图形
    if export_data_config is None:
        fig.show("browser")

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
                x=0,
                y=0,
                z=0),
            eye=dict(x=0, y=0, z=1.5)
        )
        fig.update_layout(scene_camera=camera)

        pic_num = export_data_config["pic_num"]
        export_path = export_data_config["export_path"]
        os.makedirs(export_data_config["export_path"], exist_ok=True)
        for i in tqdm(range(export_data_config["pic_num"]+1)):
            img_path = os.path.join(export_path, f"{i}".zfill(3)+".png")

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
                title=dict(text="", automargin=True),
                scene_camera=camera,
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
                paper_bgcolor='rgba(0,0,0,0)'  # Transparent paper background
            )
            fig.write_image(img_path, width=1920, height=1920, scale=2)
            # fig.show()


def pointcloud_and_stitch_logits_visualize(vertices:np.array, stitches:np.array, logistic, colormap="tab20", colornum=32, color_norm = [0,10], title="", export_data_config=None):
    vertices = deepcopy(vertices)

    if isinstance(vertices, torch.Tensor):
        vertices = vertices.clone()
        vertices = vertices.detach().cpu().numpy()
    if isinstance(vertices, list):
        if isinstance(vertices[0], torch.Tensor):
            for idx, _ in enumerate(vertices):
                vertices[idx] = vertices[idx].detach().cpu().numpy()
        elif isinstance(vertices[0], np.ndarray):
            pass
        else:
            vertices = np.array(vertices)

    if isinstance(vertices, np.ndarray) and vertices.ndim==2:
        vertices = vertices.reshape(1,-1,3)

    if isinstance(logistic,torch.Tensor):
        logistic = logistic.detach().cpu().numpy()
    if logistic.ndim==2:
        logistic = logistic.reshape(-1)


    if not export_data_config:
        point_size = 6
        line_width = 5
    else:
        point_size = 5
        line_width = 13


    # 场景
    fig = go.Figure()

    cur_points_list = []
    pcs_num_cumsum = np.cumsum([len(n) for n in vertices])
    cur_points_normalized = min_max_normalize_pc(np.concatenate(vertices))
    for i in range(len(pcs_num_cumsum)):
        if i == 0:
            st = 0
        else:
            st = pcs_num_cumsum[i - 1]
        ed = pcs_num_cumsum[i]
        cur_points_list.append(cur_points_normalized[st:ed])

    min_val = np.min(cur_points_normalized)
    max_val = np.max(cur_points_normalized)

    all_coords = np.concatenate(cur_points_list, axis=0)

    # min_val = np.min(all_coords)
    # max_val = np.max(all_coords)
    # 定义颜色映射
    colors = cm.get_cmap(colormap, colornum)
    color_norm = mcolors.Normalize(vmin=color_norm[0], vmax=color_norm[1])

    for i, vertex in enumerate(cur_points_list):
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
                opacity=0.65
            ),
            showlegend=False
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
        title='3D Global Coordinates of Faces'
    )

    colors_2 = cm.get_cmap('coolwarm_r', 10)
    color_norm_2 = mcolors.Normalize(vmin=0, vmax=1)
    for i, pair in enumerate(stitches):
        # 获取颜色
        color = mcolors.to_hex(colors_2(color_norm_2(logistic[i])))

        # 一个piece上的点
        surf_pnts = all_coords[np.array(pair)]

        # x, y, z坐标
        x = surf_pnts[:, 0]
        y = surf_pnts[:, 1]
        z = surf_pnts[:, 2]

        # # 在场景中添加点云
        # fig.add_trace(go.Scatter3d(
        #     x=x,
        #     y=y,
        #     z=z,
        #     mode='markers',
        #     marker=dict(
        #         size=2,
        #         color=color,
        #         opacity=0.8
        #     )
        # ))

        # 定义一对顶点的坐标 (例如：点A和点B)
        x_line = [x[0], x[1]]  # 顶点 A 和 B 的 x 坐标
        y_line = [y[0], y[1]]  # 顶点 A 和 B 的 y 坐标
        z_line = [z[0], z[1]]  # 顶点 A 和 B 的 z 坐标

        # 添加一条线段，连接点A和点B
        fig.add_trace(go.Scatter3d(
            x=x_line,
            y=y_line,
            z=z_line,
            mode='lines',
            line=dict(
                color=color,  # 线段颜色
                width=line_width  # 线段宽度
            ),
            showlegend=False
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
            aspectmode='cube',  # 确保各个轴的比例相同
            camera=camera
        ))

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
                x=0,
                y=0,
                z=0,),
            eye=dict(x=0, y=0, z=1.5)
        )
        fig.update_layout(scene_camera=camera)

        pic_num = export_data_config["pic_num"]
        os.makedirs(export_data_config["export_path"], exist_ok=True)
        for i in tqdm(range(export_data_config["pic_num"]+1)):
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

    # 显示图形
    if export_data_config is None:
        fig.show("browser")


def pointcloud_visualize(vertices_:np.array, title="", colormap="tab20", colornum=32, color_norm = [-10,10], export_data_config=None):
    # num_parts = 11
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


    cur_points_list = []
    pcs_num_cumsum = np.cumsum([len(n) for n in vertices])
    cur_points_normalized = min_max_normalize_pc(np.concatenate(vertices))
    for i in range(len(pcs_num_cumsum)):
        if i == 0:
            st = 0
        else:
            st = pcs_num_cumsum[i - 1]
        ed = pcs_num_cumsum[i]
        cur_points_list.append(cur_points_normalized[st:ed])

    min_val = np.min(cur_points_normalized)
    max_val = np.max(cur_points_normalized)

    all_coords = np.concatenate(cur_points_list, axis=0)

    # all_coords = np.concatenate(vertices, axis=0)
    #
    # min_val = np.min(all_coords)
    # max_val = np.max(all_coords)
    # 定义颜色映射
    colors = cm.get_cmap(colormap, colornum)
    color_norm = mcolors.Normalize(vmin=color_norm[0], vmax=color_norm[1])
    # part_colors = plt.get_cmap('coolwarm', num_parts)
    for i, vertex in enumerate(cur_points_list):
        # 获取颜色
        color = mcolors.to_hex(colors(color_norm(i)))
        # part_color = mcolors.to_hex(part_colors(i))
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
                # color=part_color,
                color=color,
                opacity=1
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
                x=0,
                y=0,
                z=0),
            eye=dict(x=0, y=0, z=1.5)
        )
        fig.update_layout(scene_camera=camera)

        pic_num = export_data_config["pic_num"]
        export_path = export_data_config["export_path"]
        os.makedirs(export_data_config["export_path"], exist_ok=True)
        for i in tqdm(range(export_data_config["pic_num"]+1)):
            img_path = os.path.join(export_path, f"{i}".zfill(3)+".png")

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



def export_video_results(batch, inf_rst,
                         stitch_pcs, unstitch_pcs,
                         stitch_indices, logits,
                         data_name, vis_resource_fp, vid_len, output_dir=""):
    # 按panel的点的旋转视频
    pcs = batch["pcs"].squeeze(0)
    num_parts = batch["num_parts"].item()
    piece_id = batch["piece_id"].squeeze(0).detach().cpu().numpy()
    part_masks = [piece_id == part_idx for part_idx in range(num_parts)]
    pcs_parts = [pcs[msk] for msk in part_masks]

    # === Geometry Image ===

    # Garmage Visualize
    orig_data_fp = vis_resource_fp.replace("vis_resource.pkl", "orig_data.pkl")
    with open(orig_data_fp, 'rb') as f:
        orig_data = pickle.load(f, fix_imports=True)
    n_surfs = orig_data["surf_bbox"].shape[0]
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
        export_data_config=get_export_config(os.path.join(output_dir, "geometry_bbox_vis_withoutbbox", f"{data_name}"), pic_num=vid_len)
    )
    draw_bbox_geometry(
        bboxes=surf_bbox,
        bbox_colors=colors,
        points=surf_wcs_,
        boundary_points = [p.detach().cpu().numpy() for p in pcs_parts],
        point_colors=colors,
        point_masks=surf_mask_,
        num_point_samples=1500,
        show_bbox=True,
        export_data_config=get_export_config(os.path.join(output_dir, "geometry_bbox_vis", f"{data_name}"), pic_num=vid_len)
    )
    # # 提取的边缘点的旋转视频
    # pointcloud_visualize(pcs_parts, colormap='coolwarm', colornum=len(pcs_parts), color_norm=[0, len(pcs_parts)],
    #                      export_data_config=get_export_config(os.path.join(output_dir, "point_cloud_vis", f"{data_name}"), pic_num=vid_len))
    # # 点分类的旋转视频
    # pointcloud_visualize([stitch_pcs, unstitch_pcs], colormap='bwr', colornum=2, color_norm=[0, 1],
    #                      export_data_config=get_export_config(os.path.join(output_dir, "point_cloud_cls_vis", f"{data_name}"), pic_num=vid_len))
    # # 缝合的旋转视频
    # pointcloud_and_stitch_logits_visualize([stitch_pcs, unstitch_pcs],
    #                                        stitch_indices, logits, colormap='bwr', colornum=2, color_norm=[0, 1],
    #                                        title=f"predict pcs classify",
    #                                        export_data_config=get_export_config(os.path.join(output_dir, "PC_and_stitch_vis", f"{data_name}"), pic_num=vid_len))



if __name__ == "__main__":
    jigsaw_output_root = "/home/Ex1/data/resources/Garmage_SigAisia2025/batch_inference_onlySketch/jigsaw_output"
    vis_resource_list = sorted(glob(os.path.join(jigsaw_output_root, "**", "vis_resource.pkl"), recursive=True))[2:]
    output_dir = "/home/Ex1/ProjectFiles/Pycharm_MyPaperWork/Jigsaw_matching/_my/only_export_vis_rotate/output"
    for vis_resource_fp in vis_resource_list:
        print(vis_resource_fp)
        print("Processing one...")
        with open(vis_resource_fp, "rb") as f:
            vis_data = pickle.load(f, fix_imports=True)

        batch = vis_data["batch"]
        inf_rst = vis_data["inf_rst"]
        stitch_pcs = vis_data["stitch_pcs"]
        unstitch_pcs = vis_data["unstitch_pcs"]
        stitch_indices = vis_data["stitch_indices"]
        logits = vis_data["logits"]

        data_name = os.path.basename(os.path.dirname(vis_resource_fp))
        export_video_results(batch, inf_rst, stitch_pcs, unstitch_pcs, stitch_indices, logits, data_name, vis_resource_fp,
                             vid_len=120, output_dir=os.path.join(output_dir, "video_rotate"))



    # batch_np = {}
    # for k in batch:
    #     if isinstance(batch[k], torch.Tensor):
    #         batch_np[k] = batch[k].detach().cpu()
    #     else:
    #         batch_np[k] = batch[k]
    # inf_rst_np = {}
    # for k in inf_rst:
    #     if isinstance(inf_rst[k], torch.Tensor):
    #         inf_rst_np[k] = inf_rst[k].detach().cpu()
    #     else:
    #         inf_rst_np[k] = inf_rst[k]
    # vis_resource = {
    #     "batch": batch_np,
    #     "inf_rst": inf_rst_np,
    #     "stitch_pcs": stitch_pcs.detach().cpu(),
    #     "unstitch_pcs": unstitch_pcs.detach().cpu(),
    #     "stitch_indices": stitch_indices,
    #     "logits": logits,
    # }
    #







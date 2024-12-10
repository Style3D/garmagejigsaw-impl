import os
import os.path
import numpy as np
import trimesh
from trimesh import Trimesh
import igl
from igl import boundary_loop
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import plotly.graph_objects as go


def stitch_visualize(vertices:np.array, stitches:np.array):
    # 场景
    fig = go.Figure()

    all_coords = vertices

    min_val = np.min(all_coords)
    max_val = np.max(all_coords)
    # 定义颜色映射
    colors = cm.get_cmap('tab20', 1000)
    color_norm = mcolors.Normalize(vmin=0, vmax=1000)

    for i, pair in enumerate(stitches):
        # 获取颜色
        color = mcolors.to_hex(colors(color_norm(i)))

        # 一个piece上的点
        surf_pnts = all_coords[np.array(pair)]

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
                size=2,
                color=color,
                opacity=0.8
            )
        ))

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
                width=1  # 线段宽度
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
        title='3D Global Coordinates of Faces'
    )

    # 显示图形
    fig.show()
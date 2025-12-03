import os
import math

import numpy as np
import plotly.graph_objects as go


def min_max_normalize(points, bbox):
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    bbox_normalized = bbox.reshape(-1, 3)
    points = points - (max_vals + min_vals)/2
    bbox_normalized = bbox_normalized - (max_vals + min_vals)/2

    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    ranges = np.array([1]*points.shape[-1]) * np.max(max_vals - min_vals)
    ranges[ranges == 0] = 1

    normalized_points = points/ranges
    bbox_normalized = bbox_normalized/ranges
    bbox_normalized = bbox_normalized.reshape(-1, 6)

    return normalized_points, bbox_normalized


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
        [x0, y1, z1]   # 7
    ])

    # Pairs of corners between which to draw lines
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
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
        [x0, y1, z1]   # 7
    ])

    # Define the triangles composing the surfaces of the box
    # Each face is composed of two triangles
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 1, 5], [0, 5, 4],  # Side face
        [1, 2, 6], [1, 6, 5],  # Side face
        [2, 3, 7], [2, 7, 6],  # Side face
        [3, 0, 4], [3, 4, 7]   # Side face
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

        pcs_num_cumsum = np.cumsum([len(n) for n in cur_points_list])
        cur_points_normalized, bboxes = min_max_normalize(np.concatenate(cur_points_list), bboxes)
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
            cur_points = np.concatenate([cur_points_list[idx],boundary_points[idx]])
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
                    showgrid = False,
                    showticklabels = False,
                    zeroline = False,
                    visible=False
                ),
                yaxis=dict(
                    title='Y Axis',
                    range=[min_val, max_val],
                    showgrid = False,
                    showticklabels = False,
                    zeroline = False,
                    visible=False
                ),
                zaxis=dict(
                    title='Z Axis',
                    range=[min_val, max_val],
                    showgrid = False,
                    showticklabels = False,
                    zeroline=False,
                    visible=False
                ),
                aspectmode='cube'
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
        for i in range(export_data_config["pic_num"]+1):
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

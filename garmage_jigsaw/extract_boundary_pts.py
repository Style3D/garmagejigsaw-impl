import os
import json
import pickle
import argparse
from glob import glob
from tqdm import tqdm
from typing import List, Tuple
from copy import deepcopy

import cv2
import uuid
import scipy
import trimesh
from matplotlib.colors import to_hex
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import (pointcloud_visualize, draw_per_panel_geo_imgs)
from utils.visualization.draw_bbox_geometry import _create_bounding_box_lines, _create_bounding_box_mesh
import plotly.graph_objects as go


def draw_bbox_geometry(
    bboxes,
    bbox_colors,
    points=None,
    point_masks=None,
    point_colors=None,
    num_point_samples=1000,
    show_bbox=True,
    show_num=False,
    all_bboxes=None,
    export_data_config=None,
    output_fp = None,
    visatt_dict={}
):
    annotations = []

    fig = go.Figure()

    for idx in range(len(bboxes)):
        # visualize point clouds if given
        if points is not None:
            cur_points, cur_points_mask = points[idx].reshape(-1, 3), point_masks[idx].reshape(-1)
            cur_points = cur_points[cur_points_mask, :]
            if cur_points.shape[0] > num_point_samples:
                rand_idx = np.random.choice(cur_points.shape[0], num_point_samples, replace=False)

                # rand_idx = np.linspace(0, cur_points.shape[0] - 1, num_point_samples, dtype=int)

                cur_points = cur_points[rand_idx, :]
            fig.add_trace(go.Scatter3d(
                x=cur_points[:, 0],
                y=cur_points[:, 1],
                z=cur_points[:, 2],
                mode='markers',
                marker=dict(size=2, color=point_colors[idx]),
                name=f'Point Cloud {idx+1}'
            ))

        # Add the bounding box lines
        min_point, max_point = bboxes[idx, :3], bboxes[idx, 3:]
        bbox_lines = _create_bounding_box_lines(min_point, max_point, color=bbox_colors[idx])
        fig.add_trace(bbox_lines)
        # Add the bounding box surfaces with transparency
        bbox_mesh = _create_bounding_box_mesh(min_point, max_point, color=bbox_colors[idx], opacity=0.05)
        fig.add_trace(bbox_mesh)

    if show_bbox:
        for idx in range(len(bboxes)):
            # Add the bounding box lines
            min_point, max_point = bboxes[idx, :3], bboxes[idx, 3:]
            bbox_lines = _create_bounding_box_lines(min_point, max_point, color=bbox_colors[idx])
            fig.add_trace(bbox_lines)
            # Add the bounding box surfaces with transparency
            bbox_mesh = _create_bounding_box_mesh(min_point, max_point, color=bbox_colors[idx],
                                                  opacity=visatt_dict.get("bboxmesh_opacity", 0.2))
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
        eye=dict(x=0, y=0, z=visatt_dict.get("camera_eye_z", 1.2))
    )

    if all_bboxes is None:
        all_mins = np.min(bboxes[:, :3], axis=0)
        all_maxs = np.max(bboxes[:, 3:], axis=0)
    else:
        all_mins = np.min(all_bboxes[:, :3], axis=0)
        all_maxs = np.max(all_bboxes[:, 3:], axis=0)
    center = (all_mins + all_maxs) / 2
    range_scale = visatt_dict.get("range_scale", 1.1)
    max_range = np.max(all_maxs - all_mins) / range_scale
    x_range = [center[0] - max_range, center[0] + max_range]
    y_range = [center[1] - max_range, center[1] + max_range]
    z_range = [center[2] - max_range, center[2] + max_range]

    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=visatt_dict.get("camera_eye_z", 1.2))
    )

    # Update layout
    fig.update_layout(
        scene=dict(
            annotations=annotations,
            camera=camera,
            xaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                showline=False,
                showticklabels=False,
                title='',
                range=x_range
            ),
            yaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                showline=False,
                showticklabels=False,
                title='',
                range=y_range
            ),
            zaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                showline=False,
                showticklabels=False,
                title='',
                range=z_range
            ),
            aspectmode='cube',
        ),
        width=800,
        height=800,
        margin=dict(r=0.1, l=0.1, b=0.1, t=0.1),
        showlegend=False,
        title=dict(text="", automargin=True),
        scene_camera=camera,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    if export_data_config is None and output_fp is None:
        fig.show("browser")

    if output_fp is not None and export_data_config is None:
        fig.write_image(output_fp, width=1080, height=1080, scale=2)


def pointcloud_condition_visualize(vertices: np.ndarray, output_fp=None):
    assert vertices.ndim == 2 and vertices.shape[1] == 3, "vertices 应为 (N, 3) 的 numpy 数组"

    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    color = "#717388"
    xrange = x.max() - x.min()
    yrange = y.max() - y.min()
    zrange = z.max() - z.min()
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=color,
                colorscale='Viridis',
                opacity=1,
                showscale=False
            ),
            showlegend=False
        )
    ])

    axis_style = dict(
        showbackground=False,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False,
        visible=False
    )
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=2)
    )
    fig.update_layout(
        scene=dict(
            xaxis=axis_style,
            yaxis=axis_style,
            zaxis=axis_style,
            aspectmode='manual',
            aspectratio=dict(
                x=xrange,
                y=yrange,
                z=zrange
            )
        ),
        scene_camera=camera,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    RESO = 512
    if output_fp:
        fig.write_image(output_fp.replace(".pkl", "_pcCond.png") , width=RESO, height=RESO)


def is_contour_OutLine(contour_idx, panel_instance_seg):
    """
    Check if the contour is the OutLine of a panel instance
    """
    return contour_idx == 0 or panel_instance_seg[contour_idx] != panel_instance_seg[contour_idx - 1]


def get_random_uuid():
    id = str(uuid.uuid4().hex)
    result = id[0:8] + "-" + id[8:12] + "-" + id[12:16] + "-" + id[16:20] + "-" + id[20:]
    return result


def angle_between_vectors(vectors1, vectors2):
    dot_product = np.sum(vectors1 * vectors2, axis=-1)
    norm_v1 = np.linalg.norm(vectors1, axis=-1)
    norm_v2 = np.linalg.norm(vectors2, axis=-1)
    cos_theta = dot_product / (norm_v1 * norm_v2 + 1e-8)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle_rad)


def _denormalize_pts(pts, bbox):
    pos_dim =  pts.shape[-1]
    bbox_min = bbox[..., :pos_dim][:, None, ...]
    bbox_max = bbox[..., pos_dim:][:, None, ...]
    bbox_scale = np.max(bbox_max - bbox_min, axis=-1, keepdims=True) * 0.5
    bbox_offset = (bbox_max + bbox_min) / 2.0
    return pts * bbox_scale + bbox_offset


def _pad_arr(arr, pad_size=10):
    return np.pad(
        arr,
        ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)),
        # pad size to each dimension, require tensor to have size (H,W, C)
        mode='constant',
        constant_values=0)

from chamferdist import ChamferDistance
CD = ChamferDistance()
def shape_dedup_cd(data):
    threshold = 1

    if "surf_bbox" in data:
        _surf_bbox = data["surf_bbox"]
    else:
        _surf_bbox = data["surf_bbox_wcs"]
    geo_orig = data["surf_ncs"]
    mask = data["surf_mask"]
    n_surfs = geo_orig.shape[0]
    if geo_orig.ndim == 4:
        geo_orig = geo_orig.reshape(n_surfs, -1, geo_orig.shape[3])
    geo_orig = _denormalize_pts(geo_orig, _surf_bbox)

    valid_mask = []
    valid_pts_list = []
    for i in range(n_surfs):
        valid_pts =  geo_orig[i][mask[i]]
        valid_pts = valid_pts[np.random.choice(len(valid_pts), 2048)]
        valid_pts = torch.tensor(valid_pts[None, ...], dtype=torch.float32)
        if len(valid_pts_list)==0:
            valid_pts_list.append(valid_pts)
            valid_mask.append(True)
        else:
            CD_loss_list = []
            for j in range(len(valid_pts_list)):
                CD_loss_list.append(CD(valid_pts_list[j], valid_pts, bidirectional=True))
            CD_loss_list = np.array(CD_loss_list)
            if CD_loss_list.min()<threshold:
                valid_mask.append(False)
            else:
                valid_pts_list.append(valid_pts)
                valid_mask.append(True)

    valid_mask = np.array(valid_mask, dtype=bool)
    for k in data:
        if isinstance(data[k], np.ndarray):
            data[k] = data[k][valid_mask]

    return data

def load_data(garment_fp, save_vis=False, shape_dedup=False):
    """
    Load a garmage file.
    :param garment_fp:
    :param save_vis: If save garmagenet generate results.
    :return:
    """
    with open(garment_fp, "rb") as f:
        data = pickle.load(f)

    # transfer list to ndarray
    for k in data:
        if isinstance(data[k], list):
            data[k] = np.array(data[k])

    # shape dedup before process
    if shape_dedup:
        data = shape_dedup_cd(data)

    # start process data
    geo_orig = data["surf_ncs"]
    mask = data["surf_mask"]
    n_surfs = len(geo_orig)
    if "surf_bbox" in data: _surf_bbox = data["surf_bbox"]
    else: _surf_bbox = data["surf_bbox_wcs"]
    if "surf_uv_bbox" in data: uv_bbox = data["surf_uv_bbox"]
    else: uv_bbox = data["surf_uv_bbox_wcs"]
    if isinstance(uv_bbox, torch.Tensor):
        uv_bbox = uv_bbox.detach().cpu().numpy()

    # save some visualize results (Only for watch) ===
    if save_vis:
        # panel wise geometry image
        mask_draw_per_panel = mask
        if np.max(mask_draw_per_panel)>0.8 and np.min(mask_draw_per_panel)<-0.8:
            mask_draw_per_panel = (mask_draw_per_panel+1)/2
            mask_draw_per_panel[mask_draw_per_panel<0.1] = 0
            mask_draw_per_panel[mask_draw_per_panel>1] = 1
        colors = [to_hex(plt.cm.coolwarm(i)) for i in np.linspace(0, 1, n_surfs)]
        draw_per_panel_geo_imgs(geo_orig.reshape(n_surfs,-1,3), mask_draw_per_panel.reshape(n_surfs,-1), colors, pad_size=5, out_dir=garment_fp.replace(".pkl","")+"_per_panel_images")

        # visualize bbox + geometry
        _surf_ncs_ = data["surf_ncs"].reshape(n_surfs,-1,3)
        _surf_wcs_ = _denormalize_pts(_surf_ncs_, _surf_bbox)
        _mask_ = data["surf_mask"].reshape(n_surfs,-1)

        if _mask_.dtype == np.float32 or _mask_.dtype == np.float64:
            _mask_ = _mask_>0

        draw_bbox_geometry(
            bboxes=_surf_bbox,
            bbox_colors=colors,
            points=_surf_wcs_,
            point_masks=_mask_,
            point_colors=colors,
            num_point_samples=2000,
            all_bboxes=_surf_bbox,
            output_fp=garment_fp.replace(".pkl","")+"_3d_PC_BBOX.png",
            visatt_dict={
                "bboxmesh_opacity": 0.12,
                "point_size": 12,
                "point_opacity": 0.8,
                "bboxline_width": 8,
            }
        )

    # geometry
    if geo_orig.ndim == 4:
        geo_orig = geo_orig.reshape(geo_orig.shape[0], -1, geo_orig.shape[3])
    geo_orig = _denormalize_pts(geo_orig, _surf_bbox)
    geo_orig = geo_orig.reshape(-1, 256, 256, 3)

    # mask
    mask = mask.reshape(-1, 256, 256, 1)
    if np.min(mask) < -0.1 or np.max(mask) > 1.1:
        mask = torch.sigmoid(torch.tensor(mask, dtype=torch.float64))
        mask = mask.detach().numpy()
        mask = mask>0.5

    # pad geometry image
    geo_orig = _pad_arr(geo_orig, pad_size=5)
    mask = _pad_arr(mask, pad_size=5)

    return geo_orig, mask, uv_bbox, data


def resample_boundary(points, contour, corner_index, delta, c_idx, outlier_thresh=0.05):
    """
    Resamples the boundary points and adjusts corner points to fit the new sampling.

    :param points:          Nx3 array of 3D points
    :param contour:         Nx2 array of 2D contour points
    :param corner_index:    Indices of corner points in the original contour
    :param delta:           Resampling arc length interval
    :param c_idx:           Contour index (unused in core logic, for debugging/context)
    :param outlier_thresh:  Used to filter points with abnormal spacing
    :return:
    """

    points = np.asarray(points)
    TEST = False

    if TEST:
        A = contour[corner_index]
        B = contour
        plt.figure(figsize=(8, 6))

        plt.scatter(B[:, 0], B[:, 1], color='blue', label='Points B')

        plt.scatter(A[:, 0], A[:, 1], color='red', label='Points A')

        plt.legend()
        plt.title('Scatter Plot of Points A and B')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        input("Press Enter to continue...")

    # remove outliers ===
    deltas_prev = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
    deltas_next = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
    # Filter valid points where both previous and next distances are less than the threshold
    valid_pts = np.logical_and(deltas_prev < outlier_thresh, deltas_next < outlier_thresh)
    valid_pts[corner_index] = True
    points = points[valid_pts, :]
    contour = contour[valid_pts, :]
    # If valid_pts exists, corner_index needs to be updated synchronously
    _old_indices = np.arange(len(valid_pts))
    _new_indices = np.cumsum(valid_pts) - 1
    corner_index = _new_indices[corner_index]

    # Compute distances between consecutive points ===
    # Close the boundary into a loop
    points = np.vstack([points, points[0]])  # Close the boundary
    contour = np.vstack([contour, contour[0]])

    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)  # Calculate distance between consecutive points

    # Compute cumulative arc length
    arc_lengths = np.insert(np.cumsum(deltas), 0, 0)

    # Total length of the boundary
    total_length = arc_lengths[-1]

    # Number of new points
    num_points = int(np.ceil(total_length / delta)) + 1

    # New equally spaced arc lengths
    new_arc_lengths = np.linspace(0, total_length, num=num_points)

    # Where should the resampled vertices be located between the original vertices?
    # Length normalization (1e-x prevents new_points_insert_idx[-1] from exceeding bounds)
    new_arc_lengths = new_arc_lengths * (arc_lengths[-1] / (new_arc_lengths[-1] + 1e-12))

    # Move the resampled points closest to the original corners onto the corners ===
    corner_arc = arc_lengths[corner_index]    # Parameter corresponding to corner points
    new_corner_index = np.abs(new_arc_lengths[:, None] - corner_arc).argmin(axis=0)  # The new sampled points closest to the corners
    dis_2_old_corner = new_arc_lengths[new_corner_index] - corner_arc
    # Find the closest edge points to these corners in the resampled points.
    min_dis_mask = np.abs(dis_2_old_corner) < delta/3
    # The edge points closest to these corners are moved onto the corners
    new_arc_lengths[new_corner_index[min_dis_mask]] = corner_arc[min_dis_mask]
    # Insert new points for other corners in the resampled points
    new_arc_lengths = np.sort(np.concatenate((new_arc_lengths, corner_arc[~min_dis_mask])))
    # Recalculate the index of each corner point
    new_corner_index = np.abs(new_arc_lengths[:, None] - corner_arc).argmin(axis=0)

    num_points = len(new_arc_lengths)

    # Delete empty fitted edges
    diff = np.abs(new_corner_index - np.roll(new_corner_index, -1)) > 0
    if np.sum(~diff)>0:
        print("Empty approx edge current, deleted.")
    new_corner_index = new_corner_index[diff]

    # Interpolate 3D-points ===
    resampled_points_3D = np.zeros((num_points, 3))
    for i in range(3):
        resampled_points_3D[:, i] = np.interp(new_arc_lengths, arc_lengths, points[:, i])
    # Interpolate 2D-points ===
    resampled_points_2D = np.zeros((num_points, 2))
    for i in range(2):
        resampled_points_2D[:, i] = np.interp(new_arc_lengths, arc_lengths, contour[:, i])
    resampled_points_2D[new_corner_index] = contour[corner_index]

    if TEST:
        A = resampled_points_2D[new_corner_index]
        B = resampled_points_2D

        plt.figure(figsize=(8, 6))

        plt.scatter(B[:, 0], B[:, 1], color='blue', label='Points B')

        plt.scatter(A[:, 0], A[:, 1], color='red', label='Points A')

        plt.legend()

        plt.title('Scatter Plot of Points A and B')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.show()
        input("Press Enter to continue...")

    return resampled_points_3D, resampled_points_2D, new_corner_index, valid_pts#, old_point_param, new_point_param


def vectorize_contour_adaptive_distance(
        contour: np.ndarray,
        smooth_sigma: float = 3.0,
        min_sample_dist: float = 10.0,
        angle_threshold: float = 25.0,
        min_point_dist: int = 10,
        local_max_window: int = 5
    ):
    """
    原始参数，效果不错：
    smooth_sigma: float = 3.0,
    min_sample_dist: float = 10.0,
    angle_threshold: float = 30.0,
    min_point_dist: int = 10,
    local_max_window: int = 5

    针对较近的端点情况进行优化：
    smooth_sigma: float = 2.0,
    min_sample_dist: float = 10.0,
    angle_threshold: float = 30.0,
    min_point_dist: int = 6,
    local_max_window: int = 4
    """

    """
    Contour vectorization algorithm based on variable-length scale curvature calculation.

    :param contour: Nx2 array of boundary points.
    :param smooth_sigma: Standard deviation for Gaussian smoothing.
    :param min_sample_dist: Minimum arc length distance for vector sampling (unit length).
    :param angle_threshold: Minimum angle change threshold (degrees) for preliminary filtering.
    :param min_point_dist: Minimum index distance (number of points) between final control points.
    :param local_max_window: Window size for local maximum search (+/- index).
    :return: List of indices of the selected control points.
    """

    def smooth_contour_convolution(contour: np.ndarray, sigma: float = 3.0) -> np.ndarray:
        """Smooths single contour coordinates using Gaussian filter to remove high-frequency noise."""
        if sigma <= 0:
            return contour.astype(np.float64)

        contour_float = contour.astype(np.float64)
        N = len(contour_float)

        # Do not smooth if too few contour points (less than 2*sigma)
        if N < 2 * sigma:
            return contour_float

        # Smooth along the contour order (axis 0). mode='wrap' handles closed contours
        x_smooth = scipy.ndimage.gaussian_filter1d(contour_float[:, 0], sigma, mode='wrap')
        y_smooth = scipy.ndimage.gaussian_filter1d(contour_float[:, 1], sigma, mode='wrap')
        return np.stack([x_smooth, y_smooth], axis=-1)

    def angle_between_vectors_fixed(vectors1: np.ndarray, vectors2: np.ndarray) -> float:
        """
        Calculates the angle (in degrees) between two 2D vectors, fixed for zero vectors.
        Args: Vectors must be of shape (2,).
        """
        # Flatten vectors for dot product
        v1 = vectors1.flatten()
        v2 = vectors2.flatten()

        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        denominator = norm_v1 * norm_v2

        # Fix for zero or near-zero norm
        if denominator < 1e-8:
            # Treat as no curvature or noise, angle is 0 degrees
            return 0.0

        cos_theta = dot_product / denominator

        # Clamp to [-1.0, 1.0] to handle floating point errors
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        angle_rad = np.arccos(cos_theta)
        return np.degrees(angle_rad)

    # Preprocessing and Robustness Enhancement ===

    # Gaussian smoothing
    smoothed_contour = smooth_contour_convolution(contour, sigma=smooth_sigma)
    N = len(smoothed_contour)
    if N < 2:
        return []

    # Calculate cumulative arc length
    diffs = np.diff(smoothed_contour, axis=0)
    distances = np.linalg.norm(diffs, axis=1)

    # Distance from the last point to the first point (closed contour)
    closure_dist = np.linalg.norm(smoothed_contour[0] - smoothed_contour[-1])

    arc_length = np.concatenate(([0], np.cumsum(distances)))
    total_arc_length = arc_length[-1] + closure_dist  # Full contour length

    if total_arc_length < 2 * min_sample_dist:
        # Contour is too short to sample
        return [0]

    # Variable-Length Scale Curvature Calculation ===

    angles = np.zeros(N)

    for i in range(N):
        current_arc = arc_length[i]

        # Find left neighbor (P_L)
        target_len_L = current_arc - min_sample_dist
        if target_len_L < 0:
            target_len_L += total_arc_length

        # Find the position of the target arc length in the arc_length array, considering the loop
        # Use simple index loop search to ensure robustness

        len_L = 0.0
        idx_L = i
        while len_L < min_sample_dist and idx_L != ((i + 1) % N):
            idx_prev = (idx_L - 1 + N) % N
            dist = np.linalg.norm(smoothed_contour[idx_L] - smoothed_contour[idx_prev])
            len_L += dist
            if len_L >= min_sample_dist: break
            idx_L = idx_prev

        # Find right neighbor (P_R)
        len_R = 0.0
        idx_R = i
        while len_R < min_sample_dist and idx_R != ((i - 1 + N) % N):
            idx_next = (idx_R + 1) % N
            dist = np.linalg.norm(smoothed_contour[idx_R] - smoothed_contour[idx_next])
            len_R += dist
            if len_R >= min_sample_dist: break
            idx_R = idx_next

        # Calculate angle
        v_L = smoothed_contour[i] - smoothed_contour[idx_L]
        v_R = smoothed_contour[idx_R] - smoothed_contour[i]

        angles[i] = angle_between_vectors_fixed(v_L, v_R)

    # Local Maximum Filtering ===

    potential_control_points = []

    for i in range(N):
        # Preliminary Filtering
        if angles[i] < angle_threshold:
            continue

        # Local Maximum Search
        is_local_max = True
        for j in range(-local_max_window, local_max_window + 1):
            if j == 0: continue

            idx = (i + j) % N
            # Must be strictly greater than, not equal to, to ensure a unique peak is found
            if angles[idx] > angles[i]:
                is_local_max = False
                break

        if is_local_max:
            potential_control_points.append({'index': i, 'angle': angles[i]})

    # Distance Constraint and NMS (Non-Maximum Suppression) ===

    # Sort by curvature from largest to smallest
    potential_control_points.sort(key=lambda x: x['angle'], reverse=True)

    final_control_point_indices = set()

    for current_point in potential_control_points:
        idx_curr = current_point['index']

        is_too_close = False
        for idx_final in final_control_point_indices:
            # Distance check: Use the difference in point indices (contour loop distance)
            dist = min(abs(idx_curr - idx_final), N - abs(idx_curr - idx_final))
            if dist < min_point_dist:
                is_too_close = True
                break

        if not is_too_close:
            final_control_point_indices.add(idx_curr)


    TEST = False
    if TEST:
        A = contour[list(final_control_point_indices)]
        B = contour

        plt.figure(figsize=(8, 6))
        plt.scatter(B[:, 0], B[:, 1], color='blue', label='Points B')
        plt.scatter(A[:, 0], A[:, 1], color='red', label='Points A')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.title('Scatter Plot of Points A and B')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        input("Press Enter to continue...")

    return sorted(list(final_control_point_indices))


def dilate_garmage(mask_img, geo_img):
    """ for each point is False on mask_img, find nearst to dilation

    :param mask_img:    RESO x RESO
    :param geo_img:     RESO x RESO x 3
    :return:            result after dilate
    """

    geo_img_dilated = deepcopy(geo_img)
    mask = mask_img

    # Get the coordinates of the points on the boundary/edge
    _, thresh = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_indices = np.concatenate(list(contours)).reshape(-1,2)

    # Geometry values at the boundary points
    geo_contour = geo_img[contour_indices[:, 1], contour_indices[:, 0], :]

    # Coordinates of points that require dilation (invalid/masked-out points)
    unvalid_indices = np.argwhere(~mask_img)
    unvalid_indices = unvalid_indices[...,::-1]

    # Distance matrix between all points needing dilation and all boundary points
    # Calculated using the expansion: (a-b)^2 = a^2 + b^2 - 2ab
    dis = np.sqrt(
        np.sum(unvalid_indices ** 2, axis=1)[:, None] +
        np.sum(contour_indices ** 2, axis=1)[None,:] -
        2 * unvalid_indices @ contour_indices.T)

    # Find the nearest boundary point for each point requiring dilation
    min_dis_indices = np.argmin(dis, axis=1)
    dilation_geo = geo_contour[min_dis_indices]

    # Apply the geometry of the nearest boundary points to the invalid regions
    geo_img_dilated[unvalid_indices[:, 1], unvalid_indices[:, 0], :] = dilation_geo
    return geo_img_dilated


def normalize_and_denormalize_uv(points_2D, panel_instance_seg, uv_bbox):
    """
    Normalizes the 2D points to their local BBOX and then denormalizes them to the UV space defined by uv_bbox.

    :param points_2D:     2D (UV) resampled points
    :param panel_instance_seg:      Determine the panel to which each contour belongs.
    :param uv_bbox:                 2D (UV) bbox of each panel
    :return:
    """

    points_2D = [ri.astype(np.float64) for ri in points_2D]
    nps = [len(ri) for ri in points_2D]
    uv_local = np.concatenate(points_2D)
    nps_cumsum = np.cumsum(nps)

    # Place the points of each panel into the corresponding bbox in uv_bbox ===
    uv_global = np.zeros_like(uv_local, dtype=np.float32)
    bbox_uv_local_prev = None
    for contour_idx in range(len(nps_cumsum)):
        instance_idx = panel_instance_seg[contour_idx]
        if contour_idx == 0: start_point_idx = 0
        else: start_point_idx = nps_cumsum[contour_idx - 1]
        end_point_idx = nps_cumsum[contour_idx]

        all_coords = uv_local[start_point_idx:end_point_idx]
        bbox = uv_bbox[instance_idx]

        if is_contour_OutLine(contour_idx, panel_instance_seg):
            # Calculate the local BBOX for the current outline contour
            bbox_uv_local = np.array([np.min(all_coords[:, 0]), np.min(all_coords[:, 1]),
                                      np.max(all_coords[:, 0]), np.max(all_coords[:, 1])])
            bbox_uv_local_prev = bbox_uv_local
        else:
            # If not an OutLine, use the BBOX of the corresponding OutLine for adjustment
            bbox_uv_local = bbox_uv_local_prev

        uv_bbox_scale = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
        uv_bbox_offset = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]

        # Normalize to local space
        all_coords[:, 0] -= (bbox_uv_local[0] + bbox_uv_local[2]) / 2
        all_coords[:, 1] -= (bbox_uv_local[1] + bbox_uv_local[3]) / 2
        all_coords[:, 0] /= bbox_uv_local[2] - bbox_uv_local[0]
        all_coords[:, 1] /= bbox_uv_local[3] - bbox_uv_local[1]

        # Denormalize to target UV BBOX space
        all_coords[:, 0] *= uv_bbox_scale[0]
        all_coords[:, 1] *= uv_bbox_scale[1]
        all_coords[:, 0] += uv_bbox_offset[0]
        all_coords[:, 1] += uv_bbox_offset[1]

        uv_global[start_point_idx:end_point_idx] = all_coords

    # Place the entire garment into a larger box (Global UV space layout) ===
    uv_bbox_scale = 1000
    uv_bbox_offset = [0, 1000]
    for contour_idx in range(len(nps_cumsum)):
        if contour_idx == 0:
            start_point_idx = 0
        else:
            start_point_idx = nps_cumsum[contour_idx - 1]
        end_point_idx = nps_cumsum[contour_idx]

        all_coords = uv_local[start_point_idx:end_point_idx]
        all_coords *= uv_bbox_scale
        all_coords += uv_bbox_offset

        uv_global[start_point_idx:end_point_idx] = all_coords

    result = []
    for contour_idx in range(len(nps_cumsum)):
        if contour_idx == 0: start_point_idx = 0
        else: start_point_idx = nps_cumsum[contour_idx - 1]
        end_point_idx = nps_cumsum[contour_idx]
        result.append(uv_global[start_point_idx:end_point_idx])
    return result


def extract_boundary_pts(geo_orig, uv_bbox, mask, delta=0.023, RESO=256, erode_size=1, show_2d_approx=False):
    panel_nes = []              # Number of edges on each panel
    contour_nes = []            # Number of edges on each contour
    resampled_points_3D = []    # Resampled 3D points
    resampled_points_2D = []    # Resampled 2D points
    panel_instance_seg = []     # The panel instance to which a contour belongs
                                # A Panel may have multiple contours; they can be merged based on the panel instance
    edge_approx = []            # Approximated edges

    thresh_dict={64:16, 128:32, 256:64, 512:128, 1024:256}
    contour_min_thresh = thresh_dict.get(RESO, 16)

    contour_list = []
    empty_GeoImg_num = 0

    # === Dilate garmage before processing
    geo_dil_list = []

    for panel_idx in range(mask.shape[0]):
        # filter empty GeoImg ===
        geo_dist = np.linalg.norm(geo_orig[panel_idx], axis=-1)
        if geo_dist.min() < 1e-6 and geo_dist.max() < 1e-6:
            empty_GeoImg_num+=1
            continue

        # erode img by mask ===
        mask_img = (mask[panel_idx] * 255.0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_img = cv2.erode(mask_img, kernel, iterations=erode_size)

        geo = geo_orig[panel_idx]
        geo_dil = dilate_garmage(mask_img, geo)
        geo_dil_list.append(geo_dil)

        mask_img = cv2.dilate(mask_img, kernel, iterations=erode_size)

        thresh = 150
        assert thresh >100 and thresh <= 230
        mask_img[mask_img >= thresh] = 255
        mask_img[mask_img < thresh] = 0

        # extract contours by mask ===
        _, thresh = cv2.threshold(mask_img, 128, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # contours_pts = [np.squeeze(contour, axis=1) for contour in contours if contour.shape[0] > 16]

        for contour in contours:
            # filter contour too small
            if contour.shape[0] < contour_min_thresh:
                continue

            contour_list.append(np.squeeze(contour, axis=-2))
            panel_instance_seg.append(panel_idx-empty_GeoImg_num)  # The panel instance to which it belongs

    resized_uv = normalize_and_denormalize_uv(contour_list, panel_instance_seg, uv_bbox)
    geo_orig = np.concatenate([g[None,...] for g in geo_dil_list], axis=0)

    TEST = False
    if TEST:
        A = np.concatenate(resized_uv)

        plt.figure(figsize=(8, 6))
        plt.scatter(A[:, 0], A[:, 1], color='blue', s=1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.title('Scatter Plot.')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        input("Press Enter to continue...")

    all_corners = []

    for contour_idx, contour in enumerate(contour_list):
        p_idx = panel_instance_seg[contour_idx]  # Index of the panel the contour belongs to

        # Get the indices of the corner points
        corner_index = vectorize_contour_adaptive_distance(resized_uv[contour_idx])

        # If there are too few endpoints, select 4 points as endpoints by average sampling ===
        if len(corner_index) <= 2:
            corner_index = np.arange(0, len(contour), len(contour)/4 + 1, dtype=np.int32)
            corner_index = np.unique(corner_index)

        all_corners.append(corner_index)

        # extract boundary points ===
        geo_arr = geo_orig[p_idx]
        geo_sample_pts = geo_arr[contour[:, 1], contour[:, 0], :]

        # resample boundary pts (delta is the resampling interval) ===
        new_points_3D, new_points_2D, new_corner_index, valid_dis_pts = (
                    resample_boundary(geo_sample_pts, contour, corner_index, delta, contour_idx, outlier_thresh=0.05))

        contour = contour[valid_dis_pts]
        contour_list[contour_idx] = contour

        contour_nes.append(len(new_corner_index))  # Number of edges on each Panel (contour)
        resampled_points_3D.append(new_points_3D)  # Sampled points in 3D
        resampled_points_2D.append(new_points_2D)  # Sampled points in 2D
        edge_approx.append(np.array([[new_corner_index[i], new_corner_index[(i+1)%len(new_corner_index)]] for i in range(len(new_corner_index))]))

        if is_contour_OutLine(contour_idx, panel_instance_seg):
            panel_nes.append(len(new_corner_index))
        else:
            panel_nes[-1] += len(new_corner_index)

    TEST = False
    if TEST:
        A = np.concatenate([resized_uv[i][all_corners[i][:]] for i in range(len(resized_uv))])
        B = np.concatenate(resized_uv)

        plt.figure(figsize=(8, 6))
        plt.scatter(B[:, 0], B[:, 1], color='blue',s=1)
        plt.scatter(A[:, 0], A[:, 1], color='red',s=3)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.title('Scatter Plot of boundary segments.')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        input("Press Enter to continue...")

    for e_i, e_ap in enumerate(edge_approx):
        edge_approx[e_i] = edge_approx[e_i][np.argsort(edge_approx[e_i][:, 0])]

    # Check whether invalid results occur. ===
    test = np.concatenate([np.array(e) for e in edge_approx])
    if np.sum(test[:,1]==test[:,0]):
        raise ValueError("Wrong edge approx result...")

    contour_nes = np.array(contour_nes)

    contour_nps = np.array([len(ri) for ri in resampled_points_2D])

    """
    resampled_points_3D:    Extracted 3D boundary points (Contour-wise)
    resampled_points_2D:    Extracted 2D boundary points (Contour-wise)
    contour_nes:        Number of edges on each Contour
    panel_nes:          Number of edges on each Panel
    edge_approx:            Fitted edges for each contour
    panel_instance_seg:     Which Panel each contour belongs to
    contour_nps:            Number of sampled points in each contour
    """
    return resampled_points_3D, resampled_points_2D, contour_nes, np.array(panel_nes), edge_approx, np.array(panel_instance_seg), contour_nps


def resample_by_arc_length(points: np.ndarray, target_count: int) -> np.ndarray:
    """
    Resamples the curve to a target number of points using uniform arc length parametrization.
    Guarantees that the original endpoints are preserved.

    :param points:       N x 2 numpy array of 2D points.
    :param target_count: The desired number of points in the output curve.
    :return:             target_count x 2 numpy array of resampled points.
    """
    N = len(points)
    if N <= 1 or target_count <= 1 or N == target_count:
        return points.copy()

    # 1. Calculate cumulative arc length for parameterization
    distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
    arc_lengths = np.concatenate(([0.0], np.cumsum(distances)))
    total_length = arc_lengths[-1]

    if total_length < 1e-9:
        # Zero length curve
        return np.repeat(points[0][np.newaxis, :], target_count, axis=0)

    # 2. Determine target parameters (uniform arc lengths)
    target_arc_lengths = np.linspace(0, total_length, target_count)

    # 3. Linear interpolation to get resampled coordinates
    # Uses arc_lengths as the x-coordinates for interpolation
    x_resampled = np.interp(target_arc_lengths, arc_lengths, points[:, 0])
    y_resampled = np.interp(target_arc_lengths, arc_lengths, points[:, 1])

    return np.column_stack((x_resampled, y_resampled))


def uniform_sample_by_arc_length(points: np.ndarray, num_samples: int) -> np.ndarray:
    if len(points) < 2 or num_samples <= 0:
        return np.empty((0, 2))

    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_len = arc[-1]

    if total_len == 0:
        return np.repeat(points[:1], num_samples, axis=0)

    targets = np.linspace(0, total_len, num_samples + 2)[1:-1]

    sampled = []
    for t in targets:
        i = np.searchsorted(arc, t) - 1
        i = np.clip(i, 0, len(seg_lengths) - 1)
        alpha = (t - arc[i]) / seg_lengths[i]
        sampled.append((1 - alpha) * points[i] + alpha * points[i + 1])

    return np.vstack(sampled)


def approx_curve(resampled_points_2D: List[np.ndarray], edge_approx: List[List[List[int]]], contour_nps) -> List[List[np.ndarray]]:
    """
    Extracts control points for garment edges using a robust pipeline:
    Arc-Length Resampling -> Moving Average Smoothing -> Global Maximum Error Simplification.

    The output strictly preserves the original input endpoints for boundary precision.
    """
    # Core Parameters
    SMOOTH_ITERATION = 4
    SMOOTH_WINDOW_SIZE = 7
    assert SMOOTH_WINDOW_SIZE%2==1
    K_EXCLUDE = 2  # Number of points excluded near endpoints (applied after resampling)
    MAX_CONTROLS = 12  # Maximum number of internal control points
    TARGET_RESAMPLE_POINTS = 50  # Fixed number of points for arc-length resampling

    garment_approx_curve = []

    def moving_average_smooth(points, window_size, iterations):
        """
        Performs recursive moving average smoothing using 'reflect' padding mode.
        Original endpoints are strictly preserved after each iteration.
        """
        current_points = points.copy()
        L = len(current_points)
        if L < window_size: return current_points
        padding = (window_size - 1) // 2
        original_start_point = points[0].copy()
        original_end_point = points[-1].copy()
        for n in range(iterations):
            # 'reflect' mode padding to handle boundary conditions
            padded_x = np.pad(current_points[:, 0], padding, mode='reflect')
            padded_y = np.pad(current_points[:, 1], padding, mode='reflect')
            if n == 0:
                window = np.ones(window_size) / window_size
            x_smooth = np.convolve(padded_x, window, mode='valid')
            y_smooth = np.convolve(padded_y, window, mode='valid')
            smoothed_points = np.column_stack((x_smooth, y_smooth))
            # Enforce preservation of original endpoints
            smoothed_points[0] = original_start_point
            smoothed_points[-1] = original_end_point
            current_points = smoothed_points
        return current_points

    # --- Main Loop ---
    for contour_idx, contour_edge_approx in enumerate(edge_approx):
        panel_approx_curve = []
        contour_point_2d = resampled_points_2D[contour_idx]

        for e_approx in contour_edge_approx:
            start_idx, end_idx = e_approx[0], e_approx[1]

            # 1. Extract current edge segment points
            if start_idx > end_idx:
                edge_points = np.vstack((contour_point_2d[start_idx:], contour_point_2d[:end_idx + 1]))
            else:
                edge_points = contour_point_2d[start_idx:end_idx + 1]

            N = len(edge_points)
            if N < 2:
                panel_approx_curve.append(edge_points)
                continue

            # 1.5. Arc-Length Resampling to a fixed count (TARGET_RESAMPLE_POINTS)
            resampled_edge_points = resample_by_arc_length(edge_points, TARGET_RESAMPLE_POINTS)
            N_resampled = len(resampled_edge_points)

            # 2. Smooth the resampled curve
            smoothed_points = moving_average_smooth(resampled_edge_points, SMOOTH_WINDOW_SIZE, iterations=SMOOTH_ITERATION)
            # smoothed_points = resampled_edge_points

            # 3. Determine the segment range for simplification (Excluding points near endpoints due to smoothing artifacts)
            middle_start_idx = K_EXCLUDE
            middle_end_idx = N_resampled - K_EXCLUDE

            if middle_start_idx >= middle_end_idx:
                # Keep only the original endpoints if the segment is too short
                final_edge_points = np.vstack((edge_points[0], edge_points[-1]))
            else:
                # Segment ready for simplification
                rdp_segment = smoothed_points[middle_start_idx:middle_end_idx]


                # 4. Uniform sampling between endpoints (no parameter changes)
                control_points = uniform_sample_by_arc_length(
                    rdp_segment,
                    MAX_CONTROLS
                )

                # 5. Construct the final output: Middle points + strictly original endpoints
                if control_points.ndim == 2 and control_points.shape[0] > 0:
                    final_edge_points = np.vstack((edge_points[0], control_points, edge_points[-1]))
                else:
                    final_edge_points = np.vstack((edge_points[0], edge_points[-1]))

            panel_approx_curve.append(final_edge_points)

        garment_approx_curve.append(panel_approx_curve)

    return garment_approx_curve


# === END of control point select algorithm ===


def panel_Layout(garment_approx_curve, uv_bbox, panel_instance_seg, show_layout_panels=False):
    """
    Place each Panel into its corresponding BBOX.

    :param garment_approx_curve:    Endpoints and fitted curves
    :param uv_bbox:                 UV BBOX for each Panel
    :param panel_instance_seg:      Panel corresponding to each Contour
    :param show_layout_panels:
    :return:
    """

    garment_approx_curve = [[u.astype(np.float64) for u in k] for k in garment_approx_curve]

    # Place the points of each panel into the corresponding bbox in uv_bbox ===
    bbox_approx_panel_prev = None
    for contour_idx in range(len(garment_approx_curve)):
        # Get the curve p_approx_curve and the UV bbox of its corresponding Panel (Panel and Contour have a one-to-many relationship)
        p_idx = panel_instance_seg[contour_idx]
        approx_curve = garment_approx_curve[contour_idx]
        bbox = uv_bbox[p_idx]

        if is_contour_OutLine(contour_idx, panel_instance_seg):
            all_coords = np.concatenate(approx_curve, axis=-2)
            bbox_approx_panel = np.array([np.min(all_coords[:, 0]), np.min(all_coords[:, 1]),
                                          np.max(all_coords[:, 0]), np.max(all_coords[:, 1])])
            bbox_approx_panel_prev = bbox_approx_panel
        else:
            bbox_approx_panel = bbox_approx_panel_prev

        uv_bbox_scale = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
        uv_bbox_offset = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]

        for curve in approx_curve:
            # Normalize to local space
            curve[:, 0] -= (bbox_approx_panel[0] + bbox_approx_panel[2]) / 2
            curve[:, 1] -= (bbox_approx_panel[1] + bbox_approx_panel[3]) / 2
            curve[:, 0] /= bbox_approx_panel[2] - bbox_approx_panel[0]
            curve[:, 1] /= bbox_approx_panel[3] - bbox_approx_panel[1]

            # Denormalize to target UV BBOX space
            curve[:, 0] *= uv_bbox_scale[0]
            curve[:, 1] *= uv_bbox_scale[1]
            curve[:, 0] += uv_bbox_offset[0]
            curve[:, 1] += uv_bbox_offset[1]

    # Place the entire garment into a larger box ===
    all = np.concatenate([u for u in [np.concatenate(k, axis=-2) for k in garment_approx_curve]], axis=-2)
    uv_bbox_scale = 1500
    uv_bbox_offset = [0, 1500]
    for contour_idx in range(len(garment_approx_curve)):
        for curve in garment_approx_curve[contour_idx]:
            curve *= uv_bbox_scale
            curve += uv_bbox_offset

    return garment_approx_curve


def get_garment_json(garment_approx_curve, panel_instance_seg, uv_bbox):
    """
    Converts the fitted curves into an AIGP-JSON file that does not contain stitches.

    :param garment_approx_curve:    Endpoints + fitted edges
    :param panel_instance_seg:      Which Panel each contour belongs to
    :param uv_bbox:
    :return:
    """

    garment_json = {"panels": [], "stitches": []}

    for contour_idx in range(len(garment_approx_curve)):
        panel_instance_idx = panel_instance_seg[contour_idx]

        # If this is an OutLine ===
        if is_contour_OutLine(contour_idx, panel_instance_seg):
            p_approx_curve = garment_approx_curve[contour_idx]
            bbox = uv_bbox[panel_instance_idx].tolist()
            panel_json = {
                "center": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                "seqEdges": [],
                "id": get_random_uuid(),
                "label": f"{panel_instance_idx}".zfill(2),
                # "translation": [0.0, 0.0, 0.0],
                # "rotation": [0.0, 0.0, 0.0],
            }
            garment_json["panels"].append(panel_json)
            seqEdge_json = {
                "type": 3,  # Basic line (OutLine)
                "circleType": 0,
                "edges": [],
                "vertices": [],
            }
            p_approx_curve = [pac.tolist() for pac in p_approx_curve]
            for e_idx in range(len(p_approx_curve)):
                edge_json = {
                    "bezierPoints": [[0, 0, 0], [0, 0, 0]],
                    "controlPoints": [[ct_p[0] - panel_json["center"][0], ct_p[1] - panel_json["center"][1], 0] for ct_p in
                                      p_approx_curve[e_idx]],
                    "id": get_random_uuid(),
                }
                seqEdge_json["edges"].append(edge_json)
                seqEdge_json["vertices"].append(edge_json["controlPoints"][0])

            panel_json["seqEdges"].append(seqEdge_json)
        else:
            # Inner line (hole or internal boundary) ===
            panel_json = garment_json["panels"][panel_instance_idx]
            p_approx_curve = garment_approx_curve[contour_idx]
            seqEdge_json = {
                "type": 4,  # Inner line
                "circleType": 1,
                "edges": [],
                "vertices": [],
            }
            for e_idx in range(len(p_approx_curve)):
                edge_json = {
                    "bezierPoints": [[0, 0, 0], [0, 0, 0]],
                    "controlPoints": [[ct_p[0] - panel_json["center"][0], ct_p[1] - panel_json["center"][1], 0] for ct_p in
                                      p_approx_curve[e_idx]],
                    "id": get_random_uuid(),
                }
                seqEdge_json["edges"].append(edge_json)
                seqEdge_json["vertices"].append(edge_json["controlPoints"][0])

            panel_json["seqEdges"].append(seqEdge_json)

    return garment_json


def get_full_uv_info(resampled_points_2D, panel_instance_seg, uv_bbox, contour_nps, show_full_uv=False):
    """
    Get the UV coordinates of the resampled 2D points.
    :param resampled_points_2D:     2D (UV) resampled points
    :param panel_instance_seg:      Determine the panel to which each contour belongs.
    :param uv_bbox:                 2D (UV) bbox of each panel
    :param show_full_uv:            For debugging use
    :return:
    """

    uv_local = np.concatenate(resampled_points_2D)
    nps_cumsum = np.cumsum(contour_nps)

    # Place the points of each panel into the corresponding bbox in uv_bbox ===
    uv_global = np.zeros_like(uv_local, dtype=np.float32)
    bbox_uv_local_prev = None
    for contour_idx in range(len(nps_cumsum)):
        instance_idx = panel_instance_seg[contour_idx]
        if contour_idx == 0: start_point_idx = 0
        else: start_point_idx = nps_cumsum[contour_idx - 1]
        end_point_idx = nps_cumsum[contour_idx]

        all_coords = uv_local[start_point_idx:end_point_idx]
        bbox = uv_bbox[instance_idx]

        if is_contour_OutLine(contour_idx, panel_instance_seg):
            bbox_uv_local = np.array([np.min(all_coords[:, 0]), np.min(all_coords[:, 1]),
                                      np.max(all_coords[:, 0]), np.max(all_coords[:, 1])])
            bbox_uv_local_prev = bbox_uv_local
        else:
            bbox_uv_local = bbox_uv_local_prev  # If not an OutLine, use the BBOX of the corresponding OutLine for adjustment

        uv_bbox_scale = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
        uv_bbox_offset = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]

        # Normalize to local space
        all_coords[:, 0] -= (bbox_uv_local[0] + bbox_uv_local[2]) / 2
        all_coords[:, 1] -= (bbox_uv_local[1] + bbox_uv_local[3]) / 2
        all_coords[:, 0] /= bbox_uv_local[2] - bbox_uv_local[0]
        all_coords[:, 1] /= bbox_uv_local[3] - bbox_uv_local[1]

        # Denormalize to target UV BBOX space
        all_coords[:, 0] *= uv_bbox_scale[0]
        all_coords[:, 1] *= uv_bbox_scale[1]
        all_coords[:, 0] += uv_bbox_offset[0]
        all_coords[:, 1] += uv_bbox_offset[1]

        uv_global[start_point_idx:end_point_idx] = all_coords

    # Place the entire garment into a larger box ===
    uv_bbox_scale = 1000
    uv_bbox_offset = [0, 1000]
    for contour_idx in range(len(nps_cumsum)):
        if contour_idx == 0:
            start_point_idx = 0
        else:
            start_point_idx = nps_cumsum[contour_idx - 1]
        end_point_idx = nps_cumsum[contour_idx]

        all_coords = uv_local[start_point_idx:end_point_idx]
        all_coords *= uv_bbox_scale
        all_coords += uv_bbox_offset

        uv_global[start_point_idx:end_point_idx] = all_coords

    if show_full_uv:
        plt.scatter(x=uv_global[:, 0], y=uv_global[:, 1], s=2)
        plt.show()

    return uv_global


def save_resaults(output_dir, g_idx, data,
                  resampled_points_3D, edge_approx, contour_nes, panel_nes, panel_instance_seg,
                  garment_json, full_uv_info, cfg, garment_fp, g_basename=None):
    """
    Save the results.

    :param output_dir:              Root directory for this batch of output
    :param g_idx:                   Garment index
    :param resampled_points_3D:     3D boundary points
    :param edge_approx:             Approximated edges
    :param contour_nes:         Number of edges on each contour
    :param panel_nes:           Number of edges on each panel
    :param panel_instance_seg:      Which Panel each contour belongs to
    :param garment_json:            Garment AIGP file (panels only)
    :param full_uv_info:            2D-UV position corresponding to each 3D boundary point
    :param cfg:                     Configuration used for resampling
    :return:
    """
    if g_basename is None:
        garment_name = "garment_" + f"{g_idx}".zfill(5)
    else:
        garment_name = "garment_" + g_basename
    garment_dir = os.path.join(output_dir,garment_name)
    os.makedirs(garment_dir, exist_ok=True)

    # save data_info ===
    datainfo_json_save_path = os.path.join(output_dir, "data_info.json")
    with open(datainfo_json_save_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4)

    # save piece_XX.obj files ===
    piece_list = [trimesh.Trimesh(vertices=b_pts, process=False) for b_pts in resampled_points_3D]
    for p_idx, piece in enumerate(piece_list):
        piece_save_path = os.path.join(garment_dir, "piece_" + f"{p_idx}".zfill(2) + ".obj")
        piece.export(piece_save_path)

    # save garment.json file ===
    annotation_dir = os.path.join(garment_dir, "annotations")
    os.makedirs(annotation_dir, exist_ok=True)
    garment_json_save_path = os.path.join(annotation_dir, garment_name + ".json")
    with open(garment_json_save_path, 'w', encoding='utf-8') as f:
        json.dump(garment_json, f, indent=4)

    # save annotations.json file ===
    annotations_json_save_path = os.path.join(annotation_dir, "annotations.json")
    annotations_json = {
        "panel_nes": panel_nes.tolist(),
        "contour_nes": contour_nes.tolist(),
        "edge_approx": np.concatenate(edge_approx).tolist(),
        "panel_instance_seg": panel_instance_seg.tolist(),
        "data_path": garment_fp
    }
    with open(annotations_json_save_path, 'w', encoding='utf-8') as f:
        json.dump(annotations_json, f, indent=4, ensure_ascii=False)

    # save uv info in the full image ===
    if full_uv_info is not None:
        full_uv_info_save_path = os.path.join(annotation_dir, "uv.npy")
        np.save(full_uv_info_save_path, full_uv_info)

    # save original data ===
    # This data is only used in the composite_visualize method in the Jigsaw project for visualization
    original_data_dir = os.path.join(garment_dir, "original_data")
    os.makedirs(original_data_dir, exist_ok=True)

    for k in data:
        if isinstance(data[k], np.ndarray):
            data[k] = data[k].tolist()
    with open(os.path.join(original_data_dir, os.path.basename(garment_fp)), "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str,
        default=None
    )
    parser.add_argument("--delta", type=float, default=0.012)
    parser.add_argument("--erode_size", type=int, default=3)
    parser.add_argument("--shape_dedup", type=bool, default=False)
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = data_dir + "_extracted"
    os.makedirs(output_dir, exist_ok=True)

    """
    Keys in cfg:
        RESO: Resolution of GeoImg
        delta: Resampling interval
        
    feature convK55D11:
        data_path_list = sorted(glob(os.path.join(data_dir, "*.pkl")))
        garment_num = len(data_path_list)
        cfg = {"RESO":256, "delta": 0.012, "erode_size":3}
        + 6 INF NOISE * 1 Iter
    """
    data_path_list = sorted(glob(os.path.join(data_dir, "*.pkl")), key=lambda x: os.path.splitext(os.path.basename(x))[0])
    garment_num = len(data_path_list)
    cfg = {"RESO":256, "delta": args.delta, "erode_size":args.erode_size}

    for g_idx in tqdm(range(0, garment_num), desc=f"Processing data: "):

        garment_fp = data_path_list[g_idx]
        g_basename = os.path.basename(garment_fp).replace(".pkl","")

        try:   g_idx = int(garment_fp.split("/")[-1].split(".")[-2])
        except Exception:   g_idx = g_idx

        # read data
        geo_orig, mask, uv_bbox, data = load_data(garment_fp, save_vis=False, shape_dedup=args.shape_dedup)

        # extract boundary points => resample boundary points => split boundary into edges (contour_nes)
        resampled_points_3D, resampled_points_2D, contour_nes, panel_nes, edge_approx, panel_instance_seg, contour_nps = (
            extract_boundary_pts(geo_orig, uv_bbox, mask, delta=cfg["delta"], RESO=cfg["RESO"], erode_size=cfg.get("erode_size", 3), show_2d_approx=False))

        # Denormalize resampled_points_2D to UV coordinates based on uv_bbox
        full_uv_info = get_full_uv_info(resampled_points_2D, panel_instance_seg, uv_bbox, contour_nps, show_full_uv=False)
        # [test]
        # pointcloud_visualize([np.pad(resampled_points_2D[0], (0, 1)), np.pad(resampled_points_2D[0][edge_approx[0][:, 0]], (0, 1))], colornum=5, color_norm=[-1, 1])


        # Get sampling points for curve fitting
        approx_curve_samples = approx_curve(resampled_points_2D, edge_approx, contour_nps)

        # # Denormalize the vector diagram of each Panel to UV coordinates based on uv_bbox
        garment_approx_curve = panel_Layout(approx_curve_samples, uv_bbox, panel_instance_seg)

        # Get the json file (panels only)
        garment_json = get_garment_json(garment_approx_curve, panel_instance_seg, uv_bbox)

        save_resaults(
            output_dir,
            g_idx,
            data,
            resampled_points_3D,
            edge_approx,
            contour_nes,
            panel_nes,
            panel_instance_seg,
            garment_json,
            full_uv_info,
            cfg, garment_fp, g_basename=g_basename
        )
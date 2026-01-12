
import json
import uuid
import torch
import numpy as np
from copy import deepcopy
from functools import cmp_to_key
from itertools import groupby
from utils import is_contour_OutLine


def get_random_uuid():
    id = str(uuid.uuid4().hex)
    result = id[0:8] + "-" + id[8:12] + "-" + id[12:16] + "-" + id[16:20] + "-" + id[20:]
    return result


def apply_point_info(stitch_edge_side, point_info):
    stitch_edge_side['clothPieceId'] = point_info['contour_id']
    stitch_edge_side['edgeId'] = point_info['edge_id']
    stitch_edge_side['param'] = point_info['param']


def get_new_stitch():
    stitch = [
        {
            "end": {"clothPieceId": None, "edgeId": None, "param": None},
            "isCounterClockWise": None,
            "start": {"clothPieceId": None, "edgeId": None, "param": None}
        },
        {
            "end": {"clothPieceId": None, "edgeId": None, "param": None},
            "isCounterClockWise": None,
            "start": {"clothPieceId": None, "edgeId": None, "param": None}
        }
    ]
    return stitch


def get_new_stitch_edge(start_point=None, end_point=None, target_edge=None, isCC=None):
    stitch_edge = {
        "target_edge":target_edge,
        "start_point":start_point.copy(),
        "end_point":end_point.copy(),
        "isCC":isCC
    }
    return stitch_edge


def filter_too_short(all_stitch_points_list, isCC_order_list=None, fliter_len = 1):
    """
    filter stitch with too less points.
    :param all_stitch_points_list:
    :param isCC_order_list:
    :param fliter_len:
    :return:
    """
    all_stitch_points_list_ = []
    if isCC_order_list is None:
        for s in all_stitch_points_list:
            if len(s)>fliter_len:
                all_stitch_points_list_.append(s)
        return all_stitch_points_list_, None
    else:
        isCC_order_list_ = []
        for s, c in zip(all_stitch_points_list, isCC_order_list):
            if len(s)>fliter_len:
                all_stitch_points_list_.append(s)
                isCC_order_list_.append(c)
        return all_stitch_points_list_, isCC_order_list_


def cal_stitch_edge_param_dis(stitch_edge, all_contour_info):
    """
    Compute the param distance of a stitch edge
    :param stitch_edge:
    :param all_contour_info:
    :return:
    """
    contour_info = all_contour_info[stitch_edge['start_point']['contour_id']]
    start_key = "start_point" if not stitch_edge["isCC"] else "end_point"
    end_key = "end_point" if not stitch_edge["isCC"] else "start_point"
    start_point = stitch_edge[start_key]
    end_point = stitch_edge[end_key]
    # Compute the param difference between the two points
    if end_point['global_param'] >= start_point['global_param']:
        param_dis = end_point['global_param'] - start_point['global_param']
    else:
        param_dis = (contour_info['param_end'] - start_point['global_param'] +
                      end_point['global_param'] - contour_info['param_start'])
    return param_dis


def cal_stitch_edge_middle_index(stitch_edge, all_contour_info):
    """
    Compute the point ID at the middle position of a stitch edge
    :param stitch_edge:
    :param all_contour_info:
    :return:
    """
    contour_info = all_contour_info[stitch_edge['start_point']['contour_id']]
    start_key = "start_point" if not stitch_edge["isCC"] else "end_point"
    end_key = "end_point" if not stitch_edge["isCC"] else "start_point"
    start_point = stitch_edge[start_key]
    end_point = stitch_edge[end_key]
    # Compute the index distance between the two points
    if end_point['id'] >= start_point['id']:
        index_dis = end_point['id'] - start_point['id']
    else:
        index_dis = (contour_info['index_end'] - start_point['id'] +
                      end_point['id'] - contour_info['index_start']) + 1
    middle_index = start_point['id'] + torch.trunc(index_dis / 2)
    if middle_index > contour_info['index_end']:
        middle_index = middle_index % contour_info['index_end'] + contour_info['index_start']
    return middle_index


def cal_stitch_edge_index_dis(stitch_edge, all_contour_info):
    """
    :param stitch_edge:
    :param all_contour_info:
    :return:
    """
    contour_info = all_contour_info[stitch_edge['start_point']['contour_id']]
    start_key = "start_point" if not stitch_edge["isCC"] else "end_point"
    end_key = "end_point" if not stitch_edge["isCC"] else "start_point"
    start_point = stitch_edge[start_key]
    end_point = stitch_edge[end_key]
    # Compute the index distance between the two points
    if end_point['id'] >= start_point['id']:
        index_dis = end_point['id'] - start_point['id']
    else:
        index_dis = (contour_info['index_end'] - start_point['id'] +
                      end_point['id'] - contour_info['index_start']) + 1
    return index_dis


def cal_neigbor_points_param_dis(start_point, end_point, all_contour_info):
    """
    Compute the bidirectional param distance between two points on a loop
    (used only for calculating the distance between relatively close points)
    :param start_point:
    :param end_point:
    :param all_contour_info:
    :return:
    """
    contour_info = all_contour_info[start_point['contour_id']]
    if start_point['contour_id']!=end_point['contour_id']:
        return None
    if end_point['global_param'] >= start_point['global_param']:
        param_dis=[
          contour_info['param_end'] - start_point['global_param'] + end_point['global_param'] - contour_info['param_start'],
          end_point['global_param'] - start_point['global_param']
        ]
    else:
        param_dis = [
          start_point['global_param'] - end_point['global_param'],
          contour_info['param_end'] - start_point['global_param'] + end_point['global_param'] - contour_info['param_start']
        ]
    return param_dis


def global_param2point_id(global_param, contour_info):
    """
    Given the input global_param and contour_info, find the (floating-point) point_id
    corresponding to this global_param, as well as the two nearest points
    :param global_param:
    :param contour_info:
    :return:
    """
    dis_tensor=abs(contour_info["all_points_global_param"] - global_param)
    values, indices = torch.topk(dis_tensor, 2, largest=False)
    first_near, second_near = contour_info["all_points_info"][indices[0]], contour_info["all_points_info"][indices[1]]

    if values[0] < 1e-5:
        point_idx_float = first_near["id"]
    else:
        point_idx_float = (first_near["id"] * (values[1] ) + second_near["id"] * values[0]) / (values[0] + values[1] )

    return  point_idx_float, first_near, second_near


def cal_neigbor_points_index_dis(start_point, end_point, all_contour_info):
    """
    Compute the bidirectional index distance between two points on a loop
    (used only for calculating the distance between relatively close points)

    :param start_point:
    :param end_point:
    :param all_contour_info:
    :return:
    """
    contour_info = all_contour_info[start_point['contour_id']]
    if start_point['contour_id']!=end_point['contour_id']:
        return None
    loop_len = contour_info['index_end'] - contour_info['index_start'] + 1

    if end_point['id'] >= start_point['id']:
        index_dis = [
            loop_len - (end_point['id'] - start_point['id']),  # counter-clockwise distance
            end_point['id'] - start_point['id']  # clockwise distance
        ]
    else:
        index_dis = [
            start_point['id'] - end_point['id'],  # clockwise distance (end < start in this case)
            loop_len - (start_point['id'] - end_point['id'])  # counter-clockwise distance
        ]

    index_dis = [i.item() for i in index_dis]
    return index_dis


def find_nearst_pattern_side_point(point_info, all_edge_info, all_contour_info):
    """
    find nearst pattern side point of point_info
    :return:
    """
    edge_points = all_edge_info[point_info["edge_id"]]["points_info"]
    edge_side_points = [edge_points[0], edge_points[-1]]

    dis_stitch_side_point2edge_side_points = [
        min(cal_neigbor_points_index_dis(point_info, edge_side_points[0], all_contour_info)),
        min(cal_neigbor_points_index_dis(point_info, edge_side_points[1], all_contour_info))
    ]
    nearst_edge_side_point = edge_side_points[dis_stitch_side_point2edge_side_points.index(min(dis_stitch_side_point2edge_side_points))]

    return nearst_edge_side_point, min(dis_stitch_side_point2edge_side_points)


def optimize_stitch_edge_list_byNeighbor(stitch_edge_list_paramOrder, all_contour_info, all_edge_info,
                                         optimize_thresh_neighbor_index_dis=6,
                                         optimize_thresh_side_index_dis=4):
    """
    TODO: this algorithm needs to be redesigned
    Optimize the relationship between neighboring stitch edges
    :param stitch_edge_list_paramOrder:
    :param all_contour_info:
    :param all_edge_info:
    :param optimize_thresh_neighbor_index_dis: threshold for optimizing distances between stitch edges
                                                (pairs of stitch edges whose distance is smaller than this threshold
                                                 will have their endpoints optimized)
    :param optimize_thresh_side_index_dis: threshold between stitch edges and edge endpoints
                                           (stitch edge endpoints within this threshold will be snapped to edge endpoints)
    :return:
    """
    # todo
    """
    Traverse each stitch edge on this contour:
        For the two endpoints (A, B) of a stitch edge:
            One stitch edge endpoint (A) finds the nearest panel endpoint (P)
            and other stitch edge endpoints (C_1-N)
            This stitch edge endpoint finds all stitch edges close to itself,
            taking at most one endpoint from each stitch edge
            If (A) lies on a panel edge endpoint, all points move to the edge endpoint
            Otherwise, find which target points are among them
        
        Need to consider the case where multiple endpoints are extremely close
    """
    current_contour_info = all_contour_info[stitch_edge_list_paramOrder[0]['start_point']['contour_id']]
    for se_idx, stitch_edge in enumerate(stitch_edge_list_paramOrder):
        if se_idx == 0:
            stitch_edge_previous = stitch_edge_list_paramOrder[-1]
        else:
            stitch_edge_previous = stitch_edge_list_paramOrder[se_idx - 1]

        # Get the potentially connectable parts of the two edges ----------------------------------------------
        # Definition: right is in the relative clockwise direction, left is in the relative counter-clockwise direction

        # The two points at the connection (pre_right_point, cur_left_point), and the two farther points ===
        cur_left_key = "start_point" if not stitch_edge["isCC"] else "end_point"
        cur_right_key = "start_point" if stitch_edge["isCC"] else "end_point"
        cur_left_point = stitch_edge[cur_left_key]  # Among the two endpoints of the current edge, the one in the relative counter-clockwise direction
        cur_right_point = stitch_edge[cur_right_key]

        pre_left_key = "end_point" if stitch_edge_previous["isCC"] else "start_point"
        pre_right_key = "end_point" if not stitch_edge_previous["isCC"] else "start_point"
        pre_left_point = stitch_edge_previous[pre_left_key]
        pre_right_point = stitch_edge_previous[pre_right_key]  # Among the two endpoints of the previous edge, the one in the relative clockwise direction

        index_dis_pre_right_cur_left_2side = cal_neigbor_points_index_dis(pre_right_point, cur_left_point, all_contour_info)
        param_dis_pre_right_cur_left_2side = cal_neigbor_points_param_dis(pre_right_point, cur_left_point, all_contour_info)
        min_dis_index_pre_right_cur_left = 0 if index_dis_pre_right_cur_left_2side[0] < index_dis_pre_right_cur_left_2side[1] else 1  # index of the smaller distance side
        index_dis_pre_right_cur_left = index_dis_pre_right_cur_left_2side[min_dis_index_pre_right_cur_left]
        param_dis_pre_right_cur_left = param_dis_pre_right_cur_left_2side[min_dis_index_pre_right_cur_left]

        dis_pre_right2cur_left = min(index_dis_pre_right_cur_left_2side)
        dis_pre_left2cur_left = min(cal_neigbor_points_index_dis(pre_left_point, cur_left_point, all_contour_info))
        dis_pre_right2cur_right = min(cal_neigbor_points_index_dis(pre_right_point, cur_right_point, all_contour_info))


        # The two panel edge endpoints closest to the two endpoints at the current connection ===
        pre_right_point_nearst_edge_side_point, pre_right_point_nearst_edge_side_dis = find_nearst_pattern_side_point(pre_right_point, all_edge_info, all_contour_info)
        cur_left_point_nearst_edge_side_point, cur_left_point_nearst_edge_side_dis = find_nearst_pattern_side_point(cur_left_point, all_edge_info, all_contour_info)

        optimized = False
        # If the distance between the two stitch edge endpoints at the connection is below the threshold,
        # and there is no other endpoint closer to the connection
        if dis_pre_right2cur_left < optimize_thresh_neighbor_index_dis:
            # Misalignment exists
            if dis_pre_right2cur_left >= dis_pre_left2cur_left or dis_pre_right2cur_left >= dis_pre_right2cur_right:
                optimized = False
            # Both points are on panel edge endpoints
            elif pre_right_point_nearst_edge_side_dis < 1 and pre_right_point_nearst_edge_side_dis < 1:
                optimized = False
            # Only one point is on a panel edge endpoint
            elif (pre_right_point_nearst_edge_side_dis < 1) ^ (pre_right_point_nearst_edge_side_dis < 1):
                target_side_point = pre_right_point_nearst_edge_side_point \
                    if pre_right_point_nearst_edge_side_dis < 1 else cur_left_point_nearst_edge_side_point
                target_global_param = target_side_point["global_param"]
                optimized = True
            # Neither point is on an endpoint; take the midpoint
            else:
                # There is a gap between the two edges
                if min_dis_index_pre_right_cur_left == 1:
                    target_global_param = pre_right_point["global_param"] + param_dis_pre_right_cur_left / 2
                    optimized = True
                # There is overlap between the two edges
                else:
                    # Use a smaller threshold for overlap cases
                    if index_dis_pre_right_cur_left <= max(optimize_thresh_neighbor_index_dis / 4, 2):
                        target_global_param = cur_left_point["global_param"] + param_dis_pre_right_cur_left / 2
                        optimized = True
                    else:
                        optimized = False
        else:
            optimized = False

        # Apply the result
        if optimized:
            # If it crosses into another contour, correct it
            if target_global_param > current_contour_info["param_end"]:
                target_global_param = current_contour_info["param_start"] + target_global_param - current_contour_info["param_end"]

            # Find the point closest to the target global_param position
            target_point_id, first_close_point, second_close_point = global_param2point_id(target_global_param, current_contour_info)
            target_edge_info = all_edge_info[first_close_point["edge_id"]]  # Get the information of the edge it lies on
            target_param = target_global_param - target_edge_info["param_start"]  # Local param

            # Assign the computed new target position to both stitch edges
            for point in [pre_right_point, cur_left_point]:
                point["edge_id"] = target_edge_info["id"]
                point["global_param"] = target_global_param
                point["param"] = target_param
                point["id"] = target_point_id


def optimize_stitch_edge_list_byApproxEdge(stitch_edge_list ,thresh, all_contour_info, all_edge_info):
    """
    Adjust the positions of stitch edge endpoints that are very close to panel edge endpoints
    and are not connected to other stitch edges.
    :param stitch_edge_list:
    :param thresh:
    :param all_contour_info:
    :param all_edge_info:
    :return:
    """

    # Stitch edges belonging to the same contour will be sorted according to the position of their middle point
    cmp_fun = lambda x: (cal_stitch_edge_middle_index(x, all_contour_info))

    stitch_edge_list_contourOrder = sorted(stitch_edge_list, key=lambda x: (x["start_point"]["contour_id"],))
    stitch_edge_list_paramOrder = []
    start_contour_id = stitch_edge_list_contourOrder[0]["start_point"]["contour_id"]
    for e_idx, stitch_edge in enumerate(stitch_edge_list_contourOrder):
        if stitch_edge["start_point"]["contour_id"] != start_contour_id or e_idx == len(stitch_edge_list_contourOrder) - 1:
            # If the last one's contour_id has not changed
            if not stitch_edge["start_point"]["contour_id"] != start_contour_id and e_idx == len(stitch_edge_list_contourOrder) - 1:
                stitch_edge_list_paramOrder.append(stitch_edge)

            # [At this moment, all stitch edges in stitch_edge_list_paramOrder lie on the same contour]
            # Sort this stitch_edge_list_paramOrder
            stitch_edge_list_paramOrder = sorted(stitch_edge_list_paramOrder, key=cmp_fun)

            # For stitch edges on the same contour, try to find attachable panel edge endpoints --------------
            for se_idx, start_stitch_edge in enumerate(stitch_edge_list_paramOrder):
                # The previous edge of the current start_stitch_edge in the clockwise direction
                if se_idx == 0:
                    start_stitch_edge_previous = stitch_edge_list_paramOrder[-1]
                else:
                    start_stitch_edge_previous = stitch_edge_list_paramOrder[se_idx - 1]

                # === Get the potentially connectable parts of the two edges ===
                # Definition: right is in the relative clockwise direction, left is in the relative counter-clockwise direction
                pre_right_key = "end_point" if not start_stitch_edge_previous["isCC"] else "start_point"
                cur_left_key = "start_point" if not start_stitch_edge["isCC"] else "end_point"
                pre_right_point = start_stitch_edge_previous[pre_right_key]  # Among the two endpoints of the previous edge, the one in the relative clockwise direction
                cur_left_point = start_stitch_edge[cur_left_key]  # Among the two endpoints of the current edge, the one in the relative counter-clockwise direction

                # The other two points
                pre_left_key = "end_point" if start_stitch_edge_previous["isCC"] else "start_point"
                cur_right_key = "start_point" if start_stitch_edge["isCC"] else "end_point"
                pre_left_point = start_stitch_edge_previous[pre_left_key]
                cur_right_point = start_stitch_edge[cur_right_key]

                # For the clockwise start point of the current stitch edge and the clockwise end point of the previous edge,
                # find attachable panel edge endpoints respectively
                for point in [pre_right_point, cur_left_point]:
                    edge = all_edge_info[point["edge_id"]]
                    edge_side_point = [edge["points_info"][0], edge["points_info"][-1]]
                    # Distance from the stitch edge endpoint to the two endpoints of the edge it lies on
                    side_point_index_dis = [
                        min(cal_neigbor_points_index_dis(point, edge_side_point[0], all_contour_info)),
                        min(cal_neigbor_points_index_dis(point, edge_side_point[1], all_contour_info)),
                    ]

                    """
                    For a stitch edge endpoint, it will first try to merge with the nearest endpoint
                    on the panel vector edge it lies on.
                    If the merge fails, it will then try the other one that is slightly farther away.
                    """
                    optimized = False
                    side_point_index_dis_index = 0 if side_point_index_dis[0] < side_point_index_dis[1] else 1
                    for i in range(2):
                        if 0 <= side_point_index_dis[side_point_index_dis_index] <= thresh:
                            panel_side_point = edge_side_point[side_point_index_dis_index]
                            # For a panel endpoint, if it is closer to pre_left_point or cur_right_point, it will be skipped
                            if (min(cal_neigbor_points_index_dis(panel_side_point, point, all_contour_info)) >
                                    min(cal_neigbor_points_index_dis(panel_side_point, pre_left_point, all_contour_info))):
                                continue
                            if (min(cal_neigbor_points_index_dis(panel_side_point, point, all_contour_info)) >
                                    min(cal_neigbor_points_index_dis(panel_side_point, cur_right_point, all_contour_info))):
                                continue
                            point["id"] = panel_side_point["id"]
                            point["param"] = panel_side_point["param"]
                            point["global_param"] = panel_side_point["global_param"]
                            optimized = True

                        if optimized:
                            break
                        # Switch to the other, slightly farther stitch edge endpoint
                        side_point_index_dis_index = 1 - side_point_index_dis_index

            # Switch to the next contour
            start_contour_id = stitch_edge["start_point"]["contour_id"]
            stitch_edge_list_paramOrder = [stitch_edge]
        else:
            stitch_edge_list_paramOrder.append(stitch_edge)


def apply_stitch_param(st_eg, param):
    st_eg['clothPieceId'] = param['panel_id']
    st_eg['edgeId'] = param['edge_id']
    st_eg['param'] = param['param']


def load_data(batch):
    garment_json_path = batch["garment_json_path"][0]
    annotations_json_path = batch["annotations_json_path"][0]

    with open(garment_json_path, "r") as gf, open(annotations_json_path,"r") as af:
        garment_json = json.load(gf)
        annotations_json = json.load(af)

    panel_nes = torch.tensor(annotations_json["panel_nes"], dtype=torch.int64, device=batch["pcs"].device)
    contour_nes = torch.tensor(annotations_json["contour_nes"], dtype=torch.int64, device=batch["pcs"].device)
    edge_approx = torch.tensor(annotations_json["edge_approx"], dtype=torch.int64, device=batch["pcs"].device)
    panel_instance_seg = torch.tensor(annotations_json["panel_instance_seg"], dtype=torch.int64, device=batch["pcs"].device)

    return garment_json, panel_nes, contour_nes, edge_approx, panel_instance_seg


def find_best_endpoints_np_left(stitch_points_info, contour_info, is_cc):
    """
    Given a series of points that may not be well-ordered and a clockwise/counter-clockwise direction,
    find the start point and end point among them.
    """

    pts = torch.tensor([p["id"] for p in stitch_points_info]).detach().cpu().numpy()
    contour_index_start = contour_info['index_start'].detach().cpu().numpy()
    contour_len = (contour_info['index_end'] - contour_info['index_start'] + 1).detach().cpu().numpy()

    # Ensure input is ndarray for convenient parallel computation
    if not isinstance(pts, np.ndarray):
        pts = np.array(pts, dtype=np.int64)

    # Half of the length of pts
    half = (len(pts) + 1) // 2

    # Indices on the circular contour
    idx_targets = (pts - contour_index_start) % contour_len
    # All possible start points, limited to the first half
    # (most start/end points appear in this range)
    if half is not None:
        idx_starts = idx_targets[:half]  # shape (half,)

    # Compute the number of steps from each start to each target
    if is_cc:
        dist = (idx_starts[:, None] - idx_targets[None, :]) % contour_len
    else:
        dist = (idx_targets[None, :] - idx_starts[:, None]) % contour_len

    # Steps required for each start point to cover all targets
    lengths_cover = dist.max(axis=1) + 1  # shape (half,)

    # Choose the point that requires the minimum steps to cover all targets as the start endpoint
    best_i = int(np.argmin(lengths_cover))
    best_start_idx = int(idx_starts[best_i])
    best_start = contour_index_start + best_start_idx
    best_length = int(lengths_cover[best_i])

    # For the selected start point, choose the last point on the path that covers all input points
    # (based on is_cc) as the end endpoint
    far_j = int(np.argmax(dist[best_i]))
    best_end_idx = int(idx_targets[far_j])
    best_end = contour_index_start + best_end_idx

    start_point_info = stitch_points_info[best_i]
    end_point_info = stitch_points_info[far_j]

    return start_point_info, end_point_info


def pointstitch_2_edgestitch(batch, inf_rst, stitch_mat, stitch_indices,
                              unstitch_thresh = 6, fliter_len=3, division_thresh = 6,
                              optimize_thresh_neighbor_index_dis = 4,
                              optimize_thresh_side_index_dis=2):

    """
    Derive edge-to-edge stitching from point-to-point stitching

    :param batch:                       # garment_jigsaw model input
    :param inf_rst:                     # garment_jigsaw model output
    :param stitch_mat:                  # matrix of point-to-point stitches
    :param stitch_indices:              # indices of point-to-point stitches
    :param unstitch_thresh:             # unstitched point sequences longer than this will form a new stitch point list
    :param fliter_len:                  # stitch point lists shorter than or equal to this will be filtered out
    :param division_thresh              # if the distance between adjacent points on a stitch edge exceeds this threshold,
                                        # it will be split into two stitch edges
    :param optimize_thresh_neighbor_index_dis:   # neighboring stitch edges with param distance below this will be optimized
    :return:
    """

    # Load data (V2 AIGP file and edge fitting results) ------------------------------------------------------
    garment_json, panel_nes, contour_nes, edge_approx, panel_instance_seg = load_data(batch)
    device_ = stitch_mat.device

    pcs = batch["pcs"][0]
    n_pcs = batch["n_pcs"][0]
    piece_id = batch["piece_id"][0]
    contour_num = torch.sum(n_pcs!=0)
    n_pcs = n_pcs[:contour_num]

    n_pcs_cumsum = torch.cumsum(n_pcs, dim=-1)
    contour_nes_cumsum = torch.cumsum(contour_nes, dim=-1)

    # Convert fitted edge indices to global coordinates
    edge_approx_global = deepcopy(edge_approx)
    for contour_idx in range(len(n_pcs_cumsum)):
        if contour_idx == 0: edge_start_idx = 0
        else: edge_start_idx = contour_nes_cumsum[contour_idx - 1]
        edge_end_idx = contour_nes_cumsum[contour_idx]
        if contour_idx!=0: edge_approx_global[edge_start_idx:edge_end_idx] += n_pcs_cumsum[contour_idx-1]

    # Aggregate key information for each point, edge, and contour --------------------------------------------
    all_contour_info = {}
    all_edge_info = {}
    all_point_info = {}
    very_small_gap = 1e-5
    global_param = 0  # global param of a point
    contourid2panelid = {}  # map contour_id to panel_id

    for contour_idx in range(len(n_pcs_cumsum)):

        panel_instance_idx = panel_instance_seg[contour_idx]
        if contour_idx == 0:
            point_start_idx = 0
            edge_start_idx = 0
        else:
            point_start_idx = n_pcs_cumsum[contour_idx - 1]
            edge_start_idx = contour_nes_cumsum[contour_idx - 1]
        point_end_idx = n_pcs_cumsum[contour_idx]
        edge_end_idx = contour_nes_cumsum[contour_idx]

        # Determine whether this is an outline contour (isOL)
        isOL, contour_pos = is_contour_OutLine(contour_idx, panel_instance_seg)

        # === Build information for each panel ===
        # Points on this contour
        contour_points = pcs[point_start_idx:point_end_idx]
        # Local / global fitted edges
        contour_edges_approx = edge_approx[edge_start_idx:edge_end_idx]
        contour_edges_approx_global = edge_approx_global[edge_start_idx:edge_end_idx]
        # Number of fitted edges and points on this contour
        contour_edge_num = contour_edges_approx.shape[-2]
        contour_point_num = contour_points.shape[-2]
        panel_json = garment_json["panels"][panel_instance_idx]

        if isOL: contour_id = panel_json["id"]
        else: contour_id = get_random_uuid()

        all_contour_info[contour_id] = {"id":contour_id, "edges_info":{}}
        contour_info = all_contour_info[contour_id]
        contour_info["param_start"] = global_param  # minimum param of this panel
        contour_info["param_end"] = []
        contour_info["index_end"] = point_end_idx-1
        contour_info["index_start"] = point_start_idx if isinstance(point_start_idx,torch.Tensor) \
                                      else torch.tensor(point_start_idx, dtype=point_end_idx.dtype,device=point_end_idx.device)

        seqEdges_json = panel_json["seqEdges"][contour_pos]
        contourid2panelid[contour_id] = panel_json["id"]
        for edge_idx, (e_approx, e_approx_global) in enumerate(zip(contour_edges_approx, contour_edges_approx_global)):
            # === Build information for each edge ===
            edge_json = seqEdges_json["edges"][edge_idx]
            all_edge_info[edge_json["id"]] = {"id":edge_json["id"], "contour_id":contour_info["id"], "points_info":[]}
            edge_info = all_edge_info[edge_json["id"]]
            contour_info["edges_info"][edge_json["id"]] = edge_info
            # Get ordered point indices for this edge on the panel
            if ((edge_idx==0 and e_approx_global[0]>e_approx_global[1]) or (edge_idx==contour_edge_num-1 and e_approx_global[0]>e_approx_global[1])):
                edge_points_idx = torch.concat([torch.arange(e_approx_global[0], point_end_idx), torch.arange(point_start_idx, e_approx_global[1]+1)])
            else:
                edge_points_idx = torch.arange(e_approx_global[0], e_approx_global[1]+1)

            # Minimum param of this edge
            edge_info["param_start"] = global_param
            for idx, point_idx in enumerate(edge_points_idx):
                # === Build information for each point ===
                # Compute param of the current point (a very small offset is added to avoid errors,
                # and these offsets will be fixed in later processing)
                param = idx / (len(edge_points_idx) - 1)  # relative position of this point on the edge
                if idx==0: param+=very_small_gap
                elif idx==len(edge_points_idx)-1: param-=very_small_gap

                point_info = {"id": point_idx, "contour_id": contour_info["id"], "edge_id": edge_info["id"],
                              "param": param, "global_param": global_param + param, "is_stitch": False}
                edge_info["points_info"].append(point_info)

                pt_key = point_idx.tolist()
                if not pt_key in all_point_info.keys():
                    all_point_info[pt_key] = [point_info]
                else:  # duplicate point (intersection of two edges)
                    all_point_info[pt_key].append(point_info)

            # After all points on this edge are processed, increment global_param by 1 for the next edge
            global_param+=1
        # contour_info["all_points_info"] will be frequently accessed during stitch optimization,
        # so it is precomputed here
        contour_info["all_points_info"] = [point for e in contour_info["edges_info"] for point in contour_info["edges_info"][e]["points_info"]]
        contour_info["all_points_global_param"] = torch.tensor([point["global_param"] for point in contour_info["all_points_info"]])
        # Maximum param of this panel
        contour_info["param_end"] = global_param

    # Obtain N lists of point-to-point stitches --------------------------------------------------------------
    """ 
    For each contour, obtain all point-to-point stitches on it
    Group these point-to-point stitches by the contour_id of the stitched endpoints
    For each group, break it at locations where the spacing is too large
    """

    # Point-to-point stitch map
    stitch_map = torch.zeros((len(pcs)), dtype=torch.int64, device=device_)-1
    # Only take the first half; otherwise importing into the software will cause bugs
    stitch_map[stitch_indices[:, 0]] = stitch_indices[:, 1]
    # stitch_map[stitch_indices[:, 1]] = stitch_indices[:, 0]

    # Traverse all stitch point pairs on this panel and mark these points as stitched
    for contour_id in all_contour_info:
        # Get edge information on this contour
        contour_info = all_contour_info[contour_id]
        contour_edges_info = contour_info["edges_info"]

        for edge_id in contour_edges_info:
            # Get point information on this edge
            edge_info = contour_edges_info[edge_id]
            edge_points_info = edge_info["points_info"]

            for idx, point_info in enumerate(edge_points_info):
                # Current point and its stitched counterpart
                point_idx = point_info["id"]
                point_idx_cor = stitch_map[point_idx]

                # Mark them as stitch points
                if point_idx.item()>=0 and point_idx_cor.item()>=0:
                    for p_l in [all_point_info[point_idx.item()], all_point_info[point_idx_cor.item()]]:
                        for p in p_l:
                            p["is_stitch"] = True

    all_stitch_points_list = []
    for contour_id in all_contour_info:
        contour_info = all_contour_info[contour_id]
        contour_edges_info = contour_info["edges_info"]

        # Collect all point-to-point stitches originating from the same contour ===
        contour_stitch_points = []
        contour_stitch_list = []
        for edge_id in contour_edges_info:
            edge_info = contour_edges_info[edge_id]
            edge_points_info = edge_info["points_info"]
            for idx, point_info in enumerate(edge_points_info):
                contour_stitch_points.append(point_info)
        contour_stitch_points = [p for p in contour_stitch_points if stitch_map[p["id"].item()].item()>=0]

        for point_info in contour_stitch_points:
            point_info_cor = all_point_info[stitch_map[point_info["id"].item()].item()]

            if len(point_info_cor)==1:
                contour_stitch_list.append([point_info, point_info_cor])
            elif len(point_info_cor)==2:
                contour_stitch_list.append([point_info, [point_info_cor[0]]])
                # contour_stitch_list.append([point_info, [point_info_cor[1]]])

        # ==================

        # === Build nodes ===
        # Each stitch pair is treated as a node
        nodes = []
        nodes_info = []
        contour_stitch_list_sorted = sorted(contour_stitch_list, key=lambda p: (p[1][0]["contour_id"]))
        for stitch in contour_stitch_list_sorted:
            nodes_info.append([stitch[0]["id"].item(), stitch[1][0]["id"].item()])
            nodes.append(stitch)
        node_count = len(nodes_info)

        # === Build adjacency lists for both ends of each stitch ===
        # If the distance between the left ends of any two nodes is below the threshold,
        # they are connected in adj1; similarly for the right ends
        adj1 = [set() for _ in range(node_count)]   # graph for left ends
        adj2 = [set() for _ in range(node_count)]   # graph for right ends

        for i in range(node_count):
            for j in range(i + 1, node_count):
                if nodes[i][1][0]["contour_id"] != nodes[j][1][0]["contour_id"]:
                    continue
                # Distance between left ends and all other nodes
                d1 = min(cal_neigbor_points_index_dis(
                    nodes[i][0], nodes[j][0], all_contour_info))
                if d1 <= unstitch_thresh:
                    adj1[i].add(j)
                    adj1[j].add(i)

                # Distance between right ends and all other nodes
                d2 = min(cal_neigbor_points_index_dis(
                    nodes[i][1][0], nodes[j][1][0], all_contour_info))
                if d2 <= unstitch_thresh:
                    adj2[i].add(j)
                    adj2[j].add(i)

        # === Check whether a subgraph is connected ===
        def is_connected_subset(subset, adj):
            if not subset:
                return True
            S = set(subset)
            start = next(iter(S))
            visited = {start}
            dq = [start]
            while dq:
                u = dq.pop()
                for v in adj[u]:
                    if v in S and v not in visited:
                        visited.add(v)
                        dq.append(v)
            return visited == S

        # === Perform double-graph connectivity clustering ===
        remaining = set(range(node_count))
        groups = [] # stores clustering results of stitches originating from the current contour

        while True:
            if len(remaining) < 2:
                break

            made_group = False
            rem_list = sorted(remaining)

            for ai in range(len(rem_list)):
                i = rem_list[ai]
                for aj in range(ai + 1, len(rem_list)):
                    j = rem_list[aj]

                    S = {i, j}

                    expanded = True
                    while expanded:
                        expanded = False
                        for v in sorted(remaining - S):
                            if is_connected_subset(S | {v}, adj1) and \
                                    is_connected_subset(S | {v}, adj2):
                                S.add(v)
                                expanded = True
                                break

                    if len(S) >= 2:
                        groups.append([nodes[x] for x in sorted(S)])
                        remaining -= S
                        made_group = True
                        break
                if made_group:
                    break

            if not made_group:
                break

        all_stitch_points_list.extend(groups)

    # Filter out overly short stitch lists -------------------------------------------------------------------
    """
    The point-to-point stitches obtained in the previous step may include very short ones
    """
    all_stitch_points_list, _ = filter_too_short(all_stitch_points_list, fliter_len = fliter_len)

    # Convert stitched points from list to dict --------------------------------------------------------------
    for s_idx, stitch_points_list in enumerate(all_stitch_points_list):
        for sp_idx, st_point in enumerate(stitch_points_list):
            if isinstance(st_point[1], list):
                st_point[1] = st_point[1][0]

    # 为每个缝合计算时针方向 ------------------------------------------------------------------------------------------------
    """
    对于一个缝合边上的点，根据它们的param判断这条缝合边的时针方向
    但是由于被缝合边上的点可能不是那么有序（这来源于预测的点点缝合不完美）
    因此，我计算 索引距离\in[1,gap]的 相邻的缝合点之间 顺+逆时针方向的间距
    对于每一对相邻缝合点之间时针方向，累积后得到大致正确的缝合边时针方向
    """
    isCC_order_list = []
    for s_idx, stitch_points_list in enumerate(all_stitch_points_list):
        isCC_order_list.append([None,None])
        # 缝边和被缝边分别计算时针方向
        for i in range(2):
            order_sum = 0
            # max_gap = min(3, len(stitch_points_list))
            max_gap=1
            for gap in range(1, max_gap+1):
                for sp_idx, st_point in enumerate(stitch_points_list):
                    if sp_idx < gap: continue
                    # 计算双向的param_dis
                    dis_d = cal_neigbor_points_param_dis(st_point[i], stitch_points_list[sp_idx - gap][i], all_contour_info)
                    ord = 1 if dis_d[1] >= dis_d[0] else -1
                    order_sum += ord
            # order_sum>=0时，我们认为这个缝边是顺时针的
            isCC_order_list[-1][i] = order_sum < 0


    # 对于每一个缝合边上的缝合点进行聚类排序 ------------------------------------------------------------------------
    """
    这一步的目的是为了在 “一个边上的点间距过长，则拆分边” 步骤时，能够有效的找到拆分点
    
    对于一个缝合边上的缝合点，如果unstitch_thresh的值设的过大，可能会包含多个缝合边
    如果其中某一个缝合边经过contour的起终点，缝合边上的点直接用id进行排序会导致错误
    因此在排序时，用点与点在contour上顺+逆时针上的距离作为依据，并且不处理间距过大的，实现聚类的效果
    """
    def compare_fun(x1, x2):
        # 或取顺逆时针方向，点x1 到 点x2 的间距
        y1, y2 = cal_neigbor_points_index_dis(x1, x2, all_contour_info)
        # 顺+逆时针间距都过大：不进行排序（从而实现聚类的效果）
        if min(y1, y2) > 8:
            return 0
        # 逆时针间距小：返回正数：x1 应该排在 x2 后面
        if y1 < y2:
            return 1
        # 顺时针间距小：返回负数：x1 应该排在 x2 前面
        elif y1 > y2:
            return -1
        # 顺+逆时针间距相等：保持原有相对顺序
        else:
            return 0

    judge_key = "contour_id"
    for s_idx, stitch_points_list in enumerate(all_stitch_points_list):
        unordered_stitch_points_list = []
        start_contour_id = stitch_points_list[0][1][judge_key]
        start_point_idx = 0
        for point_idx, stitch_point in enumerate(stitch_points_list):
            # 如果edge_id发生了变化，或是这个Panel结束了
            if stitch_point[1][judge_key] != start_contour_id or point_idx == len(stitch_points_list)-1:
                if stitch_point[1][judge_key] == start_contour_id and point_idx == len(stitch_points_list)-1:
                    unordered_stitch_points_list.append(stitch_point[1])

                # 先按照顺时针进行聚类排序，如果标记为逆时针，则将排序后的
                if isCC_order_list[s_idx][1]: reverse = True
                else: reverse = False

                ordered_stitch_points_list = deepcopy(sorted(unordered_stitch_points_list, key=cmp_to_key(compare_fun), reverse=reverse))
                for i in range(len(unordered_stitch_points_list)):
                    stitch_points_list[start_point_idx + i][1] = ordered_stitch_points_list[i]

                start_point_idx = point_idx
                start_contour_id = stitch_point[1][judge_key]
                unordered_stitch_points_list = [stitch_point[1]]
            else:
                unordered_stitch_points_list.append(stitch_point[1])

    # 一个边上的点间距过长，则拆分边 -------------------------------------------------------------------------------------------------
    thresh = division_thresh
    new_all_stitch_points_list = []
    new_isCC_order_list = []
    for idx, stitch_points_list in enumerate(all_stitch_points_list):

        stitch_point_stack = []
        for point_idx, stitch_point in enumerate(stitch_points_list):
            if len(stitch_point_stack) == 0:
                stitch_point_stack.append(stitch_point)
            else:
                dis = np.array([
                    min(cal_neigbor_points_index_dis(stitch_point_stack[-1][0], stitch_point[0], all_contour_info)),
                    min(cal_neigbor_points_index_dis(stitch_point_stack[-1][1], stitch_point[1], all_contour_info))
                ])
                dis_mask = dis<thresh
                if np.sum(dis_mask)==2:
                    stitch_point_stack.append(stitch_point)
                elif np.sum(dis_mask)==1:
                    new_all_stitch_points_list.append(stitch_point_stack)
                    new_isCC_order_list.append(isCC_order_list[idx])
                    stitch_point_stack = []
                    stitch_point_stack.append(stitch_point)
                else:
                    new_all_stitch_points_list.append(stitch_point_stack)
                    new_isCC_order_list.append(isCC_order_list[idx])
                    stitch_point_stack = []
                    stitch_point_stack.append(stitch_point)
        if len(stitch_point_stack) != 0:
            new_all_stitch_points_list.append(stitch_point_stack)
            new_isCC_order_list.append(isCC_order_list[idx])

    all_stitch_points_list = new_all_stitch_points_list
    isCC_order_list = new_isCC_order_list

    # 将太短的全部过滤掉 ---------------------------------------------------------------------------------------------------
    """
    拆分边时，会拆出特别短的，这些需要剔除
    """
    all_stitch_points_list, isCC_order_list = filter_too_short(all_stitch_points_list, isCC_order_list, fliter_len = fliter_len) # [modified]

    # [todo]一个缝合边上的相邻点之间的平均间距过长，则删除边 ------------------------------------------------------------------------
    thresh = 5
    new_all_stitch_points_list = []
    new_isCC_order_list = []
    for idx, stitch_points_list in enumerate(all_stitch_points_list):
        from_dis_list = []
        to_dis_list = []
        for point_idx, stitch_point in enumerate(stitch_points_list[1:]):
            from_dis = cal_neigbor_points_index_dis(
                stitch_points_list[point_idx-1][0], stitch_points_list[point_idx][0], all_contour_info)
            from_dis_list.append(min(from_dis))
            to_dis = cal_neigbor_points_index_dis(
                stitch_points_list[point_idx-1][1], stitch_points_list[point_idx][1], all_contour_info)
            to_dis_list.append(min(to_dis))
        from_dis_mean = np.mean(from_dis_list)
        to_dis_mean = np.mean(to_dis_list)
        if from_dis_mean < thresh and to_dis_mean < thresh:
            new_all_stitch_points_list.append(stitch_points_list)
            new_isCC_order_list.append(isCC_order_list[idx])
        else:
            pass
    all_stitch_points_list = new_all_stitch_points_list
    isCC_order_list = new_isCC_order_list

    # [todo]一个边上的点点缝合的平均距离过大（3D间距），则删除边 -----------------------------------------------------------------

    # 根据缝合距离，检测是否有缝边的cc预测错了 ----------------------------------------------------------------------------------
    for idx, stitch_points_list in enumerate(all_stitch_points_list):
        from_indices = torch.tensor([p[0]["id"] for p in stitch_points_list]).detach().cpu().numpy()
        to_indices = torch.tensor([p[1]["id"] for p in stitch_points_list]).detach().cpu().numpy()
        from_pcs = pcs[from_indices]
        to_pcs = pcs[to_indices]

        # 计算当前情况下，缝合距离之和
        diff = from_pcs-to_pcs
        dist = np.linalg.norm(diff, axis=1).sum()
        # 计算被缝合边反向情况下，缝合距离之和
        diff_rev = from_pcs - torch.flip(to_pcs, dims=[0])
        dist_rev = np.linalg.norm(diff_rev, axis=1).sum()

        # 如果反向后缝合距离更短了，大概率是因为之前预测反了
        if dist_rev < dist:
            isCC_order_list[idx][1] = not isCC_order_list[idx][1]
            stitch_points_list_to_rev = deepcopy([p[1] for p in stitch_points_list[::-1]])
            for i in range(len(stitch_points_list)):
                stitch_points_list[i][1] = stitch_points_list_to_rev[i]

    # 获取缝合信息 --------------------------------------------------------------------------------------------------------
    # 用于存放所有缝合边信息(用于后续优化)
    stitch_edge_list = []
    for s_idx, stitch_points_list in enumerate(all_stitch_points_list):
        # 【方法1】如果前面进行了排序，则不用这个
        from_stitch_points = [p[0] for p in stitch_points_list]
        from_contour_info = all_contour_info[from_stitch_points[0]["contour_id"]]
        to_stitch_points = [p[1] for p in stitch_points_list]
        to_contour_info = all_contour_info[to_stitch_points[0]["contour_id"]]
        # 边1
        from_start_point_info, from_end_point_info = find_best_endpoints_np_left(from_stitch_points, from_contour_info, isCC_order_list[s_idx][0])
        # 边2（被缝合）
        to_start_point_info, to_end_point_info = find_best_endpoints_np_left(to_stitch_points, to_contour_info, isCC_order_list[s_idx][1])
        stitch_edge_start = get_new_stitch_edge(from_start_point_info, from_end_point_info, isCC=isCC_order_list[s_idx][0])
        stitch_edge_end = get_new_stitch_edge(to_start_point_info, to_end_point_info, isCC=isCC_order_list[s_idx][1])
        stitch_edge_start["target_edge"] = stitch_edge_end
        stitch_edge_end["target_edge"] = stitch_edge_start
        stitch_edge_list.append(stitch_edge_start)
        stitch_edge_list.append(stitch_edge_end)


    # 将 离边端点特别近 且 不与其它缝边相衔接 的缝边的端点 的位置进行调整 ---------------------------------------------------------
    optimize_stitch_edge_list_byApproxEdge(stitch_edge_list, max(optimize_thresh_side_index_dis, 2), all_contour_info, all_edge_info)

    # 将长度特别短的缝边删除 ------------------------------------------------------------------------------------------------
    thresh = 0.06
    filtered_stitch_edge_list = []
    for start_stitch_edge, end_stitch_edge in zip(stitch_edge_list[::2], stitch_edge_list[1::2]):
        param_dis_st = cal_stitch_edge_param_dis(start_stitch_edge, all_contour_info)
        param_dis_ed = cal_stitch_edge_param_dis(end_stitch_edge, all_contour_info)
        if param_dis_st<thresh or param_dis_ed<thresh:
            continue
        filtered_stitch_edge_list.append(start_stitch_edge)
        filtered_stitch_edge_list.append(end_stitch_edge)
    stitch_edge_list = filtered_stitch_edge_list

    # 优化相邻缝合的param -------------------------------------------------------------------------------------------------
    stitch_edge_list_contourOrder = sorted(stitch_edge_list, key=lambda x: x["start_point"]["contour_id"])
    cmp_fun = lambda x: (cal_stitch_edge_middle_index(x, all_contour_info))
    for contour_id, edge_group in groupby(stitch_edge_list_contourOrder, key=lambda x: x["start_point"]["contour_id"]):
        edge_group = list(edge_group)
        edge_group = sorted(edge_group, key=cmp_fun)
        optimize_stitch_edge_list_byNeighbor(edge_group, all_contour_info, all_edge_info,
                                             optimize_thresh_neighbor_index_dis=optimize_thresh_neighbor_index_dis,
                                             optimize_thresh_side_index_dis=optimize_thresh_side_index_dis)

    # 将长度特别短的缝边删除 ------------------------------------------------------------------------------------------------
    thresh = 0.06
    filtered_stitch_edge_list = []
    for start_stitch_edge, end_stitch_edge in zip(stitch_edge_list[::2], stitch_edge_list[1::2]):
        param_dis_st = cal_stitch_edge_param_dis(start_stitch_edge, all_contour_info)
        param_dis_ed = cal_stitch_edge_param_dis(end_stitch_edge, all_contour_info)
        if param_dis_st<thresh or param_dis_ed<thresh:
            continue
        filtered_stitch_edge_list.append(start_stitch_edge)
        filtered_stitch_edge_list.append(end_stitch_edge)
    stitch_edge_list = filtered_stitch_edge_list

    # 将缝合两端都完全衔接的相邻的缝合进行合并 ---------------------------------------------------------------------------------
    max_iter = 5
    for it in range(max_iter):
        changed = False

        stitch_edge_list_merged = []
        # 按所在板片进行排序后的 stitch_edge_list
        stitch_edge_list_contourOrder = sorted(stitch_edge_list[::2], key=lambda x: (x["start_point"]["contour_id"],))
        stitch_edge_list_paramOrder = []
        start_contour_id = stitch_edge_list_contourOrder[0]["start_point"]["contour_id"]
        for e_idx, stitch_edge in enumerate(stitch_edge_list_contourOrder):
            if stitch_edge["start_point"]["contour_id"] != start_contour_id or e_idx == len(stitch_edge_list_contourOrder) - 1:
                # 如果最后一个的contour_id没变化
                if stitch_edge["start_point"]["contour_id"] == start_contour_id and e_idx == len(stitch_edge_list_contourOrder) - 1:
                    stitch_edge_list_paramOrder.append(stitch_edge)

                # 【此时，stitch_edge_list_paramOrder中所有的缝边都位于同一contour上】
                # 对这个stitch_edge_list_paramOrder根据global_param进行排序（isCC=false根据起始点排序，isCC=true根据终点点排序）
                stitch_edge_list_paramOrder = sorted(stitch_edge_list_paramOrder, key=lambda x: (x["start_point"]["global_param"] if not x["isCC"] else x["end_point"]["global_param"]))

                # 对同一contour上的缝边，尝试合并它们 ------------------------------------------------------------------------------
                for se_idx, start_stitch_edge in enumerate(stitch_edge_list_paramOrder):
                    if se_idx == 0:
                        start_stitch_edge_previous_index = -1
                        start_stitch_edge_previous = stitch_edge_list_paramOrder[start_stitch_edge_previous_index]
                    else:
                        start_stitch_edge_previous_index = se_idx - 1
                        start_stitch_edge_previous = stitch_edge_list_paramOrder[start_stitch_edge_previous_index]

                    # === 获取两个边的可能衔接的部分 ===
                    # 定义：right是位于相对顺时针的方向，left是位于相对逆时针的方向
                    pre_right_key = "end_point" if not start_stitch_edge_previous["isCC"] else "start_point"
                    cur_left_key = "start_point" if not start_stitch_edge["isCC"] else "end_point"
                    pre_right_point = start_stitch_edge_previous[pre_right_key]  # 上一条边的两个端点中，处于相对顺时针方向的点
                    cur_left_point = start_stitch_edge[cur_left_key]  # 当前边的两个端点中，处于相对逆时针方向的点

                    # 其它部分的俩个点
                    pre_left_key = "end_point" if start_stitch_edge_previous["isCC"] else "start_point"
                    cur_right_key = "start_point" if start_stitch_edge["isCC"] else "end_point"
                    pre_left_point = start_stitch_edge_previous[pre_left_key]
                    cur_right_point = start_stitch_edge[cur_right_key]

                    # 计算相邻 缝边 的双向间距
                    st_param_dis_d = cal_neigbor_points_param_dis(pre_right_point, cur_left_point, all_contour_info)
                    if st_param_dis_d and min(st_param_dis_d) < 0.001:
                        end_stitch_edge = start_stitch_edge["target_edge"]
                        end_stitch_edge_previous = start_stitch_edge_previous["target_edge"]
                        # 计算相邻 被缝边 的双向间距
                        ed_param_dis_d = cal_neigbor_points_param_dis(end_stitch_edge_previous[pre_right_key], end_stitch_edge[cur_left_key], all_contour_info)
                        if ed_param_dis_d and min(ed_param_dis_d) < 0.001:
                            stitch_edge_list_paramOrder[se_idx][cur_left_key] = start_stitch_edge_previous[pre_left_key]
                            stitch_edge_list_paramOrder[se_idx]["target_edge"][cur_left_key] = end_stitch_edge_previous[pre_left_key]
                            del stitch_edge_list_paramOrder[start_stitch_edge_previous_index]
                            changed=True

                for se in stitch_edge_list_paramOrder:
                    stitch_edge_list_merged.extend([se, se["target_edge"]])

                # 如果最后一个的contour_id有变化
                if stitch_edge["start_point"]["contour_id"] != start_contour_id and e_idx == len(stitch_edge_list_contourOrder) - 1:
                    stitch_edge_list_merged.extend([stitch_edge, stitch_edge["target_edge"]])

                # 切换到下一个 contour
                start_contour_id = stitch_edge["start_point"]["contour_id"]
                stitch_edge_list_paramOrder = [stitch_edge]
            else:
                stitch_edge_list_paramOrder.append(stitch_edge)
        stitch_edge_list = stitch_edge_list_merged

        if not changed:
            break

    for stitch_edge in stitch_edge_list:
        for k in ['start_point', 'end_point']:
            stitch_edge[k]["panel_id"] = contourid2panelid[stitch_edge[k]["contour_id"]]

    # change format
    stitch_edge_json_list = []
    for start_stitch_edge, end_stitch_edge in zip(stitch_edge_list[::2], stitch_edge_list[1::2]):
        stitch_edge_json = get_new_stitch()

        stitch_edge_json[0]["isCounterClockWise"] = start_stitch_edge["isCC"]
        stitch_edge_json[1]["isCounterClockWise"] = end_stitch_edge["isCC"]

        apply_stitch_param(stitch_edge_json[0]["start"], start_stitch_edge["start_point"])
        apply_stitch_param(stitch_edge_json[0]["end"], start_stitch_edge["end_point"])
        apply_stitch_param(stitch_edge_json[1]["start"], end_stitch_edge["start_point"])
        apply_stitch_param(stitch_edge_json[1]["end"], end_stitch_edge["end_point"])

        stitch_edge_json_list.append(stitch_edge_json)

    garment_json["stitches"] = stitch_edge_json_list

    results = {
        "garment_json":garment_json,
        }

    return results
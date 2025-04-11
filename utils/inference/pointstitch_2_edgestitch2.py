# 从点点缝合关系获取边边缝合关系
# 第二版，为了将单一板片多个contour的情况考虑进去
import json
import uuid
import torch
import numpy as np
from copy import deepcopy

from numpy.compat import contextlib_nullcontext

from utils import is_contour_OutLine

# 获取一段随机的 uuid
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

# 过滤掉太短的缝合
def filter_too_short(all_stitch_points_list, isCC_order_list=None, fliter_len = 1):
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

# 判断两个点有没有关联性
def is_valid_stitch_point(stitch_points, stitch_points_compare):
    edge_id_list = [sp["edge_id"] for sp in stitch_points_compare[1]]
    trigger = False
    for p_idx, point in enumerate(stitch_points[1]):
        if point["edge_id"] in edge_id_list:
            trigger = True
            break
    return trigger, p_idx


# 计算一个缝合边的param_dis
def cal_stitch_edge_param_dis(stitch_edge, all_contour_info):
    contour_info = all_contour_info[stitch_edge['start_point']['contour_id']]
    start_key = "start_point" if not stitch_edge["isCC"] else "end_point"
    end_key = "end_point" if not stitch_edge["isCC"] else "start_point"
    start_point = stitch_edge[start_key]
    end_point = stitch_edge[end_key]
    # 计算两个点的param的差距
    if end_point['global_param'] >= start_point['global_param']:
        param_dis = end_point['global_param'] - start_point['global_param']
    else:
        param_dis = (contour_info['param_end'] - start_point['global_param'] +
                      end_point['global_param'] - contour_info['param_start'])
    return param_dis


# 计算一个环上的两个点之间的双向param_dis(仅用于计算较近的点之间的距离)
def cal_neigbor_points_param_dis(start_point, end_point, all_contour_info):
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


# 根据输入的global_param + contour_info，找到这个global_param对应的（浮点数的）point_id，以及最近的俩 point
def global_param2point_id(global_param, contour_info):
    dis_tensor=abs(contour_info["all_points_global_param"] - global_param)
    values, indices = torch.topk(dis_tensor, 2, largest=False)
    first_near, second_near = contour_info["all_points_info"][indices[0]], contour_info["all_points_info"][indices[1]]

    # # [todo] 如果first_near的global_param在contour的某一端上，将second_near设置为另一端
    # if first_near==contour_info['all_points_global_param'][0] and global_param <= first_near["global_param"]:
    #     second_near = contour_info['all_points_global_param'][-1]
    #     values[1] =
    #
    # elif first_near==contour_info['all_points_global_param'][-1] and global_param >= first_near["global_param"]:
    #     second_near = contour_info['all_points_global_param'][0]

    if values[0] < 1e-5:
        point_idx_float = first_near["id"]
    else:
        point_idx_float = (first_near["id"] * (values[1] ) + second_near["id"] * values[0]) / (values[0] + values[1] )

    return  point_idx_float, first_near, second_near


# 计算一个环上的两个点之间的双向index_dis(仅用于计算较近的点之间的距离)
def cal_neigbor_points_index_dis(start_point, end_point, all_contour_info):
    contour_info = all_contour_info[start_point['contour_id']]
    if start_point['contour_id']!=end_point['contour_id']:
        return None
    if end_point['id'] >= start_point['id']:
        index_dis=[
          contour_info['index_end'] - start_point['id'] + end_point['id'] - contour_info['index_start'],
          end_point['id'] - start_point['id']
        ]
    else:
        index_dis = [
          start_point['id'] - end_point['id'],
          contour_info['index_end'] - start_point['id'] + end_point['id'] - contour_info['index_start']
        ]
    index_dis = [i.item() for i in index_dis]
    return index_dis


def optimize_stitch_edge_list_byNeighbor(stitch_edge_list_paramOrder, all_contour_info, all_edge_info,
                                         optimize_thresh_neighbor_index_dis=6,
                                         optimize_thresh_side_index_dis=4):
    """

    :param stitch_edge_list_paramOrder:
    :param all_contour_info:
    :param all_edge_info:
    :param optimize_thresh_neighbor_index_dis:       优化缝边之间距离的阈值（间距小于这一阈值的一对缝边，它们之间的端点会被优化）
    :param optimize_thresh_side_index_dis:           缝边与边端点之间的阈值（小于阈值的缝边的端点将在优化时吸附到边的端点上）
    :return:
    """
    current_contour_info = all_contour_info[stitch_edge_list_paramOrder[0]['start_point']['contour_id']]
    for se_idx, stitch_edge in enumerate(stitch_edge_list_paramOrder):
        if se_idx == 0:
            stitch_edge_previous = stitch_edge_list_paramOrder[-1]
        else:
            stitch_edge_previous = stitch_edge_list_paramOrder[se_idx - 1]

        # 获取两个边的可能衔接的部分 ---------------------------------------------------------------------------------------------------
        # 定义：right是位于相对顺时针的方向，left是位于相对逆时针的方向
        pre_right_key = "end_point" if not stitch_edge_previous["isCC"] else "start_point"
        cur_left_key = "start_point" if not stitch_edge["isCC"] else "end_point"
        pre_right_point = stitch_edge_previous[pre_right_key]  # 上一条边的两个端点中，处于相对顺时针方向的点
        cur_left_point = stitch_edge[cur_left_key]  # 当前边的两个端点中，处于相对逆时针方向的点

        # # 其它部分的俩点
        pre_left_key = "end_point" if stitch_edge_previous["isCC"] else "start_point"
        cur_right_key = "start_point" if stitch_edge["isCC"] else "end_point"
        pre_left_point = stitch_edge_previous[pre_left_key]
        cur_right_point = stitch_edge[cur_right_key]

        # 计算这两个点的 index距离 和 param差距 ---------------------------------------------------------------------------------------
        index_dis_2side = cal_neigbor_points_index_dis(pre_right_point, cur_left_point, all_contour_info)
        param_dis_2side = cal_neigbor_points_param_dis(pre_right_point, cur_left_point, all_contour_info)
        min_dis_index = 0 if index_dis_2side[0] < index_dis_2side[1] else 1     # 距离较小的那一端的 index
        index_dis = index_dis_2side[min_dis_index]
        param_dis = param_dis_2side[min_dis_index]

        # 设定阈值，根据采样频率delta来动态调整阈值的大小 ---------------------------------------------------------------------------------
        adjust_param = 1  # 这个用于根据采样频率调整阈值的大小 [todo] 预留，可在这里将采样频率考虑进来
        optimize_thresh_neighbor_index_dis = optimize_thresh_neighbor_index_dis * adjust_param
        optimize_thresh_side_index_dis = optimize_thresh_side_index_dis * adjust_param

        # 对于端点距离 index_dis 小于阈值的缝边进行优化 ---------------------------------------------------------------------------------
        optimized = False  # 是否已经进行过优化了
        # === 检测两个缝边离端点的距离 ===
        # 同一 edge 上的所有点
        edge_points_pre_right = all_edge_info[pre_right_point["edge_id"]]["points_info"]
        edge_points_cur_left = all_edge_info[cur_left_point["edge_id"]]["points_info"]
        # 找到 edge 上最近的端点
        side_point_index_dis = [
            [
                # 前一条缝边的右点 到其所在边的两个端点的距离
                min(cal_neigbor_points_index_dis(pre_right_point, edge_points_pre_right[0], all_contour_info)),
                min(cal_neigbor_points_index_dis(pre_right_point, edge_points_pre_right[-1], all_contour_info))
            ],
            [
                # 当前缝边的左点 到其所在边的两个端点的距离
                min(cal_neigbor_points_index_dis(cur_left_point, edge_points_cur_left[0], all_contour_info)),
                min(cal_neigbor_points_index_dis(cur_left_point, edge_points_cur_left[-1], all_contour_info))
            ]
        ]
        # 获取 前一条缝边的右点 和 当前缝边的左点 离各自边上两个端点的距离
        side_point_index_dis_index = [
            0 if side_point_index_dis[0][0] < side_point_index_dis[0][1] else 1,
            0 if side_point_index_dis[1][0] < side_point_index_dis[1][1] else 1,
        ]
        # 获取离 前一条缝边的右点 和 当前缝边的左点 最近的两个端点
        side_point_closet =[
            edge_points_pre_right[0] if side_point_index_dis_index[0]==0 else edge_points_pre_right[-1],
            edge_points_cur_left[0]  if side_point_index_dis_index[1]==0 else edge_points_cur_left[-1],
        ]
        # 计算 前一条边的右点 和 当前边的左点 离最近端点的距离
        pre_right_dis2side = min(cal_neigbor_points_index_dis(pre_right_point, side_point_closet[0], all_contour_info))
        cur_left_dis2side = min(cal_neigbor_points_index_dis(cur_left_point, side_point_closet[1], all_contour_info))

        target_global_param_list = []
        # 如果缝边间距小于阈值，且存在缝边端点离拟合边端点距离小于阈值，则将这一拟合边端点位置设为优化后的位置 -----------------------------------------------------------------------------------
        if not optimized:
            if index_dis < optimize_thresh_neighbor_index_dis/2:
                # 如果两个相邻缝边上的某个点离边的端点的距离小于阈值，则将这个边的端点设定为目标位置
                if pre_right_dis2side < optimize_thresh_side_index_dis or cur_left_dis2side < optimize_thresh_side_index_dis:
                    order = ["pre", "cur"] if pre_right_dis2side < cur_left_dis2side else  ["cur", "pre"]
                    for o in order:
                        # 如果仅一个点离拟合边端点近端点
                        if o=="pre" and pre_right_dis2side < optimize_thresh_side_index_dis and cur_left_dis2side >= optimize_thresh_side_index_dis:
                            target_side_point, second_target = side_point_closet[0], side_point_closet[1]
                            pre_left_point_dis_2side = cal_neigbor_points_index_dis(pre_left_point, target_side_point, all_contour_info)    # 缝边左点 到 端点 的双向距离
                            pre_right_point_dis_2side = cal_neigbor_points_index_dis(pre_right_point, target_side_point, all_contour_info)  # 缝边右点 到 端点 的双向距离
                            cur_right_point_dis_2side = cal_neigbor_points_index_dis(cur_right_point, target_side_point, all_contour_info)  # 缝边右点 到 端点 的双向距离
                            if not (min(pre_right_point_dis_2side) < min(pre_left_point_dis_2side) and min(pre_right_point_dis_2side) < min(cur_right_point_dis_2side)):
                                pass
                            else:
                                target_global_param = target_side_point["global_param"]
                                target_global_param_list = [target_global_param, target_global_param]
                                optimized = True
                        # 如果仅一个点离拟合边端点近端点
                        elif o=="cur" and cur_left_dis2side < optimize_thresh_side_index_dis and pre_right_dis2side >= optimize_thresh_side_index_dis:
                            target_side_point, second_target = side_point_closet[1], side_point_closet[0]
                            cur_left_point_dis_2side = cal_neigbor_points_index_dis(cur_left_point, target_side_point, all_contour_info)    # 缝边左点 到 端点 的双向距离
                            cur_right_point_dis_2side = cal_neigbor_points_index_dis(cur_right_point, target_side_point, all_contour_info)  # 缝边右点 到 端点 的双向距离
                            pre_left_point_dis_2side = cal_neigbor_points_index_dis(pre_left_point, target_side_point, all_contour_info)    # 缝边左点 到 端点 的双向距离
                            if not (min(cur_left_point_dis_2side) < min(pre_left_point_dis_2side) and min(cur_left_point_dis_2side) < min(cur_right_point_dis_2side)):
                                pass
                            else:
                                target_global_param = target_side_point["global_param"]
                                target_global_param_list = [target_global_param, target_global_param]
                                optimized = True

                # 如果两个点都离拟合边端点近
                if cur_left_dis2side < optimize_thresh_side_index_dis and pre_right_dis2side < optimize_thresh_side_index_dis:
                    target_global_param_list = [side_point_closet[0]["global_param"], side_point_closet[1]["global_param"]]
                    optimized = True

            # 应用结果
            if optimized:
                # # [todo] 可以考虑改成中间的点中不缝合点超过多少时，就不进行优化
                # === 将计算出的新的目标位置赋给两个缝合边 ===
                for point, target_global_param in zip([pre_right_point, cur_left_point], target_global_param_list):
                    # 找到离 目标global_param所在位置 最近的点
                    target_point_id, first_close_point, second_close_point = global_param2point_id(target_global_param, current_contour_info)
                    target_edge_info = all_edge_info[first_close_point["edge_id"]]  # 获取所在边的信息
                    target_param = target_global_param - target_edge_info["param_start"]  # 局部param

                    point["edge_id"] = target_edge_info["id"]
                    point["global_param"] = target_global_param
                    point["param"] = target_param
                    point["id"] = target_point_id

        # 如过并非上述情况，且缝边间距小于阈值，则将二者的中间点设为优化后的位置 --------------------------------------------------------------------------------------------------------------
        if not optimized:
            # [todo] 如果某个缝边端点离拟合边端点太近了（1e-3），则不进行优化
            if index_dis < optimize_thresh_neighbor_index_dis and not (pre_right_dis2side<1e-3 or cur_left_dis2side<1e-3):
                if min_dis_index == 1:  # 两个边中间有缝隙
                    if index_dis < optimize_thresh_neighbor_index_dis:
                        target_global_param = pre_right_point["global_param"] + param_dis / 2
                        optimized=True
                    else: pass
                else:  # 两个边中间有重合
                    if index_dis < optimize_thresh_neighbor_index_dis:  # 对于中间有重合的情况，我们需要更积极的处理
                        target_global_param = cur_left_point["global_param"] + param_dis / 2
                        optimized=True
                    else: pass
            # 应用结果
            if optimized:
                # 如果越界到别的 contour 上了，则修正
                if target_global_param > current_contour_info["param_end"]:
                    target_global_param = current_contour_info["param_start"] +  target_global_param - current_contour_info["param_end"]
                # 找到离 目标global_param所在位置 最近的点
                target_point_id, first_close_point, second_close_point = global_param2point_id(target_global_param, current_contour_info)
                target_edge_info =  all_edge_info[first_close_point["edge_id"]]         # 获取所在边的信息
                target_param = target_global_param - target_edge_info["param_start"]    # 局部param
                # # [todo] 可以考虑改成中间的点中不缝合点超过多少时，就不进行优化
                # 将计算出的新的目标位置赋给两个缝合边
                for point in [pre_right_point, cur_left_point]:
                    point["edge_id"] = target_edge_info["id"]
                    point["global_param"] = target_global_param
                    point["param"] = target_param
                    point["id"] = target_point_id



def optimize_stitch_edge_list_byApproxEdge(stitch_edge_list ,thresh, all_contour_info, all_edge_info):
    """
    将 离边端点特别近 且 不与其它缝边相衔接 的缝边的端点 的位置进行调整
    :param stitch_edge_list:
    :param thresh:
    :param all_contour_info:
    :param all_edge_info:
    :return:
    """
    stitch_edge_list_contourOrder = sorted(stitch_edge_list, key=lambda x: (x["start_point"]["contour_id"],))
    stitch_edge_list_paramOrder = []
    start_contour_id = stitch_edge_list_contourOrder[0]["start_point"]["contour_id"]
    for e_idx, stitch_edge in enumerate(stitch_edge_list_contourOrder):
        if stitch_edge["start_point"]["contour_id"] != start_contour_id or e_idx == len(stitch_edge_list_contourOrder) - 1:
            # 如果最后一个的contour_id没变化
            if not stitch_edge["start_point"]["contour_id"] != start_contour_id and e_idx == len(stitch_edge_list_contourOrder) - 1:
                stitch_edge_list_paramOrder.append(stitch_edge)

            # 【此时，stitch_edge_list_paramOrder中所有的缝边都位于同一contour上】
            # 对这个stitch_edge_list_paramOrder根据global_param进行排序（isCC=false根据起始点排序，isCC=true根据终点点排序）
            stitch_edge_list_paramOrder = sorted(stitch_edge_list_paramOrder,
                                                 key=lambda x: (x["start_point"]["global_param"] if not x["isCC"] else x["end_point"]["global_param"]))

            # 对同一contour上的缝边，尝试合并它们 ------------------------------------------------------------------------------
            for se_idx, start_stitch_edge in enumerate(stitch_edge_list_paramOrder):
                if se_idx == 0:
                    start_stitch_edge_previous = stitch_edge_list_paramOrder[-1]
                else:
                    start_stitch_edge_previous = stitch_edge_list_paramOrder[se_idx - 1]

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

                # 如果相邻缝边间距较大
                st_index_dis_d = cal_neigbor_points_index_dis(pre_right_point, cur_left_point, all_contour_info)
                # if st_index_dis_d and min(st_index_dis_d) > 1e-3:
                for point in [pre_right_point, cur_left_point]:
                    edge = all_edge_info[point["edge_id"]]
                    edge_side_point = [edge["points_info"][0], edge["points_info"][-1]]
                    # 缝边端点 到 所在边的两个端点 的距离
                    side_point_index_dis = [
                        min(cal_neigbor_points_index_dis(point, edge_side_point[0], all_contour_info)),
                        min(cal_neigbor_points_index_dis(point, edge_side_point[1], all_contour_info)),
                    ]
                    # 获取 前一条缝边的右点 和 当前缝边的左点 离各自边上两个端点的距离
                    side_point_index_dis_index = 0 if side_point_index_dis[0] < side_point_index_dis[1] else 1
                    if 0 < side_point_index_dis[side_point_index_dis_index] <= thresh:
                        closed_side_point = edge_side_point[side_point_index_dis_index]
                        # 如果当前缝边（或前一个缝边）上另一端点离这个拟合边端点更近，则这次优化不会进行
                        if (min(cal_neigbor_points_index_dis(closed_side_point, point, all_contour_info)) >
                                min(cal_neigbor_points_index_dis(closed_side_point, pre_left_point, all_contour_info))):
                            continue
                        if (min(cal_neigbor_points_index_dis(closed_side_point, point, all_contour_info)) >
                                min(cal_neigbor_points_index_dis(closed_side_point, cur_right_point, all_contour_info))):
                            continue
                        point["id"] = closed_side_point["id"]
                        point["param"] = closed_side_point["param"]
                        point["global_param"] = closed_side_point["global_param"]

            # 切换到下一个 contour
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

# 从点点缝合推导出线线缝合
def pointstitch_2_edgestitch2(batch, inf_rst, stitch_mat, stitch_indices,
                              unstitch_thresh = 6, fliter_len=3,
                              optimize_thresh_neighbor_index_dis = 4,
                              optimize_thresh_side_index_dis=2):

    """
    :param batch:                       # garment_jigsaw model input
    :param inf_rst:                     # garment_jigsaw model output
    :param stitch_mat:                  # mat of point&point stitch
    :param stitch_indices:              # indices of point&point stitch
    :param unstitch_thresh:             # continue unstitched point longer than this will create new stitch point list
    :param fliter_len:                  # stitch point list shorter than this will be filtered
    :param optimize_thresh_neighbor_index_dis:   # neighbor stitch edges param_dis lower than this will be optimized
    :return:
    """

    # 读取数据（V2的AIGP文件 和 边拟合结果）
    garment_json, panel_nes, contour_nes, edge_approx, panel_instance_seg = load_data(batch)
    device_ = stitch_mat.device

    pcs = batch["pcs"][0]
    n_pcs = batch["n_pcs"][0]
    piece_id = batch["piece_id"][0]
    contour_num = torch.sum(n_pcs!=0)
    n_pcs = n_pcs[:contour_num]

    n_pcs_cumsum = torch.cumsum(n_pcs, dim=-1)
    contour_nes_cumsum = torch.cumsum(contour_nes, dim=-1)

    # 将拟合边的index转换成全局坐标
    edge_approx_global = deepcopy(edge_approx)
    for contour_idx in range(len(n_pcs_cumsum)):
        if contour_idx == 0: edge_start_idx = 0
        else: edge_start_idx = contour_nes_cumsum[contour_idx - 1]
        edge_end_idx = contour_nes_cumsum[contour_idx]
        if contour_idx!=0: edge_approx_global[edge_start_idx:edge_end_idx] += n_pcs_cumsum[contour_idx-1]

    # 将每个点、边、contour的关键信息进行汇总 ----------------------------------------------------------------------------------
    all_contour_info = {}
    all_edge_info = {}
    all_point_info = {}
    very_small_gap = 1e-5
    global_param = 0  # 一个点的全局param
    contourid2panelid = {}  # 用于将contour_id映射到panel_id

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

        # 判断是否净边（isOL）
        isOL, contour_pos = is_contour_OutLine(contour_idx, panel_instance_seg)

        # === 构建每个Panel的信息 ===
        # 这个contour上的点
        contour_points = pcs[point_start_idx:point_end_idx]
        # 局部/全局 拟合边
        contour_edges_approx = edge_approx[edge_start_idx:edge_end_idx]
        contour_edges_approx_global = edge_approx_global[edge_start_idx:edge_end_idx]
        # 这个contour上有几个拟合边，几个点
        contour_edge_num = contour_edges_approx.shape[-2]
        contour_point_num = contour_points.shape[-2]
        panel_json = garment_json["panels"][panel_instance_idx]

        if isOL: contour_id = panel_json["id"]
        else: contour_id = get_random_uuid()

        all_contour_info[contour_id] = {"id":contour_id, "edges_info":{}}
        contour_info = all_contour_info[contour_id]
        contour_info["param_start"] = global_param  # 这个Panel的param的最小值
        contour_info["param_end"] = []
        contour_info["index_end"] = point_end_idx-1
        contour_info["index_start"] = point_start_idx if isinstance(point_start_idx,torch.Tensor) \
                                      else torch.tensor(point_start_idx, dtype=point_end_idx.dtype,device=point_end_idx.device)

        seqEdges_json = panel_json["seqEdges"][contour_pos]
        contourid2panelid[contour_id] = panel_json["id"]
        for edge_idx, (e_approx, e_approx_global) in enumerate(zip(contour_edges_approx, contour_edges_approx_global)):
            # === 构建每个Edge的信息 ===
            edge_json = seqEdges_json["edges"][edge_idx]
            all_edge_info[edge_json["id"]] = {"id":edge_json["id"], "contour_id":contour_info["id"], "points_info":[]}
            edge_info = all_edge_info[edge_json["id"]]
            contour_info["edges_info"][edge_json["id"]] = edge_info
            # 获取这个Panel上每个edge上的所有点的按顺序的index
            if ((edge_idx==0 and e_approx_global[0]>e_approx_global[1]) or (edge_idx==contour_edge_num-1 and e_approx_global[0]>e_approx_global[1])):
                edge_points_idx = torch.concat([torch.arange(e_approx_global[0], point_end_idx), torch.arange(point_start_idx, e_approx_global[1]+1)])
                edge_points_idx.to(device_)
            else:
                edge_points_idx = torch.arange(e_approx_global[0], e_approx_global[1]+1)
            # 这个边的param的最小值
            edge_info["param_start"] = global_param
            for idx, point_idx in enumerate(edge_points_idx):
                # === 构建每个Point的信息 ===
                # 计算当前点的param (我添加了一个很小的offset来防止一些报错，并且这些offset会在后续处理中被修复)
                param = idx/(len(edge_points_idx)-1)  # 这个点在边上的位置
                if idx==0: param+=very_small_gap
                elif idx==len(edge_points_idx)-1: param-=very_small_gap

                point_info = {"id": point_idx, "contour_id": contour_info["id"], "edge_id": edge_info["id"],
                              "param": param, "global_param": global_param + param, "is_stitch": False}
                edge_info["points_info"].append(point_info)

                pt_key = point_idx.tolist()
                if not pt_key in all_point_info.keys():
                    all_point_info[pt_key] = [point_info]
                else:  # 重复的点（两个边的交汇处）
                    all_point_info[pt_key].append(point_info)

            # 一条边上的点全部计算完后，global_param+1后作为下一条边的起始 param_start
            global_param+=1
        # contour_info["all_points_info"]在优化缝合时会大量重复调用，因此我在这里提前获取
        contour_info["all_points_info"] = [point for e in contour_info["edges_info"] for point in contour_info["edges_info"][e]["points_info"]]
        contour_info["all_points_global_param"] = torch.tensor([point["global_param"] for point in contour_info["all_points_info"]])
        # 这个Panel的param的最大值
        contour_info["param_end"] = global_param

    # 将信息转换为N个由缝合点对组成的list ------------------------------------------------------------------------------------
    # 缝合关系的映射
    stitch_mat = torch.zeros((len(pcs)), dtype=torch.int64, device=device_)-1
    stitch_mat[stitch_indices[:, 0]] = stitch_indices[:, 1]  # 只取前一半的部分，因为软件中的缝合不能重叠
    # stitch_mat[stitch_indices[:, 1]] = stitch_indices[:, 0]

    all_stitch_points_list = []
    stitch_points_list = []
    unstitch_num = 0

    for contour_id in all_contour_info:
        contour_info = all_contour_info[contour_id]
        edges_info = contour_info["edges_info"]

        # 遍历这个Panel上的所有缝合点对
        for edge_id in edges_info:
            edge_info = edges_info[edge_id]
            points_info = edge_info["points_info"]

            for idx, point_info in enumerate(points_info):
                # 当前点 和它缝合的点
                point_idx = point_info["id"]
                point_idx_cor = stitch_mat[point_idx]

                # 将它们标记为缝合点
                lst = []
                if point_idx>0: lst.append(all_point_info[point_idx.item()])
                if point_idx_cor>0: lst.append(all_point_info[point_idx_cor.item()])
                for p_l in lst:
                    for p in p_l:
                        p["is_stitch"] = True

                # 防止不缝合的点导致的报错
                if point_idx_cor.tolist() != -1:
                    point_info_cor = all_point_info[point_idx_cor.tolist()]
                else:
                    point_info_cor = None

                # 对几种特殊情况进行分别处理 ---------------------------------------------------------------------------------
                for _ in range(1):
                    # === 如果当前点不缝合 === ，且已经连续数个点都不缝合时，开一条新的缝合
                    if not point_info_cor:
                        unstitch_num+=1  # [modified]
                        if unstitch_num>unstitch_thresh:
                            unstitch_num=0
                            if len(stitch_points_list) > 0:
                                all_stitch_points_list.append(stitch_points_list)
                                stitch_points_list = []
                        break
                    # === 如果当前点缝合 ===
                    else: unstitch_num = 0

                    # 根据“缝合点”的信息 判断是否开新的缝合 ------------------------------------------------------------------
                    if len(stitch_points_list) > 0:
                        # 缝合点所在Panel发生变化
                        if point_info["contour_id"] != stitch_points_list[-1][0]["contour_id"]:
                            all_stitch_points_list.append(stitch_points_list)
                            stitch_points_list=[]
                            stitch_points_list.append([point_info, point_info_cor])
                            break

                    # 根据“被缝合点”的信息 判断是否开新的缝合 ------------------------------------------------------------------
                    if len(stitch_points_list) > 0:
                        # === 判断contour_id是否发生变化 ===（trigger假时表示发生了变化）
                        trigger = False
                        for p in point_info_cor:
                            for i_tmp, p_id_tmp in enumerate([e["contour_id"] for e in stitch_points_list[-1][1]]):
                                if p["contour_id"] == p_id_tmp:
                                    trigger = True


                        # === 如果被缝合点的 contour_id 发生了变化 ===
                        if not trigger:
                            all_stitch_points_list.append(stitch_points_list)
                            stitch_points_list = []
                            stitch_points_list.append([point_info, point_info_cor])
                            break

                        # # === 如果被缝合点的 contour_id 没有发生变化 ===
                        # # 当前和上一个被缝合点的间距如果过大，则开新的缝合边 [todo] 改成根据 index_dis 判断
                        # thresh_side_dis = 8
                        # # thresh_side_dis = 0.6
                        # if trigger:
                        #     # 计算两个被缝合点的双向param_dis
                        #     index_dis_d = cal_neigbor_points_index_dis(stitch_points_list[-1][1][0], point_info_cor[0], all_contour_info)
                        #     index_dis_mask = np.array(index_dis_d)<thresh_side_dis
                        #     # param_dis_d = cal_neigbor_points_param_dis(stitch_points_list[-1][1][0], point_info_cor[0], all_contour_info)
                        #     # param_dis_mask = np.array(param_dis_d)<thresh_side_dis
                        #     # 如果两边的距离都超出了阈值
                        #     # if np.sum(param_dis_mask)==0:
                        #     if np.sum(index_dis_mask)==0:
                        #         all_stitch_points_list.append(stitch_points_list)
                        #         stitch_points_list = []
                        #         stitch_points_list.append([point_info, point_info_cor])
                        #         break

                        # 被缝合点是歧义点（这代表这个点是拟合边的端点），取更近的那个
                        if trigger and len(point_info_cor)>1:
                            param_dis_d = [
                                min(cal_neigbor_points_param_dis(stitch_points_list[-1][1][0],
                                                             point_info_cor[0], all_contour_info)),
                                min(cal_neigbor_points_param_dis(stitch_points_list[-1][1][0],
                                                             point_info_cor[1], all_contour_info)),
                            ]
                            param_dis_d = np.array(param_dis_d)
                            near = int(np.argmin(param_dis_d))
                            far = int(np.argmax(param_dis_d))

                            # [todo] 第一个append近的，第二个append远的
                            stitch_points_list.append([point_info, [point_info_cor[near]]])
                            # all_stitch_points_list.append(stitch_points_list)
                            # stitch_points_list = []
                            # stitch_points_list.append([point_info, [point_info_cor[far]]])
                            break

                    # 其它情况
                    stitch_points_list.append([point_info, point_info_cor])
                    break

    # 如果最后的stitch_points_list不为空，append到all
    if len(stitch_points_list) > 0:
        all_stitch_points_list.append(stitch_points_list)

    # 将太短的全部过滤掉 ---------------------------------------------------------------------------------------------------
    all_stitch_points_list, _ = filter_too_short(all_stitch_points_list, fliter_len = fliter_len) # [modified]

    # 将被缝合点由list换成dict ---------------------------------------------------------------------------------------------
    for s_idx, stitch_points_list in enumerate(all_stitch_points_list):
        for sp_idx, st_point in enumerate(stitch_points_list):
            st_point[1] = st_point[1][0]

    # 为每个缝合计算时针方向 ------------------------------------------------------------------------------------------------
    isCC_order_list = []
    for s_idx, stitch_points_list in enumerate(all_stitch_points_list):
        isCC_order_list.append([None,None])
        # 缝边和被缝边分别计算时针方向
        for i in range(2):
            order_sum = 0
            max_gap = min(3, len(stitch_points_list))
            for gap in range(1, max_gap+1):
                for sp_idx, st_point in enumerate(stitch_points_list):
                    if sp_idx < gap: continue
                    # 计算双向的param_dis
                    dis_d = cal_neigbor_points_param_dis(st_point[i], stitch_points_list[sp_idx - gap][i], all_contour_info)
                    ord = gap if dis_d[1] >= dis_d[0] else -gap
                    order_sum += ord
            # order_sum>=0时，我们认为这个缝边是顺时针的，此时
            isCC_order_list[-1][i] = order_sum < 0


    # 为每组连续且属于同一个edge(contour?)的点按照param进行排序 --------------------------------------------------------------------------
    cmpfun = lambda x: (x["global_param"])
    judge_key = "contour_id"  # [todo] 如果有问题，改回 "edge_id" 试试
    for s_idx, stitch_points_list in enumerate(all_stitch_points_list):
        unordered_stitch_points_list = []
        start_contour_id = stitch_points_list[0][1][judge_key]
        start_point_idx = 0
        for point_idx, stitch_point in enumerate(stitch_points_list):
            # 如果edge_id发生了变化，或是这个Panel结束了
            if stitch_point[1][judge_key] != start_contour_id or point_idx == len(stitch_points_list)-1:
                if stitch_point[1][judge_key] == start_contour_id and point_idx == len(stitch_points_list)-1:
                    unordered_stitch_points_list.append(stitch_point[1])

                if isCC_order_list[s_idx][1]: reverse = True
                else: reverse = False

                ordered_stitch_points_list = deepcopy(sorted(unordered_stitch_points_list, key=cmpfun, reverse=reverse))
                for i in range(len(unordered_stitch_points_list)):
                    stitch_points_list[start_point_idx + i][1] = ordered_stitch_points_list[i]

                start_point_idx = point_idx
                start_contour_id = stitch_point[1][judge_key]
                unordered_stitch_points_list = [stitch_point[1]]
            else:
                unordered_stitch_points_list.append(stitch_point[1])

    # 一个边上的点间距过长，则拆分边 -------------------------------------------------------------------------------------------------
    thresh = 4
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
                    pass
        if len(stitch_point_stack) != 0:
            new_all_stitch_points_list.append(stitch_point_stack)
            new_isCC_order_list.append(isCC_order_list[idx])
    all_stitch_points_list = new_all_stitch_points_list
    isCC_order_list = new_isCC_order_list

    # 将太短的全部过滤掉 ---------------------------------------------------------------------------------------------------
    all_stitch_points_list, isCC_order_list = filter_too_short(all_stitch_points_list, isCC_order_list, fliter_len = fliter_len) # [modified]

    # 获取缝合信息 --------------------------------------------------------------------------------------------------------
    # 用于存放所有缝合边信息(用于后续优化)
    stitch_edge_list = []
    for s_idx, stitch_points_list in enumerate(all_stitch_points_list):
        start_stitch_points = stitch_points_list[0]
        end_stitch_points = stitch_points_list[-1]
        stitch_edge_start = get_new_stitch_edge(start_stitch_points[0], end_stitch_points[0], isCC=isCC_order_list[s_idx][0])
        stitch_edge_end = get_new_stitch_edge(start_stitch_points[1], end_stitch_points[1], isCC=isCC_order_list[s_idx][1])
        stitch_edge_start["target_edge"] = stitch_edge_end
        stitch_edge_end["target_edge"] = stitch_edge_start
        stitch_edge_list.append(stitch_edge_start)
        stitch_edge_list.append(stitch_edge_end)

    # # 将 离边端点特别近 且 不与其它缝边相衔接 的缝边的端点 的位置进行调整 ---------------------------------------------------------
    # optimize_stitch_edge_list_byApproxEdge(stitch_edge_list, optimize_thresh_side_index_dis, all_contour_info, all_edge_info)

    # 优化相邻缝合的param -------------------------------------------------------------------------------------------------
    # 按所在板片进行排序后的 stitch_edge_list
    stitch_edge_list_contourOrder = sorted(stitch_edge_list, key=lambda x: (x["start_point"]["contour_id"],))
    stitch_edge_list_paramOrder = []
    start_contour_id = stitch_edge_list_contourOrder[0]["start_point"]["contour_id"]
    for e_idx, stitch_edge in enumerate(stitch_edge_list_contourOrder):
        # 如果contour_id发生了变化，或是这个edge是最后一个
        if stitch_edge["start_point"]["contour_id"] != start_contour_id or e_idx == len(stitch_edge_list_contourOrder) - 1:
            # 如果最后一个的contour_id没变化
            if not stitch_edge["start_point"]["contour_id"] != start_contour_id and e_idx == len(stitch_edge_list_contourOrder) - 1:
                stitch_edge_list_paramOrder.append(stitch_edge)
            # 对这个stitch_edge_list_paramOrder根据global_param进行排序（isCC=false根据起始点排序，isCC=true根据终点点排序）
            stitch_edge_list_paramOrder = sorted(stitch_edge_list_paramOrder, key=lambda x:(x["start_point"]["global_param"] if not x["isCC"] else x["end_point"]["global_param"]))
            # 对同一contour上的缝边，优化它们的param
            optimize_stitch_edge_list_byNeighbor(stitch_edge_list_paramOrder, all_contour_info, all_edge_info,
                                                 optimize_thresh_neighbor_index_dis=optimize_thresh_neighbor_index_dis,
                                                 optimize_thresh_side_index_dis=optimize_thresh_side_index_dis)
            # 切换到下一个 contour
            start_contour_id = stitch_edge["start_point"]["contour_id"]
            stitch_edge_list_paramOrder = [stitch_edge]
        else:
            stitch_edge_list_paramOrder.append(stitch_edge)

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
    stitch_edge_list_merged = []  # 合并后的缝边
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

    # 将 离边端点特别近 且 不与其它缝边相衔接 的缝边的端点 的位置进行调整 ---------------------------------------------------------
    optimize_stitch_edge_list_byApproxEdge(stitch_edge_list, optimize_thresh_side_index_dis, all_contour_info, all_edge_info)


    # 将缝合数据转换为 AIGP 文件中 "stitches" 的格式 ------------------------------------------------------------------------
    # 将stitch_edge_list中，非outLine的contourid改为对应的panel的id
    for stitch_edge in stitch_edge_list:
        for k in ['start_point', 'end_point']:
            stitch_edge[k]["panel_id"] = contourid2panelid[stitch_edge[k]["contour_id"]]
    # 转格式
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
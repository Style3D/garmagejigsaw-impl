# 从点点缝合关系获取边边缝合关系
import json
import math
import torch
import numpy as np
from copy import deepcopy



def apply_point_info(stitch_edge_side, point_info):
    stitch_edge_side['clothPieceId'] = point_info['panel_id']
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
def filter_too_short(all_stitch_points_list, fliter_len = 1):
    all_stitch_points_list_ = []
    for s in all_stitch_points_list:
        if len(s)>fliter_len:
            all_stitch_points_list_.append(s)
    return all_stitch_points_list_

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
def cal_stitch_edge_param_dis(stitch_edge, all_panel_info):
    panel_info = all_panel_info[stitch_edge['start_point']['panel_id']]
    start_key = "start_point" if not stitch_edge["isCC"] else "end_point"
    end_key = "end_point" if not stitch_edge["isCC"] else "start_point"
    start_point = stitch_edge[start_key]
    end_point = stitch_edge[end_key]
    # 计算两个点的param的差距
    if end_point['global_param'] >= start_point['global_param']:
        param_dis = end_point['global_param'] - start_point['global_param']
    else:
        param_dis = (panel_info['param_end'] - start_point['global_param'] +
                      end_point['global_param'] - panel_info['param_start'])
    return param_dis


# 计算一个环上的两个点之间的双向param_dis(仅用于计算较近的点之间的距离)
def cal_neigbor_points_param_dis(start_point, end_point, all_panel_info):
    panel_info = all_panel_info[start_point['panel_id']]
    if start_point['panel_id']!=end_point['panel_id']:
        return None
    if end_point['global_param'] >= start_point['global_param']:
        param_dis=[
          panel_info['param_end'] - start_point['global_param'] + end_point['global_param'] - panel_info['param_start'],
          end_point['global_param'] - start_point['global_param']
        ]
    else:
        param_dis = [
          start_point['global_param'] - end_point['global_param'],
          panel_info['param_end'] - start_point['global_param'] + end_point['global_param'] - panel_info['param_start']
        ]
    return param_dis

# # [todo] 如果无用，删掉
# # 根据输入的global_param，找到所在的panel
# def global_param2panel_id(global_param, all_panel_info):
#     bins = torch.tensor([p["param_start"] for p in [all_panel_info[k] for k in all_panel_info]])
#     index = torch.searchsorted(bins, torch.tensor([global_param]), right=True) - 1
#     target_panel_id = list(all_panel_info.keys())[index]
#     return target_panel_id

# 根据输入的global_param + panel_info，找到这个global_param对应的（浮点数的）point_id，以及最近的俩 point
def global_param2point_id(global_param, panel_info):
    dis_tensor=abs(panel_info["all_points_global_param"] - global_param)
    values, indices = torch.topk(dis_tensor, 2, largest=False)
    first_near, second_near = panel_info["all_points_info"][indices[0]], panel_info["all_points_info"][indices[1]]

    # # [todo] 如果first_near的global_param在panel的某一端上，将second_near设置为另一端
    # if first_near==panel_info['all_points_global_param'][0] and global_param <= first_near["global_param"]:
    #     second_near = panel_info['all_points_global_param'][-1]
    #     values[1] =
    #
    # elif first_near==panel_info['all_points_global_param'][-1] and global_param >= first_near["global_param"]:
    #     second_near = panel_info['all_points_global_param'][0]

    if values[0] < 1e-5:
        point_idx_float = first_near["id"]
    else:
        point_idx_float = (first_near["id"] * (values[1] ) + second_near["id"] * values[0]) / (values[0] + values[1] )

    return  point_idx_float, first_near, second_near


# 计算一个环上的两个点之间的双向index_dis(仅用于计算较近的点之间的距离)
def cal_neigbor_points_index_dis(start_point, end_point, all_panel_info):
    panel_info = all_panel_info[start_point['panel_id']]
    if start_point['panel_id']!=end_point['panel_id']:
        return None
    if end_point['id'] >= start_point['id']:
        index_dis=[
          panel_info['index_end'] - start_point['id'] + end_point['id'] - panel_info['index_start'],
          end_point['id'] - start_point['id']
        ]
    else:
        index_dis = [
          start_point['id'] - end_point['id'],
          panel_info['index_end'] - start_point['id'] + end_point['id'] - panel_info['index_start']
        ]
    index_dis = [i.item() for i in index_dis]
    return index_dis


def optimize_stitch_edge_list(stitch_edge_list_paramOrder, index_dis_optimize_thresh, all_panel_info, all_edge_info):
    current_panel_info = all_panel_info[stitch_edge_list_paramOrder[0]['start_point']['panel_id']]
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
        index_dis_2side = cal_neigbor_points_index_dis(pre_right_point, cur_left_point, all_panel_info)
        param_dis_2side = cal_neigbor_points_param_dis(pre_right_point, cur_left_point, all_panel_info)
        min_dis_index = 0 if index_dis_2side[0] < index_dis_2side[1] else 1     # 距离较小的那一端的 index
        index_dis = index_dis_2side[min_dis_index]
        param_dis = param_dis_2side[min_dis_index]

        # 设定阈值，根据采样频率delta来动态调整阈值的大小 ---------------------------------------------------------------------------------
        index_dis_optimize_thresh = 9   # 优化缝边之间距离的阈值（间距小于这一阈值的一对缝边，它们之间的端点会被优化）
        index_dis_side_thresh = 3       # 缝边与边端点之间的阈值（小于阈值的缝边的端点将在优化时吸附到边的端点上）
        adjust_param = 0.023 / 0.023  # 这个用于根据采样频率调整阈值的大小 [todo] 这是个预留接口 看到那个 “0.023/0.023” 了吗？后续需要在这里将采样频率考虑进来
        index_dis_optimize_thresh = index_dis_optimize_thresh * adjust_param
        index_dis_side_thresh = index_dis_side_thresh * adjust_param

        # 对于端点距离 index_dis 小于阈值的缝边进行优化 ---------------------------------------------------------------------------------
        # 对于中间存在空隙的相邻缝边，用单倍阈值；对于中间存在重合的相邻缝边，要更积极的处理，采样大于2倍的阈值。
        if index_dis < index_dis_optimize_thresh * 5:
            # === 检测两个缝边离端点的距离 ===
            # 同一 edge 上的所有点
            edge_points_pre_right = all_edge_info[pre_right_point["edge_id"]]["points_info"]
            edge_points_cur_left = all_edge_info[cur_left_point["edge_id"]]["points_info"]
            # 找到 edge 上最近的端点
            side_point_index_dis = [
                [
                    # 前一条缝边的右点 到其所在边的两个端点的距离
                    min(cal_neigbor_points_index_dis(pre_right_point, edge_points_pre_right[0], all_panel_info)),
                    min(cal_neigbor_points_index_dis(pre_right_point, edge_points_pre_right[-1], all_panel_info))
                ],
                [
                    # 当前缝边的左点 到其所在边的两个端点的距离
                    min(cal_neigbor_points_index_dis(cur_left_point, edge_points_cur_left[0], all_panel_info)),
                    min(cal_neigbor_points_index_dis(cur_left_point, edge_points_cur_left[-1], all_panel_info))
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
            pre_right_dis2side = min(cal_neigbor_points_index_dis(pre_right_point, side_point_closet[0], all_panel_info))
            cur_left_dis2side = min(cal_neigbor_points_index_dis(cur_left_point, side_point_closet[1], all_panel_info))

            # === 计算两个点需要调整到的位置的 global_param ===
            optimized = False  # 是否已经进行过优化了
            # 如果两个相邻缝边上的某个点离边的端点的距离小于阈值，则将这个边的端点设定为目标位置
            if pre_right_dis2side < index_dis_side_thresh or cur_left_dis2side < index_dis_side_thresh:

                order = ["pre", "cur"] if pre_right_dis2side < cur_left_dis2side else  ["cur", "pre"]
                for o in order:
                    if optimized: break
                    if o=="pre" and pre_right_dis2side < index_dis_side_thresh:
                        target_side_point, second_target = side_point_closet[0], side_point_closet[1]
                        left_point, right_point = pre_left_point, pre_right_point
                        left_point_dis_2side = cal_neigbor_points_index_dis(left_point, target_side_point, all_panel_info)    # 缝边左点 到 端点 的双向距离
                        right_point_dis_2side = cal_neigbor_points_index_dis(right_point, target_side_point, all_panel_info)  # 缝边右点 到 端点 的双向距离
                        if left_point_dis_2side[0]<left_point_dis_2side[1] and right_point_dis_2side[0]<right_point_dis_2side[1]:
                            pass
                        else:
                            target_global_param = target_side_point["global_param"]
                            optimized = True
                    elif o=="cur" and cur_left_dis2side < index_dis_side_thresh:
                        target_side_point, second_target = side_point_closet[1], side_point_closet[0]
                        left_point, right_point = cur_left_point, cur_right_point
                        left_point_dis_2side = cal_neigbor_points_index_dis(left_point, target_side_point, all_panel_info)  # 缝边左点 到 端点 的双向距离
                        right_point_dis_2side = cal_neigbor_points_index_dis(right_point, target_side_point, all_panel_info)  # 缝边右点 到 端点 的双向距离
                        if left_point_dis_2side[0] > left_point_dis_2side[1] and right_point_dis_2side[0] > right_point_dis_2side[1]:
                            pass
                        else:
                            target_global_param = target_side_point["global_param"]
                            optimized = True


            # 其它情况（到目前为止还没进行任何优化）：在这两个缝边的邻接处的两个点的中间作为目标位置 [todo] 这里的判断条件可能有问题
            if not optimized:
                if min_dis_index == 1:  # 两个边中间有缝隙
                    if index_dis < index_dis_optimize_thresh:
                        target_global_param = pre_right_point["global_param"] + param_dis / 2
                    else:
                        return
                else:  # 两个边中间有重合
                    if index_dis < index_dis_optimize_thresh * 5:  # 对于中间有重合的情况，我们需要更积极的处理
                        target_global_param = cur_left_point["global_param"] + param_dis / 2
                    else:
                        return

            # === 如果越界到别的 panel 上了，则修正 ===
            if target_global_param > current_panel_info["param_end"]:
                target_global_param = current_panel_info["param_start"] +  target_global_param - current_panel_info["param_end"]

            # === 找到离 目标global_param所在位置 最近的点 ===
            target_point_id, first_close_point, second_close_point = global_param2point_id(target_global_param, current_panel_info)
            target_edge_info =  all_edge_info[first_close_point["edge_id"]]         # 获取所在边的信息
            target_param = target_global_param - target_edge_info["param_start"]    # 局部param

            # [todo] 可以考虑改成中间的点中不缝合点超过多少时，就不进行优化
            # === 如果目标点位置是不缝合点，则不进行合并 ===
            if min_dis_index == 1: check_param = pre_right_point["global_param"] + param_dis / 2
            else: check_param = cur_left_point["global_param"] + param_dis / 2
            if check_param > current_panel_info["param_end"]:
                check_param = current_panel_info["param_start"] + check_param - current_panel_info["param_end"]
            _, check_point, _ = global_param2point_id(check_param, current_panel_info)
            if not check_point["is_stitch"]:
                return

            # === 将计算出的新的目标位置赋给两个缝合边 ===
            for point in [pre_right_point, cur_left_point]:
                point["edge_id"] = target_edge_info["id"]
                point["global_param"] = target_global_param
                point["param"] = target_param
                point["id"] = target_point_id



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
    edge_approx = torch.tensor(annotations_json["edge_approx"], dtype=torch.int64, device=batch["pcs"].device)

    return garment_json, panel_nes, edge_approx

# 从点点缝合推导出线线缝合
def pointstitch_2_edgestitch(batch, inf_rst, stitch_mat, stitch_indices,
                             unstitch_thresh = 6, fliter_len=3,
                             param_dis_optimize_thresh = 0.3):

    """
    :param batch:                       # garment_jigsaw model input
    :param inf_rst:                     # garment_jigsaw model output
    :param stitch_mat:                  # mat of point&point stitch
    :param stitch_indices:              # indices of point&point stitch
    :param unstitch_thresh:             # continue unstitched point longer than this will create new stitch point list
    :param fliter_len:                  # stitch point list shorter than this will be filtered
    :param param_dis_optimize_thresh:   # neighbor stitch edges param_dis lower than this will be optimized
    :return:
    """

    # 读取数据（V2的AIGP文件 和 边拟合结果）
    garment_json, panel_nes, edge_approx = load_data(batch)
    device_ = stitch_mat.device

    pcs = batch["pcs"][0]
    n_pcs = batch["n_pcs"][0]
    pcs_idx = torch.arange(pcs.shape[-2])
    piece_id = batch["piece_id"][0]
    panel_num = torch.sum(n_pcs!=0)
    n_pcs = n_pcs[:panel_num]

    n_pcs_cumsum = torch.cumsum(n_pcs, dim=-1)
    panel_nes_cumsum = torch.cumsum(panel_nes, dim=-1)

    # 将拟合边的index转换成全局坐标
    edge_approx_global = deepcopy(edge_approx)
    for panel_idx in range(len(n_pcs_cumsum)):
        if panel_idx == 0: edge_start_idx = 0
        else: edge_start_idx = panel_nes_cumsum[panel_idx - 1]
        edge_end_idx = panel_nes_cumsum[panel_idx]
        if panel_idx!=0: edge_approx_global[edge_start_idx:edge_end_idx] += n_pcs_cumsum[panel_idx-1]

    # 将每个点、边、panel的关键信息进行汇总 ----------------------------------------------------------------------------------
    all_panel_info = {}
    all_edge_info = {}
    all_point_info = {}
    very_small_gap = 1e-5
    global_param = 0  # 一个点的全局param
    for panel_idx in range(len(n_pcs_cumsum)):
        if panel_idx == 0:
            point_start_idx = 0
            edge_start_idx = 0
        else:
            point_start_idx = n_pcs_cumsum[panel_idx - 1]
            edge_start_idx = panel_nes_cumsum[panel_idx - 1]
        point_end_idx = n_pcs_cumsum[panel_idx]
        edge_end_idx = panel_nes_cumsum[panel_idx]

        # ===== 构建每个Panel的信息 =====
        # 这个panel上的点
        panel_points = pcs[point_start_idx:point_end_idx]
        # 局部/全局 拟合边
        panel_edges_approx = edge_approx[edge_start_idx:edge_end_idx]
        panel_edges_approx_global = edge_approx_global[edge_start_idx:edge_end_idx]
        # 这个panel上有几个拟合边，几个点
        panel_edge_num = panel_edges_approx.shape[-2]
        panel_point_num = panel_points.shape[-2]
        panel_json =  garment_json["panels"][panel_idx]
        all_panel_info[panel_json["id"]] = {"id":panel_json["id"], "edges_info":{}}
        panel_info = all_panel_info[panel_json["id"]]
        panel_info["param_start"] = global_param  # 这个Panel的param的最小值
        panel_info["param_end"] = []
        panel_info["index_end"] = point_end_idx-1
        panel_info["index_start"] = point_start_idx if isinstance(point_start_idx,torch.Tensor) \
                                    else torch.tensor(point_start_idx,
                                                        dtype=point_end_idx.dtype,device=point_end_idx.device)
        for edge_idx, (e_approx, e_approx_global) in enumerate(zip(panel_edges_approx, panel_edges_approx_global)):
            # ===== 构建每个Edge的信息 =====
            edge_json = panel_json["seqEdges"][0]["edges"][edge_idx]
            all_edge_info[edge_json["id"]] = {"id":edge_json["id"], "panel_id":panel_info["id"], "points_info":[]}
            edge_info = all_edge_info[edge_json["id"]]
            panel_info["edges_info"][edge_json["id"]] = edge_info

            # 获取这个Panel上每个edge上的所有点的按顺序的index
            if ((edge_idx==0 and e_approx_global[0]>e_approx_global[1]) or (edge_idx==panel_edge_num-1 and e_approx_global[0]>e_approx_global[1])):
                edge_points_idx = torch.concat([torch.arange(e_approx_global[0], point_end_idx), torch.arange(point_start_idx, e_approx_global[1]+1)])
                edge_points_idx.to(device_)
            else:
                edge_points_idx = torch.arange(e_approx_global[0], e_approx_global[1]+1)
            # 这个边的param的最小值
            edge_info["param_start"] = global_param
            edge_info["index_start"] = min(edge_points_idx)
            edge_info["index_end"] = max(edge_points_idx)
            for idx, point_idx in enumerate(edge_points_idx):
                # ===== 构建每个Point的信息 =====
                # 计算当前点的param (我添加了一个很小的offset来防止一些报错，并且这些offset会在后续处理中被修复)
                param = idx/(len(edge_points_idx)-1)  # 这个点在边上的位置
                if idx==0: param+=very_small_gap
                elif idx==len(edge_points_idx)-1: param-=very_small_gap

                point_info = {"id": point_idx, "panel_id": panel_info["id"], "edge_id": edge_info["id"],
                              "param": param, "global_param": global_param + param, "is_stitch": False}
                edge_info["points_info"].append(point_info)
                pt_key = point_idx.tolist()
                if not pt_key in all_point_info.keys():
                    all_point_info[pt_key] = [point_info]
                else:  # 重复的点（两个边的交汇处）
                    all_point_info[pt_key].append(point_info)

            # 一条边上的点全部计算完后，global_param+1后作为下一条边的起始 param_start
            global_param+=1
        # panel_info["all_points_info"]在优化缝合时会大量重复调用，因此我在这里提前获取
        panel_info["all_points_info"] = [point for e in panel_info["edges_info"] for point in panel_info["edges_info"][e]["points_info"]]
        panel_info["all_points_global_param"] = torch.tensor([point["global_param"] for point in panel_info["all_points_info"]])
        # 这个Panel的param的最大值
        panel_info["param_end"] = global_param


    # 将信息转换为N个由缝合点对组成的list ------------------------------------------------------------------------------------
    # 缝合关系的映射
    stitch_mat = torch.zeros((len(pcs)), dtype=torch.int64, device=device_)-1
    stitch_mat[stitch_indices[:, 0]] = stitch_indices[:, 1]  # 只去前一半的部分，因为软件中的缝合不能重复
    # stitch_mat[stitch_indices[:, 1]] = stitch_indices[:, 0]

    all_stitch_points_list = []
    stitch_points_list = []
    unstitch_num = 0

    for panel_id in all_panel_info:
        panel_info = all_panel_info[panel_id]
        edges_info = panel_info["edges_info"]

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
                        if point_info["panel_id"] != stitch_points_list[-1][0]["panel_id"]:
                            all_stitch_points_list.append(stitch_points_list)
                            stitch_points_list=[]
                            stitch_points_list.append([point_info, point_info_cor])
                            break

                    # 根据“被缝合点”的信息 判断是否开新的缝合 ------------------------------------------------------------------
                    if len(stitch_points_list) > 0:
                        # === 判断panel_id是否发生变化 ===（trigger假时表示发生了变化）
                        trigger = False
                        for p in point_info_cor:
                            for i_tmp, p_id_tmp in enumerate([e["panel_id"] for e in stitch_points_list[-1][1]]):
                                if p["panel_id"] == p_id_tmp:
                                    trigger = True


                        # === 如果被缝合点的 panel_id 发生了变化 ===
                        if not trigger:
                            all_stitch_points_list.append(stitch_points_list)
                            stitch_points_list = []
                            stitch_points_list.append([point_info, point_info_cor])
                            break

                        # === 如果被缝合点的 panel_id 没有发生变化 ===
                        # 当前和上一个被缝合点的间距如果过大，则开新的缝合边 [todo] 改成根据 index_dis 判断
                        thresh_side_dis = 0.6
                        if trigger:
                            # 计算两个被缝合点的双向param_dis
                            param_dis_d = cal_neigbor_points_param_dis(stitch_points_list[-1][1][0], point_info_cor[0], all_panel_info)
                            param_dis_mask = np.array(param_dis_d)<thresh_side_dis
                            # 如果两边的距离都超出了阈值
                            if np.sum(param_dis_mask)==0:
                                all_stitch_points_list.append(stitch_points_list)
                                stitch_points_list = []
                                stitch_points_list.append([point_info, point_info_cor])
                                break

                        # 被缝合点是歧义点（这代表这个点是拟合边的端点），取更近的那个
                        if trigger and len(point_info_cor)>1:
                            param_dis_d = [
                                min(cal_neigbor_points_param_dis(stitch_points_list[-1][1][0],
                                                             point_info_cor[0], all_panel_info)),
                                min(cal_neigbor_points_param_dis(stitch_points_list[-1][1][0],
                                                             point_info_cor[1], all_panel_info)),
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
    all_stitch_points_list = filter_too_short(all_stitch_points_list, fliter_len = fliter_len) # [modified]

    # # （暂时没用了）消除歧义点 --------------------------------------------------------------------------------------------
    # for s_idx, stitch_points_list in enumerate(all_stitch_points_list):
    #     is_previous_matched = False
    #     for sp_idx, st_point in enumerate(stitch_points_list):
    #         is_current_matched = False
    #         # 非歧义点不处理
    #         if len(st_point[1]) == 1: continue
    #         # 对于歧义点，找到用于对比的点的idx：compare_idx
    #         # if sp_idx == 0:
    #         if sp_idx==0:
    #             compare_idx = sp_idx+1
    #         elif is_previous_matched:
    #             if sp_idx==len(stitch_points_list)-1:
    #                 compare_idx = 0
    #             else:
    #                 compare_idx = sp_idx+1
    #         else:
    #             compare_idx = sp_idx-1
    #
    #         # 对目标点进行匹配
    #         is_valid, p_idx = is_valid_stitch_point(st_point, stitch_points_list[compare_idx])
    #
    #         # 如果左点对比失败，且不是最后点
    #         if not is_valid and compare_idx==sp_idx-1 and sp_idx!=len(stitch_points_list)-1:
    #             # 对右点进行匹配
    #             compare_idx = sp_idx - 1
    #             is_valid, p_idx = is_valid_stitch_point(st_point, stitch_points_list[compare_idx])
    #
    #         # 如果歧义点找到了合理的邻近匹配
    #         if is_valid:
    #             st_point[1] = [st_point[1][p_idx]]
    #             is_current_matched = True
    #
    #         # 没有邻近匹配
    #         else:
    #             # [todo] 可能有能改进的地方
    #             if sp_idx == 0:
    #                 st_point[1] = [st_point[1][1]]
    #             elif sp_idx == len(stitch_points_list)-1:
    #                 st_point[1] = [st_point[1][0]]
    #             else:
    #                 stitch_points_list.remove(st_point)
    #
    #         is_previous_matched = is_current_matched

    pass
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
            for sp_idx, st_point in enumerate(stitch_points_list):
                if sp_idx == 0: continue
                # 计算双向的param_dis
                dis_d = cal_neigbor_points_param_dis(st_point[i], stitch_points_list[sp_idx - 1][i], all_panel_info)
                ord = 1 if dis_d[1] >= dis_d[0] else -1
                order_sum += ord
            # order_sum>=0时，我们认为这个缝边是顺时针的，此时
            isCC_order_list[-1][i] = order_sum < 0


    # 为每组连续且属于同一个edge的点按照param进行排序 --------------------------------------------------------------------------
    cmpfun = lambda x: (x["global_param"])
    for s_idx, stitch_points_list in enumerate(all_stitch_points_list):
        unordered_stitch_points_list = []
        start_panel_id = stitch_points_list[0][1]["edge_id"]
        start_point_idx = 0
        for point_idx, stitch_point in enumerate(stitch_points_list):
            # 如果edge_id发生了变化，或是这个Panel结束了
            if stitch_point[1]["edge_id"] != start_panel_id or point_idx == len(stitch_points_list)-1:
                if stitch_point[1]["edge_id"] == start_panel_id and point_idx == len(stitch_points_list)-1:
                    unordered_stitch_points_list.append(stitch_point[1])

                if isCC_order_list[s_idx][1]: reverse = True
                else: reverse = False

                ordered_stitch_points_list = deepcopy(sorted(unordered_stitch_points_list, key=cmpfun, reverse=reverse))
                for i in range(len(unordered_stitch_points_list)):
                    stitch_points_list[start_point_idx + i][1] = ordered_stitch_points_list[i]

                start_point_idx = point_idx
                start_panel_id = stitch_point[1]["edge_id"]
                unordered_stitch_points_list = [stitch_point[1]]
            else:
                unordered_stitch_points_list.append(stitch_point[1])


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



    # 优化相邻缝合的param -------------------------------------------------------------------------------------------------
    # 按所在板片进行排序后的 stitch_edge_list
    stitch_edge_list_panelOrder = sorted(stitch_edge_list, key=lambda x: (x["start_point"]["panel_id"],))
    stitch_edge_list_paramOrder = []
    start_panel_id = stitch_edge_list_panelOrder[0]["start_point"]["panel_id"]
    for e_idx, stitch_edge in enumerate(stitch_edge_list_panelOrder):
        # 如果panel_id发生了变化，或是这个edge是最后一个
        if stitch_edge["start_point"]["panel_id"] != start_panel_id or e_idx == len(stitch_edge_list_panelOrder) - 1:
            # 如果最后一个的panel_id没变化
            if not stitch_edge["start_point"]["panel_id"] != start_panel_id and e_idx == len(stitch_edge_list_panelOrder) - 1:
                stitch_edge_list_paramOrder.append(stitch_edge)

            # 对这个stitch_edge_list_paramOrder根据global_param进行排序（isCC=false根据起始点排序，isCC=true根据终点点排序）
            stitch_edge_list_paramOrder = sorted(stitch_edge_list_paramOrder, key=lambda x:(x["start_point"]["global_param"] if not x["isCC"] else x["end_point"]["global_param"]))

            # 对同一panel上的缝边，优化它们的param
            optimize_stitch_edge_list(stitch_edge_list_paramOrder, param_dis_optimize_thresh, all_panel_info, all_edge_info)

            # 切换到下一个 panel
            start_panel_id = stitch_edge["start_point"]["panel_id"]
            stitch_edge_list_paramOrder = [stitch_edge]
        else:
            stitch_edge_list_paramOrder.append(stitch_edge)



    # 将长度特别短的缝边删除 ------------------------------------------------------------------------------------------------
    thresh = 0.08
    filtered_stitch_edge_list = []
    for start_stitch_edge, end_stitch_edge in zip(stitch_edge_list[::2], stitch_edge_list[1::2]):
        # [modified]
        # param_dis_st = cal_stitch_edge_param_dis(end_stitch_edge, all_panel_info)
        # param_dis_ed = cal_stitch_edge_param_dis(end_stitch_edge, all_panel_info)
        param_dis_st = cal_stitch_edge_param_dis(start_stitch_edge, all_panel_info)
        param_dis_ed = cal_stitch_edge_param_dis(end_stitch_edge, all_panel_info)
        if param_dis_st<thresh or param_dis_ed<thresh:
            continue
        filtered_stitch_edge_list.append(start_stitch_edge)
        filtered_stitch_edge_list.append(end_stitch_edge)
    stitch_edge_list = filtered_stitch_edge_list


    # 将缝合两端都完全衔接的相邻的缝合进行合并 ---------------------------------------------------------------------------------
    stitch_edge_list_merged = []  # 合并后的缝边
    # 按所在板片进行排序后的 stitch_edge_list
    stitch_edge_list_panelOrder = sorted(stitch_edge_list[::2], key=lambda x: (x["start_point"]["panel_id"],))
    stitch_edge_list_paramOrder = []
    sum=0
    start_panel_id = stitch_edge_list_panelOrder[0]["start_point"]["panel_id"]
    for e_idx, stitch_edge in enumerate(stitch_edge_list_panelOrder):
        sum+=1
        if stitch_edge["start_point"]["panel_id"] != start_panel_id or e_idx == len(stitch_edge_list_panelOrder) - 1:
            # 如果最后一个的panel_id没变化
            if stitch_edge["start_point"]["panel_id"] == start_panel_id and e_idx == len(stitch_edge_list_panelOrder) - 1:
                stitch_edge_list_paramOrder.append(stitch_edge)

            # 【此时，stitch_edge_list_paramOrder中所有的缝边都位于同一panel上】
            # 对这个stitch_edge_list_paramOrder根据global_param进行排序（isCC=false根据起始点排序，isCC=true根据终点点排序）
            stitch_edge_list_paramOrder = sorted(stitch_edge_list_paramOrder, key=lambda x: (x["start_point"]["global_param"] if not x["isCC"] else x["end_point"]["global_param"]))

            # 对同一panel上的缝边，尝试合并它们 ------------------------------------------------------------------------------
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
                st_param_dis_d = cal_neigbor_points_param_dis(pre_right_point, cur_left_point, all_panel_info)
                if st_param_dis_d and min(st_param_dis_d) < 0.001:
                    end_stitch_edge = start_stitch_edge["target_edge"]
                    end_stitch_edge_previous = start_stitch_edge_previous["target_edge"]
                    # 计算相邻 被缝边 的双向间距
                    ed_param_dis_d = cal_neigbor_points_param_dis(end_stitch_edge_previous[pre_right_key], end_stitch_edge[cur_left_key], all_panel_info)
                    if ed_param_dis_d and min(ed_param_dis_d) < 0.001:
                        stitch_edge_list_paramOrder[se_idx][cur_left_key] = start_stitch_edge_previous[pre_left_key]
                        stitch_edge_list_paramOrder[se_idx]["target_edge"][cur_left_key] = end_stitch_edge_previous[pre_left_key]
                        del stitch_edge_list_paramOrder[start_stitch_edge_previous_index]

            for se in stitch_edge_list_paramOrder:
                stitch_edge_list_merged.extend([se, se["target_edge"]])

            # 如果最后一个的panel_id有变化
            if stitch_edge["start_point"]["panel_id"] != start_panel_id and e_idx == len(stitch_edge_list_panelOrder) - 1:
                stitch_edge_list_merged.extend([stitch_edge, stitch_edge["target_edge"]])

            # 切换到下一个 panel
            start_panel_id = stitch_edge["start_point"]["panel_id"]
            stitch_edge_list_paramOrder = [stitch_edge]
        else:
            stitch_edge_list_paramOrder.append(stitch_edge)

    stitch_edge_list = stitch_edge_list_merged




    # 将 离边端点特别近 且 不与其它缝边相衔接 的缝边的端点 的位置进行调整 ------------------------------------------------------------------------------------
    thresh = 5
    adjust_param = 0.023 / 0.023  # 这个用于根据采样频率调整阈值的大小 [todo] 预留接口
    thresh = thresh * adjust_param

    stitch_edge_list_panelOrder = sorted(stitch_edge_list, key=lambda x: (x["start_point"]["panel_id"],))
    stitch_edge_list_paramOrder = []
    start_panel_id = stitch_edge_list_panelOrder[0]["start_point"]["panel_id"]
    for e_idx, stitch_edge in enumerate(stitch_edge_list_panelOrder):
        if stitch_edge["start_point"]["panel_id"] != start_panel_id or e_idx == len(stitch_edge_list_panelOrder) - 1:
            # 如果最后一个的panel_id没变化
            if not stitch_edge["start_point"]["panel_id"] != start_panel_id and e_idx == len(stitch_edge_list_panelOrder) - 1:
                stitch_edge_list_paramOrder.append(stitch_edge)

            # 【此时，stitch_edge_list_paramOrder中所有的缝边都位于同一panel上】
            # 对这个stitch_edge_list_paramOrder根据global_param进行排序（isCC=false根据起始点排序，isCC=true根据终点点排序）
            stitch_edge_list_paramOrder = sorted(stitch_edge_list_paramOrder, key=lambda x: (x["start_point"]["global_param"] if not x["isCC"] else x["end_point"]["global_param"]))

            # 对同一panel上的缝边，尝试合并它们 ------------------------------------------------------------------------------
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

                # 如果相邻缝边间距较大
                st_index_dis_d = cal_neigbor_points_index_dis(pre_right_point, cur_left_point, all_panel_info)
                if st_index_dis_d and min(st_index_dis_d) > thresh:
                    for point in [pre_right_point, cur_left_point]:
                        edge = all_edge_info[point["edge_id"]]
                        edge_side_point = [edge["points_info"][0], edge["points_info"][-1]]
                        # 缝边端点 到 所在边的两个端点 的距离
                        side_point_index_dis = [
                            min(cal_neigbor_points_index_dis(point, edge_side_point[0],  all_panel_info)),
                            min(cal_neigbor_points_index_dis(point, edge_side_point[1], all_panel_info)),
                        ]
                        # 获取 前一条缝边的右点 和 当前缝边的左点 离各自边上两个端点的距离
                        side_point_index_dis_index = 0 if side_point_index_dis[0] < side_point_index_dis[1] else 1
                        if 0 < side_point_index_dis[side_point_index_dis_index] <= thresh:
                            closed_side_point = edge_side_point[side_point_index_dis_index]
                            point["id"] = closed_side_point["id"]
                            point["param"] = closed_side_point["param"]
                            point["global_param"] = closed_side_point["global_param"]

            # 切换到下一个 panel
            start_panel_id = stitch_edge["start_point"]["panel_id"]
            stitch_edge_list_paramOrder = [stitch_edge]
        else:
            stitch_edge_list_paramOrder.append(stitch_edge)



    # 将缝合数据转换为 AIGP 文件中 "stitches" 的格式 ------------------------------------------------------------------------
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
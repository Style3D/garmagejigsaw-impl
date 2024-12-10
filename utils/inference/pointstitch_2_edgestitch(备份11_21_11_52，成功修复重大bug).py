# 从点点缝合关系获取边边缝合关系
import json
from copy import deepcopy

import torch

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
        "start_point":start_point,
        "end_point":end_point,
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

def calculate_param_distance(start_point_info, end_point_info, all_panel_info):
    """
    :param start_point_info:
    :param end_point_info:
    :param all_panel_info:
    :return:
    """

    # 不在同一个panel上的就不计算了
    if start_point_info["panel_id"] != end_point_info["panel_id"]:
        return None

    panel_id = start_point_info["panel_id"]
    panel_info = all_panel_info[panel_id]

    p_start, p_end = start_point_info["global_param"], end_point_info["global_param"]
    panel_p_start, panel_p_end = panel_info["param_start"], panel_info["param_end"]

    # 计算在panel的这个环上，start_point从左右两端到end_point的param distance
    if p_start < p_end:
        dis_left = p_start - panel_p_start + panel_p_end - p_end
        dis_right = p_end - p_start
    elif p_start > p_end:
        dis_left = p_start - p_end
        dis_right = p_end - panel_p_start + panel_p_end - p_start
    else:
        dis_left=dis_right=0
    return [dis_left, dis_right]




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
                             param_dis_optimize_thresh = 0.7, filter_len_realworld=0.12):

    """
    :param batch:           # garment_jigsaw model input
    :param inf_rst:         # garment_jigsaw model output
    :param stitch_mat:          # mat of point&point stitch
    :param stitch_indices:      # indices of point&point stitch
    :param unstitch_thresh:     # continue unstitched point longer than this will create new stitch point list
    :param fliter_len:          # stitch point list shorter than this will be filtered
    :param filter_len_realworld:        # stitch edge whose realworld length less than this will be filtered
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
        # 这个Panel的param的最小值
        panel_info["param_start"] = global_param
        for edge_idx, (e_approx, e_approx_global) in enumerate(zip(panel_edges_approx, panel_edges_approx_global)):
            edge_json = panel_json["seqEdges"][0]["edges"][edge_idx]
            all_edge_info[edge_json["id"]] = {"id":edge_json["id"], "panel_id":panel_info["id"], "points_info":[], "points_index":None}
            edge_info = all_edge_info[edge_json["id"]]
            panel_info["edges_info"][edge_json["id"]] = edge_info

            # 获取这个Panel上每个edge上的所有点的按顺序的index
            if ((edge_idx==0 and e_approx_global[0]>e_approx_global[1]) or (edge_idx==panel_edge_num-1 and e_approx_global[0]>e_approx_global[1])):
                edge_points_idx = torch.concat([torch.arange(e_approx_global[0], point_end_idx), torch.arange(point_start_idx, e_approx_global[1]+1)])
                edge_points_idx.to(device_)
            else:
                edge_points_idx = torch.arange(e_approx_global[0], e_approx_global[1]+1)
            edge_info["points_index"] = edge_points_idx
            # 这个边的param的最小值
            edge_info["param_start"] = global_param

            for idx, point_idx in enumerate(edge_points_idx):
                # 计算当前点的param (我添加了一个很小的offset来防止一些报错，并且这些offset会在后续处理中被修复)
                param = idx/(len(edge_points_idx)-1)  # 这个点在边上的位置
                if idx==0: param+=very_small_gap
                elif idx==len(edge_points_idx)-1: param-=very_small_gap
                point_info = {"id": point_idx, "panel_id": panel_info["id"], "edge_id": edge_info["id"],
                              "param": param, "global_param": global_param + param}
                edge_info["points_info"].append(point_info)

                pt_key = point_idx.tolist()
                if not pt_key in all_point_info.keys():

                    all_point_info[pt_key] = [point_info]
                else:  # 重复的点（两个边的交汇处）
                    all_point_info[pt_key].append(point_info)

            # 一条边上的点全部计算完后，global_param+1后作为下一条边的起始 param_start
            global_param+=1

        # 这个Panel的param的最大值
        panel_info["param_end"] = global_param

    # 消除具有歧义的 边与边的中间点 ------------------------------------------------------------------------------------------
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

                # 防止不缝合的点导致的报错
                if point_idx_cor.tolist() != -1:
                    point_info_cor = all_point_info[point_idx_cor.tolist()]
                else:
                    point_info_cor = None

                # 对几种特殊情况进行分别处理 --------------------------------------------------------------------------------
                # 不缝合的点，连续出现数个则开新的缝合
                for _ in range(1):
                    # [modified]
                    # 当前点不缝合，且已经连续数个点都不缝合时，开一条新的缝合
                    if not point_info_cor:
                        unstitch_num+=1  # [modified]
                        if unstitch_num>unstitch_thresh:
                            unstitch_num=0
                            if len(stitch_points_list) > 0:
                                all_stitch_points_list.append(stitch_points_list)
                                stitch_points_list = []
                        break
                    else:
                        unstitch_num = 0

                    # # 缝合点所在拟合边发生变化
                    # if len(stitch_points_list) > 0:
                    #     if point_info["edge_id"] != stitch_points_list[-1][0]["edge_id"]:
                    #         all_stitch_points_list.append(stitch_points_list)
                    #         stitch_points_list=[]
                    #         stitch_points_list.append([point_info, point_info_cor])
                    #         break

                    # 根据 缝合点 的信息判断
                    if len(stitch_points_list) > 0:
                        # 缝合点所在Panel发生变化
                        if point_info["panel_id"] != stitch_points_list[-1][0]["panel_id"]:
                            all_stitch_points_list.append(stitch_points_list)
                            stitch_points_list=[]
                            stitch_points_list.append([point_info, point_info_cor])
                            break

                    # 根据 被缝合点 的信息判断
                    if len(stitch_points_list) > 0:
                        trigger = False
                        for p in point_info_cor:
                            if p["panel_id"] in [e["panel_id"] for e in stitch_points_list[-1][1]]:  # [modified]
                                trigger = True

                        # 被缝合点所在panel发生变化
                        if not trigger:
                            all_stitch_points_list.append(stitch_points_list)
                            stitch_points_list = []
                            stitch_points_list.append([point_info, point_info_cor])
                            break

                        # 被缝合点是歧义点（这代表这个点是拟合边的端点），且与上一个点是相关的
                        if trigger and len(point_info_cor)>1:
                            # stitch_points_list.append([point_info, point_info_cor])
                            # [modified]
                            stitch_points_list.append([point_info, point_info_cor])
                            all_stitch_points_list.append(stitch_points_list)
                            stitch_points_list = []
                            stitch_points_list.append([point_info, point_info_cor])
                            break

                    # 其它情况
                    stitch_points_list.append([point_info, point_info_cor])
                    break
    # 如果最后的stitch_points_list不为空，append到all
    if len(stitch_points_list) > 0:
        all_stitch_points_list.append(stitch_points_list)

    pass
    # # 到目前为止，可以保证all_stitch_points_list中的每一段缝合的两个边都各自位于同一Panel上
    # # [TEST] 下面循环仅用于验证这一结论：
    # for s_idx, stitch_points_list in enumerate(all_stitch_points_list):
    #     for point_idx, point in enumerate(stitch_points_list):
    #         if point_idx==0: continue
    #         point_left = stitch_points_list[point_idx-1]
    #         if point_left[0]["panel_id"] != point[0]["panel_id"]:
    #             raise ValueError("出现不满足条件的缝合边")
    #         trigger = False
    #         for p in point[1]:
    #             if p["panel_id"] in [e["panel_id"] for e in point_left[1]]:
    #                 trigger = True
    #         if not trigger:
    #             raise ValueError("出现不满足条件的被缝合边")
    pass

    # 将太短的全部过滤掉
    all_stitch_points_list = filter_too_short(all_stitch_points_list, fliter_len = fliter_len) # [modified]

    # 消除歧义点 ---------------------------------------------------------------------------------------------------------
    for s_idx, stitch_points_list in enumerate(all_stitch_points_list):
        is_previous_matched = False
        for sp_idx, st_point in enumerate(stitch_points_list):
            is_current_matched = False
            # 非歧义点不处理
            if len(st_point[1]) == 1: continue
            # 对于歧义点，找到用于对比的点的idx：compare_idx
            # if sp_idx == 0:
            if sp_idx==0 or is_previous_matched:
                compare_idx = 1
            else:
                compare_idx = sp_idx-1

            # 对目标点进行匹配
            is_valid, p_idx = is_valid_stitch_point(st_point, stitch_points_list[compare_idx])

            # 如果左点对比失败，且不是最后点
            if not is_valid and compare_idx==sp_idx-1 and sp_idx!=len(stitch_points_list)-1:
                # 对右点进行匹配
                compare_idx = sp_idx - 1
                is_valid, p_idx = is_valid_stitch_point(st_point, stitch_points_list[compare_idx])

            # 如果歧义点找到了合理的邻近匹配
            if is_valid:
                st_point[1] = [st_point[1][p_idx]]
                is_current_matched = True

            # 没有邻近匹配
            else:
                if sp_idx == 0:
                    st_point[1] = [st_point[1][1]]
                elif sp_idx == len(stitch_points_list)-1:
                    st_point[1] = [st_point[1][0]]
                else:
                    stitch_points_list.remove(st_point)

            # is_previous_matched = is_current_matched
    # 再将歧义点由list换成dict
    for s_idx, stitch_points_list in enumerate(all_stitch_points_list):
        for sp_idx, st_point in enumerate(stitch_points_list):
            st_point[1] = st_point[1][0]

    # 为每个缝合计算是否逆时针 ----------------------------------------------------------------------------------------------
    # 每个缝合边是否逆时针
    isCC_order_list = []
    for s_idx, stitch_points_list in enumerate(all_stitch_points_list):
        isCC_order_list.append([None,None])
        # 缝边和被缝边分别计算时针方向
        for i in range(2):
            order_sum = 0
            for sp_idx, st_point in enumerate(stitch_points_list):
                if sp_idx == 0: continue
                # [todo] 两种计算ord的方法二选一

                # 方法1：根据下一个点的循环global_param上的距离来判断ord
                param_dis =  calculate_param_distance(stitch_points_list[sp_idx - 1][i], st_point[i], all_panel_info)
                ord = 1 if param_dis[1] < param_dis[0] else -1
                # # 方法2：根据和下一个点的global_param上的距离来判断ord
                # ord = st_point[i]["global_param"] - stitch_points_list[sp_idx - 1][i]["global_param"]
                # ord = 1 if ord >= 0 else -1
                order_sum += ord
            # order_sum>=0时，我们认为这个缝边是顺时针的
            isCC_order_list[-1][i] = order_sum < 0
        # for sp_idx, st_point in enumerate(stitch_points_list):
        #     if sp_idx == 0: continue
        #     ord = st_point[0]["global_param"] - stitch_points_list[sp_idx - 1][0]["global_param"]
        #     ord = 1 if ord > 0 else -1
        #     order_sum += ord
        # order_list[-1][0] = order_sum < 0
        # order_sum = 0
        # for sp_idx, st_point in enumerate(stitch_points_list):
        #     if sp_idx == 0: continue
        #     ord = st_point[1]["global_param"] - stitch_points_list[sp_idx - 1][1]["global_param"]
        #     ord = 1 if ord > 0 else -1
        #     order_sum += ord
        # order_list[-1][1] = order_sum < 0


    # 为每组连续且属于同一个edge的点按照param进行排序
    cmpfun = lambda x: (x["global_param"])
    for s_idx, stitch_points_list in enumerate(all_stitch_points_list):
        unordered_stitch_points_list = []
        start_panel_id = stitch_points_list[0][1]["edge_id"]
        start_point_idx = 0
        for point_idx, stitch_point in enumerate(stitch_points_list):
            # 如果edge_id发生了变化，或是这个Panel结束了
            if stitch_point[1]["edge_id"] != start_panel_id or point_idx == len(stitch_points_list)-1:
                if not stitch_point[1]["edge_id"] != start_panel_id and point_idx == len(stitch_points_list)-1:
                    unordered_stitch_points_list.append(stitch_point[1])

                if isCC_order_list[s_idx][1]:
                    reverse = True
                else:
                    reverse = False
                ordered_stitch_points_list = deepcopy(sorted(unordered_stitch_points_list, key=cmpfun, reverse=reverse))
                for i in range(len(unordered_stitch_points_list)):
                    stitch_points_list[start_point_idx + i][1] = ordered_stitch_points_list[i]
                start_point_idx = point_idx
                unordered_stitch_points_list = []
            else:
                unordered_stitch_points_list.append(stitch_point[1])

    # 获取缝合信息 --------------------------------------------------------------------------------------------------------
    # # [todo] 将一些每缝到底的线扩充下（比如end的para为0.96，但下条缝合边不在这个边上了）
    #  # 如果当前边的起点的param离边的某一端低于expend_thresh，则调整param
    # for se_idx, se_cur in enumerate(stitch_edge_list):
    #     for i in range(2):
    #         se_pre = stitch_edge_list[se_idx - 1]
    #
    #         # 判断起点和前面的边是否完全衔接，如果是则不进行操作
    #         if (se_pre[i]["end"]["edgeId"] == se_cur[i]["start"]["edgeId"] and
    #             se_pre[i]["end"]["param"] == se_cur[i]["start"]["param"]):
    #             continue
    #         # 如果当前边的起点的param离边的某一端很近，则调整param
    #         if se_cur[i]["start"]["param"] < expend_thresh and not se_cur[i]['isCounterClockWise']:
    #             se_cur[i]["start"]["param"] = 0
    #         elif 1 - se_cur[i]["start"]["param"] < expend_thresh and se_cur[i]['isCounterClockWise']:
    #             se_cur[i]["start"]["param"] = 1
    #         # # [OPTIONAL] 看看是否在同一边上，如果是，让他们衔接
    #         # if(se_pre[i]["end"]["edgeId"] == se_cur[i]["start"]["edgeId"] and
    #         #    se_pre[i]["end"]["param"] != se_cur[i]["start"]["param"]):
    #         #     mean_param = (se_pre[i]["end"]["param"]+se_cur[i]["start"]["param"])/2
    #         #     se_pre[i]["end"]["param"] = se_cur[i]["start"]["param"] = mean_param
    #
    #         if se_idx == len(stitch_edge_list)-1:
    #             se_next = stitch_edge_list[0]
    #         else:
    #             se_next = stitch_edge_list[se_idx+1]
    #         # 判断终点和后面的边是否完全衔接，如果是则不进行操作
    #         if (se_cur[i]["end"]["edgeId"] == se_next[i]["start"]["edgeId"] and
    #             se_cur[i]["end"]["param"] == se_next[i]["start"]["param"]):
    #             continue
    #         # 如果当前边的终点的param离边的某一端很近，则调整param
    #         if 1-se_cur[i]["end"]["param"] <expend_thresh and not se_cur[i]['isCounterClockWise']:
    #             se_cur[i]["end"]["param"] = 1
    #         elif se_cur[i]["end"]["param"] <expend_thresh and se_cur[i]['isCounterClockWise']:
    #             se_cur[i]["end"]["param"] = 0
    #         else:
    #             # 如果当前边的终点的param离边的某一端很近，则调整param
    #             if 1 - se_cur[i]["end"]["param"] < expend_thresh and not se_cur[i]['isCounterClockWise']:
    #                 se_cur[i]["end"]["param"] = 1
    #             elif se_cur[i]["end"]["param"] < expend_thresh and se_cur[i]['isCounterClockWise']:
    #                 se_cur[i]["end"]["param"] = 0
    pass
    # [todo:OPT] 看看能不能将线线优化的结果应用到点点缝合（说不定有用）


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

    # target_idx = 0
    # panel_id_list = []
    # new_se_list = []
    # for se_idx, stitch_edge in enumerate(stitch_edge_list[::2]):
    #     if stitch_edge["start_point"]["panel_id"] not in panel_id_list:
    #         panel_id_list.append(stitch_edge["start_point"]["panel_id"])
    #     if len(panel_id_list)-1==target_idx:
    #         new_se_list.append(stitch_edge)
    #         new_se_list.append(stitch_edge_list[se_idx*2+1])
    # stitch_edge_list = new_se_list
    # a=1

    pass
    # # 优化相邻缝合的param -------------------------------------------------------------------------------------------------
    # # 先将离端点较近的处理下 =======================
    # thresh = 0.1
    # for stitch_edge in stitch_edge_list:
    #     for pt in [stitch_edge["start_point"], stitch_edge["end_point"]]:
    #         if pt["param"] < thresh:
    #             pt["global_param"] -= pt["param"]
    #             pt["param"] = 0
    #         elif 1-pt["param"] < thresh:
    #             pt["global_param"] += 1-pt["param"]
    #             pt["param"] = 1
    pass

    # 优化param ==========================
    # 按所在板片进行排序后的 stitch_edge_list
    stitch_edge_list_panelOrder = sorted(stitch_edge_list,
        key=lambda x: (x["start_point"]["panel_id"],))
    stitch_edge_list_paramOrder = []
    start_panel_id = stitch_edge_list_panelOrder[0]["start_point"]["panel_id"]
    for e_idx, stitch_edge in enumerate(stitch_edge_list_panelOrder):
        # 如果panel_id发生了变化，或是这个edge是最后一个
        if stitch_edge["start_point"]["panel_id"] != start_panel_id or e_idx == len(stitch_edge_list_panelOrder) - 1:
            # 如果最后一个的panel_id没变化
            if not stitch_edge["start_point"]["panel_id"] != start_panel_id and e_idx == len(stitch_edge_list_panelOrder) - 1:
                stitch_edge_list_paramOrder.append(stitch_edge)
            # 对这个stitch_edge_list_paramOrder根据global_param进行排序（isCC=false根据起始点排序，isCC=true根据终点点排序）
            stitch_edge_list_paramOrder = sorted(stitch_edge_list_paramOrder,
                                                 key=lambda x:(x["start_point"]["global_param"] if not x["isCC"]
                                                               else x["end_point"]["global_param"]))
            # [todo] 让按照顺时针排序完的edges，相邻之间进行判断是否有某种情况，以及怎么处理
            def optimize_stitch_edge_list_paramOrder(stitch_edge_list_paramOrder, param_dis_optimize_thresh, all_panel_info):
                try:
                    current_panel_info = all_panel_info[stitch_edge_list_paramOrder[0]['start_point']['panel_id']]
                except IndexError:
                    a=1
                for se_idx, stitch_edge in enumerate(stitch_edge_list_paramOrder):
                    if se_idx == 0:
                        stitch_edge_previous = stitch_edge_list_paramOrder[-1]
                    else:
                        stitch_edge_previous = stitch_edge_list_paramOrder[se_idx-1]

                    # 获取两个边的可能衔接的部分
                    pre_end_key = "end_point" if not stitch_edge_previous["isCC"] else "start_point"
                    cur_start_key = "start_point" if not stitch_edge["isCC"] else "end_point"
                    pre_end_point = stitch_edge_previous[pre_end_key]
                    cur_start_point = stitch_edge[cur_start_key]

                    # 计算两个点的param的差距
                    if cur_start_point['global_param'] >= pre_end_point['global_param']:
                        param_dis = cur_start_point['global_param'] - pre_end_point['global_param']
                    else:
                        param_dis = (current_panel_info['param_end'] - pre_end_point['global_param'] +
                                    cur_start_point['global_param'] - current_panel_info['param_start'])

                    a=1
                    # 对于 param dis 小于阈值的相邻缝合边，对它们相衔接的部分进行优化
                    if abs(param_dis) < param_dis_optimize_thresh:
                        if cur_start_point['edge_id'] == pre_end_point['edge_id']:
                            mid = (cur_start_point['global_param'] + pre_end_point['global_param']) / 2
                            cur_start_point['global_param'] = pre_end_point['global_param'] = mid
                            cur_start_point['param'] = pre_end_point['param'] = mid % 1
                            a = 1
                        elif cur_start_point['edge_id'] != pre_end_point['edge_id']:
                            if cur_start_point['global_param'] >= pre_end_point['global_param']:
                                # 如果两个属于同一边
                                # [todo] 如果属于同一边，且有一个是端点，则另一个点移过来
                                # 不属于同一边，但这两个边是相邻的
                                pre_end_point['global_param'] += 1-pre_end_point['param']
                                pre_end_point['param'] = 1
                                cur_start_point['global_param'] -= cur_start_point['param']
                                cur_start_point['param'] = 0
                                a=1
                            else:
                                pre_end_point['global_param'] += 1 - pre_end_point['param']
                                pre_end_point['param'] = 1
                                cur_start_point['global_param'] -= cur_start_point['param']
                                cur_start_point['param'] = 0
                                a=1
                a=1
            optimize_stitch_edge_list_paramOrder(stitch_edge_list_paramOrder, param_dis_optimize_thresh, all_panel_info)

            start_panel_id = stitch_edge["start_point"]["panel_id"]
            stitch_edge_list_paramOrder = []
            stitch_edge_list_paramOrder.append(stitch_edge)
        else:
            stitch_edge_list_paramOrder.append(stitch_edge)
    a=1

    # 将缝合数据转换为 AIGP 文件中 "stitches" 的格式 这一步分离出来到这里
    stitch_edge_json_list = []
    for start_stitch_edge, end_stitch_edge in zip(stitch_edge_list[::2], stitch_edge_list[1::2]):
        stitch_edge_json = get_new_stitch()
        # [todo] 具体定义 “从尾到头” 的情况（比如param突然骤降，但是依旧是在同一个边上）
        # [todo] start&end_stitch_points取param最靠近端点的
        # [todo] 又或者可以试试先对所有点按照globalparam进行一次排序（要考虑到 “从尾到头” 的情况）

        # [todo] 计算param的过程要考虑到 “从尾到头” 的情况
        stitch_edge_json[0]["isCounterClockWise"] = start_stitch_edge["isCC"]
        stitch_edge_json[1]["isCounterClockWise"] = end_stitch_edge["isCC"]

        apply_stitch_param(stitch_edge_json[0]["start"], start_stitch_edge["start_point"])
        apply_stitch_param(stitch_edge_json[0]["end"], start_stitch_edge["end_point"])
        apply_stitch_param(stitch_edge_json[1]["start"], end_stitch_edge["start_point"])
        apply_stitch_param(stitch_edge_json[1]["end"], end_stitch_edge["end_point"])

        # 删除一些para过短的进行过滤处理 ------------------------------------------------------------------------------------
        # [todo] 改成根据"真实距离"删（小于0.1CM）
        if ((stitch_edge_json[0]["start"]["param"] == stitch_edge_json[0]["end"]["param"] and
             stitch_edge_json[0]["start"]["edgeId"] == stitch_edge_json[0]["end"]["edgeId"])
                or
                (stitch_edge_json[1]["start"]["param"] == stitch_edge_json[1]["end"]["param"] and
                 stitch_edge_json[1]["start"]["edgeId"] == stitch_edge_json[1]["end"]["edgeId"])):
            continue
        stitch_edge_json_list.append(stitch_edge_json)

    # # [todo]将缝合数据转换为 AIGP 文件中 "stitches" 的格式 这一步分离出来到这里
    # for s_idx, stitch_points_list in enumerate(all_stitch_points_list):
    #     stitch_edge = get_new_stitch()
    #     # [todo] 具体定义 “从尾到头” 的情况（比如param突然骤降，但是依旧是在同一个边上）
    #     # [todo] start&end_stitch_points取param最靠近端点的
    #     # [todo] 又或者可以试试先对所有点按照globalparam进行一次排序（要考虑到 “从尾到头” 的情况）
    #     start_stitch_points = stitch_points_list[0]
    #     end_stitch_points = stitch_points_list[-1]
    #
    #     # [todo] 计算param的过程要考虑到 “从尾到头” 的情况
    #     # [todo] 计算order的部分移到前面去了，记得改这里的
    #     order_sum = 0
    #     for sp_idx, st_point in enumerate(stitch_points_list):
    #         if sp_idx == 0: continue
    #         ord = st_point[0]["global_param"] - stitch_points_list[sp_idx - 1][0]["global_param"]
    #         ord = 1 if ord > 0 else -1
    #         order_sum += ord
    #     stitch_edge[0]["isCounterClockWise"] = order_sum < 0
    #
    #     order_sum = 0
    #     for sp_idx, st_point in enumerate(stitch_points_list):
    #         if sp_idx == 0: continue
    #         ord = st_point[1]["global_param"] - stitch_points_list[sp_idx - 1][1]["global_param"]
    #         # # 属于同一panel的被缝合点会出现套圈的情况（），如果出现这一情况则将ord乘-1(有问题，最好别用)
    #         # if abs(ord) >= 2. and st_point[1]["panel_id"]==stitch_points_list[sp_idx - 1][1]["panel_id"]:
    #         #     ord *= -1
    #         ord = 1 if ord > 0 else -1
    #         order_sum += ord
    #     stitch_edge[1]["isCounterClockWise"] = order_sum < 0
    #
    #     apply_stitch_param(stitch_edge[0]["start"], start_stitch_points[0])
    #     apply_stitch_param(stitch_edge[0]["end"], end_stitch_points[0])
    #     apply_stitch_param(stitch_edge[1]["start"], start_stitch_points[1])
    #     apply_stitch_param(stitch_edge[1]["end"], end_stitch_points[1])
    #
    #     # 删除一些para过短的进行过滤处理 ------------------------------------------------------------------------------------
    #     # [todo] 改成根据"真实距离"删（小于0.1CM）
    #     # calculate_param_distance(stitch_edge[0]["start"], stitch_edge[0]["end"])  # todo[TEST]
    #     if ((stitch_edge[0]["start"]["param"] == stitch_edge[0]["end"]["param"] and
    #          stitch_edge[0]["start"]["edgeId"] == stitch_edge[0]["end"]["edgeId"])
    #             or
    #             (stitch_edge[1]["start"]["param"] == stitch_edge[1]["end"]["param"] and
    #              stitch_edge[1]["start"]["edgeId"] == stitch_edge[1]["end"]["edgeId"])):
    #         continue
    #     stitch_edge_list.append(stitch_edge)


    garment_json["stitches"] = stitch_edge_json_list
    return garment_json
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

def calculate_param_distance(start, end, is_cc=False):
    """
    :param start:
    :param end:
    :param is_cc: is counterclockwise
    :return:
    """







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
                             unstitch_thresh = 3, fliter_len=3, filter_para_len=0.15, expend_thresh=0.15):

    """
    :param batch:           # garment_jigsaw model input
    :param inf_rst:         # garment_jigsaw model output
    :param stitch_mat:          # mat of point&point stitch
    :param stitch_indices:      # indices of point&point stitch
    :param unstitch_thresh:
    :param fliter_len:
    :param expend_thresh:
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
            edge_info["param_start"] = global_param
            for idx, point_idx in enumerate(edge_points_idx):
                param = idx/(len(edge_points_idx)-1)  # 这个点在边上的位置
                point_info={"id":point_idx ,"panel_id":panel_info["id"], "edge_id":edge_info["id"],
                            "param":param, "global_param":global_param+param}
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
    stitch_mat[stitch_indices[:, 0]] = stitch_indices[:, 1]
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

                    # 根据缝合点的信息判断
                    if len(stitch_points_list) > 0:
                        # [modified] 缝合点所在Panel发生变化
                        if point_info["panel_id"] != stitch_points_list[-1][0]["panel_id"]:
                            all_stitch_points_list.append(stitch_points_list)
                            stitch_points_list=[]
                            stitch_points_list.append([point_info, point_info_cor])
                            break

                    # 根据被缝合点的信息判断
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

                        # 被缝合点是歧义点（这代表这个点是拟合边的端点）
                        if trigger and len(point_info_cor)>1:
                            stitch_points_list.append([point_info, point_info_cor])
                            # [modified]
                            # stitch_points_list.append([point_info, point_info_cor])
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

    # 到目前为止，可以保证all_stitch_points_list中的每一段缝合的两个边都各自位于同一Panel上
    # [TEST] 下面循环仅用于验证这一结论：
    for s_idx, stitch_points_list in enumerate(all_stitch_points_list):
        for point_idx, point in enumerate(stitch_points_list):
            if point_idx==0: continue
            point_left = stitch_points_list[point_idx-1]
            if point_left[0]["panel_id"] != point[0]["panel_id"]:
                raise ValueError("出现不满足条件的缝合边")
            trigger = False
            for p in point[1]:
                if p["panel_id"] in [e["panel_id"] for e in point_left[1]]:
                    trigger = True
            if not trigger:
                raise ValueError("出现不满足条件的被缝合边")

    all_stitch_points_list = filter_too_short(all_stitch_points_list, fliter_len = fliter_len) # [modified]

    # 消除歧义点 ---------------------------------------------------------------------------------------------------------
    for s_idx, stitch_points_list in enumerate(all_stitch_points_list):
        for sp_idx, st_point in enumerate(stitch_points_list):
            # 非歧义点不处理
            if len(st_point[1]) == 1: continue
            # 对于歧义点，找到用于对比的点的idx：compare_idx
            if sp_idx==0:
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
            # 如果歧义点找到了合理的匹配
            if is_valid:    # 有邻近匹配
                st_point[1] = [st_point[1][p_idx]]
            else:   # 没有邻近匹配
                if sp_idx == 0:
                    st_point[1] = [st_point[1][1]]
                elif sp_idx == len(stitch_points_list)-1:
                    st_point[1] = [st_point[1][0]]
                else:
                    # [todo] 暂时默认采用0点
                    stitch_points_list.remove(st_point)
    # 再将歧义点由list换成dict
    for s_idx, stitch_points_list in enumerate(all_stitch_points_list):
        for sp_idx, st_point in enumerate(stitch_points_list):
            st_point[1] = st_point[1][0]

    # 存放所有边缝合信息
    stitch_edge_list = []
    for s_idx, stitch_points_list in enumerate(all_stitch_points_list):
        stitch_edge = get_new_stitch()
        # [todo] 具体定义 “从尾到头” 的情况（不如param突然骤降，但是依旧是在同一个边上）
        # [todo] start&end_stitch_points取param最靠近端点的
        # [todo] 又或者可以试试先对所有点按照globalparam进行一次排序（要考虑到 “从尾到头” 的情况）
        start_stitch_points = stitch_points_list[0]
        end_stitch_points = stitch_points_list[-1]

        # [todo] 计算param的过程要考虑到 “从尾到头” 的情况
        order_sum = 0
        for sp_idx, st_point in enumerate(stitch_points_list):
            if sp_idx==0: continue
            ord = st_point[0]["global_param"]-stitch_points_list[sp_idx-1][0]["global_param"]
            ord = 1 if ord>0 else -1
            order_sum+=ord
        stitch_edge[0]["isCounterClockWise"] = order_sum < 0

        order_sum = 0
        for sp_idx, st_point in enumerate(stitch_points_list):
            if sp_idx == 0: continue
            ord = st_point[1]["global_param"] - stitch_points_list[sp_idx - 1][1]["global_param"]
            ord = 1 if ord > 0 else -1
            order_sum+=ord
        stitch_edge[1]["isCounterClockWise"] = order_sum < 0

        apply_stitch_param(stitch_edge[0]["start"], start_stitch_points[0])
        apply_stitch_param(stitch_edge[0]["end"], end_stitch_points[0])
        apply_stitch_param(stitch_edge[1]["start"], start_stitch_points[1])
        apply_stitch_param(stitch_edge[1]["end"], end_stitch_points[1])

        # 删除一些para过短的进行过滤处理 ------------------------------------------------------------------------------------
        # [todo] 改成根据"global_param"来计算param distance，小于filter_para_len的都可以删掉
        if ((stitch_edge[0]["start"]["param"] == stitch_edge[0]["end"]["param"] and
            stitch_edge[0]["start"]["edgeId"] == stitch_edge[0]["end"]["edgeId"])
                or
            (stitch_edge[1]["start"]["param"] == stitch_edge[1]["end"]["param"] and
            stitch_edge[1]["start"]["edgeId"] == stitch_edge[1]["end"]["edgeId"])):
            continue
        stitch_edge_list.append(stitch_edge)
    a=1


    # [todo:OPT] 将一些被分隔开的（但是属于同一缝合的）边缝合 合并到一起去
    pass


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
    # [todo:OPT] 看看能不能将线线优化的结果应用到点点缝合（说不定有用）
    garment_json["stitches"] = stitch_edge_list
    return garment_json
# 从点点缝合关系获取边边缝合关系
# 第二版，为了将单一板片多个contour的情况考虑进去
import json
import uuid
import torch
import numpy as np
from copy import deepcopy

def sl2graph(stitch_indices):
    stitch_indices = torch.sort(stitch_indices, dim=1).values
    Graph = torch.unique(stitch_indices, dim=0)
    return Graph
# 获取一段随机的 uuid
def get_random_uuid():
    id = str(uuid.uuid4().hex)
    result = id[0:8] + "-" + id[8:12] + "-" + id[12:16] + "-" + id[16:20] + "-" + id[20:]
    return result

def apply_point_info(stitch_edge_side, point_info):
    stitch_edge_side['clothPieceId'] = point_info['contour_id']
    stitch_edge_side['edgeId'] = point_info['edge_id']
    stitch_edge_side['param'] = point_info['param']

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

def apply_stitch_param(st_eg, param):
    st_eg['clothPieceId'] = param['panel_id']
    st_eg['edgeId'] = param['edge_id']
    st_eg['param'] = param['param']



def load_data(batch):
    annotations_json_path = batch["annotations_json_path"][0]

    with open(annotations_json_path,"r") as af:
        annotations_json = json.load(af)

    panel_nes = torch.tensor(annotations_json["panel_nes"], dtype=torch.int64, device=batch["pcs"].device)
    contour_nes = torch.tensor(annotations_json["contour_nes"], dtype=torch.int64, device=batch["pcs"].device)
    edge_approx = torch.tensor(annotations_json["edge_approx"], dtype=torch.int64, device=batch["pcs"].device)
    panel_instance_seg = torch.tensor(annotations_json["panel_instance_seg"], dtype=torch.int64, device=batch["pcs"].device)

    return panel_nes, contour_nes, edge_approx, panel_instance_seg

# 从点点缝合推导出线线缝合
def ps2es4eval(batch, inf_rst, stitch_mat, stitch_indices,
                              unstitch_thresh = 6, fliter_len=3):

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
    num_parts = batch["num_parts"][0]
    # panel_nes, contour_nes, edge_approx, panel_instance_seg = load_data(batch)
    device_ = stitch_mat.device

    pcs = batch["pcs"][0]
    n_pcs = batch["n_pcs"][0]
    contour_num = torch.sum(n_pcs!=0)
    n_pcs = n_pcs[:contour_num]

    n_pcs_cumsum = torch.cumsum(n_pcs, dim=-1)
    contour_id2idx = {}

    # 将每个点、边、contour的关键信息进行汇总 ----------------------------------------------------------------------------------
    all_contour_info = {}
    all_point_info = {}
    all_edge_info = {}
    very_small_gap = 1e-5
    global_param = 0  # 一个点的全局param

    for contour_idx in range(len(n_pcs_cumsum)):
        if contour_idx == 0:
            point_start_idx = 0
        else:
            point_start_idx = n_pcs_cumsum[contour_idx - 1]
        point_end_idx = n_pcs_cumsum[contour_idx]

        # === 构建每个Panel的信息 ===
        contour_id = get_random_uuid()
        contour_id2idx[contour_id] = contour_idx

        all_contour_info[contour_id] = {"id":contour_id, "edges_info":{}}
        contour_info = all_contour_info[contour_id]
        contour_info["param_start"] = global_param  # 这个Panel的param的最小值
        contour_info["param_end"] = []
        contour_info["index_end"] = point_end_idx-1
        contour_info["index_start"] = point_start_idx if isinstance(point_start_idx,torch.Tensor) \
                                      else torch.tensor(point_start_idx, dtype=point_end_idx.dtype,device=point_end_idx.device)
        a=1
        for edge_idx in range(1):
            # === 构建每个Edge的信息 ===
            edge_id = get_random_uuid()
            all_edge_info[edge_id] = {"id":edge_id, "contour_id":contour_info["id"], "points_info":[]}
            edge_info = all_edge_info[edge_id]
            contour_info["edges_info"][edge_id] = edge_info
            # 获取这个Panel上每个edge上的所有点的按顺序的index
            edge_points_idx = torch.arange(point_start_idx, point_end_idx)
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
    stitch_mat[stitch_indices[:, 1]] = stitch_indices[:, 0]

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
                            # far = int(np.argmax(param_dis_d))

                            stitch_points_list.append([point_info, [point_info_cor[near]]])
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

    graph_result = torch.tensor([[contour_id2idx[sp[0][0]["contour_id"]],contour_id2idx[sp[0][1]["contour_id"]]] for sp in all_stitch_points_list])
    graph_result = sl2graph(graph_result)
    return graph_result
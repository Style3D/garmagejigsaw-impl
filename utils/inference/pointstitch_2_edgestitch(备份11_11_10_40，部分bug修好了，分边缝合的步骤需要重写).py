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

def load_data(batch):
    garment_json_path = batch["garment_json_path"][0]
    annotations_json_path = batch["annotations_json_path"][0]

    with open(garment_json_path, "r") as gf, open(annotations_json_path,"r") as af:
        garment_json = json.load(gf)
        annotations_json = json.load(af)

    panel_nes = torch.tensor(annotations_json["panel_nes"], dtype=torch.int64, device=batch["pcs"].device)
    edge_approx = torch.tensor(annotations_json["edge_approx"], dtype=torch.int64, device=batch["pcs"].device)

    return garment_json, panel_nes, edge_approx

 # [
 #            {
 #                "start": {
 #                    "clothPieceId": "panel_0",
 #                    "edgeId": "panel_0_e01",
 #                    "param": 0
 #                },
 #                "end": {
 #                    "clothPieceId": "panel_0",
 #                    "edgeId": "panel_0_e01",
 #                    "param": 1
 #                },
 #                "isCounterClockWise": false
 #            },
 #            {
 #                "start": {
 #                    "clothPieceId": "panel_1",
 #                    "edgeId": "panel_1_e04",
 #                    "param": 1
 #                },
 #                "end": {
 #                    "clothPieceId": "panel_1",
 #                    "edgeId": "panel_1_e04",
 #                    "param": 0
 #                },
 #                "isCounterClockWise": true
 #            }
 #        ],

# 从点点缝合推导出线线缝合
def pointstitch_2_edgestitch(batch, inf_rst, stitch_mat, stitch_indices):
    # 读取数据（aigp文件 和 边拟合结果）
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

    # 将拟合边的坐标转换成全局坐标
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

            for idx, point_idx in enumerate(edge_points_idx):
                param = idx/(len(edge_points_idx)-1)  # 这个点在边上的位置
                point_info={"id":point_idx ,"panel_id":panel_info["id"], "edge_id":edge_info["id"], "param":param}
                edge_info["points_info"].append(point_info)

                pt_key = point_idx.tolist()
                if not pt_key in all_point_info.keys():
                    all_point_info[pt_key] = [point_info]
                else:  # 重复的点（两个边的交汇处）
                    all_point_info[pt_key].append(point_info)
                a=1
            a=1
        a=1
    a=1



    # 消除具有歧义的边与边的中间点 ------------------------------------------------------------------------------------------
    # ===================================================
    # [TODO] 先去获得所有缝合点对的缝合划分，着重处理边与边的中间点
    # 缝合关系的映射
    stitch_mat = torch.zeros((len(pcs)), dtype=torch.int64, device=device_)-1
    stitch_mat[stitch_indices[:, 0]] = stitch_indices[:, 1]
    stitch_mat[stitch_indices[:, 1]] = stitch_indices[:, 0]

    all_stitch_points_list = []
    for panel_id in all_panel_info:
        panel_info = all_panel_info[panel_id]
        edges_info = panel_info["edges_info"]

        # 遍历这个Panel上的所有缝合点对
        for edge_id in edges_info:
            edge_info = edges_info[edge_id]
            points_info = edge_info["points_info"]

            stitch_points_list = []
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
                # 不缝合的点
                for _ in range(1):
                    if not point_info_cor:
                        # if len(stitch_points_list)>0:
                        #     all_stitch_points_list.append(stitch_points_list)
                        #     stitch_points_list = []
                        break
                    # 边上的第一个点
                    elif idx==0:
                        stitch_points_list.append([point_info, point_info_cor])
                        break
                    # 歧义点（处于边与边的交界处，同时作为起点和终点的点）
                    elif len(stitch_points_list)>0:
                        trigger = False
                        for p in point_info_cor:
                            if p["edge_id"] == stitch_points_list[-1][1][0]["edge_id"]:
                                trigger=True
                        if not trigger:
                            stitch_points_list.append([point_info, point_info_cor])
                            all_stitch_points_list.append(stitch_points_list)
                            stitch_points_list = []
                            stitch_points_list.append([point_info, point_info_cor])
                            break
                    # 其它情况
                    if len(point_info_cor)==1:
                        stitch_points_list.append([point_info, point_info_cor])
                        break

                # 边上的最后一个点
                if idx == len(points_info) - 1:
                    if len(stitch_points_list) > 0:
                        all_stitch_points_list.append(stitch_points_list)
                        stitch_points_list = []

    # 到目前为止，可以保证all_stitch_points_list中的每一段缝合的两个边都各自位于同一拟合边上
    # 下面循环仅用于验证这一结论：
    for s_idx, stitch_points_list in enumerate(all_stitch_points_list):
        tow_edges_idx = []
        for point_idx, point in enumerate(stitch_points_list):
            if point_idx == 0:
                tow_edges_idx = [point[0]["edge_id"], point[1][0]["edge_id"]]
            else:
                if point[0]["edge_id"]!=tow_edges_idx[0] or not tow_edges_idx[1] in [p["edge_id"] for p in point[1]]:
                    raise ValueError("出现不满足条件的缝合边!")

    # [todo] 根据边上的点的信息去消除歧义点
    # 用边内信息判断完后，可以紧接一轮边之间的判断
    for s_idx, stitch_points_list in enumerate(all_stitch_points_list):
        order_dict = {}
        is_des = False
        # cor_list =

    # [todo] 将一些被分隔开的（但是属于同一缝合的）边缝合 合并到一起去

    # [todo] 将一些每缝到底的线扩充下（比如end的para为0.96，但下条缝合边不在这个边上了）

    a=1
    # ===================================================

    # 生成 stitch -------------------------------------------------------------------------------------------------------
    stitch = {}
    for panel_id in all_panel_info:
        panel_info = all_panel_info[panel_id]
        edges_info = panel_info["edges_info"]
        # 这个Panel上的所有缝合点对
        all_stitch_points_list = []
        for edge_id in edges_info:
            edge_info = edges_info[edge_id]
            points_info = edge_info["points_info"]

            stitch = None
            stitch_points_list = []
            for idx, point_info in enumerate(points_info):
                # 当前点 和它缝合的点
                point_idx = point_info["id"]
                point_idx_cor = stitch_mat[point_idx]
                point_info_cor = all_point_info[point_idx_cor.tolist()]
                if len(point_info_cor) == 1:
                    point_info_cor = point_info_cor[0]
                elif len(point_info_cor) == 2 :
                    pass
                stitch_points_list.append([point_info, point_info_cor])

                # [TODO] 判断是否结束边

                # 判断是否生成新的 stitch
                stitch_start = False
                if idx==0:
                    stitch_start = True
                    if idx==0 and stitch:
                        raise ValueError
                elif True:
                    a=1

                # 生成新的 stitch
                if stitch_start:
                    stitch = get_new_stitch()
                    apply_point_info(stitch[0]["start"], point_info)
                    apply_point_info(stitch[1]["start"], point_info_cor)
                    stitch_points_list = []

                # 结束边的判断条件
                if idx == len(points_info)-1:pass
    pass
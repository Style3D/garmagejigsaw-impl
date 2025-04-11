# 本代码用于计算论文中需要的metric
import os
import concurrent
import json
import time

import torch

from model import build_model
from dataset import build_stylexd_dataloader_train_val
from utils import  to_device, get_pointstitch, stitch_mat2indices # , pointstitch_2_edgestitch2
from _my.cal_metric.utils_cal_metric.ps2es4eval import ps2es4eval
import multiprocessing
from utils import pointcloud_visualize, pointcloud_and_stitch_visualize, pointcloud_and_stitch_logits_visualize, composite_visualize
# from utils.inference.save_result import save_result
import threading
import time
from functools import wraps

def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 定义一个事件来控制超时
            timeout_event = threading.Event()

            # 创建一个线程来运行目标函数
            def run():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
                finally:
                    timeout_event.set()  # 标记函数完成

            result = [None]  # 用列表存储结果，因为线程中变量需要可变对象
            thread = threading.Thread(target=run)
            thread.start()

            # 等待指定的时间
            if not timeout_event.wait(timeout=seconds):
                raise TimeoutError(f"方法 {func.__name__} 运行超过 {seconds} 秒")
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]

        return wrapper

    return decorator

def ps2graph(stitch_indices, piece_id):
    stitch_indices = piece_id[stitch_indices]
    stitch_indices = torch.sort(stitch_indices, dim=1).values
    Graph = torch.unique(stitch_indices, dim=0)
    return Graph

@timeout(10)
def cal_metric_topology(gt_, pred_):
    import networkx as nx
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    G_gt = nx.Graph()
    G_gt.add_edges_from(gt_.tolist())

    G_pred = nx.Graph()
    G_pred.add_edges_from(pred_.tolist())  # 误预测了 (3,4)

    # 获取邻接矩阵
    adj_gt = nx.to_numpy_array(G_gt, nodelist=sorted(G_gt.nodes()))
    adj_pred = nx.to_numpy_array(G_pred, nodelist=sorted(G_gt.nodes()))
    # 计算 Precision、Recall、F1
    acc = accuracy_score(adj_gt.flatten(), adj_pred.flatten())
    precision = precision_score(adj_gt.flatten(), adj_pred.flatten())
    recall = recall_score(adj_gt.flatten(), adj_pred.flatten())
    f1 = f1_score(adj_gt.flatten(), adj_pred.flatten())
    GED = nx.graph_edit_distance(G_gt, G_pred)

    return acc, precision, recall, f1, GED


if __name__ == "__main__":
    from utils.config import cfg
    from utils.parse_args import parse_args

    args = parse_args("Jigsaw")

    model_save_path = cfg.MODEL_SAVE_PATH
    output_path = cfg.OUTPUT_PATH
    point_num = cfg.DATA.NUM_PC_POINTS
    model = build_model(cfg).load_from_checkpoint(cfg.WEIGHT_FILE)
    _, val_loader = build_stylexd_dataloader_train_val(cfg)

    metric_dict = {"PRECISION_CLS":[], "RECALL_CLS":[], "PRECISION_STITCH":[], "RECALL_STITCH":[], "STITCH_AMD":[],
                   "TOPOLOGHY_ACC_PS":[], "TOPOLOGHY_PRECISION_PS":[], "TOPOLOGHY_RECALL_PS":[], "TOPOLOGHY_F1_PS":[], "TOPOLOGHY_GED_PS":[],
                   "TOPOLOGHY_ACC_ES": [], "TOPOLOGHY_PRECISION_ES": [], "TOPOLOGHY_RECALL_ES": [], "TOPOLOGHY_F1_ES": [], "TOPOLOGHY_GED_ES": []}

    for batch in val_loader:
        batch = to_device(batch, model.device)

        try:
            inf_rst = model(batch)
        except Exception as e:
            continue

        B_size, N_point, _ = batch["pcs"].shape
        stitch_mat_ = inf_rst["ds_mat"]
        mat_gt = batch["mat_gt"]
        pc_cls = inf_rst["pc_cls"]
        threshold = cfg.MODEL.PC_CLS_THRESHOLD

        # 计算点分类的metrics ------------------------------------------------------------------------------------
        pc_cls_ = pc_cls.squeeze(-1)
        pc_cls_gt = (torch.sum(mat_gt[:, :N_point] + mat_gt[:, :N_point].transpose(-1, -2),
                               dim=-1) == 1) * 1.0

        indices = pc_cls_ > threshold
        TP_CLS = torch.sum(torch.sum(indices[pc_cls_gt == 1] * 1))
        indices = pc_cls_ > threshold
        FP_CLS = torch.sum(torch.sum((indices[pc_cls_gt == 1] == False) * 1))
        indices = pc_cls_ < threshold
        TN_CLS = torch.sum(torch.sum(indices[pc_cls_gt == 0] * 1))
        indices = pc_cls_ < threshold
        FN_CLS = torch.sum(torch.sum((indices[pc_cls_gt == 0] == False) * 1))
        # === Precision和Recall ===
        PRECISION_CLS = TP_CLS / (TP_CLS + FP_CLS + 1e-13)
        RECALL_CLS = TP_CLS / (TP_CLS + FN_CLS + 1e-13)


        # 计算点缝合的metric ------------------------------------------------------------------------------------
        stitch_mat_full, stitch_indices_full, logits = (
            get_pointstitch(batch, inf_rst,
                            sym_choice="", mat_choice="hun",
                            filter_neighbor_stitch=False, filter_neighbor=3,
                            filter_too_long=False, filter_length=0.08,
                            filter_too_small=True, filter_logits=0.2,
                            only_triu=False, filter_uncontinue=False,
                            show_pc_cls=False, show_stitch=False, export_vis_result=False))


        mat_gt = mat_gt==1
        pcs = batch["pcs"].squeeze(0)
        piece_id = batch["piece_id"][0]
        mat_gt = mat_gt + mat_gt.transpose(-1, -2)
        stitch_mat_full = stitch_mat_full>=0.9


        # === 缝合Recall ===
        RECALL_STITCH = torch.sum(torch.bitwise_and(stitch_mat_full==mat_gt, mat_gt))/torch.sum(mat_gt)

        # === 缝合 AMD (average mean distance) ===
        stitch_indices_pred = stitch_mat2indices(stitch_mat_full)
        stitch_indices_gt = stitch_mat2indices(mat_gt)
        mask_cls_pred = (pc_cls_ > threshold).squeeze(0)
        idx_range = torch.range(0,point_num-1,dtype=torch.int64,device=mat_gt.device)
        stitch_map = torch.zeros((point_num), dtype=torch.int64) - 1
        stitch_map[stitch_indices_pred[:, 0]] = stitch_indices_pred[:, 1]
        stitch_map[stitch_indices_pred[:, 1]] = stitch_indices_pred[:, 0]

        stitch_map_gt = torch.zeros((point_num), dtype=torch.int64) - 1
        stitch_map_gt[stitch_indices_gt[:, 0]] = stitch_indices_gt[:, 1]
        stitch_map_gt[stitch_indices_gt[:, 1]] = stitch_indices_gt[:, 0]

        stitch_point_idx = idx_range[mask_cls_pred]

        stitch_cor_pred = stitch_map[stitch_point_idx]
        stitch_cor_gt = stitch_map_gt[stitch_point_idx]

        stitch_pair_valid_mask = torch.bitwise_and(stitch_cor_pred!=-1, stitch_cor_gt!=-1)
        stitch_point_idx_valid = stitch_point_idx[stitch_pair_valid_mask]

        stitch_cor_pred = stitch_map[stitch_point_idx_valid]
        stitch_cor_gt = stitch_map_gt[stitch_point_idx_valid]

        stitch_cor_position_pred = pcs[stitch_cor_pred]
        stitch_cor_position_gt = pcs[stitch_cor_gt]

        normalize_range = batch['normalize_range'].squeeze(0)
        STITCH_AMD = torch.sum(torch.norm(stitch_cor_position_pred - stitch_cor_position_gt, dim=1)) / len(stitch_point_idx_valid)
        STITCH_AMD *= normalize_range


        # === 计算点缝合拓扑准确性 ===
        timeout = 10
        graph_pred_ps = ps2graph(stitch_indices_pred, piece_id)
        graph_gt = ps2graph(stitch_indices_gt, piece_id)
        try:
            typology_acc_ps, topology_precision_ps, topology_recall_ps, topology_f1_ps, topology_GED_ps = cal_metric_topology(graph_gt, graph_pred_ps)
        except Exception:
            print("TIME OUT")
            continue

        # === 计算处理后的拓扑准确性 ===
        stitch_mat_pred = stitch_mat_full
        graph_pred_es = ps2es4eval(batch, inf_rst, stitch_mat_pred, stitch_indices_full, unstitch_thresh=5, fliter_len=3)
        try:
            typology_acc_es, topology_precision_es, topology_recall_es, topology_f1_es, topology_GED_es = (
                cal_metric_topology(graph_gt, graph_pred_es))
        except Exception:
            print("TIME OUT")
            continue

        # === 将这一轮的结果加入到list中 ===
        metric_dict["PRECISION_CLS"].append(PRECISION_CLS)
        metric_dict["RECALL_CLS"].append(RECALL_CLS)

        metric_dict["RECALL_STITCH"].append(RECALL_STITCH)

        metric_dict["STITCH_AMD"].append(STITCH_AMD)

        metric_dict["TOPOLOGHY_ACC_PS"].append(typology_acc_ps)
        metric_dict["TOPOLOGHY_PRECISION_PS"].append(topology_precision_ps)
        metric_dict["TOPOLOGHY_RECALL_PS"].append(topology_recall_ps)
        metric_dict["TOPOLOGHY_F1_PS"].append(topology_f1_ps)
        metric_dict["TOPOLOGHY_GED_PS"].append(topology_GED_ps)

        metric_dict["TOPOLOGHY_ACC_ES"].append(typology_acc_es)
        metric_dict["TOPOLOGHY_PRECISION_ES"].append(topology_precision_es)
        metric_dict["TOPOLOGHY_RECALL_ES"].append(topology_recall_es)
        metric_dict["TOPOLOGHY_F1_ES"].append(topology_f1_es)
        metric_dict["TOPOLOGHY_GED_ES"].append(topology_GED_es)

        # === 用于保存结果的dict ===
        out_dict = {
            "PRECISION_CLS": torch.mean(torch.Tensor(metric_dict['PRECISION_CLS'])).float().item(),
            "RECALL_CLS": torch.mean(torch.Tensor(metric_dict['RECALL_CLS'])).float().item(),
            "RECALL_STITCH": torch.mean(torch.Tensor(metric_dict["RECALL_STITCH"])).item(),
            "STITCH_AMD": torch.mean(torch.Tensor(metric_dict["STITCH_AMD"])).item(),
            "TOPOLOGHY_ACC_PS": torch.mean(torch.Tensor(metric_dict['TOPOLOGHY_ACC_PS'])).float().item(),
            "TOPOLOGHY_PRECISION_PS": torch.mean(torch.Tensor(metric_dict['TOPOLOGHY_PRECISION_PS'])).float().item(),
            "TOPOLOGHY_RECALL_PS": torch.mean(torch.Tensor(metric_dict['TOPOLOGHY_RECALL_PS'])).float().item(),
            "TOPOLOGHY_F1_PS": torch.mean(torch.Tensor(metric_dict['TOPOLOGHY_F1_PS'])).float().item(),
            "TOPOLOGHY_GED_PS": torch.mean(torch.Tensor(metric_dict['TOPOLOGHY_GED_PS'])).float().item(),
            "TOPOLOGHY_ACC_ES": torch.mean(torch.Tensor(metric_dict['TOPOLOGHY_ACC_ES'])).float().item(),
            "TOPOLOGHY_PRECISION_ES": torch.mean(torch.Tensor(metric_dict['TOPOLOGHY_PRECISION_ES'])).float().item(),
            "TOPOLOGHY_RECALL_ES": torch.mean(torch.Tensor(metric_dict['TOPOLOGHY_RECALL_ES'])).float().item(),
            "TOPOLOGHY_F1_ES": torch.mean(torch.Tensor(metric_dict['TOPOLOGHY_F1_ES'])).float().item(),
            "TOPOLOGHY_GED_ES": torch.mean(torch.Tensor(metric_dict['TOPOLOGHY_GED_ES'])).float().item(),
        }
        with open(os.path.join("/home/Ex1/ProjectFiles/Pycharm_MyPaperWork/Jigsaw_matching/_tmp/metric", 'metric.json'), 'w') as f:
            json.dump(out_dict, f, indent=2)

        print(f"\nPRECISION_CLS: {out_dict['PRECISION_CLS']}\n"
              f"RECALL_CLS: {out_dict['RECALL_CLS']}\n"
              f"RECALL_STITCH: {out_dict['RECALL_STITCH']}\n"
              f"STITCH_AMD: {out_dict['STITCH_AMD']}\n"
              f"TOPOLOGHY_ACC_PS: {out_dict['TOPOLOGHY_ACC_PS']}\n"
              f"TOPOLOGHY_PRECISION_PS: {out_dict['TOPOLOGHY_PRECISION_PS']}\n"
              f"TOPOLOGHY_RECALL_PS: {out_dict['TOPOLOGHY_RECALL_PS']}\n"
              f"TOPOLOGHY_F1_PS: {out_dict['TOPOLOGHY_F1_PS']}\n"
              f"TOPOLOGHY_GED_PS: {out_dict['TOPOLOGHY_GED_PS']}\n"
              f"TOPOLOGHY_ACC_ES: {out_dict['TOPOLOGHY_ACC_ES']}\n"
              f"TOPOLOGHY_PRECISION_ES: {out_dict['TOPOLOGHY_PRECISION_ES']}\n"
              f"TOPOLOGHY_RECALL_ES: {out_dict['TOPOLOGHY_RECALL_ES']}\n"
              f"TOPOLOGHY_F1_ES: {out_dict['TOPOLOGHY_F1_ES']}\n"
              f"TOPOLOGHY_GED_ES: {out_dict['TOPOLOGHY_GED_ES']}\n"
              )

        torch.cuda.empty_cache()
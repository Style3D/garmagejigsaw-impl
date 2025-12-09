"""
GEMINI version
"""

import json
import uuid
import torch
import numpy as np
from copy import deepcopy
from functools import cmp_to_key
from itertools import groupby
from utils import is_contour_OutLine
from collections import defaultdict, deque


def get_random_uuid():
    id = str(uuid.uuid4().hex)
    result = id[0:8] + "-" + id[8:12] + "-" + id[12:16] + "-" + id[16:20] + "-" + id[20:]
    return result


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



# --------------------------
# 辅助：环上距离 / 最小覆盖弧 / outlier 检测
# --------------------------


def is_contour_OutLine(contour_idx, panel_instance_seg):
    """
    Determine whether a contour is a OutLine
    :param contour_idx:
    :param panel_instance_seg:
    :return:
    """
    if contour_idx == 0 or panel_instance_seg[contour_idx] != panel_instance_seg[contour_idx - 1]:
        is_OL = True
        pos = 0
    else:
        is_OL = False
        pos = torch.sum(panel_instance_seg[:contour_idx+1]==panel_instance_seg[contour_idx])-1
    """
    is_OL:  Is outline
    pos:    The index of this contour on the panel (starting from 0)
    """
    return is_OL, pos


def cw_distance(a, b, L):
    return (b - a) % L

def shortest_cyclic_distance(a, b, L):
    d_cw = cw_distance(a, b, L)
    d_ccw = (L - d_cw) if d_cw != 0 else 0
    if d_cw <= d_ccw:
        return int(d_cw), 1
    else:
        return int(d_ccw), -1

def minimal_cyclic_interval(pos_tensor, L):
    """
    pos_tensor: 1D tensor of ints in [0..L-1]
    return: (start_pos, end_pos, length)
    """
    if pos_tensor.numel() == 0:
        return None, None, 0
    pos = torch.unique(pos_tensor).sort().values.tolist()
    if len(pos) == 1:
        return pos[0], pos[0], 1
    gaps = []
    for i in range(len(pos)-1):
        gaps.append((pos[i+1]-pos[i], i))
    gaps.append(((pos[0] + L - pos[-1]), len(pos)-1))
    max_gap, idx = max(gaps, key=lambda x: x[0])
    start_idx = (idx + 1) % len(pos)
    start = pos[start_idx]
    end = pos[idx]
    if end >= start:
        length = end - start + 1
    else:
        length = (L - start) + (end + 1)
    return start, end, int(length)

def detect_outliers_by_nearest_neighbor(pos_tensor, L, outlier_gap_thresh):
    """
    Return boolean mask (torch.bool) same length as pos_tensor
    A point is outlier if min(distance to neighbors on ring) > outlier_gap_thresh
    """
    n = pos_tensor.numel()
    if n <= 2:
        return torch.zeros(n, dtype=torch.bool)
    pos = pos_tensor.clone()
    # sort
    sorted_idx = torch.argsort(pos)
    sorted_pos = pos[sorted_idx].tolist()
    out = torch.zeros(n, dtype=torch.bool)
    # neighbors on ring (circular)
    for k in range(n):
        p = sorted_pos[k]
        left = cw_distance(sorted_pos[(k-1) % n], p, L)
        right = cw_distance(p, sorted_pos[(k+1) % n], L)
        nearest = min(left, right)
        if nearest > outlier_gap_thresh:
            out[sorted_idx[k]] = True
    return out

# --------------------------
# Helper: build garment stitch json pair (compatible with original pipeline)
# returns list of two dicts: [side0_dict, side1_dict]
# each side dict contains keys: 'isCounterClockWise', 'start', 'end'
# 'start'/'end' are dicts with clothPieceId, edgeId, param
# --------------------------
def build_stitch_json_pair(sideA_info, sideB_info):
    """
    sideX_info: dict with fields:
        panel_id, edge_id, start_param (float 0..1), end_param (float 0..1), is_ccw (bool)
    Both start_param/end_param are relative to the edge (0..1).
    """
    def make_side_dict(side):
        return {
            "isCounterClockWise": side["is_ccw"],
            "start": {
                "clothPieceId": side["panel_id"],
                "edgeId": side["edge_id"],
                "param": float(side["start_param"])
            },
            "end": {
                "clothPieceId": side["panel_id"],
                "edgeId": side["edge_id"],
                "param": float(side["end_param"])
            }
        }
    # original code expects a sequence/list/dict indexable; return [sideA, sideB]
    return [make_side_dict(sideA_info), make_side_dict(sideB_info)]

# --------------------------
# 主函数替换体
# --------------------------
def pointstitch_2_edgestitch5(batch, inf_rst, stitch_mat, stitch_indices,
                              max_gap = 2, outlier_gap_thresh = 20):
    """
    Replacement implementation that:
      - reads garment_json, panel_nes, contour_nes, edge_approx, panel_instance_seg via load_data(batch)
      - constructs all_contour_info (index ranges / edge info) similarly to original code
      - groups stitch point pairs into edge-stitch groups using:
            * outlier detection (per-side nearest neighbor gap)
            * adjacency: two stitches adjacent iff both side distances <= max_gap
            * connected components as groups
      - produces garment_json["stitches"] with pairs of side dicts compatible with original pipeline
    Returns: {"garment_json": garment_json}
    """
    # load existing data structure (use your file's load_data)
    garment_json, panel_nes, contour_nes, edge_approx, panel_instance_seg = load_data(batch)

    device_ = stitch_mat.device if hasattr(stitch_mat, "device") else torch.device("cpu")
    pcs = batch["pcs"][0]
    n_pcs = batch["n_pcs"][0]
    contour_num = torch.sum(n_pcs!=0)
    n_pcs = n_pcs[:contour_num]

    # cumulative sums for mapping local->global indices (same logic as original code)
    n_pcs_cumsum = torch.cumsum(n_pcs, dim=-1)
    contour_nes_cumsum = torch.cumsum(contour_nes, dim=-1)

    # build global edge_approx indices (same as original)
    edge_approx_global = deepcopy(edge_approx)
    for contour_idx in range(len(n_pcs_cumsum)):
        if contour_idx == 0:
            edge_start_idx = 0
        else:
            edge_start_idx = contour_nes_cumsum[contour_idx - 1]
        edge_end_idx = contour_nes_cumsum[contour_idx]
        if contour_idx != 0:
            edge_approx_global[edge_start_idx:edge_end_idx] += n_pcs_cumsum[contour_idx-1]

    # build all_contour_info and all_edge_info (similar to original)
    all_contour_info = {}
    all_edge_info = {}
    contourid2panelid = {}
    global_param = 0.0
    very_small_gap = 1e-5

    for contour_idx in range(len(n_pcs_cumsum)):
        panel_instance_idx = panel_instance_seg[contour_idx]
        if contour_idx == 0:
            point_start_idx = 0
            edge_start_idx = 0
        else:
            point_start_idx = n_pcs_cumsum[contour_idx - 1].item()
            edge_start_idx = contour_nes_cumsum[contour_idx - 1].item()
        point_end_idx = n_pcs_cumsum[contour_idx].item()
        edge_end_idx = contour_nes_cumsum[contour_idx].item()

        # original code uses is_contour_OutLine to determine id mapping -- try to reuse it if exists
        try:
            isOL, contour_pos = is_contour_OutLine(contour_idx, panel_instance_seg)
        except Exception:
            # fallback: assume contour is outline and pos = 0
            isOL, contour_pos = True, 0

        panel_json = garment_json["panels"][panel_instance_idx]
        contour_id = panel_json["id"] if isOL else get_random_uuid()

        contour_info = {
            "id": contour_id,
            "index_start": torch.tensor(point_start_idx, dtype=torch.int64, device=device_),
            "index_end": torch.tensor(point_end_idx-1, dtype=torch.int64, device=device_),
            "param_start": global_param,
            "edges_info": {}
        }
        all_contour_info[contour_id] = contour_info
        contourid2panelid[contour_id] = panel_json["id"]

        # get local edges approximation for this contour
        seqEdges_json = panel_json["seqEdges"][contour_pos]
        contour_edges_approx = edge_approx[edge_start_idx:edge_end_idx]
        contour_edges_approx_global = edge_approx_global[edge_start_idx:edge_end_idx]
        contour_edge_num = contour_edges_approx.shape[-2]

        for edge_idx, (e_approx_local, e_approx_global) in enumerate(zip(contour_edges_approx, contour_edges_approx_global)):
            edge_json = seqEdges_json["edges"][edge_idx]
            edge_id = edge_json["id"]
            # assemble edge points indices (handle wrap-around edges)
            if ((edge_idx==0 and e_approx_global[0] > e_approx_global[1]) or
                (edge_idx==contour_edge_num-1 and e_approx_global[0] > e_approx_global[1])):
                # wrap case
                edge_points_idx = torch.concat([
                    torch.arange(int(e_approx_global[0]), point_end_idx, dtype=torch.int64),
                    torch.arange(point_start_idx, int(e_approx_global[1])+1, dtype=torch.int64)
                ])
            else:
                edge_points_idx = torch.arange(int(e_approx_global[0]), int(e_approx_global[1]) + 1, dtype=torch.int64)
            # build points_info for this edge
            edge_info = {"id": edge_id, "contour_id": contour_info["id"], "points_info": [], "param_start": global_param}
            for idx_in_edge, point_idx in enumerate(edge_points_idx.tolist()):
                # compute param along edge (0..1)
                L = len(edge_points_idx)
                param = idx_in_edge / float(max(L-1, 1))
                if idx_in_edge == 0:
                    param += very_small_gap
                elif idx_in_edge == L-1:
                    param -= very_small_gap
                point_info = {
                    "id": int(point_idx),
                    "contour_id": contour_info["id"],
                    "edge_id": edge_id,
                    "param": float(param),
                    "global_param": global_param + float(param),
                    "is_stitch": False
                }
                edge_info["points_info"].append(point_info)
            contour_info["edges_info"][edge_id] = edge_info
            all_edge_info[edge_id] = edge_info
            # update global_param by edge length (approx)
            global_param += 1.0  # keep global_param monotonic; exact scale not critical here

    # ---------- build helper maps: global point idx -> contour_id & pos_in_contour & panel_id ----------
    # we can compute pos_in_contour as (global_idx - contour_index_start) % contour_len
    contour_list = list(all_contour_info.values())
    # build array of ranges for faster lookup: each contour store start,end
    contour_ranges = []
    for c in contour_list:
        start = int(c["index_start"].item())
        end = int(c["index_end"].item())
        length = end - start + 1
        contour_ranges.append((c["id"], start, end, length, contourid2panelid[c["id"]]))

    # helper to find contour & local pos for a global point idx
    def find_contour_for_global_idx(gidx):
        # linear scan (contour count typically not huge)
        for cid, start, end, length, panelid in contour_ranges:
            if start <= gidx <= end:
                pos = (gidx - start) % length
                return cid, pos, length, panelid
        return None, None, None, None

    # ---------- Parse stitch_indices (Mx2 global indices) ----------
    # stitch_indices can be from AIGP model output; ensure it's a LongTensor Nx2
    stitches = stitch_indices.clone().long()
    M = stitches.shape[0]

    # bucket stitches by contour pair (ordered: sideA contour id smaller for deterministic)
    contourpair_buckets = defaultdict(list)
    stitch_meta = []  # will store for each stitch: dict with global indices and contour/pos info

    for s in range(M):
        a = int(stitches[s,0].item())
        b = int(stitches[s,1].item())
        ca, pa, La, panelA = find_contour_for_global_idx(a)
        cb, pb, Lb, panelB = find_contour_for_global_idx(b)
        if ca is None or cb is None:
            # skip unknown points (shouldn't happen normally)
            continue
        # keep original orientation but for grouping we will set sideA as min contour id to avoid duplicate processing
        if ca <= cb:
            key = (ca, cb)
            sideA_global = a; sideB_global = b
            sideA_pos = pa; sideB_pos = pb
            L_sideA = La; L_sideB = Lb
            panelA_id = panelA; panelB_id = panelB
        else:
            key = (cb, ca)
            sideA_global = b; sideB_global = a
            sideA_pos = pb; sideB_pos = pa
            L_sideA = Lb; L_sideB = La
            panelA_id = panelB; panelB_id = panelA
        meta = {
            "globalA": int(sideA_global),
            "globalB": int(sideB_global),
            "posA": int(sideA_pos),
            "posB": int(sideB_pos),
            "L_A": int(L_sideA),
            "L_B": int(L_sideB),
            "panelA": panelA_id,
            "panelB": panelB_id
        }
        idx_in_bucket = len(contourpair_buckets[key])
        contourpair_buckets[key].append(len(stitch_meta))
        stitch_meta.append(meta)

    # ---------- For each contour-pair, run outlier filter + adjacency + connected components ----------
    stitch_edge_groups = []  # list of group dicts
    for (cA, cB), idx_indices in contourpair_buckets.items():
        if len(idx_indices) == 0:
            continue
        # gather arrays
        indices = idx_indices
        posA = torch.tensor([stitch_meta[i]["posA"] for i in indices], dtype=torch.long)
        posB = torch.tensor([stitch_meta[i]["posB"] for i in indices], dtype=torch.long)
        globalA = torch.tensor([stitch_meta[i]["globalA"] for i in indices], dtype=torch.long)
        globalB = torch.tensor([stitch_meta[i]["globalB"] for i in indices], dtype=torch.long)
        panelA_id = stitch_meta[indices[0]]["panelA"]
        panelB_id = stitch_meta[indices[0]]["panelB"]
        L1 = stitch_meta[indices[0]]["L_A"]
        L2 = stitch_meta[indices[0]]["L_B"]

        # detect outliers on each side
        outA_mask = detect_outliers_by_nearest_neighbor(posA, L1, outlier_gap_thresh)
        outB_mask = detect_outliers_by_nearest_neighbor(posB, L2, outlier_gap_thresh)
        is_outlier_mask = outA_mask | outB_mask

        normal_idx = [i for idx_i, i in enumerate(indices) if not is_outlier_mask[idx_i].item()]
        outlier_idx = [i for idx_i, i in enumerate(indices) if is_outlier_mask[idx_i].item()]

        # adjacency among normal set
        groups_local = []

        if len(normal_idx) > 0:
            # reindex local normal entries 0..K-1
            normal_positionsA = torch.tensor([stitch_meta[i]["posA"] for i in normal_idx], dtype=torch.long)
            normal_positionsB = torch.tensor([stitch_meta[i]["posB"] for i in normal_idx], dtype=torch.long)
            normal_globalA = torch.tensor([stitch_meta[i]["globalA"] for i in normal_idx], dtype=torch.long)
            normal_globalB = torch.tensor([stitch_meta[i]["globalB"] for i in normal_idx], dtype=torch.long)

            K = normal_positionsA.shape[0]
            # build adjacency (naive O(K^2) for clarity; K per bucket usually small; can optimize if needed)
            adj = [[] for _ in range(K)]
            for i in range(K):
                for j in range(i+1, K):
                    dA, _ = shortest_cyclic_distance(int(normal_positionsA[i].item()), int(normal_positionsA[j].item()), L1)
                    if dA > max_gap:
                        continue
                    dB, _ = shortest_cyclic_distance(int(normal_positionsB[i].item()), int(normal_positionsB[j].item()), L2)
                    if dB > max_gap:
                        continue
                    adj[i].append(j); adj[j].append(i)
            # connected components
            visited = [False]*K
            for i in range(K):
                if visited[i]:
                    continue
                q = deque([i]); visited[i]=True; comp=[]
                while q:
                    u = q.popleft(); comp.append(u)
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v]=True; q.append(v)
                # comp is indices into normal_* arrays; map back to global indices referencing stitch_meta
                comp_global_indices = [normal_idx[c] for c in comp]
                groups_local.append({
                    "members": comp_global_indices,
                    "is_outlier_group": False
                })

        # each outlier becomes its own group (marked)
        for oi in outlier_idx:
            groups_local.append({
                "members": [oi],
                "is_outlier_group": True
            })

        # For each local group, compute side intervals, direction, and create side info
        for g in groups_local:
            members = g["members"]
            # build arrays for this group
            Apos = torch.tensor([stitch_meta[i]["posA"] for i in members], dtype=torch.long)
            Bpos = torch.tensor([stitch_meta[i]["posB"] for i in members], dtype=torch.long)
            GA = torch.tensor([stitch_meta[i]["globalA"] for i in members], dtype=torch.long)
            GB = torch.tensor([stitch_meta[i]["globalB"] for i in members], dtype=torch.long)
            # minimal covering arcs
            sA, eA, lenA = minimal_cyclic_interval(Apos, L1)
            sB, eB, lenB = minimal_cyclic_interval(Bpos, L2)
            # order group by side A cw from sA
            if sA is not None:
                distsA = [cw_distance(sA, int(p.item()), L1) for p in Apos]
                order = sorted(range(len(distsA)), key=lambda x: distsA[x])
            else:
                order = list(range(len(Apos)))
            GA_ord = GA[order]
            GB_ord = GB[order]
            Apos_ord = Apos[order]
            Bpos_ord = Bpos[order]
            # decide B direction by monotonicity
            if sB is not None and len(Bpos_ord) > 1:
                db = [cw_distance(sB, int(p.item()), L2) for p in Bpos_ord]
                inc = sum(1 for i in range(1,len(db)) if db[i] >= db[i-1])
                dec = (len(db)-1) - inc
                is_B_cw = (inc >= dec)
            else:
                is_B_cw = True
            # build side info structures consistent with original pipeline
            sideA_info = {
                "panel_id": contourid2panelid[cA],
                "edge_id": None,   # attempt to find an edge id that contains start global point if possible
                "start_param": 0.0,
                "end_param": 1.0,
                "is_ccw": False
            }
            sideB_info = {
                "panel_id": contourid2panelid[cB],
                "edge_id": None,
                "start_param": 0.0,
                "end_param": 1.0,
                "is_ccw": not is_B_cw  # note: original 'isCounterClockWise' bool intent
            }
            # Try to infer edge_id and start/end param from all_edge_info if possible:
            # search for an edge in contour cA whose points_info include GA_ord[0] and GA_ord[-1] similarly for cB
            def infer_edge_and_params(global_indices, contour_id):
                # return (edge_id, start_param, end_param) if found else (None, 0,1)
                edge_map = all_contour_info[contour_id]["edges_info"]
                gstart = int(global_indices[0].item())
                gend = int(global_indices[-1].item())
                for eid, einfo in edge_map.items():
                    pts = [p["id"] for p in einfo["points_info"]]
                    if gstart in pts and gend in pts:
                        # compute params by locating indices within edge points and linearly mapping to [0,1]
                        i0 = pts.index(gstart)
                        i1 = pts.index(gend)
                        Lp = len(pts)
                        if Lp==1:
                            p0 = 0.0; p1 = 1.0
                        else:
                            p0 = i0/(Lp-1)
                            p1 = i1/(Lp-1)
                        return eid, float(p0), float(p1)
                return None, 0.0, 1.0

            eidA, p0A, p1A = infer_edge_and_params(GA_ord.tolist(), cA)
            eidB, p0B, p1B = infer_edge_and_params(GB_ord.tolist(), cB)
            if eidA is not None:
                sideA_info["edge_id"] = eidA
                sideA_info["start_param"] = p0A
                sideA_info["end_param"] = p1A
            if eidB is not None:
                sideB_info["edge_id"] = eidB
                sideB_info["start_param"] = p0B
                sideB_info["end_param"] = p1B

            # build final group dict
            group_record = {
                "ring_a": cA,
                "ring_b": cB,
                "stitch_pairs": torch.stack([GA_ord, GB_ord], dim=1),  # G x 2 global indices tensor
                "side_a": {
                    "ring": cA,
                    "positions": Apos_ord,
                    "start_pos": sA,
                    "end_pos": eA,
                    "arc_length": lenA,
                    "is_clockwise": True,
                    "panel_id": sideA_info["panel_id"],
                    "edge_id": sideA_info["edge_id"],
                    "start_param": sideA_info["start_param"],
                    "end_param": sideA_info["end_param"]
                },
                "side_b": {
                    "ring": cB,
                    "positions": Bpos_ord,
                    "start_pos": sB,
                    "end_pos": eB,
                    "arc_length": lenB,
                    "is_clockwise": is_B_cw,
                    "panel_id": sideB_info["panel_id"],
                    "edge_id": sideB_info["edge_id"],
                    "start_param": sideB_info["start_param"],
                    "end_param": sideB_info["end_param"]
                },
                "is_outlier_group": g["is_outlier_group"]
            }
            stitch_edge_groups.append(group_record)

    # ---------- Convert groups to garment_json["stitches"] format ----------
    stitch_json_list = []
    for g in stitch_edge_groups:
        # each group corresponds to one stitch segment between side_a and side_b
        sideA = {
            "panel_id": g["side_a"]["panel_id"],
            "edge_id": g["side_a"]["edge_id"] if g["side_a"]["edge_id"] is not None else "",
            "start_param": g["side_a"]["start_param"],
            "end_param": g["side_a"]["end_param"],
            "is_ccw": not g["side_a"]["is_clockwise"]  # align semantics: isCounterClockWise
        }
        sideB = {
            "panel_id": g["side_b"]["panel_id"],
            "edge_id": g["side_b"]["edge_id"] if g["side_b"]["edge_id"] is not None else "",
            "start_param": g["side_b"]["start_param"],
            "end_param": g["side_b"]["end_param"],
            "is_ccw": not g["side_b"]["is_clockwise"]
        }
        # build pair of dicts compatible with original pipeline
        pair = build_stitch_json_pair(
            {
                "panel_id": sideA["panel_id"],
                "edge_id": sideA["edge_id"],
                "start_param": sideA["start_param"],
                "end_param": sideA["end_param"],
                "is_ccw": sideA["is_ccw"]
            },
            {
                "panel_id": sideB["panel_id"],
                "edge_id": sideB["edge_id"],
                "start_param": sideB["start_param"],
                "end_param": sideB["end_param"],
                "is_ccw": sideB["is_ccw"]
            }
        )
        stitch_json_list.append(pair)

    # assign into garment_json, preserving other fields
    garment_json["stitches"] = stitch_json_list

    results = {"garment_json": garment_json}
    return results

import json
import math
import os
import random
from glob import glob

import numpy as np
import torch
# [todo] 将新的加噪方式（BBOX加噪）做出来
from diffusers import DDPMScheduler

import igl
import trimesh
import trimesh.sample

from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import convolve1d

from utils import get_sphere_noise, min_max_normalize, styleXD_normalize, get_pc_bbox, compute_adjacency_list
from utils import meshes_visualize, stitch_visualize, pointcloud_visualize, pointcloud_and_stitch_visualize
from utils import LatinHypercubeSample, random_point_in_convex_hull
from utils import stitch_mat2indices, stitch_indices2mat, stitch_indices_order, stitch_mat_order, cal_mean_edge_len

from scipy.stats.qmc import LatinHypercube

class AllPieceMatchingDataset_stylexd(Dataset):
    """Geometry part assembly dataset, with fracture surface information.

    We follow the data prepared by Breaking Bad dataset:
        https://breaking-bad-dataset.github.io/
    """

    def __init__(
            self,
            data_dir,
            data_keys,
            data_types = (),
            category="all",
            num_points=1000,
            min_num_part=2,
            max_num_part=20,
            shuffle_parts=False,

            shrink_mesh=False,
            shrink_mesh_param=1,

            panel_noise_type="default",
            trans_range=1,
            rot_range=-1,
            scale_range=0,

            overfit=10,

            pcs_sample_type="area",
            pcs_noise_type="default",
            pcs_noise_strength=1.,
            use_stitch_noise = False,  # 是否沿着缝合线添加noise
            stitch_noise_strength = 1,
            stitch_noise_random_range = (0.8, 2.2),
            read_uv = False,

            min_part_point=30,
            mode= "train",
    ):
        if mode not in ["train", "val", "test", "inference"]:
            raise ValueError(f"mode=\"{mode}\" is not valid.")

        self.mode = mode
        self.data_dir = data_dir
        self.data_types = data_types

        self.data_list = self._read_data()
        # self.data_list = self.data_list[::200]

        try:
            with open(os.path.join(data_dir,self.mode,"data_info.json"), "r", encoding="utf-8") as f:
                self.data_info = json.load(f)
        except:
            self.data_info = None

        self.category = category if category.lower() != "all" else ""

        self.num_points = num_points

        self.min_num_part = min_num_part
        self.max_num_part = max_num_part  # ignore shapes with more parts
        self.min_part_point = min_part_point  # ensure that each piece has at least # points
        # self.shuffle_parts = shuffle_parts  # shuffle part orders

        self.panel_noise_type = panel_noise_type
        self.scale_range = scale_range
        self.rot_range = rot_range  # rotation range in degree
        self.trans_range = trans_range  # translation range
        """
        stitch（only training）：根据缝合关系按比例采样
        boundary_mesh：在mesh的边缘点上采样
        boundary_pcs：直接在采样出的边缘点上采样
        """
        if pcs_sample_type not in [ "stitch", "boundary_mesh", "boundary_pcs"]:
            raise ValueError(f"pcs_sample_type=\"{pcs_sample_type}\" is not valid.")
        self.pcs_sample_type = pcs_sample_type

        self.shrink_mesh=shrink_mesh
        self.shrink_mesh_param=shrink_mesh_param

        self.pcs_noise_strength = pcs_noise_strength
        if pcs_noise_type not in ["default", "normal"]:
            raise ValueError(f"pcs_noise_type=\"{pcs_noise_type}\" is not valid.")
        self.pcs_noise_type = pcs_noise_type
        self.use_stitch_noise = use_stitch_noise
        self.stitch_noise_strength = stitch_noise_strength
        self.stitch_noise_random_min = min(stitch_noise_random_range)
        self.stitch_noise_random_max = max(stitch_noise_random_range)

        print("dataset length: ", len(self.data_list))

        # additional data to load, e.g. ('part_ids', 'instance_label')
        self.data_keys = data_keys

        # overfit = 10

        if overfit > 0:
            self.data_list = self.data_list[:overfit]

        self.length = len(self.data_list)

        self.read_uv = read_uv


    def __len__(self):
        return self.length

    def _read_data(self):
        if self.mode in ["train", "val", "test"]:
            with open(os.path.join(self.data_dir, "dataset_split", f"{self.mode}.json") ,"r", encoding="utf-8") as f:
                split = json.load(f)
            mesh_dir = os.path.join(self.data_dir, self.mode)
            data_list = [os.path.join(mesh_dir, dir_) for dir_ in split]
            return data_list
        # 根据数据类型读取不同的数据去inference
        if self.mode == "inference":
            assert len(self.data_types)!=0, "self.data_types can't be empty in inference."
            self.data_types = list(dict.fromkeys(self.data_types))  # 去除重复
            data_list = []
            for type_name in self.data_types:
                mesh_dir = os.path.join(self.data_dir, self.mode)
                mesh_dir = os.path.join(mesh_dir, type_name)
                assert os.path.exists(mesh_dir), f"No data folder corresponding to data_types:{self.data_types}"
                data_list.extend(sorted(glob(os.path.join(mesh_dir, "garment_*"))))
            return data_list        # 根据数据类型读取不同的数据去inference
        # if self.mode == "inference":
        #     assert len(self.data_types)!=0, "self.data_types can't be empty in inference."
        #     self.data_types = list(dict.fromkeys(self.data_types))  # 去除重复
        #     data_list = []
        #     for type_name in self.data_types:
        #         mesh_dir = os.path.join(self.data_dir, self.mode)
        #         mesh_dir = os.path.join(mesh_dir, type_name)
        #         assert os.path.exists(mesh_dir), f"No data folder corresponding to data_types:{self.data_types}"
        #         data_list.extend(sorted(glob(os.path.join(mesh_dir, "garment_*"))))
        #     return data_list


    # 用于训练inference董远数据的模型
    # random scale+rotate+move panel (default)
    def _random_SRM_default(self, pc, pcs, pc_idx, mean_edge_len):
        """
        pc: [N, 3]
        """
        noise_strength_S = random.random()*0.5 + 0.2
        noise_strength_R = random.random()*0.5 + 0.5
        noise_strength_M = random.random()*0.6 + 0.8

        pc_centroid = get_pc_bbox(pc)[0]
        # SCALE ------------------------------
        pc = pc - pc_centroid[None]
        # [todo]0.95改回去1
        # scale_gt = (np.random.rand(1) * self.scale_range) + 1.0
        scale_gt = (np.random.rand(1) * self.scale_range) * noise_strength_S + 1
        pc = pc*scale_gt

        # ROTATE ------------------------------
        if self.rot_range >= 0.0:
            rot_euler = (np.random.rand(3) - 0.5) * 2.0 * self.rot_range
            rot_mat = R.from_euler("xyz", rot_euler, degrees=True).as_matrix()
        else:
            rot_mat = R.random().as_matrix()
        pc = (rot_mat @ pc.T).T
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        # we use scalar-first quaternion
        quat_gt = quat_gt[[3, 0, 1, 2]] * noise_strength_R
        pc = pc + pc_centroid[None]

        # MOVE ------------------------------
        bkg_pcs = np.concatenate(pcs[:pc_idx]+pcs[pc_idx+1:])
        pc_centroid = get_pc_bbox(pc)[0]
        bkg_pcs_centroid = get_pc_bbox(bkg_pcs)[0]

        move_vec = pc_centroid-bkg_pcs_centroid
        move_vec = move_vec/np.linalg.norm(move_vec)
        # 水平投影，引导Panel在水平上向外移动
        move_vec_horizon = np.zeros_like(move_vec)
        move_vec_horizon[:] = move_vec[:]
        move_vec_horizon[1] = move_vec_horizon[1]/2
        move_vec_horizon = move_vec_horizon/np.linalg.norm(move_vec_horizon)

        move_vec = move_vec + move_vec_horizon
        move_vec = move_vec / np.linalg.norm(move_vec)

        trans_gt = (mean_edge_len * move_vec * (np.random.rand(3) + 1.) / 4. * self.trans_range).reshape(-1, 3)
        pc = pc + trans_gt

        # trans_gt = (mean_edge_len * (np.random.rand(3) - 0.5) * 2.0 * self.trans_range).reshape(-1,3) * noise_strength_M
        # pc = pc + trans_gt
        return pc, trans_gt, quat_gt

    # random scale+move panel by add noise on panel bbox
    def _random_SM_byBbox(self, pc):
        # [todo] 给BBOX随机加噪
        surfPos = get_pc_bbox(pc, type="xyxy")
        surfPos = torch.Tensor(np.concatenate(surfPos), device="cpu").unsqueeze(0)
        aug_ts = torch.randint(0, 15, (1,), device="cpu").long()
        # aug_noise = torch.randn(surfPos.shape,device="cpu")
        # surfPos = self.noise_scheduler.add_noise(surfPos, aug_noise, aug_ts)
        # [todo] 让部件填入加噪后的BBOX，返回

        raise NotImplementedError

    def _shuffle_pc(self, pc, pc_gt):
        """pc: [N, 3]"""
        order_indices = np.arange(pc.shape[0])
        random.shuffle(order_indices)
        pc = pc[order_indices]
        pc_gt = pc_gt[order_indices]
        return pc, pc_gt, order_indices

    def _pad_data(self, data, pad_size=None):
        """Pad data to shape [`self.max_num_part`, data.shape[1], ...]."""
        if pad_size is None:
            pad_size = self.max_num_part
        data = np.array(data)
        if len(data.shape) > 1:
            pad_shape = (pad_size,) + tuple(data.shape[1:])
        else:
            pad_shape = (pad_size,)
        pad_data = np.zeros(pad_shape, dtype=data.dtype)
        pad_data[: data.shape[0]] = data
        return pad_data

    def sample_point_byBoundaryMesh(self, meshes, num_points):
        pcs, piece_id, nps = [], [], [len(mesh.vertices) for mesh in meshes]
        sample_fid = []  # 采样点的faceid

        # 所有的顶点、面、边界点
        vertices = np.concatenate([np.array(mesh.vertices) for mesh in meshes], axis=0)
        faces = []
        boundary_point_list = []
        adjacency_list = []

        # PART 1 -------------------------------------------------------------------------------------------------------
        # 因为读取的mesh被划分为多个，因此，面中index和边界点index都需要转换为全局index
        points_end_idxs = np.cumsum(nps)
        n_loop_pts = []  # 每个loop的顶点数
        for idx, mesh in enumerate(meshes):
            if idx == 0:
                point_start_idx = 0
            else:
                point_start_idx = points_end_idxs[idx - 1]
            # point_end_idx = points_end_idxs[idx]
            # 获取边界点的全局index
            for loop in igl.all_boundary_loop(mesh.faces):
                loop = np.array(loop) + point_start_idx
                boundary_point_list.append(loop)
                boundary_adjacency = np.zeros((len(loop), 4), dtype=np.int32)
                if len(loop) > 4:
                    boundary_adjacency[:, 0] = np.concatenate([loop[2:], loop[:2]],axis=-1)
                    boundary_adjacency[:, 1] = np.concatenate([loop[1:], loop[:1]],axis=-1)
                    boundary_adjacency[:, 2] = np.concatenate([loop[-1:], loop[:-1]],axis=-1)
                    boundary_adjacency[:, 3] = np.concatenate([loop[-2:], loop[:-2]],axis=-1)
                else:
                    boundary_adjacency[:, :] = loop
                n_loop_pts.append(len(loop))
                adjacency_list.append(boundary_adjacency)

            # 获取面的全局index
            faces.append(np.array(mesh.faces) + point_start_idx)
        faces = np.concatenate(faces, axis=0)
        boundary_points_idx = np.concatenate(boundary_point_list, axis=0)
        adjacency_list = np.concatenate(adjacency_list, axis=-2)
        # pointcloud_visualize(vertices[boundary_points_idx])

        boundary_points = vertices[boundary_points_idx]
        piece_id = np.concatenate([np.array([idx] * n) for idx, n in enumerate(nps)], axis=0)[boundary_points_idx]
        points_normalized, normalize_range = min_max_normalize(boundary_points)
        mean_edge_len = cal_mean_edge_len(meshes) / normalize_range
        # pointcloud_visualize(points_normalized)
        for i in range(len(nps)):
            pcs.append(points_normalized[piece_id == i])
        nps = np.array([len(pc) for pc in pcs])

        sample_result =dict(
            pcs=pcs,
            piece_id=piece_id,
            nps=nps,
            mat_gt=None,
            mean_edge_len=mean_edge_len,
            normalize_range=normalize_range
        )
        return sample_result

    def sample_point_byBoundaryPcs(self, meshes, num_points):
        pcs = [np.array(mesh.vertices) for mesh in meshes]
        num_parts = len(pcs)
        piece_id = np.concatenate([[idx]*len(pcs[idx]) for idx in range(len(pcs))], axis=0)
        # pcs_normalized, normalize_range = styleXD_normalize(np.concatenate(pcs))
        # pcs_normalized, normalize_range = np.concatenate(pcs), 1.
        pcs_normalized, normalize_range = min_max_normalize(np.concatenate(pcs))
        pcs = [pcs_normalized[piece_id==panel_idx] for panel_idx in range(num_parts)]
        nps = np.array([len(pc) for pc in pcs])
        mean_edge_len = 0.0072/normalize_range
        sample_result =dict(
            pcs=pcs,
            piece_id=piece_id,
            nps=nps,
            mat_gt=None,
            mean_edge_len=mean_edge_len,
            normalize_range=normalize_range
        )
        return sample_result


    def sample_point_byStitch(self, meshes, num_points, stitches, full_uv_info, n_rings = 2, max_check_times=4, min_samplenum_prepanel=4):
        """
        根据缝合点和不缝合点的数量，按比例进行对这两种点进行分别采样

        :param meshes:          List of trimesh.Trimesh
        :param num_points:      Target point sample num
        :param stitches:        Stitch relation in original mesh
        :param full_uv_info:    UV info in original mesh
        :param n_rings:         Expand rings in getting sampled_points_CH
        :param max_check_times:         Max re-sample times
        :param min_samplenum_prepanel:  Min point sample number in each Panel
        :return:
        """

        pcs = []
        nps =  [len(mesh.vertices) for mesh in meshes]      # 每个 Panel 的点的数量
        num_parts = len(nps)    # Panel 数量
        piece_id_global = np.concatenate([[idx]*i for idx,i in enumerate(nps)], axis=-1)    # 每个点顶点的 piece id



        # PART 1 -------------------------------------------------------------------------------------------------------
        # 获取 所有顶点、所有面(面中的index转全局index，因为一个衣服包含很多个mesh)、所有的边界点，分别保存在 vertices、faces、boundary_point_list 中

        vertices = np.concatenate([np.array(mesh.vertices) for mesh in meshes], axis=0)     # 所有的 顶点
        faces = []      # 所有的 面
        boundary_point_list = []     # 所有的 边界点

        points_end_idxs = np.cumsum(nps)
        for idx, mesh in enumerate(meshes):
            if idx ==0 : point_start_idx = 0
            else: point_start_idx = points_end_idxs[idx-1]
            # === 获取边界点的全局index ===
            new_list = []  # 一个Panel可能包含多个contours
            for loop in igl.all_boundary_loop(mesh.faces):
                new_list.extend(np.array(loop)+point_start_idx)
            boundary_point_list.append(new_list)
            # === 获取面的全局index ===
            faces.append(np.array(mesh.faces)+point_start_idx)
        faces = np.concatenate(faces,axis=0)
        boundary_points_idx = np.concatenate(boundary_point_list,axis=0)

        # PART 2 -------------------------------------------------------------------------------------------------------
        # 按比例对具有缝合关系的点 和 不具有缝合关系的 边界点 进行采样

        # === 获取点点缝合关系的映射 ===
        stitch_map = np.zeros(shape=(len(vertices)), dtype=np.int32) - 1
        stitch_map[stitches[:, 0]] = stitches[:, 1]
        stitch_map[stitches[:, 1]] = stitches[:, 0]

        # === 获取所有的两类点的idx ===
        # 所有具有缝合关系的点的index
        stitched_vertices_idx = np.concatenate([stitches[:, 0], stitches[:, 1]], axis=0)
        # 所有不具有缝合关系的边界点的index
        unstitched_vertices_idx = boundary_points_idx[
            np.array([s not in stitched_vertices_idx for s in boundary_points_idx])]

        # === 分配 缝合点\不缝合点 的采样数量 ===
        # 在具有缝合关系的点上进行采样的数量
        stitched_sample_num = int((num_points * (len(stitched_vertices_idx) / (len(stitched_vertices_idx) + len(unstitched_vertices_idx)))) / 2)
        # 在不具有缝合关系的边界点上进行采样的数量
        unstitched_sample_num = num_points - stitched_sample_num * 2

        # === 计算每个Panel的采样点数量期望值 ===
        boundary_len = sum([len(b) for b in boundary_point_list])
        expect_sample_nums = np.array([int(num_points*len(b)/boundary_len) for b in boundary_point_list])

        # === 对于期望值过低的Panel，我们提前分配一部分采样点 ===
        sample_num_arrangement = np.zeros(num_parts, dtype=np.int16)
        """
        为什么用 int(min_samplenum_prepanel * 2) 作为单个Panel预设的最小采样点数量 ?
            * 2.5 是因为后面分配每个 Panel 的缝合点采样数量会先除以2
            假如一个很小的 Panel 没有不缝合点，如果在这里仅分配了 min_samplenum_prepanel 个点，则只会在这个Panel上采样 min_samplenum_prepanel/2 个点
            这大概率会导致在这个 Panel 上在最终采样的点的数量达不到 min_samplenum_prepanel 这个数量
        因此：
            min_expect = min_samplenum_prepanel * X>2 ，是为了尽可能的为更多可能出问题的（也就是最后采样数量可能太少的）Panel 提前分配一些采样点数量
            pre_arranged = int(min_samplenum_prepanel * X>2)，是为了能百分之百保证采样一次就能达到每个 Panel 的采样数量都大于 min_samplenum_prepanel
        由于 min_expect 不能设的过大（会导致给太多的Panel预分配采样点），因此依旧可能有Panel采样点数量过少，因此还是需要 check 最多 max_check_times 次
        """
        min_expect = min_samplenum_prepanel * 4             # 采样点的期望值低于 min_expect 的会被预分配采样点数量
        pre_arranged = int(min_samplenum_prepanel * 2.5)    # 预分配的采样点数量
        assert pre_arranged * num_parts < num_points, "min_samplenum_prepanel too large may cause error"
        sample_num_arrangement[expect_sample_nums <= min_expect] = pre_arranged
        sample_num_arrangement[expect_sample_nums <= min_expect] = int(pre_arranged/2)  # 其它的也得在这时分配一点，防止采样点数量太不平衡
        # 提前分配的采样点
        arranged_num = np.sum(sample_num_arrangement)

        # === 获取每个 Panel 的采样点总数 ===
        sample_num_arrangement = np.array([int((num_points-arranged_num)*len(b)/boundary_len) for b in boundary_point_list]) + sample_num_arrangement
        # 分配没分完的采样点
        for i in range(num_points - np.sum(sample_num_arrangement)):
            sample_num_arrangement[random.randint(0, num_parts - 1)]+=1

        # === 获取每个panel上的 缝合点 和 被缝合点 的数量 ===
        panel_stitched_num = [sum(piece_id_global[stitched_vertices_idx]==i) for i in range(num_parts)]
        panel_unstitched_num = [sum(piece_id_global[unstitched_vertices_idx]==i) for i in range(num_parts)]
        # 每个Panel上分配的 缝合点 的采样数量
        sample_num_arrangement_stitched = np.array([int((sample_num_arrangement[i] * panel_stitched_num[i] / (panel_stitched_num[i]+panel_unstitched_num[i]))/2) for i in range(num_parts)])
        # 每个Panel上分配的 不缝合点 的采样数量
        sample_num_arrangement_unstitched = np.array([sample_num_arrangement[i]-sample_num_arrangement_stitched[i]*2 for i in range(num_parts)])

        # === 获取每个Panel上的 缝合点 和 不缝合点===
        stitched_list, unstitched_list = [], []
        for part_idx in range(num_parts):
            mask = piece_id_global[stitched_vertices_idx] == part_idx
            stitched_list.append(stitched_vertices_idx[mask])
            mask = piece_id_global[unstitched_vertices_idx] == part_idx
            unstitched_list.append(unstitched_vertices_idx[mask])

        # === 修正 每个Panel上的 缝合点\不缝合点 的分配 ===
        unarranged = 0  # 多余未分配的点数量，在不缝合点转移到缝合点，且不缝合点数量为奇数时，会产生一个offset
        # 没有不缝合点的panel，将分配的不缝合点数转移到缝合点
        for i in range(num_parts):
            if len(unstitched_list[i])==0:
                transfer_volumn = math.floor((sample_num_arrangement_unstitched[i] + unarranged)/2)
                unarranged = (sample_num_arrangement_unstitched[i] + unarranged)%2
                sample_num_arrangement_stitched[i] += transfer_volumn
                sample_num_arrangement_unstitched[i] = 0
        # 没有缝合点的panel，将分配的缝合点数转移到不缝合点
        for i in range(num_parts):
            if len(stitched_list[i])==0:
                transfer_volumn = sample_num_arrangement_stitched[i]*2
                if unarranged>0:
                    transfer_volumn+=unarranged
                    unarranged=0
                sample_num_arrangement_unstitched[i] += transfer_volumn
                sample_num_arrangement_stitched[i] = 0
        # 如果依旧有未分配的点，找到分配不缝合点大于0且分配数量最少的Panel，将这个点分配给它
        if unarranged>0:
            min_idx, min_arranged = 0, 9999999
            for i in range(num_parts):
                if min_arranged > sample_num_arrangement_unstitched[i] > 0:
                    min_arranged = sample_num_arrangement_unstitched[i]
                    min_idx = i
            sample_num_arrangement_unstitched[min_idx]+=unarranged
            unaranged = 0

        # PART 3 -------------------------------------------------------------------------------------------------------
        # 根据上一部分计算的 采样点数量分配，来进行采样。可进行多轮采样
        for check_time in range(max_check_times):
            # === 成对采样具有缝合关系的点 ===
            sample_list = []
            for part_idx in range(num_parts):
                # 一个panel上的缝合点
                part_points = stitched_list[part_idx]
                # 采样结果加入list
                if sample_num_arrangement_stitched[part_idx]>0:
                    sample_list.append(part_points[LatinHypercubeSample(len(part_points), sample_num_arrangement_stitched[part_idx])])
            stitched_sample_idx = np.concatenate(sample_list)
            stitched_sample_idx = sorted(stitched_sample_idx)
            stitched_sample_idx_cor = stitch_map[stitched_sample_idx]

            # === 采样不具有缝合关系的边界点 ===
            sample_list = []
            for part_idx in range(num_parts):
                # 一个panel上的缝合点
                part_points = unstitched_list[part_idx]
                # 采样结果加入list
                if sample_num_arrangement_unstitched[part_idx]>0:
                    sample_list.append(part_points[LatinHypercubeSample(len(part_points), sample_num_arrangement_unstitched[part_idx])])

            if len(sample_list) > 0:
                unstitched_sample_idx = np.concatenate(sample_list)
            else:
                unstitched_sample_idx = None

            # === 获得采样结果 ===
            # 所有采样点的index
            if unstitched_sample_idx is not None:
                all_sample_idx = np.concatenate([stitched_sample_idx, stitched_sample_idx_cor, unstitched_sample_idx], axis=0)
            else:
                all_sample_idx = np.concatenate([stitched_sample_idx, stitched_sample_idx_cor], axis=0)

            # 检查是否满足条件：每个Panel的采样点总数都大于等于min_samplenum_prepanel
            sampled_piece_id = piece_id_global[all_sample_idx]
            piece_sample_num = np.array([np.sum(sampled_piece_id==i) for i in range(num_parts)])
            if np.sum(piece_sample_num<min_samplenum_prepanel)==0:
                break
            else:
                if check_time!=max_check_times-1:
                    continue
                else:
                    raise ValueError(f"sample num:{np.sum(piece_sample_num<min_samplenum_prepanel)}, NUM_PC_POINTS too small in cfg")  # 如果超过最大调用次数

        # 在具有缝合关系的点上进行采样的数量
        stitched_sample_num = np.sum(sample_num_arrangement_stitched)
        # 采样点中的缝合关系
        mat_gt = np.array([np.array([i, i + stitched_sample_num]) for i in range(stitched_sample_num)])

        # 每个采样点属于的part
        piece_id = np.concatenate([np.array([idx]*n) for idx, n in enumerate(nps)],axis=0)[all_sample_idx]
        # stitch_visualize(np.concatenate([stitched_sample,stitched_sample_cor,unstitched_sample],axis=0),mat_gt)

        # PART 4 -------------------------------------------------------------------------------------------------------
        # 对每个边界点，进行凸包集采样

        # # === 获取每个采样点的相邻点集 (集合中包括自身)===
        # # 获得点与点邻居关系的邻接矩阵
        # adjacency_list = np.array(compute_adjacency_list(faces, len(vertices)), dtype=object)
        # # 获取每个采样点的邻居顶点的index的集合
        # all_sample_neighbor_idx = adjacency_list[all_sample_idx]
        # # 将每个采样点本身也加入这个集合，这个集合对应的顶点可以构成一个凸包集
        # all_sample_neighbor_idx = [np.array(n+[i]) for n,i in zip(all_sample_neighbor_idx, all_sample_idx)]
        # all_sample_neighbor_points = [vertices[n] for n in all_sample_neighbor_idx]
        # # 在每个凸包集内随机采样一个顶点
        # sampled_points_CH = np.array([random_point_in_convex_hull(elem) for elem in all_sample_neighbor_points])
        # # pointcloud_visualize([sampled_points_CH[0:stitched_sample_num*2],sampled_points_CH[stitched_sample_num*2:]])
        #
        # # === 对采样结果进行扩张 ===
        # all_sample_points = vertices[all_sample_idx]
        # V_P = sampled_points_CH - all_sample_points
        # sampled_points_CH = all_sample_points + n_rings * V_P * self.pcs_noise_strength
        # # pointcloud_visualize([sampled_points_CH[0:stitched_sample_num*2],sampled_points_CH[stitched_sample_num*2:]])
        # # pointcloud_and_stitch_visualize(sampled_points_CH,mat_gt[:100])

        # [todo] 可能存在冗余代码，记得清理下
        sampled_points_CH =  vertices[all_sample_idx]

        # PART 5 -------------------------------------------------------------------------------------------------------
        # 将点云大小按最BBOX的最长边归一化到[-0.5 - 0.5]，计算平均边长

        # points_normalized, normalize_range = min_max_normalize(sampled_points_CH)
        points_normalized, normalize_range = styleXD_normalize(sampled_points_CH)
        mean_edge_len = cal_mean_edge_len(meshes)/normalize_range
        # pointcloud_visualize(points_normalized)

        # PART 6 -------------------------------------------------------------------------------------------------------
        # 让采样的缝合关系满足：在index上接近的缝合关系，在位置上也尽可能接近

        # === 将采样点按index进行排序 ===，这样同一个part的会凑到一块去
        sorted_indices = np.argsort(all_sample_idx)  # 创建排序索引
        points_normalized = points_normalized[sorted_indices]
        piece_id = piece_id[sorted_indices]

        # 采样边缘部分的UV，并排序
        if full_uv_info is not None:
            uv_sampled = full_uv_info[all_sample_idx]
            uv_sampled = uv_sampled[sorted_indices]

        # 对缝合关系对应的两端顶点的index应用相同的排序，然后按起始点index进行降序排序
        mat_gt = stitch_indices_order(mat_gt, sorted_indices)

        # 如果一个缝合关系末端顶点的index 大于 起始顶点的index，则让它们交换
        mask = mat_gt[:, 0] > mat_gt[:, 1]
        mat_gt[mask] = mat_gt[mask][:, ::-1]

        mat_gt = mat_gt[np.argsort(mat_gt[:, 0])]
        # pointcloud_and_stitch_visualize(points_normalized, mat_gt)

        # === 将归一化后的采样点按Panel区分 ===
        for i in range(len(nps)):
            pcs.append(points_normalized[piece_id==i])
        nps = np.array([len(pc) for pc in pcs])

        if(sum([len(pc) for pc in pcs])!=num_points):
            raise AssertionError("Pointcloud Sample Num Wrong.")

        pcs_dict = dict(
            pcs=pcs,
            piece_id=piece_id,
            nps=np.array(nps),
            mat_gt=mat_gt,
            mean_edge_len=mean_edge_len,
            normalize_range=normalize_range,
        )
        if full_uv_info is not None:
            pcs_dict["uv"] = uv_sampled
        return pcs_dict

    # 在mesh的顶点上添加噪声(暂不使用)
    def add_noise_on_mesh(self, meshes, noise_strength=2):
        for mesh in meshes:
            # 计算平均边长
            edges = np.array(mesh.edges)
            v_el = np.array(mesh.vertices)[np.concatenate(edges,axis=0)[0::2]]
            v_er = np.array(mesh.vertices)[np.concatenate(edges,axis=0)[1::2]]
            mean_edge_len = np.mean(np.sqrt(np.sum((v_el - v_er) ** 2, axis=1)))

            # 噪声的基数
            noise_base = noise_strength * mean_edge_len

            v_n =  mesh.vertex_normals
            V1 = np.zeros_like(v_n)
            V2 = np.zeros_like(v_n)

            for i in range(v_n.shape[0]):
                vec = v_n[i]

                # Normalize the input vector
                vec = vec / np.linalg.norm(vec)

                # Find the first orthogonal vector
                if vec[0] != 0 or vec[1] != 0:
                    V1[i] = np.array([-vec[1], vec[0], 0])
                else:
                    V1[i] = np.array([1, 0, 0])

                # Find the second orthogonal vector using cross product
                V2[i] = np.cross(vec, V1[i])

            V1 = np.array(V1) / np.linalg.norm(V1)
            V2 = np.array(V2) / np.linalg.norm(V2)

            # Initialize an array to store points
            points_offset = np.zeros((V1.shape[0], 3))

            angle = np.random.uniform(0, 2 * np.pi, V1.shape[0])
            r = np.sqrt(np.random.uniform(0, noise_base ** 2, V1.shape[0]))
            x = np.tile((r * np.cos(angle))[:, np.newaxis], (1, 3))
            y = np.tile((r * np.sin(angle))[:, np.newaxis], (1, 3))
            points_offset = x * V1 + y * V2

            mesh.vertices = mesh.vertices + points_offset
        return

    def add_noise_on_pcs(self, pcs, meshes, piece_id,mean_edge_len, noise_strength=50, noise_type="default", fid=None):
        # 噪声基数
        noise_base = noise_strength * mean_edge_len

        # 最简单的加噪方式
        if noise_type == "default":
            results_pcs = []
            for pc in pcs:
                pc = pc + get_sphere_noise(len(pc), noise_base)
                results_pcs.append(np.array(pc))
            return results_pcs
        # 根据法线进行加噪的方式
        elif noise_type == "normal":
            # 对每个顶点，获取它所在面的法线
            v_n = np.concatenate(np.array([np.array(mesh.face_normals[fid[i]]) for i, mesh in enumerate(meshes)]),
                                 axis=0)

            V1 = np.zeros_like(v_n)
            V2 = np.zeros_like(v_n)

            for i in range(v_n.shape[0]):
                vec = v_n[i]

                # Normalize the input vector
                vec = vec / np.linalg.norm(vec)

                # Find the first orthogonal vector
                if vec[0] != 0 or vec[1] != 0:
                    V1[i] = np.array([-vec[1], vec[0], 0])
                else:
                    V1[i] = np.array([1, 0, 0])

                # Find the second orthogonal vector using cross product
                V2[i] = np.cross(vec, V1[i])

            V1 = np.array(V1) / np.linalg.norm(V1)
            V2 = np.array(V2) / np.linalg.norm(V2)

            # Initialize an array to store points
            points_offset = np.zeros((V1.shape[0], 3))

            angle = np.random.uniform(0, 2 * np.pi, V1.shape[0])
            r = np.sqrt(np.random.uniform(0, noise_base ** 2, V1.shape[0]))
            x = np.tile((r * np.cos(angle))[:, np.newaxis], (1, 3))
            y = np.tile((r * np.sin(angle))[:, np.newaxis], (1, 3))
            points_offset = x * V1 + y * V2

            sum = 0
            results_pcs = []
            for pc in pcs:
                pc = pc + points_offset[sum:sum + len(pc)]
                sum += len(pc)
                results_pcs.append(np.array(pc))

            return results_pcs
        else:
            raise ValueError(f"noise_type \"{noise_type}\" is not valid.")

    # [todo] 根据训练、测试去读取不同的数据
    def load_meshes(self, data_folder):
        mesh_files = sorted(glob(os.path.join(data_folder, "piece_*.obj")))

        if not self.min_num_part <= len(mesh_files) <= self.max_num_part:
            raise ValueError(f"Part num of {data_folder}({len(mesh_files)}) out of range [{self.min_num_part}:{self.max_num_part}]")

        meshes = [trimesh.load(mesh_file, force="mesh", process=False) for mesh_file in mesh_files]

        return meshes

    def load_full_uv_info(self, data_folder):
        if self.read_uv:
            full_uv_info = np.load(os.path.join(data_folder, "annotations", "uv.npy"))
            return full_uv_info
        else:
            return None

    def shrink_meshes(self, meshes, shrink_param=0.5):
        """
        将一个Panel上的边界点往内部缩一些

        :param meshes:
        :param shrink_param:
        :return:
        """
        if shrink_param > 1 or shrink_param < 0:
            raise ValueError(f"shrink_param={shrink_param} is invalid")

        for mesh in meshes:
            # 跳过太小的Panel
            if len(mesh.vertices) < 400: continue
            # 对每个边缘点，找其不是边缘点的相邻点 ----------------------------------------------------------------------------
            edge_points_loops = igl.all_boundary_loop(mesh.faces)  # 获取边界上的点
            all_boundary_points = np.concatenate(edge_points_loops)
            neighbor_points = {}    # 每个边缘点的所有相邻点
            neighbor_boundary_points = {}  # 每个边缘点的所有相邻边缘点
            for loop in edge_points_loops:
                for p_idx, point in enumerate(loop):
                    # 对最外面的每个边界点，找到与其相邻的点
                    neighbors = mesh.vertex_neighbors[point]
                    neighbor_points[point] = neighbors

                    if p_idx==0:
                        neighbors_boundaey = [loop[-1], loop[p_idx+1]]
                    elif p_idx==len(loop)-1:
                        neighbors_boundaey = [loop[p_idx-1], loop[0]]
                    else:
                        neighbors_boundaey = [loop[p_idx - 1], loop[p_idx+1]]
                    neighbor_boundary_points[point] = neighbors_boundaey


            # 对每个边界点，和neighbor_points中的点，计算shrink后的位置 -------------------------------------------------------
            # 先计算新位置
            new_vertices_positions = {}
            for b_point in all_boundary_points:

                neighbors_boundaey = neighbor_boundary_points[b_point]
                neighbors_boundaey_position = mesh.vertices[neighbors_boundaey]
                neighbors_boundaey_vector = neighbors_boundaey_position[1] - neighbors_boundaey_position[0]
                neighbors_boundaey_vector = neighbors_boundaey_vector/np.linalg.norm(neighbors_boundaey_vector)

                b_point_pos = np.array(mesh.vertices[b_point])
                center = get_pc_bbox(mesh.vertices[neighbor_points[b_point]], type="ccwh")[0]

                AB = center - b_point_pos
                proj_v_AB = (np.dot(AB, neighbors_boundaey_vector) / np.dot(neighbors_boundaey_vector, neighbors_boundaey_vector)) * neighbors_boundaey_vector
                perpendicular = AB - proj_v_AB

                target_position = b_point_pos + perpendicular

                new_position = (shrink_param * b_point_pos +
                                (1 - shrink_param) * target_position)
                new_vertices_positions[b_point] = new_position

            # 再应用到mesh里
            for b_point in all_boundary_points:
                mesh.vertices[b_point] = new_vertices_positions[b_point]
        # 保存结果
        # meshes_visualize(meshes, "after_shrink")

    def _get_pcs(self, data_folder):
        meshes = self.load_meshes(data_folder)

        # 让mesh的每个边界点往内缩
        if self.shrink_mesh:
            self.shrink_meshes(meshes, self.shrink_mesh_param)

        full_uv_info = self.load_full_uv_info(data_folder)

        if self.pcs_sample_type == "boundary_mesh":
            sample_result = self.sample_point_byBoundaryMesh(meshes, self.num_points)
            if full_uv_info is not None:
                sample_result["uv"] = full_uv_info
            return sample_result
        elif self.pcs_sample_type == "boundary_pcs":  # 不会去采样固定数量个点
            sample_result = self.sample_point_byBoundaryPcs(meshes, self.num_points)
            if full_uv_info is not None:
                sample_result["uv"] = full_uv_info
            return sample_result
        elif self.pcs_sample_type == "stitch":
            if self.num_points%2!=0:
                raise ValueError("self.num_points should be an even number when self.pcs_sample_type==\"stitch\"")
            stitches = np.load(os.path.join(data_folder, "annotations", "stitch.npy"))
            # stitch_visualize(np.concatenate([np.array(mesh.vertices) for mesh in meshes], axis = 0), stitch)
            # [todo] 将来如果有空，试着从根本上解决这个问题
            max_check_times = 8  # 最大重复采样次数
            min_samplenum_prepanel = 4  # 单个Panel上的最少采样点数量
            sample_result = self.sample_point_byStitch(meshes, self.num_points, stitches, full_uv_info, n_rings=2,
                                                       max_check_times=max_check_times, min_samplenum_prepanel=min_samplenum_prepanel)
            return sample_result
        else:
            raise NotImplementedError(f"pcs_sample_type: {self.pcs_sample_type} hasen't been implemented")


    def __getitem__(self, index):
        # 获取所需路径 ------------------------------------------------------------------------------------------------
        mesh_file_path = self.data_list[index]
        try:
            garment_json_path = glob(os.path.join(mesh_file_path, "annotations", "garment*.json"))[0]
            garment_json_path = garment_json_path if os.path.exists(garment_json_path) else ""
        except Exception:
            garment_json_path = ""

        annotations_json_path = os.path.join(mesh_file_path, "annotations", "annotations.json")
        annotations_json_path = annotations_json_path if os.path.exists(annotations_json_path) else ""

        # 进行采样 ---------------------------------------------------------------------------------------------------
        sample_result= self._get_pcs(mesh_file_path)
        pcs = sample_result["pcs"]
        nps = sample_result["nps"]
        mat_gt = sample_result["mat_gt"]
        piece_id = sample_result["piece_id"]
        mean_edge_len = sample_result["mean_edge_len"]
        normalize_range = sample_result["normalize_range"]
        # 对uv进行和pc相同的normalize
        if "uv" in sample_result.keys():
            uv = sample_result["uv"]
            uv = styleXD_normalize(uv)[0]
        else: uv=None

        # pointcloud_visualize(np.concatenate(pcs), colormap="tab10", colornum=2)
        num_parts = len(pcs)

        # # 在位置变化前，先加一次很小的缝合噪声 ---------------------------------------------------------------------------
        # if self.use_stitch_noise:
        #     # pointcloud_visualize(pcs)
        #     # pointcloud_visualize([pcs[piece_id==i] for i in range(len(pcs))])
        #     # pointcloud_and_stitch_visualize(pcs, mat_gt)
        #
        #     # stitch_noise_strength=1 在训练数据上inference结果更好
        #     # stitch_noise_strength=2 在董远数据上inference结果更好
        #
        #     # === 对缝合点 按缝合线加远离噪声 ===
        #     all_pcs = np.concatenate(pcs)
        #     # 随机噪声强度
        #     rand_param = random.random()
        #     stitch_noise_strength_base = 0.05 * rand_param + 0.1 * (1 - rand_param)
        #     stitch_noise_strength_base *= self.stitch_noise_strength
        #
        #     # mat_gt的shape=Nx2，保存的是每一对缝合点的index
        #     vec = all_pcs[mat_gt[:, 1]] - all_pcs[mat_gt[:, 0]]
        #     # vec 是每一对缝合点之间的向量
        #     vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-6)
        #     # 在缝合的两边加的噪声
        #     noise1 = (vec * np.random.rand(len(mat_gt), 1) * 0.7 + 0.3) * stitch_noise_strength_base * mean_edge_len * 1.2
        #     noise2 = (vec * np.random.rand(len(mat_gt), 1) * 0.7 + 0.3) * stitch_noise_strength_base * mean_edge_len * 1.2
        #     # cur_pcs的shape为Mx3，是所有点的位置
        #     all_pcs[mat_gt[:, 1]] += noise1
        #     all_pcs[mat_gt[:, 0]] -= noise2
        #
        #     nps_cumsum = np.cumsum(nps)
        #     for idx in range(len(pcs)):
        #         if idx == 0:
        #             point_start_idx = 0
        #         else:
        #             point_start_idx = nps_cumsum[idx - 1]
        #         point_end_idx = nps_cumsum[idx]
        #         pcs[idx] = all_pcs[point_start_idx:point_end_idx]
        #
        # # pointcloud_visualize(all_pcs)
        # # pointcloud_visualize([all_pcs[piece_id==i] for i in range(len(all_pcs))])
        # # pointcloud_and_stitch_visualize(all_pcs, mat_gt)


        cur_pcs, cur_pcs_gt ,cur_quat, cur_trans= [], [], [], []

        points_end_idxs = np.cumsum(nps)

        for idx, (pc, n_p) in enumerate(zip(pcs, nps)):
            if idx == 0: point_start_idx = 0
            else: point_start_idx = points_end_idxs[idx - 1]
            # point_end_idx = points_end_idxs[idx]

            pc_gt = pc.copy()
            if self.panel_noise_type == "default":
                # [todo] 这里改成用 byBBOX
                # self._random_SM_byBbox(pc)
                pc, gt_trans, gt_quat = self._random_SRM_default(pc, pcs, idx, mean_edge_len)
            else:
                raise NotImplementedError("")

            cur_pcs.append(pc)
            cur_pcs_gt.append(pc_gt)
            cur_quat.append(gt_quat)
            cur_trans.append(gt_trans)

        cur_pcs = np.concatenate(cur_pcs).astype(np.float32)  # [N_sum, 3]
        cur_pcs_gt = np.concatenate(cur_pcs_gt).astype(np.float32)  # [N_sum, 3]
        # pointcloud_visualize(cur_pcs)
        cur_quat = self._pad_data(np.stack(cur_quat, axis=0), self.max_num_part).astype(np.float32)  # [P, 4]
        cur_trans = self._pad_data(np.stack(cur_trans, axis=0), self.max_num_part).astype(np.float32)  # [P, 3]

        # pointcloud_and_stitch_visualize(cur_pcs,mat_gt)
        n_pcs = self._pad_data(np.array(nps), self.max_num_part).astype(np.int64)  # [P]
        valids = np.zeros(self.max_num_part, dtype=np.float32)
        valids[:num_parts] = 1.0

        data_dict = {
            "pcs": cur_pcs,                             # pointclouds after random transformation
            "pcs_gt": cur_pcs_gt,                       # pointclouds before random transformation
            "n_pcs": n_pcs,                             # point num of each part
            "part_quat": cur_quat,
            "part_trans": cur_trans,
            "num_parts": num_parts,
            "part_valids": valids,
            "data_id": index,
            "piece_id": piece_id,
            "mesh_file_path": mesh_file_path,               # path of this garment
            "garment_json_path": garment_json_path,         # path of garment.json if exist
            "annotations_json_path": annotations_json_path, # path of annotations.json if exist
            "normalize_range": normalize_range
        }
        if uv is not None:
            # 添加一列零，不然pointnet好像提取不了特征
            uv = np.hstack((uv, np.zeros((uv.shape[0], 1))))
            data_dict["uv"] = uv

        # 添加缝合噪声 ---------------------------------------------------------------------------------------------
        if self.mode == "train" or self.mode == "val" or self.mode == "test":
            # pointcloud_visualize(cur_pcs)
            # pointcloud_visualize([cur_pcs[piece_id==i] for i in range(len(pcs))])
            # pointcloud_and_stitch_visualize(cur_pcs, mat_gt)
            pass
            # === 在位置变化后，加一次缝合噪声 ===
            if self.use_stitch_noise:
                def smooth_using_convolution(noise, k=[1,1,1]):
                    kernel = np.array(k) / sum(k)  # 平滑卷积核
                    smoothed_noise = convolve1d(noise, kernel, axis=0, mode='nearest')
                    return smoothed_noise

                # 随机噪声强度
                rand_param = random.random()
                stitch_noise_strength_base = (self.stitch_noise_random_min * rand_param +
                                              self.stitch_noise_random_max * (1-rand_param))
                stitch_noise_strength_base *= self.stitch_noise_strength

                # === 对缝合点 按缝合线加远离噪声 ===
                stitch_noise_strength1 = stitch_noise_strength_base
                vec =  cur_pcs[mat_gt[:, 1]] - cur_pcs[mat_gt[:,0]]
                vec2 = cur_pcs[mat_gt[:, 0]] - cur_pcs[mat_gt[:, 1]]
                # 是每一对缝合点之间的向量
                vec = vec/(np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
                vec2 = vec2/(np.linalg.norm(vec2, axis=1, keepdims=True) + 1e-12)

                # vec = smooth_using_convolution(vec, [1,1,1,1,1])

                # 在缝合的两边加的噪声
                noise1 = (vec * np.random.rand(len(mat_gt), 1)*0.9+0.1) * stitch_noise_strength1 * mean_edge_len
                noise2 = (vec2 * np.random.rand(len(mat_gt), 1)*0.9+0.1) * stitch_noise_strength1 * mean_edge_len
                # cur_pcs的shape为Mx3，是所有点的位置
                # noise1 = smooth_using_convolution(noise1)
                # noise2 = smooth_using_convolution(noise2)
                cur_pcs[mat_gt[:, 1]] += noise1
                cur_pcs[mat_gt[:, 0]] += noise2

                # # === 对不缝合的点加随机噪声 ===
                # stitch_noise_strength3 = stitch_noise_strength_base * (random.random()*0.4+0.2)
                # unstitch_mask = np.zeros(cur_pcs.shape[0])
                # unstitch_mask[mat_gt[:, 0]] = 1
                # unstitch_mask[mat_gt[:, 1]] = 1
                # unstitch_mask = unstitch_mask == 0
                # noise3 = (np.random.rand(np.sum(unstitch_mask), 3)*2.-1.)
                # noise3 = noise3 / (np.linalg.norm(noise3, axis=1, keepdims=True) + 1e-6)
                # noise3 = noise3 * stitch_noise_strength3 * mean_edge_len
                # cur_pcs[unstitch_mask] += noise3

                # === 对缝合的点的两端加相同噪声（只加一点点） ===
                stitch_noise_strength4 = stitch_noise_strength_base * (random.random()*0.3+0.15)
                noise4 = (np.random.rand(len(mat_gt), 3)*2.-1.)
                noise4 = noise4 / (np.linalg.norm(noise4, axis=1, keepdims=True) + 1e-6)
                noise4 = noise4 * stitch_noise_strength4 * mean_edge_len
                cur_pcs[mat_gt[:, 0]] += noise4
                cur_pcs[mat_gt[:, 1]] += noise4

                # # === 对缝合的点再加点噪声（只加一点点） ===
                # stitch_mask = ~unstitch_mask
                # noise5 = (np.random.rand(np.sum(stitch_mask), 3) * 2. - 1.)
                # noise5 = noise5 / (np.linalg.norm(noise5, axis=1, keepdims=True) + 1e-6)
                # noise5 = noise5 * stitch_noise_strength * mean_edge_len * 0.2
                # cur_pcs[stitch_mask] += noise5
            # pointcloud_visualize(cur_pcs)
            # pointcloud_visualize([cur_pcs[piece_id==i] for i in range(len(pcs))])
            # pointcloud_and_stitch_visualize(cur_pcs, mat_gt)

            # GT 缝合关系
            mat_gt = stitch_indices2mat(self.num_points, mat_gt)
            data_dict["mat_gt"] = mat_gt

            # [todo] 改完后用下面的代码测试下
            # pointcloud_and_stitch_visualize(cur_pcs,mat_gt)

            # 平均缝合长度
            Dis = np.sqrt(np.sum(((cur_pcs[:,None,:] - cur_pcs[None,:,:])**2), axis=-1))
            data_dict["mean_stitch_dis_gt"] = np.mean(Dis[mat_gt == 1])
        else:
            pass
        return data_dict

def build_stylexd_dataloader_train_val(cfg):
    # train、val用的是StyleXD带l的obj文件，用stitch采样
    # TRAIN DATASET ----------------------------------------------------------------------------------------------------
    data_dict = dict(
        mode="train",
        data_dir=cfg.DATA.DATA_DIR,
        data_keys=cfg.DATA.DATA_KEYS,

        num_points=cfg.DATA.NUM_PC_POINTS,
        min_num_part=cfg.DATA.MIN_NUM_PART,
        max_num_part=cfg.DATA.MAX_NUM_PART,

        shrink_mesh=cfg.DATA.SHRINK_MESH.TRAIN,
        shrink_mesh_param=cfg.DATA.SHRINK_MESH_PARAM.TRAIN,

        # stitch_noise only used for train data
        use_stitch_noise=cfg.DATA.USE_STITCH_NOISE,
        stitch_noise_strength=cfg.DATA.STITCH_NOISE_STRENGTH,
        stitch_noise_random_range=cfg.DATA.STITCH_NOISE_RANDOM_RANGE,

        pcs_sample_type=cfg.DATA.PCS_SAMPLE_TYPE.TRAIN,
        pcs_noise_type=cfg.DATA.PCS_NOISE_TYPE,
        pcs_noise_strength=cfg.DATA.PCS_NOISE_STRENGTH,
        panel_noise_type=cfg.DATA.PANEL_NOISE_TYPE.TRAIN,

        scale_range=cfg.DATA.SCALE_RANGE,
        rot_range=cfg.DATA.ROT_RANGE,
        trans_range=cfg.DATA.TRANS_RANGE,

        overfit=cfg.DATA.OVERFIT,
        min_part_point=cfg.DATA.MIN_PART_POINT,

        read_uv=cfg.MODEL.USE_UV_FEATURE,
    )
    train_set = AllPieceMatchingDataset_stylexd(**data_dict)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=cfg.DATA.SHUFFLE,  # [todo] 真的开始训练之后改成True
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    # VAL DATASET ------------------------------------------------------------------------------------------------------
    # data_dict["shuffle_parts"] = False
    data_dict["mode"] = "val"
    data_dict["pcs_sample_type"] = cfg.DATA.PCS_SAMPLE_TYPE.VAL
    data_dict["panel_noise_type"] = cfg.DATA.PANEL_NOISE_TYPE.VAL
    data_dict["shrink_mesh"] = cfg.DATA.SHRINK_MESH.VAL
    data_dict["shrink_mesh_param"] = cfg.DATA.SHRINK_MESH_PARAM.VAL

    val_set = AllPieceMatchingDataset_stylexd(**data_dict)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    return train_loader, val_loader

def build_stylexd_dataloader_test(cfg):
    # test用的是董远生成的不带l的obj文件，用边界点采样法
    data_dict = dict(
        mode="test",
        data_dir=cfg.DATA.DATA_DIR,
        data_keys=cfg.DATA.DATA_KEYS,
        data_types=cfg.DATA.DATA_TYPES.TEST,

        num_points=cfg.DATA.NUM_PC_POINTS,
        min_num_part=cfg.DATA.MIN_NUM_PART,
        max_num_part=cfg.DATA.MAX_NUM_PART,

        shrink_mesh=cfg.DATA.SHRINK_MESH.TEST,
        shrink_mesh_param=cfg.DATA.SHRINK_MESH_PARAM.TEST,

        pcs_sample_type=cfg.DATA.PCS_SAMPLE_TYPE.TEST,
        pcs_noise_type=cfg.DATA.PCS_NOISE_TYPE,
        pcs_noise_strength=cfg.DATA.PCS_NOISE_STRENGTH,
        panel_noise_type=cfg.DATA.PANEL_NOISE_TYPE.TEST,

        scale_range=cfg.DATA.SCALE_RANGE,
        rot_range=cfg.DATA.ROT_RANGE,
        trans_range=cfg.DATA.TRANS_RANGE,

        overfit=cfg.DATA.OVERFIT,
        min_part_point=cfg.DATA.MIN_PART_POINT,

        read_uv=True,
    )
    test_set = AllPieceMatchingDataset_stylexd(**data_dict)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=cfg.DATA.SHUFFLE,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    return test_loader

def build_stylexd_dataloader_inference(cfg):
    # test用的是董远生成的不带l的obj文件，用边界点采样法
    data_dict = dict(
        mode="inference",
        data_dir=cfg.DATA.DATA_DIR,
        data_keys=cfg.DATA.DATA_KEYS,
        data_types=cfg.DATA.DATA_TYPES.TEST,

        num_points=cfg.DATA.NUM_PC_POINTS,
        min_num_part=cfg.DATA.MIN_NUM_PART,
        max_num_part=cfg.DATA.MAX_NUM_PART,

        shrink_mesh=cfg.DATA.SHRINK_MESH.TEST,
        shrink_mesh_param=cfg.DATA.SHRINK_MESH_PARAM.TEST,

        pcs_sample_type=cfg.DATA.PCS_SAMPLE_TYPE.TEST,
        pcs_noise_type=cfg.DATA.PCS_NOISE_TYPE,
        pcs_noise_strength=cfg.DATA.PCS_NOISE_STRENGTH,
        panel_noise_type=cfg.DATA.PANEL_NOISE_TYPE.TEST,

        scale_range=cfg.DATA.SCALE_RANGE,
        rot_range=cfg.DATA.ROT_RANGE,
        trans_range=cfg.DATA.TRANS_RANGE,

        overfit=cfg.DATA.OVERFIT,
        min_part_point=cfg.DATA.MIN_PART_POINT,

        read_uv = True
    )
    test_set = AllPieceMatchingDataset_stylexd(**data_dict)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=cfg.DATA.SHUFFLE,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    return test_loader
import os
import pickle
import random
from binascii import Error

import numpy as np
import trimesh
import trimesh.sample
import igl
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import pdist, squareform
from torch.utils.data import Dataset, DataLoader
from glob import glob

from trimesh.permutate import noise
from wandb.wandb_torch import torch

from utils.lr import set_lr
from utils import get_sphere_noise, min_max_normalize, get_pc_bbox, compute_adjacency_list
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
            category="all",
            num_points=1000,
            min_num_part=2,
            max_num_part=20,
            shuffle_parts=False,
            shrink_mesh=False,
            shrink_mesh_param=1,
            panel_noise_type="default",
            scale_range=0,
            rot_range=-1,
            trans_range=1,
            overfit=10,
            length=-1,
            pcs_sample_type="area",
            shuffle_points_after_sample=False,
            pcs_noise_strength=1.,
            pcs_noise_type="default",
            use_stitch_noise = False,  # 是否沿着缝合线添加noise
            min_part_point=30,
            fracture_label_threshold=0.025,
            mode= "train",
    ):
        if mode not in ["train","val","test"]:
            raise ValueError(f"mode=\"{mode}\" is not valid.")

        self.mode = mode

        # store parameters
        self.category = category if category.lower() != "all" else ""

        self.data_dir = data_dir
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
        self.shuffle_points_after_sample = shuffle_points_after_sample

        self.shrink_mesh=shrink_mesh
        self.shrink_mesh_param=shrink_mesh_param

        self.pcs_noise_strength = pcs_noise_strength
        if pcs_noise_type not in ["default", "normal"]:
            raise ValueError(f"pcs_noise_type=\"{pcs_noise_type}\" is not valid.")
        self.pcs_noise_type = pcs_noise_type
        self.use_stitch_noise = use_stitch_noise

        self.data_list = self._read_data()

        print("dataset length: ", len(self.data_list))

        # additional data to load, e.g. ('part_ids', 'instance_label')
        self.data_keys = data_keys

        # overfit = 10

        if overfit > 0:
            self.data_list = self.data_list[:overfit]

        self.length = len(self.data_list)
        # if 0 < length < len(self.data_list):
        #     self.length = length
        # else:
        #     self.length = len(self.data_list)

        self.fracture_label_threshold = fracture_label_threshold

    def __len__(self):
        return self.length

    def _read_data(self):
        mesh_dir = os.path.join(self.data_dir, self.mode)
        data_list = sorted(glob(os.path.join(mesh_dir, "garment_*")))
        return data_list

    # 用于训练inference董远数据的模型
    # random scale+rotate+move panel (default)
    def _random_SRM_default(self, pc, mean_edge_len):
        """
        pc: [N, 3]
        """
        pc_centroid = get_pc_bbox(pc)[0]
        # SCALE ------------------------------
        pc = pc - pc_centroid[None]
        # [todo]0.95改回去1
        # scale_gt = (np.random.rand(1) * self.scale_range) + 1.0
        scale_gt = (np.random.rand(1) * self.scale_range) + 0.95
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
        quat_gt = quat_gt[[3, 0, 1, 2]]
        pc = pc + pc_centroid[None]

        # MOVE ------------------------------
        trans_gt = (mean_edge_len * (np.random.rand(3) - 0.5) * 2.0 * self.trans_range).reshape(-1,3)
        pc = pc + trans_gt
        return pc, trans_gt, quat_gt

    # 用于训练inference部件组合的模型
    # random move+scale+rotate panel (default)
    def _random_MSR_partcombin(self, pc, pcs, pc_idx, mean_edge_len):
        """
        pc: [N, 3]
        pcs: List[[N1,3],[N2,3],[N3,3],......]
        """
        pc=pc.copy()
        bkg_pcs = np.concatenate(pcs[:pc_idx]+pcs[pc_idx+1:])
        pc_centroid = get_pc_bbox(pc)[0]
        bkg_pcs_centroid = get_pc_bbox(bkg_pcs)[0]

        # MOVE ------------------------------
        move_vec = pc_centroid-bkg_pcs_centroid
        move_vec = move_vec/np.linalg.norm(move_vec)
        # 水平投影，引导Panel在水平上向外移动
        move_vec_horizon = np.zeros_like(move_vec)
        move_vec_horizon[:] = move_vec[:]
        move_vec_horizon[1] = 0.
        move_vec_horizon = move_vec_horizon/np.linalg.norm(move_vec_horizon)

        move_vec = move_vec + move_vec_horizon
        move_vec = move_vec / np.linalg.norm(move_vec)

        trans_gt = (mean_edge_len * move_vec * (np.random.rand(3) + 1.) / 4. * self.trans_range).reshape(-1, 3)
        pc = pc + trans_gt

        # SCALE ------------------------------
        pc_centroid = get_pc_bbox(pc)[0]
        pc=pc-pc_centroid
        scale_gt = np.random.rand(3)*2-1
        scale_gt=scale_gt*np.abs(scale_gt)
        scale_gt = scale_gt/np.linalg.norm(scale_gt)
        scale_gt = np.ones(3) + scale_gt * self.scale_range
        pc = pc * scale_gt

        # ROTATE ------------------------------
        if self.rot_range >= 0.0:
            rot_euler = (np.random.rand(3) - 0.5) * 2.0 * self.rot_range
            rot_mat = R.from_euler("xyz", rot_euler, degrees=True).as_matrix()
        else:
            rot_mat = R.random().as_matrix()
        pc = (rot_mat @ pc.T).T
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        # we use scalar-first quaternion
        quat_gt = quat_gt[[3, 0, 1, 2]]
        pc = pc + pc_centroid[None]

        return pc, trans_gt, quat_gt

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

    # @staticmethod
    # def sample_points_by_areas(areas, num_points):
    #     """areas: [P], num_points: N"""
    #     total_area = np.sum(areas)
    #     nps = np.ceil(areas * num_points / total_area).astype(np.int32)
    #     nps[np.argmax(nps)] -= np.sum(nps) - num_points
    #     return np.array(nps, dtype=np.int64)

    # def sample_reweighted_points_by_areas(self, areas, numpoints):
    #     """ Sample points by areas, but ensures that each part has at least # points.
    #     areas: [P]
    #     """
    #     nps = self.sample_points_by_areas(areas, numpoints)
    #     if self.min_part_point <= 1:
    #         return nps
    #     delta = 0
    #     for i in range(len(nps)):
    #         if nps[i] < self.min_part_point:
    #             delta += self.min_part_point - nps[i]
    #             nps[i] = self.min_part_point
    #     while delta > 0:
    #         k = np.argmax(nps)
    #         if nps[k] - delta >= self.min_part_point:
    #             nps[k] -= delta
    #             delta = 0
    #         else:
    #             delta -= nps[k] - self.min_part_point
    #             nps[k] = self.min_part_point
    #     # simply take points from the largest parts
    #     # This implementation is not very elegant, could improve by resample by areas.
    #     return np.array(nps, dtype=np.int64)

    # # 对一批mesh按照表面积进行随机采样
    # def sample_point_byArea(self, meshes, num_points):
    #     meshes_boundary, meshes_inner = [], []
    #
    #     for mesh_idx, mesh in enumerate(meshes):
    #         vertices = np.array(mesh.vertices)
    #         faces = np.array(mesh.faces)
    #
    #         # 获取所有的边界点
    #         boundary_loops = igl.all_boundary_loop(faces)
    #         v_idx_boundary = []
    #         for boundary_loop in boundary_loops:
    #             v_idx_boundary += boundary_loop
    #
    #         # PART 1 ---------------------------------------------------------------------------------------------------
    #         # 在边界点上进行扩张expand_epoch圈的面
    #
    #         # 计算扩张的圈数
    #         expand_epoch = 1
    #         if self.mode == "test":
    #             # 特别小的mesh没必要扩张
    #             if len(faces) < 400:
    #                 expand_epoch = 1
    #             else:
    #                 expand_epoch = 2
    #         else:
    #             raise NotImplementedError()
    #
    #         f_idx_boundary = []
    #         for e in range(expand_epoch):
    #             v_idx_boundary_new = []
    #             f_idx_boundary_new = []
    #             for f_i, face in enumerate(faces):
    #                 # 判断面上是否有边界点
    #                 for v_i in face:
    #                     if (v_i in v_idx_boundary) and (f_i not in f_idx_boundary):
    #                         if f_i not in f_idx_boundary_new:
    #                             f_idx_boundary_new.append(int(f_i))
    #                         for v_i_f in face:
    #                             if (v_i_f not in f_idx_boundary) and (v_i_f not in f_idx_boundary_new):
    #                                 v_idx_boundary_new.append(v_i_f)
    #                         continue
    #             v_idx_boundary = np.concatenate([v_idx_boundary, v_idx_boundary_new], axis=0)
    #             f_idx_boundary = np.concatenate(
    #                 [np.array(f_idx_boundary, dtype=np.int32), np.array(f_idx_boundary_new, dtype=np.int32)],
    #                 axis=0)
    #
    #         mesh_boundary = mesh.submesh([f_idx_boundary])
    #         meshes_boundary.append(mesh_boundary[0])
    #
    #     # 保存边界mesh
    #     for mesh_idx, mesh in enumerate(meshes_boundary):
    #         mesh.export(os.path.join("_tmp/mesh_split/beforenoised", f'selected_mesh{mesh_idx}.obj'))
    #
    #     # 在mesh上加噪声
    #     # self.add_noise_on_mesh(meshes_boundary)
    #
    #     areas = [mesh.area for mesh in meshes]
    #     areas = np.array(areas)
    #     pcs, piece_id, nps = [], [], []
    #     mat_gt = None
    #     sample_fid = []  # 采样点的faceid
    #
    #     nps = self.sample_reweighted_points_by_areas(areas, num_points)
    #
    #     for i, (mesh) in enumerate(meshes):
    #         num_points = nps[i]
    #
    #         # 两种随机采样方式
    #         samples, fid = trimesh.sample.sample_surface_even(mesh, num_points)  # 更均匀
    #         # samples, fid = mesh.sample(num_points, return_index=True)  # 更随机
    #
    #         pcs.append(samples)
    #         piece_id.append([i] * num_points)
    #         sample_fid.append(fid)
    #
    #     piece_id = np.concatenate(piece_id).astype(np.int64).reshape((-1, 1))
    #
    #     pcs_dict = dict(
    #         pcs=pcs,
    #         piece_id=piece_id,
    #         nps=nps,
    #         mat_gt=mat_gt,
    #         # areas=areas,
    #     )
    #     return pcs_dict, sample_fid

    def sample_point_byBoundaryMesh(self, meshes, num_points):
        # [todo] 过滤毛刺的方法存在隐患
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

        # # PART 2 -------------------------------------------------------------------------------------------------------
        # # 过滤掉毛刺
        #
        # # e_len_list = []
        # # pt_l = adjacency_list[:, 1]
        # # pt_r = adjacency_list[:, 2]
        # # e_len_l = np.sum(np.sqrt((vertices[pt_l] - vertices[boundary_points_idx]) ** 2), axis=-1)
        # # e_len_r = np.sum(np.sqrt((vertices[boundary_points_idx] - vertices[pt_r]) ** 2), axis=-1)
        # # threshold = 0.04
        # # e_len_l = e_len_l < threshold
        # # e_len_r = e_len_r < threshold
        #
        # # loop_end_idxs = np.cumsum(n_loop_pts)
        # # filtered_loop_list = []
        # # adjacency_list = []
        # # for idx, loop_len in enumerate(n_loop_pts):
        # #     if idx == 0:
        # #         loop_start_idx = 0
        # #     else:
        # #         loop_start_idx = loop_end_idxs[idx - 1]
        # #     loop_end_idx = loop_end_idxs[idx]
        # #     loop = boundary_points_idx[loop_start_idx:loop_end_idx][e_len_l[loop_start_idx:loop_end_idx]]
        # #     filtered_loop_list.append(loop)
        # #
        # #     boundary_adjacency = np.zeros((len(loop), 4), dtype=np.int32)
        # #     if len(loop) > 4:
        # #         boundary_adjacency[:, 0] = np.concatenate([loop[2:], loop[:2]], axis=-1)
        # #         boundary_adjacency[:, 1] = np.concatenate([loop[1:], loop[:1]], axis=-1)
        # #         boundary_adjacency[:, 2] = np.concatenate([loop[-1:], loop[:-1]], axis=-1)
        # #         boundary_adjacency[:, 3] = np.concatenate([loop[-2:], loop[:-2]], axis=-1)
        # #     else:
        # #         boundary_adjacency[:, :] = loop
        # #     adjacency_list.append(boundary_adjacency)
        # # boundary_points_idx = np.concatenate(filtered_loop_list,axis=-1)
        # # adjacency_list = np.concatenate(adjacency_list,axis=-2)
        # # # pointcloud_visualize(vertices[boundary_points_idx])
        #
        # # PART 3 -------------------------------------------------------------------------------------------------------
        # # 随机采样边界点
        #
        # sample_idx = LatinHypercubeSample(len(boundary_points_idx), num_points)
        # sample_idx = sorted(sample_idx)
        # boundary_points_sample_idx = boundary_points_idx[sample_idx]
        # boundary_points_sample = vertices[boundary_points_sample_idx]
        # adjacent_sample = adjacency_list[sample_idx]
        # adjacent_sample_point = vertices[adjacent_sample]
        # piece_id = np.concatenate([np.array([idx] * n) for idx, n in enumerate(nps)], axis=0)[boundary_points_sample_idx]
        #
        # # PART 4 -------------------------------------------------------------------------------------------------------
        # # 对采样的边界点进行平滑化
        #
        # # # 平滑参数
        # # weight_flat = 0
        # # sample_points = (
        # #     weight_flat *
        # #         (np.mean(adjacent_sample_point[:, 1:3, :], axis=1) * 0.2 +
        # #          np.mean(adjacent_sample_point[:, 0::3, :], axis=1) * 0.8 ) +
        # #     boundary_points_sample *
        # #         (1-weight_flat)
        # #     )
        # # # pointcloud_visualize(sample_points)
        # # points_normalized, normalize_range = min_max_normalize(sample_points)
        #
        #
        # points_normalized, normalize_range = min_max_normalize(boundary_points_sample)
        # mean_edge_len = cal_mean_edge_len(meshes) / normalize_range
        # # pointcloud_visualize(points_normalized)
        #
        # points_normalized += get_sphere_noise(self.num_points,radius=self.pcs_noise_strength * mean_edge_len)
        # points_normalized, normalize_range = min_max_normalize(boundary_points_sample)

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
        )
        return sample_result, mean_edge_len

    def sample_point_byBoundaryPcs(self, meshes, num_points):
        pcs = [np.array(mesh.vertices) for mesh in meshes]
        piece_id = np.concatenate([[idx]*len(pcs[idx]) for idx in range(len(pcs))], axis=0)
        nps = np.array([len(pc) for pc in pcs])
        mean_edge_len = 0.0072
        sample_result =dict(
            pcs=pcs,
            piece_id=piece_id,
            nps=nps,
            mat_gt=None,
        )
        return sample_result, mean_edge_len


    def sample_point_byStitch(self, meshes, num_points, stitches, n_rings = 2):
        pcs, piece_id, nps = [], [], [len(mesh.vertices) for mesh in meshes]
        # sample_fid = []  # 采样点的faceid

        # 所有的顶点、面、边界点
        vertices = np.concatenate([np.array(mesh.vertices) for mesh in meshes], axis=0)
        faces = []
        boundary_point_list = []

        # PART 1 -------------------------------------------------------------------------------------------------------
        # 因为读取的mesh被划分为多个，因此，面中index和边界点index都需要转换为全局index

        points_end_idxs = np.cumsum(nps)
        for idx, mesh in enumerate(meshes):
            if idx ==0 : point_start_idx = 0
            else: point_start_idx = points_end_idxs[idx-1]
            # point_end_idx = points_end_idxs[idx]
            # 获取边界点的全局index
            for loop in igl.all_boundary_loop(mesh.faces):
                boundary_point_list.append(np.array(loop)+point_start_idx)
            # 获取面的全局index
            faces.append(np.array(mesh.faces)+point_start_idx)
        faces = np.concatenate(faces,axis=0)
        boundary_points_idx = np.concatenate(boundary_point_list,axis=0)


        # PART 2 -------------------------------------------------------------------------------------------------------
        # 按比例对具有缝合关系的点 和 不具有缝合关系的 边界点 进行采样

        # 所有具有缝合关系的点的index
        stitch_verties_idx = np.concatenate([stitches[:, 0], stitches[:, 1]], axis=0)
        # 所有不具有缝合关系的边界点的index
        unstitched_boundary_points_idx = boundary_points_idx[
            np.array([s not in stitch_verties_idx for s in boundary_points_idx])]

        # 在具有缝合关系的点上进行采样的数量
        stitched_sample_num = int((num_points * (
                    len(stitch_verties_idx) / (len(stitch_verties_idx) + len(unstitched_boundary_points_idx)))) / 2)
        # 在不具有缝合关系的边界点上进行采样的数量
        unstitched_sample_num = num_points - stitched_sample_num * 2

        # 点点缝合关系的映射
        stitch_map = np.zeros(shape=(len(vertices)), dtype=np.int32) - 1
        stitch_map[stitches[:, 0]] = stitches[:, 1]
        stitch_map[stitches[:, 1]] = stitches[:, 0]

        # 成对采样具有缝合关系的点：缝合起点 stitched_sample_idx；缝合终点 stitched_sample_idx_cor
        stitched_sample_idx = stitch_verties_idx[LatinHypercubeSample(len(stitch_verties_idx), stitched_sample_num)]
        stitched_sample_idx = sorted(stitched_sample_idx)
        # stitched_sample = vertices[stitched_sample_idx]
        # pointcloud_visualize(stitched_sample)
        stitched_sample_idx_cor = stitch_map[stitched_sample_idx]
        # stitched_sample_cor = vertices[stitched_sample_idx_cor]
        # pointcloud_visualize(stitched_sample_cor)
        # 采样不具有缝合关系的边界点
        unstitched_sample_idx = unstitched_boundary_points_idx[LatinHypercubeSample(len(unstitched_boundary_points_idx), unstitched_sample_num)]
        # unstitched_sample = vertices[unstitched_sample_idx]
        # pointcloud_visualize([stitched_sample,unstitched_sample])

        # 采样点中的缝合关系
        mat_gt = np.array([np.array([i,i+stitched_sample_num]) for i in range(stitched_sample_num)])
        # 所有采样点的index
        all_sample_idx = np.concatenate([stitched_sample_idx, stitched_sample_idx_cor, unstitched_sample_idx],axis=0)
        # 每个采样点属于的part
        piece_id = np.concatenate([np.array([idx]*n) for idx, n in enumerate(nps)],axis=0)[all_sample_idx]
        # stitch_visualize(np.concatenate([stitched_sample,stitched_sample_cor,unstitched_sample],axis=0),mat_gt)

        # PART 3 -------------------------------------------------------------------------------------------------------
        # 对每个边界点，进行凸包集采样

        # 获得点与点邻居关系的邻接矩阵
        # adjacency_list = np.array(igl.adjacency_list(faces), dtype=object)  # igl.adjacency_list has memory bug
        adjacency_list = np.array(compute_adjacency_list(faces, len(vertices)), dtype=object)
        # 获取每个采样点的邻居顶点的index的集合
        all_sample_neighbor_idx = adjacency_list[all_sample_idx]
        # 将每个采样点本身也加入这个集合，这个集合对应的顶点可以构成一个凸包集
        all_sample_neighbor_idx = [np.array(n+[i]) for n,i in zip(all_sample_neighbor_idx, all_sample_idx)]
        all_sample_neighbor_points = [vertices[n] for n in all_sample_neighbor_idx]
        # 在每个凸包集内随机采样一个顶点
        sampled_points_CH = np.array([random_point_in_convex_hull(elem) for elem in all_sample_neighbor_points])
        # pointcloud_visualize([sampled_points_CH[0:stitched_sample_num*2],sampled_points_CH[stitched_sample_num*2:]])

        # 对采样结果进行扩张
        all_sample_points = vertices[all_sample_idx]
        V_P = sampled_points_CH - all_sample_points
        sampled_points_CH = all_sample_points + n_rings * V_P * self.pcs_noise_strength
        # pointcloud_visualize([sampled_points_CH[0:stitched_sample_num*2],sampled_points_CH[stitched_sample_num*2:]])
        # pointcloud_and_stitch_visualize(sampled_points_CH,mat_gt[:100])
        # test = np.zeros(self.num_points)
        # for face in faces:
        #     for v_idx in face:
        #         if v_idx in all_sample_idx:
        #             idx = np.where(all_sample_idx==v_idx)
        #             test[idx] += 1

        # PART 4 -------------------------------------------------------------------------------------------------------
        # 将点云大小按最BBOX的最长边归一化到[-0.5 - 0.5]，计算平均边长

        points_normalized, normalize_range = min_max_normalize(sampled_points_CH)
        mean_edge_len = cal_mean_edge_len(meshes)/normalize_range
        # pointcloud_visualize(points_normalized)

        # # PART 5 -------------------------------------------------------------------------------------------------------
        # # pointcloud_and_stitch_visualize(points_normalized, mat_gt)
        # # pointcloud_visualize([points_normalized[piece_id==i] for i in range(len(meshes))])
        # scale_factor = 3
        # points_end_idxs = np.cumsum(np.array([np.sum(piece_id==i) for i in range(len(meshes))]))
        # # garment_center = np.array([0.,-5.,0.])
        # garment_center = np.mean(points_normalized, axis=0)
        # panel_vec_rel_list = []
        # for idx in range(len(meshes)):
        #     if idx == 0:
        #         point_start_idx = 0
        #     else:
        #         point_start_idx = points_end_idxs[idx - 1]
        #     point_end_idx = points_end_idxs[idx]
        #     # panel_center = np.mean(points_normalized[point_start_idx:point_end_idx], axis=0)
        #     panel_center = get_pc_bbox(points_normalized[point_start_idx:point_end_idx], type="ccwh")[0]
        #     panel_vec_rel = panel_center-garment_center
        #     panel_vec_rel = panel_vec_rel/np.linalg.norm(panel_vec_rel)
        #     panel_vec_rel_list.append(panel_vec_rel)
        #     points_normalized[point_start_idx:point_end_idx] += panel_vec_rel * mean_edge_len * scale_factor
        # # for idx in range(len(meshes)):
        # #     if idx == 0:
        # #         point_start_idx = 0
        # #     else:
        # #         point_start_idx = points_end_idxs[idx - 1]
        # #     point_end_idx = points_end_idxs[idx]
        # #
        # #     center = panel_center[idx]
        # #     points_normalized[point_start_idx:point_end_idx] = points_normalized[point_start_idx:point_end_idx] - center
        # #     points_normalized[point_start_idx:point_end_idx] = points_normalized[point_start_idx:point_end_idx] / scale_factor
        # #     points_normalized[point_start_idx:point_end_idx] = points_normalized[point_start_idx:point_end_idx] + center
        #
        # # 顺着缝合线，在缝合点上加噪
        # # pointcloud_visualize([points_normalized[piece_id==i] for i in range(len(meshes))])
        # stitch_pair_num = len(stitched_sample_idx)
        # noise_strength = mean_edge_len * 2
        # stitch_vec = points_normalized[stitch_pair_num:stitch_pair_num*2] - points_normalized[:stitch_pair_num]
        # stitch_vec = stitch_vec/np.linalg.norm(stitch_vec, axis=1, keepdims=True)
        # stitch_vec = stitch_vec.copy()
        # random_noise1 = (stitch_vec * np.random.rand(stitch_pair_num, 1)*0.7+0.3) * noise_strength
        # random_noise2 = (stitch_vec * np.random.rand(stitch_pair_num, 1)*0.7+0.3) * noise_strength
        # points_normalized[:stitch_pair_num] += random_noise1
        # points_normalized[stitch_pair_num:stitch_pair_num * 2] -= random_noise2
        # # pointcloud_visualize([points_normalized[:stitch_pair_num],points_normalized[stitch_pair_num:stitch_pair_num * 2]])
        # for idx in range(len(meshes)):
        #     if idx == 0:
        #         point_start_idx = 0
        #     else:
        #         point_start_idx = points_end_idxs[idx - 1]
        #     point_end_idx = points_end_idxs[idx]
        #     panel_vec_rel = panel_vec_rel_list[idx]
        #     points_normalized[point_start_idx:point_end_idx] -= panel_vec_rel * mean_edge_len * scale_factor

        # for idx in range(len(meshes)):
        #     if idx == 0:
        #         point_start_idx = 0
        #     else:
        #         point_start_idx = points_end_idxs[idx - 1]
        #     point_end_idx = points_end_idxs[idx]
        #
        #     center = panel_center[idx]
        #     points_normalized[point_start_idx:point_end_idx] = points_normalized[point_start_idx:point_end_idx] - center
        #     points_normalized[point_start_idx:point_end_idx] = points_normalized[point_start_idx:point_end_idx] * scale_factor
        #     points_normalized[point_start_idx:point_end_idx] = points_normalized[point_start_idx:point_end_idx] + center

        # PART 6 -------------------------------------------------------------------------------------------------------
        # 让采样的缝合关系满足：在index上接近的缝合关系，在位置上也尽可能接近

        # 将采样点按index进行排序，这样同一个part的会凑到一块去
        sorted_indices = np.argsort(all_sample_idx)  # 创建排序索引
        points_normalized = points_normalized[sorted_indices]
        piece_id = piece_id[sorted_indices]

        # 对缝合关系对应的两端顶点的index应用相同的排序，然后按起始点index进行降序排序
        mat_gt = stitch_indices_order(mat_gt, sorted_indices)

        # 如果一个缝合关系末端顶点的index 大于 起始顶点的index，则让它们交换
        mask = mat_gt[:, 0] > mat_gt[:, 1]
        mat_gt[mask] = mat_gt[mask][:, ::-1]

        # 进行排序
        mat_gt = mat_gt[np.argsort(mat_gt[:, 0])]
        # pointcloud_and_stitch_visualize(points_normalized, mat_gt)

        for i in range(len(nps)):
            pcs.append(points_normalized[piece_id==i])
        nps = np.array([len(pc) for pc in pcs])

        pcs_dict = dict(
            pcs=pcs,
            piece_id=piece_id,
            nps=np.array(nps),
            mat_gt=mat_gt,
        )
        return pcs_dict, mean_edge_len

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
            raise ValueError

        meshes = [trimesh.load(mesh_file, force="mesh", process=False) for mesh_file in mesh_files]

        return meshes

    def shrink_meshes(self, meshes, shrink_param = 0.5):
        # 保存结果
        # meshes_visualize(meshes, "before_shrink")

        if shrink_param>1 or shrink_param<0:
            raise ValueError(f"shrink_param={shrink_param} is invalid")

        for mesh in meshes:
            if len(mesh.vertices)<400: continue
            # 对每个边缘点，找其不是边缘点的相邻点 ----------------------------------------------------------------------------
            edge_points_loops = igl.all_boundary_loop(mesh.faces)  # 获取边界上的点
            all_boundary_points = np.concatenate(edge_points_loops)
            neighbor_points = {}
            for loop in edge_points_loops:

                for point in loop:
                    # 对最外面的每个边界点，找到里面一圈的点钟与其相邻的点
                    neighbors = mesh.vertex_neighbors[point]
                    if len(neighbors) >= 3:
                        for n_p in neighbors:
                            if n_p in all_boundary_points:
                                neighbors.remove(n_p)
                    # 如果有一个边界点的相邻点数量小于3，代表这个点的相邻点都是边界点
                    else:
                        pass
                    neighbor_points[point] = neighbors

            # 对每个边界点，和neighbor_points中的点，计算shrink后的位置 -------------------------------------------------------
            # 先计算新位置
            new_vertices_positions = {}
            for b_point in all_boundary_points:
                # 将边缘点往内部收缩
                new_position = (shrink_param * mesh.vertices[b_point] +
                                 (1-shrink_param) * np.mean(mesh.vertices[neighbor_points[b_point]],axis=-2))
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
        # if self.pcs_sample_type == "area":
        #     raise NotImplementedError
        #     sample_result, sample_fid = self.sample_point_byArea(meshes, self.num_points)
        #
        #     # [todo] 计算mean_edge_len前先对点云进行标准化，参考：
        #     # points_normalized, normalize_range = min_max_normalize(sampled_points_CH)
        #     # mean_edge_len = cal_mean_edge_len(meshes) / normalize_range
        #     mean_edge_len = cal_mean_edge_len(meshes)
        #
        #     # # 保存加噪前的采样点 (不区分部件)
        #     # pc_boundary = trimesh.Trimesh(
        #     #     vertices=np.concatenate([np.array(sample_result["pcs"][i]) for i in range(len(sample_result["pcs"]))],
        #     #                             axis=0))
        #     # pc_boundary.export(os.path.join("_tmp/pc_sample", 'before_noised.obj'))
        #
        #     # # 保存加噪前的采样点 (区分部件)
        #     # for idx, pc in enumerate(sample_result["pcs"]):
        #     #     pc_boundary = trimesh.Trimesh(vertices=pc)
        #     #     pc_boundary.export(os.path.join("_tmp/pc_sample", f'before_noised{idx}.obj'))
        #
        #     # 在采样点上加噪
        #     sample_result["pcs"] = self.add_noise_on_pcs(
        #         sample_result["pcs"],
        #         meshes, sample_result["piece_id"], mean_edge_len=mean_edge_len, fid=sample_fid,
        #         noise_strength=0, noise_type=self.pcs_noise_type)
        #
        #     # # 保存加噪后的采样点 (不区分部件)
        #     # pc_boundary = trimesh.Trimesh(
        #     #     vertices=np.concatenate([np.array(sample_result["pcs"][i]) for i in range(len(sample_result["pcs"]))],
        #     #                             axis=0))
        #     # pc_boundary.export(os.path.join("_tmp/pc_sample", 'after_noised.obj'))
        #
        #     # # 保存加噪后的采样点 (区分部件)
        #     # for idx ,pc in enumerate(sample_result["pcs"]):
        #     #     pc_boundary = trimesh.Trimesh(vertices=pc)
        #     #     pc_boundary.export(os.path.join("_tmp/pc_sample", f'after_noised{idx}.obj'))
        #
        #     pcs = sample_result["pcs"]  # [P,N,3]
        #     piece_id = sample_result["piece_id"]  # [self.num_points]
        #     nps = sample_result["nps"]  # [P]
        #     mat_gt = sample_result["mat_gt"]
        #     return pcs, piece_id, nps, mat_gt, mean_edge_len # areas

        if self.pcs_sample_type == "boundary_mesh":
            sample_result, mean_edge_len = self.sample_point_byBoundaryMesh(meshes, self.num_points)
            pcs = sample_result["pcs"]  # [P,N,3]
            piece_id = sample_result["piece_id"]  # [self.num_points]
            nps = sample_result["nps"]  # [P]
            mat_gt = sample_result["mat_gt"]
            return pcs, piece_id, nps, mat_gt, mean_edge_len
        elif self.pcs_sample_type == "boundary_pcs":
            # 不会去采样固定数量个点
            sample_result, mean_edge_len = self.sample_point_byBoundaryPcs(meshes, self.num_points)
            pcs = sample_result["pcs"]  # [P,N,3]
            piece_id = sample_result["piece_id"]  # [self.num_points]
            nps = sample_result["nps"]  # [P]
            mat_gt = sample_result["mat_gt"]
            return pcs, piece_id, nps, mat_gt, mean_edge_len
        elif self.pcs_sample_type == "stitch":
            if self.num_points%2!=0:
                raise ValueError("self.num_points should be an even number when self.pcs_sample_type==\"stitch\"")

            if self.mode=="train": n_rings = 2
            else: n_rings = 1

            stitches = np.load(os.path.join(data_folder, "annotations", "stitch.npy"))
            # stitch_visualize(np.concatenate([np.array(mesh.vertices) for mesh in meshes], axis = 0), stitch)
            sample_result, mean_edge_len = self.sample_point_byStitch(meshes, self.num_points, stitches, n_rings=n_rings)

            pcs = sample_result["pcs"]  # [P,N,3]
            piece_id = sample_result["piece_id"]  # [self.num_points]
            nps = sample_result["nps"]  # [P]
            mat_gt = sample_result["mat_gt"]
            return pcs, piece_id, nps, mat_gt, mean_edge_len
        else:
            raise NotImplementedError(f"pcs_sample_type: {self.pcs_sample_type} hasen't been implemented")


    def __getitem__(self, index):
        mesh_file_path = self.data_list[index]
        pcs, piece_id, nps, mat_gt, mean_edge_len = self._get_pcs(mesh_file_path)
        # pointcloud_visualize(np.concatenate(pcs), colormap="tab10", colornum=2)
        num_parts = len(pcs)
        cur_pcs, cur_pcs_gt ,cur_quat, cur_trans= [], [], [], []

        points_end_idxs = np.cumsum(nps)
        if self.shuffle_points_after_sample:
            indices_list = []

        for idx, (pc, n_p) in enumerate(zip(pcs, nps)):
            if idx == 0: point_start_idx = 0
            else: point_start_idx = points_end_idxs[idx - 1]
            # point_end_idx = points_end_idxs[idx]

            pc_gt = pc.copy()
            if self.panel_noise_type == "default":
                pc, gt_trans, gt_quat = self._random_SRM_default(pc, mean_edge_len)
            elif  self.panel_noise_type == "partcombin":
                pc, gt_trans, gt_quat = self._random_MSR_partcombin(pc, pcs, idx, mean_edge_len)
            else:
                raise NotImplementedError("")

            # 将采样点打乱
            if self.shuffle_points_after_sample:
                pc, pc_gt, order_indices = self._shuffle_pc(pc, pc_gt)
                indices_list.append(order_indices+point_start_idx)

            cur_pcs.append(pc)
            cur_pcs_gt.append(pc_gt)
            cur_quat.append(gt_quat)
            cur_trans.append(gt_trans)

        # 按采样点打乱的顺序对缝合关系中的顶点index进行映射
        if mat_gt is not None and self.shuffle_points_after_sample:
            order_indices = np.concatenate(indices_list, axis=0)
            mat_gt = stitch_indices_order(mat_gt, order_indices)
            # pointcloud_and_stitch_visualize(np.concatenate(cur_pcs,axis=0),mat_gt)

        cur_pcs = np.concatenate(cur_pcs).astype(np.float32)  # [N_sum, 3]
        cur_pcs_gt = np.concatenate(cur_pcs_gt).astype(np.float32)  # [N_sum, 3]
        # pointcloud_visualize(cur_pcs)
        cur_quat = self._pad_data(np.stack(cur_quat, axis=0), self.max_num_part).astype(np.float32)  # [P, 4]
        cur_trans = self._pad_data(np.stack(cur_trans, axis=0), self.max_num_part).astype(np.float32)  # [P, 3]

        # pointcloud_and_stitch_visualize(cur_pcs,mat_gt)
        n_pcs = self._pad_data(np.array(nps), self.max_num_part).astype(np.int64)  # [P]
        valids = np.zeros(self.max_num_part, dtype=np.float32)
        valids[:num_parts] = 1.0

        try:
            garment_json_path = glob(os.path.join(mesh_file_path, "garment*.json"))[0]
            garment_json_path = garment_json_path if os.path.exists(garment_json_path) else ""
        except Exception:
            garment_json_path = ""
        annotations_json_path = os.path.join(mesh_file_path,"annotations.json")
        annotations_json_path = annotations_json_path if os.path.exists(annotations_json_path) else ""
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
        }
        if self.mode == "train":
            # pointcloud_visualize(cur_pcs)
            # pointcloud_visualize([cur_pcs[piece_id==i] for i in range(len(pcs))])
            # pointcloud_and_stitch_visualize(cur_pcs,mat_gt)

            # 对缝合点 和 不缝合的点 分别加噪声
            if self.use_stitch_noise:
                # stitch_noise_strength=1 在训练数据上inference结果更好
                # stitch_noise_strength=2 在董远数据上inference结果更好

                # === 对缝合点 按缝合线加远离噪声 ===
                # stitch_noise_strength = 1
                # 让噪声强度在1-2之间随机变化
                rand_param = random.random()
                stitch_noise_strength = 1. * rand_param + 2. * (1-rand_param)

                # mat_gt的shape=Nx2，保存的是每一对缝合点的index
                vec =  cur_pcs[mat_gt[:, 1]] - cur_pcs[mat_gt[:,0]]
                # vec 是每一对缝合点之间的向量
                vec = vec/(np.linalg.norm(vec, axis=1, keepdims=True) + 1e-6)
                # 在缝合的两边加的噪声
                noise1 = (vec * np.random.rand(len(mat_gt), 1)*0.7+0.3) * stitch_noise_strength * mean_edge_len * 1.2
                noise2 = (vec * np.random.rand(len(mat_gt), 1)*0.7+0.3) * stitch_noise_strength * mean_edge_len * 1.2
                # cur_pcs的shape为Mx3，是所有点的位置
                cur_pcs[mat_gt[:, 1]] += noise1
                cur_pcs[mat_gt[:, 0]] -= noise2

                # === 对不缝合的点加随机噪声 ===
                unstitch_mask = np.zeros(cur_pcs.shape[0])
                unstitch_mask[mat_gt[:, 0]] = 1
                unstitch_mask[mat_gt[:, 1]] = 1
                unstitch_mask = unstitch_mask == 0
                noise3 = (np.random.rand(np.sum(unstitch_mask), 3)*2.-1.)
                noise3 = noise3 / (np.linalg.norm(noise3, axis=1, keepdims=True) + 1e-6)
                noise3 = noise3 * stitch_noise_strength * mean_edge_len
                cur_pcs[unstitch_mask] += noise3*0.4

                # === 对缝合的点的两端加相同噪声（只加一点点） ===
                noise4 = (np.random.rand(len(mat_gt), 3)*2.-1.)
                noise4 = noise4 / (np.linalg.norm(noise4, axis=1, keepdims=True) + 1e-6)
                noise4 = noise4 * stitch_noise_strength * mean_edge_len * 0.2
                cur_pcs[mat_gt[:, 0]] += noise4
                cur_pcs[mat_gt[:, 1]] += noise4

                # # === 对缝合的点再加点噪声（只加一点点） ===
                # stitch_mask = ~unstitch_mask
                # noise5 = (np.random.rand(np.sum(stitch_mask), 3) * 2. - 1.)
                # noise5 = noise5 / (np.linalg.norm(noise5, axis=1, keepdims=True) + 1e-6)
                # noise5 = noise5 * stitch_noise_strength * mean_edge_len * 0.2
                # cur_pcs[stitch_mask] += noise5

            # pointcloud_and_stitch_visualize(cur_pcs, mat_gt)

            # GT 缝合关系
            mat_gt = stitch_indices2mat(self.num_points, mat_gt)
            data_dict["mat_gt"] = mat_gt

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
        data_dir=cfg.DATA.DATA_DIR,
        data_keys=cfg.DATA.DATA_KEYS,
        category=cfg.DATA.CATEGORY,
        num_points=cfg.DATA.NUM_PC_POINTS,
        min_num_part=cfg.DATA.MIN_NUM_PART,
        max_num_part=cfg.DATA.MAX_NUM_PART,

        shrink_mesh=cfg.DATA.SHRINK_MESH.TRAIN,
        shrink_mesh_param=cfg.DATA.SHRINK_MESH_PARAM.TRAIN,

        # shuffle_parts=cfg.DATA.SHUFFLE_PARTS,
        pcs_noise_strength = cfg.DATA.PCS_NOISE_STRENGTH,
        use_stitch_noise =  cfg.DATA.USE_STITCH_NOISE,

        panel_noise_type = cfg.DATA.PANEL_NOISE_TYPE.TRAIN,
        scale_range=cfg.DATA.SCALE_RANGE,
        rot_range=cfg.DATA.ROT_RANGE,
        trans_range=cfg.DATA.TRANS_RANGE,

        overfit=cfg.DATA.OVERFIT,
        pcs_sample_type=cfg.DATA.PCS_SAMPLE_TYPE.TRAIN,
        pcs_noise_type=cfg.DATA.PCS_NOISE_TYPE,
        min_part_point=cfg.DATA.MIN_PART_POINT,
        length=cfg.DATA.LENGTH * cfg.BATCH_SIZE,
        mode="train",
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
    data_dict["length"] = cfg.DATA.TEST_LENGTH
    data_dict["mode"] = "val"
    data_dict["pcs_sample_type"] = cfg.DATA.PCS_SAMPLE_TYPE.VAL
    data_dict["panel_noise_type"] = cfg.DATA.PANEL_NOISE_TYPE.VAL
    data_dict["shrink_mesh"] = cfg.DATA.SHRINK_MESH.VAL
    data_dict["shrink_mesh_param"] = cfg.DATA.SHRINK_MESH_PARAM.VAL
    val_set = AllPieceMatchingDataset_stylexd(**data_dict)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=1,  # [modified]
        # shuffle=cfg.DATA.SHUFFLE,  # 空数据+True导致在Docker里报错，因此暂时先不应用上去
        shuffle=False,
        num_workers=1,  # [modified]
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    return train_loader, val_loader

def build_stylexd_dataloader_test(cfg):
    # test用的是董远生成的不带l的obj文件，用边界点采样法
    data_dict = dict(
        data_dir=cfg.DATA.DATA_DIR,
        data_keys=cfg.DATA.DATA_KEYS,
        category=cfg.DATA.CATEGORY,

        num_points=cfg.DATA.NUM_PC_POINTS,
        min_num_part=cfg.DATA.MIN_NUM_PART,
        max_num_part=cfg.DATA.MAX_NUM_PART,

        shrink_mesh=cfg.DATA.SHRINK_MESH.TEST,
        shrink_mesh_param=cfg.DATA.SHRINK_MESH_PARAM.TEST,

        pcs_noise_strength=cfg.DATA.PCS_NOISE_STRENGTH,

        panel_noise_type=cfg.DATA.PANEL_NOISE_TYPE.TEST,
        scale_range=cfg.DATA.SCALE_RANGE,
        rot_range=cfg.DATA.ROT_RANGE,
        trans_range=cfg.DATA.TRANS_RANGE,
        overfit=cfg.DATA.OVERFIT,
        pcs_sample_type=cfg.DATA.PCS_SAMPLE_TYPE.TEST,
        pcs_noise_type=cfg.DATA.PCS_NOISE_TYPE,
        min_part_point=cfg.DATA.MIN_PART_POINT,
        length=cfg.DATA.LENGTH * cfg.BATCH_SIZE,
        mode="test",
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
import json
import os
import pickle
import random
from glob import glob
import numpy as np

import igl
import trimesh
import trimesh.sample

from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader


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
        if mode not in ["train","val","test"]:
            raise ValueError(f"mode=\"{mode}\" is not valid.")

        self.mode = mode
        self.data_dir = data_dir
        self.data_types = data_types
        self.data_list = self._read_data()
        if self.mode=="train":
            self.data_list = self.data_list[:10000]
        elif self.mode=="val":
            self.data_list = self.data_list[10000:]
        elif self.mode=="test":
            self.data_list = self.data_list

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
        # if 0 < length < len(self.data_list):
        #     self.length = length
        # else:
        #     self.length = len(self.data_list)


    def __len__(self):
        return self.length

    def _read_data(self):
        if self.mode in ["train", "val"]:
            mesh_dir = os.path.join(self.data_dir, self.mode)
            data_list = sorted(glob(os.path.join(mesh_dir, "garment_*")))
            return data_list
        # [todo] 根据数据类型读取不同的数据去inference
        if self.mode == "test":
            assert len(self.data_types)!=0, "self.data_types can't be empty in inference."
            self.data_types = list(dict.fromkeys(self.data_types))  # 去除重复
            data_list = []
            for type_name in self.data_types:
                mesh_dir = os.path.join(self.data_dir, self.mode)
                mesh_dir = os.path.join(mesh_dir, type_name)
                assert os.path.exists(mesh_dir), f"No data folder corresponding to data_types:{self.data_types}"
                data_list.extend(sorted(glob(os.path.join(mesh_dir, "garment_*"))))
            return data_list


    # 用于训练inference董远数据的模型
    # random scale+rotate+move panel (default)
    def _random_SRM_default(self, pc, mean_edge_len):
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
        scale_gt = (np.random.rand(1) * self.scale_range) * noise_strength_S + 0.95
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
        trans_gt = (mean_edge_len * (np.random.rand(3) - 0.5) * 2.0 * self.trans_range).reshape(-1,3) * noise_strength_M
        pc = pc + trans_gt
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
            mean_edge_len=mean_edge_len
        )
        return sample_result

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
            mean_edge_len=mean_edge_len
        )
        return sample_result


    def sample_point_byStitch(self, meshes, num_points, stitches, full_uv_info, n_rings = 2):
        pcs, nps = [], [len(mesh.vertices) for mesh in meshes]
        piece_id = np.concatenate([[idx]*i for idx,i in enumerate(nps)], axis=-1)

        # 所有的顶点、面、边界点
        vertices = np.concatenate([np.array(mesh.vertices) for mesh in meshes], axis=0)
        faces, boundary_point_list = [], []

        # PART 1 -------------------------------------------------------------------------------------------------------
        # === 将点和面的index转换为全局index ===
        points_end_idxs = np.cumsum(nps)
        for idx, mesh in enumerate(meshes):
            if idx ==0 : point_start_idx = 0
            else: point_start_idx = points_end_idxs[idx-1]
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
        # [modified]
        # boundary_len_per_piece = np.array([np.sum(piece_id[boundary_points_idx]==i) for i in range(len(nps))])
        sample_stitch = LatinHypercubeSample(len(stitch_verties_idx), stitched_sample_num)
        stitched_sample_idx = stitch_verties_idx[sample_stitch]
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

        # PART 4 -------------------------------------------------------------------------------------------------------
        # 将点云大小按最BBOX的最长边归一化到[-0.5 - 0.5]，计算平均边长

        points_normalized, normalize_range = min_max_normalize(sampled_points_CH)
        mean_edge_len = cal_mean_edge_len(meshes)/normalize_range
        # pointcloud_visualize(points_normalized)

        # PART 5 -------------------------------------------------------------------------------------------------------
        # 让采样的缝合关系满足：在index上接近的缝合关系，在位置上也尽可能接近

        # 将采样点按index进行排序，这样同一个part的会凑到一块去
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
            mean_edge_len=mean_edge_len
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
            raise ValueError

        meshes = [trimesh.load(mesh_file, force="mesh", process=False) for mesh_file in mesh_files]

        return meshes

    def load_full_uv_info(self, data_folder):
        if self.read_uv:
            full_uv_info = np.load(os.path.join(data_folder, "annotations", "uv.npy"))
            return full_uv_info
        else:
            return None

    def shrink_meshes(self, meshes, shrink_param = 0.5):
        # 保存结果
        # meshes_visualize(meshes, "before_shrink")

        if shrink_param>1 or shrink_param<0:
            raise ValueError(f"shrink_param={shrink_param} is invalid")

        for mesh in meshes:
            # if len(mesh.vertices)<400: continue
            # # 对每个边缘点，找其不是边缘点的相邻点 ----------------------------------------------------------------------------
            # edge_points_loops = igl.all_boundary_loop(mesh.faces)  # 获取边界上的点
            # all_boundary_points = np.concatenate(edge_points_loops)
            # neighbor_points = {}
            # for loop in edge_points_loops:
            #
            #     for point in loop:
            #         # 对最外面的每个边界点，找到里面一圈的点钟与其相邻的点
            #         neighbors = mesh.vertex_neighbors[point]
            #         if len(neighbors) >= 3:
            #             for n_p in neighbors:
            #                 if n_p in all_boundary_points:
            #                     neighbors.remove(n_p)
            #         # 如果有一个边界点的相邻点数量小于3，代表这个点的相邻点都是边界点
            #         else:
            #             pass
            #         neighbor_points[point] = neighbors
            #
            # # 对每个边界点，和neighbor_points中的点，计算shrink后的位置 -------------------------------------------------------
            # # 先计算新位置
            # new_vertices_positions = {}
            # for b_point in all_boundary_points:
            #     # 将边缘点往内部收缩
            #     new_position = (shrink_param * mesh.vertices[b_point] +
            #                      (1-shrink_param) * np.mean(mesh.vertices[neighbor_points[b_point]],axis=-2))
            #     new_vertices_positions[b_point] = new_position
            # # 再应用到mesh里
            # for b_point in all_boundary_points:
            #     mesh.vertices[b_point] = new_vertices_positions[b_point]
            if len(mesh.vertices)<400: continue
            # 对每个边缘点，找其不是边缘点的相邻点 ----------------------------------------------------------------------------
            edge_points_loops = igl.all_boundary_loop(mesh.faces)  # 获取边界上的点
            all_boundary_points = np.concatenate(edge_points_loops)
            neighbor_points = {}
            for loop in edge_points_loops:
                for point in loop:
                    # 对最外面的每个边界点，找到与其相邻的点
                    neighbors = mesh.vertex_neighbors[point]
                    neighbor_points[point] = neighbors

            # 对每个边界点，和neighbor_points中的点，计算shrink后的位置 -------------------------------------------------------
            # 先计算新位置
            new_vertices_positions = {}
            for b_point in all_boundary_points:
                # 将边缘点往内部收缩
                new_position = (shrink_param * mesh.vertices[b_point] +
                                 (1-shrink_param) * get_pc_bbox(mesh.vertices[neighbor_points[b_point]], type="ccwh")[0])
                new_vertices_positions[b_point] = new_position
            # 再应用到mesh里
            for b_point in all_boundary_points:
                mesh.vertices[b_point] = new_vertices_positions[b_point]
        # 保存结果
        # meshes_visualize(meshes, "after_shrink")
        a=1

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
            if self.mode=="train": n_rings = 2
            else: n_rings = 1
            stitches = np.load(os.path.join(data_folder, "annotations", "stitch.npy"))
            # stitch_visualize(np.concatenate([np.array(mesh.vertices) for mesh in meshes], axis = 0), stitch)
            # [todo] 将来如果有空，试着从根本上解决这个问题
            max_check_times = 4  # 最大重复采样次数
            for i in range(max_check_times):
                sample_result = self.sample_point_byStitch(meshes, self.num_points, stitches, full_uv_info, n_rings=n_rings,)
                # 如果没有未被采样的panel
                if sum([len(pc)==0 for pc in sample_result["pcs"] ]) == 0: break
                else:
                    if i!=max_check_times-1: continue  # 如果还有继续调用次数，则重新采样一遍
                    else: raise ValueError("NUM_PC_POINTS too small in cfg ")  # 如果超过最大调用次数
            return sample_result
        else:
            raise NotImplementedError(f"pcs_sample_type: {self.pcs_sample_type} hasen't been implemented")


    def __getitem__(self, index):
        # === 获取所需路径 ===
        mesh_file_path = self.data_list[index]
        try:
            garment_json_path = glob(os.path.join(mesh_file_path, "annotations", "garment*.json"))[0]
            garment_json_path = garment_json_path if os.path.exists(garment_json_path) else ""
        except Exception:
            garment_json_path = ""

        annotations_json_path = os.path.join(mesh_file_path, "annotations", "annotations.json")
        annotations_json_path = annotations_json_path if os.path.exists(annotations_json_path) else ""

        # === 获取点云以及标注信息 ===
        sample_result= self._get_pcs(mesh_file_path)
        pcs = sample_result["pcs"]
        nps = sample_result["nps"]
        mat_gt = sample_result["mat_gt"]
        piece_id = sample_result["piece_id"]
        mean_edge_len = sample_result["mean_edge_len"]

        # 对uv进行和pc相同的normalize
        if "uv" in sample_result.keys():
            uv = sample_result["uv"]
            uv = min_max_normalize(uv)[0]
        else: uv=None

        # pointcloud_visualize(np.concatenate(pcs), colormap="tab10", colornum=2)
        num_parts = len(pcs)

        # === 在位置变化前，先加一次很小的缝合噪声 ===
        if self.use_stitch_noise:
            # pointcloud_visualize(pcs)
            # pointcloud_visualize([pcs[piece_id==i] for i in range(len(pcs))])
            # pointcloud_and_stitch_visualize(pcs, mat_gt)

            # stitch_noise_strength=1 在训练数据上inference结果更好
            # stitch_noise_strength=2 在董远数据上inference结果更好

            # === 对缝合点 按缝合线加远离噪声 ===
            all_pcs = np.concatenate(pcs)
            # 随机噪声强度
            rand_param = random.random()
            stitch_noise_strength_base = 0.05 * rand_param + 0.1 * (1 - rand_param)
            stitch_noise_strength_base *= self.stitch_noise_strength

            # mat_gt的shape=Nx2，保存的是每一对缝合点的index
            vec = all_pcs[mat_gt[:, 1]] - all_pcs[mat_gt[:, 0]]
            # vec 是每一对缝合点之间的向量
            vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-6)
            # 在缝合的两边加的噪声
            noise1 = (vec * np.random.rand(len(mat_gt), 1) * 0.7 + 0.3) * stitch_noise_strength_base * mean_edge_len * 1.2
            noise2 = (vec * np.random.rand(len(mat_gt), 1) * 0.7 + 0.3) * stitch_noise_strength_base * mean_edge_len * 1.2
            # cur_pcs的shape为Mx3，是所有点的位置
            all_pcs[mat_gt[:, 1]] += noise1
            all_pcs[mat_gt[:, 0]] -= noise2

            nps_cumsum = np.cumsum(nps)
            for idx in range(len(pcs)):
                if idx == 0:
                    point_start_idx = 0
                else:
                    point_start_idx = nps_cumsum[idx - 1]
                point_end_idx = nps_cumsum[idx]
                pcs[idx] = all_pcs[point_start_idx:point_end_idx]

        # pointcloud_visualize(all_pcs)
        # pointcloud_visualize([all_pcs[piece_id==i] for i in range(len(all_pcs))])
        # pointcloud_and_stitch_visualize(all_pcs, mat_gt)

        cur_pcs, cur_pcs_gt ,cur_quat, cur_trans= [], [], [], []

        points_end_idxs = np.cumsum(nps)

        for idx, (pc, n_p) in enumerate(zip(pcs, nps)):
            if idx == 0: point_start_idx = 0
            else: point_start_idx = points_end_idxs[idx - 1]
            # point_end_idx = points_end_idxs[idx]

            pc_gt = pc.copy()
            if self.panel_noise_type == "default":
                pc, gt_trans, gt_quat = self._random_SRM_default(pc, mean_edge_len)
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
        }
        if uv is not None:
            # 添加一列零，不然pointnet好像提取不了特征
            uv = np.hstack((uv, np.zeros((uv.shape[0], 1))))
            data_dict["uv"] = uv

        if self.mode == "train":
            # pointcloud_visualize(cur_pcs)
            # pointcloud_visualize([cur_pcs[piece_id==i] for i in range(len(pcs))])
            # pointcloud_and_stitch_visualize(cur_pcs, mat_gt)
            pass
            # === 在位置变化后，加一次缝合噪声 ===
            if self.use_stitch_noise:
                # 随机噪声强度
                rand_param = random.random()
                stitch_noise_strength_base = (self.stitch_noise_random_min * rand_param +
                                              self.stitch_noise_random_max * (1-rand_param))
                stitch_noise_strength_base *= self.stitch_noise_strength

                # === 对缝合点 按缝合线加远离噪声 ===
                stitch_noise_strength1 = stitch_noise_strength_base
                vec =  cur_pcs[mat_gt[:, 1]] - cur_pcs[mat_gt[:,0]]
                # 是每一对缝合点之间的向量
                vec = vec/(np.linalg.norm(vec, axis=1, keepdims=True) + 1e-6)
                # 在缝合的两边加的噪声
                noise1 = (vec * np.random.rand(len(mat_gt), 1)*0.7+0.3) * stitch_noise_strength1 * mean_edge_len * 1.2
                noise2 = (vec * np.random.rand(len(mat_gt), 1)*0.7+0.3) * stitch_noise_strength1 * mean_edge_len * 1.2
                # cur_pcs的shape为Mx3，是所有点的位置
                cur_pcs[mat_gt[:, 1]] += noise1
                cur_pcs[mat_gt[:, 0]] -= noise2

                # === 对不缝合的点加随机噪声 ===
                stitch_noise_strength3 = stitch_noise_strength_base * (random.random()*0.4+0.2)
                unstitch_mask = np.zeros(cur_pcs.shape[0])
                unstitch_mask[mat_gt[:, 0]] = 1
                unstitch_mask[mat_gt[:, 1]] = 1
                unstitch_mask = unstitch_mask == 0
                noise3 = (np.random.rand(np.sum(unstitch_mask), 3)*2.-1.)
                noise3 = noise3 / (np.linalg.norm(noise3, axis=1, keepdims=True) + 1e-6)
                noise3 = noise3 * stitch_noise_strength3 * mean_edge_len
                cur_pcs[unstitch_mask] += noise3

                # === 对缝合的点的两端加相同噪声（只加一点点） ===
                stitch_noise_strength4 = stitch_noise_strength_base * (random.random()*0.2+0.1)
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
        batch_size=1,  # [modified]
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

        read_uv=cfg.MODEL.USE_UV_FEATURE,
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
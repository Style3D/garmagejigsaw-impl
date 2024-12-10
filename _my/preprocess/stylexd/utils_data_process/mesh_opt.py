# 用于解析 StyleXD中的单个obj文件
import os.path

import numpy as np
import trimesh
from trimesh import Trimesh
import igl
from igl import boundary_loop

# def split_mesh_into_parts(obj_dict):
#     nps = obj_dict["nps"]
#     nfs = obj_dict["nfs"]
#     vertices = obj_dict["vertices"]
#     faces = obj_dict["faces"]
#
#     meshes = []
#     num_parts = len(nps)
#     end_point_idx = np.cumsum(nps)
#     enc_face_idx = np.cumsum(nfs)
#     for idx in range(num_parts):
#         if idx == 0:
#             point_start = 0
#             face_start = 0
#         else:
#             point_start = end_point_idx[idx - 1]
#             face_start = enc_face_idx[idx - 1]
#         point_end = end_point_idx[idx]
#         face_end = enc_face_idx[idx]
#
#         mesh = Trimesh(vertices=vertices, faces=faces)
#         face_to_ex = list(range(face_start, face_end))
#         mesh_splited = mesh.submesh([face_to_ex], append=True)
#         meshes.append(mesh_splited)
#
#     # # 每个mesh边界点的idx
#     # boundary_points_idx = []
#     # for idx, mesh in enumerate(meshes):
#     #     boundary_points_idx.append(boundary_loop(mesh.faces))
#     #
#     # boundary_points = np.concatenate([mesh.vertices[boundary_points_idx[idx]] for idx, mesh in enumerate(meshes)],
#     #                                  axis=0)
#
#     return meshes


def split_mesh_into_parts(obj_dict):
    nps = np.array(obj_dict["nps"])
    nfs = np.array(obj_dict["nfs"])
    vertices = np.array(obj_dict["vertices"])
    faces = np.array(obj_dict["faces"])

    meshes = []
    num_parts = len(nps)
    end_point_idx = np.cumsum(nps)
    end_face_idx = np.cumsum(nfs)

    sum=0

    for idx in range(num_parts):
        if idx == 0:
            point_start = 0
            face_start = 0
        else:
            point_start = end_point_idx[idx - 1]
            face_start = end_face_idx[idx - 1]
        point_end = end_point_idx[idx]
        face_end = end_face_idx[idx]

        Sub_vertices_idx = list(range(point_start, point_end))
        Sub_faces_idx = list(range(face_start, face_end))
        Sub_vertices, Sub_faces = get_sub_mesh(vertices, faces, Sub_vertices_idx, Sub_faces_idx)

        mesh = Trimesh(vertices=Sub_vertices, faces=Sub_faces, process=False)
        meshes.append(mesh)

    # # 每个mesh边界点的idx
    # boundary_points_idx = []
    # for idx, mesh in enumerate(meshes):
    #     boundary_points_idx.append(boundary_loop(mesh.faces))
    #
    # boundary_points = np.concatenate([mesh.vertices[boundary_points_idx[idx]] for idx, mesh in enumerate(meshes)],
    #                                  axis=0)

    return meshes


def get_sub_mesh(Vertices:np.array, Faces:np.array, Sub_vertices_idx:np.array, Sub_faces_idx:np.array):
    Sub_faces = Faces[Sub_faces_idx]
    # 步骤1：创建 Sub_vertices 中的顶点的索引映射
    # 映射：原来的顶点索引 -> 在新顶点数组 L 中的索引
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(Sub_vertices_idx)}

    # 步骤2：创建新的顶点数组 L
    Sub_vertices = Vertices[Sub_vertices_idx]

    # 步骤3：将 Sub_faces 的索引更新为相对于 L 的索引
    Sub_faces = np.array([[index_map[vert] for vert in face] for face in Sub_faces])

    return Sub_vertices, Sub_faces


if __name__ == "__main__":
    Vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])

    Faces = np.array([[0, 1, 2], [2, 1, 3], [4, 5, 6], [6, 5, 7]])

    Sub_vertices_idx = np.array([0, 1, 2, 3])  # 选择部分顶点

    Sub_faces_idx = np.array([0, 1])  # 选择部分面

    Sub_vertices, Sub_faces = get_sub_mesh(Vertices, Faces, Sub_vertices_idx, Sub_faces_idx)

    print("Sub_vertices:")
    print(Sub_vertices)
    print("\nnew_Sub_faces:")
    print(Sub_faces)
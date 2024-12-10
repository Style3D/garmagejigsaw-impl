import numpy as np
import open3d as o3d
import torch


def square_distance(src, dst, normalised=False):
    """
    Calculate Euclid distance between each two points.
    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    if isinstance(src, torch.Tensor):
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        if normalised:
            dist += 2
        else:
            dist += torch.sum(src ** 2, dim=-1)[:, :, None]
            dist += torch.sum(dst ** 2, dim=-1)[:, None, :]

        dist = torch.clamp(dist, min=1e-12, max=None)
    elif isinstance(src, np.ndarray):
        dist = -2 * np.matmul(src, dst.transpose(0, 2, 1))
        if normalised:
            dist += 2
        else:
            dist += np.sum(src ** 2, axis=-1)[:, :, None]
            dist += np.sum(dst ** 2, axis=-1)[:, None, :]
        dist = dist.clip(1e-12, None)
    else:
        raise NotImplementedError(
            f"square distance not implemented for src type {type(src)}"
        )
    return dist


def to_array(tensor):
    """
    Conver tensor to array
    """
    if not isinstance(tensor, np.ndarray):
        if tensor.device == torch.device("cpu"):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor



def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
    return pcd


def to_o3d_feats(embedding):
    """
    Convert tensor/array to open3d features
    embedding:  [N, 3]
    """
    feats = o3d.pipelines.registration.Feature()
    feats.data = to_array(embedding).T
    return feats


def to_tensor(array):
    """
    Convert array to tensor
    """
    if not isinstance(array, torch.Tensor):
        return torch.from_numpy(array).float()
    else:
        return array


def mutual_selection(score_mat):
    """
    Return a {0,1} matrix, the element is 1 if and only if it's maximum along both row and column

    Args: np.array()
        score_mat:  [B,N,N]
    Return:
        mutuals:    [B,N,N]
    """
    score_mat = to_array(score_mat)
    if score_mat.ndim == 2:
        score_mat = score_mat[None, :, :]

    mutuals = np.zeros_like(score_mat)
    for i in range(score_mat.shape[0]):  # loop through the batch
        c_mat = score_mat[i]
        flag_row = np.zeros_like(c_mat)
        flag_column = np.zeros_like(c_mat)

        max_along_row = np.argmax(c_mat, 1)[:, None]
        max_along_column = np.argmax(c_mat, 0)[None, :]
        np.put_along_axis(flag_row, max_along_row, 1, 1)
        np.put_along_axis(flag_column, max_along_column, 1, 0)
        mutuals[i] = (flag_row.astype(np.bool)) & (flag_column.astype(np.bool))
    return mutuals.astype(np.bool)


# from utils.visualization.pointcloud_visualize import pointcloud_visualize
def min_max_normalize(points):
    """
    对 Nx3 点进行 Min-Max 归一化，使得点的坐标在 [-1, 1] 范围内。

    Args:
        points (ndarray): Nx3 的点集。

    Returns:
        ndarray: 归一化后的 Nx3 点集。
    """
    # 计算每一列的最小值和最大值
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # 移动到中心
    points = points - (max_vals + min_vals)/2

    # 计算每一列的最小值和最大值
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # 防止除零
    ranges = np.array([1]*points.shape[-1]) * np.max(max_vals - min_vals)
    ranges[ranges == 0] = 1

    # 归一化到 [-0.5, 0.5]
    normalized_points = points/ranges

    return normalized_points, ranges[0]


def get_pc_bbox(pc: np.array, type: object = "ccwh") -> object:
    if type not in ["xyxy", "ccwh", "xywh"]:
        raise ValueError()

    if isinstance(pc, np.ndarray):
        if type == "ccwh":
            max_ = np.array([np.max(pc[:,ax]) for ax in range(3)])
            min_ = np.array([np.min(pc[:,ax]) for ax in range(3)])
            center = (max_+min_)/2
            scale = max_-min_
            result = [center, scale]
        else:
            raise NotImplementedError
        return result
    elif isinstance(pc, torch.Tensor):
        if type == "ccwh":
            max_ = torch.tensor([torch.max(pc[:,ax]) for ax in range(3)], device=pc.device)
            min_ = torch.tensor([torch.min(pc[:,ax]) for ax in range(3)], device=pc.device)
            center = (max_+min_)/2
            scale = max_-min_
            result = [center, scale]
        else:
            raise NotImplementedError
        return result
    else:
        raise NotImplementedError

def pc_rescale(pcs, scale):
    center = torch.tensor(get_pc_bbox(pcs)[0]).unsqueeze(0)
    pcs_u = pcs - center
    pcs_u *= scale
    pcs_u += center
    return pcs_u
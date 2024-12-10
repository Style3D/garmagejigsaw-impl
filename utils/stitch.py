# Transfer the data format of stitch.

import numpy as np
import torch


def stitch_indices2mat(num_points, indices):
    if isinstance(indices, torch.Tensor):
        matrix = torch.zeros((num_points, num_points), dtype=torch.int, device=indices.device)
        matrix[indices[:, 0], indices[:, 1]] = 1
        return matrix
    elif isinstance(indices, np.ndarray):
        matrix = np.zeros((num_points, num_points), dtype=int)
        matrix[indices[:, 0], indices[:, 1]] = 1
        return matrix


def stitch_mat2indices(matrix):
    if isinstance(matrix, torch.Tensor):
        indices = torch.nonzero(matrix[0])
    elif isinstance(matrix, np.ndarray):
        rows, cols = np.nonzero(matrix)
        indices = np.vstack((rows, cols)).T
    else:
        raise TypeError
    return indices

def stitch_indices_order(stitch_indices, order_indices):
    """
    Sort the stitch_indices by order (high efficiency)

    :param stitch_indices:
    :param order_indices:
    :return:
    """
    mapping = np.zeros(len(order_indices), dtype=int)
    mapping[order_indices] = np.arange(len(order_indices))
    stitch_indices = mapping[stitch_indices]
    return stitch_indices

def stitch_mat_order(matrix, order_indices):
    """
    Sort the stitch_mat by order (low efficiency)

    :param matrix:
    :param order_indices:
    :return:
    """
    matrix_order = matrix[np.ix_(order_indices, order_indices)]
    return matrix_order

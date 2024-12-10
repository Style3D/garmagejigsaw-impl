import copy
import itertools
import pickle

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from torch.nn import LayerNorm, GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm


def array_equal(a, b):
    """Compare if two arrays are the same.

    Args:
        a/b: can be np.ndarray or torch.Tensor.
    """
    if a.shape != b.shape:
        return False
    try:
        assert (a == b).all()
        return True
    except:
        return False


def array_in_list(array, lst):
    """Judge whether an array is in a list."""
    for v in lst:
        if array_equal(array, v):
            return True
    return False


def filter_wd_parameters(model, skip_list=()):
    """Create parameter groups for optimizer.

    We do two things:
        - filter out params that do not require grad
        - exclude bias and Norm layers
    """
    # we need to sort the names so that we can save/load ckps
    w_name, b_name, no_decay_name = [], [], []
    for name, m in model.named_modules():
        # exclude norm weight
        if isinstance(m, (LayerNorm, GroupNorm, _BatchNorm, _InstanceNorm)):
            w_name.append(name)
        # exclude bias
        if hasattr(m, "bias") and m.bias is not None:
            b_name.append(name)
        if name in skip_list:
            no_decay_name.append(name)
    w_name.sort()
    b_name.sort()
    no_decay_name.sort()
    no_decay = [model.get_submodule(m).weight for m in w_name] + [
        model.get_submodule(m).bias for m in b_name
    ]
    for name in no_decay_name:
        no_decay += [
            p
            for p in model.get_submodule(m).parameters()
            if p.requires_grad and not array_in_list(p, no_decay)
        ]

    decay_name = []
    for name, param in model.named_parameters():
        if param.requires_grad and not array_in_list(param, no_decay):
            decay_name.append(name)
    decay_name.sort()
    decay = [model.get_parameter(name) for name in decay_name]
    return {"decay": list(decay), "no_decay": list(no_decay)}


def get_batch_length_from_part_points(n_pcs, n_valids=None, part_valids=None):
    """
    :param n_pcs: [B, P] number of points per batch
    :param n_valids: [B] number of parts per batch
    :param part_valids: [B, P] 0/1
    :return: batch_length [\sum n_valids, ]
    """
    B, P = n_pcs.shape
    if n_valids is None:
        if part_valids is None:
            n_valids = torch.ones(B, device=n_pcs, dtype=torch.long) * P
        else:
            n_valids = torch.sum(part_valids, dim=1).to(torch.long)

    batch_length_list = []
    for b in range(B):
        batch_length_list.append(n_pcs[b, : n_valids[b]])
    batch_length = torch.cat(batch_length_list)
    assert batch_length.shape[0] == torch.sum(n_valids)
    return batch_length

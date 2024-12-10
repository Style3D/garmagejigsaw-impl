import math
import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter


def build_pc_classifier(dim):
    affinity_layer = nn.Sequential(
        nn.BatchNorm1d(dim),
        nn.ReLU(inplace=True),
        nn.Conv1d(dim, 1, 1),
    )
    return affinity_layer

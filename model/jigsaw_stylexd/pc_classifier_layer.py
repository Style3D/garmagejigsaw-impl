
from torch import nn


def build_pc_classifier(dim):
    affinity_layer = nn.Sequential(
        nn.BatchNorm1d(dim),
        nn.ReLU(inplace=True),
        nn.Conv1d(dim, 1, 1),
    )
    return affinity_layer

"""
如果后续需要改进，可参考PTv1的点分类方法
    self.cls = nn.Sequential(
        nn.Linear(planes[4], 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(128, num_classes),
    )
"""
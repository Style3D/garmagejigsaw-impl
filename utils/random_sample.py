from scipy.stats.qmc import LatinHypercube
# [todo] 下面调用这玩意有问题，换成正常的
from wandb.sdk.internal.internal import logger

import numpy as np
from scipy.spatial import ConvexHull

def LatinHypercubeSample(sample_size, n_samples):
    """
    Latin Hypercube sample.

    :param sample_size:
    :param n_samples:
    :return:
    """
    sampler = LatinHypercube(d=1)  # 创建一个 Latin Hypercube 采样器，维度为1
    sample = sampler.random(n=n_samples)
    sample_idx = (sample * sample_size).astype(int).flatten()
    return sample_idx

# def balancedSample(sample_size, n_samples, iteration=20):
#     """
#     一种较为均衡的采样方式，能尽可能减少重复的采样
#     对于采样结果，我们会进行迭代式的优化
#     :param sample_size:
#     :param n_samples:
#     :return:
#     """
#     sample_list = []
#     sample_remain = n_samples
#     while sample_remain > 0:
#         sample_num = min(sample_remain, sample_size)
#         sample = np.arange(sample_size)
#         np.random.shuffle(sample)
#         sample = sample[:sample_num]
#         if len(sample) != sample_size:
#             iteration = iteration
#             for i in range(iteration):
#                 sample = np.sort(sample)
#                 for i in range(len(sample)):
#                     pre, cur, next  = i-1, i, i+1
#                     if pre < 0:
#                         pre = len(sample) - 1
#                     if next == len(sample):
#                         next = 0
#
#                     dis_pre = sample[cur] - sample[pre]
#                     if dis_pre < 0:
#                         dis_pre = sample_size + dis_pre
#                     dis_next = sample[next] - sample[cur]
#                     if dis_next < 0:
#                         dis_next = sample_size + dis_next
#
#                     if dis_pre - dis_next > 0:
#                         sample[i] -= 1
#                         if sample[i] < 0:
#                             sample[i] = sample_size-1
#                     elif dis_next - dis_pre > 0:
#                         sample[i] += 1
#                         if sample[i] == sample_size:
#                             sample[i] = 0
#                 a=1
#
#         sample_list.append(sample)
#         sample_remain -= sample_num
#     sample_idx = np.concatenate(sample_list)
#     np.random.shuffle(sample_idx)
#     return sample_idx

def balancedSample(sample_size, n_samples, iteration=20):
    samples = np.linspace(0, sample_size - 1, n_samples)
    samples = np.round(samples).astype(int)  # 四舍五入为整数
    # np.random.shuffle(samples)  # 可选：打乱顺序
    return samples

def random_point_in_convex_hull(points):
    # print(points)
    """
    get a random point in the convex_hull of a pointcloud.

    :param points: pointcloud
    :return:
    """
    # Generate a random point inside the convex hull
    if len(points) == 1:
        return points[0]
    elif len(points) == 2:
        return (points[0] + points[1]) / 2
    elif len(points) == 3:
        barycentric_coords = np.random.dirichlet(np.ones(3))
        return np.dot(barycentric_coords, points)
    else:
        try:
            # 加噪声可以有效防止所有点挤在一个面上
            pts = points + np.random.normal(-1e-8, 1e-8, size=points.shape)
            hull = ConvexHull(pts)
        except:
            logger.warning("很不幸，加的噪声让点处于了一个平面上")
            # 如果很不幸，加的噪声让点处于了一个平面上，那就再加一回
            pts = points + np.random.normal(-1e-8, 1e-8, size=points.shape)
            hull = ConvexHull(pts)
        simplices = hull.simplices
        # Choose a random simplex (face of the convex hull)
        simplex = simplices[np.random.choice(simplices.shape[0])]
        # Generate a random point in the simplex
        barycentric_coords = np.random.dirichlet(np.ones(len(simplex)))
        return np.dot(barycentric_coords, pts[simplex])

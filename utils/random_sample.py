from scipy.stats.qmc import LatinHypercube
from wandb.sdk.internal.internal import logger


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


import numpy as np
from scipy.spatial import ConvexHull, Delaunay

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

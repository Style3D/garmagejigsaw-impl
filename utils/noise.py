# [modified] 自己写的
import numpy as np

from trimesh import Trimesh

#
def get_sphere_noise(n_points, radius=1.0):
    """
    生成一个球形范围内的随机噪声

    :param n_points: 噪声采样数
    :param radius: 球半径
    :return:(n_points X 3)
    """
    # 随机生成点的半径，均匀分布
    r = np.random.uniform(0, radius, n_points) # 均匀分布在体积内
    # 随机生成点的角度，球坐标系
    theta = np.random.uniform(0, 2 * np.pi, n_points)  # Azimuthal angle
    phi = np.random.uniform(0, np.pi, n_points)  # Polar angle

    # 转换为笛卡尔坐标系
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    return np.vstack((x, y, z)).T

if __name__ == "__main__":
    A=get_sphere_noise(1000)
    m = Trimesh(vertices=A)
    m.export("../_tmp/m.obj")
# 新增功能：composite_visualize 在一个页面中显示多个内容，并能够保存为html文件
import math

import torch
from torchvision.utils import make_grid
import numpy as np

import plotly.express as px
from plotly.subplots import make_subplots



def geoimg_visualize(geo, mask):
    """
    :param batch:       dataset output with 1 batch_size
    :param inf_rst:     inference result of model
    :return:
    """

    # === fig的布局设置 ===
    fig = make_subplots(
        rows=1, cols=1, subplot_titles=("Geometry Image"),
        specs=[
            [{"type": "xy", "colspan": 1, "rowspan": 1}]
        ],
        vertical_spacing=0.02,  # 设置垂直间距
        horizontal_spacing=0.02  # 设置水平间距
    )

    num_parts = len(geo)
    grid_imgs = make_grid(torch.cat([torch.FloatTensor((geo.transpose(0,3,1,2) + 1.0) * 0.5), torch.FloatTensor(mask.transpose(0,3,1,2))], dim=1),
                          nrow=math.ceil(math.sqrt(num_parts)), ncol=math.ceil(math.sqrt(num_parts)), padding=5)
    grid_imgs = grid_imgs.permute(1, 2, 0).cpu().numpy()
    grid_imgs = np.concatenate([grid_imgs[:, :, :3], np.repeat(grid_imgs[:, :, -1:], 3, axis=-1)], axis=0)
    fig.add_trace(px.imshow(grid_imgs).data[0], row=1, col=1)


    # fig的设置 ------------------------------------------------------------------------------------
    fig.update_layout(
        height=1400, width=2400, title_text="Garment",
    )
    fig.show()
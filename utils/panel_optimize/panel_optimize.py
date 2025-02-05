import torch
from torch import nn
from chamferdist import ChamferDistance
from utils import pointcloud_visualize, pc_rescale


"""
Optimize the Scale and Transformation of Panels
"""
class STModel(nn.Module):
    def __init__(self, panel_num):
        super().__init__()
        self.panel_s = nn.Parameter(torch.ones((1,panel_num,3), dtype=torch.float32))
        self.panel_t = nn.Parameter(torch.zeros((1,panel_num,3), dtype=torch.float32))

def loss_function(loss_func,pcs_cur,pcs_bkg,target_dis):
    loss = loss_func(pcs_cur.unsqueeze(0), pcs_bkg.unsqueeze(0), bidirectional=False)
    loss = torch.abs(loss-target_dis)
    return loss

def get_optimize_ST(pcs_list, options = ('s','t')):
    if options is None or len(options)==0:
        raise ValueError("options empty.")
    options = [e.lower() for e in options]

    loss_func = ChamferDistance()
    w_loss = 1.
    max_iter_t = 100
    max_iter_s = 120
    target_dis = 0.

    st_model = STModel(len(pcs_list))
    st_model = st_model.cuda().train()
    optimizer_st = torch.optim.AdamW(
        st_model.parameters(),
        lr=5e-4,
        betas=(0.95, 0.99),
        weight_decay=1e-6,
        eps=1e-08,
    )

    for it in range(max(max_iter_s, max_iter_t)):
        # optimizer_st.lr = 2e-3 * (math.sin((1 - it / max_iter) * math.pi / 2))
        panel_loss = 0
        panel_s, panel_t = st_model.panel_s[0], st_model.panel_t[0]

        # 应用位置变化，计算位置变化后损失
        pcs_updated = []
        for idx, pcs in enumerate(pcs_list):
            offset_s = panel_s[idx].unsqueeze(0)
            offset_t = panel_t[idx].unsqueeze(0)
            if it>=max_iter_s: offset_s = offset_s.detach()
            elif it>=max_iter_t: offset_t = offset_t.detach()
            pcs_u = pcs + offset_t
            pcs_u = pc_rescale(pcs_u, offset_s)
            pcs_updated.append(pcs_u)  # 更新点云位置

        for idx, pcs in enumerate(pcs_updated):
            pcs_cur = pcs_updated[idx]
            pcs_bkg =  torch.concat([pcs for i, pcs in enumerate(pcs_updated) if i != idx],dim=-2)
            panel_loss += loss_function(loss_func,pcs_cur,pcs_bkg,target_dis)
        panel_loss /= len(pcs_updated)  # 计算平均损失

        optimizer_st.zero_grad()  # 清零梯度
        loss_total = w_loss * panel_loss
        loss_total.backward()  # 反向传播损失
        optimizer_st.step()  # 更新模型参数

        # print(f'Iter {it} loss_total:{loss_total:.5f}')  # 打印当前迭代损失


    results= {"s":st_model.panel_s, "t":st_model.panel_t}

    return results


def get_pcs_selected(pcs_list):
    """
    对每个Panel，把离其他Panel太远的点筛掉
    :param pcs_list:
    :return:
    """
    result_pcs_list=[]
    for idx, pcs in enumerate(pcs_list):
        pcs_cur = pcs
        pcs_bkg = torch.concat([pcs for i, pcs in enumerate(pcs_list) if i != idx], dim=-2)

        A_expanded = pcs_cur.unsqueeze(1)
        B_expanded = pcs_bkg.unsqueeze(0)

        distances = torch.norm(A_expanded - B_expanded, dim=2)  # (M, N)

        min_distances, min_indices_B = torch.min(distances, dim=1)  # (M,)

        nearest_distances, nearest_indices_A = torch.topk(min_distances, pcs_cur.shape[0], largest=False)

        mask = nearest_distances < 0.12

        result_pcs_list.append(pcs_cur[nearest_indices_A[mask]])
    return result_pcs_list

def panel_optimize(pcs_list, max_iter_t = 100, max_iter_s = 120, target_dis = 0., filter_distance = 0.12):

    pcs_list_selected = get_pcs_selected(pcs_list)

    opt_results = get_optimize_ST(pcs_list_selected)
    s_panel = opt_results.get('s', torch.ones((1,len(pcs_list),3), dtype=torch.float32))
    t_panel = opt_results.get('t', torch.ones((1,len(pcs_list),3), dtype=torch.float32))

    pointcloud_visualize(pcs_list, title="original")

    panel_updated = []
    for pcs, t in zip(pcs_list, t_panel[0]):
        panel_updated.append(pcs+t*1.)
    pointcloud_visualize(panel_updated, title="optimized translation")

    for idx, (pcs, s) in enumerate(zip(panel_updated, s_panel[0])):
        pcs_u = pc_rescale(pcs, s*1.)
        panel_updated[idx] = pcs_u
    pointcloud_visualize(panel_updated, title="optimized scale")

    return panel_updated  # 返回表面和边的坐标
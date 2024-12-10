import torch
import torch.nn.functional as F


def _valid_mean(loss_per_part, valids):
    """Average loss values according to the valid parts.

    Args:
        loss_per_part: [B, P]
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch, averaged over valid parts
    """
    if valids is not None:
        valids = valids.float().detach()
        loss_per_data = (loss_per_part * valids).sum(1) / valids.sum(1)
    else:
        loss_per_data = loss_per_part.sum(1) / loss_per_part.shape[1]
    return loss_per_data


# 二元交叉熵？
def permutation_loss(pred_mat, gt_mat, src_ns, tgt_ns):
    """
    Permutation loss
    $$L_mat = -\frac{1}{N} {\sum_{1\leq i j \leq N} x_{ij}^{gt} \log \hat{x}_{ij}^{gt} + (1-x_{ij}^{gt}) \log (1-\hat{x}_{ij}^{gt})}$$
    @param pred_mat: [B, N_src, N_tgt]
    @param gt_mat: [B, N_src, N_tgt]
    @param src_ns: [B], the number of points of the source in each batch
    @param tgt_ns: [B], the number of points of the target in each batch
    @return: L_mat
    """
    batch_num = pred_mat.shape[0]

    pred_dsmat = pred_mat.to(dtype=torch.float32)

    try:
        assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
        assert torch.all((gt_mat >= 0) * (gt_mat <= 1))
    except AssertionError as err:
        print(pred_dsmat)
        raise err

    loss = torch.tensor(0.0).to(pred_dsmat.device)
    n_sum = torch.zeros_like(loss)
    for b in range(batch_num):
        batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
        loss += F.binary_cross_entropy(
            pred_dsmat[batch_slice], gt_mat[batch_slice], reduction="sum"
        )
        n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

    return loss / n_sum

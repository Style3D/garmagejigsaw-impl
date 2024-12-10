import torch
# 让 dataloader 读取的dict中的tensor集体转移到cuda上
def to_device(data, device):
    if isinstance(data, dict):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(x, device) for x in data]
    elif isinstance(data, tuple):
        return tuple(to_device(x, device) for x in data)
    elif torch.is_tensor(data):
        return data.to(device)
    return data
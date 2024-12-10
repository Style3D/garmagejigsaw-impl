from .all_piece_matching_dataset_styleXD import build_stylexd_dataloader_train_val
from .all_piece_matching_dataset_styleXD import build_stylexd_dataloader_test
from .dataset_config_styleXD import dataset_cfg

# def build_dataloader_train_val(cfg):
#     dataset = cfg.DATASET.lower().split(".")
#     if dataset[0] == "stylexd":
#         if dataset[1] == "all_piece_matching":
#             return build_all_piece_matching_stylexd_dataloader_train_val(cfg)
#         else:
#             raise NotImplementedError(f"Dataset {dataset} not implemented")
#     else:
#         raise NotImplementedError(f"Dataset {dataset} not implemented")
#
def get_dataset_config(dataset_name):
    if dataset_name == "stylexd":
        from .dataset_config_styleXD import dataset_cfg
        return dataset_cfg
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")
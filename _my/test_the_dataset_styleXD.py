# 用于测试 all_piece_matching_dataset_styleXD.py 文件是否能正常运行

from dataset import build_stylexd_dataloader_test
from dataset import build_stylexd_dataloader_train_val
from utils import pointcloud_visualize

def test_model(cfg):
    train_loader, val_loader = build_stylexd_dataloader_train_val(cfg)
    test_loader = build_stylexd_dataloader_test(cfg)
    for batch in train_loader:
        a=10
        break
    # for batch in val_loader:
    #     a = 10
    #     break
    # for batch in tqdm(test_loader):
    #     # pointcloud_visualize(batch["pcs_gt"])
    #     a=1
    # for batch in val_loader:
    #     pointcloud_visualize(batch["pcs_gt"])
    #     a=10
    #     break

if __name__ == "__main__":
    from utils.config import cfg
    from utils.parse_args import parse_args
    args = parse_args("Jigsaw")

    test_model(cfg)
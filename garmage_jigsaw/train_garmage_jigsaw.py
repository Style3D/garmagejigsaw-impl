import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from model import build_model
from dataset import build_stylexd_dataloader_train_val


def train_model(cfg):
    import faulthandler
    faulthandler.enable(all_threads=True)

    if len(cfg.WEIGHT_FILE) > 0 and os.path.exists(cfg.WEIGHT_FILE):
        ckp_path = cfg.WEIGHT_FILE
        print(f"cfg.WEIGHT_FILE : {ckp_path}")
    else:
        ckp_path = None
        if len(cfg.WEIGHT_FILE) > 0:
            print(f"cfg.WEIGHT_FILE {cfg.WEIGHT_FILE} not found.")

    # initial Wandb logger
    logger_name = f"{cfg.MODEL_NAME}"
    logger_id = cfg.WANDB.ID if cfg.RESUME else None
    logger = WandbLogger(
        project=cfg.PROJECT,
        name=logger_name,
        id=logger_id,
        save_dir=cfg.OUTPUT_PATH,
        resume=cfg.RESUME
    )

    # configure callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.MODEL_SAVE_PATH,
        filename="model_{epoch:03d}",
        monitor="epoch",
        every_n_epochs=5,  # save epoch per n epochs
        save_top_k=-1,
        mode=cfg.CALLBACK.CHECKPOINT_MODE,
        save_last=True,
    )

    callbacks = [
        LearningRateMonitor("epoch"),
        checkpoint_callback,
    ]

    all_gpus = list(cfg.GPUS)

    trainer_dict = dict(
        logger=logger,
        accelerator='gpu',
        devices=all_gpus,
        max_epochs=cfg.TRAIN.NUM_EPOCHS,
        callbacks=callbacks,
        benchmark=cfg.CUDNN,
        gradient_clip_val=0.1,
        check_val_every_n_epoch=cfg.TRAIN.VAL_EVERY,
        log_every_n_steps=5,
        profiler='simple',
        detect_anomaly=True,
    )

    if len(all_gpus) > 1:
        trainer_dict.update({
            "strategy": cfg.PARALLEL_STRATEGY
        })

    model = build_model(cfg)

    trainer = pl.Trainer(**trainer_dict)
    train_loader, val_loader = build_stylexd_dataloader_train_val(cfg)

    with torch.autograd.set_detect_anomaly(True):
        if cfg.TRAIN.FINETUNE and ckp_path is not None:
            model.load_state_dict(torch.load(ckp_path)['state_dict'])
        print("Start finetuning")
        trainer.fit(model, train_loader, val_loader)
        print("Done finetuning")


if __name__ == "__main__":
    from utils.config import cfg
    from utils.parse_args import parse_args
    from utils.print_easydict import print_easydict

    args = parse_args("GarmageJigsaw")

    pl.seed_everything(cfg.RANDOM_SEED)

    print_easydict(cfg)
    train_model(cfg)
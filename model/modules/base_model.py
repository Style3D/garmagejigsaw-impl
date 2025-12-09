import time

import torch
from torch import optim
import pytorch_lightning

from utils import filter_wd_parameters, CosineAnnealingWarmupRestarts


class BaseModel(pytorch_lightning.LightningModule):
    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.test_results = None
        self.cd_list = []

    # The flow for this base model is:
    # training_step -> forward_pass -> loss_function ->
    # _loss_function -> forward

    def forward(self, data_dict):
        """Forward pass to predict matching."""
        raise NotImplementedError("forward function should be implemented per model")

    def training_step(self, data_dict, batch_idx, optimizer_idx=-1):
        loss_dict = self.forward_pass(
            data_dict, mode='train', optimizer_idx=optimizer_idx
        )
        return loss_dict['loss']

    def training_epoch_end(self, outputs) -> None:
        torch.cuda.empty_cache()

    def validation_step(self, data_dict, batch_idx):
        loss_dict = self.forward_pass(data_dict, mode='val', optimizer_idx=-1)
        return loss_dict

    def validation_epoch_end(self, outputs):
        func = torch.tensor if \
            isinstance(outputs[0]['batch_size'], int) else torch.stack
        batch_sizes = func([output.pop('batch_size') for output in outputs
                            ]).type_as(outputs[0]['loss'])
        losses = {
            f'val/{k}': torch.stack([output[k] for output in outputs]).reshape(-1)
            for k in outputs[0].keys()
        }
        avg_loss = {
            k: (v * batch_sizes).sum() / batch_sizes.sum()
            for k, v in losses.items()
        }
        self.log_dict(avg_loss, sync_dist=True)
        torch.cuda.empty_cache()

    def test_step(self, data_dict, batch_idx):
        torch.cuda.synchronize()
        start = time.time()
        loss_dict = self.forward_pass(data_dict, mode='test', optimizer_idx=-1)
        torch.cuda.synchronize()
        end = time.time()
        time_elapsed = end - start
        loss_dict['time'] = torch.tensor(time_elapsed, device=loss_dict['loss'].device, dtype=torch.float64)
        return loss_dict

    def test_epoch_end(self, outputs):
        if isinstance(outputs[0]['batch_size'], int):
            func_bs = torch.tensor
            func_loss = torch.stack
        else:
            func_bs = torch.cat
            func_loss = torch.cat
        batch_sizes = func_bs([output.pop('batch_size') for output in outputs
                               ]).type_as(outputs[0]['loss'])
        losses = {
            f'test/{k}': func_loss([output[k] for output in outputs])
            for k in outputs[0].keys()
        }
        avg_loss = {
            k: (v * batch_sizes).sum() / batch_sizes.sum()
            for k, v in losses.items()
        }
        print('; '.join([f'{k}: {v.item():.6f}' for k, v in avg_loss.items()]))

        total_shape_cd = torch.mean(torch.cat(self.cd_list))
        print(f'total_shape_cd: {total_shape_cd.item():.6f}')

        self.test_results = avg_loss
        self.log_dict(avg_loss, sync_dist=True)

    def _loss_function(self, data_dict, out_dict={}, optimizer_idx=-1):
        raise NotImplementedError("loss_function should be implemented per model")

    def loss_function(self, data_dict, optimizer_idx, mode):
        out_dict = self.forward(data_dict)
        loss_dict = self._loss_function(data_dict, out_dict, optimizer_idx)

        if 'loss' not in loss_dict:
            total_loss = 0.
            for k, v in loss_dict.items():
                if k.endswith('_loss'):
                    total_loss += v * eval(f'self.cfg.LOSS.{k.upper()}_W')
            loss_dict['loss'] = total_loss

        total_loss = loss_dict['loss']
        if total_loss.numel() != 1:
            loss_dict['loss'] = total_loss.mean()

        if not self.training:
            if 'batch_size' not in loss_dict:
                loss_dict['batch_size'] = out_dict['batch_size']
        return loss_dict

    def forward_pass(self, data_dict, mode, optimizer_idx):
        loss_dict = self.loss_function(data_dict, optimizer_idx=optimizer_idx, mode=mode)
        # log
        if (mode == 'train' or mode == 'val' or mode == 'test') and self.local_rank == 0:
            log_dict = {f'{mode}/{k}': v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
            data_name = [k for k in self.trainer.profiler.recorded_durations.keys() if 'prepare_data' in k][0]
            log_dict[f'{mode}/data_time'] = self.trainer.profiler.recorded_durations[data_name][-1]
            self.log_dict(log_dict, logger=True, sync_dist=False, rank_zero_only=True)

        return loss_dict

    def configure_optimizers(self):
        """Build optimizer and lr scheduler."""
        lr = self.cfg.TRAIN.LR
        wd = self.cfg.TRAIN.WEIGHT_DECAY

        if wd > 0.:
            params_dict = filter_wd_parameters(self)
            params_list = [{
                'params': params_dict['no_decay'],
                'weight_decay': 0.,
            }, {
                'params': params_dict['decay'],
                'weight_decay': wd,
            }]
            optimizer = optim.AdamW(params_list, lr=lr)
        else:
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.)

        if len(self.cfg.TRAIN.LR_SCHEDULER)>0 and self.cfg.TRAIN.LR_SCHEDULER:
            assert self.cfg.TRAIN.LR_SCHEDULER.lower() in ['cosine']
            total_epochs = self.cfg.TRAIN.NUM_EPOCHS
            warmup_epochs = int(total_epochs * self.cfg.TRAIN.WARMUP_RATIO)
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                total_epochs,
                max_lr=lr,
                min_lr=lr / self.cfg.TRAIN.LR_DECAY,
                warmup_steps=warmup_epochs,
            )
            return (
                [optimizer],
                [{
                    'scheduler': scheduler,
                    'interval': 'epoch',
                }],
            )
        return optimizer
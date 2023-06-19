import os
import torch
import pytorch_lightning as pl
import clip
from pytorch_metric_learning.miners import BatchHardMiner
from timm.optim.adamp import AdamP
from timm.scheduler import CosineLRScheduler
from .losses import ArcFaceFocalLoss
from .pl_metrics import compute_metrics

from data.dataset_clip import dataloader
from data.dataset import dataloader as notxt_dataloader
from data.augmentations import train_augs


# hook for the pl.Module and already build model
class PLModule(pl.LightningModule):
    def __init__(self,
                 data_root: str,
                 model,
                 norm_emb=False,
                 augmenations=False,
                 num_epoches=40,
                 num_features=2048,
                 num_workers=6,
                 text_loss_weight=0.1):
        super().__init__()

        self.p_to_train_imgs                = os.path.join(data_root, 'Training')
        self.p_to_val_imgs                  = os.path.join(data_root, 'Validation')

        self.norm_emb                       = norm_emb
        self.image_size                     = 224
        self.batch_size                     = 64
        self.num_classes                    = 2
        self.num_workers                    = num_workers
        self.num_epochs                     = num_epoches

        self.loss_margin                    = 0.5
        self.loss_scale                     = 1
        self.model                          = model
        self.model_parameters               = [p for p in model.parameters() if p.requires_grad]
        self.model.to(self.device)
        self.clip_workaround                = True
        self.loss                           = ArcFaceFocalLoss(num_classes=self.num_classes,
                                                               margin=self.loss_margin,
                                                               scale=self.loss_scale,
                                                               num_features=num_features)

        self.miner                          = BatchHardMiner()

        # augmenters for dataloader
        self.train_augs                     = None
        if augmenations:
            self.train_augs                 = train_augs()

        self.training_step_outputs          = []
        self.validation_step_outputs        = []

    def forward(self, x):
        self.model.forward(x)

    def training_step(self, batch, batch_nb):
        img, target, caption                = batch
        embeddings_image                    = self.model(img)
        hard_pairs                          = self.miner(embeddings_image, target)
        loss_train                          = self.loss(embeddings_image, target, hard_pairs)

        self.log('loss_train', loss_train, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.training_step_outputs.append(loss_train)
        return {'loss': loss_train}

    def on_train_epoch_end(self):
        train_loss = torch.stack(self.training_step_outputs)
        self.training_step_outputs.clear()

        self.log_dict = {'loss_train': sum(train_loss) / len(train_loss)}
        self.model.float()

    def validation_step(self, batch, batch_idx):
        img, target, caption                = batch
        embeddings_image                    = self.model(img)
        hard_pairs                          = self.miner(embeddings_image, target)
        loss_val                            = self.loss(embeddings_image, target, hard_pairs)

        self.log('loss_val', loss_val, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.validation_step_outputs.append(loss_val)
        return {'loss_val': loss_val}

    def on_validation_epoch_end(self):
        loss_val = torch.stack(self.validation_step_outputs)
        self.validation_step_outputs.clear()

        clip.model.convert_weights(self.model)
        precision, _, mean_average_precision = compute_metrics(self.no_txt_val_dataloader(), self.model,
                                                               normalize=self.norm_emb)
        loss_val = sum(loss_val) / len(loss_val)
        self.log('precision', precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('mean_average_precision', mean_average_precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss_val', loss_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        results = {'loss_val': loss_val}
        self.model.float()
        return results

    def configure_optimizers(self):
        optimizer = AdamP(lr=1e-4, weight_decay=0.1, params=self.model.parameters())
        scheduler = CosineLRScheduler(optimizer,
                                      cycle_limit=1,
                                      lr_min=1e-6,
                                      warmup_lr_init=1e-5,
                                      warmup_t=3,
                                      t_initial=self.num_epochs)
        if 'timm' in scheduler.__str__():
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        else:
            return [optimizer], [scheduler]

    # workaround for timm
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)

    def train_dataloader(self):
        return dataloader(p_to_images=self.p_to_train_imgs, batch_size=self.batch_size, augs=self.train_augs,
                          num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return dataloader(p_to_images=self.p_to_val_imgs, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False, drop_last=False)

    def no_txt_val_dataloader(self):
        return notxt_dataloader(p_to_images=self.p_to_val_imgs, batch_size=self.batch_size,
                                num_workers=self.num_workers, shuffle=False, drop_last=False, precision='fp16')

# workaround for https://github.com/openai/CLIP/issues/57

    def on_before_optimizer_step(self, optimizer) -> None:
        self.model.float()
        # taken as convergence trick @ https://github.com/mlfoundations/wise-ft/blob/master/src/models/finetune.py
        torch.nn.utils.clip_grad_norm_(self.model_parameters, 1.0)

    def on_before_zero_grad(self, optimizer) -> None:
        clip.model.convert_weights(self.model)

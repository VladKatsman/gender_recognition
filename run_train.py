import os
from modelling import models
from training.pl_model import PLModule
from training.pl_model_clip import PLModule as PLModuleClip
from training.pl_trainer import BaseTrainer
from misc.visualization_utils import center_print
import cfg

# num_features = model.classifier.in_features


def train_baseline(norm_emb=True, num_workers=6, n_features=2048):
    model = models.baseline_backbone()
    do_train(model, norm_emb, num_workers, n_features)


def train_baseline_custom(norm_emb=False, num_workers=6, n_features=128):
    model = models.baseline_backbone_neck()
    do_train(model, norm_emb, num_workers, n_features)


def do_train(model, norm_emb=True, num_workers=6, n_features=2048):
    pl_model = PLModuleClip(cfg.DATA_ROOT, model=model, norm_emb=norm_emb, num_workers=num_workers, num_features=n_features)
    pl_trainer = BaseTrainer(cfg.OUT_DIR, debug_ratio=cfg.DEBUG_RATIO)
    best_model_score = pl_trainer.run_trainer(pl_model)
    center_print(f"Best model average precision: {best_model_score}")


def train_clip(norm_emb=False, num_workers=6, n_features=128, text_loss_w=0.1):
    model = models.clip_vitb16_with_neck()
    pl_model = PLModuleClip(cfg.DATA_ROOT,
                            model=model,
                            norm_emb=norm_emb,
                            num_workers=num_workers,
                            num_features=n_features,
                            text_loss_weight=text_loss_w)
    pl_trainer = BaseTrainer(cfg.OUT_DIR, debug_ratio=cfg.DEBUG_RATIO)
    best_model_score = pl_trainer.run_trainer(pl_model)
    center_print(f"Best model average precision: {best_model_score}")


if __name__ == '__main__':
    # train_baseline_custom()
    train_clip()

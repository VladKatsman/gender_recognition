import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar


class BaseTrainer:
    """ This class realizes the experiment interface """

    def __init__(self, train_dir, debug_ratio=1.0, num_epoches=40):
        self.num_epoches = num_epoches
        self.data_usage_ratio = debug_ratio
        self.train_dir = train_dir
        # init train dir if it not exists
        os.makedirs(train_dir, exist_ok=True)

    # run for metrics (stored @ gcloud, check configuration)
    def run_trainer(self, model):
        trainer = self.build_default_trainer()

        # train loop
        trainer.fit(model)

        # test model in order to get report (update it to be able to send docs to the slack without saving it locally)
        best_model_score = trainer.checkpoint_callback.best_model_score

        return trainer.log_dir, best_model_score

    # default trainer
    def build_default_trainer(self):
        logger          = TensorBoardLogger(name='logger', save_dir=self.train_dir)
        checkpointer    = ModelCheckpoint(dirpath=logger.log_dir,
                                          monitor='precision',
                                          save_top_k=1,
                                          save_weights_only=True,
                                          mode='max',
                                          filename="{epoch:02d}-{precision:.4f}")
        early_stop      = EarlyStopping(monitor='loss_val',
                                        patience=5,
                                        mode='min')

        trainer = pl.Trainer(
            devices='auto',
            strategy='auto',  # dp for multiple gpus
            enable_checkpointing=True,
            logger=logger,
            callbacks=[early_stop, checkpointer, MyProgressBar()],
            max_epochs=self.num_epoches,
            precision=16,
            benchmark=True,
            limit_train_batches=self.data_usage_ratio,  # debug purposes
            # limit_train_batches=0.01,  # debug purposes
            limit_val_batches=self.data_usage_ratio,    # debug purposes
        )
        return trainer


class MyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

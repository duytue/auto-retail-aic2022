import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from abc import abstractmethod


class BaseModel(pl.LightningModule):
    """
    Base class for all models
    """
    def __init__(self, criterion, metric_ftns, config):
        super().__init__()
        self.config = config

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']

        self.checkpoint_dir = config.save_dir
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()
        # setup visualization writer instance                
        self.save_hyperparameters(config.config)

    def training_step(self, batch, batch_idx):
        data, target = batch

        output = self.forward(data)
        loss = self.criterion(output, target)

        self.log('train_acc_step', self.train_accuracy(output, target))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.train_accuracy.compute())

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = self.config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = self.config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
        return [optimizer], [lr_scheduler]

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = self.criterion(output, target)

        self.log('val_acc_step', self.val_accuracy(output, target))
        self.log('val_loss', loss)
        return loss

    def validation_epoch_end(self, outs):
        # log epoch metric
        self.log('val_acc_epoch', self.val_accuracy.compute())

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = self.criterion(output, target)

        self.test_accuracy(output, target)
        self.log('test_loss', loss)
        return loss

    def test_epoch_end(self, outs):
        self.log('test_acc_epoch', self.test_accuracy.compute())

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

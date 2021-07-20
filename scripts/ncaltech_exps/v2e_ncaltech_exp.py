"""V2E N-Caltech101 Exp.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy

from v2e_exps.ncaltech_model import V2ENCaltechModel
from v2e_exps.ncaltech_dataset import V2ENCaltechData


class V2ENCaltechExp(pl.LightningModule):
    def __init__(self, cfg):
        super(V2ENCaltechExp, self).__init__()

        self.cfg = cfg

        self.net = V2ENCaltechModel(
            pretrained=self.cfg.use_pretrained,
            num_voxel_bins=self.cfg.num_voxel_bins)

        self.loss = nn.CrossEntropyLoss()
        self.metric = Accuracy()

    def training_step(self, batch, batch_nb):
        (sample, label) = batch

        preds = self.net(sample)

        loss = self.loss(preds, label)
        acc = self.metric(torch.argmax(preds, dim=1), label)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_accuracy", acc, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_nb):
        (sample, label) = batch

        with torch.no_grad():
            preds = self.net(sample)

            loss = self.loss(preds, label)
            acc = self.metric(torch.argmax(preds, dim=1), label)

        self.log("val_loss", loss, on_epoch=True)
        self.log("val_accuracy", acc, on_epoch=True)

        return loss

    def test_step(self, batch, batch_nb):
        """Test on real N-Caltech data."""
        (sample, label) = batch

        with torch.no_grad():
            preds = self.net(sample)

            loss = self.loss(preds, label)
            acc = self.metric(torch.argmax(preds, dim=1), label)

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_accuracy", acc, on_epoch=True)

        return loss

    def train_dataloader(self):
        dataset = V2ENCaltechData(
            self.cfg.train_data_root, self.cfg.data_list, self.cfg.train_ext,
            num_voxel_bins=self.cfg.num_voxel_bins, is_train=True,
            augmentation=self.cfg.augmentation)
        return DataLoader(
            dataset, batch_size=self.cfg.batch_size,
            shuffle=True, num_workers=8)

    def val_dataloader(self):
        dataset = V2ENCaltechData(
            self.cfg.valid_data_root, self.cfg.data_list, self.cfg.valid_ext,
            num_voxel_bins=self.cfg.num_voxel_bins, is_train=False,
            augmentation=self.cfg.augmentation)
        return DataLoader(
            dataset, batch_size=self.cfg.batch_size*4,
            shuffle=False, num_workers=8)

    def test_dataloader(self):
        """Test on the real N-Caltech dataset."""
        dataset = V2ENCaltechData(
            self.cfg.test_data_root, self.cfg.data_list, self.cfg.test_ext,
            num_voxel_bins=self.cfg.num_voxel_bins, is_train=False,
            augmentation=self.cfg.augmentation)
        return DataLoader(
            dataset, batch_size=self.cfg.batch_size*4,
            shuffle=False, num_workers=8)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.cfg.step_size, gamma=0.1, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            "monitor": "val_loss"
        }

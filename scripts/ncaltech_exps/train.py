"""Training script for N-Caltech 101 experiments.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""
import argparse
import os
import json
import time
import random

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger

from v2e_ncaltech_exp import V2ENCaltechExp
from finetune_v2e_ncaltech_exp import FinetuneV2ENCaltechExp
from v2e_exps.utils import DaemonTensorboard
from v2e_exps.utils import expandpath, make_unique_dir


parser = argparse.ArgumentParser()

# experiment setting
parser.add_argument("--v2e_ncaltech_exp", action="store_true")
parser.add_argument("--finetune_v2e_ncaltech_exp", action="store_true")
# turn on evaluation
parser.add_argument("--evaluate", action="store_true")

# data configs
parser.add_argument("--train_data_root", type=expandpath,
                    help="the train data location.")
parser.add_argument("--valid_data_root", type=expandpath,
                    help="the valid data location.")
parser.add_argument("--test_data_root", type=expandpath,
                    help="the test data.")
# extension
parser.add_argument("--train_ext", type=str, default=".h5")
parser.add_argument("--valid_ext", type=str, default=".h5")
parser.add_argument("--test_ext", type=str, default=".h5")

parser.add_argument("--data_list", type=expandpath)
parser.add_argument("--num_voxel_bins", type=int, default=12)
parser.add_argument("--augmentation", action="store_true")

# logging
parser.add_argument("--log_root", type=expandpath)
parser.add_argument("--saved_checkpoint_path", type=str, default="")

# training parameters
parser.add_argument("--use_pretrained", action="store_true")
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--step_size", type=int, default=30)

# tensorboard
parser.add_argument("--tb_start", action="store_true")
parser.add_argument("--tb_port", type=int, default=6006)

cfg = parser.parse_args()

# put some sleep time to make sure no folder conflict
time.sleep(random.uniform(3, 5))

# make unique log root
if not cfg.evaluate:
    cfg.log_root = make_unique_dir(cfg.log_root)

print(json.dumps(cfg.__dict__,
                 indent=4, sort_keys=True))

# make sure the checkpoint is available
if cfg.evaluate:
    assert os.path.isfile(cfg.saved_checkpoint_path)

# define exp
if cfg.v2e_ncaltech_exp:
    if cfg.evaluate:
        exp = V2ENCaltechExp.load_from_checkpoint(
            cfg.saved_checkpoint_path, cfg=cfg)
        exp.eval()
    else:
        exp = V2ENCaltechExp(cfg)
if cfg.finetune_v2e_ncaltech_exp:
    exp = FinetuneV2ENCaltechExp.load_from_checkpoint(
        cfg.saved_checkpoint_path, cfg=cfg)

    if cfg.evaluate:
        exp.eval()

# define a logger
logger = TestTubeLogger(
    save_dir=cfg.log_root,
    name='tube'
)

# define a checkpoint callback
monitor = 'val_accuracy'
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(cfg.log_root, 'checkpoint'),
    save_top_k=1,
    save_last=True,
    verbose=True,
    monitor=monitor,
    mode='max' if 'accuracy' in monitor else 'min',
)

# define trainer
if cfg.evaluate:
    trainer = Trainer(
        gpus="-1" if torch.cuda.is_available() else None)
else:
    trainer = Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        gpus="-1" if torch.cuda.is_available() else None,
        max_epochs=cfg.num_epochs,
        weights_save_path=cfg.log_root,
        accumulate_grad_batches=1)

daemon_tb = None
daemon_ngrok = None

if cfg.tb_start:
    daemon_tb = DaemonTensorboard(cfg.log_root, cfg.tb_port)
    daemon_tb.start()

if not cfg.evaluate:
    trainer.fit(exp)
else:
    result = trainer.test(exp)
    print("Evaluation Result: {}".format(result))

if daemon_tb is not None:
    daemon_tb.stop()

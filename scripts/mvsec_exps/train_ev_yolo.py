"""Training script for the event based YOLO.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import argparse
import os
import json

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from evtrans.pyyolo.model import YoloNetV3
from evtrans.networks import EVYOLOFrontend
from evtrans.networks import FrameYOLOFrontend
from evtrans.networks import EVYOLOEvalNet
from evtrans.data import EVYOLODataset
from evtrans.loss import NGA_loss
from evtrans import log_history

from v2e_exps.utils import make_unique_dir
from v2e_exps.utils import expandpath


# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#  get on a parser
parser = argparse.ArgumentParser()

# image size
parser.add_argument("--img_size", type=int, default=416)
# number of epochs
parser.add_argument("--num_epochs", type=int, default=10)
# batch size
parser.add_argument("--batch_size", type=int, default=4)
# training folder
parser.add_argument("--train_data_dir", type=str)
# validation folder
parser.add_argument("--val_data_dir", type=str)
# logging directory
parser.add_argument("--log_dir", type=str)

# network weight path
parser.add_argument("--weights_path", type=expandpath,
                    default=os.path.join("..", "res",
                                         "yolov3_original.pth"))
# cutoff layer
parser.add_argument("--cut_stage", type=int)
# eval end layer
parser.add_argument("--eval_stage", type=int)
# number of outputs in middle net
parser.add_argument("--num_eval_outputs", type=int)

# number of convolution input dimension
parser.add_argument("--conv_input_dim", type=int)

# loss term coefficient
parser.add_argument("--frl", type=float, default=0.)
parser.add_argument("--fel", type=float, default=0.)
parser.add_argument("--frl_gram", type=float, default=0.)
parser.add_argument("--fel_gram", type=float, default=0.)
parser.add_argument("--fel_context", type=float, default=0.)
parser.add_argument("--tv", type=float, default=0.)

parser.add_argument("--sample_limit", type=int, default=0)

# if it's mixed day and night condition
parser.add_argument("--mixed", action="store_true")
parser.add_argument("--night", action="store_true")

# parse argument
args = parser.parse_args()

# make unique log root
args.log_dir = make_unique_dir(args.log_dir)

print(json.dumps(args.__dict__,
                 indent=4, sort_keys=True))

# set up logging server
checkpoint_dir = os.path.join(args.log_dir, "checkpoints")
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    print("Setup the logging directory at {}".format(args.log_dir))

log_file = os.path.join(args.log_dir, "log.csv")

# train and val data loader
train_dl = DataLoader(
    EVYOLODataset(args.train_data_dir, img_size=args.img_size,
                  is_train=True,
                  parsing="/*/*.npz" if args.mixed else "/*.npz",
                  is_recurrent=args.r1, no_resampling=True,
                  sample_limit=args.sample_limit,
                  intensity_offset=20 if args.night else 0),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=False)

val_dl = DataLoader(
    EVYOLODataset(args.val_data_dir, img_size=args.img_size,
                  is_train=False, parsing="/*.npz",
                  is_recurrent=args.r1, no_resampling=True,
                  intensity_offset=20 if args.night else 0),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    drop_last=False)


# load pretrained model
pretrain_model = YoloNetV3(nms=True)
pretrain_model.load_state_dict(torch.load(args.weights_path))
pretrain_model.to(DEVICE)
pretrain_model.eval()

# build Event driven front end
ev_yolo = EVYOLOFrontend(
    stage=args.cut_stage,
    input_dim=args.conv_input_dim,
    pretrained_model=None).to(DEVICE)

# made sure YOLO frontend and evaluation are at evaluation mode
yolo_frontend = FrameYOLOFrontend(
    pretrain_model.darknet, stage=args.cut_stage).to(DEVICE)
yolo_frontend.eval()

print("Initialize EV Frontend and Frontend Net")

yolo_eval_net = EVYOLOEvalNet(
    pretrain_model.darknet,
    stages=[args.cut_stage+1, args.eval_stage]).to(DEVICE)
yolo_eval_net.eval()

print("Initialized Middle Net")

optimizerG = Adam(ev_yolo.parameters(), lr=1e-4)

for epoch_i in range(args.num_epochs):
    print("*"*50)
    print("TRAINING EPOCH {}".format(epoch_i+1))

    #  if lstm_frontend is not None:
    #      h_state, c_state = get_recurrent_initial_state(
    #          args.batch_size, args.img_size, args.r1, args.r2, DEVICE)
    #  else:
    #      h_state, c_state = None, None

    epoch_losses = []
    epoch_frl = []
    epoch_fel = []
    epoch_tv = []
    epoch_frl_gram = []
    epoch_fel_gram = []
    epoch_fel_context = []
    for batch_i, (ev_batch, img_batch) in enumerate(train_dl):
        ev_yolo.train()

        # calculate generator loss and prepare features
        loss_output = NGA_loss(
            ev_batch.to(DEVICE), img_batch.to(DEVICE),
            ev_yolo, yolo_frontend, yolo_eval_net,
            h_state=None, c_state=None,
            frl=args.frl, fel=args.fel,
            frl_gram=args.frl_gram, fel_gram=args.fel_gram,
            tv=args.tv, fel_context=args.fel_context,
            return_all_parts=True)

        # decode batch output
        #  if lstm_frontend is not None:
        #      (b_frl, b_fel, b_tv, b_frl_gram, b_fel_gram, b_fel_context,
        #       batch_loss, ev_feature,
        #       img_feature, h_state, c_state) = loss_output
        #  else:
        (b_frl, b_fel, b_tv, b_frl_gram, b_fel_gram, b_fel_context,
         batch_loss, ev_feature, img_feature) = loss_output

        optimizerG.zero_grad()
        batch_loss.backward()

        optimizerG.step()

        #  if lstm_frontend is not None:
        #      # detach the state from previous batch
        #      h_state = h_state.detach()
        #      c_state = c_state.detach()

        # collecting log
        print("[TRAIN] L at batch {}: {:.3f} FRL: {:.3f} "
              "FEL: {:.3f} TV: {:.3f} FRLG: {:.3f} "
              "FELG: {:.3f} FELC: {:.3f}".format(
                  batch_i, batch_loss,
                  b_frl, b_fel, b_tv, b_frl_gram,
                  b_fel_gram, b_fel_context), end='\r')

        epoch_losses.append(float(batch_loss.item()))
        epoch_frl.append(float(b_frl))
        epoch_fel.append(float(b_fel))
        epoch_tv.append(float(b_tv))
        epoch_frl_gram.append(float(b_frl_gram))
        epoch_fel_gram.append(float(b_fel_gram))
        epoch_fel_context.append(float(b_fel_context))

    print("\n"+"-"*50)
    # record training loss
    train_epoch_loss = np.mean(epoch_losses)
    train_frl = np.mean(epoch_frl)
    train_fel = np.mean(epoch_fel)
    train_tv = np.mean(epoch_tv)
    train_frl_gram = np.mean(epoch_frl_gram)
    train_fel_gram = np.mean(epoch_fel_gram)
    train_fel_context = np.mean(epoch_fel_context)

    print("[TRAIN] L at epoch {}: {:.3f} FRL: {:.3f} "
          "FEL: {:.3f} TV: {:.3f} FRLG: {:.3f} "
          "FELG: {:.3f} FELC: {:.3f}".format(
              epoch_i+1, train_epoch_loss, train_frl, train_fel,
              train_tv, train_frl_gram, train_fel_gram,
              train_fel_context))
    print("\n"+"-"*50)

    # evaluation pipeline
    epoch_losses = []
    epoch_frl = []
    epoch_fel = []
    epoch_tv = []
    epoch_frl_gram = []
    epoch_fel_gram = []
    epoch_fel_context = []

    # get initial state
    #  if lstm_frontend is not None:
    #      h_state, c_state = get_recurrent_initial_state(
    #          args.batch_size, args.img_size, args.r1, args.r2, DEVICE)
    #  else:
    #      h_state, c_state = None, None

    for batch_i, (ev_batch, img_batch) in enumerate(val_dl):
        ev_yolo.eval()

        # calculate loss
        with torch.no_grad():
            # calculate loss
            loss_output = NGA_loss(
                ev_batch.to(DEVICE), img_batch.to(DEVICE),
                ev_yolo, yolo_frontend, yolo_eval_net,
                h_state=None, c_state=None,
                frl=args.frl, fel=args.fel,
                frl_gram=args.frl_gram, fel_gram=args.fel_gram,
                tv=args.tv, fel_context=args.fel_context,
                return_all_parts=True)

            # decode batch output
            #  if lstm_frontend is not None:
            #      (b_frl, b_fel, b_tv, b_frl_gram, b_fel_gram, b_fel_context,
            #       batch_loss, ev_feature,
            #       img_feature, h_state, c_state) = loss_output
            #  else:
            (b_frl, b_fel, b_tv, b_frl_gram, b_fel_gram, b_fel_context,
             batch_loss, ev_feature, img_feature) = loss_output

            # collect log
            epoch_losses.append(float(batch_loss.item()))
            epoch_frl.append(float(b_frl))
            epoch_fel.append(float(b_fel))
            epoch_tv.append(float(b_tv))
            epoch_frl_gram.append(float(b_frl_gram))
            epoch_fel_gram.append(float(b_fel_gram))
            epoch_fel_context.append(float(b_fel_context))

            #  if lstm_frontend is not None:
            #      # detach the state from previous batch
            #      h_state = h_state.detach()
            #      c_state = c_state.detach()

            print("[VALID] L at batch {}: {:.3f} FRL: {:.3f} "
                  "FEL: {:.3f} TV: {:.3f} FRLG: {:.3f} "
                  "FELG: {:.3f} FELC: {:.3f}".format(
                      batch_i, batch_loss,
                      b_frl, b_fel, b_tv, b_frl_gram,
                      b_fel_gram, b_fel_context), end='\r')

    # calculate validation loss
    val_epoch_loss = np.mean(epoch_losses)
    val_frl = np.mean(epoch_frl)
    val_fel = np.mean(epoch_fel)
    val_tv = np.mean(epoch_tv)
    val_frl_gram = np.mean(epoch_frl_gram)
    val_fel_gram = np.mean(epoch_fel_gram)
    val_fel_context = np.mean(epoch_fel_context)
    print("-"*50)
    print("[VALID] L at epoch {}: {:.3f} FRL: {:.3f} "
          "FEL: {:.3f} TV: {:.3f} FRLG: {:.3f} "
          "FELG: {:.3f} FELC: {:.3f}".format(
              epoch_i+1, val_epoch_loss, val_frl, val_fel,
              val_tv, val_frl_gram, val_fel_gram,
              val_fel_context))

    # write log into file
    log_history(log_file,
                [train_epoch_loss,
                 train_frl, train_fel, train_tv, train_frl_gram,
                 train_fel_gram, train_fel_context,
                 val_epoch_loss,
                 val_frl, val_fel, val_tv, val_frl_gram,
                 val_fel_gram, val_fel_context],
                idx=epoch_i+1,
                header=["epoch",
                        "train_loss",
                        "train_frl", "train_fel", "train_tv",
                        "train_frl_gram", "train_fel_gram",
                        "train_fel_context",
                        "val_loss",
                        "val_frl", "val_fel", "val_tv",
                        "val_frl_gram", "val_fel_gram",
                        "val_fel_context"])

    print("*"*50)

    # Saving the network
    checkpoint_path = os.path.join(
        checkpoint_dir, "checkpoint_{}.pt".format(epoch_i+1))
    torch.save(
        {"epoch": epoch_i,
         "model_state_dict": ev_yolo.state_dict()},
        checkpoint_path)

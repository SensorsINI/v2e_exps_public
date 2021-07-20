"""Visualize N-Caltech data.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

import argparse
import pickle
import os
import random

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import resize

from v2e_exps.utils import load_rpg_events
from v2e_exps.utils import load_ncaltech_events_bin
from v2e_exps.utils import load_v2e_events
from v2e_exps.ncaltech_dataset import prepare_voxel
from v2e_exps.utils import pad_frame
from v2e_exps.utils import expandpath

import matplotlib.pyplot as plt
import seaborn as sns


def robust_img(data):
    robust_max = np.percentile(data.flatten(), 99)
    robust_min = np.percentile(data.flatten(), 1)

    data = np.clip(data, robust_min, robust_max)

    return data


# load data functions
load_events = {
    ".npz": load_rpg_events,
    ".bin": load_ncaltech_events_bin,
    ".h5": load_v2e_events}

parser = argparse.ArgumentParser()

parser.add_argument("--data_list", type=expandpath)

parser.add_argument("--ncaltech_root", type=expandpath)
parser.add_argument("--v2e_ideal_root", type=expandpath)
parser.add_argument("--v2e_bright_root", type=expandpath)
parser.add_argument("--v2e_dark_root", type=expandpath)

args = parser.parse_args()

with open(args.data_list, "rb") as f:
    file_list = pickle.load(f, encoding="utf-8")

files = file_list["train"]
random.shuffle(files)

#  files = [("chair/image_0001", 1)]

for file_name in files:
    ncaltech_path = os.path.join(
        args.ncaltech_root, file_name[0]+".bin")
    ideal_path = os.path.join(
        args.v2e_ideal_root, file_name[0]+".h5")
    bright_path = os.path.join(
        args.v2e_bright_root, file_name[0]+".h5")
    dark_path = os.path.join(
        args.v2e_dark_root, file_name[0]+".h5")

    # load events
    ncaltech_events = load_events[".bin"](ncaltech_path)
    ideal_events = load_events[".h5"](ideal_path)
    bright_events = load_events[".h5"](bright_path)
    dark_events = load_events[".h5"](dark_path)

    ncaltech_events = ncaltech_events[
        ncaltech_events[:, 0] <= (
            ncaltech_events[-1, 0]-ncaltech_events[0, 0])//3]
    # prepare voxel
    n_height, n_width = \
        ncaltech_events[:, 2].max()+1, ncaltech_events[:, 1].max()+1
    ncaltech_voxel = prepare_voxel(
        ncaltech_events, num_voxel_bins=15,
        height=n_height, width=n_width, normalize=True,
        remove_outlier=True)

    ncaltech_voxel = pad_frame(ncaltech_voxel, 180, 240)
    ncaltech_voxel = torch.tensor(ncaltech_voxel, dtype=torch.float32)
    ncaltech_voxel = resize(ncaltech_voxel, [180, 240])

    ideal_events = ideal_events[
        ideal_events[:, 0] <= (ideal_events[-1, 0]-ideal_events[0, 0])//3]
    bright_events = bright_events[
        bright_events[:, 0] <= (bright_events[-1, 0]-bright_events[0, 0])//3]
    dark_events = dark_events[
        dark_events[:, 0] <= (dark_events[-1, 0]-dark_events[0, 0])//3]

    ideal_voxel = prepare_voxel(
        ideal_events, num_voxel_bins=15,
        height=180, width=240, normalize=True, remove_outlier=True)
    bright_voxel = prepare_voxel(
        bright_events, num_voxel_bins=15,
        height=180, width=240, normalize=True, remove_outlier=True)
    dark_voxel = prepare_voxel(
        dark_events, num_voxel_bins=15,
        height=180, width=240, normalize=True, remove_outlier=True)

    # prepare visualization
    ncaltech_hm = ncaltech_voxel.data.numpy().mean(axis=0)
    #  ideal_hm = robust_img(ideal_voxel.mean(axis=0))
    #  bright_hm = robust_img(bright_voxel.mean(axis=0))
    #  dark_hm = robust_img(dark_voxel.mean(axis=0))
    ideal_hm = ideal_voxel.mean(axis=0)
    bright_hm = bright_voxel.mean(axis=0)
    dark_hm = dark_voxel.mean(axis=0)

    # visualize
    fig = plt.figure(figsize=(16, 8))

    plt.subplot(2, 4, 1)
    sns.heatmap(ncaltech_hm, square=True, cbar=False, cmap="gray")
    plt.axis("off")
    plt.title("Real Events")

    plt.subplot(2, 4, 2)
    sns.heatmap(ideal_hm, square=True, cbar=False, cmap="gray")
    plt.axis("off")
    plt.title("V2E Ideal Events")

    plt.subplot(2, 4, 3)
    sns.heatmap(bright_hm, square=True, cbar=False, cmap="gray")
    plt.axis("off")
    plt.title("V2E Bright Events")

    plt.subplot(2, 4, 4)
    sns.heatmap(dark_hm, square=True, cbar=False, cmap="gray")
    plt.axis("off")
    plt.title("V2E Dark Events")

    plt.subplot(2, 4, 5)
    plt.hist(ncaltech_hm.flatten(), bins=100)

    plt.subplot(2, 4, 6)
    plt.hist(ideal_hm.flatten(), bins=100)

    plt.subplot(2, 4, 7)
    plt.hist(bright_hm.flatten(), bins=100)

    plt.subplot(2, 4, 8)
    plt.hist(dark_hm.flatten(), bins=100)

    plt.tight_layout()
    #  plt.show()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    cv2.imshow("V2E Visualize", data)
    cv2.waitKey(30)

    plt.cla()
    fig.clear()
    plt.close()
    del fig

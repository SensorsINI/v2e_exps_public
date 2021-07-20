"""Make Video.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import
import argparse
import os
import glob

import numpy as np

from skimage.io import imread
from skimage.transform import rescale
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from v2ecore.v2e_utils import video_writer
from v2e_exps.utils import expandpath


parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=expandpath)
parser.add_argument("--gt_root", type=expandpath)
parser.add_argument("--pred_root", type=expandpath)
parser.add_argument("--video_output_path", type=expandpath)
parser.add_argument("--frame_rate", type=int, default=43)

args = parser.parse_args()

files_list = sorted(glob.glob("{}".format(args.data_root)+"/*.npz"))

clahe = cv2.createCLAHE()
out_writer = None

for file_idx in range(len(files_list)):
    # read the data
    file_path = files_list[file_idx]
    base_name = os.path.basename(file_path)[:-4]

    data = np.load(file_path)
    rgb_img = data["img"][..., np.newaxis]
    rgb_img = np.concatenate((rgb_img, rgb_img, rgb_img), axis=2)

    ev_img = data["ev_img"]
    ev_img = ev_img.mean(axis=0)
    robust_max = np.percentile(ev_img, 99)
    robust_min = np.percentile(ev_img, 1)
    ev_img = np.clip(ev_img, robust_min, robust_max)
    ev_img = (ev_img-robust_min)/(robust_max-robust_min)
    ev_img = (ev_img*255).astype(np.uint8)
    ev_img = clahe.apply(ev_img)[..., np.newaxis]
    ev_img = np.concatenate((ev_img, ev_img, ev_img), axis=2)

    # reading predictions
    rgb_detect = os.path.join(args.gt_root, base_name+".txt")
    ev_detect = os.path.join(args.pred_root, base_name+".txt")
    try:
        gt = np.loadtxt(rgb_detect, usecols=(1, 2, 3, 4))
        if gt.ndim == 1:
            gt = gt[np.newaxis, ...]
    except Exception:
        print("No prediction at {}".format(rgb_detect))
        gt = None
        # if no ground truth, then pass
        continue

    try:
        dt = np.loadtxt(ev_detect, usecols=(2, 3, 4, 5))
        if dt.ndim == 1:
            dt = dt[np.newaxis, ...]
    except Exception:
        print("No prediction at {}".format(ev_detect))
        dt = None

    fig = plt.figure(figsize=(12, 5))
    ax = plt.subplot(121)
    plt.imshow(rgb_img)
    plt.axis("off")
    if gt is not None:
        for box in gt:
            box = box.astype(int)
            box[box < 0] = 0
            rect = patches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                linewidth=3, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    ax1 = plt.subplot(122)
    plt.imshow(ev_img)
    plt.axis("off")
    if gt is not None:
        for box in gt:
            box = box.astype(int)
            box[box < 0] = 0
            rect = patches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                linewidth=3, edgecolor='r', facecolor='none')
            ax1.add_patch(rect)
    if dt is not None:
        for box in dt:
            box = box.astype(int)
            box[box < 0] = 0
            rect = patches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                linewidth=3, edgecolor='blue', facecolor='none')
            ax1.add_patch(rect)

    plt.tight_layout()

    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    if out_writer is None:
        out_writer = video_writer(
            args.video_output_path, data.shape[0], data.shape[1],
            frame_rate=args.frame_rate)

    out_writer.write(data)

    cv2.imshow("V2E Visualize", data)
    cv2.waitKey(1)

    plt.cla()
    fig.clear()
    plt.close()
    del fig

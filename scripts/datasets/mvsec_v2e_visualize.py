"""Visualize V2E, perhaps selection as well.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

import argparse
import os
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from v2e_exps.utils import expandpath

parser = argparse.ArgumentParser()

parser.add_argument("--data_root", type=expandpath)

args = parser.parse_args()

frame_list = sorted(
    glob.glob(
        os.path.join(args.data_root, "*.npz")))

for file_path in frame_list:
    # load events
    data = np.load(file_path)

    img = data["img"]
    #  img = data["img"].astype(np.int32)
    #  img[img == 255] = 0
    #  img = np.clip(img-20, 0, None)
    #  img = img.astype(np.uint8)

    ev_img = data["ev_img"]

    fig = plt.figure(figsize=(8, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    ev_img_present = ev_img.mean(axis=0)
    sns.heatmap(ev_img_present, cbar=False, square=True, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 3)

    nonzero_ev = (ev_img != 0)
    select = ev_img[nonzero_ev]
    robust_max = np.percentile(ev_img.flatten(), 99)
    robust_min = np.percentile(ev_img.flatten(), 1)

    plt.hist(select.flatten(), bins=100)
    plt.title("Robust Max {:.2f} and Min {:.2f}".format(
        robust_max, robust_min))

    plt.subplot(2, 2, 4)

    plt.hist(img.flatten(), bins=50)
    plt.title("Intensity Dist")

    plt.tight_layout()

    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    cv2.imshow("V2E Visualize", data)
    cv2.waitKey(1)

    plt.cla()
    fig.clear()
    plt.close()
    del fig

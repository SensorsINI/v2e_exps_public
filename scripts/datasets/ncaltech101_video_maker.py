"""N-Caltech101 Video maker.

This file makes Caltech101 Luma frames by using published
ncaltech_syn_images.

The data source is here:
https://github.com/uzh-rpg/rpg_vid2e

Each video has 161 frames, according to the provided timestamps,
the frame interval is:
298507462/160 = 1865671.6375 nanos

therefore, the intrinsic frame rate is:
1/(1865671.6375/1e9) = 536.000001233 FPS

The videos in the script are rendered at 30 FPS,
the input slowmotion factor is:
536.000001233/30 = 17.866666708 times

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

import argparse
import os
import glob

import subprocess

parser = argparse.ArgumentParser()

parser.add_argument("--data_root", type=str)
parser.add_argument("--output_root", type=str)

args = parser.parse_args()

valid_folders = sorted(
    glob.glob(
        os.path.join(args.data_root, "*", "image_*")))
valid_folders = [x for x in valid_folders if ".npz" not in x]

for folder in valid_folders:
    assert os.path.isdir(folder)

    video_base_name = os.path.basename(folder)
    parent_folder_name = os.path.dirname(folder)

    images_list = os.path.join(folder, "images", "*.png")

    # target output folder
    target_folder_name = parent_folder_name.replace(
        args.data_root, args.output_root)
    if not os.path.isdir(target_folder_name):
        os.makedirs(target_folder_name)

    target_file_name = os.path.join(
        target_folder_name, video_base_name+".mp4")

    # run subprocess to create videos
    subprocess.run(
        ["ffmpeg", "-framerate", "30",
         "-pattern_type", "glob", "-i", images_list,
         "-c:v", "libx265", "-x265-params", "lossless=1",
         target_file_name])

    print("Saving to {}".format(target_file_name))

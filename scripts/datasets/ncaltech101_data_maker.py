"""V2E N-Caltech101 dataset maker.

Use V2E to simulate the entire dataset.

For Ideal range:
    - threshold: 0.2-0.5
    - sigma: 0.03-0.05
    - cutoff_hz: 0
    - leak_rate_hz: 0
    - shot_noise_rate_hz: 0

For bright light range:
    - threshold: 0.2-0.5
    - sigma: 0.03-0.05
    - cutoff_hz: 80-120
    - leak_rate_hz: 0.1
    - shot_noise_rate_hz: 0.1-5 Hz

For low light range:
    - threshold: 0.2-0.5
    - sigma: 0.03-0.05
    - cutoff_hz: 20-60
    - leak_rate_hz: 0.1
    - shot_noise_rate_hz: 10-30 Hz

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

import argparse
import os
import glob
import subprocess
import random
import json

parser = argparse.ArgumentParser()

parser.add_argument("--data_root", type=str)
parser.add_argument("--output_root", type=str)
parser.add_argument("--dataset_config", type=str,
                    help="'ideal', 'bright', 'dark'")

args = parser.parse_args()

# set configs
if args.dataset_config == "ideal":
    thre_low, thre_high = 0.05, 0.5
    sigma_low, sigma_high = 0, 0
    cutoff_hz_low, cutoff_hz_high = 0, 0
    leak_rate_hz_low, leak_rate_hz_high = 0, 0
    shot_noise_rate_hz_low, shot_noise_rate_hz_high = 0, 0
elif args.dataset_config == "bright":
    thre_low, thre_high = 0.05, 0.5
    sigma_low, sigma_high = 0.03, 0.05
    cutoff_hz_low, cutoff_hz_high = 200, 200
    leak_rate_hz_low, leak_rate_hz_high = 0.1, 0.5
    shot_noise_rate_hz_low, shot_noise_rate_hz_high = 0, 0
elif args.dataset_config == "dark":
    thre_low, thre_high = 0.05, 0.5
    sigma_low, sigma_high = 0.03, 0.05
    cutoff_hz_low, cutoff_hz_high = 10, 100
    leak_rate_hz_low, leak_rate_hz_high = 0, 0
    shot_noise_rate_hz_low, shot_noise_rate_hz_high = 1, 10

# get root folder list
valid_folders = sorted(
    glob.glob(
        os.path.join(args.data_root, "*", "image_*")))
valid_folders = [x for x in valid_folders if ".npz" not in x]

params_collector = {}

for folder in valid_folders:
    out_filename = os.path.basename(folder)+".h5"
    out_folder = os.path.dirname(folder)
    out_folder = out_folder.replace(args.data_root, args.output_root)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    folder = os.path.join(folder, "images")

    # configure paramters
    thres = random.uniform(thre_low, thre_high)
    # sigma should be about 15%~25% range as low and high
    # threshold higher than 0.2: 0.03-0.05
    # threshold lower than 0.2: 15%~25%
    #  sigma = random.uniform(sigma_low, sigma_high)
    sigma = random.uniform(
        min(thres*0.15, sigma_low), min(thres*0.25, sigma_high)) \
        if args.dataset_config != "ideal" else 0

    leak_rate_hz = random.uniform(leak_rate_hz_low, leak_rate_hz_high)
    shot_noise_rate_hz = random.uniform(
        shot_noise_rate_hz_low, shot_noise_rate_hz_high)

    if args.dataset_config == "dark":
        # cutoff hz follows shot noise config
        cutoff_hz = shot_noise_rate_hz*10
    else:
        cutoff_hz = random.uniform(cutoff_hz_low, cutoff_hz_high)

    params_collector[os.path.join(out_folder, out_filename)] = {
        "thres": thres,
        "sigma": sigma,
        "cutoff_hz": cutoff_hz,
        "leak_rate_hz": leak_rate_hz,
        "shot_noise_rate_hz": shot_noise_rate_hz}

    # dump bias configs all the time
    with open(os.path.join(args.output_root,
                           "dvs_params_settings.json"), "w") as f:
        json.dump(params_collector, f, indent=4)

    v2e_command = [
        "v2e.py",
        "-i", folder,
        "-o", out_folder,
        "--overwrite",
        "--unique_output_folder", "false",
        "--no_preview",
        "--skip_video_output",
        "--disable_slomo",
        "--pos_thres", "{}".format(thres),
        "--neg_thres", "{}".format(thres),
        "--sigma_thres", "{}".format(sigma),
        "--cutoff_hz", "{}".format(cutoff_hz),
        "--leak_rate_hz", "{}".format(leak_rate_hz),
        "--shot_noise_rate_hz", "{}".format(shot_noise_rate_hz),
        "--input_frame_rate", "30",
        "--input_slowmotion_factor", "17.866666708",
        "--dvs_h5", out_filename,
        "--dvs_aedat2", "None",
        "--dvs_text", "None",
        "--dvs_exposure", "duration", "0.001",
        "--auto_timestamp_resolution", "false"]

    subprocess.run(v2e_command)

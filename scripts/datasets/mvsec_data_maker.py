"""Synthesis MVSEC data.


- threshold: 0.54
- sigma: 0.03-0.05
- cutoff_hz: 0
- leak_rate_hz: 0.1-0.5 Hz
- shot_noise_rate_hz: 1-10Hz

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

import argparse
import os
import glob
import subprocess
import random
import json

import cv2
import numpy as np
import h5py

import matplotlib.pyplot as plt

from v2e_exps.utils import expandpath
from v2e_exps.utils import make_dvs_frame
from v2ecore.v2e_utils import video_writer

parser = argparse.ArgumentParser()

parser.add_argument("--data_root", type=str)
parser.add_argument("--output_root", type=str)
parser.add_argument("--start", type=float, default=0)
parser.add_argument("--stop", type=float, default=5)
parser.add_argument("--slomo_model", type=expandpath)
parser.add_argument("--visualize", action="store_true",
                    help="to compare with reference")
parser.add_argument("--vis_frame", action="store_true")
parser.add_argument("--reference_event_file", type=expandpath,
                    default=".")
parser.add_argument("--h5_filename", type=str, default="v2e_mvsec_day_2.h5")
parser.add_argument("--cutoff_hz", type=float, default=0)
parser.add_argument("--video_out_path", type=str)

args = parser.parse_args()

# synthesis parameters
on_thres = 0.73
off_thres = 0.43
#  sigma = random.uniform(0.03, 0.05)
#  cutoff_hz = 0
#  leak_rate_hz = random.uniform(0.1, 0.5)
#  shot_noise_rate_hz = random.uniform(0.5, 5)

# no noise, no nothing
sigma = 0.03
#  cutoff_hz = 0
#  leak_rate_hz = 0
#  shot_noise_rate_hz = 0
leak_rate_hz = 0.5
shot_noise_rate_hz = 2

if not args.visualize:
    v2e_command = [
        "v2e.py",
        "-i", args.data_root,
        "-o", args.output_root,
        "--overwrite",
        "--unique_output_folder", "false",
        "--no_preview",
        "--pos_thres", "{}".format(on_thres),
        "--neg_thres", "{}".format(off_thres),
        "--sigma_thres", "{}".format(sigma),
        "--cutoff_hz", "{}".format(args.cutoff_hz),
        "--leak_rate_hz", "{}".format(leak_rate_hz),
        "--shot_noise_rate_hz", "{}".format(shot_noise_rate_hz),
        "--input_frame_rate", "43.745746932880046",
        "--dvs_h5", args.h5_filename,
        "--dvs_aedat2", "None",
        "--dvs_text", "None",
        "--dvs_exposure", "duration", "0.03",
        "--auto_timestamp_resolution", "True",
        #  "--dvs_emulator_seed", "42",
        "--start_time", "{}".format(args.start),
        "--stop_time", "{}".format(args.stop),
        #  "--disable_slomo",
        "--slomo_model", args.slomo_model,
        "--skip_video_output",
        ]

    subprocess.run(v2e_command)

if args.vis_frame:
    # V2E Data
    v2e_events_path = os.path.join(args.output_root, args.h5_filename)
    assert os.path.isfile(v2e_events_path)

    v2e_data = h5py.File(v2e_events_path, "r")
    v2e_events = v2e_data["events"][()]
    v2e_events[:, 0] -= v2e_events[0, 0]
    v2e_events = v2e_events.astype("float32")
    v2e_events[:, 0] /= 1e6

    v2e_event_count = v2e_events.shape[0]

    event_rate = v2e_event_count/(v2e_events[-1, 0]-v2e_events[0, 0])

    print("Event Rate: {} events/sec".format(event_rate))

    start_event_idx = 0
    increment = 20000
    end_event_idx = v2e_event_count-1

    out_writer = None

    while (start_event_idx+increment < end_event_idx):
        # select events
        selected_data = v2e_data["events"][
            start_event_idx:start_event_idx+increment]

        dvs_frame = make_dvs_frame(
            selected_data, height=260, width=346, color=False, clip=3)

        fig = plt.figure()
        plt.imshow(dvs_frame, cmap="gray")

        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        start_event_idx += increment

        if out_writer is None:
            out_writer = video_writer(
                os.path.join(args.output_root, args.video_out_path),
                data.shape[0], data.shape[1],
                frame_rate=30)

        out_writer.write(data)

        #  cv2.imshow("V2E Visualize", data)
        #  cv2.waitKey(10)

        plt.cla()
        fig.clear()
        plt.close()
        del fig
        print("Plotting {}/{}".format(start_event_idx, end_event_idx))

    quit()

# CAUTION: make sure the data is not so large
if args.visualize:
    # V2E Data
    v2e_events_path = os.path.join(args.output_root, args.h5_filename)
    assert os.path.isfile(v2e_events_path)

    v2e_data = h5py.File(v2e_events_path, "r")
    v2e_events = v2e_data["events"][()]
    v2e_events[:, 0] -= v2e_events[0, 0]
    v2e_events = v2e_events.astype("float32")
    v2e_events[:, 0] /= 1e6

    v2e_event_count = v2e_events.shape[0]

    # Reference data
    assert os.path.isfile(args.reference_event_file)
    ref_data = h5py.File(args.reference_event_file, "r")

    event_dataset = ref_data["davis"]["left"]["events"]
    event_img_inds = ref_data["davis"]["left"][
        "image_raw_event_inds"][()]
    frame_ts = ref_data["davis"]["left"]["image_raw_ts"][()]

    # find out the range
    frame_ts -= frame_ts[0]
    start_idx = np.searchsorted(frame_ts, args.start, side="right")
    stop_idx = np.searchsorted(frame_ts, args.stop, side="right")

    start_event_idx = event_img_inds[start_idx]
    stop_event_idx = event_img_inds[stop_idx]

    dvs_events = event_dataset[start_event_idx:stop_event_idx][()]
    # MVSEC has a event arrangement of (x, y, t, p), change it to
    # (t, x, y, p)
    dvs_events = dvs_events[:, [2, 0, 1, 3]]
    # zero time stamp
    dvs_events[:, 0] -= dvs_events[0, 0]

    #  dvs_events = select_events_in_roi(dvs_events, args.x, args.y)
    ref_event_count = dvs_events.shape[0]

    # calculating number of ON and Off events
    dvs_on_count = np.count_nonzero((dvs_events[:, -1] == 1))
    dvs_off_count = ref_event_count-dvs_on_count

    print("Start Frame Time: {} at {}".format(
          frame_ts[start_idx], start_event_idx))
    print("Stop Frame Time: {} at {}".format(
          frame_ts[stop_idx], stop_event_idx))
    print("Number of reference events: {}".format(ref_event_count))
    print("Number of V2E events: {}".format(v2e_event_count))

    # plot event distribution
    print("Reference Duration: {:.4f}".format(dvs_events[-1, 0]))
    print("V2E Duration: {:.4f}".format(v2e_events[-1, 0]))

    v2e_on_events_idx = (v2e_events[:, 3] == 1)
    v2e_off_events_idx = np.logical_not(v2e_on_events_idx)

    v2e_on_events = v2e_events[v2e_on_events_idx, :]
    v2e_off_events = v2e_events[v2e_off_events_idx, :]

    ref_on_events_idx = (dvs_events[:, 3] == 1)
    ref_off_events_idx = np.logical_not(ref_on_events_idx)

    ref_on_events = dvs_events[ref_on_events_idx, :]
    ref_off_events = dvs_events[ref_off_events_idx, :]

    plt.figure(figsize=(18, 12))
    plt.subplot(2, 1, 1)
    plt.hist([v2e_on_events[:, 0],
              ref_on_events[:, 0]], bins=300,
             label=["No. V2E ON Events {}".format(v2e_on_events.shape[0]),
                    "No. MVSEC ON Events {}".format(ref_on_events.shape[0])])

    plt.legend(fontsize=16)
    plt.xlabel("Time in s", fontsize=16)
    plt.ylabel("Event count", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.subplot(2, 1, 2)
    plt.hist([v2e_off_events[:, 0],
              ref_off_events[:, 0]], bins=300,
             label=["No. V2E OFF Events {}".format(v2e_off_events.shape[0]),
                    "No. MVSEC OFF Events {}".format(ref_off_events.shape[0])])

    plt.legend(fontsize=16)
    plt.xlabel("Time in s", fontsize=16)
    plt.ylabel("Event count", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

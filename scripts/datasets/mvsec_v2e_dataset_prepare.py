"""Prepare train and valid for V2E MVSEC recording.

This script also does some plotting and calculating
statistics.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

import argparse
import glob
import os
import pickle

import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt

from v2ecore.v2e_utils import video_writer
from v2e_exps.utils import expandpath
from v2e_exps.utils import find_nearest
from v2e_exps.utils import make_dvs_frame
from v2e_exps.compute_voxel import events_to_voxel_grid


parser = argparse.ArgumentParser()

parser.add_argument("--v2e_mvsec_data_path", type=expandpath)
parser.add_argument("--ori_mvsec_data_path", type=expandpath)
parser.add_argument("--mvsec_frame_data_path", type=expandpath)
parser.add_argument("--v2e_inds_save_path", type=expandpath)
parser.add_argument("--show", action="store_true")
parser.add_argument("--out_video_path", type=expandpath, default="")
parser.add_argument("--frame_rate", type=int, default=43)

# data generation params
parser.add_argument("--data_output_root", type=expandpath)
parser.add_argument("--num_events", type=int, default=25000)
parser.add_argument("--num_slices", type=int, default=10)


args = parser.parse_args()

# open v2e mvsec data
v2e_data = h5py.File(args.v2e_mvsec_data_path, "r")
v2e_events = v2e_data["events"]

v2e_time = (v2e_events[-1, 0]-v2e_events[0, 0])/1e6

print("V2E MVSEC Number of Events: {}".format(v2e_events.shape[0]))
print("V2E MVSEC Duration: {}s".format(v2e_time))

# open mvsec data
mvsec_data = h5py.File(args.ori_mvsec_data_path, "r")
mvsec_events = mvsec_data["davis"]["left"]["events"]
mvsec_img_inds = mvsec_data["davis"]["left"][
    "image_raw_event_inds"][()]

mvsec_time = (mvsec_events[-1, 2]-mvsec_events[0, 2])

print("Original MVSEC Number of Events: {}".format(mvsec_events.shape[0]))
print("Original MVSEC Duration: {}s".format(mvsec_time))

# mvsec frame time
frame_ts = mvsec_data["davis"]["left"]["image_raw_ts"][()]
frame_ts -= frame_ts[0]  # convert into ms
frame_ts *= 1e6

# select time, ignore the ones that are larger than total duration
# TODO: WARNING: FOR NIGHT ONLY
frame_ts = np.array([ts for ts in frame_ts if ts < v2e_time*1e6])

frame_list = sorted(
    glob.glob(
        os.path.join(args.mvsec_frame_data_path, "*.png")))

print("Number of MVSEC Frames: {}".format(len(frame_list)))
print("Number of MVSEC Frame Ts: {}".format(len(frame_ts)))

# build up a event index array
v2e_event_inds = []
out_writer = None

start_idx = 0
# skipping at least the first frame since it does not represent
# any events
skip_start = 1

for idx, frame_t in enumerate(frame_ts[skip_start:-1]):
    # TODO: WARNING: FOR NIGHT ONLY
    # skip for adapting frame rate
    if idx % 2 != 0:
        continue

    if not os.path.isfile(args.v2e_inds_save_path) or args.show:
        nearest_value_idx = find_nearest(
            v2e_events, start_idx, frame_t)

        start_idx = nearest_value_idx

        frame_path = frame_list[idx+skip_start]
        v2e_event_inds.append([frame_path, nearest_value_idx, frame_t])

        print("Frame {}:".format(idx+skip_start))
        print("Frame Ts: {}".format(frame_t))
        print("Event TS: {}".format(v2e_events[nearest_value_idx, 0]))
    else:
        print("Index calculated before.")
        break

    if args.show:

        aps_img = cv2.imread(frame_path)

        # V2E Events
        dvs_events = v2e_events[
            max(0, nearest_value_idx-10000):
            min(nearest_value_idx+10000, v2e_events.shape[0])][()]

        dvs_img = make_dvs_frame(
            dvs_events, height=260, width=346, color=False, clip=3)

        # original MVSEC events
        ori_ind = mvsec_img_inds[idx+skip_start]
        ori_events = mvsec_events[
            max(0, ori_ind-10000):
            min(ori_ind+10000, mvsec_events.shape[0])][()]
        ori_events = ori_events[:, [2, 0, 1, 3]]

        ori_dvs_img = make_dvs_frame(
            ori_events, height=260, width=346, color=False, clip=3)

        fig = plt.figure(figsize=(15, 5))
        fig.add_subplot(3, 3, (1, 4))
        plt.imshow(aps_img)
        plt.title("APS Frame")
        plt.axis("off")
        fig.add_subplot(3, 3, (2, 5))
        plt.imshow(dvs_img, cmap="gray")
        plt.title("V2E MVSEC DVS Frame")
        plt.axis("off")
        fig.add_subplot(3, 3, (3, 6))
        plt.imshow(ori_dvs_img, cmap="gray")
        plt.title("Original MVSEC DVS Frame")
        plt.axis("off")

        fig.add_subplot(3, 3, (7, 9))
        ts = frame_ts[:idx+skip_start+1]/1e6
        y = np.zeros_like(ts)
        plt.plot(ts, y, color="blue",
                 linewidth=5)
        plt.plot(frame_ts[idx+skip_start]/1e6, 0, color="red",
                 linewidth=5, marker="o", markersize=10)
        plt.xlabel("Time (s)")
        plt.xlim([frame_ts[skip_start]/1e6, frame_ts[-1]/1e6])
        plt.yticks([])
        plt.tight_layout(pad=0.5)

        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        if out_writer is None:
            out_writer = video_writer(
                args.out_video_path, data.shape[0], data.shape[1],
                frame_rate=args.frame_rate)

        out_writer.write(data)

        #  cv2.imshow("V2E MVSEC Show", data)
        #  cv2.waitKey(1)

        plt.cla()
        fig.clear()
        plt.close()
        del fig

if out_writer is not None:
    out_writer.release()

# save if necessary
if not os.path.isfile(args.v2e_inds_save_path):
    with open(args.v2e_inds_save_path, "wb") as f:
        pickle.dump(v2e_event_inds, f)

if not args.show:
    with open(args.v2e_inds_save_path, "rb") as f:
        v2e_event_inds = pickle.load(f)

    # prepare the training pairs, no need testing pairs
    num_v2e_events = v2e_events.shape[0]
    num_frame_generated = 0

    if not os.path.isdir(args.data_output_root):
        os.makedirs(args.data_output_root)

    for frame_idx, frame_item in enumerate(v2e_event_inds[1:-1]):
        # adding this because we skipped the first frame.
        true_frame_idx = frame_idx+1
        frame_path, frame_ind, frame_t = frame_item

        ev_vol_start_idx = max(frame_ind-args.num_events//2, 0)
        ev_vol_end_idx = min(frame_ind+args.num_events//2, num_v2e_events)

        # condition on too few events
        if (ev_vol_end_idx - ev_vol_start_idx) < args.num_events:
            print("Bad frame {}, not enough events".format(
                  frame_idx))
            continue

        # if the start and the end reached another frame, then we dump this
        # means too few events here
        ev_start_ts = v2e_events[ev_vol_start_idx, 0]
        ev_end_ts = v2e_events[ev_vol_end_idx, 0]

        curr_frame_ts = frame_t
        pre_frame_ts = v2e_event_inds[true_frame_idx-1][2]
        next_frame_ts = v2e_event_inds[true_frame_idx+1][2]

        pre_frame_ind = v2e_event_inds[true_frame_idx-1][1]
        next_frame_ind = v2e_event_inds[true_frame_idx+1][1]

        #  print("Pre: ", frame_ind-pre_frame_ind)
        #  print("Next: ", next_frame_ind-frame_ind)
        #  print(pre_frame_ind, frame_ind, next_frame_ind)
        #  if (next_frame_ind-pre_frame_ind) < args.num_events:
        if ev_start_ts < pre_frame_ts or ev_end_ts > next_frame_ts:
            print("Bad frame {}, too long".format(
                frame_idx))
            print("Pre ts {} {} {}".format(
                pre_frame_ts, ev_start_ts, ev_start_ts < pre_frame_ts))
            print("Next ts {} {} {}".format(
                next_frame_ts, ev_end_ts, ev_end_ts > pre_frame_ts))
            print("Num events: {}".format(next_frame_ind-pre_frame_ind))
            continue

        # select events
        candidate_events = v2e_events[
            ev_vol_start_idx:ev_vol_end_idx][()].astype("float32")
        candidate_events[:, 0] /= 1e6

        # get voxel grid
        voxel = events_to_voxel_grid(
            candidate_events.copy(), num_bins=args.num_slices,
            width=346, height=260)

        # normalization
        nonzero_ev = (voxel != 0)
        num_nonzeros = nonzero_ev.sum()

        # has events
        if num_nonzeros > 0:
            mean = voxel.sum()/num_nonzeros
            stddev = np.sqrt((voxel**2).sum()/num_nonzeros-mean**2)
            mask = nonzero_ev.astype("float32")
            voxel = mask*(voxel-mean)/stddev

        # output path
        output_path = os.path.join(
            args.data_output_root, "v2e_frame_ev_pair_{:05d}".format(
                num_frame_generated+1))

        # frame
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

        # save data
        np.savez(output_path+".npz", ev_img=voxel,
                 img=frame)

        num_frame_generated += 1

        print("Generated data for frame {} at {}".format(
            frame_idx, num_frame_generated))

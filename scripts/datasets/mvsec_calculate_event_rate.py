"""Calculate Event Rate.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

import argparse
import pickle
import os

import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import ks_2samp
from scipy.stats import ttest_ind
from scipy.stats import epps_singleton_2samp
import h5py

import matplotlib.pyplot as plt
import seaborn as sns

from v2e_exps.utils import expandpath

parser = argparse.ArgumentParser()

parser.add_argument("--v2e_event_file", type=expandpath)
parser.add_argument("--disable_v2e", action="store_true")
parser.add_argument("--v2e_inds_save_path", type=expandpath)
parser.add_argument("--reference_event_file", type=expandpath,
                    default=".")

args = parser.parse_args()

if not args.disable_v2e:
    # V2E Data
    v2e_data = h5py.File(args.v2e_event_file, "r")
    v2e_events = v2e_data["events"]
    #  v2e_events[:, 0] -= v2e_events[0, 0]
    #  v2e_events = v2e_events.astype("float32")
    #  v2e_events[:, 0] /= 1e6

    v2e_event_count = v2e_events.shape[0]

    with open(args.v2e_inds_save_path, "rb") as f:
        v2e_event_inds = pickle.load(f)

    v2e_rate_collector = []
    for index, (ind_pre, ind_post) in enumerate(
            zip(v2e_event_inds[:-1], v2e_event_inds[1:])):

        num_events = ind_post[1]-ind_pre[1]
        time_pre = v2e_events[ind_pre[1], 0]
        time_post = v2e_events[ind_post[1], 0]

        duration = (time_post-time_pre)/1e6
        event_rate = num_events/duration

        v2e_rate_collector.append(event_rate)
        print("{}/{}".format(index, len(v2e_event_inds)-1), end="\r")

    print("Average Event Rate: ", np.mean(v2e_rate_collector))
    print("STD Event Rate: ", np.std(v2e_rate_collector))

# Reference data
ref_data = h5py.File(args.reference_event_file, "r")

event_dataset = ref_data["davis"]["left"]["events"]
event_img_inds = ref_data["davis"]["left"][
    "image_raw_event_inds"][()]
frame_ts = ref_data["davis"]["left"]["image_raw_ts"][()]
frame_ts -= frame_ts[0]

ref_frames = ref_data["davis"]["left"]["image_raw"]

ref_rate_collector = []
event_img_inds = event_img_inds[1:-1]  # remove first and last as in V2E
for index, (ind_pre, ind_post) in enumerate(
        zip(event_img_inds[:-1], event_img_inds[1:])):

    num_events = ind_post-ind_pre
    time_pre = event_dataset[ind_pre, 2]
    time_post = event_dataset[ind_post, 2]

    duration = time_post-time_pre
    event_rate = num_events/duration

    ref_rate_collector.append(event_rate)

    print("{}/{}".format(index, event_img_inds.shape[0]-1), end="\r")

print("Ref Average Event Rate: ", np.mean(ref_rate_collector))
print("Ref STD Event Rate: ", np.std(ref_rate_collector))

ref_intensity_collector = []
for frame_idx in range(ref_frames.shape[0]):
    frame = ref_frames[frame_idx][()]

    frame[frame == 255] = 0

    avg_intensity = frame.mean()
    ref_intensity_collector.append(avg_intensity)

    print("Frame {}/{}".format(frame_idx, ref_frames.shape[0]-1), end="\r")

print("Frame Intensity Mean {:.2f} STD {:.2f}".format(
    np.mean(ref_intensity_collector),
    np.std(ref_intensity_collector)))

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, (1, 2))
plt.plot(frame_ts[1:-2], ref_rate_collector, label="real events")
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("Event Rate", fontsize=16)
plt.legend(fontsize=16)
plt.grid()

plt.subplot(1, 3, 3)
sns.violinplot(data=[ref_rate_collector], split=True)
plt.xticks([0], ["real events"], fontsize=16)
plt.ylabel("Event Rate", fontsize=16)

plt.show()

if not args.disable_v2e:
    print("R2", r2_score(ref_rate_collector, v2e_rate_collector))

    v2e_rate_collector = np.array(v2e_rate_collector)
    #  diff_to_reduce = v2e_rate_collector.mean()-np.mean(ref_rate_collector)
    v2e_rate_collector -= 224900
    v2e_rate_collector = np.clip(v2e_rate_collector, 0, None)

    ref_rate_normed = np.array(ref_rate_collector)/np.sum(ref_rate_collector)
    v2e_rate_normed = v2e_rate_collector/v2e_rate_collector.sum()
    print("KS", ks_2samp(ref_rate_normed, v2e_rate_normed))
    print("t-Test", ttest_ind(ref_rate_normed, v2e_rate_normed))

    # visualize
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, (1, 2))
    plt.plot(frame_ts[1:-2], v2e_rate_collector, label="V2E events")
    plt.plot(frame_ts[1:-2], ref_rate_collector, label="real events",
             alpha=0.7)
    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Event Rate", fontsize=16)
    plt.legend(fontsize=16)
    plt.grid()

    #  plt.subplot(1, 3, 2)
    #  plt.hist([v2e_rate_collector, ref_rate_collector], bins=100)
    #  plt.xlabel("Event Rate", fontsize=16)
    #  plt.ylabel("Count", fontsize=16)
    #  plt.grid()

    plt.subplot(1, 3, 3)
    sns.violinplot(data=[v2e_rate_collector, ref_rate_collector], split=True)
    plt.xticks([0, 1], ["V2E events", "real events"], fontsize=16)
    plt.ylabel("Event Rate", fontsize=16)

    plt.tight_layout()
    plt.savefig("mvsec_data_compare.pdf", dpi=300)

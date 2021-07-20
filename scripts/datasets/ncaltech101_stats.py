"""Calculate some N-Caltech 101 Statistics.

1. NCaltech
2. RPG Events
3. V2E Events

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

import argparse
import os
import glob

import numpy as np

from v2e_exps.utils import load_ncaltech_events_bin
from v2e_exps.utils import load_rpg_events
from v2e_exps.utils import load_v2e_events

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str)
parser.add_argument("--dataset", type=str)

args = parser.parse_args()

if args.dataset == "ncaltech":
    valid_files = sorted(
        glob.glob(
            os.path.join(args.data_root, "*", "image_*.bin")))
elif args.dataset == "rpg":
    valid_files = sorted(
        glob.glob(
            os.path.join(args.data_root, "*", "image_*.npz")))
elif args.dataset == "v2e":
    valid_files = sorted(
        glob.glob(
            os.path.join(args.data_root, "*", "image_*.h5")))

load_functions = {
    "ncaltech": load_ncaltech_events_bin,
    "rpg": load_rpg_events,
    "v2e": load_v2e_events}

load_data = load_functions[args.dataset]

print("Number of files {}".format(len(valid_files)))

event_rates = []

for idx, filename in enumerate(valid_files):
    events = load_data(filename)

    num_events = events.shape[0]
    total_time_in_sec = (events[-1, 0]-events[0, 0])/1e6

    event_rate = num_events/total_time_in_sec

    event_rates.append(event_rate)

    print("File {} {}/{} Num Events: {} Time: {} Rate: {}".format(
        filename, idx+1, len(valid_files), num_events, total_time_in_sec,
        event_rate))

print("Average event rate: {}".format(np.mean(event_rates)))
print("STD event rate: {}".format(np.std(event_rates)))

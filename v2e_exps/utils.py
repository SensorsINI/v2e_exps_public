"""Utilities functions.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

import os
import random
import copy
import numpy as np
import h5py
import subprocess
import time
from threading import Thread, Event


def expandpath(path):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def make_unique_dir(path, idx=1):
    temp_path = os.path.join(path, "{:03d}".format(idx))

    if os.path.isdir(temp_path):
        return make_unique_dir(path, idx+1)

    return temp_path


def find_nearest(dataset, start_idx, search_value, search_gap=10000):
    num_events = dataset.shape[0]
    nearest_value_idx = 0

    for event_batch in range((num_events-start_idx) // search_gap):
        start_pos = start_idx+event_batch*search_gap
        end_pos = min(start_idx+(event_batch+1)*search_gap,
                      num_events)
        selected_events = dataset[start_pos:end_pos, 0]

        nearest_idx = np.searchsorted(
            selected_events, search_value, side="right")

        if nearest_idx != search_gap:
            nearest_value_idx = start_idx+event_batch*search_gap+nearest_idx
            break

    return nearest_value_idx


def pad_frame(frame, out_height, out_width):
    """Pad frame so that the output follows a specific aspect ratio.

    Mainly for padding N-caltech 101 data.

    The last two dimensions of the frame is height and width.
    """
    frame_height, frame_width = frame.shape[-2], frame.shape[-1]

    padding_list = [(0, 0)]*frame.ndim

    # pad according to height
    resize_ratio = out_height/frame_height
    new_out_width = int(out_width/resize_ratio)
    if new_out_width >= frame_width:
        total_pad = new_out_width-frame_width
        if total_pad // 2 == 1:
            pads = (total_pad//2, total_pad//2+1)
        else:
            pads = (total_pad//2, total_pad//2)

        padding_list[-1] = pads
    else:
        # padding according to width
        resize_ratio = out_width/frame_width
        new_out_height = int(out_height/resize_ratio)
        total_pad = new_out_height-frame_height

        if total_pad // 2 == 1:
            pads = (total_pad//2, total_pad//2+1)
        else:
            pads = (total_pad//2, total_pad//2)

        padding_list[-2] = pads

    frame = np.pad(frame, pad_width=padding_list,
                   mode="constant", constant_values=0)

    return frame


def corp_frame(frame, out_height, out_width):
    """corp frame based on the expected output height and width.

    Mainly serve for visualization purpose

    The last two dimensions of the frame is height and width.
    """
    frame_height = frame.shape[-2]
    frame_width = frame.shape[-1]

    if out_height >= out_width:
        final_width = int((out_width/out_height)*frame_height)
        frame = frame[
            ..., :, (frame_width-final_width)//2:(frame_width+final_width)//2]
    elif out_height < out_width:
        final_height = int((out_height/out_width)*frame_width)
        frame = frame[
            ...,
            (frame_height-final_height)//2:(frame_height+final_height)//2, :]

    return frame


def make_dvs_frame(events, height=None, width=None, color=True, clip=None):
    """Create a single frame.

    Mainly for visualization purposes

    # Arguments
    events : np.ndarray
        (t, x, y, p)
    x_pos : np.ndarray
        x positions
    """
    if height is None or width is None:
        height = events[:, 2].max()+1
        width = events[:, 1].max()+1

    histrange = [(0, v) for v in (height, width)]

    pol_on = (events[:, 3] == 1)
    pol_off = np.logical_not(pol_on)
    img_on, _, _ = np.histogram2d(
            events[pol_on, 2], events[pol_on, 1],
            bins=(height, width), range=histrange)
    img_off, _, _ = np.histogram2d(
            events[pol_off, 2], events[pol_off, 1],
            bins=(height, width), range=histrange)

    on_non_zero_img = img_on.flatten()[img_on.flatten() > 0]
    on_mean_activation = np.mean(on_non_zero_img)
    # on clip
    if clip is None:
        on_std_activation = np.std(on_non_zero_img)
        img_on = np.clip(
            img_on, on_mean_activation-3*on_std_activation,
            on_mean_activation+3*on_std_activation)
    else:
        img_on = np.clip(
            img_on, -clip, clip)

    # off clip
    off_non_zero_img = img_off.flatten()[img_off.flatten() > 0]
    off_mean_activation = np.mean(off_non_zero_img)
    if clip is None:
        off_std_activation = np.std(off_non_zero_img)
        img_off = np.clip(
            img_off, off_mean_activation-3*off_std_activation,
            off_mean_activation+3*off_std_activation)
    else:
        img_off = np.clip(
            img_off, -clip, clip)

    if color:
        frame = np.zeros((height, width, 3))
        frame[..., 0] = img_on
        frame[..., 1] = img_off

        frame /= np.abs(frame).max()
    else:
        frame = img_on-img_off
        frame -= frame.min()
        frame /= frame.max()

    return frame


def load_v2e_events(filename):
    """Load V2E Events, all HDF5 records."""
    assert os.path.isfile(filename)

    v2e_data = h5py.File(filename, "r")
    events = v2e_data["events"][()]

    return events


def load_rpg_events(filename):
    """Load RPG N-Caltech101 simulated events."""
    assert os.path.isfile(filename)

    rpg_events = np.load(filename)

    xyp = rpg_events["xyp"].astype(np.float32)
    t = rpg_events["t"]*1e6

    return np.append(t, xyp, axis=-1)


def load_ncaltech_events_bin(filename):
    """Load one binary N-Caltech recording.
    # Arguments
    filename : str
        the file name of the event binary file.
    """
    if not os.path.isfile(filename):
        raise ValueError("File {} does not exist".format(filename))

    # open data
    f = open(filename, "rb")
    raw_data = np.fromfile(f, dtype=np.uint8)

    f.close()
    raw_data = np.uint32(raw_data)

    # read  events
    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7  # bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | \
        (raw_data[4::5])

    events = np.concatenate(
        (all_ts[..., np.newaxis],
         all_x[..., np.newaxis],
         all_y[..., np.newaxis],
         all_p[..., np.newaxis]), axis=-1)

    return events


class DaemonTensorboard(Thread):
    def __init__(self, log_dir, port):
        super().__init__()
        self.validate_requirements(log_dir)
        self.log_dir = log_dir
        self.port = port
        self.event = Event()
        self.daemon = True

    @staticmethod
    def _cmd_exists(cmd):
        return any(
            os.access(os.path.join(path, cmd), os.X_OK)
            for path in os.environ["PATH"].split(os.pathsep)
        )

    @staticmethod
    def validate_requirements(log_dir):
        assert DaemonTensorboard._cmd_exists(
            'tensorboard'), 'TensorBoard not found'
        os.makedirs(log_dir, exist_ok=True)

    @staticmethod
    def kill_old():
        try:
            subprocess.check_output(
                ['killall', 'tensorboard'], stderr=subprocess.DEVNULL)
            print(
                'Killed some stale Tensorboard process before running '
                'a managed daemon')
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    def create_tensorboard_process(self):
        self.kill_old()
        pid = subprocess.Popen(
            ["tensorboard", "--logdir", self.log_dir, "--host",
             "localhost", "--port", str(self.port),
             "--samples_per_plugin", "images=1000"],
            stdout=open(
                os.path.join(self.log_dir, 'tensorboard_server.log'), 'a'),
            stderr=subprocess.STDOUT,
        )
        time.sleep(5)
        assert pid.poll() is None, 'TensorBoard launch failed (port occupied?)'
        return pid

    def run(self):
        pid = self.create_tensorboard_process()
        print(f'Running TensorBoard daemon on port {self.port}')

        while not self.event.is_set():
            time.sleep(1)
            assert pid.poll() is None, 'TensorBoard was killed'

        print('Stopping TensorBoard daemon')
        pid.terminate()
        pid.communicate()

    def stop(self):
        self.event.set()
        self.join()

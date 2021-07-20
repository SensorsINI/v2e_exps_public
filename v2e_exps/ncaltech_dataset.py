"""N-Caltech101 dataset routines.

This module consists of both event simulation routine and
training dataset routine.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomRotation
from torchvision.transforms import RandomAffine

from v2e_exps.compute_voxel import events_to_voxel_grid
from v2e_exps.utils import load_rpg_events
from v2e_exps.utils import load_ncaltech_events_bin
from v2e_exps.utils import load_v2e_events
from v2e_exps.utils import pad_frame


def prepare_voxel(
        events, num_voxel_bins, height, width, normalize=True,
        remove_outlier=False):
    # processing events
    events = events.astype("float32")
    events[:, 0] /= 1e6  # convert in to seconds instead of microseconds

    voxel = events_to_voxel_grid(
        events, num_bins=num_voxel_bins,
        height=height, width=width)

    if remove_outlier:
        robust_min = np.percentile(voxel.flatten(), 1)
        robust_max = np.percentile(voxel.flatten(), 99)
        voxel = np.clip(voxel, robust_min, robust_max)

    if normalize:
        # normalization
        nonzero_ev = (voxel != 0)
        num_nonzeros = nonzero_ev.sum()

        # has events
        if num_nonzeros > 0:
            mean = voxel.sum()/num_nonzeros
            stddev = np.sqrt((voxel**2).sum()/num_nonzeros-mean**2)
            mask = nonzero_ev.astype("float32")
            voxel = mask*(voxel-mean)/stddev

    return voxel


class V2ENCaltechData(Dataset):
    def __init__(self, data_root, file_list_file, extension,
                 num_voxel_bins=3, is_train=True,
                 augmentation=False):
        """V2E Caltech data.

        file_list_file: stores train and test list of files

        This function opens a single dataset from:
        - NCaltech
        - RPG events
        - V2E events

        """
        super(V2ENCaltechData, self).__init__()

        self.data_root = data_root
        self.extension = extension
        self.is_train = is_train

        # To handle the combined dataset case
        # specifically designed for V2E datasets
        # right now, we use "all" as a special indicator
        # for grasp all data roots
        # it's a fake & empty data root that exists
        if "all" in data_root:
            self.use_all_datasets = True
            self.ideal_data_root = self.data_root.replace("all", "ideal")
            self.bright_data_root = self.data_root.replace("all", "bright")
            self.dark_data_root = self.data_root.replace("all", "dark")
        elif "bnd" in data_root:
            # for bright + dark scenario
            self.use_all_datasets = True
            self.bright_data_root = self.data_root.replace("bnd", "bright")
            self.dark_data_root = self.data_root.replace("bnd", "dark")
        elif "addb" in data_root:
            # for bright + additional bright
            self.use_all_datasets = True
            self.bright_data_root = self.data_root.replace("addb", "bright")
            self.dark_data_root = self.data_root.replace("addb", "add_bright")
        else:
            self.use_all_datasets = False

        with open(file_list_file, "rb") as f:
            file_list = pickle.load(f, encoding="utf-8")

        if is_train:
            self.files = file_list["train"]

            self.all_files = []
            if self.use_all_datasets:
                # preappend in the list
                if "all" in data_root:
                    self.all_files += [
                        (os.path.join(self.ideal_data_root,
                                      file_data[0]+self.extension),
                         file_data[1])
                        for file_data in self.files]

                self.all_files += [
                    (os.path.join(self.bright_data_root,
                                  file_data[0]+self.extension),
                     file_data[1])
                    for file_data in self.files]
                self.all_files += [
                    (os.path.join(self.dark_data_root,
                                  file_data[0]+self.extension),
                     file_data[1])
                    for file_data in self.files]
        else:
            self.files = file_list["test"]

        self.height, self.width = 180, 240
        self.num_voxel_bins = num_voxel_bins

        # load data functions
        self.load_events = {
            ".npz": load_rpg_events,
            ".bin": load_ncaltech_events_bin,
            ".h5": load_v2e_events}

        # augumentation
        self.augmentation = augmentation
        self.hflip = RandomHorizontalFlip()
        self.rand_rotate = RandomRotation(15)
        self.rand_affine = RandomAffine(15, translate=(0.1, 0.1))

    def __getitem__(self, index):
        if self.use_all_datasets is True:
            filename, label = self.all_files[index]
        else:
            filename, label = self.files[index]
            # real file name
            filename = os.path.join(
                self.data_root, filename+self.extension)

        # load image
        events = self.load_events[self.extension](filename)

        if self.extension == ".bin":
            # original N-Caltech
            height, width = events[:, 2].max()+1, events[:, 1].max()+1
        else:
            height, width = self.height, self.width

        # processing events
        voxel = prepare_voxel(
            events, num_voxel_bins=self.num_voxel_bins,
            height=height, width=width,
            normalize=True,
            remove_outlier=True if self.extension == ".bin" else False)

        if self.extension == ".bin":
            # resize N-Caltech datset
            voxel = pad_frame(voxel, self.height, self.width)
            voxel = torch.tensor(voxel, dtype=torch.float32)
            voxel = resize(voxel, [self.height, self.width])
        else:
            voxel = torch.tensor(voxel, dtype=torch.float32)

        if self.augmentation:
            # data augmentation
            if self.is_train:
                voxel = self.hflip(voxel)
                #  voxel = self.rand_rotate(voxel)
                voxel = self.rand_affine(voxel)

        return voxel, label

    def __len__(self):
        if self.use_all_datasets:
            return len(self.all_files)
        else:
            return len(self.files)

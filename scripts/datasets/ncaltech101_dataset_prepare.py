"""This scripts focus on splitting N-Caltech 101 dataset.

Split to train and test dataset.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

import argparse
import os
import glob
import random
import pickle

parser = argparse.ArgumentParser()

parser.add_argument("--reference_data_folder", type=str)
parser.add_argument("--num_train_per_class", type=int, default=30)
parser.add_argument("--data_split_out_path", type=str)
parser.add_argument("--percentage_split", action="store_true")
parser.add_argument("--train_split", type=float, default=0.45)
parser.add_argument("--valid_split", type=float, default=0.30)
parser.add_argument("--test_split", type=float, default=0.25)

args = parser.parse_args()

classes_folders = sorted(
    glob.glob(os.path.join(args.reference_data_folder, "*")))
classes_folders = [x for x in classes_folders if os.path.isdir(x)]

train_dataset = []
valid_dataset = []
test_dataset = []

for idx, class_folder in enumerate(classes_folders):
    # get folder name
    folder_name = os.path.basename(class_folder)

    data_list = sorted(glob.glob(os.path.join(class_folder, "*.*")))
    data_list = [x for x in data_list if x[-4:] != ".txt"]
    data_names = [
        os.path.join(
            folder_name,
            os.path.splitext(os.path.basename(x))[0]) for x in data_list]

    file_idx = list(range(len(data_names)))
    random.shuffle(file_idx)

    if args.percentage_split:
        # using percentage split
        num_files = len(file_idx)
        num_train = int(num_files*args.train_split)
        num_valid = int(num_files*args.valid_split)
        num_test = num_files-num_train-num_valid

        train_data_names = sorted([
            data_names[x] for x in file_idx[:num_train]])
        valid_data_names = sorted([
            data_names[x] for x in file_idx[num_train:num_train+num_valid]])
        test_data_names = sorted([
            data_names[x] for x in file_idx[-num_test:]])

        # assign train and test split
        for name in train_data_names:
            train_dataset.append((name, idx))

        for name in valid_data_names:
            valid_dataset.append((name, idx))

        for name in test_data_names:
            test_dataset.append((name, idx))
    else:
        train_data_names = sorted([
            data_names[x] for x in file_idx[:args.num_train_per_class]])
        test_data_names = sorted([
            data_names[x] for x in file_idx[args.num_train_per_class:]])

        # assign train and test split
        for name in train_data_names:
            train_dataset.append((name, idx))

        for name in test_data_names:
            test_dataset.append((name, idx))

print("Train data:", len(train_dataset))
print("Valid data:", len(valid_dataset))
print("Test data:", len(test_dataset))
if args.percentage_split:
    with open(args.data_split_out_path, "wb") as f:
        pickle.dump({"train": train_dataset,
                     "valid": valid_dataset,
                     "test": test_dataset}, f)
else:
    with open(args.data_split_out_path, "wb") as f:
        pickle.dump({"train": train_dataset, "test": test_dataset}, f)

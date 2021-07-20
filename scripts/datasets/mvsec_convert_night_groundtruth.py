"""Groundtruth saved in XML, saved as compatible format in txt.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uch.ch
"""

import argparse
import os
import glob
import json

import xmltodict

from v2e_exps.utils import expandpath

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=expandpath)
parser.add_argument("--output_root", type=expandpath)

args = parser.parse_args()


file_list = sorted(
    glob.glob(
        os.path.join(args.data_root, "*.xml")))

if not os.path.isdir(args.output_root):
    os.makedirs(args.output_root)

for file_path in file_list:
    file_base = os.path.basename(file_path)
    output_path = os.path.join(
        args.output_root, file_base[:-4]+".txt")

    with open(file_path) as f:
        data = xmltodict.parse(f.read())

    objects = data["annotation"]["object"]

    #  print(output_path, len(objects))
    #  print(json.dumps(data,
    #                   indent=4, sort_keys=True))

    if type(objects) is list:
        for obj in objects:
            bndbox = obj["bndbox"]

            xmin = bndbox["xmin"]
            ymin = bndbox["ymin"]

            xmax = bndbox["xmax"]
            ymax = bndbox["ymax"]

            with open(output_path, "a+") as f:
                f.write("car {} {} {} {}\n".format(xmin, ymin, xmax, ymax))
    else:
        bndbox = objects["bndbox"]

        xmin = bndbox["xmin"]
        ymin = bndbox["ymin"]

        xmax = bndbox["xmax"]
        ymax = bndbox["ymax"]

        with open(output_path, "a+") as f:
            f.write("car {} {} {} {}\n".format(xmin, ymin, xmax, ymax))

    print("Write to {}".format(output_path))

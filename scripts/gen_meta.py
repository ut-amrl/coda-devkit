import os
import sys
import copy
import time
import json
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default="/robodata/CODa",
                    help="CODa directory")

# For imports
sys.path.append(os.getcwd())

from helpers.metadata import *

def main(args):
    indir = args.data_path

    """
    Metadata Generation Steps
    1. Generate Auto Statistics 
    2. Random divide Train, Test, Val Splits for Object Detection
    """
    assert os.path.isdir(indir), '%s does not exist for root directory' % indir

    outdir = os.path.join(indir, "metadata")
    if not os.path.exists(outdir):
        print("Metadata directory does not exist, creating at %s..."%outdir)
        os.mkdir(outdir)

    pc_subdir= os.path.join("3d_label", "os1")
    pc_fulldir = os.path.join(indir, pc_subdir)
    assert os.path.isdir(pc_fulldir), '%s does not exist for pc directory' % pc_fulldir
    traj_list = [traj for traj in os.listdir(pc_fulldir) if os.path.isdir(
        os.path.join(pc_fulldir, traj) )]
    traj_list = sorted(traj_list, key=lambda x: int(x), reverse=False)



    train_percent = 0.8
    val_percent = 0.0
    test_percent = 0.2
    rng = np.random.default_rng()
    for traj in traj_list:
        print("Creating metadata file for traj %s"%traj)
        metadata_dict = copy.deepcopy(METADATA_DICT)

        traj_subdir = os.path.join(pc_subdir, traj)
        traj_fulldir= os.path.join(indir, traj_subdir)

        bin_files = np.array([os.path.join(traj_subdir, bin_file) for bin_file in os.listdir(traj_fulldir) if bin_file.endswith(".json")])
        num_bin_files   = len(bin_files)
        num_train       = int(num_bin_files * train_percent)
        num_val         = int(num_bin_files * val_percent)

        #1
        indices = np.arange(0, len(bin_files), 1)
        rng.shuffle(indices)

        train, val, test    = indices[:num_train], indices[num_train:num_train+num_val], \
            indices[num_train+num_val:]

        #2
        metadata_dict["ObjectTracking"]["train"].extend(bin_files[train].tolist())
        metadata_dict["ObjectTracking"]["val"].extend(bin_files[val].tolist())
        metadata_dict["ObjectTracking"]["test"].extend(bin_files[test].tolist())

        metadata_dict["trajectory"] = int(traj)

        metadata_path = os.path.join(outdir, "%s.json"%traj)
        metadata_file = open(metadata_path, "w+")
        json.dump(metadata_dict, metadata_file, indent=4)



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
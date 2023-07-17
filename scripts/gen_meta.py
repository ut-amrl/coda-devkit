import os
import sys
import copy
import json
import time
import argparse

from os.path import join

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default="/robodata/arthurz/Datasets/CODa_dev",
                    help="CODa directory")
parser.add_argument('--size', default="sm",
                    help="CODa dataset size (tiny, sm, md, full)")

# For imports
sys.path.append(os.getcwd())

from helpers.metadata import *
from helpers.constants import *

rng = np.random.default_rng(seed=42)

from multiprocessing import Pool
import tqdm

def get_label_list(indir, label_dir, traj):
    traj = str(traj)
    pc_subdir= os.path.join(label_dir, "os1")
    traj_subdir = os.path.join(pc_subdir, traj)
    traj_fulldir= os.path.join(indir, traj_subdir)
    filetype = SENSOR_DIRECTORY_FILETYPES[pc_subdir]
    bin_files = np.array([os.path.join(traj_subdir, bin_file) for bin_file in os.listdir(traj_fulldir) if bin_file.endswith(filetype)])
    return bin_files

def gen_dataset_split(label_file_list, split_size=[0.7, 0.15, 0.15]):
    train_percent, val_percent, test_percent = split_size
    assert (train_percent+val_percent<1), "Train and validation percent should not be entire dataset..." 
    num_bin_files   = len(label_file_list)
    num_train       = int(num_bin_files * train_percent)
    num_val         = int(num_bin_files * val_percent)
    num_test        = int(num_bin_files * test_percent)

    #1
    indices = np.arange(0, len(label_file_list), 1)
    rng.shuffle(indices)

    train, val, test    = indices[:num_train], indices[num_train:num_train+num_val], \
        indices[num_train+num_val:num_train+num_val+num_test]

    return label_file_list[train], label_file_list[val], label_file_list[test]


def get_unique_objects(indir, traj, label_file_list):
    # parse unique objects from each file in mp thread
    
    print("Start processing traj %s for unique objects" % str(traj))
    start = time.time()
    unique_objects = set()
    for label_file in label_file_list:
        label_path = join(indir, label_file)
        anno_json = json.load(open(label_path, 'r'))

        for anno_dict in anno_json["3dbbox"]:
            unique_objects.add(anno_dict["classId"])

    print("Done processing traj %s for unique objects in %6.2f seconds" %
        (str(traj), time.time() - start) )
    unique_objects = sorted(list(unique_objects))

    return unique_objects
    # pool = Pool(processes=num_workers)
    # for _ in tqdm.tqdm(pool.imap_unordered( self.deepen_decode_single_sem_file, zip(traj_dir_multi, annotation_files_multi)), total=len(annotation_files_multi)):
    #     pass

def gen_metadata_file(indir, outdir, bbox_traj_list=[], sem_traj_list=[], 
    split_size=[0.7, 0.15, 0.15], split_suffix=""):
    if len(bbox_traj_list)==0 and len(sem_traj_list)==0:
        print("No trajectories to process, returning...")
        return
    
    operators_path = './helpers/helper_utils/operators.json'
    dates_path = './helpers/helper_utils/dates.json'
    operators_dict = None
    dates_dict = None
    if not os.path.exists(operators_path):
        print("No operators path exists %s, skipping for metadata files..."%operators_path)
    elif not os.path.exists(dates_path):
        print("No dates path exists %s, skipping for metadata files..."%dates_path)
    else:
        operators_dict = json.load(open(operators_path, 'r'))
        dates_dict = json.load(open(dates_path, 'r'))

    traj_list = sorted(list(set(bbox_traj_list+sem_traj_list)))
    for traj in traj_list:
        print("Creating metadata file for traj %d"%traj)
        metadata_dict = copy.deepcopy(METADATA_DICT)
        bbox_traj_fulldir, sem_traj_fulldir = None, None
        metadata_dict["date"] = dates_dict[str(traj)] if dates_dict is not None else None
        metadata_dict["operator"] = operators_dict[str(traj)] if metadata_dict is not None else None
        metadata_dict["poses"] = join(POSES_DIR, DENSE_POSES_DIR, "%s.txt"%str(traj))

        tasks = {}
        if traj in bbox_traj_list:
            pc_subdir= os.path.join(TRED_BBOX_LABEL_DIR, "os1")
            task = SENSOR_DIRECTORY_TO_TASK[pc_subdir]
            tasks[task] = TRED_BBOX_LABEL_DIR

        if traj in sem_traj_list:
            pc_subdir= os.path.join(SEMANTIC_LABEL_DIR, "os1")
            task = SENSOR_DIRECTORY_TO_TASK[pc_subdir]
            tasks[task] = SEMANTIC_LABEL_DIR

        for task, task_dir in tasks.items():
            label_files = get_label_list(indir, task_dir, traj)
            train_files, val_files, test_files = gen_dataset_split(label_files, split_size)

            label_files = np.concatenate((train_files, val_files, test_files))
            if task==OBJECT_DETECTION_TASK:
                metadata_dict["objects"].extend(get_unique_objects(indir, traj, label_files))

            metadata_dict[task]["training"].extend(train_files.tolist())
            metadata_dict[task]["validation"].extend(val_files.tolist())
            metadata_dict[task]["testing"].extend(test_files.tolist())

        metadata_dict["trajectory"] = int(traj)
        metadata_path = os.path.join(outdir, "%d.json"%traj)
        metadata_file = open(metadata_path, "w+")
        json.dump(metadata_dict, metadata_file, indent=4)
        metadata_file.close()

def main(args):
    indir = args.data_path
    datasize = args.size

    """
    Metadata Generation Steps
    1. Generate Auto Statistics 
    2. Random divide Train, Test, Val Splits for Object Detection
    """
    assert os.path.isdir(indir), '%s does not exist for root directory' % indir

    outdir = os.path.join(indir, "metadata")
    if datasize!="full":
        outdir = os.path.join(indir, "metadata_%s"%datasize)
    if not os.path.exists(outdir):
        print("Metadata directory does not exist, creating at %s..."%outdir)
        os.mkdir(outdir)

    pc_subdir= os.path.join(TRED_BBOX_LABEL_DIR, "os1")
    pc_fulldir = os.path.join(indir, pc_subdir)
    pc_sem_subdir = os.path.join(SEMANTIC_LABEL_DIR, "os1")
    pc_sem_fulldir = os.path.join(indir, pc_sem_subdir)
    if not os.path.exists(pc_fulldir):
        print('%s does not exist for pc directory' % pc_fulldir)
        os.makedirs(pc_fulldir)
    if not os.path.exists(pc_sem_fulldir):
        print('%s does not exist for pc semantic directory' % pc_sem_fulldir)
        os.makedirs(pc_sem_fulldir)
    bbox_traj_list = [int(traj) for traj in os.listdir(pc_fulldir) if os.path.isdir(
        os.path.join(pc_fulldir, traj) )]
    bbox_traj_list = sorted(bbox_traj_list, key=lambda x: int(x), reverse=False)

    sem_traj_list = [int(traj) for traj in os.listdir(pc_sem_fulldir) if os.path.isdir(
        os.path.join(pc_sem_fulldir, traj) )]
    sem_traj_list = sorted(sem_traj_list, key=lambda x: int(x), reverse=False)

    # Generation annotation split
    if datasize=="full":
        split_size = [0.7, 0.15, 0.15] # Use for full dataset 100%
    elif datasize=="md":
        split_size = [0.35, 0.075, 0.075] # Use for medium dataset 50%
    elif datasize=="sm":
        split_size = [0.15, 0.05, 0.05] # Use for small dataset 25%
    elif datasize=="tiny":
        split_size = [0.03, 0.01, 0.01] # Use for dataset teaser 5%
    else:
        print("Invalid datasize specified, exiting...")
        return

    gen_metadata_file(indir, outdir, bbox_traj_list, sem_traj_list, split_size)

    # for traj in traj_list:
    #     print("Creating metadata file for traj %s"%traj)
    #     metadata_dict = copy.deepcopy(METADATA_DICT)

    #     traj_subdir = os.path.join(pc_subdir, traj)
    #     traj_fulldir= os.path.join(indir, traj_subdir)

    #     bin_files = np.array([os.path.join(traj_subdir, bin_file) for bin_file in os.listdir(traj_fulldir) if bin_file.endswith(".json")])
    #     num_bin_files   = len(bin_files)
    #     num_train       = int(num_bin_files * train_percent)
    #     num_val         = int(num_bin_files * val_percent)

    #     #1
    #     indices = np.arange(0, len(bin_files), 1)
    #     rng.shuffle(indices)

    #     train, val, test    = indices[:num_train], indices[num_train:num_train+num_val], \
    #         indices[num_train+num_val:]

    #     #2
    #     metadata_dict["ObjectTracking"]["training"].extend(bin_files[train].tolist())
    #     metadata_dict["ObjectTracking"]["validation"].extend(bin_files[val].tolist())
    #     metadata_dict["ObjectTracking"]["testing"].extend(bin_files[test].tolist())

    #     metadata_dict["trajectory"] = int(traj)

    #     metadata_path = os.path.join(outdir, "%s.json"%traj)
    #     metadata_file = open(metadata_path, "w+")
    #     json.dump(metadata_dict, metadata_file, indent=4)



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
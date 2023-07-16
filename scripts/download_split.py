import os
from os.path import join

import sys
import json
import shutil

sys.path.append(os.getcwd())

from helpers.constants import *
from helpers.sensors import *
from helpers.metadata import *

from multiprocessing import Pool
import tqdm

SUBSET_DIR_COPY_LIST = [TWOD_RAW_DIR, TRED_COMP_DIR, TRED_BBOX_LABEL_DIR, SEMANTIC_LABEL_DIR]

def task_files(meta_path, task, return_frames=True):
    json_dict = json.load(open(meta_path, 'r'))

    files_list = []
    splits = ["training", "validation", "testing"]
    for split in splits:
        files_list.extend(json_dict[task][split])
    
    frames_list = []
    if return_frames:
        for subpath in files_list:
            annofile = subpath.split('/')[-1]
            _, _, traj, frame = get_filename_info(annofile)
            frames_list.append(frame)

    return files_list, frames_list

def copy_single_traj_files(args):
    indir, outdir, meta_path, split = args
    json_dict = json.load(open(meta_path, 'r'))

    traj = str(json_dict['trajectory'])

    frames_dict = {}
    # Object List
    obj_files_list, obj_frames_list = task_files(meta_path, OBJECT_DETECTION_TASK)

    # Semantic List
    sem_files_list, sem_frames_list = task_files(meta_path, SEMANTIC_SEGMENTATION_TASK)

    # Ensures all frames are unique
    full_frames_list = list(set(obj_frames_list + sem_frames_list))
    full_frames_list = sorted(full_frames_list, key=lambda frame: int(frame))

    # print("Copying %i files from traj %s to %s" % (len(full_frames_list), traj, outdir))

    # Copy 2d_raw, 3d_comp, 3d bbox, 3d semantic files
    for subset_dir in SUBSET_DIR_COPY_LIST:
        modality= subset_dir
        # Default to downloading cam0, cam1, and os1 for now
        if subset_dir==TWOD_RAW_DIR:
            sensor_list = ["cam0", "cam1"]
        else:
            sensor_list = ["os1"]
        
        subset_frames_list = full_frames_list
        if subset_dir == TRED_BBOX_LABEL_DIR:
            subset_frames_list = obj_frames_list
        elif subset_dir == SEMANTIC_LABEL_DIR:
            subset_frames_list = sem_frames_list

        # Copy all subdirectory files for outdir
        for sensor in sensor_list:
            for frame in subset_frames_list:
                frame_in_path = set_filename_dir(indir, modality, sensor, traj, frame)
                frame_out_path = set_filename_dir(outdir, modality, sensor, traj, frame)
                shutil.copyfile(frame_in_path, frame_out_path)

    # metadata_dir = METADATA_DIR if split=="full" else "%s_%s" % (METADATA_DIR, split)
    # FULL_DIR_COPY_LIST = [metadata_dir, TIMESTAMPS_DIR, DENSE_POSES_FULL_DIR, CALIBRATION_DIR]

    # for full_dir in FULL_DIR_COPY_LIST:
    #     src_dir = join(indir, full_dir)
    #     out_dir = join(outdir, full_dir)
    #     shutil.copytree(src_dir, out_dir, dirs_exist_ok=True)

    # print("Copying %i files from traj %s to %s" % (len(full_frames_list), traj, outdir))


def copy_split_files(indir, outdir, split="sm"):
    
    meta_dir = join(indir, "metadata_%s"%split)
    meta_files = [meta_file for meta_file in os.listdir(meta_dir) if meta_file.endswith(".json")]
    meta_files = sorted(meta_files, key=lambda mfile: int(mfile.split('.')[0]))

    meta_path_list = [join(meta_dir, meta_file) for meta_file in meta_files]
    indir_list = [indir] * len(meta_files)
    outdir_list = [outdir] * len(meta_files)
    split_list =  [split] * len(meta_files)
    for meta_file in meta_files:
        meta_path = join(meta_dir, meta_file)
        json_dict = json.load(open(meta_path, 'r'))

        traj = str(json_dict['trajectory'])

        # Make subset directories

        for subset_dir in SUBSET_DIR_COPY_LIST:
            if subset_dir==TWOD_RAW_DIR:
                sensor_list = ["cam0", "cam1"]
            else:
                sensor_list = ["os1"]
            
            for sensor in sensor_list:
                traj_path = join(outdir, subset_dir, sensor, traj)
                if not os.path.exists(traj_path):
                    print("Subdirectory %s dne, creating..."%traj_path )
                    os.makedirs(traj_path)

        # copy_single_traj_files((indir, outdir, meta_path, split))
    pool = Pool(processes=32)
    for _ in tqdm.tqdm(pool.imap_unordered(copy_single_traj_files, \
        zip(indir_list, outdir_list, meta_path_list, split_list)), total=len(meta_files)):
        pass

def main():
    split = "tiny"
    indir="/robodata/arthurz/Datasets/CODa_dev"
    outdir="/scratch/arthurz/Datasets/CODa_%s" % split
    if not os.path.exists(outdir):
        print("Making outdir if only one additional folder needed %s"%outdir)
        os.mkdir(outdir)

    copy_split_files(indir, outdir, split=split)

if __name__ == '__main__':
    main()
import cv2
import numpy as np
import pdb 
import os
from os.path import join

import sys
sys.path.append(os.getcwd())

from helpers.constants import *
from helpers.sensors import *
# closest_img_path = "/robodata/arthurz/Datasets/CODa_dev/3d_raw/cam3/13/3d_raw_cam3_13_1675698198145955089.png"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--traj', default="0",
                    help="Select a trajectory to visualize ")
parser.add_argument('-f', '--frame', default="0",
                    help="Select a frame to visualize ")

def extract_ts(img_file):
    ts_str = img_file.split("_")[-1].split(".")[0]
    dec = 10
    ts_str_dec = ts_str[:dec] + "." + ts_str[dec:]
    return float(ts_str_dec)

def extract_frame(img_file):
    frame = img_file.split("_")[-1].split(".")[0]
    return int(frame)

def find_closest_frame_idx(sync_img_file, sorted_sync_img_ts, sorted_unsync_img_files, sorted_unsync_img_ts):
    modality, sensor, _, frame = get_filename_info(sync_img_file)
    target_ts = sorted_sync_img_ts[int(frame)]

    print("Searching in %s for timestamp %s" % (unsync_img_dir, str(target_ts)))
    closest_img_idx = np.argmin(np.abs(target_ts - sorted_unsync_img_ts))
    closest_img_path = join(unsync_img_dir, sorted_unsync_img_files[closest_img_idx])
    print("Closest image found with timestamp %s ", sorted_unsync_img_ts[closest_img_idx])

    return closest_img_path

def main(args):
    indir = os.getenv(ENV_CODA_ROOT_DIR)
    assert indir is not None, f'Directory for CODa cannot be found, set {ENV_CODA_ROOT_DIR}'
    traj  = args.traj
    target_frame = args.frame

    sync_img_dir = "%s/2d_raw/cam0/%s" % (indir, traj)
    unsync_img_dir = "%s/3d_raw/cam3/%s" % (indir, traj)
    sync_ts_file = join(indir, TIMESTAMPS_DIR, "%s.txt"%traj)
    unsync_img_files = np.array([img_file for img_file in os.listdir(unsync_img_dir) if img_file.endswith('.png')])
    unsync_img_ts = np.array([extract_ts(img_file) for img_file in unsync_img_files])
    unsync_sort_mask = np.argsort(unsync_img_ts) # low to high

    sorted_unsync_img_ts = unsync_img_ts[unsync_sort_mask]
    sorted_unsync_img_files = unsync_img_files[unsync_sort_mask]

    sync_img_files = [img_file for img_file in os.listdir(sync_img_dir) if img_file.endswith('.png')]
    sorted_sync_img_files = np.array(sorted(sync_img_files, key=extract_frame)) # sorted low to high
    sorted_sync_img_ts = np.fromfile(sync_ts_file, sep=' ').reshape(-1,) # already sorted

    for sync_img_file in sorted_sync_img_files:
        modality, sensor, _, frame = get_filename_info(sync_img_file)
        if int(frame)<int(target_frame):
            continue

        target_ts = sorted_sync_img_ts[int(frame)]
        print("Searching in %s for timestamp %s" % (unsync_img_dir, str(target_ts)))
        closest_img_idx = np.argmin(np.abs(target_ts - sorted_unsync_img_ts))
        closest_img_path = join(unsync_img_dir, sorted_unsync_img_files[closest_img_idx])
        print("Closest image found with timestamp %s ", sorted_unsync_img_ts[closest_img_idx])

        sync_img_np = cv2.imread(join(sync_img_dir, sync_img_file))
        unsync_img_np = cv2.imread(closest_img_path, cv2.IMREAD_GRAYSCALE)
        max_val = np.max(unsync_img_np)
        unsync_img_np = unsync_img_np * (255.0 / max_val)
        
        print(f'Writing image to directory {os.getcwd()}/testsync.png')
        print(f'Writing image to directory {os.getcwd()}/testsunync.png')
        cv2.imwrite("testsync.png", sync_img_np)
        cv2.imwrite("testunsync.png", unsync_img_np)
        import pdb; pdb.set_trace()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
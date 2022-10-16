import os
import pdb
import numpy as np

from Research.UTPeDa.scripts.label_utils import *

def save_bins(dataset_root, save_root, header):

    data_dir = dataset_root
    save_dir = os.path.join(save_root, "velo")
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    convert_bag_to_bins(data_dir, save_dir, header)

def main():
    # User defined filepaths
    DATASET_ROOT_DIR    = "/home/arthur/Data/SCAND-label"
    SAVE_ROOT_DIR       = "/home/arthur/AMRL/SCAND/data" 
    SCENE_ENV           = "outdoor"
    FRAME_SCALE         = 10 # 10Hz / 10 = 1Hz

    bin_header = {
        "rate": FRAME_SCALE,
        "scene": SCENE_ENV
    }

    manifest_header = {
        "prefix": "s3://scand-trial1-sagemaker/artifacts/",
        "rate": FRAME_SCALE, # Measured in terms of Data Hz / rate Ex: Lidar @ 10Hz, rate=10 = Lidar @ 1Hz
        "scene": SCENE_ENV 
    }

    image_header = {
        "rate": FRAME_SCALE,
        "scene": SCENE_ENV
    }

    # Convert all pcd files in data directory to bin files
    # convert_bag_to_bins(DATASET_ROOT_DIR, SAVE_ROOT_DIR, bin_header)
    create_manifest(DATASET_ROOT_DIR, SAVE_ROOT_DIR, manifest_header)
    # create_images(DATASET_ROOT_DIR, SAVE_ROOT_DIR, image_header)

    # Uncomment for debugging
    # SCENE_DIR = os.path.join(SAVE_ROOT_DIR, "velo", "seq10")
    # visualize_bins(SCENE_DIR)
    """
    open_bin_as_np(os.path.join(SAVE_ROOT_DIR, "velo", "2021-11-10-10-44-32"))
    """

if __name__ == "__main__":
    main()
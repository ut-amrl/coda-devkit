import os
import sys
import yaml
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default="config/bagdecoder.yaml",
                    help="decode config file (see config/decode.yaml for example)")
parser.add_argument('-a', '--all_days', default=False, help="decode all sudirs or the one specified in .yaml")
parser.add_argument('-p', '--densify_poses', type=bool, default=False, help="Only densify existing poses")

# For imports
sys.path.append(os.getcwd())

#CustomImports
from helpers.bagdecoder import BagDecoder


def main(args):
    # Process the following days in CODa_bags
    settings_file =  open(args.config, 'r')
    settings = yaml.safe_load(settings_file)
    root_repo = settings['repository_root']
    if args.all_days:
        subdirs_to_process = -1
        trajectory_curr = 0

        if subdirs_to_process==-1: # Default to all subdirs
            subdirs_to_process = [entry.path for entry in os.scandir(root_repo) if entry.is_dir()]

        for dir_path in subdirs_to_process:
            bag_files = [ subdir_file.path.split('/')[-1] for subdir_file in os.scandir(dir_path) if subdir_file.path.endswith(".bag") and
                "calibration" not in subdir_file.path]

            subdir = dir_path.split('/')[-1]
            # Modify bagdecoder.yaml settings with correct config
            settings['bag_date'] = subdir
            settings['bags_to_process'] = bag_files
            settings['bags_to_traj_ids'] = np.arange(trajectory_curr, trajectory_curr+len(bag_files), 1)
            trajectory_curr += len(bag_files)

            bag_decoder = BagDecoder(settings, is_config_dict=True)

            print("Starting conversion for date ", subdir, " days ", bag_files)            
            bag_decoder.convert_bag()
            print("Finished converting bag, sending bag complete signal...")

    else:
        if args.densify_poses:
            poses_indir = os.path.join(settings['dataset_output_root'], "poses")
            poses_outdir= os.path.join("./dense")
            ts_dir      = os.path.join(settings['dataset_output_root'], "timestamps")
            BagDecoder.densify_poses(poses_indir, poses_outdir, ts_dir)
        else:
            bag_decoder = BagDecoder(args.config, is_config_dict=False)
            bag_decoder.convert_bag()
        # bag_decoder.rectify_images(num_workers=24)
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
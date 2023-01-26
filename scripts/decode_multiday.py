import os
import sys
import yaml
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', default="config/bagdecoder.yaml",
                    help="decode config file (see config/decode.yaml for example)")
parser.add_argument('--all_days', default=False, help="decode all sudirs or the one specified in .yaml")

# For imports
sys.path.append(os.getcwd())

#CustomImports
from helpers.bagdecoder import BagDecoder


def main(args):
    if args.all_days:
        # Process the following days in CODa_bags
        settings_file =  open(args.config, 'r')
        settings = yaml.safe_load(settings_file)
        root_repo = settings['repository_root']

        subdirs_to_process = ["20230118"]
        trajectory_curr = 8

        if subdirs_to_process==-1: # Default to all subdirs
            subdirs_to_process = [ subdir for subdir in os.listdir(root_repo) if os.path.isdir(os.path.join(
                root_repo, subdir) ) ]

        for subdir in subdirs_to_process:
            dir_path = os.path.join(root_repo, subdir)

            bag_files = [ subdir_file for subdir_file in os.listdir(dir_path) if subdir_file.endswith(".bag") and
                "calibration" not in subdir_file]

            if subdir=="20230119":
                bag_files = ["1674164498.bag"]

            # Modify bagdecoder.yaml settings with correct config
            settings['bag_date'] = subdir
            settings['bags_to_process'] = bag_files
            settings['bags_to_traj_ids'] = np.arange(trajectory_curr, trajectory_curr+len(bag_files), 1)
            trajectory_curr += len(bag_files)

            bag_decoder = BagDecoder(settings, is_config_dict=True)
            bag_decoder.convert_bag()
    else:
        bag_decoder = BagDecoder(args.config, is_config_dict=False)
        bag_decoder.convert_bag()
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
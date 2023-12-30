import os
import sys
import yaml
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default="config/bagdecoder.yaml",
                    help="decode config file (see config/decode.yaml for example)")
parser.add_argument('-a', '--all_days', type=bool, default=False, help="decode all sudirs or the one specified in .yaml")
parser.add_argument('-p', '--densify_poses', type=bool, default=False, help="Only densify existing poses")
parser.add_argument('-ca', '--calibrations', type=bool, default=False, help="Only process calibrations")


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
        # if args.calibrations:
        #     subdirs_to_process = [
        #         "20230116",
        #         "20230117",
        #         "20230118",
        #         "20230126",
        #         "20230127",
        #         "20230130", 
        #         "20230203", 
        #         "20230206", 
        #         "20230207", 
        #         "20230208", 
        #         "20230209", 
        #         "20230210"
        #     ]
        #     subdirs_to_process = [os.path.join(root_repo, subdir) for subdir in subdirs_to_process]
        #     traj_ids = [
        #         0, 1, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22
        #     ]
            
        if subdirs_to_process==-1: # Default to all subdirs
            subdirs_to_process = [entry.path for entry in os.scandir(root_repo) if entry.is_dir()]
            traj_ids = np.arange(0, 23).tolist()

        traj_idx = 0
        for dir_idx, dir_path in enumerate(subdirs_to_process):
            bag_files = [ subdir_file.path.split('/')[-1] for subdir_file in os.scandir(dir_path) 
            if subdir_file.path.endswith(".bag") and
                ( ("calibration" not in subdir_file.path) ^ args.calibrations)] 

            subdir = dir_path.split('/')[-1]
            # Modify bagdecoder.yaml settings with correct config
            settings['bag_date'] = subdir
            settings['bags_to_process'] = bag_files
            settings['bags_to_traj_ids'] = traj_ids[traj_idx:traj_idx+len(bag_files)]
            traj_idx += len(bag_files)
            
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
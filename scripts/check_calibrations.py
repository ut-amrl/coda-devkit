import os
from os.path import join
import sys
import pdb
import json

import rospy
from sensor_msgs.msg import PointCloud2, Image

import cv2
import numpy as np

# For imports
sys.path.append(os.getcwd())

from helpers.sensors import get_filename_info, set_filename_by_prefix, read_bin, read_sem_label, set_filename_dir
from helpers.visualization import *
from helpers.metadata import read_metadata_anno
from helpers.constants import METADATA_DIR, CALIBRATION_DIR, TWOD_RAW_DIR, TRED_RAW_DIR, SEM_POINT_SIZE

import argparse
from multiprocessing import Pool
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--mod', default="3d_bbox", help="Visualize annotation type")
parser.add_argument('--gen_all', default=True, help="Generate all annotations")
parser.add_argument('--cfg_file', default='config/checker_annotation.yaml', help="Config file to use for input files")
parser.add_argument('--cam_id', default="cam0", help="Camera id to generate calibration for")

def generate_point_calibration_img(image_np, bin_np, calib_ext_file, calib_intr_file):
    image_pts, pts_mask = project_3dto2d_points(bin_np, calib_ext_file, calib_intr_file)
    in_bounds = np.logical_and(
            np.logical_and(image_pts[:, 0]>=0, image_pts[:, 0]<1224),
            np.logical_and(image_pts[:, 1]>=0, image_pts[:, 1]<1024)
        )
    valid_point_mask = in_bounds & pts_mask
    valid_points = image_pts[valid_point_mask, :]

    for pt in valid_points:
        image_np = cv2.circle(image_np, (pt[0], pt[1]), radius=SEM_POINT_SIZE, color=(0, 0, 255), thickness=-1)
    return image_np
    
def dump_calibration_img(outdir, traj, frame, cam_id, pt_image_np):
    #Assumes images are rectified now
    out_image_dir = join(outdir, TWOD_RAW_DIR, cam_id, traj)
    if not os.path.exists(out_image_dir):
        print("Output image directory %s  does not exist, creating now..." % out_image_dir)
        os.makedirs(out_image_dir)
    out_image_file = set_filename_by_prefix("2d_rect", cam_id, traj, frame)
    out_image_path = join(out_image_dir, out_image_file)

    # Write trajectory and frame to calibration image
    text = "Traj %s Frame %s" % (str(traj), str(frame))
    org = (1224-350, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2
    pt_image_np = cv2.putText(pt_image_np, text, org, font, fontScale, 
                 color, thickness, cv2.LINE_AA, False)

    print("Saving %s projection for traj %s frame %s at %s..." % (cam_id, traj, frame, out_image_path))
    cv2.imwrite(out_image_path, pt_image_np)

def generate_calibration_summary(args):
    cfg, traj = args

    # assumes images are not rectified by default
    TRAJ_SUB_KEY = "[TRAJECTORY]"
    CAM_SUB_KEY = "[CAM]"
    cam_list = ["cam0"]

    traj = str(traj)
    calibextr_path = join(  cfg['indir'], 
        cfg['calibextr_subdir'].replace(TRAJ_SUB_KEY, traj) )
    calibintr_path = join(  cfg['indir'], 
        cfg['calibintr_subdir'].replace(TRAJ_SUB_KEY, traj) )
    tred_path = join( cfg['indir'],
        cfg['tred_subdir'].replace(TRAJ_SUB_KEY, traj)
    )

    twod_paths = []
    for cam_id in cam_list:
        twod_paths.append(join( cfg['indir'],
            cfg['cam_subdir'].replace(CAM_SUB_KEY, cam_id).replace(TRAJ_SUB_KEY, traj)
        ))

    ts_path = join( cfg['indir'],
        cfg['ts_subdir'].replace(TRAJ_SUB_KEY, traj)
    )

    # Project 3D LiDAR to Images
    frame_to_ts_np = np.loadtxt(ts_path)
    for frame, ts in enumerate(frame_to_ts_np):
        if frame % cfg['fr']==0:
            for cam_list_idx, cam_id in enumerate(cam_list):
                twod_img_file   = set_filename_by_prefix(TWOD_RAW_DIR, cam_id, traj, frame)
                twod_img_path = os.path.join(twod_paths[cam_list_idx], twod_img_file)
                image_np = cv2.imread(twod_img_path)

                tred_bin_file = set_filename_by_prefix(TRED_COMP_DIR, "os1", traj, frame)
                tred_bin_path = join(tred_path, tred_bin_file)
                bin_np = read_bin(tred_bin_path)
                pt_image_np = project_3dpoint_image(image_np, bin_np, calibextr_path, calibintr_path)

                dump_calibration_img(cfg['outdir'], traj, frame, cam_id, pt_image_np)

def generate_single_anno_file(args):
    indir, outdir, modality, sensor, traj, frame, cam_list = args
    traj = str(traj)

    # Assumes images are not rectified by default
    TRAJ_SUB_KEY = "[TRAJECTORY]"
    CAM_SUB_KEY = "[CAM]"

    # Set IO paths
    calibextr_path  = join( indir, CALIBRATION_DIR, traj, "calib_os1_to_%s.yaml"%cam_list[0])
    calibintr_path  = join( indir, CALIBRATION_DIR, traj, "calib_%s_intrinsics.yaml"%cam_list[0])
    tred_anno_path  = set_filename_dir(indir, modality, sensor, traj, frame)
    twod_img_path   = set_filename_dir(indir, TWOD_RAW_DIR, cam_list[0], traj, frame)
    
    # Project 3D anno to 2D
    image_np = cv2.imread(twod_img_path)
    if "3d_bbox"==modality:
        tred_anno_dict = json.load(open(tred_anno_path))
        tred_anno_image = project_3dbbox_image(tred_anno_dict, calibextr_path, calibintr_path, image_np)
    elif "3d_semantic"==modality:
        pc_path = set_filename_dir(indir, TRED_RAW_DIR, sensor, traj, frame)
        pc_np   = read_bin(pc_path, keep_intensity=False)
        tred_anno_image = project_3dpoint_image(image_np, pc_np, calibextr_path, calibintr_path, tred_anno_path)

    dump_calibration_img(outdir, traj, frame, cam_list[0], tred_anno_image)

def generate_annotation_visualization(args):
    cfg, traj, frame = args

    # assumes images are not rectified by default
    TRAJ_SUB_KEY = "[TRAJECTORY]"
    CAM_SUB_KEY = "[CAM]"
    cam_list = ["cam0"]

    traj = str(traj)
    calibextr_path = join(  cfg['indir'], 
        cfg['calibextr_subdir'].replace(TRAJ_SUB_KEY, traj) )
    calibintr_path = join(  cfg['indir'], 
        cfg['calibintr_subdir'].replace(TRAJ_SUB_KEY, traj) )
    bbox_path = join( cfg['indir'],
        cfg['bbox_subdir'].replace(TRAJ_SUB_KEY, traj)
    )

    twod_paths = []
    for cam_id in cam_list:
        twod_paths.append(join( cfg['indir'],
            cfg['cam_subdir'].replace(CAM_SUB_KEY, cam_id).replace(TRAJ_SUB_KEY, traj)
        ))

    # Project 3D annotations to images
    for cam_list_idx, cam_id in enumerate(cam_list):
        twod_img_file   = set_filename_by_prefix(TWOD_RAW_DIR, cam_id, traj, frame)
        twod_img_path = os.path.join(twod_paths[cam_list_idx], twod_img_file)
        image_np = cv2.imread(twod_img_path)

        tred_bbox_file = set_filename_by_prefix(TRED_BBOX_LABEL_DIR, "os1", traj, frame)
        tred_bbox_path = join(bbox_path, tred_bbox_file)
        bbox_anno_dict = json.load(open(tred_bbox_path))
        tred_bbox_image = project_3dbbox_image(bbox_anno_dict, calibextr_path, calibintr_path, image_np)

        dump_calibration_img(cfg['outdir'], traj, frame, cam_id, tred_bbox_image)

def main(args):
    """
    indir - CODa directory (assumes 3d_labels exists)
    outdir - directory to save bbox projections to

    This script can be used to project the point cloud to corresponding images. 
    """
    cfg_path    = args.cfg_file
    gen_all     = args.gen_all
    modality    = args.mod
    cam_id      = args.cam_id

    checker_fp = os.path.join(os.getcwd(), cfg_path)
    with open(checker_fp, 'r') as checker_file:
        checker_cfg = yaml.safe_load(checker_file)
        indir       = checker_cfg["indir"]
        outdir       = checker_cfg["outdir"]

        if gen_all:
            for trajectory in np.arange(23):
                # Read all 3d bbox files from coda metadata
                metadata_path = join(indir, METADATA_DIR, "%i.json"%trajectory)

                if not os.path.exists(metadata_path):
                    print("Trajectory %i does not contain metadata file %s" % (trajectory, metadata_path))
                    continue
                anno_subpaths = read_metadata_anno(metadata_path, modality=modality)
                num_annos = len(anno_subpaths)

                if num_annos==0:
                    continue
                
                _, sensor_name, _, frame = get_filename_info(anno_subpaths[0].split("/")[-1])
                indir_list = [indir] * num_annos
                outdir_list = [outdir] * num_annos
                modality_list = [modality] * num_annos
                sensor_list = [sensor_name] * num_annos
                traj_list = [trajectory] * num_annos
                frame_list = []
                cam_list = [[cam_id]] * num_annos
                for anno_path in anno_subpaths:
                    modality, sensor_name, _, frame = get_filename_info(anno_path.split("/")[-1])

                    # Make trajectory ehre
                    output_img_dir = join(outdir, TWOD_RAW_DIR, cam_id, str(trajectory))
                    if not os.path.exists(output_img_dir):
                        print("Output image dir for %s does not exist, creating..."%output_img_dir)
                        os.makedirs(output_img_dir)
                    frame_list.append(frame)

                    # #Uncomment below for testing
                    # if not (trajectory==0 and int(frame) == 4965):
                    #     continue
                    # generate_single_anno_file((indir, ".", modality, sensor_name, trajectory, frame, [cam_id]))

                pool = Pool(processes=checker_cfg['num_workers'])
                for _ in tqdm.tqdm(pool.imap_unordered(generate_single_anno_file, \
                    zip(indir_list, outdir_list, modality_list, sensor_list, traj_list, frame_list, cam_list)), total=num_annos):
                    pass

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
import os
import sys
import pdb
import json

import rospy
from sensor_msgs.msg import PointCloud2, Image

import cv2
import numpy as np

# For imports
sys.path.append(os.getcwd())

from helpers.sensors import get_filename_info, set_filename_by_prefix, read_bin
from helpers.visualization import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--traj', default=0,
                    help="trajectory to visualization calibation for (default 0)")
parser.add_argument('--frame', default=0, help="first frame to visualization calibration for")

def main(args):
    """
    indir - CODa directory (assumes 3d_labels exists)
    outdir - directory to save bbox projections to
    """
    indir   = "/robodata/arthurz/Datasets/CODa"
    trajectory = int(args.traj)
    start_frame = int(args.frame)
    use_wcs = False
    use_sem = False

    #Project 3d bbox annotations to 2d
    calib_ext_file = os.path.join(indir, "calibrations", str(trajectory), "calib_os1_to_cam0.yaml")
    calib_intr_file= os.path.join(indir, "calibrations", str(trajectory), "calib_cam0_intrinsics.yaml")
    tred_bin_dir   = os.path.join(indir, "3d_raw", "os1", str(trajectory))
    tred_anno_dir  = os.path.join(indir, "3d_bbox", "os1", str(trajectory))
    twod_anno_dir  = os.path.join(indir, "2d_rect", "cam0", str(trajectory))

    #Locate closest pose from frame
    ts_to_frame_path = os.path.join(indir, "timestamps", "%s_frame_to_ts.txt"%trajectory)
    ts_to_poses_path = os.path.join(indir, "poses", "%s.txt"%trajectory)
    frame_to_poses_np = np.loadtxt(ts_to_poses_path).reshape(-1, 8)
    frame_to_ts_np = np.loadtxt(ts_to_frame_path)

    # frame_order  = np.argsort([ int(get_filename_info(file)[-1]) for file in os.listdir(tred_anno_dir) ])
    # tred_anno_files = np.array(os.listdir(tred_anno_dir))[frame_order]
    
    frame_order  = np.argsort([ int(get_filename_info(file)[-1]) for file in os.listdir(tred_bin_dir) ])
    bin_files   = np.array(os.listdir(tred_bin_dir))[frame_order]

    image_pub   = rospy.Publisher('/coda/stereo/%s'%"cam0", Image, queue_size=10)
    pc_pub      = rospy.Publisher('/coda/ouster/lidar_packets', PointCloud2, queue_size=10)
    dense_poses = densify_poses_between_ts(frame_to_poses_np, frame_to_ts_np)

    for (idx, bin_file) in enumerate(bin_files):
        if idx < start_frame:
            continue

        modality, sensor_name, trajectory, frame = get_filename_info(bin_file)

        bin_path = os.path.join(tred_bin_dir, bin_file)
        bin_np = read_bin(bin_path, False)

        wcs_pose = None
        if use_wcs:
            wcs_pose = pose_to_homo(dense_poses[idx])
            bin_np_homo = np.hstack((bin_np, np.ones( (bin_np.shape[0], 1) ) ))
            bin_np      = (wcs_pose @ bin_np_homo.T).T[:, :3]
      
        twod_img_file   = set_filename_by_prefix("2d_rect", "cam0", trajectory, frame)
        twod_img_path = os.path.join(twod_anno_dir, twod_img_file)
        image = cv2.imread(twod_img_path)

        tred_label_file = set_filename_by_prefix("3d_bbox", "os1", trajectory, frame)
        tred_label_path = os.path.join(tred_anno_dir, tred_label_file)

        if use_sem:
            valid_points = project_3dsem_image(bin_np, calib_ext_file, calib_intr_file, wcs_pose)

            for pt in valid_points:
                image = cv2.circle(image, (pt[0], pt[1]), radius=1, color=(0, 0, 255))
            cv2.imwrite("testcalibration.png", image)
        else:
            anno_dict       = json.load(open(tred_label_path))
            image = project_3dbbox_image(anno_dict, calib_ext_file, calib_intr_file, image)
            cv2.imwrite("testbboxcalibration.png", image)

            bbox_coords = project_3dto2d_bbox_image(anno_dict, calib_ext_file, calib_intr_file)
            image = cv2.imread(twod_img_path)
            image = draw_2d_bbox(image, bbox_coords)
            cv2.imwrite("test2dbboxcalibration.png", image)

        import pdb; pdb.set_trace()

        # cv2.imshow('img', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
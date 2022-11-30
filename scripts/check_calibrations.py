import os
import pdb
import json

import rospy
from sensor_msgs.msg import PointCloud2, Image

import cv2
import numpy as np

from helpers.sensors import get_filename_info, set_filename_by_prefix
from helpers.geometry import find_closest_pose, project_3dto2d_bbox, draw_bbox
from helpers.visualization import *

def main():
    """
    indir - CODa directory (assumes 3d_labels exists)
    outdir - directory to save bbox projections to
    """
    indir   = "/home/arthur/AMRL/Datasets/CODa"
    outdir = "/home/arthur/AMRL/Datasets/CODa"
    trajectory = 0
    use_wcs = True

    #Project 3d bbox annotations to 2d
    calib_ext_file = os.path.join(indir, "calibrations", str(trajectory), "calib_os1_to_cam0.yaml")
    calib_intr_file= os.path.join(indir, "calibrations", str(trajectory), "calib_cam0_intrinsics.yaml")
    tred_anno_dir  = os.path.join(indir, "3d_label", "os1", str(trajectory))
    twod_anno_dir  = os.path.join(indir, "2d_raw", "cam0", str(trajectory))

    #Locate closest pose from frame
    ts_to_frame_path = os.path.join(indir, "timestamps", "%s_frame_to_ts.txt"%trajectory)
    ts_to_poses_path = os.path.join(indir, "poses", "%s.txt"%trajectory)
    frame_to_poses_np = np.loadtxt(ts_to_poses_path).reshape(-1, 8)
    frame_to_ts_np = np.loadtxt(ts_to_frame_path)

    frame_order  = np.argsort([ int(get_filename_info(file)[-1]) for file in os.listdir(tred_anno_dir) ])
    tred_anno_files = np.array(os.listdir(tred_anno_dir))[frame_order]

    image_pub   = rospy.Publisher('/coda/stereo/%s'%"cam0", Image, queue_size=10)
    pc_pub      = rospy.Publisher('/coda/bin/points', PointCloud2, queue_size=10)

    for tred_anno_file in tred_anno_files:
        modality, sensor_name, trajectory, frame = get_filename_info(tred_anno_file)

        tred_anno_path = os.path.join(tred_anno_dir, tred_anno_file)
        anno_dict       = json.load(open(tred_anno_path))

        # Load necessary files
        ts = frame_to_ts_np[int(frame)]
        pose    = find_closest_pose(frame_to_poses_np, ts)
        bbox_pts, bbox_mask = project_3dto2d_bbox(anno_dict, pose, calib_ext_file, calib_intr_file)

        twod_img_file   = set_filename_by_prefix("2d_raw", "cam0", trajectory, frame)
        twod_img_path   = os.path.join(twod_anno_dir, twod_img_file)
        image = cv2.imread(twod_img_path)
        
        for obj_idx in range(0, bbox_pts.shape[0]):
            # pdb.set_trace()
            # in_fov_points = bbox_pts[obj_idx, bbox_mask[obj_idx]]
            # if in_fov_points.shape[0]==0:
            #     continue

            # in_bounds = np.logical_and(
            #     np.logical_and(in_fov_points[:, 0]>=0, in_fov_points[:, 0]<1224),
            #     np.logical_and(in_fov_points[:, 1]>=0, in_fov_points[:, 1]<1024)
            # )
            in_bounds = np.logical_and(
                np.logical_and(bbox_pts[obj_idx, :, 0]>=0, bbox_pts[obj_idx, :, 0]<1224),
                np.logical_and(bbox_pts[obj_idx, :, 1]>=0, bbox_pts[obj_idx, :, 1]<1024)
            )
            # pdb.set_trace()
            # points = bbox_pts[obj_idx, bbox_mask[obj_idx]]
            # in_bounds_and_fov = np.logical_and(
            #     in_bounds, bbox_mask[obj_idx]
            # )
            # pdb.set_trace()
            valid_point_mask = bbox_mask[obj_idx] & in_bounds
            valid_points = bbox_pts[obj_idx, valid_point_mask, :]
            if valid_points.shape[0]==0:
                continue
            # pdb.set_trace()
            image = draw_bbox(image, valid_points, valid_point_mask)


            # for point in valid_points:
            #     image = cv2.circle(image, (point[0], point[1]), radius=5, color=(255, 0, 0), thickness=-1)
        # Save image locally overwrite
        tmp_img_path =os.path.join(twod_anno_dir, "test.png")
        cv2.imwrite(tmp_img_path, image)
        pub_img(image_pub, tmp_img_path)

        # print("valid_points", in_bounds_and_fov)
        # pdb.set_trace()
        # cv2.imshow('img', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
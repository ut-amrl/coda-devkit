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
    trajectory = 2
    use_wcs = False

    #Project 3d bbox annotations to 2d
    calib_ext_file = os.path.join(indir, "calibrations", str(trajectory), "calib_os1_to_cam0.yaml")
    calib_intr_file= os.path.join(indir, "calibrations", str(trajectory), "calib_cam0_intrinsics.yaml")
    tred_bin_dir   = os.path.join(indir, "3d_raw", "os1", str(trajectory))
    tred_anno_dir  = os.path.join(indir, "3d_label", "os1", str(trajectory))
    twod_anno_dir  = os.path.join(indir, "2d_raw", "cam0", str(trajectory))

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
    pc_pub      = rospy.Publisher('/coda/bin/points', PointCloud2, queue_size=10)
    dense_poses = densify_poses_between_ts(frame_to_poses_np, frame_to_ts_np)

    for (idx, bin_file) in enumerate(bin_files):
        if idx < 2760:
            continue
        modality, sensor_name, trajectory, frame = get_filename_info(bin_file)

        indir="/home/arthur/AMRL/Datasets/CODa"
        calib_ext_file = os.path.join(indir, "calibrations", str(trajectory), "calib_os1_to_cam0.yaml")
        calib_intr_file= os.path.join(indir, "calibrations", str(trajectory), "calib_cam0_intrinsics.yaml")
        
        bin_path = os.path.join(tred_bin_dir, bin_file)
        bin_np = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 3)

        wcs_pose = None
        if use_wcs:
            wcs_pose = pose_to_homo(dense_poses[idx])
            bin_np_homo = np.hstack((bin_np, np.ones( (bin_np.shape[0], 1) ) ))
            bin_np      = (wcs_pose @ bin_np_homo.T).T[:, :3]

        image_pts, pts_mask = project_3dto2d_points(bin_np, calib_ext_file, calib_intr_file, wcs_pose)
        # pdb.set_trace()
        in_bounds = np.logical_and(
                np.logical_and(image_pts[:, 0]>=0, image_pts[:, 0]<1224),
                np.logical_and(image_pts[:, 1]>=0, image_pts[:, 1]<1024)
            )
        valid_point_mask = in_bounds & pts_mask
        valid_points = image_pts[valid_point_mask, :]

        twod_img_file   = set_filename_by_prefix("2d_raw", "cam0", trajectory, frame)
        twod_img_path = os.path.join(twod_anno_dir, twod_img_file)
        image = cv2.imread(twod_img_path)
        for pt in valid_points:
            image = cv2.circle(image, (pt[0], pt[1]), radius=1, color=(0, 0, 255))
        # print("valid_points", in_bounds_and_fov)
        # pdb.set_trace()
        cv2.imshow('img', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # for tred_anno_file in tred_anno_files:
        # modality, sensor_name, trajectory, frame = get_filename_info(tred_anno_file)

        # tred_anno_path = os.path.join(tred_anno_dir, tred_anno_file)
        # anno_dict       = json.load(open(tred_anno_path))

        # # Load necessary files
        # ts = frame_to_ts_np[int(frame)]
        # pose    = find_closest_pose(frame_to_poses_np, ts)

        # twod_img_file   = set_filename_by_prefix("2d_raw", "cam0", trajectory, frame)
        # twod_img_path   = os.path.join(twod_anno_dir, twod_img_file)
        # image = cv2.imread(twod_img_path)

        # label_filename = set_filename_by_prefix("3d_label", "os1",
        #             str(trajectory), frame)

        # anno_dict       = json.load(open(tred_anno_path))
        # image = cv2.imread(twod_img_path)
        # indir="/home/arthur/AMRL/Datasets/CODa"
        # calib_ext_file = os.path.join(indir, "calibrations", str(trajectory), "calib_os1_to_cam0.yaml")
        # calib_intr_file= os.path.join(indir, "calibrations", str(trajectory), "calib_cam0_intrinsics.yaml")
        # bbox_pts, bbox_mask, bbox_idxs = project_3dto2d_bbox(anno_dict, pose, calib_ext_file, calib_intr_file)

        # for obj_idx in range(0, bbox_pts.shape[0]):
        #     in_bounds = np.logical_and(
        #         np.logical_and(bbox_pts[obj_idx, :, 0]>=0, bbox_pts[obj_idx, :, 0]<1224),
        #         np.logical_and(bbox_pts[obj_idx, :, 1]>=0, bbox_pts[obj_idx, :, 1]<1024)
        #     )
        #     # pdb.set_trace()
        #     valid_point_mask = bbox_mask[obj_idx] & in_bounds
        #     valid_points = bbox_pts[obj_idx, valid_point_mask, :]
        #     if valid_points.shape[0]==0:
        #         continue

        #     bbox_idx = bbox_idxs[obj_idx][0]
        #     obj_id = CLASS_TO_ID[anno_dict["3dannotations"][bbox_idx]["classId"]]
        #     obj_color = ID_TO_COLOR[obj_id]
        #     image = draw_bbox(image, valid_points, valid_point_mask, color=obj_color)


        #     # for point in valid_points:
        #     #     image = cv2.circle(image, (point[0], point[1]), radius=5, color=(255, 0, 0), thickness=-1)
        # # Save image locally overwrite
        # tmp_img_path =os.path.join(twod_anno_dir, "test.png")
        # cv2.imwrite(tmp_img_path, image)

        # # print("valid_points", in_bounds_and_fov)
        # # pdb.set_trace()
        # cv2.imshow('img', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import rospy
from sensor_msgs.msg import Image, PointCloud2
import cv2
from cv_bridge import CvBridge

import glob
import numpy as np
import os
from os.path import join
import open3d as o3d

import sys
sys.path.append(os.getcwd())
from helpers.sensors import get_calibration_info, get_filename_info, set_filename_dir, read_bin
from helpers.constants import TRED_RAW_DIR, CALIBRATION_DIR, TIMESTAMPS_DIR, STEREO_SETTINGS
from helpers.visualization import pub_pc_to_rviz
from scipy.spatial.transform import Rotation as R

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--indir', default="/robodata/arthurz/Datasets/CODa_dev",
                    help="Provide the path to where CODa is installed")
parser.add_argument('--traj', default="0",
                    help="Select a trajectory 0-22")
parser.add_argument('--cam', default="cam3",
                    help="Select a stereo camera to visualize (cam2, cam3)")

def convert_depth_to_pc(depth_img_np, fx, fy, cx, cy, range=[0.4, 25.0]):
    """
    Depth to point cloud is inherently noisy and results in large z steps
    https://github.com/stereolabs/zed-ros-wrapper/issues/412
    """
    depth_img_np = depth_img_np / 1000.0 # convert to meters
    max_depth, min_depth = np.max(depth_img_np), np.min(depth_img_np)
    depth_to_z = depth_img_np * (range[1] - range[0]) / (max_depth - min_depth)
    points = []

    # Create the intrinsics matrix (camera matrix)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(depth_to_z.shape[1], depth_to_z.shape[0], fx, fy, cx, cy)

    # Create the point cloud from the depth image
    depth_array = np.array(depth_to_z, dtype=np.float32) # by defaul tin meters
    point_cloud = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth_array), intrinsics)
    pc_np = np.asarray(point_cloud.points)
    return pc_np

def process_stereo_to_pointcloud(depth_img_path, calib_intr_path, calib_extr_path, depth_range=[0.4, 25.0]):
    depth_img_np = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
    calib_intr, _, _ = get_calibration_info(calib_intr_path)
    calib_extr, _, _ = get_calibration_info(calib_extr_path)

    fx, fy = calib_intr['camera_matrix']['data'][0], calib_intr['camera_matrix']['data'][4]
    cx, cy = calib_intr['camera_matrix']['data'][2], calib_intr['camera_matrix']['data'][5]
    os12cam = np.array(calib_extr['extrinsic_matrix']['data']).reshape(4, 4)
    cam2os1 = np.linalg.inv(os12cam)

    pc_np = convert_depth_to_pc(depth_img_np, fx, fy, cx, cy, range=depth_range)
    
    pc_np_homo = np.hstack((pc_np, np.ones((pc_np.shape[0], 1))))

    # BEGIN Sanity Check
    # euler_rot = R.from_euler('xyz', [81.9, 180, 92.5], degrees=True)
    # trans = np.array([0.19, 0.16, -0.31])

    # euler_rot = R.from_euler('xyz', [81.9, 180, 83.5], degrees=True)
    # trans = np.array([0.01, -0.6, -0.32])
    # # trans = np.array([0, 0, 0])
   
    # import pdb; pdb.set_trace()
    # rot_extr = euler_rot.as_matrix().reshape(3,3)
    # cam2os1 = np.eye(4)
    # cam2os1[:3, :3] = rot_extr
    # cam2os1[:3, 3] = trans
    # END Sanity Check

    # BEGIN kinect depth check
    # euler_rot = R.from_euler('xyz', [0, 0, 0], degrees=True)
    # trans = np.array([0, 0, 0])
    # rot_extr = euler_rot.as_matrix().reshape(3,3)
    # cam2os1 = np.eye(4)
    # cam2os1[:3, :3] = rot_extr
    # cam2os1[:3, 3] = trans
    # END Sanity check

    trans_pc_np = (cam2os1 @ pc_np_homo.T).T
    trans_pc_np = trans_pc_np[:, :3].astype(np.float32)
    
    return trans_pc_np

def main(args):
    traj    = args.traj
    camid   = args.cam
    indir   = args.indir
    # stereo_dir = "/robodata/arthurz/Datasets/CODa_cal"
    depth_dir = join(indir, TRED_RAW_DIR, camid, str(traj))
    pc_dir      = join(indir, TRED_RAW_DIR, "os1", str(traj))
    ts_path     = join(indir, TIMESTAMPS_DIR, f'{traj}_frame_to_ts.txt')
    calib_extr_path = join(indir, CALIBRATION_DIR, str(traj), f'calib_os1_to_{camid}.yaml')
    calib_intr_path = join(indir, CALIBRATION_DIR, str(traj), f'calib_{camid}_intrinsics.yaml')
    rospy.init_node('coda_check_stereo')

    assert camid in STEREO_SETTINGS, f'Cam with id {camid} not found in stereo depth range defintions...'
    depth_range = [STEREO_SETTINGS[camid]['min_depth'], STEREO_SETTINGS[camid]['max_depth'] ]

    # Set up ROS publishers for transformed depth image and point cloud
    depth_pc_pub = rospy.Publisher(f'/coda/{camid}/depth_cloud', PointCloud2, queue_size=10)
    lidar_pc_pub = rospy.Publisher('/coda/ouster/lidar_packets', PointCloud2, queue_size=10)
    rate = rospy.Rate(5)

    # Read all 3d_files from directory
    pattern = join(depth_dir, f"*.png")
    file_paths = glob.glob(pattern)
    file_paths.sort()

    # Read in frame to ts map
    ts_np = np.loadtxt(ts_path).reshape(-1)

    for file_path in file_paths:
        filename = file_path.split('/')[-1]
        _, _, _, ts_str = get_filename_info(filename)
        dec = 10
        ts_str_dec = ts_str[:dec] + "." + ts_str[dec:]
        depth_ts =  float(ts_str_dec)

        depth_pc_np = process_stereo_to_pointcloud(file_path, calib_intr_path, calib_extr_path, depth_range=depth_range)
        pub_pc_to_rviz(depth_pc_np, depth_pc_pub, depth_ts)

        # Find closest point cloud to ts
        closest_lidar_frame = np.searchsorted(ts_np, depth_ts, side='left')
        lidar_ts = ts_np[closest_lidar_frame]
        lidar_path = set_filename_dir(indir, TRED_RAW_DIR, "os1", str(traj), closest_lidar_frame, include_name=True)
        lidar_np = read_bin(lidar_path, keep_intensity=False)

        pub_pc_to_rviz(lidar_np, lidar_pc_pub, lidar_ts)
        rate.sleep()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
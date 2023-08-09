# import pdb; pdb.set_trace()
import os
from os.path import join
import sys
import argparse
import numpy as np
import time
import json

from scipy.spatial.transform import Rotation as R
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped

# For imports
sys.path.append(os.getcwd())
from helpers.sensors import set_filename_dir, read_bin
from helpers.geometry import pose_to_homo, find_closest_pose, densify_poses_between_ts
from helpers.visualization import pub_pc_to_rviz, pub_pose

parser = argparse.ArgumentParser()
parser.add_argument('--traj', default="0",
                    help="number of trajectory, e.g. 1")


def yaw_to_homo(pose_np, yaw):
    trans = pose_np[:, 1:4]
    rot_mat = R.from_euler('z', yaw, degrees=True).as_matrix()
    tmp = np.expand_dims(np.eye(4, dtype=np.float64), axis=0)
    homo_mat = np.repeat(tmp, len(trans), axis=0)
    homo_mat[:, :3, :3] = rot_mat
    # homo_mat[:, :3, 3 ] = trans
    return homo_mat

def apply_hom_mat(pose_np, homo_mat, non_origin):
    _, x, y, z, _, _, _, _ = pose_np.transpose()
    x, y, z, ones = [p.reshape(-1, 1) for p in [x, y, z, np.ones(len(x))]]
    xyz1 = np.expand_dims(np.hstack((x, y, z, ones)), -1)

    if non_origin:
        xyz1_center = np.copy(xyz1)
        xyz1_center[:, :2] = xyz1[:, :2] - xyz1[0, :2]
        xyz1_center_rotated = np.matmul(homo_mat, xyz1_center)[:, :3, 0]
        xyz1_center_rotated[:, :2] = xyz1_center_rotated[:, :2] + xyz1[0, :2].reshape(1, -1)
        corrected_pose_np = xyz1_center_rotated
    else:
        corrected_pose_np = np.matmul(homo_mat, xyz1)[:, :3, 0]
    return corrected_pose_np

def correct_pose(pose_np, trajectory):
    dir = './json'
    fpath = os.path.join(dir, 'pose_correction.json')
    f = open(fpath, "r")
    pose_correction = json.load(f)
    f.close()

    JSON_NAMES = ["start_arr", "end_arr", "yaw_arr"]
    start_arr, end_arr, yaw_arr = [], [], []

    if trajectory in pose_correction.keys():
        traj_dict = pose_correction[trajectory]
        start_arr, end_arr, yaw_arr = traj_dict[JSON_NAMES[0]], traj_dict[JSON_NAMES[1]], traj_dict[JSON_NAMES[2]]

    corrected_pose = np.copy(pose_np)
    
    # handles multiple rotation
    for i in range(len(start_arr)): 
        start, end, yaw = start_arr[i], end_arr[i], yaw_arr[i]
        non_origin = False
        if start != 0:
            non_origin = True
        homo_mat = yaw_to_homo(corrected_pose[start:end, :], yaw)
        corrected_pose[start:end, 1:4] = apply_hom_mat(corrected_pose[start:end, :], homo_mat, non_origin)
    return corrected_pose


def main(args):
    trajectory = args.traj
    indir = "/robodata/arthurz/Datasets/CODa_dev"
    pose_path = f"{indir}/poses/{trajectory}.txt"
    # pose_path = f"{trajectory}.txt"
    ts_path = f"{indir}/timestamps/{trajectory}_frame_to_ts.txt"
    bin_dir   = f"{indir}/3d_comp/os1/{trajectory}/"
    outdir    = f"./cloud_to_laser/%s" % args.traj

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Initialize ros publishing
    rospy.init_node('bin_publisher', anonymous=True)
    pc_pub = rospy.Publisher('/coda/ouster/lidar_packets', PointCloud2, queue_size=10)
    pose_pub = rospy.Publisher('/coda/pose', PoseStamped, queue_size=10)

    pose_np = np.loadtxt(pose_path).reshape(-1, 8)
    pose_np = correct_pose(pose_np, trajectory)
    timestamp_np = np.loadtxt(ts_path).reshape(-1)
    
    print("\n" + "---"*15)
    print(f"\nNumber of lidar frames: {len(pose_np)}\n")

    # Iterate through all poses
    LIDAR_HEIGHT = 0.8 # meters
    ZBAND_HEIGHT = 0.5 # meters
    ZMIN_OFFSET = 1.9 # meters

    rate = rospy.Rate(100)

    dense_pose_np = densify_poses_between_ts(pose_np, timestamp_np)
    last_pose = None

    for pose_idx, pose in enumerate(pose_np):
        pose_ts = pose[0]
        print("pose ", pose_idx)
        closest_lidar_frame = np.searchsorted(timestamp_np, pose_ts, side='left')
        lidar_ts = timestamp_np[closest_lidar_frame]
        bin_path = set_filename_dir(indir, "3d_raw", "os1", trajectory, closest_lidar_frame, include_name=True)
        lidar_np = read_bin(bin_path, keep_intensity=False)

        # Filter all point between zmin and zmax, downsample angular to 1/4 original size
        lidar_np = lidar_np.reshape(128, 1024, -1)
        lidar_np = np.transpose(lidar_np, [1, 0, 2]) # result (1024, 128, 3)
        lidar_np = lidar_np[::, ::8, :] # downsample: 1024 -> 512 and 128 -> 32
        lidar_np = lidar_np.reshape(-1, 3)

        zmin = ZMIN_OFFSET - LIDAR_HEIGHT
        zmax = zmin+ZBAND_HEIGHT
        z_mask = np.logical_and(lidar_np[:, 2] > zmin, lidar_np[:, 2] < zmax)
        lidar_np = lidar_np[z_mask].reshape(-1, 3).astype(np.float32)

        LtoG                = pose_to_homo(pose) # lidar to global frame
        homo_lidar_np       = np.hstack((lidar_np, np.ones((lidar_np.shape[0], 1))))
        trans_homo_lidar_np = (LtoG @ homo_lidar_np.T).T
        trans_lidar_np      = trans_homo_lidar_np[:, :3] #.reshape(128, 1024, -1)
        trans_lidar_np      = trans_lidar_np.reshape(-1, 3).astype(np.float32)

        pub_pc_to_rviz(trans_lidar_np, pc_pub, lidar_ts)
        pub_pose(pose_pub, pose, pose[0])

        if last_pose is None:
            last_pose = pose
        else:
            # print("check is last poses are similar")
            # print(np.allclose(last_pose[1:], pose[1:], rtol=1e-5) )
            last_pose = pose
        rate.sleep()

if __name__ == "__main__":
    start_time = time.time()
    args = parser.parse_args()
    main(args)
    print("--- Final: %s seconds ---" % (time.time() - start_time))

# python -W ignore scripts/3d_legoloam_to_2d_rviz.py --traj 0
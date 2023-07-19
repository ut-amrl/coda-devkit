# import pdb; pdb.set_trace()
import os
from os.path import join
import sys
import argparse
import numpy as np
import time

from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import KDTree

import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped

from multiprocessing import Pool
import tqdm

# For imports
sys.path.append(os.getcwd())
from helpers.sensors import set_filename_dir, read_bin
from helpers.geometry import pose_to_homo, find_closest_pose, densify_poses_between_ts
from helpers.visualization import pub_pc_to_rviz, pub_pose

parser = argparse.ArgumentParser()
parser.add_argument('--traj', default="0",
                    help="number of trajectory, e.g. 1")
parser.add_argument('--option', default="hitl",
                    help="hitl for hitl SLAM and vis for visualization ")

def main(args):
    global option
    trajectory, option = args.traj, args.option
    indir = "/robodata/arthurz/Datasets/CODa_dev"
    # pose_path = f"{indir}/poses/{trajectory}.txt"
    pose_path = f"{trajectory}.txt"
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
    timestamp_np = np.loadtxt(ts_path).reshape(-1)
    
    # number of lidar frames
    # n_of_bin_files = 1000
    n_of_bin_files = len([entry for entry in os.listdir(bin_dir) if os.path.isfile(os.path.join(bin_dir, entry))])
    print("\n" + "---"*15)
    print(f"\nNumber of lidar frames: {len(pose_np)}\n")

    # Iterate through all poses
    LIDAR_HEIGHT = 0.8 # meters
    ZBAND_HEIGHT = 0.5 # meters
    ZMIN_OFFSET = 1.9 # meters

    rate = rospy.Rate(100)

    dense_pose_np = densify_poses_between_ts(pose_np, timestamp_np)
    last_pose = None

    # for frame in range(n_of_bin_files):
    #     if frame%frame_ds!=0:
    #         continue

    #     bin_path = set_filename_dir(indir, "3d_raw", "os1", trajectory, frame)
    #     lidar_np = read_bin(bin_path, keep_intensity=False)
    #     lidar_ts = timestamp_np[frame]
        
    #     # Find closest LiDAR frame to pose timestamp
    #     closest_pose        = dense_pose_np[frame]
    #     # closest_pose        = np.array([closest_pose[0], closest_pose[3], -closest_pose[1], -closest_pose[2],  closest_pose[4], closest_pose[7], -closest_pose[5], -closest_pose[6]])

    #     LtoG                = pose_to_homo(closest_pose) # lidar to global frame
    #     homo_lidar_np       = np.hstack((lidar_np, np.ones((lidar_np.shape[0], 1))))
    #     trans_homo_lidar_np = (LtoG @ homo_lidar_np.T).T
    #     trans_lidar_np      = trans_homo_lidar_np[:, :3].reshape(-1, 3).astype(np.float32)
    #     # Filter all point between zmin and zmax, downsample angular to 1/4 original size
    #     # zmin = LIDAR_HEIGHT + ZMIN_OFFSET
    #     # zmax = zmin+ZBAND_HEIGHT
    #     # z_mask = np.logical_and(trans_lidar_np[:, :, 2] > zmin, trans_lidar_np[:, :, 2] < zmax)
    #     # filtered_lidar_np = trans_lidar_np[z_mask]

    #     # LtoG = pose_to_homo(closest_pose) # lidar to global frame
    #     # trans_homo_lidar_np = (LtoG @ homo_lidar_np.T).T
    #     # trans_lidar_np = trans_homo_lidar_np[:, :3]

    #     print("Publishing lidar frame %s" % str(frame))
    #     # Transform and publish point cloud to rviz
    #     pub_pc_to_rviz(trans_lidar_np, pc_pub, lidar_ts)
    #     pub_pose(pose_pub, closest_pose, closest_pose[0])
    #     if last_pose is None:
    #         last_pose = closest_pose
    #     else:
    #         print("check is last poses are similar")
    #         print("last pose ", last_pose[1:])
    #         print("last pose ", closest_pose[1:])
    #         print(np.allclose(last_pose[1:], closest_pose[1:], rtol=1e-5) )
    #         last_pose = closest_pose
    #     # import pdb; pdb.set_trace()
    #     rate.sleep()
    for pose_idx, pose in enumerate(pose_np):
        pose_ts = pose[0]
        if pose_idx < 1000:
            continue
        print("pose ", pose_idx)
        closest_lidar_frame = np.searchsorted(timestamp_np, pose_ts, side='left')
        lidar_ts = timestamp_np[closest_lidar_frame]
        bin_path = set_filename_dir(indir, "3d_raw", "os1", trajectory, closest_lidar_frame)
        lidar_np = read_bin(bin_path, keep_intensity=False)

        # Filter all point between zmin and zmax, downsample angular to 1/4 original size
        lidar_np = lidar_np.reshape(128, 1024, -1)
        zmin = ZMIN_OFFSET - LIDAR_HEIGHT
        zmax = zmin+ZBAND_HEIGHT
        z_mask = np.logical_and(lidar_np[:, :, 2] > zmin, lidar_np[:, :, 2] < zmax)
        lidar_np = lidar_np[z_mask].reshape(-1, 3).astype(np.float32)

        LtoG                = pose_to_homo(pose) # lidar to global frame
        homo_lidar_np       = np.hstack((lidar_np, np.ones((lidar_np.shape[0], 1))))
        trans_homo_lidar_np = (LtoG @ homo_lidar_np.T).T
        trans_lidar_np      = trans_homo_lidar_np[:, :3] #.reshape(128, 1024, -1)
        trans_lidar_np      = trans_lidar_np.reshape(-1, 3).astype(np.float32)

        # # Filter all point between zmin and zmax, downsample angular to 1/4 original size
        # zmin = LIDAR_HEIGHT + ZMIN_OFFSET
        # zmax = zmin+ZBAND_HEIGHT
        # z_mask = np.logical_and(trans_lidar_np[:, :, 2] > zmin, trans_lidar_np[:, :, 2] < zmax)
        # trans_lidar_np = trans_lidar_np[z_mask].reshape(-1, 3).astype(np.float32)

        pub_pc_to_rviz(trans_lidar_np, pc_pub, lidar_ts)
        pub_pose(pose_pub, pose, pose[0])
        if last_pose is None:
            last_pose = pose
        else:
            print("check is last poses are similar")
            # print("last pose ", last_pose[1:])
            # print("last pose ", pose[1:])
            print(np.allclose(last_pose[1:], pose[1:], rtol=1e-5) )
            last_pose = pose
        # import pdb; pdb.set_trace()
        # rate.sleep()

if __name__ == "__main__":
    start_time = time.time()
    args = parser.parse_args()
    main(args)
    print("--- Final: %s seconds ---" % (time.time() - start_time))
    
    if (args.option == "vis"):
        os.system(f"scripts/debug_visualize.py --traj {args.traj}")

# python -W ignore scripts/3d_legoloam_to_2d_hitl.py --traj 0 --option hitl
# python -W ignore scripts/3d_legoloam_to_2d_hitl.py --traj 0 --option vis
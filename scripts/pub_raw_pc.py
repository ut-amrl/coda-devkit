import rospy
import os
from os.path import join
import argparse
import sys
import glob
import struct
import time
from sensor_msgs.msg import PointCloud2, Imu
from nav_msgs.msg import Odometry

import queue

import numpy as np
# For imports
sys.path.append(os.getcwd())

from helpers.visualization import pub_pc_to_rviz, pub_imu_to_rviz
from helpers.sensors import *
from helpers.geometry import *
from helpers.constants import TRED_RAW_DIR

from scripts.gen_lidar_egocomp import compensate_frame

pose_q = queue.Queue()

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--traj', default="0",
                    help="number of trajectory, defaults to 0")
parser.add_argument('-f', '--frame', default="0",
                    help="lidar frame to begin at, defaults to 0")
parser.add_argument('-hz', type=float, default=1,
                    help="lidar publish rate, defaults to 1 Hz")

OS1_POINTCLOUD_SHAPE  = [1024, 128]

def publish_single_bin(bin_path, pc_pub, frame, ts=None, frame_id="os_sensor", do_frame_comp=False):
    bin_np = read_bin(bin_path)

    if do_frame_comp:
        bin_to_ply(bin_np[:, :3].astype(np.float64), "./nocomp.pcd")

        filename = bin_path.split('/')[-1]
        _, _, _, frame = get_filename_info(filename)
        pose_path = join(indir, "poses", "%s.txt"%trajectory)
        pose_np     = np.fromfile(pose_path, sep=' ').reshape(-1, 8)
        right_ts_idx  = find_closest_pose(pose_np, ts, return_idx=True)
        left_ts_idx = np.clip(right_ts_idx-1, 0, pose_np.shape[0]-1)
        start_pose = pose_np[left_ts_idx]
        end_pose = pose_np[right_ts_idx]

        bin_np = compensate_frame(bin_path, ts, start_pose, end_pose)
        bin_to_ply(bin_np[:, :3].astype(np.float64), "./comp.pcd")
    else:
        bin_np = bin_np.reshape(OS1_POINTCLOUD_SHAPE[1], OS1_POINTCLOUD_SHAPE[0], -1)
        # Add ring and timestamp to point cloud
        sim_rel_ts = np.linspace(0, 0.099, 1024)
        ts_points = np.tile(sim_rel_ts, (OS1_POINTCLOUD_SHAPE[1], 1))

        ring_idx = np.arange(0, 128, 1).reshape(-1, 1)
        ring = np.tile(ring_idx, (1, OS1_POINTCLOUD_SHAPE[0]))
        full_bin_np = np.dstack((bin_np, ts_points, ring)).astype(np.float32)
        bin_np = full_bin_np.reshape(-1, 6)

    if ts is None:
        ts = rospy.Time.now()

    # Publish pc to rviz
    pub_pc_to_rviz(full_bin_np, pc_pub, ts, point_type="x y z i", frame_id=frame_id, seq=frame)

def pose_handler(data):
    print("Received LeGO-LOAM pose message")
    pose_q.put(data)

def pose_to_str(pose_ros):
    pose_str = ""
    ts = pose_ros.header.stamp.to_sec()

    ts_str = f"{ts:.6f}"
    pose_list = [pose_ros.pose.pose.position.x, pose_ros.pose.pose.position.y, pose_ros.pose.pose.position.z,
        pose_ros.pose.pose.orientation.w, pose_ros.pose.pose.orientation.x, pose_ros.pose.pose.orientation.y, pose_ros.pose.pose.orientation.z]
    formatted_pose = [f"{num:.8f}" for num in pose_list]

    pose_str = " ".join([ts_str] + formatted_pose)
    return pose_str

if __name__ == '__main__':
    args = parser.parse_args()
    trajectory = int(args.traj)
    start_frame = int(args.frame)
    pub_hz = float(args.hz)

    rospy.init_node('bin_to_pointcloud_publisher', anonymous=True)

    # Set the directory path where the .bin files are located
    indir = "/robodata/arthurz/Datasets/CODa_dev"

    # Set the topic name to publish the PointCloud2 data
    ouster_topic = "/coda/ouster/points"
    imu_topic = "/coda/vectornav/IMU"

    # Get IMU data
    imu_dir = join(indir, POSES_DIR, "imu")
    imu_path = join(imu_dir, "%i.txt"%trajectory)
    imu_np = np.fromfile(imu_path, sep=' ').reshape(-1, 11)

    # Get all .bin file paths in the directory
    bin_dir = set_filename_dir(indir, TRED_RAW_DIR, "os1", trajectory, include_name=False)
    bin_paths = [ bin_path.path for bin_path in os.scandir(bin_dir) if bin_path.path.endswith(".bin") ]
    bin_paths.sort(
        key=lambda binpath: int(binpath.split("/")[-1].split(".")[0].split("_")[-1])
    )
    frame_to_ts_path = join(indir, "timestamps", "%i.txt"%trajectory)
    frame_to_ts_np = np.fromfile(frame_to_ts_path, sep=' ').reshape(-1,)

    pc_pub_rate = rospy.Rate(pub_hz)
    imu_pub_rate = rospy.Rate(pub_hz * 2) # Assume 2 times speed increase
    # Set pc publisher
    pc_pub = rospy.Publisher(ouster_topic, PointCloud2, queue_size=5)
    imu_pub = rospy.Publisher(imu_topic, Imu, queue_size=5)
    # pose_sub = rospy.Subscriber("/integrated_to_init", Odometry, pose_handler)
    
    # pose_path = "laserodomposes/%i.txt"%trajectory

    # # Reset pose file
    # pose_txt = open(pose_path, 'w')
    # pose_txt.close()

    if not bin_paths:
        rospy.loginfo("No .bin files found in the directory!")
    else:
        rospy.loginfo("Found {} .bin files in the directory.".format(len(bin_paths)))

        lidar_ts = frame_to_ts_np[start_frame]
        imu_index = max(np.searchsorted(imu_np[:, 0], lidar_ts)-1, 0) # Start one frame before

        # Publish all .bin files in the directory + imu data
        for frame in range(start_frame, len(bin_paths)):
            lidar_ts = frame_to_ts_np[frame]

            # Publish all imu messages earlier than this frame
            imu_ts = imu_np[imu_index][0]
            while imu_ts < lidar_ts and imu_index < len(imu_np):
                pub_imu_to_rviz(imu_np[imu_index], imu_pub)
                imu_index += 1
                imu_ts = imu_np[imu_index][0]
                imu_pub_rate.sleep()

            # Publish latest lidar message
            bin_path = bin_paths[frame]
            rospy.loginfo("Publishing file: " + bin_path)
            publish_single_bin(bin_path, pc_pub, frame, lidar_ts)
            pc_pub_rate.sleep()
            # while not pose_q.empty():
            #     pose_txt = open(pose_path, 'a')
            #     pose = pose_q.get()
            #     pose_str = pose_to_str(pose)
            #     print("pose str ", pose_str)
            #     pose_txt.write(pose_str + "\n")
            #     pose_txt.close()
            # import pdb; pdb.set_trace()
        # num_frames_published = 0
        # for bin_path in bin_paths:
        #     filename = bin_path.split('/')[-1]
        #     _, _, _, frame = get_filename_info(filename)
        #     frame = int(frame)

        #     ts = frame_to_ts_np[frame]
        #     rospy.loginfo("Publishing file: " + bin_path)
        #     publish_single_bin(bin_path, pc_pub, ts)
        #     pub_rate.sleep()

        #     num_frames_published += 1
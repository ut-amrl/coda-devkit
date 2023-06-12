import rospy
import os
from os.path import join
import sys
import glob
import struct
import time
from sensor_msgs.msg import PointCloud2

import numpy as np
# For imports
sys.path.append(os.getcwd())

from helpers.visualization import pub_pc_to_rviz
from helpers.sensors import *
from helpers.geometry import *
from helpers.constants import TRED_RAW_DIR

from scripts.egomotion_comp import compensate_frame

def publish_single_bin(bin_path, pc_pub, ts=None, frame_id="os_sensor"):
    bin_np = read_bin(bin_path)
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
    import pdb; pdb.set_trace()
    if ts is None:
        ts = rospy.Time.now()

    # Publish pc to rviz
    pub_pc_to_rviz(bin_np, pc_pub, rospy.Time.now(), frame_id)

    # time.sleep(0.5)
    # pub_pc_to_rviz(bin_np, pc_pub, ts, frame_id)
    # import pdb; pdb.set_trace()

if __name__ == '__main__':
    rospy.init_node('bin_to_pointcloud_publisher', anonymous=True)

    # Set the directory path where the .bin files are located
    indir = "/robodata/arthurz/Datasets/CODa"

    # Set the topic name to publish the PointCloud2 data
    topic_name = "/coda/ouster/points"

    # Set the trajectory to publish
    trajectory = 6 # 11
    frame_list = [119]
    if len(frame_list)==1:
        for i in range(2):
            frame_list.append(frame_list[-1]+1)

    # Get all .bin file paths in the directory
    bin_dir = set_filename_dir(indir, TRED_RAW_DIR, "os1", trajectory, include_name=False)
    bin_paths = [ bin_path.path for bin_path in os.scandir(bin_dir) if bin_path.path.endswith(".bin") ]
    bin_paths.sort(
        key=lambda binpath: int(binpath.split("/")[-1].split(".")[0].split("_")[-1])
    )
    frame_to_ts_path = join(indir, "timestamps", "%i_frame_to_ts.txt"%trajectory)
    frame_to_ts_np = np.fromfile(frame_to_ts_path, sep=' ').reshape(-1, 1)

    pub_rate = rospy.Rate(10)
    # Set pc publisher
    pc_pub = rospy.Publisher(topic_name, PointCloud2, queue_size=5)

    if not bin_paths:
        rospy.loginfo("No .bin files found in the directory!")
    else:
        rospy.loginfo("Found {} .bin files in the directory.".format(len(bin_paths)))

        num_frames_published = 0
        for bin_path in bin_paths:
            filename = bin_path.split('/')[-1]
            _, _, _, frame = get_filename_info(filename)
            frame = int(frame)
            if frame not in frame_list:
                continue

            ts = frame_to_ts_np[frame][0]
            rospy.loginfo("Publishing file: " + bin_path)
            publish_single_bin(bin_path, pc_pub, ts)
            pub_rate.sleep()
            # import pdb; pdb.set_trace()

            num_frames_published += 1
            if num_frames_published==len(frame_list):
                break   

import os
import sys
import pdb
import yaml
import numpy as np
from helpers.sensors import get_filename_info, set_filename_by_prefix

#ROS Imports
import rospy
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PoseStamped

import cv2
from cv_bridge import CvBridge

# For imports
sys.path.append(os.getcwd())

#CustomImports
from helpers.visualization import pub_pc_to_rviz
from helpers.geometry import inter_pose
from helpers.constants import SENSOR_DIRECTORY_FILETYPES, OS1_POINTCLOUD_SHAPE

def main():

    settings_fp = os.path.join(os.getcwd(), "config/visualize.yaml")
    with open(settings_fp, 'r') as settings_file:
        settings = yaml.safe_load(settings_file)
        indir   = settings['dataset_input_root']
        outdir  = settings['dataset_output_root']

        assert os.path.exists(outdir), "%s does not exist, provide valid dataset root directory."

    assert len(sys.argv)>=2, "Specify the trajectory number you wish to visualize\n"
    trajectory = sys.argv[1]

    bin_subdir = "3d_raw/os1"
    bin_dir = os.path.join(outdir, bin_subdir, str(trajectory))
    img_subdir = "2d_raw/cam0"
    img_dir = os.path.join(outdir, img_subdir, str(trajectory))
    assert os.path.exists(bin_dir), "%s does not exist, generate the trajectory's .bin files first\n"

    #Initialize ros point cloud publisher
    rospy.init_node('bin_publisher', anonymous=True)
    pc_pub = rospy.Publisher('/coda/bin/points', PointCloud2, queue_size=10)
    img_pub = rospy.Publisher('/coda/stereo/cam0', Image, queue_size=10)
    pose_pub = rospy.Publisher('/coda/pose', PoseStamped, queue_size=10)
    pub_rate = rospy.Rate(10) # Publish at 10 hz

    bin_files       = np.array([
        int(bin_file.split('.')[0].split("_")[-1])
        for bin_file in os.listdir(bin_dir)
    ])
    bin_files_idx   = np.argsort(bin_files)
    bin_files       = np.array(os.listdir(bin_dir))[bin_files_idx]

    frame_to_ts_file = os.path.join(indir, "timestamps", "%s_frame_to_ts.txt"%trajectory)
    pose_file   = os.path.join(indir, "poses", "%s.txt"%trajectory)
    pose_np     = np.fromfile(pose_file, sep=' ').reshape(-1, 8)
    frame_ts_np = np.fromfile(frame_to_ts_file, sep=' ').reshape(-1, 1)

    for filename in bin_files:
        # pdb.set_trace()
        _, _, trajectory, frame = get_filename_info(filename)
        filetype = filename.split(".")[-1]
        if filetype!=SENSOR_DIRECTORY_FILETYPES[bin_subdir]:
            continue

        #Publish image
        img_file = set_filename_by_prefix("2d_raw", "cam0", trajectory, frame)
        # pdb.set_trace()
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        bridge = CvBridge()
        img_msg = bridge.cv2_to_imgmsg(img, encoding="passthrough")
        img_pub.publish(img_msg)

        #Publish point cloud
        bin_file = os.path.join(bin_dir, filename)
        bin_np = np.fromfile(bin_file, dtype=np.float32).reshape(OS1_POINTCLOUD_SHAPE)
        pub_pc_to_rviz(bin_np, pc_pub, rospy.get_rostime(), frame_id="os_sensor")
        
        #Publish pose
        ts  = frame_ts_np[int(frame)][0]
        curr_ts_idx = np.searchsorted(pose_np[:, 0], ts, side="left")
        next_ts_idx = curr_ts_idx + 1
        if next_ts_idx>=pose_np.shape[0]:
            next_ts_idx = pose_np.shape[0] - 1
        
        pose = inter_pose(pose_np[curr_ts_idx], pose_np[next_ts_idx], ts)
        
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = "os_sensor"
        p.pose.position.x = pose[1]
        p.pose.position.y = pose[2]
        p.pose.position.z = pose[3]
        # Make sure the quaternion is valid and normalized
        p.pose.orientation.x = pose[5]
        p.pose.orientation.y = pose[6]
        p.pose.orientation.z = pose[7]
        p.pose.orientation.w = -pose[4]
        pose_pub.publish(p)

        pub_rate.sleep()


if __name__ == "__main__":
    main()
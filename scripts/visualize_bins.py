
import os
from signal import pause
import sys
import pdb
import yaml
import numpy as np

#ROS Imports
import rospy
from sensor_msgs.msg import PointCloud2, Image

import cv2
from cv_bridge import CvBridge

# For imports
sys.path.append(os.getcwd())

#CustomImports
from helpers.visualization import pub_pc_to_rviz
from helpers.constants import SENSOR_DIRECTORY_FILETYPES, OS1_POINTCLOUD_SHAPE

def main():

    settings_fp = os.path.join(os.getcwd(), "config/visualize.yaml")
    with open(settings_fp, 'r') as settings_file:
        settings = yaml.safe_load(settings_file)
        outdir    = settings['dataset_output_root']

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
    pc_pub = rospy.Publisher('/ouster/bin/points', PointCloud2, queue_size=10)
    img_pub = rospy.Publisher('/stereo/cam0', Image, queue_size=10)
    pub_rate = rospy.Rate(10) # Publish at 10 hz

    bin_files       = np.array([int(bin_file.split('.')[0]) for bin_file in os.listdir(bin_dir)])
    bin_files_idx   = np.argsort(bin_files)
    bin_files       = np.array(os.listdir(bin_dir))[bin_files_idx]

    for filename in bin_files:
        # pdb.set_trace()
        filetype = filename.split(".")[-1]
        if filetype!=SENSOR_DIRECTORY_FILETYPES[bin_subdir]:
            continue

        img_name = filename.split(".")[0]
        img_path = os.path.join(img_dir, "%s.%s"%(img_name, SENSOR_DIRECTORY_FILETYPES[img_subdir]))
        img = cv2.imread(img_path)
        bridge = CvBridge()
        img_msg = bridge.cv2_to_imgmsg(img, encoding="passthrough")
        img_pub.publish(img_msg)

        bin_file = os.path.join(bin_dir, filename)
        bin_np = np.fromfile(bin_file, dtype=np.float32).reshape(OS1_POINTCLOUD_SHAPE)
        pub_pc_to_rviz(bin_np, pc_pub, rospy.get_rostime(), frame_id="os_sensor")
        
        pub_rate.sleep()


if __name__ == "__main__":
    main()
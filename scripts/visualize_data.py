
import os
import sys
import pdb
import yaml
import numpy as np
from helpers.sensors import get_filename_info, set_filename_by_prefix

#ROS Imports
import rospy
from sensor_msgs.msg import PointCloud2, Image

# For imports
sys.path.append(os.getcwd())

#CustomImports
from helpers.visualization import *
from helpers.geometry import inter_pose
from helpers.constants import OS1_POINTCLOUD_SHAPE

def main():

    settings_fp = os.path.join(os.getcwd(), "config/visualize.yaml")
    with open(settings_fp, 'r') as settings_file:
        settings = yaml.safe_load(settings_file)
        viz_anno    = settings['viz_anno']
        viz_img     = settings['viz_img']
        viz_pc      = settings['viz_pc']
        viz_pose    = settings['viz_pose']

        img_root_dir    = settings['img_root_dir']
        bin_root_dir    = settings['bin_root_dir']
        pose_root_dir   = settings['pose_root_dir']
        ts_root_dir     = settings['ts_root_dir']
        bbox3d_root_dir = settings['bbox3d_root_dir']

        assert os.path.exists(img_root_dir), "%s does not exist, provide valid img_root_dir."
        assert os.path.exists(bin_root_dir), "%s does not exist, provide valid bin_root_dir."
        assert os.path.exists(pose_root_dir), "%s does not exist, provide valid pose_root_dir."
        assert os.path.exists(ts_root_dir), "%s does not exist, provide valid ts_root_dir."
        assert os.path.exists(bbox3d_root_dir), "%s does not exist, provide valid ts_root_dir."

    assert len(sys.argv)>=2, "Specify the trajectory number you wish to visualize\n"
    trajectory = sys.argv[1]

    bin_dir = os.path.join(bin_root_dir, str(trajectory))
    img_dir = os.path.join(img_root_dir, str(trajectory))
    assert os.path.exists(bin_dir), "%s does not exist, generate the trajectory's .bin files first\n"

    #Initialize ros point cloud publisher
    rospy.init_node('bin_publisher', anonymous=True)
    pc_pub = rospy.Publisher('/coda/bin/points', PointCloud2, queue_size=10)
    img_pub = rospy.Publisher('/coda/stereo/cam0', Image, queue_size=10)
    pose_pub = rospy.Publisher('/coda/pose', PoseStamped, queue_size=10)
    marker_pub = rospy.Publisher('/coda/bbox', MarkerArray, queue_size=10)
    pub_rate = rospy.Rate(10) # Publish at 10 hz

    bin_files       = np.array([
        int(bin_file.split('.')[0].split("_")[-1])
        for bin_file in os.listdir(bin_dir)
    ])
    bin_files_idx   = np.argsort(bin_files)
    bin_files       = np.array(os.listdir(bin_dir))[bin_files_idx]

    frame_to_ts_file = os.path.join(ts_root_dir, "%s_frame_to_ts.txt"%trajectory)
    pose_file   = os.path.join(pose_root_dir, "%s.txt"%trajectory)
    pose_np     = np.fromfile(pose_file, sep=' ').reshape(-1, 8)
    frame_ts_np = np.fromfile(frame_to_ts_file, sep=' ').reshape(-1, 1)

    for filename in bin_files:
        frame_time = rospy.get_rostime()
        modality, sensor_name, trajectory, frame = get_filename_info(filename)
        print("Visualizing frame ", frame)

        #Publish image
        if viz_img:
            img_file = set_filename_by_prefix("2d_raw", "cam0", trajectory, frame)
            # pdb.set_trace()
            img_path = os.path.join(img_dir, img_file)
            pub_img(img_pub, img_path)
        
        #Publish pose
        ts  = frame_ts_np[int(frame)][0]
        if viz_pose:
            curr_ts_idx = np.searchsorted(pose_np[:, 0], ts, side="left")
            next_ts_idx = curr_ts_idx + 1
            if next_ts_idx>=pose_np.shape[0]:
                next_ts_idx = pose_np.shape[0] - 1
            pose = inter_pose(pose_np[curr_ts_idx], pose_np[next_ts_idx], ts)
            pub_pose(pose_pub, pose, frame, frame_time)

        #Publish point cloud
        if viz_pc:
            bin_file = os.path.join(bin_dir, filename)
            bin_np = np.fromfile(bin_file, dtype=np.float32).reshape(OS1_POINTCLOUD_SHAPE)
            pub_pc_to_rviz(bin_np, pc_pub, frame_time, frame_id="os_sensor")

        # Publish bboxes
        if viz_anno:
            label_modality = modality.replace("raw", "label")
            label_filename = set_filename_by_prefix(label_modality, sensor_name, 
                str(trajectory), frame)
            threed_label_file = os.path.join(bbox3d_root_dir, str(trajectory), 
                label_filename)

            pub_3dbbox_to_rviz(marker_pub, threed_label_file, frame_time, verbose=True)
        
        pub_rate.sleep()


if __name__ == "__main__":
    main()
import os
import argparse
import json
import select
from os.path import join

import numpy as np
from scipy.spatial.transform import Rotation as R

import rospy
from cv_bridge import CvBridge
import cv2
import tf2_ros
import tf.transformations as tf_trans

from sensor_msgs.msg import PointCloud2, Image
from visualization_msgs.msg import  Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Point

# For imports
import sys
sys.path.append(os.getcwd())

from helpers.visualization import (clear_marker_array, create_3d_bbox_marker, pub_pose,
                                    project_3dbbox_image, pub_pc_to_rviz, apply_semantic_cmap,
                                    apply_rgb_cmap, project_3dpoint_image)
from helpers.calibration import load_extrinsic_matrix, load_camera_params
from helpers.sensors import (get_calibration_info, set_filename_dir, read_bin, read_sem_label)
from helpers.geometry import pose_to_homo
from helpers.constants import *
from scripts.check_stereo_rgb import extract_ts

from scripts.gen_pc_for_js import (downsample_point_cloud, save_bin_file, read_bbox_file, save_bbox_file)

# from helpers.ros_visualization import publish_3d_bbox

import sys
import termios
import tty

parser = argparse.ArgumentParser(description="CODa rviz visualizer")
parser.add_argument("-s", "--sequence", type=str, default="0", 
                    help="Sequence number (Default 0)")
parser.add_argument("-f", "--start_frame", type=str, default="0",
                    help="Frame to start at (Default 0)")
parser.add_argument("-c", "--color_type", type=str, default="classId", 
                    help="Color map to use for coloring boxes Options: [isOccluded, classId] (Default classId)")
parser.add_argument("-l", "--log", type=str, default="",
                    help="Logs point cloud and bbox annotations to file for external usage")
parser.add_argument("-n", "--namespace", type=str, default="coda",
                    help="Select a namespace to use for published topics")

def get_key():
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    return ch

def vis_annos_rviz(args):
    indir = os.getenv(ENV_CODA_ROOT_DIR)
    assert indir is not None, f'Directory for CODa cannot be found, set {ENV_CODA_ROOT_DIR}'
    sequence, start_frame, color_type, log_dir = args.sequence, int(args.start_frame), args.color_type, \
        args.log
    namespace = args.namespace
    rospy.init_node('CODa_publisher')

    # Define Frames
    global_frame = 'map'
    base_frame   = 'os_sensor'
    lidar_frame  = 'os_sensor'

    lidar_pub   = rospy.Publisher(f'/{namespace}/ouster/points', PointCloud2, queue_size=10)
    cam0_pub    = rospy.Publisher(f'/{namespace}/cam0', Image, queue_size=10)
    cam1_pub    = rospy.Publisher(f'/{namespace}/cam1', Image, queue_size=10)
    bbox_3d_pub = rospy.Publisher(f'/{namespace}/bbox_3d', MarkerArray, queue_size=10)
    pose_pub    = rospy.Publisher(f'/{namespace}/pose', PoseStamped, queue_size=10)
    cam3_pub    = rospy.Publisher(f'/{namespace}/cam3/depth', Image, queue_size=10)

    trajectory_pub    = rospy.Publisher('/trajectory', Marker, queue_size=10)
    trajectory_marker = Marker()
    trajectory_marker.header.frame_id = global_frame
    trajectory_marker.ns = "trajectory"
    trajectory_marker.id = 0
    trajectory_marker.type = Marker.LINE_STRIP
    trajectory_marker.action = Marker.ADD
    trajectory_marker.pose.orientation.w = 1.0  # Identity orientation
    trajectory_marker.scale.x = 0.5  # Line width
    trajectory_marker.color.g = 1.0
    trajectory_marker.color.b = 1.0
    trajectory_marker.color.a = 1.0  # Don't forget to set alpha!

    # Define TF Broadcaster
    tf_broadcaster = tf2_ros.TransformBroadcaster()
    rate = rospy.Rate(300)
    
    # for sequence in sequences:
    # Path to the data
    calib_dir       = join(indir, CALIBRATION_DIR, sequence)
    lidar_ts_dir    = join(indir, TIMESTAMPS_DIR,
                                    f"{sequence}.txt")
    poses_dir       = join(indir, POSES_DIR, "global")
    
    # Pose DATA
    pose_file   = join(poses_dir, f'{sequence}.txt')
    pose_np     = np.fromfile(pose_file, sep=' ').reshape(-1, 8)

    lidar_ts_np = np.loadtxt(lidar_ts_dir, dtype=np.float64)
    
    # Calibration DATA (Extrinsic and Intrinsic)
    os1_to_base_ext_file = join(calib_dir, "calib_os1_to_base.yaml")
    os1_to_cam0_ext_file = join(calib_dir, "calib_os1_to_cam0.yaml")
    os1_to_cam1_ext_file = join(calib_dir, "calib_os1_to_cam1.yaml")

    cam0_intrinsics_file = join(calib_dir, "calib_cam0_intrinsics.yaml")
    cam1_intrinsics_file = join(calib_dir, "calib_cam1_intrinsics.yaml")
    
    os1_to_base_ext = load_extrinsic_matrix(os1_to_base_ext_file) 
    os1_to_cam0_ext = load_extrinsic_matrix(os1_to_cam0_ext_file)
    os1_to_cam1_ext = load_extrinsic_matrix(os1_to_cam1_ext_file)

    cam0_K, cam0_D, cam0_size = load_camera_params(cam0_intrinsics_file)
    cam1_K, cam1_D, cam1_size = load_camera_params(cam1_intrinsics_file)

    # Stereo Depth DATA
    stereo_img_dir = set_filename_dir("/robodata/arthurz/Datasets/CODa", TRED_RAW_DIR, "cam3", sequence)
    if os.path.exists(stereo_img_dir):
        stereo_img_files = np.array([img_file for img_file in os.listdir(stereo_img_dir) if img_file.endswith('.png')])
        stereo_img_ts = np.array([extract_ts(img_file) for img_file in stereo_img_files])
        stereo_sort_indices = np.argsort(stereo_img_ts) # low to high
        stereo_img_files = stereo_img_files[stereo_sort_indices]
        stereo_img_ts = stereo_img_ts[stereo_sort_indices]

    pose_skip = 1
    last_frame = 0
    for pose_idx, pose in enumerate(pose_np):
        if pose_idx % pose_skip != 0:
            continue
        print(f'Publishing sensor info at pose {pose_idx}')
        # Get Pose
        pose_ts, x, y, z, qw, qx, qy, qz = pose
        base_pose = np.eye(4)
        base_pose[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
        base_pose[:3, 3] = [x, y, z]

        # Publish Pose
        pub_pose(pose_pub, pose, pose_ts, global_frame)

        point = Point()
        point.x = x
        point.y = y
        point.z = z
        trajectory_marker.points.append(point)
        trajectory_pub.publish(trajectory_marker)

        # Get the closest frame
        # frame = last_frame + np.searchsorted(lidar_ts_np[last_frame:],
        #                                      pose_ts, side='left')
        frame = np.searchsorted(lidar_ts_np, pose_ts, side='left')

        if frame == last_frame:
            continue

        lidar_ts = rospy.Time.from_sec(lidar_ts_np[frame])
        last_frame = frame

        if frame < start_frame:
            continue

        # Broadcast TF (odom -> os1)
        tf_msg = tf2_ros.TransformStamped()
        tf_msg.header.stamp = lidar_ts
        tf_msg.header.frame_id = global_frame
        tf_msg.child_frame_id = lidar_frame

        tf_msg.transform.translation.x = base_pose[0, 3]
        tf_msg.transform.translation.y = base_pose[1, 3]
        tf_msg.transform.translation.z = base_pose[2, 3]

        rotation = tf_trans.quaternion_from_matrix(base_pose)
        tf_msg.transform.rotation.x = rotation[0]
        tf_msg.transform.rotation.y = rotation[1]
        tf_msg.transform.rotation.z = rotation[2]
        tf_msg.transform.rotation.w = rotation[3]

        tf_broadcaster.sendTransform(tf_msg)
        
        # Get the path to the data
        pc_file   = set_filename_dir(indir, TRED_COMP_DIR, "os1", sequence, frame, include_name=True)
        cam0_file = set_filename_dir(indir, TWOD_RECT_DIR, "cam0", sequence, frame, include_name=True)
        cam1_file = set_filename_dir(indir, TWOD_RECT_DIR, "cam1", sequence, frame, include_name=True)
        bbox_file = set_filename_dir(indir, TRED_BBOX_LABEL_DIR, "os1", sequence, frame, include_name=True)
        sem_file = set_filename_dir(indir, SEMANTIC_LABEL_DIR, "os1", sequence, frame, include_name=True)

        # Publish Point Cloud
        if os.path.exists(pc_file):
            lidar_np = read_bin(pc_file, keep_intensity=False)

            LtoG                = pose_to_homo(pose) # lidar to global frame
            homo_lidar_np       = np.hstack((lidar_np, np.ones((lidar_np.shape[0], 1))))
            trans_homo_lidar_np = (LtoG @ homo_lidar_np.T).T
            trans_lidar_np      = trans_homo_lidar_np[:, :3]
            trans_lidar_np      = trans_lidar_np.reshape(-1, 3).astype(np.float32)

            point_type = "x y z"
            sem_color = None
            if os.path.exists(cam0_file):
                sem_color, pc_mask = apply_rgb_cmap(cam0_file, lidar_np, os1_to_cam0_ext_file,
                    cam0_intrinsics_file, return_pc_mask=True)

                trans_lidar_np = trans_lidar_np[pc_mask, :]
                sem_color = sem_color[pc_mask, :].astype(np.float32) / 255.0

            if os.path.exists(sem_file):
                sem_color_anno = apply_semantic_cmap(sem_file) / 255.0
                sem_color_anno = np.array([[b, g, r] for r, g, b in sem_color_anno], dtype=np.float32)
                
                if sem_color is not None:
                    sem_labels = read_sem_label(sem_file).astype(np.int32)
                    sem_color_anno_fov = sem_color_anno[pc_mask, :]
                    sem_labels_fov = sem_labels[pc_mask]

                    sem_label_nonbg = sem_labels_fov!=0
                    sem_color[sem_label_nonbg] = sem_color_anno_fov[sem_label_nonbg]
                else:
                    sem_color = sem_color_anno

            point_type="x y z r g b"
            trans_lidar_np = np.hstack((trans_lidar_np, sem_color))

            # Dump to file feature for external visualization
            if log_dir!="":
                point_cloud_cm = np.ones((pc_mask.shape[0], 3), dtype=np.float16) * 170 # Default point color is grey
                # Color with semantic label is exists
                if os.path.exists(sem_file):
                    sem_label_full_nonbg = sem_labels!=0
                    sem_color_full_nonbg = sem_color_anno[sem_label_full_nonbg, :]*255.0
                    point_cloud_cm[sem_label_full_nonbg, :] = sem_color_full_nonbg # 128*1024 x 3

                # Augment ego pc with colormap and ds
                nontrans_lidar_np = np.hstack((lidar_np, point_cloud_cm))
                downsampled_cloud = downsample_point_cloud(nontrans_lidar_np, 2, 2)

                assert os.path.exists(log_dir), f'{log_dir} does not exist, exiting...'
                pc_outpath = f'{log_dir}/{last_frame}.bin'
                save_bin_file(downsampled_cloud, pc_outpath)

            pub_pc_to_rviz(trans_lidar_np, lidar_pub, lidar_ts, 
                point_type=point_type, 
                frame_id=global_frame)


        # Publish the 3D Bounding Box
        if os.path.exists(bbox_file):
            bbox_3d_json = json.load(open(bbox_file, 'r'))

            # Draw 3d Bounding Box 
            clear_marker_array(bbox_3d_pub)
            bbox_3d_markers = MarkerArray()
            for bbox_3d in bbox_3d_json['3dbbox']:
                
                bbox_3d_color = (0, 0, 0, 1.0) # Black by default
                if color_type=="isOccluded":
                    color_id = OCCLUSION_TO_ID[bbox_3d['labelAttributes']['isOccluded']]
                    bbox_3d_color = OCCLUSION_ID_TO_COLOR[color_id]
                elif color_type=="classId":
                    color_id = BBOX_CLASS_TO_ID[bbox_3d['classId']]
                    bbox_3d_color_scaled_bgr = [c/255.0 for c in BBOX_ID_TO_COLOR[color_id] ] + [1]
                    bbox_3d_color_scaled_rgb = [
                        bbox_3d_color_scaled_bgr[2], bbox_3d_color_scaled_bgr[1], bbox_3d_color_scaled_bgr[0]
                    ]
                    bbox_3d_color = (bbox_3d_color_scaled_rgb)

                bbox_marker = create_3d_bbox_marker(
                    bbox_3d['cX'], bbox_3d['cY'], bbox_3d['cZ'],
                    bbox_3d['l'],  bbox_3d['w'],  bbox_3d['h'],
                    bbox_3d['r'],  bbox_3d['p'],  bbox_3d['y'],
                    lidar_frame, lidar_ts, bbox_3d['instanceId'],
                    int(bbox_3d['instanceId'].split(':')[-1]),
                    *bbox_3d_color,
                )
                bbox_3d_markers.markers.append(bbox_marker)

            # Log 3d Bbox for external viewing
            if log_dir!="":
                bbox_outpath = f'{log_dir}/{last_frame}bbox.bin'
                save_bbox_file(read_bbox_file(bbox_file), bbox_outpath)
                print(f'Logged bbox frame {last_frame} to {bbox_outpath}')
            
            bbox_3d_pub.publish(bbox_3d_markers)
        

        # Publish Camera Images
        if os.path.exists(cam0_file) and os.path.exists(cam1_file):
            # Camera 0
            cam0_image = cv2.imread(cam0_file, cv2.IMREAD_COLOR)
            # Camera 1
            cam1_image = cv2.imread(cam1_file, cv2.IMREAD_COLOR)

            if os.path.exists(sem_file):
                cam0_image = project_3dpoint_image(cam0_image, lidar_np, os1_to_cam0_ext_file, cam0_intrinsics_file, sem_file
                )

                cam1_image = project_3dpoint_image(cam1_image, lidar_np, os1_to_cam1_ext_file, cam1_intrinsics_file, sem_file
                )

            # Project 3D Bounding Box to 2D
            if os.path.exists(bbox_file):
                bbox_3d_json = json.load(open(bbox_file, 'r'))
                cam0_image = project_3dbbox_image(
                    bbox_3d_json, os1_to_cam0_ext_file, cam0_intrinsics_file,
                    cam0_image
                )

                cam1_image = project_3dbbox_image(
                    bbox_3d_json, os1_to_cam1_ext_file, cam1_intrinsics_file,
                    cam1_image
                )

            cam0_msg   = CvBridge().cv2_to_imgmsg(cam0_image)
            cam0_msg.header.stamp = lidar_ts
            cam0_pub.publish(cam0_msg)

            cam1_msg   = CvBridge().cv2_to_imgmsg(cam1_image)
            cam1_msg.header.stamp = lidar_ts
            cam1_pub.publish(cam1_msg)

        # Publish Stereo Depth Image if it exists
        if os.path.exists(stereo_img_dir):
            closest_stereo_img_idx = np.searchsorted(stereo_img_ts, pose_ts, side='left')
            closest_stereo_file = stereo_img_files[closest_stereo_img_idx]
            closest_stereo_path = join(stereo_img_dir, closest_stereo_file)
            stereo_img_np = cv2.imread(closest_stereo_path, cv2.IMREAD_GRAYSCALE)
            max_val = np.max(stereo_img_np)
            stereo_img_np = stereo_img_np * (255.0 / max_val)
            stereo_img_np = stereo_img_np.astype(np.uint16)
            cam3_ts = rospy.Time.from_sec(stereo_img_ts[closest_stereo_img_idx])
            cam3_msg   = CvBridge().cv2_to_imgmsg(stereo_img_np, encoding="mono16")

            cam3_msg.header.stamp = cam3_ts
            cam3_pub.publish(cam3_msg)

        rate.sleep()


if __name__ == '__main__':
    args = parser.parse_args()
    vis_annos_rviz(args)

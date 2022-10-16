import os
import pdb
import yaml

# Utility Libraries
import cv2 as cv
import numpy as np

# ROS Libraries
import rospy
import ros_numpy # Used in sensor_msgs.msg apt-get install ros-noetic-ros-numpy
import rosbag
import sensor_msgs
import nav_msgs
from visualization_msgs.msg import *
from geometry_msgs.msg import Point32
from std_msgs.msg import ColorRGBA

# User Defined Libraries
from constants import *

# https://stackoverflow.com/questions/39772424/how-to-effeciently-convert-ros-pointcloud2-to-pcl-point-cloud-and-visualize-it-i
def convert_pc_msg_to_np(pc_msg):
    # Fix rosbag issues, see: https://github.com/eric-wieser/ros_numpy/issues/23
    pc_msg.__class__ = sensor_msgs.msg._PointCloud2.PointCloud2
    offset_sorted = {f.offset: f for f in pc_msg.fields}
    pc_msg.fields = [f for (_, f) in sorted(offset_sorted.items())]

    # Extract x, y, z, intensity
    raw_buffer = np.frombuffer(pc_msg.data, dtype=np.uint8).reshape(pc_msg.width, -1)
    raw_pc = raw_buffer[:, 0:16].reshape(-1).tobytes()
    pc_np = np.frombuffer(raw_pc, dtype=np.float32).reshape(pc_msg.width, 4)

    return pc_np

def convert_pose_to_np(pose_msg):
    pose_msg.__class__ = nav_msgs.msg.Odometry
    ts, tns = pose_msg.header.stamp.secs, pose_msg.header.stamp.nsecs
    position, heading = pose_msg.pose.pose.position, pose_msg.pose.pose.orientation

    # TODO: check for coordinate transform between odom and velodyne frames
    return position, heading, ts, tns

def convert_bag_to_bins(data_root, save_root, header):
    rate, scene = header.values()

    data_dir = os.path.join(data_root, scene)
    if not os.path.exists(data_dir): os.makedirs(data_dir)

    bag_names = [file for file in sorted(os.listdir(data_dir)) 
        if not os.path.isdir(os.path.join(data_root, file) ) ]

    scene_settings_file = os.path.join(data_root, "scenes.yaml")
    scene_to_traj = get_scene_seq_map(scene_settings_file)
 
    for bag_name in bag_names:
        bag_path = os.path.join(data_dir, bag_name)
        idx = 0

        save_dir = os.path.join(save_root, "scans", "seq%d" % scene_to_traj[bag_name])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        odom_rec, velo_rec, image_rec = False, False, False
        last_msg = None
        for topic, msg, t in rosbag.Bag(bag_path).read_messages():
            if topic == "/velodyne_points":
                last_msg = msg
                velo_rec = True
            elif topic=="/odom" or topic=="/jackal_velocity_controller/odom":
                odom_rec = True
            elif topic=="/image_raw/compressed" or topic=="/camera/rgb/image_raw/compressed":
                image_rec = True

            if velo_rec and odom_rec and image_rec:
                velo_rec, odom_rec, image_rec = False, False, False
                if idx%rate==0:
                    pc_np = convert_pc_msg_to_np(last_msg)
                    ts  = last_msg.header.stamp.secs + last_msg.header.stamp.nsecs*1e-9
                    pc_filename = os.path.join(save_dir, str(ts).replace('.', '') + ".bin")
                    flat_pc = pc_np.reshape(-1)

                    # Save pc to binary pack format
                    flat_pc.tofile(pc_filename) # empty sep=bytes
                idx+=1

        pdb.set_trace()

def get_scene_seq_map(scene_settings_file):
    with open(scene_settings_file) as file:
        # The FullLoader parameter handles the conversion from YAML
        scene_list = yaml.load(file, Loader=yaml.FullLoader)
        scene_names = scene_list['filename']

        traj_to_scene = {}
        for scene_name in scene_names:
            traj_to_scene.update(scene_names[scene_name])
        scene_to_traj = {v: k for k, v in traj_to_scene.items()}
    return scene_to_traj


def create_manifest(data_root, save_root, manifest_header):
    prefix, rate, scene = manifest_header.values()

    data_dir = os.path.join(data_root, scene)
    save_dir = os.path.join(save_root, "sequences")
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    bag_names = [file for file in sorted(os.listdir(data_dir)) 
        if not os.path.isdir(os.path.join(data_dir, file) ) and file.endswith(".bag")]
    
    scene_settings_file = os.path.join(data_root, "scenes.yaml")
    scene_to_traj = get_scene_seq_map(scene_settings_file)

    prefix_text = PREFIX_TEXT % prefix

    manifest_frames_str = ""
    for bag_name in bag_names:
        bag_path = os.path.join(data_dir, bag_name)
        seq_no = scene_to_traj[bag_name]

        idx = 0
        frame_idx = 0
        frame_info = FRAME_TEXT_DICT
        odom_rec, velo_rec, img_rec = False, False, False
        for topic, msg, t in rosbag.Bag(bag_path).read_messages():
            if topic=="/odom" or topic=="/jackal_velocity_controller/odom":
                frame_info['ts']        = msg.header.stamp.secs + msg.header.stamp.nsecs*1e-9
                frame_info['frameno']   = msg.header.seq
                frame_info['evppx']     = msg.pose.pose.position.x
                frame_info['evppy']     = msg.pose.pose.position.y
                frame_info['evppz']     = msg.pose.pose.position.z
                frame_info['evphx']     = msg.pose.pose.orientation.x
                frame_info['evphy']     = msg.pose.pose.orientation.y
                frame_info['evphz']     = msg.pose.pose.orientation.z
                frame_info['evphw']     = msg.pose.pose.orientation.w
                
                # Estimated camera extrinsics from robot pose for now
                frame_info['ipx'], frame_info['ipy'], frame_info['ipz'] = \
                    msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z
                frame_info['ihx'], frame_info['ihy'], frame_info['ihz'], frame_info['ihw'] = \
                    msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
                odom_rec = True
            elif topic=="/velodyne_points":
                ts = str(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9).replace('.', '')
                frame_info['frame'] = "scans/seq%d/%s.bin" % (seq_no, ts)
                frame_info['ipath'] = "images/seq%d/%s.jpg" % (seq_no, ts)

                velo_rec = True
            elif topic=="/image_raw/compressed" or topic=="/camera/rgb/image_raw/compressed":
                frame_info['fx'], frame_info['fy'] = 603.6859, 606.3391
                frame_info['cx'], frame_info['cy'] = 646.9208, 380.1066
                frame_info['its'] = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
                img_rec = True

            if odom_rec and velo_rec and img_rec:
                odom_rec, velo_rec, img_rec = False, False, False
                frame_curr = FRAME_TEXT % frame_info

                if idx%rate==0:
                    # Format line breaks between frames
                    if frame_idx>0:
                        manifest_frames_str += ",\n"
                    manifest_frames_str += frame_curr
                    frame_idx += 1
                idx += 1

        # Accumulate manifest file
        seq_text = SEQ_TEXT % seq_no
        manifest_header_str = seq_text + prefix_text + NUM_FRAMES_TEXT % frame_idx
        manifest_file_str = manifest_header_str + FRAMES_START_TEXT + \
            manifest_frames_str + FRAMES_END_TEXT
        
        manifest_file = open(os.path.join(save_dir, "seq%d.json" % seq_no), "w")
        manifest_file.write(manifest_file_str)
        manifest_file.close()
        pdb.set_trace()

def ros_to_img(ros_img):
    raw_np = np.fromstring(ros_img.data, np.uint8)
    image_np = cv.imdecode(raw_np, cv.IMREAD_COLOR)

    return image_np

def create_images(data_root, save_root, header):
    rate, scene = header['rate'], header['scene']

    data_dir = os.path.join(data_root, scene)
    if not os.path.exists(data_dir): os.makedirs(data_dir)

    bag_names = [file for file in sorted(os.listdir(data_dir)) 
        if not os.path.isdir(os.path.join(data_dir, file) ) and file.endswith(".bag")]

    scene_settings_file = os.path.join(data_root, "scenes.yaml")
    scene_to_traj = get_scene_seq_map(scene_settings_file)

    for bag_name in bag_names:
        bag_path = os.path.join(data_dir, bag_name)
        idx = 0

        save_dir = os.path.join(save_root, "rgb", "seq%d" % scene_to_traj[bag_name])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        last_ts = 0
        last_raw_msg = None
        odom_rec, velo_rec, img_rec = False, False, False
        for topic, msg, t in rosbag.Bag(bag_path).read_messages():

            if topic=="/odom" or topic=="/jackal_velocity_controller/odom":
                odom_rec = True
            elif topic=="/velodyne_points":
                last_ts = msg.header.stamp.secs + msg.header.stamp.nsecs*1e-9
                velo_rec= True
            elif topic=="/image_raw/compressed" or topic=="/camera/rgb/image_raw/compressed":
                last_raw_msg = msg
                img_rec=True

            if odom_rec and img_rec and velo_rec:
                odom_rec, img_rec, velo_rec = False, False, False
                if idx%rate==0:
                    image_np = ros_to_img(last_raw_msg)

                    img_filename = os.path.join(save_dir, str(last_ts).replace('.', '') + ".jpg")
                    cv.imwrite(img_filename, image_np)
                idx+=1
        
        pdb.set_trace()

def open_bin_as_np(save_dir):
    bin_names = sorted(os.listdir(save_dir))

    for bin_name in bin_names:
        bin_path = os.path.join(save_dir, bin_name)
        bin_np = np.fromfile(bin_path, sep="", dtype=np.float32).reshape(-1, 4)
        pdb.set_trace()

def visualize_bins(save_dir):
    rospy.init_node('talker',disable_signals=True)
    map_pub = rospy.Publisher('PointCloud', MarkerArray, queue_size=10)
    bin_names = sorted(os.listdir(save_dir))

    frame_idx = 0
    next_map = MarkerArray()
    for bin_name in bin_names:
        bin_path = os.path.join(save_dir, bin_name)
        bin_np = np.fromfile(bin_path, sep="", dtype=np.float32).reshape(-1, 4)

        marker = Marker()
        marker.id = 0
        marker.ns = "point cloud"
        marker.header.frame_id = "map" # change this to match model + scene name LMSC_000001
        marker.type = marker.CUBE_LIST
        marker.action = marker.ADD
        marker.lifetime.secs = 0
        # marker.header.stamp = 0

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        for i in range(bin_np.shape[0]):              
            pc = bin_np[i].astype(np.uint32)

            point = Point32()
            color = ColorRGBA()
            point.x = pc[0]
            point.y = pc[1]
            point.z = pc[2]

            color.r, color.g, color.b = (255, 255, 255)

            color.a = 0.5
            marker.points.append(point)
            marker.colors.append(color)
        
        next_map.markers.append(marker)
        if frame_idx%10==0:
            map_pub.publish(next_map)
            next_map = MarkerArray()
            pdb.set_trace()

        frame_idx += 1
        

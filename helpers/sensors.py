import os
import pdb
import yaml
import shutil

#Sys Tools
from more_itertools import nth

#ROS
import numpy as np
np.float = np.float64  # temp fix for following import https://github.com/eric-wieser/ros_numpy/issues/37
import ros_numpy # Used in sensor_msgs.msg apt-get install ros-noetic-ros-numpy

#Libraries
import cv2
from cv_bridge import CvBridge
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from ouster import client
import matplotlib.pyplot as plt 

from helpers.constants import *
import sensor_msgs

def process_ouster_packet(os1_info, packet_arr, topic, sensor_ts):
    #Process Header
    packets = client.Packets(packet_arr, os1_info)
    scans = client.Scans(packets)
    rg = nth(scans, 0).field(client.ChanField.RANGE)
    rf =  nth(scans, 0).field(client.ChanField.REFLECTIVITY)
    signal =  nth(scans, 0).field(client.ChanField.SIGNAL)
    nr = nth(scans, 0).field(client.ChanField.NEAR_IR)
    ts = nth(scans, 0).timestamp
    
    # Set relative timestamp for each point
    init_ts = ts[0]
    ts_horizontal_rel = ts - init_ts
    ts_horizontal_rel[ts_horizontal_rel<0] = 0
    ts_points = np.tile(ts_horizontal_rel,  (OS1_POINTCLOUD_SHAPE[1], 1) )

    # Set ring to correspond to row idx
    ring_idx = np.arange(0, 128, 1).reshape(-1, 1)
    ring = np.tile(ring_idx, (1, OS1_POINTCLOUD_SHAPE[0]))

    # Project Points to ouster LiDAR Frame
    xyzlut              = client.XYZLut(os1_info)
    xyz_points          = client.destagger(os1_info, xyzlut(rg))

    # Homogeneous xyz coordinates
    homo_xyz    = np.ones((xyz_points.shape[0], xyz_points.shape[1], 1))
    xyz_points  = np.dstack((xyz_points, homo_xyz))

    #Change from LiDAR to sensor coordinate system
    signal      = np.expand_dims(signal, axis=-1)
    rf          = np.expand_dims(rf, axis=-1)
    ts_points   = np.expand_dims(ts_points, axis=-1)
    rg          = np.expand_dims(rg, axis=-1)
    nr          = np.expand_dims(nr, axis=-1)
    ring        = np.expand_dims(ring, axis=-1)

    pc = np.dstack((xyz_points, signal, ts_points, rf, ring, nr, rg)).astype(np.float32)

    return pc, sensor_ts

def set_filename_by_topic(topic, trajectory, frame):
    sensor_subpath  = SENSOR_DIRECTORY_SUBPATH[topic]
    sensor_prefix   = sensor_subpath.replace("/", "_") #get sensor name
    sensor_filetype = SENSOR_DIRECTORY_FILETYPES[sensor_subpath]

    sensor_filename = "%s_%s_%s.%s" % (sensor_prefix, str(trajectory), 
        str(frame), sensor_filetype)

    return sensor_filename

def set_filename_by_prefix(modality, sensor_name, trajectory, frame):
    filetype = SENSOR_DIRECTORY_FILETYPES['/'.join([modality, sensor_name])]
    sensor_filename = "%s_%s_%s_%s.%s" % (
        modality, 
        sensor_name, 
        str(trajectory),
        str(frame),
        filetype
        )
    return sensor_filename

def set_filename_dir(indir, modality, sensor_name, trajectory, frame=None, include_name=False):
    assert (frame is not None and include_name) or (frame is None and not include_name), \
        "Invalid frame and include name argument combination..."
    trajectory = str(trajectory)

    filepath = os.path.join(indir, modality, sensor_name, trajectory)
    if include_name:
        filename = set_filename_by_prefix(modality, sensor_name, trajectory, str(frame))
        return os.path.join(filepath, filename)
    return filepath

def get_filename_info(filename):
    filename_prefix  = filename.split('.')[0]
    filename_prefix  = filename_prefix.split('_')
    
    modality        = filename_prefix[0]+"_"+filename_prefix[1]
    sensor_name     = filename_prefix[2]
    trajectory      = filename_prefix[3]
    frame           = filename_prefix[4]
    return (modality, sensor_name, trajectory, frame)

def get_calibration_info(filepath):
    filename = filepath.split('/')[-1]
    filename_prefix = filename.split('.')[0]
    filename_split = filename_prefix.split('_')

    calibration_info = None
    src, tar = filename_split[1], filename_split[-1]
    if len(filename_split) > 3:
        #Sensor to Sensor transform
        extrinsic = yaml.safe_load(open(filepath, 'r'))
        calibration_info = extrinsic
    else:
        #Intrinsic transform
        intrinsic = yaml.safe_load(open(filepath, 'r'))
        calibration_info = intrinsic
    
    return calibration_info, src, tar

def read_sem_label(label_path):
    assert os.path.exists(label_path), "%s does not exist " % label_path
    sem_tred_np = np.array(list(open(label_path, "rb").read()))
    return sem_tred_np

def read_bin(bin_path, keep_intensity=True):
    num_points = OS1_POINTCLOUD_SHAPE[0]*OS1_POINTCLOUD_SHAPE[1]
    bin_np = np.fromfile(bin_path, dtype=np.float32).reshape(num_points, -1)

    if not keep_intensity:
        bin_np = bin_np[:, :3]
    return bin_np

def pc_to_bin(pc, filename, include_time=True):
    # pc_np = np.array(ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pc))

    pc_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(pc)
    pc_dim = 4
    if include_time:
        pc_dim = 5
    pc_np = np.zeros((pc_cloud.shape[0], pc_cloud.shape[1], pc_dim), dtype=np.float32)
    pc_np[...,0] = pc_cloud['x']
    pc_np[...,1] = pc_cloud['y']
    pc_np[...,2] = pc_cloud['z']
    pc_np[...,3] = pc_cloud['intensity']
    if pc_dim==5:
        pc_np[...,4] = pc_cloud['t']

    pc_np = pc_np.reshape(-1, pc_dim)
    
    flat_pc = pc_np.reshape(-1).astype(np.float32)
    flat_pc.tofile(filename) # empty sep=bytes

def imu_to_txt(imu, filename):
    ts = imu.header.stamp.secs + imu.header.stamp.nsecs * 1e-9
    imu_np = np.array([
        ts, imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z,
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z,
        imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z
    ], dtype=np.float64).reshape(1, -1)

    with open(filename, "a") as imu_file:
        np.savetxt(imu_file, imu_np, fmt='%6.8f', delimiter=" ")
    imu_file.close()

def mag_to_txt(mag, filename):
    ts = mag.header.stamp.secs + mag.header.stamp.nsecs*1e-9
    mag_np = np.array([
        ts, mag.magnetic_field.x, mag.magnetic_field.y, mag.magnetic_field.z
    ], dtype=np.float64).reshape(1, -1)

    with open(filename, "a") as mag_file:
        np.savetxt(mag_file, mag_np, fmt='%6.8f', delimiter=" ")
    mag_file.close()

def gps_to_txt(gps, filename):
    ts = gps.header.stamp.secs + gps.header.stamp.nsecs*1e-9
    gps_np = np.array([
        ts, gps.latitude, gps.longitude, gps.altitude
    ], dtype=np.float64).reshape(1, -1)

    with open(filename, "a") as gps_file:
        np.savetxt(gps_file, gps_np, fmt='%6.8f', delimiter=" ")
    gps_file.close()

def odom_to_txt(odom, filename):
    ts = odom.header.stamp.secs + odom.header.stamp.nsecs*1e-9
    odom_np = np.array([
        ts, 
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z,
        odom.pose.pose.orientation.w, odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
        odom.pose.pose.orientation.z, odom.twist.twist.linear.x, odom.twist.twist.linear.y, 
        odom.twist.twist.linear.z, odom.twist.twist.angular.x, odom.twist.twist.angular.y, 
        odom.twist.twist.angular.z
    ], dtype=np.float64).reshape(1, -1)

    with open(filename, "a") as odom_file:
        np.savetxt(odom_file, odom_np, fmt='%6.8f', delimiter=" ")
    odom_file.close()

def img_to_file(img_np, filename, depth=False):
    if depth: # Convert depth from m to mm before saving
        img_np = img_np * 1000
        img_np = img_np.astype(np.uint16)
    cv2.imwrite(filename, img_np)

def bin_to_ply(bin_np, ply_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(bin_np)
    o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False)
    return bin_np

def pcd_to_np(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    out_arr = np.asarray(pcd.points)
    return out_arr

def copy_image(inpath, outpath):
    outdir = '/'.join(outpath.split('/')[:-1])
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    shutil.copy(inpath, outpath)

def get_ouster_packet_info(os1_info, data):
    return client.LidarPacket(data, os1_info)

def process_image(img_data, encoding="passthrough"):
    sensor_ts = img_data.header.stamp
    cv_image = CvBridge().imgmsg_to_cv2(img_data, desired_encoding=encoding)
    image_np = np.array(cv_image)

    return image_np, sensor_ts

def process_compressed_image(img_data, encoding="bgr8"):
    sensor_ts = img_data.header.stamp
    # Decode mono16 separately due to compressed decoding bug in CvBridge()
    # import pdb; pdb.set_trace()
    if encoding=="mono16":
        compressed_image_np = np.frombuffer(img_data.data, np.uint8)
        image_np = cv2.imdecode(compressed_image_np, -1)
    else:
        cv_image = CvBridge().compressed_imgmsg_to_cv2(img_data, desired_encoding=encoding)
        image_np = np.array(cv_image)

    return image_np, sensor_ts

def rectify_image(img_path, intrinsics):
    img = np.array(cv2.imread(img_path))
    w, h = intrinsics['image_width'], intrinsics['image_height']
    #Need to be float32s to match cv type
    K = np.float32(intrinsics['camera_matrix']['data']).reshape(3,3)
    D = np.float32(intrinsics['distortion_coefficients']['data'])
    
    Kn, roi = cv2.getOptimalNewCameraMatrix(K, D, (w,h), 1, (w,h))
    mapx, mapy = cv2.initUndistortRectifyMap(K,D, None, Kn, (w,h), 5)
    img_rect = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    #Uncomment below to crop rectified image
    # x,y,w,h = roi
    # img_rect = img_rect[y:y+h, x:x+w]
    return img_rect


def process_imu(imu_data, trans):
    orientation = np.array([
        imu_data.orientation.x, imu_data.orientation.y, 
        imu_data.orientation.z, imu_data.orientation.w
    ])
    angular_vel = np.array([
        imu_data.angular_velocity.x, imu_data.angular_velocity.y, 
        imu_data.angular_velocity.z, 0])
    linear_acc  = np.array([
        imu_data.linear_acceleration.x, imu_data.linear_acceleration.y,
        imu_data.linear_acceleration.z, 0
    ])
    orientation = R.from_quat(orientation).as_euler('xyz', degrees=True)
    orientation = np.append(orientation, 0)

    # Transform imu coordinates to sensor coordinate frame
    o_trans     = np.dot(trans, orientation)
    a_trans     = np.dot(trans, angular_vel)
    l_trans     = np.dot(trans, linear_acc)

    o_trans = R.as_quat( R.from_euler('xyz', o_trans[:3], degrees=True) )
    imu_data.orientation.x = o_trans[0]
    imu_data.orientation.y = o_trans[1]
    imu_data.orientation.z = o_trans[2]
    imu_data.orientation.w = o_trans[3]

    imu_data.angular_velocity.x = a_trans[0]
    imu_data.angular_velocity.y = a_trans[1]
    imu_data.angular_velocity.z = a_trans[2]
    imu_data.linear_acceleration.x  = l_trans[0]
    imu_data.linear_acceleration.y  = l_trans[1]
    imu_data.linear_acceleration.z  = l_trans[2]
    return imu_data, imu_data.header.stamp

# def process_vnav_odometry(odom_data, trans):


def process_mag(mag_data):
    pass

def process_gps(gps_data):
    sensor_ts = gps_data.header.stamp

    if gps_data.status==-1:
        return None, sensor_ts
    
    return gps_data, sensor_ts
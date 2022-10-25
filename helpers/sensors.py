import os
import pdb


#Sys Tools
from more_itertools import nth

#ROS
from sensor_msgs.msg import CompressedImage
import ros_numpy # Used in sensor_msgs.msg apt-get install ros-noetic-ros-numpy

#Libraries
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from ouster import client

from helpers.constants import *
import sensor_msgs

def process_ouster_packet(os1_info, packet_arr, topic):
    #Process Header
    packets = client.Packets(packet_arr, os1_info)
    scans = client.Scans(packets)
    scan = nth(scans, 0).field(client.ChanField.RANGE)
    intensity =  nth(scans, 0).field(client.ChanField.REFLECTIVITY)
    ring =  nth(scans, 0).field(client.ChanField.SIGNAL)
    sensor_ts = sum(nth(scans, 0).timestamp) / len(nth(scans, 0).timestamp)
    
    # Project Points to ouster LiDAR Frame
    xyzlut              = client.XYZLut(os1_info)
    xyz_points          = client.destagger(os1_info, xyzlut(scan))

    #Change from LiDAR to sensor coordinate system
    # h, w, d = xyz_points.shape
    # lidar_to_sens = np.array(SENSOR_TO_XYZ_FRAME[topic]).reshape(4, 4)

    # xyz_points  = np.hstack( ( xyz_points.reshape(-1, 3), np.ones((h*w,1)) ) )
    # xyz_points  = np.dot(lidar_to_sens, xyz_points.T).T
    # xyz_points  = xyz_points[:, :3].reshape(h, w, d)
    intensity   = np.expand_dims(intensity, axis=-1)
    ring   = np.expand_dims(ring, axis=-1)

    # TODO figure out how to add ring to publisher
    pc = np.dstack((xyz_points, intensity, ring)).astype(np.float32)
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
        trajectory,
        frame,
        filetype
        )
    return sensor_filename
    
def get_filename_info(filename):
    # pdb.set_trace()
    filename_prefix  = filename.split('.')[0]
    filename_prefix  = filename_prefix.split('_')
    
    modality        = filename_prefix[0]+"_"+filename_prefix[1]
    sensor_name     = filename_prefix[2]
    trajectory      = filename_prefix[3]
    frame           = filename_prefix[4]
    return (modality, sensor_name, trajectory, frame)

def pc_to_bin(pc, filename):
    pc_np = np.array(ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pc))

    flat_pc = pc_np.reshape(-1).astype(np.float32)
    flat_pc.tofile(filename) # empty sep=bytes

def img_to_png(img_np, filename):
    cv2.imwrite(filename, img_np)

def get_ouster_packet_info(os1_info, data):
    return client.LidarPacket(data, os1_info)

def process_compressed_image(img_data):
    sensor_ts = img_data.header.stamp
    np_arr = np.fromstring(img_data.data, np.uint8)
    image_np = np.array(cv2.imdecode(np_arr, cv2.COLOR_BAYER_BG2BGR))

    return image_np, sensor_ts

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
    orientation = R.from_quat(orientation).as_rotvec(degrees=True)
    orientation = np.append(orientation, 0)

    # Transform imu coordinates to LiDAR coordinate frame
    o_trans     = np.dot(trans, orientation)
    a_trans     = np.dot(trans, angular_vel)
    l_trans     = np.dot(trans, linear_acc)

    o_trans = R.as_quat( R.from_rotvec(o_trans[:3]) )
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
    return imu_data

def process_mag(mag_data):
    pass



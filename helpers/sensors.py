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

def process_ouster_packet(os1_info, packet_arr):
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

    #Change Res
    xyz_points  = xyz_points.astype(np.float32)
    intensity   = np.expand_dims(intensity, axis=-1).astype(np.float32)
    ring   = np.expand_dims(ring, axis=-1).astype(np.float32)

    # TODO figure out how to add ring to publisher
    pc = np.dstack((xyz_points, intensity, ring))
    return pc, sensor_ts

def pc_to_bin(pc, save_dir, frame):
    pc_filename = os.path.join(save_dir, str(frame).replace('.', '')+ ".bin")
    pc_np = np.array(ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pc))
    flat_pc = pc_np.reshape(-1)

    flat_pc.tofile(pc_filename) # empty sep=bytes

def img_to_png(img_np, save_dir, frame):
    img_filename = os.path.join(save_dir, str(frame).replace('.', '')+ ".png")

    cv2.imwrite(img_filename, img_np)

def get_ouster_packet_info(os1_info, data):
    return client.LidarPacket(data, os1_info)

def process_compressed_image(img_data):
    sensor_ts = img_data.header.stamp
    np_arr = np.fromstring(img_data.data, np.uint8)
    image_np = np.array(cv2.imdecode(np_arr, cv2.COLOR_BAYER_BG2BGR))

    return image_np, sensor_ts

def process_imu(imu_data, trans):
    trans_np = trans.as_matrix()
    orientation = np.array([
        imu_data.orientation.x, imu_data.orientation.y, 
        imu_data.orientation.z, imu_data.orientation.w
    ])
    angular_vel = np.array([
        imu_data.angular_velocity.x, imu_data.angular_velocity.y, 
        imu_data.angular_velocity.z])
    linear_acc  = np.array([
        imu_data.linear_acceleration.x, imu_data.linear_acceleration.y,
        imu_data.linear_acceleration.z
    ])
    orientation = R.from_quat(orientation).as_rotvec(degrees=True)
    # pdb.set_trace()
    o_trans     = np.dot(trans_np, orientation)
    a_trans     = np.dot(trans_np, angular_vel)
    l_trans     = np.dot(trans_np, linear_acc)
    # pdb.set_trace()
    o_trans = R.as_quat( R.from_rotvec(o_trans) )
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

def wcs_mat(angles):
    """
    assumes angles order is zyx degrees
    """
    r = R.from_euler('zyx', angles, degrees=True)
    return r



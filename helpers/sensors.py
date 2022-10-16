import os
import pdb

#Sys Tools
from more_itertools import nth

#ROS
from sensor_msgs.msg import CompressedImage

#Libraries
import cv2
import numpy as np
from ouster import client

def process_ouster_packet(os1_info, packet_arr):
    #Process Header
    packets = client.Packets(packet_arr, os1_info)
    scans = client.Scans(packets)
    scan = nth(scans, 0).field(client.ChanField.RANGE)
    intensity =  nth(scans, 0).field(client.ChanField.REFLECTIVITY)
    sensor_ts = sum(nth(scans, 0).timestamp) / len(nth(scans, 0).timestamp)
    
    # Project Points to ouster LiDAR Frame
    xyzlut = client.XYZLut(os1_info)
    xyz_points = client.destagger(os1_info, xyzlut(scan))

    #Change Res
    xyz_points  = xyz_points.astype(np.float32)
    intensity   = np.expand_dims(intensity, axis=-1).astype(np.float32)

    pc = np.dstack((xyz_points, intensity))
    return pc, sensor_ts

def pc_to_bin(pc, save_dir, frame):
    pc_filename = os.path.join(save_dir, str(frame).replace('.', '')+ ".bin")
    flat_pc = pc.reshape(-1)

    flat_pc.tofile(pc_filename) # empty sep=bytes

def img_to_png(img_np, save_dir, frame):
    img_filename = os.path.join(save_dir, str(frame).replace('.', '')+ ".png")

    cv2.imwrite(img_filename, img_np)

def get_ouster_packet_info(os1_info, data):
    return client.LidarPacket(data, os1_info)

def process_compressed_image(img_data):
    sensor_ts = img_data.header.stamp
    np_arr = np.fromstring(img_data.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.COLOR_BAYER_BG2BGR)

    return image_np, sensor_ts

def process_imu(imu_data):
    pass

def process_mag(mag_data):
    pass

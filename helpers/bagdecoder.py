import os
import pdb
import queue
from struct import pack
from tabnanny import verbose

# Utility Libraries
import cv2 as cv
import numpy as np
import yaml

# ROS Libraries
import rospy
import ros_numpy # Used in sensor_msgs.msg apt-get install ros-noetic-ros-numpy
import rosbag
import sensor_msgs
import nav_msgs
from visualization_msgs.msg import *
from sensor_msgs.msg import PointCloud2

from helpers.sensors import *
from helpers.visualization import pub_pc_to_rviz
from helpers.constants import OS1_PACKETS_PER_FRAME, DATASET_L1_DIR_LIST, SENSOR_DIRECTORY_SUBPATH

class BagDecoder(object):
    def __init__(
        self, 
        bag_dir):
        
        self.bag_dir = bag_dir
        self.settings_fp = os.path.join(os.getcwd(), "config/settings.yaml")

        assert os.path.isdir(self.bag_dir), '%s does not exist' % self.bag_dir
        assert os.path.isfile(self.settings_fp), '%s does not exist' % self.settings_fp

        #Load available bag files
        print("Loading bags from  ", self.bag_dir)
        print("Loading settings from ", self.settings_fp)

        #Load topics to process
        self.topics = {}
        self.bags_to_process = []
        self.load_settings()

        #Load visualization
        if self.viz_points:
            rospy.init_node('bagdecoder')
            self.pc_pub = rospy.Publisher('/ouster/points', PointCloud2, queue_size=10)
        
        #Generate Dataset
        if self.gen_data:
            self.gen_dataset_structure()

        self.curr_ts        = -1
        self.num_curr_ts    = 0
        self.curr_frame     = 0
        self.last_pc, self.last_cam0, self.last_cam1 = None, None, None

    def gen_dataset_structure(self):
        print("Generating processed dataset subdirectories...")
        for subdir in DATASET_L1_DIR_LIST:
            subdir_path = os.path.join(self.outdir, subdir)
            
            if not os.path.exists(subdir_path):
                if self.verbose:
                    print("Creating directory ", subdir_path)
                os.makedirs(subdir_path)

    def load_settings(self):
        with open(self.settings_fp, 'r') as settings_file:
            settings = yaml.safe_load(settings_file)
            # Load decoder settings
            self.viz_points = settings['viz_points']
            self.gen_data   = settings['gen_data']
            self.verbose    = settings['verbose']
            self.outdir     = settings['dataset_output_root']
            if not os.path.exists(self.outdir):
                os.mkdir(self.outdir)
    
            # Load bag settings
            self.topics = {topic for topic in settings['sensor_topics'] }
            
            if self.verbose:
                print("Saving topics: ", self.topics)

            self.bags_to_process = settings['bags_to_process']
            if len(self.bags_to_process)==0:
                self.bags_to_process = [
                    file for file in sorted(os.listdir(self.bag_dir)) 
                    if os.path.splitext(file)[-1]==".bag"]
    

        if "/ouster/lidar_packets" in self.topics:
            os_metadata = os.path.join(self.bag_dir, "OS1metadata.json")
            assert os.path.isfile(os_metadata), '%s does not exist' % os_metadata
            
            #Load OS1 metadata file
            with open(os_metadata, 'r') as f:
                self.os1_info = client.SensorInfo(f.read())

    def gen_metadata(self, metadata_file):
        """
        Formats metadata file for a given bag file
        """
        None

    def convert_bag(self):
        """
        Decodes requested topics in bag file to individual files
        """
        for trajectory, bag_file in enumerate(self.bags_to_process):
            self.curr_frame = 0
            bag_fp = os.path.join(self.bag_dir, bag_file)
            print("Processing bag ", bag_fp)

            rosbag_info = yaml.safe_load( rosbag.Bag(bag_fp)._get_yaml_info())
            self._topic_to_type = {topic_entry['topic']:topic_entry['type'] \
                for topic_entry in rosbag_info['topics']}

            num_lidar_packets   = 0
            frame_id            = -1
            scan_queue          = []
            for topic, msg, ts in rosbag.Bag(bag_fp).read_messages():
                # pdb.set_trace()
                ts_ms = int(ts.to_sec()*1000)
                if topic in self.topics:
                    topic_type = self._topic_to_type[topic]

                    if topic_type=="ouster_ros/PacketMsg":
                        packet = get_ouster_packet_info(self.os1_info, msg.buf)

                        if packet.frame_id!=frame_id:
                            if num_lidar_packets==OS1_PACKETS_PER_FRAME: # 64 columns per packet
                                pc, _ = self.process_topic(topic_type, scan_queue, ts)
                                self.update_sensor_sync(pc, ts_ms, topic, trajectory)

                                if self.viz_points:
                                    pub_pc_to_rviz(pc, self.pc_pub, ts)

                                if self.verbose:
                                    print("topic: ", topic, " sensor timestamp: ", ts_ms)

                                pdb.set_trace()
                            print("Reset LiDAR packet @ ", num_lidar_packets)
                            # last_lidar_ts = ts
                            num_lidar_packets   = 0
                            frame_id            = packet.frame_id
                            scan_queue          = []

                        num_lidar_packets +=1
                        scan_queue.append(packet)
                    elif topic=="/stereo/left/image_raw/compressed" or \
                        topic=="/stereo/right/image_raw/compressed":
                        
                        if self.verbose:
                            print("topic: ", topic, " sensor timestamp: ", ts_ms)
                        cam, _ = self.process_topic(topic_type, msg, ts)
                        self.update_sensor_sync(cam, ts_ms, topic, trajectory)

    def update_sensor_sync(self, sensor_data, ts_ms, topic, trajectory):
        """
        Images will have the same timestamp but lidar will not
        """
        if  topic=="/stereo/left/image_raw/compressed" or \
            topic=="/stereo/right/image_raw/compressed" or \
            topic=="/ouster/lidar_packets":
            # print("curr ts ", self.curr_ts)
            if ts_ms > self.curr_ts:
                # pdb.set_trace()
                self.curr_ts = ts_ms
                self.num_curr_ts = 0
                self.last_pc, self.last_cam0, self.last_cam1 = None, None, None
            else:
                if topic=="/ouster/lidar_packets":
                    self.last_pc    = sensor_data
                elif topic=="/stereo/left/image_raw/compressed":
                    self.last_cam0  = sensor_data
                else:
                    self.last_cam1  = sensor_data
                self.num_curr_ts += 1
            
            if self.num_curr_ts==3:
                self.save_topic(self.last_pc,   topic, trajectory, self.curr_frame)
                self.save_topic(self.last_cam0, topic, trajectory, self.curr_frame)
                self.save_topic(self.last_cam1, topic, trajectory, self.curr_frame)
                self.curr_frame += 1
                pdb.set_trace()
                self.num_curr_ts = 0
                self.last_pc, self.last_cam0, self.last_cam1 = None, None, None

    def save_topic(self, data, topic_type, trajectory, frame):
        topic_type_subpath = SENSOR_DIRECTORY_SUBPATH[topic_type]
        save_dir = os.path.join(self.outdir, topic_type_subpath, str(trajectory))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if topic_type=="ouster_ros/PacketMsg":
            pc_to_bin(data, save_dir, frame)
        elif topic_type=="sensor_msgs/CompressedImage":
            img_to_png(data, save_dir, frame)
        else:
            pass

    def process_topic(self, type, msg, t):
        data        = None
        sensor_ts   = t
        if type=="ouster_ros/PacketMsg":
            # pdb.set_trace()
            data, sensor_ts = process_ouster_packet(self.os1_info, msg)
        elif type=="sensor_msgs/CompressedImage":
            data, sensor_ts = process_compressed_image(msg)
        elif type=="sensor_msgs/Imu":
            pass
        elif type=="sensor_msgs/MagneticField":
            pass
        return data, sensor_ts

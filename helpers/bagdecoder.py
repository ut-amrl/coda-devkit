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
import rosbag
import message_filters
from visualization_msgs.msg import *
from sensor_msgs.msg import PointCloud2, CompressedImage

from helpers.sensors import *
from helpers.visualization import pub_pc_to_rviz
from helpers.constants import OS1_PACKETS_PER_FRAME, DATASET_L1_DIR_LIST, SENSOR_DIRECTORY_SUBPATH

class BagDecoder(object):
    def __init__(
        self, 
        bag_dir):
        
        self._bag_dir = bag_dir
        self._settings_fp = os.path.join(os.getcwd(), "config/settings.yaml")

        assert os.path.isdir(self._bag_dir), '%s does not exist' % self._bag_dir
        assert os.path.isfile(self._settings_fp), '%s does not exist' % self._settings_fp

        #Load available bag files
        print("Loading bags from  ", self._bag_dir)
        print("Loading settings from ", self._settings_fp)

        #Load topics to process
        self._topics = {}
        self._bags_to_process = []
        self.load_settings()

        #Load visualization
        if self._viz_points:
            rospy.init_node('bagdecoder')
            self.pc_pub = rospy.Publisher('/ouster/points', PointCloud2, queue_size=10)
        
        #Generate Dataset
        if self._gen_data:
            self.gen_dataset_structure()

        self._trajectory = 0

        #Load sync publisher
        rospy.init_node('bagdecoder', anonymous=True)
        self._topic_to_type = None

    def gen_dataset_structure(self):
        print("Generating processed dataset subdirectories...")
        for subdir in DATASET_L1_DIR_LIST:
            subdir_path = os.path.join(self._outdir, subdir)
            
            if not os.path.exists(subdir_path):
                if self._verbose:
                    print("Creating directory ", subdir_path)
                os.makedirs(subdir_path)

    def load_settings(self):
        with open(self._settings_fp, 'r') as settings_file:
            settings = yaml.safe_load(settings_file)
            # Load decoder settings
            self._viz_points = settings['viz_points']
            self._gen_data   = settings['gen_data']
            self._verbose    = settings['verbose']
            self._outdir     = settings['dataset_output_root']
            if not os.path.exists(self._outdir):
                os.mkdir(self._outdir)
    
            # Load bag settings
            self._topics         = settings['sensor_topics']
            self._sync_topics   = settings['sync_topics']
            
            if self._verbose:
                print("Saving topics: ", self._topics)

            self.bags_to_process = settings['bags_to_process']
            if len(self.bags_to_process)==0:
                self.bags_to_process = [
                    file for file in sorted(os.listdir(self._bag_dir)) 
                    if os.path.splitext(file)[-1]==".bag"]
    

        if "/ouster/lidar_packets" in self._topics:
            os_metadata = os.path.join(self._bag_dir, "OS1metadata.json")
            assert os.path.isfile(os_metadata), '%s does not exist' % os_metadata

            #Load OS1 metadata file
            with open(os_metadata, 'r') as f:
                self._os1_info = client.SensorInfo(f.read())

    def gen_metadata(self, metadata_file):
        """
        Formats metadata file for a given bag file
        """
        None

    def sync_callback(self, *argv):
        if len(argv) > len(self._sync_topics):
            print("Unequal number of sync sensors instantiated in callback, exiting...")
            exit(1)

        frame = self._curr_frame
        for index, sensor in enumerate(argv):
            self.save_topic(sensor, self._sync_topics[index], self._trajectory, frame)
        self._curr_frame += 1

    def setup_sync_filter(self):
        if self._topic_to_type==None:
            print("Error: Topic to type mappings not defined yet. Exiting...")
            exit(1)

        self._sync_subs = []
        self._sync_pubs = {}
        for topic in self._sync_topics:
            topic_class = None
            if self._topic_to_type[topic]=="sensor_msgs/CompressedImage":
                topic_class = CompressedImage
            elif self._topic_to_type[topic]=="ouster_ros/PacketMsg":
                topic_class = PointCloud2 #Post Process PacketMsg to Pointcloud2 later
            else:
                print("Undefined topic %s for sync filter, skipping..." % topic)

            if topic_class!=None:
                self._sync_subs.append(
                    message_filters.Subscriber(topic, topic_class)
                )
                self._sync_pubs[topic] = rospy.Publisher(
                    topic, topic_class, queue_size=10
                )

        ts = message_filters.ApproximateTimeSynchronizer(self._sync_subs, 10, 0.05, allow_headerless=True)
        ts.registerCallback(self.sync_callback)

    def convert_bag(self):
        """
        Decodes requested topics in bag file to individual files
        """
        for trajectory, bag_file in enumerate(self.bags_to_process):
            self._trajectory = trajectory
            self._curr_frame = 0
            bag_fp = os.path.join(self._bag_dir, bag_file)
            print("Processing bag ", bag_fp)

            rosbag_info = yaml.safe_load( rosbag.Bag(bag_fp)._get_yaml_info())
            self._topic_to_type = {topic_entry['topic']:topic_entry['type'] \
                for topic_entry in rosbag_info['topics']}
            self.setup_sync_filter()

            num_lidar_packets = 0
            frame_id = -1
            scan_queue = []
            for topic, msg, ts in rosbag.Bag(bag_fp).read_messages():
                if topic in self._sync_topics:
                    topic_type = self._topic_to_type[topic]
                    if topic_type=="ouster_ros/PacketMsg":
                        packet = get_ouster_packet_info(self._os1_info, msg.buf)

                        if packet.frame_id!=frame_id:
                            if num_lidar_packets==OS1_PACKETS_PER_FRAME: # 64 columns per packet
                                pc, _ = self.process_topic(topic_type, scan_queue, ts)
                                pub_pc_to_rviz(pc, self._sync_pubs[topic], ts)
                            num_lidar_packets   = 0
                            frame_id            = packet.frame_id
                            scan_queue          = []

                        num_lidar_packets +=1
                        scan_queue.append(packet)
                    else:
                        self._sync_pubs[topic].publish(msg)
                else:
                    pass

    def save_topic(self, data, topic, trajectory, frame):
        topic_type      = self._topic_to_type[topic]

        topic_type_subpath = SENSOR_DIRECTORY_SUBPATH[topic]
        save_dir = os.path.join(self._outdir, topic_type_subpath, str(trajectory))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if self._verbose:
            print("Saving sync topic ", topic_type, " with timestamp ", 
                data.header.stamp, " at frame ", frame)
        if topic_type=="ouster_ros/PacketMsg":
            pc_to_bin(data, save_dir, frame)
        elif topic_type=="sensor_msgs/CompressedImage":
            img, _ = self.process_topic(topic_type, data, data.header.stamp)
            img_to_png(img, save_dir, frame)

        else:
            print("Entered undefined topic to be saved...")
            pass

    def process_topic(self, type, msg, t):
        data        = None
        sensor_ts   = t
        if type=="ouster_ros/PacketMsg":
            # pdb.set_trace()
            data, sensor_ts = process_ouster_packet(self._os1_info, msg)
        elif type=="sensor_msgs/CompressedImage":
            data, sensor_ts = process_compressed_image(msg)
        elif type=="sensor_msgs/Imu":
            pass
        elif type=="sensor_msgs/MagneticField":
            pass
        return data, sensor_ts

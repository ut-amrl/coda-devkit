import os
import pdb

# Utility Libraries
import yaml
import time

# ROS Libraries
import rospy
import rosbag
import message_filters
from visualization_msgs.msg import *
from sensor_msgs.msg import PointCloud2, CompressedImage, Imu
from rosgraph_msgs.msg import Clock

from helpers.sensors import *
from helpers.visualization import pub_pc_to_rviz
from helpers.constants import *

class BagDecoder(object):
    """
    Decodes directory of bag files into dataset structure defined in README.md
    
    This decoder requires the bag files to contain /ouster/lidar_packets and
    two compressed image topics to be published within 50 milliseconds of each other. It
    also requires that a ros master be running as well to synchronize these three topics. 
    """
    def __init__(self):
        self._settings_fp = os.path.join(os.getcwd(), "config/decoder.yaml")
        assert os.path.isfile(self._settings_fp), '%s does not exist' % self._settings_fp

        #Load available bag files
        print("Loading settings from ", self._settings_fp)

        #Load topics to process
        self._topics = {}
        self._bags_to_process = []
        self.load_settings()

        #Load visualization
        if self._viz_imu:
            self._imu_pub = rospy.Publisher('/vectornav/IMU', Imu, queue_size=10)
        
        #Generate Dataset
        if self._gen_data:
            self.gen_dataset_structure()

        self._trajectory = 0

        #Load sync publisher
        rospy.init_node('bagdecoder', anonymous=True)
        self._topic_to_type = None

        self._qp_counter = 0
        self._qp_frame_id= -1
        self._qp_scan_queue = []

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

            #Load bag file directory
            self._repository_root   = settings["repository_root"]
            self._bag_date          = settings["bag_date"]
            self._bag_dir           = os.path.join(self._repository_root, self._bag_date)
            assert os.path.isdir(self._bag_dir), '%s does not exist' % self._bag_dir
            print("Loading bags from  ", self._bag_dir)


            # Load decoder settings
            self._gen_data  = settings['gen_data']
            self._viz_imu   = settings['viz_imu']
            self._verbose   = settings['verbose']
            self._outdir    = settings['dataset_output_root']
            if not os.path.exists(self._outdir):
                os.mkdir(self._outdir)
    
            # Load bag settings
            self._topics         = settings['sensor_topics']
            self._sync_topics   = settings['sync_topics']
            
            if self._verbose:
                print("Saving topics: ", self._topics)

            self._bags_to_process = settings['bags_to_process']
            if len(self._bags_to_process)==0:
                self._bags_to_process = [
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

        for trajectory, bag_file in enumerate(self._bags_to_process):
            self._trajectory = trajectory
            self._curr_frame = 0
            bag_fp = os.path.join(self._bag_dir, bag_file)
            print("Processing bag ", bag_fp)

            #Preprocess topics
            rosbag_info = yaml.safe_load( rosbag.Bag(bag_fp)._get_yaml_info())
            self._topic_to_type = {topic_entry['topic']:topic_entry['type'] \
                for topic_entry in rosbag_info['topics']}
            self.setup_sync_filter()

            #Create frame to timestamp map
            frame_to_ts_path= os.path.join(self._outdir, "timestamps", "%i_frame_to_ts.txt"%trajectory)
            frame_to_ts     = open(frame_to_ts_path, 'w+')
            if self._verbose:
                print("Writing frame to timestamp map to %s\n" % frame_to_ts_path)

            for topic, msg, ts in rosbag.Bag(bag_fp).read_messages():
                if topic in self._sync_topics:
                    topic_type = self._topic_to_type[topic]
                    if topic_type=="ouster_ros/PacketMsg":
                        did_publish = self.qpacket(topic, msg, ts)
                        
                        if did_publish:
                            frame_to_ts.write("%10.6f\n" % ts.to_sec())
                    else:
                        self._sync_pubs[topic].publish(msg)
                else:
                    data, ts = self.process_topic(topic, msg, ts)
            
            frame_to_ts.close()
            print("Completed processing bag ", bag_fp)
            pdb.set_trace()

    def qpacket(self, topic, msg, ts):
        published_packet = False
        packet = get_ouster_packet_info(self._os1_info, msg.buf)

        if packet.frame_id!=self._qp_frame_id:
            if self._qp_counter==OS1_PACKETS_PER_FRAME: # 64 columns per packet
                pc, _ = self.process_topic(topic, self._qp_scan_queue, ts)
                pub_pc_to_rviz(pc, self._sync_pubs[topic], ts)
                published_packet = True
                
            self._qp_counter     = 0

            self._qp_frame_id    = packet.frame_id
            self._qp_scan_queue  = []

        self._qp_counter +=1
        self._qp_scan_queue.append(packet)
        return published_packet


    def save_topic(self, data, topic, trajectory, frame):
        if not self._gen_data:
            return

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

    def process_topic(self, topic, msg, t):
        topic_type = self._topic_to_type[topic]
        data        = None
        sensor_ts   = t
        if topic_type=="ouster_ros/PacketMsg":
            data, sensor_ts = process_ouster_packet(self._os1_info, msg)
        elif topic_type=="sensor_msgs/CompressedImage":
            data, sensor_ts = process_compressed_image(msg)
        elif topic_type=="sensor_msgs/Imu":
            imu_to_wcs = wcs_mat(SENSOR_TO_WCS[topic])
            data = process_imu(msg, imu_to_wcs)
            if self._viz_imu: 
                self._imu_pub.publish(data) 
        elif topic_type=="sensor_msgs/MagneticField":
            pass
        return data, sensor_ts

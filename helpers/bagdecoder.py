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
from helpers.geometry import wcs_mat
from helpers.visualization import pub_pc_to_rviz
from helpers.constants import *

class BagDecoder(object):
    """
    Decodes directory of bag files into dataset structure defined in README.md
    
    This decoder requires the bag files to contain /ouster/lidar_packets and
    two compressed image topics to be published within 50 milliseconds of each other. It
    also requires that a ros master be running as well to synchronize these three topics. 
    """
    def __init__(self, config):
        self._settings_fp = os.path.join(os.getcwd(), config)
        assert os.path.isfile(self._settings_fp), '%s does not exist' % self._settings_fp

        #Load available bag files
        print("Loading settings from ", self._settings_fp)

        #Load topics to process
        self._sensor_topics = {}
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

        self._past_sync_ts = None

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
            self._bags_to_traj_ids  = settings['bags_to_traj_ids']
            if not os.path.exists(self._outdir):
                os.mkdir(self._outdir)
    
            # Load bag settings
            self._sensor_topics = settings['sensor_topics']
            self._sync_topics   = settings['sync_topics']
            
            if self._verbose:
                print("Saving topics: ", self._sensor_topics)

            self._bags_to_process   = settings['bags_to_process']
            self._all_bags          = [ file for file in sorted(os.listdir(self._bag_dir)) 
                if os.path.splitext(file)[-1]==".bag"]
            if len(self._bags_to_process)==0:
                self._bags_to_process = self._all_bags
    
        if "/ouster/lidar_packets" in self._sensor_topics:
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

    def setup_sync_filter(self):
        if self._topic_to_type==None:
            print("Error: Topic to type mappings not defined yet. Exiting...")
            exit(1)

        self._sync_pubs = {}
        self._sync_msg_queue = {}
        for topic in self._sync_topics:
            topic_class = None
            if self._topic_to_type[topic]=="sensor_msgs/CompressedImage":
                topic_class = CompressedImage
            elif self._topic_to_type[topic]=="ouster_ros/PacketMsg":
                topic_class = PointCloud2 #Post Process PacketMsg to Pointcloud2 later
            else:
                print("Undefined topic %s for sync filter, skipping..." % topic)

            if topic_class!=None:
                self._sync_pubs[topic] = rospy.Publisher(
                    topic, topic_class, queue_size=10
                )
                self._sync_msg_queue[topic] = []

    def convert_bag(self):
        """
        Decodes requested topics in bag file to individual files
        """
        for trajectory_idx, bag_file in enumerate(self._all_bags):
            if bag_file not in self._bags_to_process:
                continue

            self._trajectory = trajectory_idx
            if len(self._bags_to_traj_ids)==len(self._bags_to_process):
                bag_idx = self._bags_to_process.index(bag_file)
                self._trajectory = self._bags_to_traj_ids[bag_idx]

            self._curr_frame = 0
            bag_fp = os.path.join(self._bag_dir, bag_file)
            print("Processing bag ", bag_fp, " as trajectory", self._trajectory)

            #Preprocess topics
            rosbag_info = yaml.safe_load( rosbag.Bag(bag_fp)._get_yaml_info())
            self._topic_to_type = {topic_entry['topic']:topic_entry['type'] \
                for topic_entry in rosbag_info['topics']}
            self.setup_sync_filter()

            #Create frame to timestamp map
            frame_to_ts_path= os.path.join(self._outdir, "timestamps", "%i_frame_to_ts.txt"%self._trajectory)
            if self._gen_data:
                self._frame_to_ts     = open(frame_to_ts_path, 'w+')
            if self._verbose:
                print("Writing frame to timestamp map to %s\n" % frame_to_ts_path)

            for topic, msg, ts in rosbag.Bag(bag_fp).read_messages():
                if topic in self._sync_topics:
                    topic_type = self._topic_to_type[topic]
                    if topic_type=="ouster_ros/PacketMsg":
                        self.qpacket(topic, msg, ts)
                    else:
                        self.sync_sensor(topic, msg, ts)
                        self._sync_pubs[topic].publish(msg)
                else:
                    if topic in self._sensor_topics:
                        data, ts = self.process_topic(topic, msg, ts)
            
            print("Completed processing bag ", bag_fp)
            if self._gen_data:
                self._frame_to_ts.close()

    def sync_sensor(self, topic, msg, ts):
        if self._past_sync_ts==None:
            self._past_sync_ts = ts
            self._past_sync_ts.secs -= 1 # Set last sync time to be less than first msg

        # print("call by topic %s with timestamp %10.6f" % (topic, ts.to_sec()))
        #Remove all timestamps earlier than last synced timestamp
        for topic_key in self._sync_msg_queue.keys():
            while len(self._sync_msg_queue[topic_key]) > 0 and \
                self._sync_msg_queue[topic_key][0].header.stamp < self._past_sync_ts:
                self._sync_msg_queue[topic_key].pop(0)

        #Insert new message into topic queue
        self._sync_msg_queue[topic].append(msg)

        #After each queue contains at least one msg, choose earliest message among queues
        while( not self.is_sync_queue_empty() ):
            #If difference between earliest message is too much larger than earliest
            # message in other queues, discard message and advance forward
            # print("Entering loop")
            latest_topic, latest_ts = self.get_latest_queue_msg(ts)
            
            self.remove_old_topics(latest_ts)
            
            if ( not self.is_sync_queue_empty() ):
                #If difference between earliest message and others is acceptable
                #save all messages in queue, discard them, and continue looping
                self._past_sync_ts = self.save_sync_topics()
    
    def save_sync_topics(self):
        earliest_sync_timestamp = None
        latest_sync_timestamp = self._past_sync_ts
        frame = self._curr_frame
        for topic, msgs in self._sync_msg_queue.items():
            self.save_topic(msgs[0], topic, self._trajectory, frame)
            
            # Store last sync timestamp
            ts = msgs[0].header.stamp
            if ts > latest_sync_timestamp:
                latest_sync_timestamp = ts
            msgs.pop(0)

            #Write earliest timestamp to file as this was the trigger sensor
            if earliest_sync_timestamp==None:
                earliest_sync_timestamp = ts
            elif ts < earliest_sync_timestamp:
                earliest_sync_timestamp = ts

        #Perform state sync updates
        self.save_frame_ts(earliest_sync_timestamp)
        self._curr_frame += 1
        return latest_sync_timestamp

    def save_frame_ts(self, timestamp):
        if self._gen_data:
            # if self._verbose:
            print("Saved frame %i timestamp %10.6f" % (self._curr_frame, timestamp.to_sec()))
            self._frame_to_ts.write("%10.6f\n" % timestamp.to_sec())

    def remove_old_topics(self, latest_ts):
        latest_ts = latest_ts.to_sec()
        for topic in self._sync_msg_queue.keys():
            delete_indices = []
            for idx, msg in enumerate(self._sync_msg_queue[topic]):
                curr_ts = msg.header.stamp.to_sec()
                if abs(curr_ts - latest_ts)>=0.1:
                    delete_indices.append(idx)
                    break
            self._sync_msg_queue[topic] = [msg for idx, msg in \
                enumerate(self._sync_msg_queue[topic]) if idx not in delete_indices]
            
    def get_latest_queue_msg(self, last_ts):
        earliest_ts     = last_ts
        earliest_topic  = None 

        for topic in self._sync_msg_queue.keys():
            msg_ts = self._sync_msg_queue[topic][0].header.stamp
            if msg_ts >= earliest_ts:
                earliest_ts = msg_ts
                earliest_topic = topic

        return earliest_topic, earliest_ts

    def is_sync_queue_empty(self):
        """
        If any queue is empty return false
        """
        are_queues_empty = False
        for topic, msgs in self._sync_msg_queue.items():
            are_queues_empty = are_queues_empty or len(msgs)==0
        return are_queues_empty

    def qpacket(self, topic, msg, ts):
        published_packet = False
        packet = get_ouster_packet_info(self._os1_info, msg.buf)

        if packet.frame_id!=self._qp_frame_id:
            if self._qp_counter==OS1_PACKETS_PER_FRAME: # 64 columns per packet
                pc, _ = self.process_topic(topic, self._qp_scan_queue, ts)
                pc_msg = pub_pc_to_rviz(pc, self._sync_pubs[topic], ts)

                self.sync_sensor(topic, pc_msg, ts)
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

        filename = set_filename_by_topic(topic, trajectory, frame)
        filepath = os.path.join(save_dir, filename)
        if self._verbose:
            print("Saving sync topic ", topic_type, " with timestamp ", 
                data.header.stamp, " at frame ", frame)
        if topic_type=="ouster_ros/PacketMsg":
            #Expects PointCloud2 Object
            pc_to_bin(data, filepath)
        elif topic_type=="sensor_msgs/CompressedImage":
            img, _ = self.process_topic(topic, data, data.header.stamp)
            img_to_png(img, filepath)
        else:
            print("Entered undefined topic to be saved...")
            pass

    def process_topic(self, topic, msg, t):
        if not topic in self._topic_to_type:
            print("Could not find topic %s in topic map for time %i, exiting..." % (topic, t.to_sec()))
            return

        topic_type = self._topic_to_type[topic]
        data        = None
        sensor_ts   = t
        if topic_type=="ouster_ros/PacketMsg":
            data, sensor_ts = process_ouster_packet(self._os1_info, msg, topic)
        elif topic_type=="sensor_msgs/CompressedImage":
            data, sensor_ts = process_compressed_image(msg)
        elif topic_type=="sensor_msgs/Imu":
            imu_to_wcs = np.array(SENSOR_TO_XYZ_FRAME[topic]).reshape(4, 4)
            data = process_imu(msg, imu_to_wcs)
            if self._viz_imu: 
                self._imu_pub.publish(data) 
        elif topic_type=="sensor_msgs/MagneticField":
            pass
        return data, sensor_ts

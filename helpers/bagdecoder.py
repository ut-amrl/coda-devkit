import os
import pdb

# Utility Libraries
import yaml

# ROS Libraries
import rospy
import rosbag
from visualization_msgs.msg import *
from sensor_msgs.msg import PointCloud2

from helpers.sensors import *
from helpers.visualization import pub_pc_to_rviz
from helpers.constants import *


class BagDecoder(object):
    def __init__(self, config):
        self._settings_fp = os.path.join(os.getcwd(), config)
        assert os.path.isfile(self._settings_fp), '%s does not exist' % self._settings_fp

        # Load available bag files
        print("Loading settings from ", self._settings_fp)

        # Load topics to process
        self._sensor_topics = {}
        self._bags_to_process = None
        self.load_settings()

        # Generate Dataset
        if self._gen_data:
            self.gen_dataset_structure()

        self._trajectory = 0

        # Load sync publisher
        rospy.init_node('coda', anonymous=True)
        self._pub_rate = rospy.Rate(10)  # Publish at 10 hz
        self._topic_to_type = None

        self._qp_counter = 0
        self._qp_frame_id = -1
        self._qp_scan_queue = []

        self._past_sync_ts = None

    def gen_dataset_structure(self):
        print("Generating processed dataset subdirectories...")
        subdir_path = os.path.join(self._outdir, "3d_raw")

        if not os.path.exists(subdir_path):
            print("Creating directory ", subdir_path)
            os.makedirs(subdir_path)

    def load_settings(self):
        settings = yaml.safe_load(open(self._settings_fp, 'r'))

        # Load bag file directory
        self._repository_root = settings["repository_root"]
        self._bag_date = settings["bag_date"]
        self._bag_dir = os.path.join(self._repository_root, self._bag_date)
        assert os.path.isdir(self._bag_dir), '%s does not exist' % self._bag_dir
        print("Loading bags from  ", self._bag_dir)

        # Load decoder settings
        self._gen_data = settings['gen_data']
        self._outdir = settings['dataset_output_root']
        self._bags_to_traj_ids = 42
        if not os.path.exists(self._outdir):
            os.mkdir(self._outdir)

        # Load bag settings
        self._sensor_topics = settings['sensor_topics']
        self._sync_topics = settings['sync_topics']
        self._sync_method = settings["sync_method"] if "sync_method" in settings else None

        self._bags_to_process = settings['bags_to_process']

        # Load LiDAR decoder settings
        if "/ouster/lidar_packets" in self._sensor_topics:
            os_metadata = os.path.join(self._bag_dir, "OS1metadata.json")

            if not os.path.exists(os_metadata):
                default_os = os.path.join("helpers/helper_utils/OS1metadata.json")
                print("Ouster metadata not found at %s, using default at %s" %
                      (os_metadata, default_os))
                os_metadata = default_os
            assert os.path.isfile(os_metadata), '%s does not exist' % os_metadata

            # Load OS1 metadata file
            with open(os_metadata, 'r') as f:
                self._os1_info = client.SensorInfo(f.read())

    def setup_sync_filter(self):
        if self._topic_to_type == None:
            print("Error: Topic to type mappings not defined yet. Exiting...")
            exit(1)

        self._sync_msg_queue = {}
        for topic in self._sync_topics:
            topic_class = None
            if self._topic_to_type[topic] == "ouster_ros/PacketMsg":
                topic_class = PointCloud2  # Post Process PacketMsg to Pointcloud2 later
            elif self._topic_to_type[topic] == "sensor_msgs/PointCloud2":
                topic_class = PointCloud2
            else:
                print("Undefined topic %s for sync filter, skipping..." % topic)

            if topic_class != None:
                self._sync_msg_queue[topic] = []

    def convert_bag(self):
        """
        Decodes requested topics in bag file to individual files
        """
        self._trajectory = 84

        self._curr_frame = 0
        bag_fp = os.path.join(self._bag_dir, self._bags_to_process)
        print("Processing bag ", bag_fp, " as trajectory", self._trajectory)

        # Preprocess topics
        rosbag_info = yaml.safe_load(rosbag.Bag(bag_fp)._get_yaml_info())

        self._topic_to_type = {topic_entry['topic']: topic_entry['type'] for topic_entry in rosbag_info['topics']}
        self.setup_sync_filter()

        for topic, msg, ts in rosbag.Bag(bag_fp).read_messages():
            if topic in self._sync_topics:
                topic_type = self._topic_to_type[topic]
                if topic_type == "ouster_ros/PacketMsg":
                    self.qpacket(topic, msg, ts)
                else:
                    self.sync_sensor(topic, msg, ts)
            elif topic in self._sensor_topics:
                topic_type = self._topic_to_type[topic]

                if topic_type == "ouster_ros/PacketMsg":
                    msg = self.qpacket(topic, msg, ts, do_sync=False)
                else:
                    # Use ts as frame for non synchronized sensors
                    filepath = self.save_topic(msg, topic, self._trajectory, ts)
        print("Completed processing bag ", bag_fp)

    def sync_sensor(self, topic, msg, ts):
        if self._past_sync_ts == None:
            self._past_sync_ts = ts
            self._past_sync_ts.secs -= 1  # Set last sync time to be less than first msg

        if self._sync_method == "FIFO":
            # Insert new message to end of queue
            self._sync_msg_queue[topic].append(msg)

            if (not self.is_sync_queue_empty()):
                self._past_sync_ts = self.save_sync_topics()
        else:
            # Remove all timestamps earlier than last synced timestamp
            for topic_key in self._sync_msg_queue.keys():
                while len(self._sync_msg_queue[topic_key]) > 0 and \
                        self._sync_msg_queue[topic_key][0].header.stamp < self._past_sync_ts:
                    self._sync_msg_queue[topic_key].pop(0)

            # Insert new message into topic queue
            self._sync_msg_queue[topic].append(msg)

            # After each queue contains at least one msg, choose earliest message among queues
            while (not self.is_sync_queue_empty()):
                # If difference between earliest message is too much larger than earliest
                # message in other queues, discard message and advance forward
                latest_topic, latest_ts = self.get_latest_queue_msg(ts)
                self.remove_old_topics(latest_ts)

                if (not self.is_sync_queue_empty()):
                    # If difference between earliest message and others is acceptable
                    # save all messages in queue, discard them, and continue looping
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

            # Write earliest timestamp to file as this was the trigger sensor
            if earliest_sync_timestamp == None:
                earliest_sync_timestamp = ts
            elif ts < earliest_sync_timestamp:
                earliest_sync_timestamp = ts

        # Perform state sync updates
        self._curr_frame += 1

        return latest_sync_timestamp

    def remove_old_topics(self, latest_ts):
        latest_ts = latest_ts.to_sec()
        for topic in self._sync_msg_queue.keys():
            delete_indices = []
            for idx, msg in enumerate(self._sync_msg_queue[topic]):
                curr_ts = msg.header.stamp.to_sec()
                if abs(curr_ts - latest_ts) >= 0.1:
                    print("Found difference of ", abs(curr_ts - latest_ts), " for topic ", topic)
                    delete_indices.append(idx)
                    break
            self._sync_msg_queue[topic] = [msg for idx, msg in enumerate(self._sync_msg_queue[topic]) if idx not in delete_indices]

    def get_latest_queue_msg(self, last_ts):
        earliest_ts = last_ts
        earliest_topic = None

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
            are_queues_empty = are_queues_empty or len(msgs) == 0
        return are_queues_empty

    def qpacket(self, topic, msg, ts, do_sync=True):
        published_packet = False
        packet = get_ouster_packet_info(self._os1_info, msg.buf)

        if packet.frame_id != self._qp_frame_id:
            if self._qp_counter == OS1_PACKETS_PER_FRAME:  # 64 columns per packet
                pc, _ = self.process_topic(topic, self._qp_scan_queue, ts)

                if do_sync:
                    pass
                else:
                    self._curr_frame += 1

                pc_msg = pub_pc_to_rviz(pc, None, ts, publish=False)

                if do_sync:
                    self.sync_sensor(topic, pc_msg, ts)
                published_packet = True

            self._qp_counter = 0

            self._qp_frame_id = packet.frame_id
            self._qp_scan_queue = []

        self._qp_counter += 1
        self._qp_scan_queue.append(packet)
        return published_packet

    def save_topic(self, data, topic, trajectory, frame):
        # Generate path of topic
        topic_type = self._topic_to_type[topic]

        topic_type_subpath = SENSOR_DIRECTORY_SUBPATH[topic]

        save_dir = os.path.join(self._outdir, topic_type_subpath, str(trajectory))
        filename = set_filename_by_topic(topic, trajectory, frame)
        filepath = os.path.join(save_dir, filename)

        if not self._gen_data:
            return filepath

        if not os.path.exists(save_dir):
            print("Creating %s because it does not exist..." % save_dir)
            os.makedirs(save_dir)

        if topic_type == "ouster_ros/PacketMsg" or topic_type == "sensor_msgs/PointCloud2":
            # Expects PointCloud2 Object
            pc_to_bin(data, filepath)
        return filepath

    def process_topic(self, topic, msg, t):
        if not topic in self._topic_to_type:
            print("Could not find topic %s in topic map for time %i, exiting..." % (topic, t.to_sec()))
            return

        topic_type = self._topic_to_type[topic]
        data = None
        sensor_ts = t
        if topic_type == "ouster_ros/PacketMsg":
            data, sensor_ts = process_ouster_packet(self._os1_info, msg, topic, sensor_ts)
        elif topic_type == "sensor_msgs/PointCloud2":
            data, sensor_ts = process_pc2(msg, sensor_ts)
        return data, sensor_ts

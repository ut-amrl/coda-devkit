import os
import pdb
import yaml
import numpy as np
from more_itertools import nth
from ouster import client

# ROS
import rospy
import rosbag
import ros_numpy  # Used in sensor_msgs.msg apt-get install ros-noetic-ros-numpy
import std_msgs.msg
from sensor_msgs.msg import PointCloud2, PointField


class BagDecoder(object):
    OS1_PACKETS_PER_FRAME = 64
    OS1_POINTCLOUD_SHAPE = [1024, 128, 3]
    TRED_RAW_DIR = "3d_raw"
    SENSOR_DIRECTORY_SUBPATH = {
        # Depth
        "/ouster/lidar_packets": "%s/os1" % TRED_RAW_DIR,
    }

    SENSOR_DIRECTORY_FILETYPES = {
        "%s/os1" % TRED_RAW_DIR: "bin",
    }

    def __init__(self, config):
        self._settings_fp = os.path.join(os.getcwd(), config)
        assert os.path.isfile(self._settings_fp), '%s does not exist' % self._settings_fp
        print("Loading settings from ", self._settings_fp)
        self._sensor_topics = {}
        self._bag_to_process = None
        self.load_settings()
        if self._gen_data:
            self.gen_dataset_structure()

        self._topic_to_type = None
        self._qp_counter = 0
        self._qp_frame_id = -1
        self._qp_scan_queue = []
        self._past_sync_ts = None

    def gen_dataset_structure(self):
        print("Generating processed dataset subdirectories...")
        subdir_path = os.path.join(self._outdir, "3d_raw")
        print("Creating directory ", subdir_path)
        os.makedirs(subdir_path, exist_ok=True)

    def load_settings(self):
        settings = yaml.safe_load(open(self._settings_fp, 'r'))

        # Load bag file directory
        self._repository_root = settings["repository_root"]
        self._bag_date = settings["bag_date"]
        self._bag_dir = os.path.join(self._repository_root, self._bag_date)
        assert os.path.isdir(self._bag_dir), '%s does not exist' % self._bag_dir

        # Load decoder settings
        self._gen_data = settings['gen_data']
        self._outdir = settings['dataset_output_root']
        self._bags_to_traj_ids = 42
        if not os.path.exists(self._outdir):
            os.mkdir(self._outdir)

        # Load bag settings
        self._sensor_topics = settings['sensor_topics']
        self._sync_topics = settings['sync_topics']
        self._bag_to_process = settings['bags_to_process']

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
        self._curr_frame = 0
        bag_fp = os.path.join(self._bag_dir, self._bag_to_process)
        print("Processing bag ", bag_fp)
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
                    filepath = self.save_topic(msg, topic, 0, ts)
        print("Completed processing bag ", bag_fp)

    def qpacket(self, topic, msg, ts, do_sync=True):
        published_packet = False
        packet = BagDecoder.get_ouster_packet_info(self._os1_info, msg.buf)
        if packet.frame_id != self._qp_frame_id:
            if self._qp_counter == BagDecoder.OS1_PACKETS_PER_FRAME:  # 64 columns per packet
                pc, _ = self.process_topic(topic, self._qp_scan_queue, ts)
                if do_sync:
                    pass
                else:
                    self._curr_frame += 1
                pc_msg = BagDecoder.pub_pc_to_rviz(pc, None, ts, publish=False)
                if do_sync:
                    self.sync_sensor(topic, pc_msg, ts)
                published_packet = True
            self._qp_counter = 0
            self._qp_frame_id = packet.frame_id
            self._qp_scan_queue = []
        self._qp_counter += 1
        self._qp_scan_queue.append(packet)
        return published_packet

    def sync_sensor(self, topic, msg, ts):
        if self._past_sync_ts == None:
            self._past_sync_ts = ts
            self._past_sync_ts.secs -= 1  # Set last sync time to be less than first msg

        for topic_key in self._sync_msg_queue.keys():
            while len(self._sync_msg_queue[topic_key]) > 0 and \
                    self._sync_msg_queue[topic_key][0].header.stamp < self._past_sync_ts:
                self._sync_msg_queue[topic_key].pop(0)
        self._sync_msg_queue[topic].append(msg)
        while (not self.is_sync_queue_empty()):
            latest_topic, latest_ts = self.get_latest_queue_msg(ts)
            self.remove_old_topics(latest_ts)
            if (not self.is_sync_queue_empty()):
                self._past_sync_ts = self.save_sync_topics()

    def save_sync_topics(self):
        earliest_sync_timestamp = None
        latest_sync_timestamp = self._past_sync_ts
        frame = self._curr_frame
        for topic, msgs in self._sync_msg_queue.items():
            self.save_topic(msgs[0], topic, 0, frame)
            ts = msgs[0].header.stamp
            if ts > latest_sync_timestamp:
                latest_sync_timestamp = ts
            msgs.pop(0)
            if earliest_sync_timestamp == None:
                earliest_sync_timestamp = ts
            elif ts < earliest_sync_timestamp:
                earliest_sync_timestamp = ts
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
        are_queues_empty = False
        for topic, msgs in self._sync_msg_queue.items():
            are_queues_empty = are_queues_empty or len(msgs) == 0
        return are_queues_empty

    def save_topic(self, data, topic, trajectory, frame):
        topic_type = self._topic_to_type[topic]
        topic_type_subpath = BagDecoder.SENSOR_DIRECTORY_SUBPATH[topic]
        save_dir = os.path.join(self._outdir, topic_type_subpath, str(trajectory))
        filename = BagDecoder.set_filename_by_topic(topic, trajectory, frame)
        filepath = os.path.join(save_dir, filename)
        if not self._gen_data:
            return filepath
        if not os.path.exists(save_dir):
            print("Creating %s because it does not exist..." % save_dir)
            os.makedirs(save_dir)
        if topic_type == "ouster_ros/PacketMsg" or topic_type == "sensor_msgs/PointCloud2":
            BagDecoder.pc_to_bin(data, filepath)  # Expects PointCloud2 Object
        return filepath

    def process_topic(self, topic, msg, t):
        if not topic in self._topic_to_type:
            print("Could not find topic %s in topic map for time %i, exiting..." % (topic, t.to_sec()))
            return
        topic_type = self._topic_to_type[topic]
        data = None
        sensor_ts = t
        if topic_type == "ouster_ros/PacketMsg":
            data, sensor_ts = BagDecoder.process_ouster_packet(self._os1_info, msg, topic, sensor_ts)
        # elif topic_type == "sensor_msgs/PointCloud2":
        #     data, sensor_ts = process_pc2(msg, sensor_ts)
        return data, sensor_ts

    @staticmethod
    def process_ouster_packet(os1_info, packet_arr, topic, sensor_ts):
        # Process Header
        packets = client.Packets(packet_arr, os1_info)
        scans = client.Scans(packets)
        rg = nth(scans, 0).field(client.ChanField.RANGE)
        rf = nth(scans, 0).field(client.ChanField.REFLECTIVITY)
        signal = nth(scans, 0).field(client.ChanField.SIGNAL)
        nr = nth(scans, 0).field(client.ChanField.NEAR_IR)
        ts = nth(scans, 0).timestamp

        # Set relative timestamp for each point
        init_ts = ts[0]
        ts_horizontal_rel = ts - init_ts
        ts_horizontal_rel[ts_horizontal_rel < 0] = 0
        ts_points = np.tile(ts_horizontal_rel, (BagDecoder.OS1_POINTCLOUD_SHAPE[1], 1))

        # Set ring to correspond to row idx
        ring_idx = np.arange(0, 128, 1).reshape(-1, 1)
        ring = np.tile(ring_idx, (1, BagDecoder.OS1_POINTCLOUD_SHAPE[0]))

        # Project Points to ouster LiDAR Frame
        xyzlut = client.XYZLut(os1_info)
        xyz_points = client.destagger(os1_info, xyzlut(rg))

        # Homogeneous xyz coordinates
        homo_xyz = np.ones((xyz_points.shape[0], xyz_points.shape[1], 1))
        xyz_points = np.dstack((xyz_points, homo_xyz))

        # Change from LiDAR to sensor coordinate system
        signal = np.expand_dims(signal, axis=-1)
        rf = np.expand_dims(rf, axis=-1)
        ts_points = np.expand_dims(ts_points, axis=-1)
        rg = np.expand_dims(rg, axis=-1)
        nr = np.expand_dims(nr, axis=-1)
        ring = np.expand_dims(ring, axis=-1)

        pc = np.dstack((xyz_points, signal, ts_points, rf, ring, nr, rg)).astype(np.float32)

        return pc, sensor_ts

    @staticmethod
    def pub_pc_to_rviz(pc, pc_pub, ts, frame_id="os_sensor", publish=True):
        if not isinstance(ts, rospy.Time):
            ts = rospy.Time.from_sec(ts)
        is_intensity = pc.shape[-1] >= 4
        is_time = pc.shape[-1] >= 5
        is_rf = pc.shape[-1] >= 6

        if is_rf:
            rf_start_offset = 5
            rf_middle_offset = rf_start_offset + 1
            rf_end_offset = 8

            pc_first = pc[:, :, :rf_start_offset]
            pc_time = pc[:, :, rf_start_offset].astype(np.uint32)
            pc_middle = pc[:, :, rf_middle_offset:rf_end_offset].astype(np.uint16)
            pc_end = pc[:, :, rf_end_offset:]

            first_bytes = pc_first.reshape(-1, 1).tobytes()
            time_bytes = pc_time.reshape(-1, 1).tobytes()
            middle_bytes = pc_middle.reshape(-1, 1).tobytes()
            # middle_pad_bytes= np.zeros((pc.shape[0], pc.shape[1], 1), dtype=np.uint16).tobytes()
            end_bytes = pc_end.reshape(-1, 1).tobytes()

            first_bytes_np = np.frombuffer(first_bytes, dtype=np.uint8).reshape(-1, rf_start_offset * 4)
            time_bytes_np = np.frombuffer(time_bytes, dtype=np.uint8).reshape(-1, 4)
            middle_bytes_np = np.frombuffer(middle_bytes, dtype=np.uint8).reshape(-1, 2 * 2)
            # middle_pad_bytes_np = np.frombuffer(middle_pad_bytes, dtype=np.uint8).reshape(
            #     -1, 2
            # )
            end_bytes_np = np.frombuffer(end_bytes, dtype=np.uint8).reshape(-1, 8)
            all_bytes_np = np.hstack((first_bytes_np, time_bytes_np,
                                      middle_bytes_np, end_bytes_np))

            # Add ambient and range bytes
            all_bytes_np = all_bytes_np.reshape(-1, 1)

        pc_flat = pc.reshape(-1, 1)

        DATATYPE = PointField.FLOAT32
        if pc.itemsize == 4:
            DATATYPE = PointField.FLOAT32
        else:
            DATATYPE = PointField.FLOAT64
            print("Undefined datatype size accessed, defaulting to FLOAT64...")

        pc_msg = PointCloud2()
        if pc.ndim > 2:
            pc_msg.width = pc.shape[1]
            pc_msg.height = pc.shape[0]
        else:
            pc_msg.width = 1
            pc_msg.height = pc.shape[0]

        pc_msg.header = std_msgs.msg.Header()
        pc_msg.header.stamp = ts
        pc_msg.header.frame_id = frame_id

        fields = [
            PointField('x', 0, DATATYPE, 1),
            PointField('y', pc.itemsize, DATATYPE, 1),
            PointField('z', pc.itemsize * 2, DATATYPE, 1)
        ]

        pc_item_position = 4
        if is_time:
            pc_msg.point_step = pc.itemsize * 7 + 6 + 2  # for actual values, 2 for padding
            fields.append(PointField('intensity', 16, DATATYPE, 1))
            fields.append(PointField('t', 20, PointField.UINT32, 1))
            fields.append(PointField('reflectivity', 24, PointField.UINT16, 1))
            fields.append(PointField('ring', 26, PointField.UINT16, 1))
            fields.append(PointField('ambient', 28, PointField.UINT16, 1))
            fields.append(PointField('range', 32, PointField.UINT32, 1))
            # fields.append(PointField('t', int(pc.itemsize*4.5), PointField.UINT32, 1))
        elif is_rf:
            pc_msg.point_step = pc.itemsize * 5 + 2
            fields.append(PointField('intensity', pc.itemsize * pc_item_position, DATATYPE, 1))
            fields.append(PointField('ring', pc.itemsize * (pc_item_position + 1), PointField.UINT16, 1))
        elif is_intensity:
            pc_msg.point_step = pc.itemsize * 4
            fields.append(PointField('intensity', pc.itemsize * pc_item_position, DATATYPE, 1))
        else:
            pc_msg.point_step = pc.itemsize * 3

        pc_msg.row_step = pc_msg.width * pc_msg.point_step
        pc_msg.fields = fields
        if is_rf:
            pc_msg.data = all_bytes_np.tobytes()
        else:
            pc_msg.data = pc_flat.tobytes()
        pc_msg.is_dense = True

        # if publish:
        #     pc_pub.publish(pc_msg)

        return pc_msg

    @staticmethod
    def set_filename_by_topic(topic, trajectory, frame):
        sensor_subpath = BagDecoder.SENSOR_DIRECTORY_SUBPATH[topic]
        sensor_prefix = sensor_subpath.replace("/", "_")  # get sensor name
        sensor_filetype = BagDecoder.SENSOR_DIRECTORY_FILETYPES[sensor_subpath]

        sensor_filename = "%s_%s_%s.%s" % (sensor_prefix, str(trajectory),
                                           str(frame), sensor_filetype)

        return sensor_filename

    @staticmethod
    def get_ouster_packet_info(os1_info, data):
        return client.LidarPacket(data, os1_info)

    @staticmethod
    def pc_to_bin(pc, filename, include_time=True):
        # pc_np = np.array(ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pc))

        pc_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(pc)
        pc_dim = 4
        if include_time:
            pc_dim = 5
        pc_np = np.zeros((pc_cloud.shape[0], pc_cloud.shape[1], pc_dim), dtype=np.float32)
        pc_np[..., 0] = pc_cloud['x']
        pc_np[..., 1] = pc_cloud['y']
        pc_np[..., 2] = pc_cloud['z']
        pc_np[..., 3] = pc_cloud['intensity']
        if pc_dim == 5:
            pc_np[..., 4] = pc_cloud['t']

        pc_np = pc_np.reshape(-1, pc_dim)

        flat_pc = pc_np.reshape(-1).astype(np.float32)
        flat_pc.tofile(filename)  # empty sep=bytes

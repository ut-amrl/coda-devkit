import os
from os.path import join
import pdb

# Utility Libraries
import yaml

# ROS Libraries
import rospy
import rosbag
from visualization_msgs.msg import *
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry

from helpers.sensors import *
from helpers.visualization import pub_pc_to_rviz, pub_img
from helpers.constants import *
from helpers.geometry import densify_poses_between_ts
from helpers.synchronization import Synchronize

from multiprocessing import Pool
import tqdm

class BagDecoder(object):
    """
    Decodes directory of bag files into dataset structure defined in README.md
    """
    def __init__(self, config, is_config_dict=False):
        self.is_config_dict = is_config_dict
        if not is_config_dict:
            self.settings_fp = os.path.join(os.getcwd(), config)
            assert os.path.isfile(self.settings_fp), '%s does not exist' % self.settings_fp
        else:
            self.settings_fp = config

        #Load available bag files
        print("Loading settings from ", self.settings_fp)

        #Setup class globals
        self.load_settings()

        #Generate Dataset
        if self.gen_data:
            self.gen_dataset_structure(self.outdir, self.sensor_topics)

        #Load sync publisher
        rospy.init_node('coda', anonymous=True)
        self.pub_rate = rospy.Rate(10) # Publish at 10 hz
        self.topic_to_type = None

        # Load bag complete header signal
        self.bagdecoder_signal = rospy.Publisher('bagdecoder_signal', String, queue_size=10)

    def gen_dataset_structure(self, outdir, sensor_topics):
        print("Generating processed dataset subdirectories...")

        for topic, info in sensor_topics.items():
            subdir_path = join(outdir, info['save_subdir'])
            if not os.path.exists(subdir_path):
                if self.verbose:
                    print("Creating directory ", subdir_path)
                os.makedirs(subdir_path)

    def load_settings(self):
        if self.is_config_dict:
            settings = self._settings_fp
        else:
            settings = yaml.safe_load(open(self.settings_fp, 'r'))

        #Load bag file directory
        self.repository_root   = settings["repository_root"]
        self.bag_date          = settings["bag_date"]
        self.bag_dir           = os.path.join(self.repository_root, self.bag_date)
        assert os.path.isdir(self.bag_dir), '%s does not exist' % self.bag_dir
        print("Loading bags from  ", self.bag_dir)

        # Load decoder settings
        self.gen_data  = settings['gen_data']
        self.vis_topics= settings['vis_topics']
        self.verbose   = settings['verbose']
        self.outdir    = settings['dataset_output_root']
        self.bags_to_traj_ids  = settings['bags_to_traj_ids']
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)

        # Load bag settings
        self.sensor_topics  = settings['sensor_topics']
        self.sync_method    = settings["sync_method"] if "sync_method" in settings else None
        self.trigger_topic  = settings['trigger_topic'] if "trigger_topic" in settings else None
        self.point_fields   = settings['point_fields'] if "point_fields" in settings else None
        
        if self.verbose:
            print("Saving topics: ", self.sensor_topics)

        self.bags_to_process   = settings['bags_to_process']
        self.all_bags          = [ file for file in sorted(os.listdir(self.bag_dir)) if os.path.splitext(file)[-1]==".bag"]
        if len(self.bags_to_process)==0:
            self.bags_to_process = self.all_bags

        # Load LiDAR decoder settings
        self.lidar_info_path = settings['lidar_info_path']
        if "/ouster/lidar_packets" in self.sensor_topics:
            os_metadata = os.path.join(self.bag_dir, self.lidar_info_path)
            self.os1_dict, self.os1_info = read_ouster_info(os_metadata)

            self.OS1_PACKETS_PER_FRAME = int(self.os1_dict['data_format']['columns_per_frame'] /
            self.os1_dict['data_format']['columns_per_packet'])

    def setup_sensor_topics(self, topic_to_type):
        if topic_to_type==None:
            print("Error: Topic to type mappings not defined yet. Exiting...")
            exit(1)

        topics_pubs = {}
        topics_msg_queue = {}
        for topic, info in self.sensor_topics.items():
            topic_type  = topic_to_type[topic]
            topic_sync  = info['sync']
            topic_class = ROSTYPE_TO_CLASS[topic_type] if topic_type in ROSTYPE_TO_CLASS else None
            
            if topic_class!=None:
                topics_pubs[topic] = rospy.Publisher(
                    f'/coda{topic}', topic_class, queue_size=10
                )
                topics_msg_queue[topic] = []
            else:
                print(f'Undefined topic {topic} for sync filter, skipping...')
        
        return topics_pubs, topics_msg_queue

    def reset_stale_files(self, ts_path, trajectory_idx):
        # Reset timestamps file
        frame_to_ts = open(ts_path, "w+")
        frame_to_ts.close()
       
        # Reset old sensor files
        for topic, info in self.sensor_topics.items():
            if info['save_ext']=='txt':
                save_dir = os.path.join(self.outdir, info['save_subdir'])
                filename = f'{trajectory_idx}.txt' 
                filepath = os.path.join(save_dir, filename)
                print(f'Resetting previous file at location {filepath}')
                topic_file  = open(filepath, 'w+')
                topic_file.close()

    def convert_bag(self):
        """
        Decodes requested topics in bag file to individual files
        """
        for traj_idx, bag_file in enumerate(self.all_bags):
            print("Bag file ", bag_file)

            #0 Setup ros message synchronizer
            synchronizer = Synchronize(
                self.sensor_topics, self.sync_method, self.trigger_topic
            )

            #1 Signal start of bag processing
            bag_date = os.path.basename(self.bag_dir)
            decode_signal_str = "START %s %s" % (bag_date, bag_file) 
            self.bagdecoder_signal.publish(decode_signal_str)

            if bag_file not in self.bags_to_process:
                continue

            if len(self.bags_to_traj_ids)==len(self.bags_to_process):
                bag_idx = self.bags_to_process.index(bag_file)
                traj_idx = self.bags_to_traj_ids[bag_idx]

            bag_fp = os.path.join(self.bag_dir, bag_file)
            print("Processing bag ", bag_fp, " as trajectory", traj_idx)

            #2 Set up ros topic listeners and publishers
            rosbag_info = yaml.safe_load( rosbag.Bag(bag_fp)._get_yaml_info() )
            topic_to_type = {topic_entry['topic']:topic_entry['type'] \
                for topic_entry in rosbag_info['topics']}
            topic_pubs, topic_msg_queues = self.setup_sensor_topics(topic_to_type)
                    
            #Create frame to timestamp map
            frame_to_ts_path= os.path.join(self.outdir, "timestamps", f'{traj_idx}.txt')
            if self.gen_data:
                self.reset_stale_files(frame_to_ts_path, traj_idx)
                if self.verbose:
                    print(f'Writing frame to timestamp map to {frame_to_ts_path}\n')

            # Reset lidar packet queue
            lidar_state_dict = {
                'frame': 0,
                'qp_ts': None,
                'qp_counter': 0,
                'qp_frame_id': -1,
                'qp_scan_queue': []
            } 
            bagfile = rosbag.Bag(bag_fp, chunk_threshold=200000000) # 200MB chunk size
            for topic, msg, ts in rosbag.Bag(bag_fp).read_messages():
                if topic in self.sensor_topics.keys():
                    topic_type = topic_to_type[topic]
                    info = self.sensor_topics[topic]

                    # Process topic and update msg with point cloud if all packets received
                    if topic_type=="ouster_ros/PacketMsg":
                        lidar_state_dict, msg = self.qpacket(topic, topic_type, msg, ts, lidar_state_dict)
                        if msg is not None: # Convert points numpy to pointcloud2
                            msg = pub_pc_to_rviz(
                                    msg, topic_pubs[topic], ts,  
                                    point_type=self.point_fields, 
                                    seq=lidar_state_dict['frame'],
                                    publish=False
                                )

                    #2 Synchronize and save topics
                    if info['sync'] and msg is not None:
                        sync_dict = synchronizer.synchronize(topic, msg)
                        
                        if sync_dict is not None:
                            # Save sync topic
                            self.save_sync_topics(
                                sync_dict, 
                                topic_to_type, 
                                traj_idx, 
                                lidar_state_dict['frame'],
                                frame_to_ts_path
                            )
                            lidar_state_dict['frame'] += 1

                            if self.vis_topics:
                                self.pub_sync_topics(topic_pubs, sync_dict)
                    elif not info['sync']:
                        self.save_topic(msg, topic, topic_type, traj_idx, ts)
                        if self.vis_topics:
                            topic_pubs[topic].publish(msg)

            print("Completed processing bag ", bag_fp)

    def pub_sync_topics(self, topic_pubs, sync_dict):
        for topic, msg in sync_dict['topics'].items():
            topic_pubs[topic].publish(msg)

    def save_sync_topics(self, sync_dict, topic_to_type, traj_idx, frame, frame_to_ts_path):
        #1 Save sync topics
        ts = sync_dict['ts']
        for topic, msg in sync_dict['topics'].items():
            topic_type = topic_to_type[topic]
            self.save_topic(msg, topic, topic_type, traj_idx, frame)

        #2 Save frame to timestamp map
        self.save_frame_ts(frame_to_ts_path, ts, frame)

    def save_frame_ts(self, frame_to_ts_path, timestamp, frame):
        if self.gen_data:
            if self.verbose:
                print("Saved frame %i timestamp %10.6f" % (frame, timestamp.to_sec()))
            
            with open(frame_to_ts_path, "a") as frame_to_ts:
                frame_to_ts.write("%10.6f\n" % timestamp.to_sec())

    def qpacket(self, topic, topic_type, msg, ts, state_dict, do_sync=True):
        published_packet = False
        packet = get_ouster_packet_info(self.os1_info, msg.buf)
        
        pc_np = None
        if packet.frame_id!=state_dict['qp_frame_id']:
            if state_dict['qp_counter']==self.OS1_PACKETS_PER_FRAME:
                pc_np, _ = self.process_topic(topic, topic_type, state_dict['qp_scan_queue'], ts)

            state_dict['qp_counter']    = 0
            state_dict['qp_frame_id']   = packet.frame_id
            state_dict['qp_scan_queue'] = []

        state_dict['qp_counter'] += 1
        state_dict['qp_scan_queue'].append(packet)

        return state_dict, pc_np

    def save_topic(self, data, topic, topic_type, trajectory, frame):
        info = self.sensor_topics[topic]
        topic_type_subpath = info['save_subdir']
        save_ext = info['save_ext']

        save_subdir = join(topic_type_subpath, str(trajectory))
        save_dir = os.path.join(self.outdir, save_subdir)
        filename = set_filename_by_topic(topic, save_subdir, save_ext, trajectory, frame)
        filepath = os.path.join(save_dir, filename)

        if not self.gen_data:
            return filepath
  
        if info['save_ext']=='txt':
            save_dir = os.path.join(self.outdir, topic_type_subpath)
            filename = f'{trajectory}.txt' 
            filepath = os.path.join(save_dir, filename)

        if not os.path.exists(save_dir):
            print("Creating %s because it does not exist..." % save_dir)
            os.makedirs(save_dir)

        if self.verbose:
            print("Saving topic ", topic_type, " with timestamp ", 
                data.header.stamp, " at frame ", frame)

        if topic_type=="ouster_ros/PacketMsg" or topic_type=="sensor_msgs/PointCloud2":
            pc_to_bin(data, filepath)
        elif topic_type=="sensor_msgs/Image":
            proc_data, _ = self.process_topic(topic, topic_type, data, data.header.stamp)
            if "depth" in topic:
                img_to_file(proc_data, filepath, depth=True)
            else:
                img_to_file(proc_data, filepath, depth=False)
        elif topic_type=="sensor_msgs/CompressedImage":
            proc_data, _ = self.process_topic(topic, topic_type, data, data.header.stamp)
            img_to_file(proc_data, filepath, depth=False)
        elif topic_type=="sensor_msgs/Imu":
            proc_data, _ = self.process_topic(topic, topic_type, data, data.header.stamp)

            imu_to_txt(proc_data, filepath)
        elif topic_type=="sensor_msgs/MagneticField":
            proc_data, _ = self.process_topic(topic, topic_type, data, data.header.stamp)

            mag_to_txt(proc_data, filepath)
        elif topic_type=="sensor_msgs/NavSatFix":
            proc_data, _ = self.process_topic(topic, topic_type, data, data.header.stamp)

            if proc_data!=None:
                gps_to_txt(proc_data, filepath)
        elif topic_type=="nav_msgs/Odometry":
            proc_data, _ = self.process_topic(topic, topic_type, data, data.header.stamp)

            odom_to_txt(proc_data, filepath)   
        else:
            if self._verbose:
                print("Entered undefined topic %s in save, skipping..."%topic)
            pass

        return filepath

    def process_topic(self, topic, topic_type, msg, t):
        data        = None
        sensor_ts   = t
        if topic_type=="ouster_ros/PacketMsg":
            data, sensor_ts = process_ouster_packet(
                self.os1_info, self.os1_dict, msg, topic, sensor_ts, point_fields=self.point_fields
            )
        elif topic_type=="sensor_msgs/Image":
            data, sensor_ts = process_image(msg)
        elif topic_type=="sensor_msgs/CompressedImage":
            encoding = "mono16" if "depth" in topic else "bgr8"
            data, sensor_ts = process_compressed_image(msg, encoding)
        elif topic_type=="sensor_msgs/Imu":
            imu_to_wcs = np.array(SENSOR_TO_XYZ_FRAME[topic]).reshape(4, 4)
            data, sensor_ts = process_imu(msg, imu_to_wcs)
        elif topic_type=="sensor_msgs/MagneticField":
            #No need to process magnetometer reading
            data = msg
        elif topic_type=="sensor_msgs/NavSatFix":
            data, sensor_ts = process_gps(msg)
        elif topic_type=="nav_msgs/Odometry":
            data = msg
        return data, sensor_ts

    """
    Post processing functions for generated datasets
    """
    def rectify_images(self, num_workers):
        cam0_dir = os.path.join(self._outdir, TWOD_RAW_DIR, 'cam0')
        assert os.path.isdir(cam0_dir), "Camera 0 directory not found at %s " % cam0_dir
        trajectories = sorted([int(traj) for traj in next(os.walk(cam0_dir))[1]])

        outdirs = [self._outdir]*len(trajectories)
        cam_calibrations = []
        for traj in trajectories:
            #Load calibrations if they exist
            calibrations_path = os.path.join(self._outdir, "calibrations", str(traj))
            if os.path.exists(calibrations_path):
                print("Calibrations directory exists %s, loading calibrations" % calibrations_path)
                cam_calibrations.append(self.load_cam_calibrations(self._outdir, traj))

            assert cam_calibrations!=None, "No camera calibrations found for traj %i" % traj

        pool = Pool(processes=num_workers)
        for _ in tqdm.tqdm(pool.imap_unordered(self.rectify_image, zip(outdirs, trajectories, cam_calibrations)), total=len(trajectories)):
            pass

    @staticmethod
    def densify_poses(poses_indir, poses_outdir, ts_dir, gen_data=True):
        if not os.path.exists(poses_outdir):
            print("Creating dense poses directory %s" % poses_outdir)
            os.makedirs(poses_outdir)
        
        pose_files = [pose_file for pose_file in os.listdir(poses_indir) if pose_file.endswith(".txt")]
        pose_files = sorted(pose_files, key=lambda x: int(x.split(".")[0]) )
        for pose_file in pose_files:
            traj = pose_file.split(".")[0]

            #Locate closest pose from frame
            ts_to_frame_path = os.path.join(ts_dir, "%s.txt"%traj)
            ts_to_poses_path = os.path.join(poses_indir, pose_file)
            frame_to_poses_np = np.loadtxt(ts_to_poses_path).reshape(-1, 8)
            frame_to_ts_np = np.loadtxt(ts_to_frame_path)
            dense_poses = densify_poses_between_ts(frame_to_poses_np, frame_to_ts_np)

            #Save dense poses
            pose_file_path = os.path.join(poses_outdir, pose_file)

            if gen_data:
                print("Saving dense poses for trajectory %s at path %s" % (traj, pose_file_path))
                np.savetxt(pose_file_path, dense_poses, fmt="%10.6f", delimiter=" ")
        print("Done generating dense poses...")

    @staticmethod
    def rectify_image(args):
        """
        Rectifies all images for a single trajectory
        """
        outdir, traj, cam_calibration = args
        cams = ['cam0', 'cam1']
        for cam in cams:
            cam_dir = os.path.join(outdir, TWOD_RAW_DIR, cam)
            cam_out_dir = os.path.join(outdir, TWOD_RECT_DIR, cam)
            if not os.path.exists(cam_out_dir):
                print("Creating %s rectified dir at %s " % (cam, cam_dir))
                os.mkdir(cam_out_dir)

            img_dir = os.path.join(outdir, '2d_raw', cam, str(traj))
            img_dir_files = os.listdir(img_dir)
            img_dir_files = sorted(img_dir_files, key=lambda x: int(x.split('.')[0].split('_')[-1]))
            
            rect_img_dir = os.path.join(img_dir.replace('raw', 'rect'))

            for img_file in img_dir_files:
                _, _, _, frame = get_filename_info(img_file)
                img_path = os.path.join(img_dir, img_file)
                rect_img_file   = set_filename_by_prefix(TWOD_RECT_DIR, cam, traj, frame)

                if not os.path.exists(rect_img_dir):
                    os.makedirs(rect_img_dir)
                if os.path.isfile(img_path):
                    intrinsics = cam_calibration['%s_intrinsics'%cam]

                    rectified_img_np = rectify_image(img_path, intrinsics)

                    rect_img_path = os.path.join(rect_img_dir, rect_img_file)
                    success = cv2.imwrite(rect_img_path, rectified_img_np)
                    if not success:
                        print("Error writing rectified image to %s " % (rect_img_path))

    @staticmethod
    def load_cam_calibrations(outdir, trajectory):
        calibrations_path = os.path.join(outdir, CALIBRATION_DIR, str(trajectory))
        calibration_fps = [os.path.join(calibrations_path, file) for file in os.listdir(calibrations_path) if file.endswith(".yaml")]

        cam_calibrations = {}
        for calibration_fp in calibration_fps:
            cal, src, tar = get_calibration_info(calibration_fp)

            if 'cam' in src:
                cal_id = "%s_%s"%(src, tar)
                cam_id = src[-1]

                if cal_id not in cam_calibrations.keys():
                    cam_calibrations[cal_id] = {}
                    cam_calibrations[cal_id]['cam_id'] = cam_id

                cam_calibrations[cal_id].update(cal)
        return cam_calibrations
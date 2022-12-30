import os
import pdb
import copy
import shutil
import yaml
import numpy as np
import shutil

import rospy
from sensor_msgs.msg import PointCloud2
from helpers.visualization import pub_pc_to_rviz
from helpers.visualization import *

from helpers.constants import *
from helpers.geometry import *
from scipy.spatial.transform import Rotation as R

from helpers.sensors import *

class AnnotationEncoder(object):
    """
    """
    def __init__(self):
        self._settings_fp = os.path.join(os.getcwd(), "config/encoder.yaml")

        with open(self._settings_fp, 'r') as settings_file:
            settings = yaml.safe_load(settings_file)

            self._trajs         = settings['trajectories']
            self._traj_frames   = settings['trajectory_frames']
            self._ds_rate       = settings['downsample_rate']
            self._indir         = settings['dataset_input_root']
            self._outdir        = settings['dataset_output_root']
            self._sensor_topics = settings['sensor_topics']
            self._copy_files    = settings['copy_files']
            self._enc_format    = settings['encoding_format']
            self._use_wcs       = settings['use_wcs']

            assert len(self._sensor_topics)

            # Sagemaker Specific
            self._prefix        = settings['s3_prefix']
        
        rospy.init_node('bin_publisher', anonymous=True)
        self._pc_pub = rospy.Publisher('/coda/bin/points', PointCloud2, queue_size=10)
        self._pose_pub = rospy.Publisher('/coda/pose', PoseStamped, queue_size=10)

        #Directory checks
        assert os.path.isdir(self._indir), '%s does not exist' % self._indir
        if not os.path.exists(self._outdir):
            print("Output directory does not exist, creating at %s " % self._outdir)
            os.mkdir(self._outdir)

        if self._enc_format=="sagemaker":
            self._sequences = os.path.join(self._outdir, "sequences")
            self._manifests = os.path.join(self._outdir, "manifests")
            if not os.path.exists(self._sequences):
                os.mkdir(self._sequences)
            if not os.path.exists(self._manifests):
                os.mkdir(self._manifests)

    def encode_annotations(self):
        if self._enc_format=="sagemaker":
            self.copy_sensor_files()
        elif self._enc_format=="scale":
            self.copy_sensor_files()
        elif self._enc_format=="deepen":
            self.create_deepen_structure()
            self.create_json_files()

    def create_deepen_structure(self):
        if not os.path.exists(self._outdir):
            print("Creating outdir %s since it does not exist..." % self._outdir)
            os.makedirs(self._outdir)

        # Json file directory
        json_dir = os.path.join(self._outdir, "3d_formatted")
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        # Camera directory
        cams_dir = ["cam0", "cam1"]
        for cam in cams_dir:
            cam_dir = os.path.join(self._outdir, DIR_2D_RAW, cam)
            if not os.path.exists(cam_dir):
                os.makedirs(cam_dir)

    def create_json_files(self):
         for traj in self._trajs:
            traj_frames = self._traj_frames[traj]

            in_pose_file = os.path.join(self._indir, "poses", "%i.txt"%traj)
            assert os.path.isfile(in_pose_file), '%s does not exist' % in_pose_file

            in_ts_file = os.path.join(self._indir, "timestamps", "%i_frame_to_ts.txt"%traj)
            assert os.path.isfile(in_ts_file), '%s does not exist' % in_ts_file

            in_pose_np = np.fromfile(in_pose_file, sep=' ').reshape(-1, 8)
            frame_to_ts_np = np.fromfile(in_ts_file, sep=' ').reshape(-1, 1)
            out_pose_np = densify_poses_between_ts(in_pose_np, frame_to_ts_np)

            for frame_seq in traj_frames:
                start, end = frame_seq[0], frame_seq[1]

                out_frame_dir = os.path.join(self._outdir, "3d_formatted", str(traj))
                if not os.path.exists(out_frame_dir):
                    print("Creating output directory for trajectory %i at %s", 
                        (traj, out_frame_dir) )
                    os.makedirs(out_frame_dir)

                for frame in range(start, end):
                    pose        = out_pose_np[frame]
                    timestamp   = frame_to_ts_np[frame]

                    # Save images and points to json formatted str
                    image_str = self.fill_json_images_dict(traj, frame, pose, timestamp)
                    points_str  = self.create_json_points_str(traj, frame, pose)
                    frame_str = image_str + points_str
                    
                    # Write json formatted string to outdir
                    json_filename = "%06d.json" % frame
                    json_path = os.path.join(out_frame_dir, json_filename)
                    json_file = open(json_path, "w+")
                    json_file.write(frame_str)
                    json_file.close()

                    # Copy .bin files
                    subdir = os.path.join(DIR_3D_RAW, "os1")
                    bin_name= set_filename_by_prefix(DIR_3D_RAW, "os1", traj, frame)
                    self.copy_file_dir(self._indir, self._outdir, subdir, str(traj), bin_name)

                    # Copy matched images
                    self.copy_cam_dir(traj, frame, "cam0")
                    self.copy_cam_dir(traj, frame, "cam1")

                    print("Wrote json output for frame %i at path %s" % (frame, json_path))

    def copy_cam_dir(self, traj, frame, sensor_name):
        modality = DIR_2D_RAW
        img_filename    = set_filename_by_prefix(modality, sensor_name, traj, frame)
        img_path        = os.path.join(self._indir, modality, sensor_name, 
            str(traj), img_filename)
        
        img_in_path = os.path.join(self._indir, img_path)
        img_out_path = img_path.replace(self._indir, self._outdir)
        copy_image(img_in_path, img_out_path)

    def create_json_points_str(self, traj, frame, pose):
        modality, sensor_name = DIR_3D_RAW, "os1"
        bin_filename    = set_filename_by_prefix(modality, sensor_name, traj, frame)
        bin_path        = os.path.join(self._indir, modality, sensor_name, str(traj), bin_filename)

        bin_np          = read_bin(bin_path, False)  
        bin_np          = np.hstack((bin_np, np.ones( (bin_np.shape[0], 1) )))
        ego_to_wcs      = pose_to_homo(pose)
        bin_np      = (ego_to_wcs @ bin_np.T).T
        bin_np      = bin_np[:, :3].astype(np.float32)

        # pub_pc_to_rviz(bin_np, self._pc_pub, rospy.get_rostime())
        # pub_pose(self._pose_pub, pose, frame, rospy.get_rostime())
        # # pdb.set_trace()
        # pcd_path = os.path.join(self._outdir, "3d_pcd", "traj")
        # if not os.path.exists(pcd_path):
        #     os.makedirs(pcd_path)
        # pcd_file = os.path.join(pcd_path, "%i.pcd"%frame)
        # bin_to_ply(bin_np, pcd_file)
        
        points_arr_str  = copy.deepcopy(DEEPEN_POINTS_PREFIX)
        point_dict = copy.deepcopy(DEEPEN_POINT_DICT)
        for entry in bin_np:
            point_dict["x"], point_dict["y"], point_dict["z"] = entry
            point_str = DEEPEN_POINT_ENTRY % point_dict
            points_arr_str += point_str

        points_arr_str = points_arr_str[:-2]
        points_arr_str += DEEPEN_POINTS_SUFFIX

        return points_arr_str


    def fill_json_images_dict(self, traj, frame, pose, ts):
        in_calib_dir = os.path.join(self._indir, "calibrations", str(traj) )
        cam0_intrinsics_path= os.path.join(in_calib_dir, "calib_cam0_intrinsics.yaml")
        cam1_intrinsics_path= os.path.join(in_calib_dir, "calib_cam1_intrinsics.yaml")
        cam0_to_cam1_path   = os.path.join(in_calib_dir, "calib_cam0_to_cam1.yaml")
        os1_to_cam0_path    = os.path.join(in_calib_dir, "calib_os1_to_cam0.yaml")

        cam0_intr_file  = open(cam0_intrinsics_path, 'r')
        cam1_intr_file  = open(cam1_intrinsics_path, 'r')
        cam0tocam1_file = open(cam0_to_cam1_path, 'r')
        os1tocam0_file  = open(os1_to_cam0_path, 'r')

        cam0_intr = yaml.safe_load(cam0_intr_file)
        cam1_intr = yaml.safe_load(cam1_intr_file)
        cam0tocam1_y  = yaml.safe_load(cam0tocam1_file)
        os1tocam0_y = yaml.safe_load(os1tocam0_file)

        cam0tocam1= np.eye(4)
        cam0tocam1[:3, :3]= np.array(cam0tocam1_y['extrinsic_matrix']\
            ['R']['data']).reshape(3, 3)
        cam0tocam1[:3, 3] = np.array(cam0tocam1_y['extrinsic_matrix']['T']
            ).reshape(3,)
        os1tocam0 = np.array(os1tocam0_y['extrinsic_matrix']\
            ['data']).reshape(4, 4)

        # TODO: once we log poses in base_link frame, add base_link to lidar transform here
        wcs_pose    = pose_to_homo(pose)

        cam0pose    = wcs_pose @ np.linalg.inv(os1tocam0)
        cam1pose    = cam0pose @ np.linalg.inv(cam0tocam1)

        modality, sensor_name = DIR_2D_RAW, "cam0"
        cam0_filename   = set_filename_by_prefix(modality, sensor_name, traj, frame)
        cam0_dict  = self.create_image_dict(cam0_filename, ts, cam0pose, cam0_intr)

        sensor_name = "cam1"
        cam1_filename   = set_filename_by_prefix(modality, sensor_name, traj, frame)
        cam1_dict  = self.create_image_dict(cam1_filename, ts, cam1pose, cam1_intr)

        image_suffix_dict = DEEPEN_IMAGE_SUFFIX_DICT
        image_suffix_dict["ts"] = ts 
        image_suffix_dict["dpx"], image_suffix_dict["dpy"], image_suffix_dict["dpz"] = \
            wcs_pose[:3, 3]
        image_suffix_dict["dhx"], image_suffix_dict["dhy"], image_suffix_dict["dhz"], \
            image_suffix_dict["dhw"] = R.from_matrix(wcs_pose[:3, :3]).as_quat()

        image_str   = copy.deepcopy(DEEPEN_IMAGE_PREFIX)
        image_str   += DEEPEN_IMAGE_ENTRY % cam0_dict
        image_str   += DEEPEN_IMAGE_ENTRY % cam1_dict
        image_str   = image_str[:-1]
        image_str   += DEEPEN_IMAGE_SUFFIX % image_suffix_dict

        return image_str

    def create_image_dict(self, img_file, ts, wcs_pose, intr):
        """
        img_file        - image filename
        frame_to_ts_np  - np array each each index is the corresponding timestamp
        wcs_pose        - pose of camera in wcs
        intr            - camera intrinsic
        """
        image_dict = copy.deepcopy(DEEPEN_IMAGE_DICT)

        modality, sensor_name, trajectory, frame = get_filename_info(img_file)
        timestamp = ts

        wcs_trans = wcs_pose[:3, 3]
        wcs_quat    = R.from_matrix(wcs_pose[:3, :3]).as_quat()

        image_dict["ipath"] = os.path.join(modality, sensor_name, trajectory, img_file)
        image_dict["ts"]  = timestamp
        image_dict["fx"]  = intr["camera_matrix"]["data"][0]
        image_dict["fy"]  = intr["camera_matrix"]["data"][4]
        image_dict["cx"]  = intr["camera_matrix"]["data"][2]
        image_dict["cy"]  = intr["camera_matrix"]["data"][5]
        image_dict["k1"], image_dict["k2"], image_dict["p1"], \
            image_dict["p2"], image_dict["k3"] = intr["distortion_coefficients"]["data"]
        image_dict["px"], image_dict["py"], image_dict["pz"] = \
            wcs_trans
        image_dict["hx"], image_dict["hy"], image_dict["hz"], image_dict["hw"] = \
            wcs_quat

        return image_dict


    def copy_sensor_files(self):
        """
        Iterates through all specified trajectories and frame pairs 
        """
        for traj in self._trajs:
            traj_frames = self._traj_frames[traj]

            for frame_seq in traj_frames:
                start, end = frame_seq[0], frame_seq[1]

                self.load_frames(traj, start, end)
            
            if self._enc_format!="sagemaker":
                #Copy calibrations
                self.copy_calibration_by_trajectory(traj)
                #Copy timestamps
                self.copy_ts_by_trajectory(traj)
                #Copy estimated poses
                self.copy_pose_by_trajectory(traj)

    def copy_calibration_by_trajectory(self, trajectory):
        in_calib_dir = os.path.join(self._indir, "calibrations", str(trajectory) )
        assert os.path.isdir(in_calib_dir), '%s does not exist' % in_calib_dir

        out_calib_root_dir = os.path.join(self._outdir, "calibrations")
        out_calib_dir   = os.path.join(out_calib_root_dir, str(trajectory))
        if not os.path.exists(out_calib_root_dir):
            os.makedirs(out_calib_root_dir)
        
        if os.path.exists(out_calib_dir):
            print("Found existing calibration directory for trajectory %i, deleting..."%trajectory)
            shutil.rmtree(out_calib_dir)

        shutil.copytree(in_calib_dir, out_calib_dir)

    def copy_ts_by_trajectory(self, trajectory):
        in_ts_file = os.path.join(self._indir, "timestamps", "%i_frame_to_ts.txt"%trajectory)
        assert os.path.isfile(in_ts_file), '%s does not exist' % in_ts_file

        out_ts_dir = os.path.join(self._outdir, "timestamps")
        out_ts_file = in_ts_file.replace(self._indir, self._outdir)
        if not os.path.exists(out_ts_dir):
            os.makedirs(out_ts_dir)

        shutil.copyfile(in_ts_file, out_ts_file)

    def copy_pose_by_trajectory(self, trajectory):
        in_pose_file    = os.path.join(self._indir, "poses", "%i.txt"%trajectory)
        assert os.path.isfile(in_pose_file), '%s does not exist' % in_pose_file

        in_ts_file = os.path.join(self._indir, "timestamps", "%i_frame_to_ts.txt"%trajectory)
        assert os.path.isfile(in_ts_file), '%s does not exist' % in_ts_file

        in_pose_np = np.fromfile(in_pose_file, sep=' ').reshape(-1, 8)
        frame_to_ts_np = np.fromfile(in_ts_file, sep=' ').reshape(-1, 1)
        
        out_pose_np = densify_poses_between_ts(in_pose_np, frame_to_ts_np)

        out_pose_dir    = os.path.join(self._outdir, "poses")
        out_pose_file   = in_pose_file.replace(self._indir, self._outdir)
        if not os.path.exists(out_pose_dir):
            os.makedirs(out_pose_dir)

        np.savetxt(out_pose_file, out_pose_np, fmt="%10.6f", delimiter=" ")
        

    def load_frames(self, traj, start, end):
        """
        Assumes that 
        """
        assert end > start, "Invalid frames, start cannot be greater than or equal to end\n"
        
        # Load pose estimate
        pose_file = os.path.join(self._indir, "poses", "%s.txt" % traj)
        frame_to_ts_file    = os.path.join(self._indir, "timestamps", "%i_frame_to_ts.txt"%traj)
        assert os.path.isfile(pose_file), "Error: pose file for trajectory %s \
            cannot be found in filepath %s\n Exiting..."%(traj, pose_file)
        assert os.path.isfile(frame_to_ts_file), "Error: pose file for trajectory %s \
            cannot be found in filepath %s\n Exiting..."%(traj, frame_to_ts_file)
        pose_np     = np.fromfile(pose_file, sep=' ').reshape(-1, 8)
        ts_frame_np = np.fromfile(frame_to_ts_file, sep=' ').reshape(-1, 1)

        manifest_frames_str = ""
        frame_count = 0
        for frame_idx, frame in enumerate(range(start, end)):
            if frame_idx%self._ds_rate==0:
                #Interpolate pose from closest timstamp
                ts  = ts_frame_np[frame][0]
                pose    = find_closest_pose(pose_np, ts)

                sensor_files = ["", "", ""]
                for idx, topic in enumerate(self._sensor_topics):
                    subdir = SENSOR_DIRECTORY_SUBPATH[topic]
                    filetype = SENSOR_DIRECTORY_FILETYPES[subdir]
                    frame_filename = set_filename_by_topic(
                        topic, str(traj), frame
                    )

                    #Copy files to aws directory
                    if self._copy_files:
                        self.copy_file_dir(self._indir, self._outdir, subdir, 
                            str(traj), frame_filename)

                        if filetype=="bin" and self._use_wcs:
                            bin_file = os.path.join(self._outdir, subdir, str(traj), frame_filename)
                            wcs_bin_np = self.ego_to_wcs(bin_file, pose)
                            #Overwrite existing bin file in outdir directory
                            wcs_bin_np.tofile(bin_file)  
                
                        if filetype=="bin" and self._enc_format == "scale":
                            bin_file = os.path.join(self._outdir, subdir, str(traj), frame_filename)
                            ply_path = bin_file.replace(".bin", ".pcd")
                            try:
                                bin_np  = read_bin(bin_file, False)
                                pub_pc_to_rviz(bin_np, self._pc_pub, rospy.get_rostime())
                            except Exception as e:
                                pdb.set_trace()
                            bin_to_ply(bin_np, ply_path)
                    
                    sensor_files[idx] = os.path.join(subdir, str(traj), frame_filename)
            
                
                frame_curr = self.fill_frame_text(sensor_files, pose, ts, frame, CAM0_CALIBRATIONS)

                if frame>start:
                    manifest_frames_str += ",\n"

                manifest_frames_str += frame_curr

                if self._enc_format=="sagemaker":
                    self.create_manifest(traj, frame_count, manifest_frames_str, start, end)

                #Accum total frame count
                frame_count+=1

    def create_manifest(self, traj, frame_count, manifest_frames_str, start, end):
        if self._enc_format!="sagemaker":
            print("Warning: running manifest generation without sagemaker format\n")

        #Write sequences/manifest file
        seq_text = SEQ_TEXT % traj
        prefix_text = PREFIX_TEXT % self._prefix
        num_frames_text = NUM_FRAMES_TEXT % frame_count
        manifest_header_str = seq_text + prefix_text + num_frames_text
        manifest_file_str   = manifest_header_str + FRAMES_START_TEXT + \
            manifest_frames_str + FRAMES_END_TEXT
        
        sequence_filename = "seq%iframes%ito%i.json" % (traj, start, end)
        manifest_filepath   = os.path.join(self._sequences, sequence_filename)
        manifest_file       = open(manifest_filepath, "w+")
        print("Writing manifest file for trajectory %i to location %s... " \
            % (traj, manifest_filepath))
        manifest_file.write(manifest_file_str)
        manifest_file.close()

        #Write manifest path description file
        manifest_path_filename = "manifest%iframes%ito%i.json" % (traj, start, end)
        manifest_path_dict = MANIFEST_PATH_DICT
        manifest_path_dict["manifest_prefix"] = os.path.join(self._prefix, "sequences")
        manifest_path_dict["sequence_filename"] = sequence_filename
        manifest_path_str = MANIFEST_PATH_STR % manifest_path_dict
        manifest_path_fp   = os.path.join(self._manifests, manifest_path_filename)
        manifest_path_file  = open(manifest_path_fp, "w+")
        manifest_path_file.write(manifest_path_str)
        manifest_path_file.close()

    def ego_to_wcs(self, filepath, pose):
        """
        pose - given as ts x y z qw qx qy qz 
        """
        pose_mat = pose_to_homo(pose)
        bin_np  = read_bin(filepath, False)
        ones_col = np.ones((bin_np.shape[0], 1), dtype=np.float32)
        bin_np  = np.hstack((bin_np, ones_col))
        wcs_bin_np = (pose_mat@bin_np.T).T
        wcs_bin_np = wcs_bin_np[:, :3].astype(np.float32)

        return wcs_bin_np     

    def copy_file_dir(self, inroot, outroot, subdir, traj, filename, pose=np.eye(4)):
        indir = os.path.join(inroot, subdir, traj)
        outdir = os.path.join(outroot, subdir, traj)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        infile  = os.path.join(indir, filename)
        outfile = os.path.join(outdir, filename)

        shutil.copyfile(infile, outfile)

    def fill_frame_text(self, filepaths, pose, ts, frameno, cam_parameters):
        assert len(filepaths)==3, "Incorrect number of sensors %i passed to manifest \
            file" % len(filepaths)

        frame, cam0, cam1 = filepaths
        #Frame header
        frame_info = FRAME_TEXT_DICT
        frame_info['ts']        = ts
        frame_info['frameno']   = frameno
        frame_info['evppx']     = pose[1]
        frame_info['evppy']     = pose[2]
        frame_info['evppz']     = pose[3]
        frame_info['evphx']     = pose[5]
        frame_info['evphy']     = pose[6]
        frame_info['evphz']     = pose[7]
        frame_info['evphw']     = pose[4]

        #Camera Extrinsics to LiDAR
        cam0_r = R.from_euler('xyz', CAM0_CALIBRATIONS['extrinsics'][3:], degrees=True)
        cam0_quat = cam0_r.as_quat()
        frame_info['ipx'], frame_info['ipy'], frame_info['ipz'] = \
           CAM0_CALIBRATIONS['extrinsics'][0], CAM0_CALIBRATIONS['extrinsics'][1], \
           CAM0_CALIBRATIONS['extrinsics'][2]
        frame_info['ihx'], frame_info['ihy'], frame_info['ihz'], frame_info['ihw'] = \
            cam0_quat[0], cam0_quat[1], cam0_quat[2], cam0_quat[3]

        frame_info['frame'] = frame
        frame_info['ipath'] = cam0

        frame_info['fx'], frame_info['fy'] = cam_parameters["camera"][0], cam_parameters["camera"][4]
        frame_info['cx'], frame_info['cy'] = cam_parameters["camera"][2], cam_parameters["camera"][5]
        frame_info['k1'], frame_info['k2'], frame_info['k3'], frame_info['k4']= \
            cam_parameters["distortion"][0], cam_parameters["distortion"][1], \
            cam_parameters["distortion"][2], cam_parameters["distortion"][3]
        frame_info['p1'], frame_info['p2'] = 0, 0
        frame_info['its'] = ts

        frame_curr = FRAME_TEXT % frame_info
        return frame_curr

        
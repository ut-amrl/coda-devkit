import os
import pdb
import yaml
import math 
import numpy as np

from helpers.constants import *
from helpers.geometry import *
from scipy.spatial.transform import Rotation as R

class ManifestGenerator(object):
    """
    """
    def __init__(self):
        self._settings_fp = os.path.join(os.getcwd(), "config/manifest.yaml")

        with open(self._settings_fp, 'r') as settings_file:
            settings = yaml.safe_load(settings_file)

            self._trajs         = settings['trajectories']
            self._traj_frames   = settings['trajectory_frames']
            self._ds_rate       = settings['downsample_rate']
            self._indir         = settings['dataset_input_root']
            self._outdir        = settings['manifest_output_root']
            self._sensor_topics = settings['sensor_topics']

            assert len(self._sensor_topics)

            # Sagemaker Specific
            self._prefix        = settings['s3_prefix']

        #Directory checks
        assert os.path.isdir(self._indir), '%s does not exist' % self._indir
        if not os.path.exists(self._outdir):
            print("Output directory does not exist, creating at %s " % self._outdir)
            os.mkdir(self._outdir)

        self._sequences = os.path.join(self._outdir, "sequences")
        self._manifests = os.path.join(self._outdir, "manifests")
        if not os.path.exists(self._sequences):
            os.mkdir(self._sequences)
        if not os.path.exists(self._manifests):
            os.mkdir(self._manifests)

    def create_manifest(self):
        """
        Iterates through all specified trajectories and frame pairs 
        """
        for traj in self._trajs:
            traj_frames = self._traj_frames[traj]

            for frame_seq in traj_frames:
                start, end = frame_seq[0], frame_seq[1]

                self.load_frames(traj, start, end)

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
                sensor_files = ["", "", ""]
                for idx, topic in enumerate(self._sensor_topics):
                    subdir = SENSOR_DIRECTORY_SUBPATH[topic]
                    filetype = SENSOR_DIRECTORY_FILETYPES[subdir]

                    sensor_files[idx] = os.path.join(subdir, str(traj), "%i.%s"% (frame, filetype))

                ts  = ts_frame_np[frame][0]
                curr_ts_idx = np.searchsorted(pose_np[:, 0], ts, side="left")
                next_ts_idx = curr_ts_idx + 1
                if next_ts_idx>=pose_np.shape[0]:
                    next_ts_idx = pose_np.shape[0] - 1
                
                pose = inter_pose(pose_np[curr_ts_idx], pose_np[next_ts_idx], ts)
                frame_curr = self.fill_frame_text(sensor_files, pose, ts, frame, CAM0_CALIBRATIONS)
                
                if frame>start:
                    manifest_frames_str += ",\n"

                manifest_frames_str += frame_curr
                #Accum total frame count
                frame_count+=1

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

        
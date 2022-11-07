from genericpath import isdir
from math import degrees
import os
import pdb
from re import sub
from sysconfig import get_python_version
from tracemalloc import start
import yaml
import json
import copy

import sys

from helpers.sensors import set_filename_by_prefix
# For imports
sys.path.append(os.getcwd())

import numpy as np
from scipy.spatial.transform import Rotation as R

from helpers.constants import *
from helpers.geometry import *

class AnnotationDecoder(object):

    def __init__(self):
        
        settings_fp = os.path.join(os.getcwd(), "config/annotationdecoder.yaml")
        with open(settings_fp, 'r') as settings_file:
            settings = yaml.safe_load(settings_file)
            self._indir   = settings['annotations_input_root']
            self._outdir  = settings['annotations_output_root']
            self._anno_type = settings['annotation_type']
            self._gen_data  = settings['gen_data']
            self._use_wcs   = settings['use_wcs']

            assert os.path.exists(self._indir), "%s does not exist, provide valid dataset root directory."

    def decode_annotations(self):
        """
        Decodes all input files in annotation directory to common annotation format
        """

        if self._anno_type=="sagemaker":
            anno_subdir, anno_files, manifest_file = "", None, None
            for dir_item in os.listdir(self._indir):
                subitem = os.path.join(self._indir, dir_item)
                if os.path.isdir(subitem):
                    anno_subdir = dir_item
                    anno_indices = np.array([
                        anno.split('.')[0] for anno in os.listdir(subitem)
                    ])
                    anno_indices = np.argsort(anno_indices)
                    
                    anno_files = np.array(os.listdir(subitem))[anno_indices]
                else:
                    manifest_file = dir_item
            
            anno_dict = self.sagemaker_decoder(anno_subdir, anno_files, manifest_file)
            if self._gen_data:
                self.save_anno_json(anno_dict)
        elif self._anno_type=="ewannotate":
            for dir_item in sorted(os.listdir(self._indir)):
                subitem = os.path.join(self._indir, dir_item)
                if os.path.isdir(subitem):
                    continue
                traj = dir_item.split('.')[0]
                self.ewannotate_decoder(subitem, traj)
        else:
            print("Undefined annotation format type specified, exiting...")

    def ewannotate_decoder(self, filepath, traj):
        with open(filepath, 'r') as annos_file:
            annos = yaml.safe_load(annos_file)

            ts_to_frame_path = os.path.join(self._outdir, "timestamps", "%s_frame_to_ts.txt"%traj)
            ts_to_poses_path = os.path.join(self._outdir, "poses", "%s.txt"%traj)

            frame_to_poses_np = np.loadtxt(ts_to_poses_path).reshape(-1, 8)

            ts_to_frame_np = np.loadtxt(ts_to_frame_path)
            annos_dir = os.path.join(self._outdir, "3d_label", "os1", str(traj))
            curr_anno_dict = None
            curr_frame = -1

            obj_track_indices = [obj["track"][0]["header"]["stamp"]["secs"] + 
                1e-9*obj["track"][0]["header"]["stamp"]["nsecs"] for obj in annos["tracks"]]
            sorted_obj_track_indices = np.argsort(obj_track_indices)
            sorted_annos_tracks = [annos["tracks"][i] for i in sorted_obj_track_indices]
            
            for object in sorted_annos_tracks:
                objtrack = object["track"][0]
                header = objtrack["header"]
                ts = header["stamp"]["secs"] + 1e-9*header["stamp"]["nsecs"]
                frame   = np.searchsorted(ts_to_frame_np, ts, side='left')
                pose    = find_closest_pose(frame_to_poses_np, ts)

                if curr_frame!=frame:
                    if curr_frame!=-1: 
                        #Save current frame dict
                        if not os.path.exists(annos_dir):
                            print("Annotation dir for traj %s frame %s does not exist, creating here %s" % 
                                (traj, frame, annos_dir))
                            os.makedirs(annos_dir)
                        print(curr_anno_dict)

                        self.write_anno_to_file(traj, frame, annos_dir, curr_anno_dict)

                    # Create new frame dict 
                    curr_frame = int(frame)
                    curr_anno_dict = copy.deepcopy(CODA_ANNOTATION_DICT)
                    curr_anno_dict["frame"] = curr_frame

                curr_anno_obj_dict = copy.deepcopy(CODA_ANNOTATION_OBJECT_DICT)
                objlabel = "Other" if objtrack["label"]=="" else objtrack["label"]
                curr_anno_obj_dict["instanceId"]= "%s:%s"%(objlabel, object["id"])
                curr_anno_obj_dict["classId"]   = objlabel
                
                # Note: Apply Pose Transform to WCS!
                quat = R.from_quat([ objtrack["rotation"]["x"], objtrack["rotation"]["y"], \
                    objtrack["rotation"]["z"], objtrack["rotation"]["w"] ])
                rot_mat = quat.as_matrix()

                bbox_pose = np.eye(4)
                bbox_pose[:3, :3]   = rot_mat
                bbox_pose[:3, 3]    = [ objtrack["translation"]["x"], objtrack["translation"]["y"], 
                    objtrack["translation"]["z"] ]

                gt_pose = np.eye(4)
                if self._use_wcs:
                    gt_pose[:3, :3] = R.from_quat([ pose[7], pose[4], pose[5], pose[6] ]).as_matrix()
                    gt_pose[:3, 3]  = pose[1:4]
                    
                trans_bbox_pose = gt_pose @ bbox_pose
                rot_vec = R.as_rotvec(R.from_matrix(trans_bbox_pose[:3, :3]), degrees=False)

                #Translation
                curr_anno_obj_dict["cX"], curr_anno_obj_dict["cY"], curr_anno_obj_dict["cZ"] = \
                    trans_bbox_pose[0, 3], trans_bbox_pose[1, 3], trans_bbox_pose[2, 3]

                #Rotation
                curr_anno_obj_dict["r"], curr_anno_obj_dict["p"], curr_anno_obj_dict["y"] = \
                    rot_vec[0], rot_vec[1], rot_vec[2]

                #Size
                curr_anno_obj_dict["l"], curr_anno_obj_dict["w"], curr_anno_obj_dict["h"] = \
                    objtrack["box"]["length"], objtrack["box"]["width"], objtrack["box"]["height"]

                curr_anno_dict["3dannotations"].append(
                    curr_anno_obj_dict
                )
            
            # Handle last frame edge case
            if curr_anno_dict!=CODA_ANNOTATION_DICT:
                self.write_anno_to_file(traj, frame, annos_dir, curr_anno_dict)

    def write_anno_to_file(self, traj, frame, annos_dir, anno_dict):
        frame_dict_path = os.path.join(annos_dir,
        set_filename_by_prefix("3d_label", "os1", str(traj), str(frame)) )
        print("Saving frame %s annotations to %s " % (frame, frame_dict_path))
        frame_dict_file = open(frame_dict_path, "w+")
        frame_dict_file.write(json.dumps(anno_dict, indent=2))
        frame_dict_file.close()

    def save_anno_json(self, anno_dict):
        subdir  = anno_dict["subdir"].replace("raw", "label")
        modality, sensor_name   = subdir.split('/')[0], subdir.split('/')[1]
        traj    = anno_dict["trajectory"]
        anno_dir = os.path.join(self._outdir, subdir, traj)

        if not os.path.exists(anno_dir):
            print("Annotation path %s does not exist, creating now..."%anno_dir)
            os.makedirs(anno_dir)

        #Write each annotated frame object to json file
        for annotation in anno_dict["annotations"]:
            frame   = annotation["frame"]
            anno_filename   = "%s_%s_%s_%s.json"%(modality, sensor_name, traj, frame)
            anno_path = os.path.join(anno_dir, anno_filename)

            anno_json = open(anno_path, "w+")
            anno_json.write(json.dumps(annotation, indent=2))
            anno_json.close()

    def sagemaker_decoder(self, anno_subdir, anno_files, manifest_file):
        anno_dict = {
            "annotations": []
        }
        for anno_filename in anno_files:
            anno_path = os.path.join(self._indir, anno_subdir, anno_filename)
            mani_path = os.path.join(self._indir, manifest_file)

            mani_file   = open(mani_path, "r")
            anno_file   = open(anno_path, "r")
            anno_json   = json.load(anno_file)
            mani_json   = json.load(mani_file)

            # Recurse through all keys in json dictionary
            # Use dictionary mapping to decide what data to copy
            # TODO: Add ability to handle streaming jobs from AWS with multiple trajectories
            labeling_job_name = mani_json["answers"][0]["answerContent"] \
                ["trackingAnnotations"]["frameData"]["s3Prefix"].split('/')[4]
            labeling_job_name = labeling_job_name.split('-')
            tracking_annotations = anno_json['trackingAnnotations']
            for frame_dict in tracking_annotations:
                #Copy over existing annotation data
                curr_dict = self.recurs_dict(frame_dict)

                #Add additional annotation information
                curr_dict["filetype"]   = curr_dict["frame"].split('.')[1]
                curr_dict["frame"]      = curr_dict["frame"].split('.')[0].split('_')[-1]
                
                #Infer modality from filetype
                if not "subdir" in anno_dict:
                    for (subdir, filetype) in SENSOR_DIRECTORY_FILETYPES.items():
                        if filetype==curr_dict["filetype"]:
                            anno_dict["subdir"] = subdir
                anno_dict["annotations"].append(curr_dict)

            anno_dict["trajectory"] = labeling_job_name[1]    
            anno_dict["sensor"]     = labeling_job_name[2]
        
        return anno_dict

    def recurs_dict(self, sub_dict_level):
        curr_dict = []
        
        level_keys = [idx[0] for idx in enumerate(sub_dict_level)]
        if isinstance(sub_dict_level, dict):
            level_keys = [ *sub_dict_level.keys() ]
            curr_dict = {}

        for key in level_keys:
            curr_level = sub_dict_level[key]
            try:            
                if key in SAGEMAKER_TO_COMMON_ANNO or isinstance(key, int):
                    key_map = key
                    if key in SAGEMAKER_TO_COMMON_ANNO:
                        key_map = SAGEMAKER_TO_COMMON_ANNO[key]

                    if isinstance(curr_level, dict) or \
                        isinstance(curr_level, list):
                        if isinstance(curr_dict, dict):
                            curr_dict[key_map] = self.recurs_dict(curr_level)
                        else:
                            curr_dict.append(self.recurs_dict(curr_level))
                    else:
                        curr_dict[key_map] = curr_level
            except Exception as e:
                print("Recursive key parsing exception on %s" % key)
        return curr_dict
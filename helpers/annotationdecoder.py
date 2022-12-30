import os
import yaml
import glob
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
        
        settings_fp = os.path.join(os.getcwd(), "config/decoder.yaml")
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

                    manifest_file = dir_item + ".json"

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
        elif self._anno_type=="deepen":
            print("Decoding deepen annotations...")
            self.deepen_decoder(self._indir)
        else:
            print("Undefined annotation format type specified, exiting...")

    def deepen_decoder(self, datadir):
        tred_dir = os.path.join(datadir, "3d_label")
        assert os.path.exists(tred_dir), "3d_label directory does not exist in %s"%tred_dir

        tred_subdirs = next(os.walk(tred_dir))[1]
        
        if "bbox" in tred_subdirs:
            bbox_subdir = os.path.join(tred_dir, "bbox")
            traj_subdirs   = next(os.walk(bbox_subdir))[1] 

            for traj in traj_subdirs:
                traj_dir = os.path.join(bbox_subdir, traj)
                anno_dict = self.deepen_decode_bbox(traj_dir, traj)
                if self._gen_data:
                    self.save_anno_json(anno_dict)

        if "semantic" in tred_subdirs:
            semantic_subdir = os.path.join(tred_dir, "semantic")
            traj_subdirs   = next(os.walk(semantic_subdir))[1]

            for traj in traj_subdirs:
                traj_dir = os.path.join(semantic_subdir, traj)
                
                start_frames = [ frame.split('.')[0] for frame in os.listdir(traj_dir) if frame.endswith(".dpn") ]
                for start_frame in start_frames:
                    anno_dict = self.deepen_decode_semantic(traj_dir, traj, start_frame)

    def deepen_decode_bbox(self, filepath, traj):
        anno_dict = {
            "tredannotations": []
        }

        label_str = DEEPEN_LABEL_STR % traj
        label_path  = os.path.join(filepath, label_str)
        anno_file   = open(label_path, "r")

        anno_json               = json.load(anno_file)
        anno_dict["trajectory"] = traj   
        anno_dict["sensor"]     = "os1"

        ts_to_frame_path = os.path.join(self._outdir, "timestamps", "%s_frame_to_ts.txt"%anno_dict["trajectory"])
        pose_path   = os.path.join(self._outdir, "poses", "%s.txt"%anno_dict["trajectory"])
        pose_np     = np.loadtxt(pose_path, dtype=np.float64).reshape(-1, 8)
        ts_np       = np.loadtxt(ts_to_frame_path)

        dense_pose_np   = densify_poses_between_ts(pose_np, ts_np)

        for frame_str in anno_json["labels"].keys():
            if frame_str == "dataset_attributes":
                continue
            frame   = frame_str.split(".")[0].lstrip("0") or "0"
            filetype = frame_str.split(".")[1]

            frame_anno = anno_json["labels"][frame_str]
            #Preprocess annotation data to not have duplicate keys 
            key_map = {"x": "qx", "y": "qy", "z": "qz", "w": "qw"}
            for (idx, anno) in enumerate(frame_anno):
                old_items = frame_anno[idx]["three_d_bbox"]["quaternion"].items()
                frame_anno[idx]["three_d_bbox"]["quaternion"] = {
                    key_map[key] if key in key_map.keys() else key:v for key,v in old_items }

            #Copy over existing annotation data
            pc_list = self.recurs_dict(frame_anno, DEEPEN_TO_COMMON_ANNO)

            #Reformat quat to euler
            for (pc_idx, pc_anno) in enumerate(pc_list):
                qx, qy, qz, qw = pc_list[pc_idx].pop("qx"), pc_list[pc_idx].pop("qy"), \
                    pc_list[pc_idx].pop("qz"), pc_list[pc_idx].pop("qw")
                euler = R.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=False)
                pc_list[pc_idx]["r"], pc_list[pc_idx]["p"], pc_list[pc_idx]["y"] = euler
                
            pc_dict = {"3dbbox": pc_list}
            #Copy frame and filetype over from existing data
            pc_dict["filetype"]   = "json"
            pc_dict["frame"]      = frame

            #Infer modality from filetype
            if not "subdir" in anno_dict:
                for (subdir, filetype) in SENSOR_DIRECTORY_FILETYPES.items():
                    if filetype==pc_dict["filetype"]:
                        anno_dict["subdir"] = subdir

            if self._use_wcs:
                frame       = int(pc_dict["frame"])
                pose        = dense_pose_np[frame]
                pose_mat    = pose_to_homo(pose)
                #Convert bbox centers to ego frame
                for index, anno in enumerate(pc_dict["3dbbox"]):
                    wcs_to_ego = np.linalg.inv(pose_mat)
                    new_anno = bbox_transform(anno, wcs_to_ego)
                    pc_dict["3dbbox"][index] = new_anno

            #Save annotations in dictionary
            anno_dict["tredannotations"].append(pc_dict)
    
        return anno_dict

    def deepen_decode_semantic(self, filepath, traj, start_frame):
        """
        Currently assumes a each file uses 10Hz rate
        """
        sem_str = DEEPEN_SEMANTIC_STR % start_frame
        sem_path  = os.path.join(filepath, sem_str)
        meta_path   = sem_path.replace(".dpn", ".json")
        
        assert os.path.exists(sem_path), "Label path %s does not exist"%sem_path
        assert os.path.exists(meta_path), "Meta path %s does not exist"%meta_path

        meta_file = yaml.safe_load(open(meta_path, "r"))
        # Check for exact match between metadata classes and constants
        assert set(SEM_ID_TO_CLASS)!=set(meta_file['paint_categories']), \
            "metadata classes inconsistent with constants"

        sem_file = open(sem_path, "rb").read()
        pc_size = np.prod(OS1_POINTCLOUD_SHAPE[:2])
        sem_np = np.frombuffer(sem_file, dtype=np.uint8).reshape(-1, pc_size)

        outdir = os.path.join(self._outdir, DIR_SEMANTIC_LABEL, "os1", traj)
        if not os.path.exists(outdir):
            print("3D Semantic directory %s does not exist, creating..." % outdir)
            os.makedirs(outdir)

        deepen_id_to_coda_id = np.array([SEM_ID_TO_CLASS.index(cat) for cat in meta_file['paint_categories']], np.uint8)
        deepen_id_to_coda_id = np.insert(deepen_id_to_coda_id, 0, 0)

        num_frames = sem_np.shape[0]
        start_frame = int(start_frame)
        for frame_idx in range(0, num_frames):
            frame = start_frame + frame_idx
            filename = set_filename_by_prefix(DIR_SEMANTIC_LABEL, "os1", traj, str(frame))
            frame_path = os.path.join(outdir, filename)

            frame_label_np = sem_np[frame_idx].reshape(-1,).astype(np.uint8)
            frame_label_np = deepen_id_to_coda_id[frame_label_np]
            print("Saving 3d semantic traj %s frame %s to %s"%(traj, frame, frame_path))
            frame_label_np.tofile(frame_path)

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
                rot_vec = R.as_euler(R.from_matrix(trans_bbox_pose[:3, :3]), degrees=False)

                #Translation
                curr_anno_obj_dict["cX"], curr_anno_obj_dict["cY"], curr_anno_obj_dict["cZ"] = \
                    trans_bbox_pose[0, 3], trans_bbox_pose[1, 3], trans_bbox_pose[2, 3]

                #Rotation
                curr_anno_obj_dict["r"], curr_anno_obj_dict["p"], curr_anno_obj_dict["y"] = \
                    rot_vec[0], rot_vec[1], rot_vec[2]

                #Size
                curr_anno_obj_dict["l"], curr_anno_obj_dict["w"], curr_anno_obj_dict["h"] = \
                    objtrack["box"]["length"], objtrack["box"]["width"], objtrack["box"]["height"]

                curr_anno_dict["3dbbox"].append(
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
        subdir  = anno_dict["subdir"].replace(DIR_3D_RAW, DIR_BBOX_LABEL)
        modality, sensor_name   = subdir.split('/')[0], subdir.split('/')[1]
        traj    = anno_dict["trajectory"]
        anno_dir = os.path.join(self._outdir, subdir, traj)

        if not os.path.exists(anno_dir):
            print("Annotation path %s does not exist, creating now..."%anno_dir)
            os.makedirs(anno_dir)

        #Write each annotated frame object to json file
        for annotation in anno_dict["tredannotations"]:
            frame   = annotation["frame"]
            anno_filename   = "%s_%s_%s_%s.json"%(modality, sensor_name, traj, frame)
            anno_path = os.path.join(anno_dir, anno_filename)

            print("Writing 3D bbox traj %s frame %s to %s..."%(traj, frame, anno_path))
            anno_json = open(anno_path, "w+")
            anno_json.write(json.dumps(annotation, indent=2))
            anno_json.close()
            
    def sagemaker_decoder(self, anno_subdir, anno_files, manifest_file):
        anno_dict = {
            "tredannotations": [],
            "twodannotations": []
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
            anno_dict["trajectory"] = labeling_job_name[1]    
            anno_dict["sensor"]     = labeling_job_name[2]

            ts_to_frame_path = os.path.join(self._outdir, "timestamps", "%s_frame_to_ts.txt"%anno_dict["trajectory"])
            pose_path   = os.path.join(self._outdir, "poses", "%s.txt"%anno_dict["trajectory"])
            pose_np     = np.loadtxt(pose_path, dtype=np.float64).reshape(-1, 8)
            ts_np       = np.loadtxt(ts_to_frame_path)
            frame       = int(anno_filename.split("_")[4].split(".")[0])

            ts          = ts_np[frame]
            pose        = find_closest_pose(pose_np, ts)
            pose_mat    = np.eye(4)
            pose_mat[:3, :3]    = R.from_quat([pose[5], pose[6], pose[7], pose[4]]).as_matrix()
            pose_mat[:3, 3]     = np.array([pose[1], pose[2], pose[3]]) 

            tracking_annotations = anno_json['trackingAnnotations']
            for frame_dict in tracking_annotations:
                #Copy over existing annotation data
                pc_dict = self.recurs_dict(frame_dict)

                #Add additional annotation information
                pc_dict["filetype"]   = pc_dict["frame"].split('.')[1]
                pc_dict["frame"]      = pc_dict["frame"].split('.')[0].split('_')[-1]

                #Infer modality from filetype
                if not "subdir" in anno_dict:
                    for (subdir, filetype) in SENSOR_DIRECTORY_FILETYPES.items():
                        if filetype==pc_dict["filetype"]:
                            anno_dict["subdir"] = subdir

                if self._use_wcs:
                    #Convert bbox centers to ego frame
                    for index, anno in enumerate(pc_dict["3dbbox"]):
                        wcs_to_ego = np.linalg.inv(pose_mat)
                        new_anno = bbox_transform(anno, wcs_to_ego)
            
                        pc_dict["3dbbox"][index] = new_anno

                #Save annotations in dictionary
                anno_dict["tredannotations"].append(pc_dict)
        
        return anno_dict

    def recurs_dict(self, sub_dict_level, src_to_common_map=SAGEMAKER_TO_COMMON_ANNO):
        curr_dict = []
        
        level_keys = [idx[0] for idx in enumerate(sub_dict_level)]
        if isinstance(sub_dict_level, dict):
            level_keys = [ *sub_dict_level.keys() ]
            curr_dict = {}

        for key in level_keys:
            curr_level = sub_dict_level[key]
            try:            
                if key in src_to_common_map or isinstance(key, int):
                    key_map = key
                    if key in src_to_common_map:
                        key_map = src_to_common_map[key]

                    if isinstance(curr_level, dict) or \
                        isinstance(curr_level, list):

                        new_keys = self.recurs_dict(curr_level, src_to_common_map)
                        if isinstance(curr_dict, dict):
                            if key_map==".":
                                curr_dict.update(new_keys)
                            else:
                                curr_dict[key_map] = new_keys
                        else:
                            curr_dict.append(self.recurs_dict(curr_level, src_to_common_map))
                    else:
                        curr_dict[key_map] = curr_level
            except Exception as e:
                print("Recursive key parsing exception on %s" % key)
        return curr_dict
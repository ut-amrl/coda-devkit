import os
import pdb
from tkinter import Frame
import yaml
import json

import sys
# For imports
sys.path.append(os.getcwd())

import numpy as np

from helpers.constants import SENSOR_DIRECTORY_FILETYPES, SAGEMAKER_TO_COMMON_ANNO

class AnnotationDecoder(object):

    def __init__(self):
        
        settings_fp = os.path.join(os.getcwd(), "config/sagedecoder.yaml")
        with open(settings_fp, 'r') as settings_file:
            settings = yaml.safe_load(settings_file)
            self._indir   = settings['annotations_input_root']
            self._outdir  = settings['annotations_output_root']
            self._anno_type = settings['annotation_type']
            self._gen_data  = settings['gen_data']

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
        else:
            print("Undefined annotation format type specified, exiting...")

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
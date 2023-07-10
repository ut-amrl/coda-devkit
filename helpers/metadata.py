"""
Metadata File
"""
import json
from enum import Enum

from helpers.constants import TRED_BBOX_LABEL_DIR, SEMANTIC_LABEL_DIR

class METALighting(str, Enum):
    DARK='dark'
    NORMAL='normal'
    BRIGHT='bright'

class METASetting(str, Enum):
    OUTDOOR='outdoor'
    INDOOR='indoor'

OBJECT_DETECTION_TASK="ObjectTracking"
SEMANTIC_SEGMENTATION_TASK="SemanticSegmentation"

METADATA_DICT = {
    "date": "",     # Manual
    "operator": "", 
    "lighting": METALighting.NORMAL,
    "setting": METASetting.OUTDOOR,
    "objects": [],     # Each entry should be OBJECT (STR): COUNT (INT)
    "attributes": [],   # Flexible human readable category
    "waypoints": {},     # Each entry should be WAYPOINT (STR): TIMESTAMP (STR)
    "trajectory": 0,    # Path to trajectory file
    "poses": "",        # Path to poses file
    OBJECT_DETECTION_TASK: {
        "training": [],
        "validation": [],
        "testing": []
    },
    SEMANTIC_SEGMENTATION_TASK: {
        "training": [],
        "validation": [],
        "testing": []
    },
    "SLAM": {
        "training": [],
        "testing": []
    }
}

SENSOR_DIRECTORY_TO_TASK = {
    "%s/os1"%TRED_BBOX_LABEL_DIR: OBJECT_DETECTION_TASK,
    "%s/os1"%SEMANTIC_LABEL_DIR: SEMANTIC_SEGMENTATION_TASK,
}

MODALITY_TO_TASK = {
    TRED_BBOX_LABEL_DIR: OBJECT_DETECTION_TASK,
    SEMANTIC_LABEL_DIR: SEMANTIC_SEGMENTATION_TASK
}

def read_metadata_anno(metadata_path, modality="3d_bbox", split="all"):
    metafile = open(metadata_path, "r")
    metajson = json.load(metafile)

    assert modality in MODALITY_TO_TASK.keys(), "Modality %s is not defined... " % modality
    task = MODALITY_TO_TASK[modality]

    task_dict = metajson[task]
    task_filepaths = []
    if split=="all":
        for split, splitpaths in task_dict.items():
            task_filepaths.extend(splitpaths)
    else:
        assert split in task_dict.keys(), "Split %s not in task splits"%split
        task_filepaths.extend(task_dict[split])

    return task_filepaths

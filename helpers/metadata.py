"""
Metadata File
"""
from enum import Enum

class METALighting(str, Enum):
    DARK='dark'
    NORMAL='normal'
    BRIGHT='bright'

class METASetting(str, Enum):
    OUTDOOR='outdoor'
    INDOOR='indoor'

METADATA_DICT = {
    "date": "",     # Manual
    "operator": "", 
    "lighting": METALighting.NORMAL,
    "setting": METASetting.OUTDOOR,
    "objects:": {},     # Each entry should be OBJECT (STR): COUNT (INT)
    "attributes": [],   # Flexible human readable category
    "waypoints": {},     # Each entry should be WAYPOINT (STR): TIMESTAMP (STR)
    "trajectory": 0,    # Path to trajectory file
    "poses": "",        # Path to poses file
    "ObjectTracking": {
        "training": [],
        "validation": [],
        "testing": []
    },
    "SLAM": {
        "training": [],
        "testing": []
    }
}
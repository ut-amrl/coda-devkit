from sensor_msgs.msg import PointField


"""
SENSOR PROCESSING CONSTANTS
"""
_DATATYPES = {}
_DATATYPES[PointField.INT8]    = ('b', 1)
_DATATYPES[PointField.UINT8]   = ('B', 1)
_DATATYPES[PointField.INT16]   = ('h', 2)
_DATATYPES[PointField.UINT16]  = ('H', 2)
_DATATYPES[PointField.INT32]   = ('i', 4)
_DATATYPES[PointField.UINT32]  = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)

OS1_PACKETS_PER_FRAME = 64

"""
DATASET PARAMETER CONSTANTS
"""
DATASET_L1_DIR_LIST = [
    "metadata",
    "calibration",
    "timestamps",
    "2d_raw",
    "3d_raw",
    "poses"
]

"""
Manifest file generation sensor to subdirectory mappings
"""
SENSOR_DIRECTORY_SUBPATH = {
    "/ouster/lidar_packets": "3d_raw/os1",
    "/stereo/left/image_raw/compressed": "2d_raw/cam0",
    "/stereo/right/image_raw/compressed": "2d_raw/cam1",
}

SENSOR_DIRECTORY_FILETYPES = {
    "3d_raw/os1": "bin",
    "2d_raw/cam0": "png",
    "2d_raw/cam1": "png",
}

"""
Sensor Hardware Constants
"""
SENSOR_TO_WCS = {
    "/ouster/lidar_packets": [0, 0, 0],
    "/vectornav/IMU": [0, 0, 180]
}

"""
Manifest Autogeneration
"""
SEQ_TEXT = '{"seq-no": %d,\n'

PREFIX_TEXT = '"prefix": "%s",\n'

NUM_FRAMES_TEXT = '"number-of-frames": %d,\n'

FRAMES_START_TEXT = '"frames": [\n'

FRAME_TEXT_DICT = {
    "frameno":  0, 
    "ts":       0.0, 
    "frame":    "", 
    "evppx":    0.0, 
    "evppy":    0.0, 
    "evppz":    0.0, 
    "evphx":    0.0, 
    "evphy":    0.0, 
    "evphz":    0.0, 
    "evphw":    0.0, 
    "ipath":    "",
    "its":       0.0,
    "fx":       0.0,
    "fy":       0.0, 
    "cx":       0.0, 
    "cy":       0.0, 
    "ipx":      0.0, 
    "ipy":      0.0, 
    "ipz":      0.0,
    "ihx":      0.0,
    "ihy":      0.0, 
    "ihz":      0.0, 
    "ihw":      0.0
}

FRAME_TEXT = '''    {
        "frame-no": %(frameno)d,
        "unix-timestamp": %(ts)0.10f,
        "frame": "%(frame)s", 
        "format": "binary/xyz", 
        "ego-vehicle-pose":{
            "position": {
                "x": %(evppx)0.10f,
                "y":  %(evppy)0.10f,
                "z":  %(evppz)0.10f
            },
            "heading": {
                "qx":  %(evphx)0.10f,
                "qy":  %(evphy)0.10f,
                "qz":  %(evphz)0.10f,
                "qw":  %(evphw)0.10f
            }
        },
        "images": [{
            "image-path": "%(ipath)s",
            "unix-timestamp":  %(ts)0.10f,
            "fx":  %(fx)0.10f,
            "fy":  %(fy)0.10f,
            "cx":  %(cx)0.10f,
            "cy":  %(cy)0.10f,
            "k1": 0,
            "k2": 0,
            "k3": 0,
            "k4": 0,
            "p1": 0,
            "p2": 0,
            "skew": 0,
            "position": {
                "x":  %(ipx)0.10f,
                "y":  %(ipy)0.10f,
                "z":  %(ipz)0.10f
            },
            "heading": {
                "qx":  %(ihx)0.10f,
                "qy":  %(ihy)0.10f,
                "qz":  %(ihz)0.10f,
                "qw":  %(ihw)0.10f
            },
            "camera-model": "pinhole"
        }]
    }'''

FRAMES_END_TEXT = ']}'
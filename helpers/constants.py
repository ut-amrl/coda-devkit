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

SENSOR_DIRECTORY_SUBPATH = {
    "/ouster/lidar_packets": "3d_raw/os1",
    "/stereo/left/image_raw/compressed": "2d_raw/cam0",
    "/stereo/right/image_raw/compressed": "2d_raw/cam1",
}


from sensor_msgs.msg import PointField

"""
VISUALIZATION SETTINGS
"""
SEM_POINT_SIZE = 2 # pixels

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
OS1_POINTCLOUD_SHAPE    = [1024, 128, 3]

"""
DATASET PARAMETER CONSTANTS
"""
TRED_RAW_DIR            = "3d_raw"
TRED_COMP_DIR           = "3d_comp"
SEMANTIC_LABEL_DIR      = "3d_semantic"
TRED_BBOX_LABEL_DIR     = "3d_bbox"
TWOD_BBOX_LABEL_TYPE    = "2d_bbox"
METADATA_DIR            = "metadata"
CALIBRATION_DIR         = "calibrations"
TWOD_RAW_DIR            = "2d_raw"
TWOD_PROJ_DIR           = "2d_proj"

DATASET_L1_DIR_LIST = [
    METADATA_DIR,
    CALIBRATION_DIR,
    "timestamps",
    TWOD_RAW_DIR,
    "2d_rect",
    TRED_RAW_DIR,
    TRED_COMP_DIR,
    TRED_BBOX_LABEL_DIR,
    TWOD_BBOX_LABEL_TYPE,
    SEMANTIC_LABEL_DIR,
    TWOD_PROJ_DIR,
    "poses"
]

DATASET_L2_DIR_LIST = [
    "poses/imu",
    "poses/gps",
    "poses/mag",
    "poses/gpsodom",
    "poses/inekfodom"
]

"""
Annotation Mappings
"""
SAGEMAKER_TO_COMMON_ANNO = {
    #Frame Specific
    "frameName": "frame",
    "boundingCuboids": "3dbbox",
    "frameAttributes": "frameAttributes",
    # Object Specific 
    "objectName": "instanceId",
    "label": "classId",
    "centerX": "cX",
    "centerY": "cY",
    "centerZ": "cZ",
    "length": "l",
    "width": "w",
    "height": "h",
    "roll": "r",
    "pitch": "p",
    "yaw": "y",
    "labelCategoryAttributes": "labelCategoryAttributes",
    "Occlusion Status": "isOccluded",
    "Nav Behavior": "behavior"
}

"""
BBOX CLASS CONSTANTS
"""
BBOX_CLASS_VIZ_LIST = [  
    # Dynamic Classes
    "Car"                   ,
    "Pedestrian"            ,
    "Bike"                  ,
    "Motorcycle"            ,
    "Golf Cart"             ,
    "Truck"                 ,
    "Scooter"               ,
    # Static Classes
    "Tree"                  ,
    "Traffic Sign"          ,
    "Canopy"                ,
    "Traffic Light"         ,
    "Bike Rack"             ,
    "Bollard"               ,
    "Construction Barrier"  ,
    "Parking Kiosk"         ,
    "Mailbox"               ,
    "Fire Hydrant"          ,
    # Static Class Mixed
    "Freestanding Plant"    ,
    "Pole"                  ,
    "Informational Sign"    ,
    "Door"                  ,
    "Fence"                 ,
    "Railing"               ,
    "Cone"                  ,
    "Chair"                 ,
    "Bench"                 ,
    "Table"                 ,
    "Trash Can"             ,
    "Newspaper Dispenser"   ,
    # Static Classes Indoor
    "Room Label"            ,
    "Stanchion"             ,
    "Sanitizer Dispenser"   ,
    "Condiment Dispenser"   ,
    "Vending Machine"       ,
    "Emergency Aid Kit"     ,
    "Fire Extinguisher"     ,
    "Computer"              ,
    "Television"            ,
    "Other"                 ,
    "Horse"                 ,
    # New Classes
    "Pickup Truck"          ,
    "Delivery Truck"        ,
    "Service Vehicle"       ,
    "Utility Vehicle"       ,
    "Fire Alarm"            ,
    "ATM"                   ,
    "Cart"                  ,
    "Couch"                 ,
    "Traffic Arm"           ,
    "Wall Sign"             ,
    "Floor Sign"            ,
    "Door Switch"           ,
    "Emergency Phone"       ,
    "Dumpster"              ,
    "Vacuum Cleaner"        ,
    "Segway"                ,
    "Bus"                   ,
    "Skateboard"
]

BBOX_CLASS_REMAP = {
    "Scooter":              "Scooter",
    "Bike":                 "Bike",
    "Car":                  "Vehicle",
    "Motorcycle":           "Motorcycle",
    "Golf Cart":            "Vehicle",
    "Truck":                "Vehicle",
    "Pedestrian":            "Pedestrian",
    "Segway":               "WheeledBoard",
    "Bus":                  "Vehicle",
    "Skateboard":           "WheeledBoard",
    # Static Classes     
    "Tree":                 "Tree",
    "Traffic Sign":         "Sign",
    "Canopy":               "Canopy",
    "Traffic Light":       "Traffic Light",
    "Bike Rack":            "Bike Rack",
    "Bollard":              "Barrier",
    "Construction Barrier": "Barrier",
    "Parking Kiosk":        "Dispenser",
    "Mailbox":              "Dispenser",
    "Fire Hydrant":         "Fire Hydrant",
    # Static Class Mixed
    "Freestanding Plant":   "Plant",
    "Pole":                 "Pole",
    "Informational Sign":   "Sign",
    "Door":                 "Barrier",
    "Fence":                "Barrier",
    "Railing":              "Barrier",
    "Cone":                 "Cone",
    "Chair":                "Chair",
    "Bench":                "Bench",
    "Table":                "Table",
    "Trash Can":            "Trash Can",
    "Newspaper Dispenser":  "Dispenser",
    # Static Classes Indoor
    "Room Label":           "Sign",
    "Stanchion":            "Barrier",
    "Sanitizer Dispenser":  "Dispenser",
    "Condiment Dispenser":  "Dispenser",
    "Vending Machine":      "Dispenser",
    "Emergency Aid Kit":    "Dispenser",
    "Fire Extinguisher":    "Dispenser",
    "Computer":             "Screen",
    "Television":           "Screen",
    "Other":                "Other",
    "Horse":                "Other",
    "Pickup Truck":         "Vehicle",
    "Delivery Truck":       "Vehicle",
    "Service Vehicle":      "Vehicle",
    "Utility Vehicle":      "Vehicle",
    "Fire Alarm":           "Fire Alarm",
    "ATM":                  "Dispenser",
    "Cart":                 "Cart",
    "Couch":                "Couch",
    "Traffic Arm":          "Traffic Arm",
    "Wall Sign":            "Sign",
    "Floor Sign":           "Sign"
}

BBOX_CLASS_TO_ID = {
    # Dynamic Classes
    "Car"                   : 0,
    "Pedestrian"            : 1,
    "Bike"                  : 2,
    "Motorcycle"            : 3,
    "Golf Cart"             : 4,
    "Truck"                 : 5,
    "Scooter"               : 6,
    # Static Classes
    "Tree"                  : 7,
    "Traffic Sign"          : 8,
    "Canopy"                : 9,
    "Traffic Light"         : 10,
    "Bike Rack"             : 11,
    "Bollard"               : 12,
    "Construction Barrier"  : 13,
    "Parking Kiosk"         : 14,
    "Mailbox"               : 15,
    "Fire Hydrant"          : 16,
    # Static Class Mixed
    "Freestanding Plant"    : 17,
    "Pole"                  : 18,
    "Informational Sign"    : 19,
    "Door"                  : 20,
    "Fence"                 : 21,
    "Railing"               : 22,
    "Cone"                  : 23,
    "Chair"                 : 24,
    "Bench"                 : 25,
    "Table"                 : 26,
    "Trash Can"             : 27,
    "Newspaper Dispenser"   : 28,
    # Static Classes Indoor
    "Room Label"            : 29,
    "Stanchion"             : 30,
    "Sanitizer Dispenser"   : 31,
    "Condiment Dispenser"   : 32,
    "Vending Machine"       : 33,
    "Emergency Aid Kit"     : 34,
    "Fire Extinguisher"     : 35,
    "Computer"              : 36,
    "Television"            : 37,
    "Other"                 : 38,
    "Horse"                 : 39,
    # New Classes
    "Pickup Truck"          : 40,
    "Delivery Truck"        : 41,
    "Service Vehicle"       : 42,
    "Utility Vehicle"       : 43,
    "Fire Alarm"            : 44,
    "ATM"                   : 45,
    "Cart"                  : 46,
    "Couch"                 : 47,
    "Traffic Arm"           : 48,
    "Wall Sign"             : 49,
    "Floor Sign"            : 50,
    "Door Switch"           : 51,
    "Emergency Phone"       : 52,
    "Dumpster"              : 53,
    "Vacuum Cleaner"        : 54,
    "Segway"                : 55,
    "Bus"                   : 56,
    "Skateboard"            : 57,
    "Water Fountain"       : 58
}

OCCLUSION_TO_ID ={
    "None": 0,
    "Light": 1,
    "Medium": 2,
    "Heavy": 3,
    "Full": 4,
    "Unknown": 5,
    "unknown": 5
}

NONRIGID_CLASS_IDS = [6, 7]

BBOX_ID_TO_COLOR = [
    (255, 0, 0),       #0 Scooter (Red)
    (0, 255, 0),       #1 Person (Green)
    (0, 0, 255),       #2 Car (Blue)
    (255, 128, 0),     #3 Motorcycle (Orange)
    (255, 255, 0),     #4 Golf Cart (Yellow)
    (128, 0, 128),     #5 Truck (Purple)
    (0, 255, 255),     #6 Person (Cyan)
    (255, 204, 0),     #7 Tree (Gold)
    (204, 0, 0),       #8 Traffic Sign (Dark Red)
    (192, 192, 192),   #9 Canopy (Silver)
    (255, 255, 51),    #10 Traffic Lights (Lime)
    (255, 102, 255),   #11 Bike Rack (Pink)
    (128, 128, 128),   #12 Bollard (Gray)
    (255, 153, 102),   #13 Construction Barrier (Light Orange)
    (0, 0, 204),       #14 Parking Kiosk (Dark Blue)
    (0, 51, 204),      #15 Mailbox (Royal Blue)
    (255, 0, 0),       #16 Fire Hydrant (Red)
    (0, 204, 0),       #17 Freestanding Plant (Green)
    (255, 167, 49),     #18 Pole (Texan Orange)
    (255, 255, 0),     #19 Informational Sign (Yellow)
    (204, 51, 0),      #20 Door (Dark Orange)
    (102, 51, 0),      #21 Fence (Brown)
    (204, 102, 0),     #22 Railing (Orange)
    (255, 153, 51),    #23 Cone (Light Orange)
    (0, 204, 204),     #24 Chair (Turquoise)
    (0, 51, 0),        #25 Bench (Dark Green)
    (102, 102, 0),     #26 Table (Olive)
    (255, 87, 51),      #27 Trash Can (Bright Blue)
    (255, 204, 153),   #28 Newspaper Dispenser (Light Orange)
    (255, 51, 255),    #29 Room Label (Magenta)
    (224, 224, 224),   #30 Stanchion (Light Gray)
    (51, 255, 255),    #31 Sanitizer Dispenser (Turquoise)
    (76, 153, 0),      #32 Condiment Dispenser (Dark Green)
    (51, 152, 255),    #33 Vending Machine (Sky Blue)
    (255, 204, 204),   #34 Emergency Aid Kit (Light Pink)
    (255, 102, 102),   #35 Fire Extinguisher (Light Red)
    (0, 153, 76),      #36 Computer (Dark Green)
    (32, 32, 32),      #37 Television (Black)
    (255, 255, 255),   #38 Other (White)
    # Temp new classes
    (255, 102, 0),     #39 Horse (Orange)
    (0, 204, 255),     #40 Pickup Truck (Sky Blue)
    (255, 16, 240),    #41 Delivery Truck (Neon Pink)
    (255, 255, 51),    #42 Service Vehicle (Lime)
    (0, 128, 0),       #43 Utility Vehicle (Green)
    (51, 0, 204),      #44 Fire Alarm (Blue)
    (255, 204, 204),   #45 ATM (Light Pink)
    (255, 102, 102),   #46 Cart (Light Red)
    (0, 153, 76),      #47 Couch (Dark Green)
    (32, 32, 32),      #48 Traffic Arm (Black)
    (255, 255, 255),   #49 Wall Sign (White)
    (255, 102, 102),   #50 Floor Sign (Light Red)
    (0, 153, 76),      #51 Door Switch (Dark Green)
    (32, 32, 32),      #52 Emergency Phone (Light Black)
    (255, 255, 255),   #53 Dumpster (White)
    (200, 200, 200),   #54 Vacuum Cleaner (Dark Gray)
    (223, 32, 32),     #55 Segway
    (255, 200, 255),   #56 Bus
    (200, 200, 105),    #57 Scooter,
    (100, 200, 90)      #58 WAter Fountain 
    #TODO ADD ADDITIONAL COLORS FOR NEW CLASSES
]

"""
TERRAIN SEMANTIC CLASS CONSTANTS
"""

SEM_CLASS_TO_ID = {
    "Unlabeled":            0,
    "Concrete":             1,
    "Grass":                2,
    "Rocks":                3,
    "Speedway Bricks":      4,
    "Red Bricks":           5,
    "Pebble Pavement":      6,
    "Light Marble Tiling":  7,
    "Dark Marble Tiling":   8,
    "Dirt Paths":           9,
    "Road Pavement":        19,
    "Short Vegetation":     11,
    "Porcelain Tile":       12,
    "Metal Grates":         13,
    "Blond Marble Tiling":  14,
    "Wood Panel":           15,
    "Patterned Tile":       16,
    "Carpet":               17,
    "Crosswalk":            18,
    "Dome Mat":             19,
    "Stairs":               20,
    "Door Mat":             21,
    "Threshold":            22,
    "Metal Floor":          23,
    "Unknown":              24
}

SEM_ID_TO_COLOR = [
    [0, 0, 0],              # 0 Unknown
    [47, 171, 97],          # 1 Concrete
    [200, 77, 159],        # 2 Grass
    [126, 49, 141],          # 3 Rocks
    [55, 128, 235],         # 4 Speedway Bricks
    [8, 149, 174],         # 5 Red Bricks
    [141, 3, 98],        # 6 Pebble Pavement
    [203, 110, 74],        # 7 Light Marble Tiling
    [49, 240, 115],          # 8 Dark Marble Tiling
    [78, 57, 127],         # 9 Dirt Paths
    [60, 143, 142],          # 10 Road Pavement
    [187, 187, 17],        # 11 Short Vegetation
    [137, 247, 165],        # 12 Porcelain Tile
    [89, 183, 27],         # 13 Metal Grates
    [134, 29, 80],        # 14 Blond Marble Tiling
    [150, 81, 244],        # 15 Wood Panel
    [163, 77, 159],        # 16 Patterned Tile
    [60, 100, 116],         # 17 Carpet
    [156, 207, 153],         # 18 Crosswalk
    [135, 138, 159],        # 19 Dome Mat
    [44, 217, 131],        # 20 Stairs
    [123, 97, 131],        # 21 Door Mat
    [115, 226, 101],          # 22 Threshold
    [156, 43, 40],          # 23 Metal Floor
    [0, 0, 0]               # 24 Unlabeled
]

# [
#     # (B, G, R) - Object Name
#     (0, 0, 0),                    # 0 Unknown
#     (148, 60, 56),                # 1 Concrete
#     (185, 104, 114),              # 2 Grass
#     (64, 90, 138),                # 3 Rocks
#     (63, 159, 213),               # 4 Speedway Bricks
#     (179, 73, 137),               # 5 Red Bricks
#     (144, 146, 214),              # 6 Pebble Pavement
#     (210, 170, 206),              # 7 Light Marble Tiling
#     (117, 17, 17),                # 8 Dark Marble Tiling
#     (108, 26, 103),               # 9 Dirt Paths
#     (56, 60, 148),                # 10 Road Pavement
#     (178, 130, 221),              # 11 Short Vegetation
#     (171, 237, 180),              # 12 Porcelain Tile
#     (217, 59, 206),               # 13 Metal Grates
#     (241, 220, 197),              # 14 Blond Marble Tiling
#     (132, 164, 196),              # 15 Wood Panel
#     (161, 220, 226),              # 16 Patterned Tile
#     (121, 120, 56),               # 17 Carpet
#     (111, 110, 51),               # 18 Crosswalk
#     (212, 195, 210),              # 19 Dome Mat
#     (243, 253, 244),              # 20 Stairs
#     (221, 220, 106),              # 21 Door Mat
#     (60, 226, 93),                # 22 Threshold
#     (108, 17, 17),                # 23 Metal Floor
#     (0, 0, 0)                     # 24 Unlabeled
# ]

"""
Manifest file generation sensor to subdirectory mappings
"""
SENSOR_DIRECTORY_SUBPATH = {
    #Depth
    "/ouster/lidar_packets": "%s/os1"%TRED_RAW_DIR,
    "/camera/depth/image_raw/compressed": "%s/cam2"%TRED_RAW_DIR,
    "/zed/zed_node/depth/depth_registered": "%s/cam3"%TRED_RAW_DIR,
    #RGB
    "/stereo/left/image_raw/compressed": "2d_raw/cam0",
    "/stereo/right/image_raw/compressed": "2d_raw/cam1",
    "/camera/rgb/image_raw/compressed": "2d_raw/cam2",
    "/zed/zed_node/left/image_rect_color/compressed": "2d_raw/cam3",
    "/zed/zed_node/right/image_rect_color/compressed": "2d_raw/cam4",
    #Inertial
    "/vectornav/IMU": "poses/imu",
    "/vectornav/Mag": "poses/mag",
    "/vectornav/GPS": "poses/gps",
    "/vectornav/Odom": "poses/gpsodom",
    "/husky_velocity_controller/odom": "poses/inekfodom"
}

SENSOR_DIRECTORY_FILETYPES = {
    #Depth
    "%s/os1"%TRED_RAW_DIR: "bin",
    "%s/cam2"%TRED_RAW_DIR: "png",
    "%s/cam3"%TRED_RAW_DIR: "png",
    #RGB
    "2d_raw/cam0": "png",
    "2d_raw/cam1": "png",
    "2d_raw/cam2": "png",
    "2d_raw/cam3": "png",
    "2d_raw/cam4": "png",
    "2d_rect/cam0": "jpg",
    "2d_rect/cam1": "jpg",
    #Labels
    "%s/cam0"%TWOD_BBOX_LABEL_TYPE: "txt", # KITTI FORMAT
    "%s/cam1"%TWOD_BBOX_LABEL_TYPE: "txt",
    "%s/os1"%TRED_BBOX_LABEL_DIR: "json",
    "%s/os1"%SEMANTIC_LABEL_DIR: "bin",
    "%s/os1"%TRED_COMP_DIR: "bin",
    "%s/cam0"%TWOD_PROJ_DIR: "jpg",
    "%s/cam1"%TWOD_PROJ_DIR: "jpg",
    #Inertial
    "poses/imu": "txt",
    "poses/mag": "txt",
    "poses/gps": "txt",
    "poses/gpsodom": "txt",
    "poses/inekfodom": "txt"
}

"""
Sensor Hardware Constants
"""
# Converts destaggered OS1 point clouds to x forward, y left, z up convention

SENSOR_TO_XYZ_FRAME = {
    "/ouster/lidar_packets": [
        -1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, 1, 0.03618,
        0, 0, 0, 1      
    ],
    "/vectornav/IMU": [
        1, 0, 0,  0, 
        0, -1, 0, 0,
        0, 0, -1, 0,
        0, 0, 0, 1
    ],
    # "/vectornav/IMU": [
    #     0, 1, 0,  0, 
    #     -1, 0, 0, 0,
    #     0, 0, 1, 0,
    #     0, 0, 0, 1
    # ],
    "/vectornav/Odom": [
        -1, 0, 0,  0, 
        0, -1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]
}
SENSOR_TO_BASE_LINK = {
    "/ouster/lidar_packets": [
        -1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, 1, 0.43,
        0, 0, 0, 1       
    ],
    "/vectornav/IMU": [
        0, -1, 0, 0, 
        1, 0, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]
}

CAM0_CALIBRATIONS = {
    "distortion": [-0.071430, 0.036078, -0.004514, -0.000722, 0.000000],
    "camera": [
        670.95818,   0.     , 617.95349, 
        0.     , 673.34226, 518.92384,
        0.     ,   0.     ,   1.
    ],
    "projection": [
        681.99707,   0.     , 596.75923,   0,
        0.     , 681.99707, 503.14468,   0,
        0.     ,   0.     ,   1.     ,   0
    ],
    "extrinsics": [0.1125, -0.1937500000000003, -0.1906249999999992, -7, -96, 94],
    "width": 1224,
    "height": 1024
}

KITTI_INTRINSIC_FORMAT = {
    'S': [],
    'K': [],
    'D': [],
    'R': [],
    'T': [],
    'S_rect': [],
    'R_rect': [],
    'P_rect': []
}

"""
DeepenAI Generation
"""
DEEPEN_TO_COMMON_ANNO = {
    #Frame Specific
    # "file_id": "frame",
    # "labels": "3dbbox",
    "attributes": "labelAttributes",
    "three_d_bbox": ".",
    "quaternion": ".",
    # Object Specific 
    "label_id": "instanceId",
    "label_category_id": "classId",
    "cx": "cX",
    "cy": "cY",
    "cz": "cZ",
    "l": "l",
    "w": "w",
    "h": "h",
    "qx": "qx",
    "qy": "qy",
    "qz": "qz",
    "qw": "qw",
    "Occlusion ": "isOccluded"
}


DEEPEN_SEMANTIC_STR = "%s.dpn"

DEEPEN_IMAGE_PREFIX = """{
    "images": ["""

DEEPEN_IMAGE_SUFFIX_DICT = {
    "ts":   0.0,
    "dhx":  0.0,
    "dhy":  0.0,
    "dhz":  0.0, 
    "dhw":  0.0, 
    "dpx":  0.0,
    "dpy":  0.0,
    "dpz":  0.0
}

DEEPEN_IMAGE_SUFFIX = """
    ],
    "timestamp": %(ts)0.10f,
    "device_heading": {
        "x": %(dhx)0.10f,
        "y": %(dhy)0.10f,
        "z": %(dhz)0.10f,
        "w": %(dhw)0.10f
    },
    "device_position": {
        "x": %(dpx)0.10f,
        "y": %(dpy)0.10f,
        "z": %(dpz)0.10f
    },"""

DEEPEN_POINTS_PREFIX = """
    "points": [
"""

DEEPEN_POINTS_SUFFIX = """
    ]
}"""

DEEPEN_POINT_DICT = {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0
}
DEEPEN_POINT_ENTRY = '''        {
            "x": %(x)0.10f,
            "y": %(y)0.10f,
            "z": %(z)0.10f
        },
'''

DEEPEN_IMAGE_DICT = {
    "ipath":    "",
    "ts":       0.0,
    "fx":       0.0,
    "fy":       0.0, 
    "cx":       0.0, 
    "cy":       0.0,
    "k1":       0.0,
    "k2":       0.0,
    "k3":       0.0,
    "p1":       0.0,
    "p2":       0.0,
    "px":      0.0, 
    "py":      0.0, 
    "pz":      0.0,
    "hx":      0.0,
    "hy":      0.0, 
    "hz":      0.0, 
    "hw":      0.0
}

DEEPEN_IMAGE_ENTRY = '''
        {
            "timestamp": %(ts)0.10f,
            "fx":  %(fx)0.10f,
            "fy":  %(fy)0.10f,
            "cx":  %(cx)0.10f,
            "cy":  %(cy)0.10f,
            "k1": %(k1)0.10f,
            "k2": %(k2)0.10f,
            "k3": %(k3)0.10f,
            "p1": %(p1)0.10f,
            "p2": %(p2)0.10f,
            "image_url": "%(ipath)s",
            "position": {
                "x": %(px)0.10f,
                "y": %(py)0.10f,
                "z": %(pz)0.10f
            },
            "heading": {
                "x": %(hx)0.10f,
                "y": %(hy)0.10f,
                "z": %(hz)0.10f,
                "w": %(hw)0.10f
            },
            "camera_model": "pinhole"
        },'''

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
    "k1":       0.0,
    "k2":       0.0,
    "k3":       0.0,
    "k4":       0.0,
    "p1":       0.0,
    "p2":       0.0,
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
            "k1": %(k1)0.10f,
            "k2": %(k2)0.10f,
            "k3": %(k3)0.10f,
            "k4": %(k4)0.10f,
            "p1": %(p1)0.10f,
            "p2": %(p2)0.10f,
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

MANIFEST_PATH_DICT = {
    "manifest_prefix":  "", 
    "sequence_filename": ""
}

MANIFEST_PATH_STR = '''{"source-ref": "%(manifest_prefix)s/%(sequence_filename)s"}'''

# Kitti360 Autogeneration

KITTI360_INTRINSIC_XML = '''<?xml version="1.0"?><intrincis_perspective><fx>%(fx)0.10f</fx><fy>%(fy)0.10f</fy><ox>%(ox)0.10f</ox><oy>%(oy)0.10f</oy><width>%(w)i</width><height>%(h)i</height></intrincis_perspective>'''

CODA_ANNOTATION_DICT = {
    "frame": 0,
    "3dbbox": []
}

CODA_ANNOTATION_OBJECT_DICT = {
    "instanceId": "",
    "classId": "",
    "cX": 0.0,
    "cY": 0.0,
    "cZ": 0.0,
    "l": 0.0,
    "w": 0.0,
    "h": 0.0,
    "r": 0.0,
    "p": 0.0,
    "y": 0.0,
    "labelCategoryAttributes": {
        "isOccluded": "No"
    }
}

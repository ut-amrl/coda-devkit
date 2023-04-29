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
OS1_POINTCLOUD_SHAPE    = [1024, 128, 3]

"""
DATASET PARAMETER CONSTANTS
"""
SEMANTIC_LABEL_TYPE     = "3d_semantic"
TRED_BBOX_LABEL_TYPE    = "3d_bbox"
TWOD_BBOX_LABEL_TYPE    = "2d_bbox"

DATASET_L1_DIR_LIST = [
    "metadata",
    "calibrations",
    "timestamps",
    "2d_raw",
    "2d_rect",
    "3d_raw",
    TRED_BBOX_LABEL_TYPE,
    TWOD_BBOX_LABEL_TYPE,
    SEMANTIC_LABEL_TYPE,
    "poses"
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
    "Vacuum Cleaner"        
]

BBOX_CLASS_REMAP = {
    "Scooter":              "Scooter",
    "Bike":                 "Bike",
    "Car":                  "Vehicle",
    "Motorcycle":           "Motorcycle",
    "Golf Cart":            "Vehicle",
    "Truck":                "Vehicle",
    "Pedestrian":            "Pedestrian",
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
    "Vacuum Cleaner"        : 54
}

OCCLUSION_TO_ID ={
    "None": 0,
    "Light": 1,
    "Medium": 2,
    "Heavy": 3,
    "Full": 4,
    "Unknown": 5
}

NONRIGID_CLASS_IDS = [6, 7]

BBOX_ID_TO_COLOR = [
    (120, 190, 33),     #0 Scooter
    (128, 128, 128),    #1 Bike
    (255, 0, 0),        #2 Car
    (255, 128, 0),      #3 Motorcyle
    (102, 204, 0),      #4 Golf Cart
    (0, 0, 255),        #5 Truck
    (102, 102, 255),    #6 Person
    (51, 102, 0),       #7 Tree
    (153, 0, 0),        #8 Traffic Sign
    (192, 192, 192),    #9 Canopy
    (255, 255, 0),      #10 Traffic Lights
    (255, 102, 255),    #11 Bike Rack
    (96, 96, 96),       #12 Bollard
    (255, 178, 102),    #13 Construction Barrier
    (0, 0, 153),        #14 Parking Kiosk
    (0, 0, 204),        #15 Mailbox
    (255, 51, 51),      #16 Fire Hydrant
    (0, 204, 0),        #17 Freestanding Plant
    (160, 160, 160),    #18 Pole
    (255, 255, 253),    #19 Informational Sign
    (153, 76, 0),       #20 Door
    (102, 51, 0),       #21 Fence
    (204, 102, 0),      #22 Railing
    (255, 152, 51),     #23 Cone
    (102, 255, 255),    #24 Chair
    (51, 25, 0),        #25 Bench
    (102, 102, 0),      #26 Table
    (64, 64, 64),       #27 Trash Can
    (255, 204, 153),    #28 Newspaper Dispenser
    (255, 51, 255),     #29 Room Label
    (224, 224, 224),    #30 Stanchion
    (51, 255, 255),     #31 Sanitizer Dispenser
    (76, 153, 0),       #32 Condiment Dispenser
    (51, 152, 255),     #33 Vending Machine
    (255, 204, 204),    #34 Emergency Aid Kit
    (255, 102, 102),    #35 Fire Extinguisher
    (0, 153, 76),       #36 Computer
    (32, 32, 32),       #37 Television
    (255, 255, 255),     #38 Other
    # Temp new classes
    (255, 204, 153),    #39 Newspaper Dispenser
    (255, 51, 255),     #40 Room Label
    (224, 224, 224),    #41 Stanchion
    (51, 255, 255),     #42 Sanitizer Dispenser
    (76, 153, 0),       #43 Condiment Dispenser
    (51, 152, 255),     #44 Vending Machine
    (255, 204, 204),    #45 Emergency Aid Kit
    (255, 102, 102),    #46 Fire Extinguisher
    (0, 153, 76),       #47 Computer
    (32, 32, 32),       #48 Television
    (255, 255, 255),     #49 Other
    (255, 102, 102),    #50 Fire Extinguisher
    (0, 153, 76),       #51 Computer
    (32, 32, 32),       #52 Television
    (255, 255, 255),    #53 Other
    (255, 255, 255)     #54 Other
    #TODO ADD ADDITIONAL COLORS FOR NEW CLASSES
]

"""
TERRAIN SEMANTIC CLASS CONSTANTS
"""

SEM_CLASS_TO_ID = {
    "Unknown":              0,
    "Concrete":             1,
    "Grass":                2,
    "Rocks":                3,
    "Speedway Bricks":      4,
    "Metal Grates":         5,
    "Red Bricks":           6,
    "Pebble Pavement":      7,
    "Light Marble Tiling":  8,
    "Blond Marble Tiling":  9,
    "Dark Marble Tiling":   10,
    "Dirt Paths":           11,
    "Road Pavement":        12,
    "Short Vegetation":     13,
    "Wood Panel":           14,
    "Porcelain Tile":       15,
    "Patterned Tile":       16,
    "Carpet":               17,
    "Crosswalk":            18,
    "Dome Mat":             19,
    "Stairs":               20,
    "Door Mat":             21,
    "Threshold":            22,
    "Metal Floor":          23
}

SEM_ID_TO_COLOR = [
    (0, 0, 0),    #Unknown
    (100, 100, 100),    #Concrete
    (0, 158, 33),       #Grass
    (120, 120, 120),    #Rocks
    (213, 126, 83),     #Speedway Bricks
    (80, 80, 80),       #Metallic Grates
    (140, 45, 0),       # Red Bricks
    (100, 90, 80),      #Pebble Pavement
    (255, 180, 140),    # Light Marble
    (250, 240, 190),    # Blond Marble
    (50, 50, 50),       # Dark Marble
    (75, 60, 40),       # Dirt
    (140, 140, 140),    # Road
    (10, 120, 50),      # Short Vegetation
    (55, 60, 40),       #Wood Panel
    (240, 200, 75),     # Porcelain Tile
    (60, 100, 180),     # PAtterned
    (0, 50, 125),       # Carpet
    (230, 230, 230),    # Crosswalk
    (165, 100, 0),      # Dome Mat
    (160, 0, 160),      # Stairs
    (240, 180, 240),    #Doot Mat
    (150, 255, 250),    #Threshold
    (0, 120, 110)       # Metal Floor
]

"""
Manifest file generation sensor to subdirectory mappings
"""
SENSOR_DIRECTORY_SUBPATH = {
    #Depth
    "/ouster/lidar_packets": "3d_raw/os1",
    "/camera/depth/image_raw/compressed": "3d_raw/cam2",
    "/zed/zed_node/depth/depth_registered": "3d_raw/cam3",
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
    "3d_raw/os1": "bin",
    "3d_raw/cam2": "png",
    "3d_raw/cam3": "png",
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
    "%s/os1"%TRED_BBOX_LABEL_TYPE: "json",
    "%s/os1"%SEMANTIC_LABEL_TYPE: "bin",
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

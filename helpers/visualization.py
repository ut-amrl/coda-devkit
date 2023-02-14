import os
import pdb
import json

#Libraries
import numpy as np

import cv2
from cv_bridge import CvBridge

#ROS
import rospy
import std_msgs.msg
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped


#Custom
from helpers.constants import PointField, BBOX_CLASS_TO_ID, BBOX_ID_TO_COLOR
from helpers.geometry import *

def pub_pc_to_rviz(pc, pc_pub, ts, frame_id="os_sensor"):
    is_intensity    = pc.shape[-1]>=4
    is_ring         = pc.shape[-1]>=5
    if is_ring:
        ring        = pc[:, :, 4].astype(np.uint16)
        pc_noring   = pc[:, :, :4]

        pc_bytes        = pc_noring.reshape(-1, 1).tobytes()
        ring_bytes      = ring.reshape(-1, 1).tobytes()

        pc_bytes_np = np.frombuffer(pc_bytes, dtype=np.uint8)
        ring_bytes_np = np.frombuffer(ring_bytes, dtype=np.uint8).reshape(-1, 2)

        pc_bytes_np = pc_bytes_np.reshape(-1, 16)
        all_bytes_np= np.hstack((pc_bytes_np, ring_bytes_np)).reshape(-1, 1)

    pc_flat = pc.reshape(-1, 1)

    DATATYPE= PointField.FLOAT32
    if pc.itemsize==PointField.UINT16:
        DATATYPE = PointField.FLOAT32
    else:
        DATATYPE = PointField.FLOAT64
        print("Undefined datatype size accessed, defaulting to FLOAT64...")

    pc_msg = PointCloud2()
    if pc.ndim>2:
        pc_msg.width        = pc.shape[1]
        pc_msg.height       = pc.shape[0]
    else:
        pc_msg.width        = 1
        pc_msg.height       = pc.shape[0]

    pc_msg.header            = std_msgs.msg.Header()
    pc_msg.header.stamp      = ts
    pc_msg.header.frame_id   = frame_id

    fields  = [
        PointField('x', 0, DATATYPE, 1),
        PointField('y', pc.itemsize, DATATYPE, 1),
        PointField('z', pc.itemsize*2, DATATYPE, 1)
    ]

    if is_ring:
        pc_msg.point_step   = pc.itemsize*4 + 2
        fields.append(PointField('intensity', pc.itemsize*3, DATATYPE, 1))
        fields.append(PointField('ring', pc.itemsize*4, PointField.UINT16, 1))
    elif is_intensity:
        pc_msg.point_step   = pc.itemsize*4
        fields.append(PointField('intensity', pc.itemsize*3, DATATYPE, 1))
    else:
        pc_msg.point_step   = pc.itemsize*3

    pc_msg.row_step     = pc_msg.width * pc_msg.point_step
    pc_msg.fields   = fields
    if is_ring:
        pc_msg.data     = all_bytes_np.tobytes()
    else:
        pc_msg.data     = pc_flat.tobytes()
    pc_msg.is_dense = True
    pc_pub.publish(pc_msg)
   
    return pc_msg

def pub_3dbbox_to_rviz(m_pub, anno_filepath, ts, track=False, verbose=False):
    """
    Annotation filepath may not be valid, double check before publishing
    """
    if not os.path.exists(anno_filepath):
        if verbose:
            print("File %s not found, not printing"%anno_filepath)
        return

    anno_file   = open(anno_filepath, 'r')
    anno_json   = json.load(anno_file)

    # Clear previous bbox markers
    bbox_markers = MarkerArray()
    bbox_marker = Marker()
    bbox_marker.id = 0
    bbox_marker.ns = "delete_markerarray"
    bbox_marker.action = Marker.DELETEALL
    bbox_markers.markers.append(bbox_marker)
    m_pub.publish(bbox_markers)

    bbox_markers = MarkerArray()
    for idx, annotation in enumerate(anno_json["3dbbox"]):
        #Pose processing
        px, py, pz          = annotation["cX"], annotation["cY"], annotation["cZ"]
        instanceId, classId = annotation["instanceId"], annotation["classId"]
        # if classId=='Tree':
        #     continue
        
        rot_mat = R.from_euler('xyz', [annotation['r'], annotation['p'], annotation['y']], degrees=False)
        quat_orien = R.as_quat(rot_mat)

        bbox_marker = Marker()
        bbox_marker.header.frame_id = "os_sensor"
        bbox_marker.header.stamp = ts
        bbox_marker.ns = instanceId # Keep ns same for the same object
        bbox_marker.id = int(instanceId.split(':')[-1])
        bbox_marker.type = bbox_marker.CUBE
        bbox_marker.action = bbox_marker.ADD
        bbox_marker.pose.position.x = px
        bbox_marker.pose.position.y = py
        bbox_marker.pose.position.z = pz
        bbox_marker.pose.orientation.x = quat_orien[0]
        bbox_marker.pose.orientation.y = quat_orien[1]
        bbox_marker.pose.orientation.z = quat_orien[2]
        bbox_marker.pose.orientation.w = quat_orien[3]
        bbox_marker.scale.x = annotation["l"]
        bbox_marker.scale.y = annotation["w"]
        bbox_marker.scale.z = annotation["h"]
        bbox_marker.color.a = 0.4
        
        if track:
            # TODO Figure out how to do instance tracking
            bbox_marker.color.r = 100 / 255.0;
            bbox_marker.color.g = 100 /255.0;
            bbox_marker.color.b = 100 / 255.0;
        else:
            class_color = BBOX_ID_TO_COLOR[BBOX_CLASS_TO_ID[classId]]
            bbox_marker.color.r = class_color[0] / 255.0;
            bbox_marker.color.g = class_color[1] /255.0;
            bbox_marker.color.b = class_color[2] / 255.0;

            if instanceId=="Person:1":
                bbox_marker.color.r = 50 / 255.0;
                bbox_marker.color.g = 205 /255.0;
                bbox_marker.color.b = 50 / 255.0;


        bbox_markers.markers.append(bbox_marker)

    m_pub.publish(bbox_markers)
    
def pub_pose(pose_pub, pose, frame, frame_time):
    p = PoseStamped()
    p.header.stamp = frame_time
    p.header.frame_id = "os_sensor"
    p.pose.position.x = pose[1]
    p.pose.position.y = pose[2]
    p.pose.position.z = pose[3]
    # Make sure the quaternion is valid and normalized
    p.pose.orientation.x = pose[5]
    p.pose.orientation.y = pose[6]
    p.pose.orientation.z = pose[7]
    p.pose.orientation.w = pose[4]
    pose_pub.publish(p)

def pub_img(img_pub, img_header, img_path, read_type=cv2.IMREAD_COLOR):
    img = cv2.imread(img_path, read_type)
    if img is None:
        print("Error reading image from file %s, skipping..." % img_path)
        return
    # img_msg = CvBridge().cv2_to_compressed_imgmsg(img, dst_format="png")
    img_msg = CvBridge().cv2_to_compressed_imgmsg(img, dst_format="png")

    img_msg.header = img_header
    img_pub.publish(img_msg)

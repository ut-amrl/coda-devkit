import os
import json

#Libraries
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.cm as cm

import cv2
from cv_bridge import CvBridge

#ROS
import rospy
import std_msgs.msg
from sensor_msgs.msg import PointCloud2, Imu
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
import sensor_msgs.point_cloud2 as pc2

#Custom
from helpers.constants import *
from helpers.sensors import read_sem_label
from helpers.geometry import *
from helpers.calibration import load_extrinsic_matrix

def pub_pc_to_rviz(pc, pc_pub, ts, point_type="x y z", frame_id="os_sensor", publish=True):
    if not isinstance(ts, rospy.Time):
        ts = rospy.Time.from_sec(ts)

    def add_field(curr_bytes_np, next_pc, field_name, fields):
        """
        curr_bytes_np - expect Nxbytes array
        next_pc - expects Nx1 array
        datatype - uint32, uint16
        """
        field2dtype = {
            "x":    np.array([], dtype=np.float32),
            "y":    np.array([], dtype=np.float32),
            "z":    np.array([], dtype=np.float32),
            "i":    np.array([], dtype=np.float32),
            "t":    np.array([], dtype=np.uint32),
            "re":   np.array([], dtype=np.uint16),
            "ri":   np.array([], dtype=np.uint16),
            "am":   np.array([], dtype=np.uint16),
            "ra":   np.array([], dtype=np.uint32),
            "r":    np.array([], dtype=np.float32),
            "g":    np.array([], dtype=np.float32),
            "b":    np.array([], dtype=np.float32)
        }
        field2pftype = {
            "x": PointField.FLOAT32,  "y": PointField.FLOAT32,  "z": PointField.FLOAT32,
            "i": PointField.FLOAT32,  "t": PointField.UINT32,  "re": PointField.UINT16,  
            "ri": PointField.UINT16, "am": PointField.UINT16, "ra": PointField.UINT32,
            "r": PointField.FLOAT32, "g": PointField.FLOAT32, "b": PointField.UINT32
        }
        field2pfname = {
            "x": "x", "y": "y", "z": "z", 
            "i": "intensity", "t": "t", 
            "re": "reflectivity",
            "ri": "ring",
            "am": "ambient", 
            "ra": "range",
            "r": "r",
            "g": "g",
            "b": "b"
        }

        #1 Populate byte data
        dtypetemp = field2dtype[field_name]

        next_entry_count = next_pc.shape[-1]
        next_bytes = next_pc.astype(dtypetemp.dtype).tobytes()

        next_bytes_width = dtypetemp.itemsize * next_entry_count
        next_bytes_np = np.frombuffer(next_bytes, dtype=np.uint8).reshape(-1, next_bytes_width)

        all_bytes_np = np.hstack((curr_bytes_np, next_bytes_np))

        #2 Populate fields
        pfname  = field2pfname[field_name]
        pftype  = field2pftype[field_name]
        pfpos   = curr_bytes_np.shape[-1]
        fields.append(PointField(pfname, pfpos, pftype, 1))
        
        return all_bytes_np, fields

    #1 Populate pc2 fields
    pc = pc.reshape(-1, pc.shape[-1]) # Reshape pc to N x pc_fields
    all_bytes_np = np.empty((pc.shape[0], 0), dtype=np.uint8)
    all_fields_list = []
    field_names = point_type.split(" ")
    for field_idx, field_name in enumerate(field_names):
        next_field_col_np = pc[:, field_idx].reshape(-1, 1)
        # import pdb; pdb.set_trace()

        all_bytes_np, all_fields_list = add_field(
            all_bytes_np, next_field_col_np, field_name, all_fields_list
        )

    #2 Make pc2 object
    pc_msg = PointCloud2()
    pc_msg.width        = 1
    pc_msg.height       = pc.shape[0]

    pc_msg.header            = std_msgs.msg.Header()
    pc_msg.header.stamp      = ts
    pc_msg.header.frame_id   = frame_id

    pc_msg.point_step = all_bytes_np.shape[-1]
    pc_msg.row_step     = pc_msg.width * pc_msg.point_step
    pc_msg.fields       = all_fields_list
    pc_msg.data         = all_bytes_np.tobytes()
    pc_msg.is_dense     = True

    if publish:
        pc_pub.publish(pc_msg)

    return pc_msg

def pub_waypoint_to_rviz(marker_publisher, waypoint, waypoint_ts):
    marker = Marker(type=Marker.CUBE, id=waypoint_ts, pose=Pose(Point(waypoint[0], waypoint[1], waypoint[2]), Quaternion(waypoint[3], waypoint[4], waypoint[5], waypoint[6])), scale = Vector3(1, 1, 1), color = stdmsgs.msg.ColorRGBA(0.0, 1.0, 1.0, 1.0))
    marker_publisher.publish(marker)


def pub_imu_to_rviz(imu_np, imu_pub, frame_id="vectornav", publish=True):
    """
    IMU data assummed to be in the same frame as LiDAR for coordinate axes
    """
    imu_msg = Imu()
    imu_msg.header.stamp    =  rospy.Time.from_sec(imu_np[0])
    imu_msg.header.frame_id = frame_id

    imu_msg.orientation.x   = imu_np[8]
    imu_msg.orientation.y   = imu_np[9]
    imu_msg.orientation.z   = imu_np[10]
    imu_msg.orientation.w   = imu_np[7]

    imu_msg.angular_velocity.x  = imu_np[4]
    imu_msg.angular_velocity.y  = imu_np[5]
    imu_msg.angular_velocity.z  = imu_np[6]

    imu_msg.linear_acceleration.x  = imu_np[1]
    imu_msg.linear_acceleration.y  = imu_np[2]
    imu_msg.linear_acceleration.z  = imu_np[3]

    if publish:
        imu_pub.publish(imu_msg)

    return imu_msg

def apply_semantic_cmap(semantic_anno_path, valid_point_mask=None):
    sem_labels = read_sem_label(semantic_anno_path).astype(np.int32)

    sem_id_to_color_np = np.array(SEM_ID_TO_COLOR,dtype=np.int32)
    color_map = sem_id_to_color_np[sem_labels]
    if valid_point_mask is not None:
        color_map = color_map[valid_point_mask]

    return color_map

def apply_rgb_cmap(img_path, bin_np, calib_ext_file, calib_intr_file, return_pc_mask=False):
    image_pts, pts_mask = project_3dto2d_points(bin_np, calib_ext_file, calib_intr_file)

    in_bounds = np.logical_and(
            np.logical_and(image_pts[:, 0]>=0, image_pts[:, 0]<1224),
            np.logical_and(image_pts[:, 1]>=0, image_pts[:, 1]<1024)
        )

    valid_point_mask = in_bounds & pts_mask
    valid_point_indices = np.where(valid_point_mask)[0]
    valid_points = image_pts[valid_point_mask, :]
    pt_color_map = np.array([(255, 255, 255)] * bin_np.shape[0], dtype=np.uint32)

    image_np = cv2.imread(img_path, cv2.IMREAD_COLOR)

    pt_color_map[valid_point_indices, :3] = image_np[valid_points[:, 1], valid_points[:, 0]]
    pt_color_map = np.stack((pt_color_map[:, 2], pt_color_map[:, 1], pt_color_map[:, 0]), axis=-1)
    return pt_color_map, valid_point_mask

def project_3dpoint_image(image_np, bin_np, calib_ext_file, calib_intr_file, colormap=None):
    image_pts, pts_mask = project_3dto2d_points(bin_np, calib_ext_file, calib_intr_file)

    in_bounds = np.logical_and(
            np.logical_and(image_pts[:, 0]>=0, image_pts[:, 0]<1224),
            np.logical_and(image_pts[:, 1]>=0, image_pts[:, 1]<1024)
        )

    valid_point_mask = in_bounds & pts_mask
    valid_points = image_pts[valid_point_mask, :]

    color_map = [(0, 0, 255)] * valid_points.shape[0]
    if colormap is not None and colormap=="camera":
        os1_to_cam = load_extrinsic_matrix(calib_ext_file)
        bin_homo_os1 = np.hstack((bin_np, np.ones( (bin_np.shape[0], 1) ) ))
        bin_homo_cam = (os1_to_cam @ bin_homo_os1.T).T
        valid_z_map = bin_homo_cam[:, 2][valid_point_mask]

        valid_z_map = np.clip(valid_z_map, 1, 40)
        # color_map = cm.get_cmap("viridis")(np.linspace(0.2, 0.7, len(valid_z_map))) * 255 # [0,1] to [0, 255]]
        norm_valid_z_map = valid_z_map / max(valid_z_map)
        color_map = cm.get_cmap("turbo")(norm_valid_z_map) * 255 # [0,1] to [0, 255]]
        color_map = color_map[:, :3]
    else:
        # Expected input np array
        color_map = apply_semantic_cmap(colormap, valid_point_mask)

    for pt_idx, pt in enumerate(valid_points):
        if color_map[pt_idx].tolist()==[0,0,0]:
            continue # Only circle non background
        image_np = cv2.circle(image_np, (pt[0], pt[1]), radius=SEM_POINT_SIZE, color=color_map[pt_idx].tolist(), thickness=-1)
    return image_np

def project_3dbbox_image(anno_dict, calib_ext_file, calib_intr_file, image):
    """
    Projects 3D Bbox to 2d image
    """
    bbox_pts, bbox_mask, bbox_idxs = project_3dto2d_bbox(anno_dict, calib_ext_file, calib_intr_file)
    for obj_idx in range(0, bbox_pts.shape[0]):
        # in_bounds = np.logical_and(
        #     np.logical_and(bbox_pts[obj_idx, :, 0]>=0, bbox_pts[obj_idx, :, 0]<1224),
        #     np.logical_and(bbox_pts[obj_idx, :, 1]>=0, bbox_pts[obj_idx, :, 1]<1024)
        # )

        valid_point_mask = bbox_mask[obj_idx]
        if np.sum(valid_point_mask)==0:
            continue
        
        # valid_points = bbox_pts[obj_idx, np.arange(8), :]
        valid_points = bbox_pts[obj_idx, valid_point_mask, :]

        bbox_idx = bbox_idxs[obj_idx][0]
        # if anno_dict["3dbbox"][bbox_idx]["classId"] not in ["Car", "Pedestrian", "Bike", "Pickup Truck", "Delivery Truck", "Service Vehicle", "Utility Vehicle"]:
        #     continue

        obj_id = BBOX_CLASS_TO_ID[anno_dict["3dbbox"][bbox_idx]["classId"]]
        obj_color = BBOX_ID_TO_COLOR[obj_id]

        image = draw_bbox(image, valid_points, valid_point_mask, color=obj_color)
    return image

def project_3dto2d_bbox_image(anno_dict, calib_ext_file, calib_intr_file):
    bbox_pts, bbox_mask, bbox_idxs = project_3dto2d_bbox(anno_dict, calib_ext_file, calib_intr_file, check_img=True)
    num_boxes = bbox_pts.shape[0]
    bbox_coords = np.zeros((num_boxes, 4)) # (left top) minxy, (right bottom) maxxy 
    for obj_idx in range(0, num_boxes):
        # in_bounds = np.logical_and(
        #     np.logical_and(bbox_pts[obj_idx, :, 0]>=0, bbox_pts[obj_idx, :, 0]<1224),
        #     np.logical_and(bbox_pts[obj_idx, :, 1]>=0, bbox_pts[obj_idx, :, 1]<1024)
        # )

        # valid_point_mask = bbox_mask[obj_idx] & in_bounds
        # valid_points = bbox_pts[obj_idx, valid_point_mask, :]
        # if valid_points.shape[0]==0:
        #     continue

        valid_points = bbox_pts[obj_idx, bbox_mask[obj_idx], :]
        if valid_points.shape[0]==0:
            continue
        valid_points[:, 0] = valid_points[:, 0].clip(min=0, max=1223)
        valid_points[:, 1] = valid_points[:, 1].clip(min=0, max=1023)

        bbox_idx = bbox_idxs[obj_idx][0]
        max_xy = np.max(valid_points, axis=0)
        min_xy = np.min(valid_points, axis=0)
        # using obj_idx instead of bbox_idx because not gauranteed to be continuous
        bbox_coords[obj_idx] = np.concatenate((min_xy, max_xy), axis=0)

    return bbox_coords

def pub_pose(pose_pub, pose, frame_time, frame_id="os_sensor"):
    """
    Create a PoseStamped msg
    
    Args:
        pose_pub: ros publisher handle
        pose: a Pose [ts, x, y, z, qw, qx, qy, qz]
        frame_time: frame time of header msgs
    Returns:
        pose_msg: a PoseStamped msg
    """
    if not isinstance(frame_time, rospy.Time):
        frame_time = rospy.Time.from_sec(frame_time)

    p = PoseStamped()
    p.header.stamp = frame_time
    p.header.frame_id = frame_id
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

def clear_marker_array(publisher):
    """
    Clear previous bbox markers
    """
    bbox_markers = MarkerArray()
    bbox_marker = Marker()
    bbox_marker.id = 0
    bbox_marker.ns = "delete_markerarray"
    bbox_marker.action = Marker.DELETEALL

    bbox_markers.markers.append(bbox_marker)
    publisher.publish(bbox_markers)

def create_3d_bbox_marker(cx, cy, cz, l, w, h, roll, pitch, yaw,
                          frame_id, time_stamp, namespace='', marker_id=0,
                          r=1.0, g=0.0, b=0.0, a=1.0, scale=0.1):
    """
    Create a 3D bounding box marker

    Args:
        cx, cy, cz: center of the bounding box
        l, w, h: length, width, and height of the bounding box
        roll, pitch, yaw: orientation of the bounding box
        frame_id: frame id of header msgs
        time_stamp: time stamp of header msgs
        namespace: namespace of the bounding box
        marker_id: id of the bounding box
        scale: scale of line width
        r, g, b, a: color and transparency of the bounding box
    
    Returns:
        marker: a 3D bounding box marker
    """

    # Create transformation matrix from the given roll, pitch, yaw
    rot_mat = R.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_matrix()

    # Half-length, half-width, and half-height
    hl, hw, hh = l / 2.0, w / 2.0, h / 2.0

    # Define 8 corners of the bounding box in the local frame
    local_corners = np.array([
        [hl, hw, hh],  [hl, hw, -hh],  [hl, -hw, hh],  [hl, -hw, -hh],
        [-hl, hw, hh], [-hl, hw, -hh], [-hl, -hw, hh], [-hl, -hw, -hh]
    ]).T

    # Transform corners to the frame
    frame_corners = rot_mat.dot(local_corners)
    frame_corners += np.array([[cx], [cy], [cz]])
    
    # Define lines for the bounding box (each line by two corner indices)
    lines = [
        [0, 1], [0, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 7], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    # Create the LineList marker
    marker = Marker()
    marker.header.frame_id = frame_id
    if type(time_stamp) is not rospy.Time:
        time_stamp = rospy.Time.from_sec(time_stamp)
    marker.header.stamp = time_stamp
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = scale
    marker.color.a = a
    marker.color.r = r 
    marker.color.g = g
    marker.color.b = b 
    marker.pose.orientation.w = 1.0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.lifetime = rospy.Duration(0)

    for start, end in lines:
        start_pt = Point(*frame_corners[:, start])
        end_pt = Point(*frame_corners[:, end])
        marker.points.extend([start_pt, end_pt])

    return marker
from fileinput import close
import os
import pdb
import yaml
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

import cv2

from helpers.constants import BBOX_CLASS_TO_ID, NONRIGID_CLASS_IDS, BBOX_CLASS_VIZ_LIST

def densify_poses_between_ts(pose_np, ts_np):
    out_pose_np = np.empty((0, pose_np.shape[1]), dtype=np.float64)
    ts_np = ts_np.reshape(-1,)
    for ts in ts_np:
        closest_pose = find_closest_pose(pose_np, ts)
        out_pose_np = np.vstack((out_pose_np, closest_pose))

    return out_pose_np

def find_closest_pose(pose_np, target_ts):
    curr_ts_idx = np.searchsorted(pose_np[:, 0], target_ts, side="right") - 1

    if curr_ts_idx>=pose_np.shape[0]:
        curr_ts_idx = pose_np.shape[0]-1
        print("Reached end of known poses at time %10.6f" % target_ts)
    elif curr_ts_idx < 0:
        curr_ts_idx = 0

    next_ts_idx = curr_ts_idx + 1
    if next_ts_idx>=pose_np.shape[0]:
        next_ts_idx = pose_np.shape[0] - 1

    if pose_np[curr_ts_idx][0] != pose_np[next_ts_idx][0]:
        pose = inter_pose(pose_np[curr_ts_idx], pose_np[next_ts_idx], target_ts)
    else:
        pose = pose_np[curr_ts_idx]

    return pose

def inter_pose(posea, poseb, sensor_ts):
    """
    Pose assumed to be in x y z qw qx qy qz
    """
    tsa = posea[0]
    tsb = poseb[0]
    if tsa==tsb:
        return posea
    quata = posea[4:]
    quatb = poseb[4:]
    transa  = posea[1:4]
    transb  = poseb[1:4]

    tparam = abs(sensor_ts - tsa) / (tsb - tsa)
    inter_trans = transa + tparam * (transb - transa)
    theta = np.arccos(np.dot(quata, quatb))
  
    # print("tsb ", tsb, " tsa ", tsa, " sensor_ts ", sensor_ts)
    # print("tparam ", tparam)

    inter_quat  =   ( np.sin( (1-tparam) * theta)  / np.sin(theta) ) * quata + \
                    ( np.sin( tparam * theta)      / np.sin(theta) ) * quatb
    # print("quat a ", quata, " inter_quat ", inter_quat, " quatb ", quatb)
    new_pose = np.concatenate((sensor_ts, inter_trans, inter_quat), axis=None)

    return new_pose

def bbox_transform(annotation, trans):
    use_quat = False

    if "r" not in annotation.keys():
        use_quat = True
    center  = np.array([annotation['cX'], annotation['cY'], annotation['cZ']])

    orient = None
    if use_quat:
        orient  = R.from_quat([annotation['qx'], annotation['qy'], annotation['qz'],
            annotation['qw']]).as_matrix()
    else:
        orient  = R.from_euler("xyz", [annotation['r'], annotation['p'], annotation['y']],
            degrees=False).as_matrix()
    homo_coords = np.eye(4)
    homo_coords[:3, :3] = orient
    homo_coords[:3, 3]  = center

    new_coords  = trans @ homo_coords
    annotation['cX'], annotation['cY'], annotation['cZ'] = new_coords[0, 3], \
        new_coords[1, 3], new_coords[2, 3]

    if use_quat:
        annotation['qx'], annotation['qy'], annotation['qz'], annotation['qw'] = R.from_matrix(
            new_coords[:3, :3]).as_quat()
    else:
        annotation['r'], annotation['p'], annotation['y'] = R.from_matrix(
            new_coords[:3, :3]).as_euler("xyz", degrees=False)
    return annotation

def wcs_mat(angles):
    """
    assumes angles order is zyx degrees
    """
    r = R.from_euler('zyx', angles, degrees=True)
    return r

def pose_to_homo(pose):
    trans = pose[1:4]
    quat = np.array([pose[5], pose[6], pose[7], pose[4]])
    rot_mat = R.from_quat(quat).as_matrix()

    homo_mat = np.eye(4, dtype=np.float64)
    homo_mat[:3, :3] = rot_mat
    homo_mat[:3, 3] = trans
    return homo_mat

def get_points_in_bboxes(pc, anno_filepath, verbose=True):
    if not os.path.exists(anno_filepath):
        if verbose:
            print("File %s not found, not printing"%anno_filepath)
        return

    anno_file   = open(anno_filepath, 'r')
    anno_json   = json.load(anno_file)

    output_mask = np.zeros((pc.shape[0], ), dtype=np.bool8)
    for idx, annotation in enumerate(anno_json["3dbbox"]):
        #Pose processing
        px, py, pz          = annotation["cX"], annotation["cY"], annotation["cZ"]
        l, w, h             = annotation["l"], annotation["w"], annotation["h"]
        _, classId = annotation["instanceId"], annotation["classId"]
        classidx = BBOX_CLASS_TO_ID[classId]

        if classidx not in NONRIGID_CLASS_IDS:
            min_x, max_x        = px - l/2.0, px + l/2.0
            min_y, max_y        = py - w/2.0, py + w/2.0
            min_z, max_z        = pz - h/2.0, pz + h/2.0

            mask_x = (pc[:, 0] >= min_x) & (pc[:, 0] <= max_x)
            mask_y = (pc[:, 1] >= min_y) & (pc[:, 1] <= max_y)
            mask_z = (pc[:, 2] >= min_z) & (pc[:, 2] <= max_z)
            mask = mask_x & mask_y & mask_z
            pc_indices = np.where(mask)
            # pdb.set_trace()
            output_mask[pc_indices] = 1
    return output_mask

def project_3dto2d_points(pc_np, calib_ext_file, calib_intr_file, wcs_pose=None):
    #Compute ego lidar to ego camera coordinate systems (Extrinsic)
    calib_ext = open(calib_ext_file, 'r')
    calib_ext = yaml.safe_load(calib_ext)['extrinsic_matrix']
    ext_homo_mat    = np.array(calib_ext['data']).reshape(
        calib_ext['rows'], calib_ext['cols']
    )
    if wcs_pose is not None:
        #Transform PC from WCS to ego lidar
        pc_np_homo = np.hstack((pc_np, np.ones(pc_np.shape[0], 1)))
        pc_np = (np.linalg.inv(wcs_pose) @ pc_np_homo.T).T[:, :3]

    #Load projection, rectification, distortion camera matrices
    intr_ext    = open(calib_intr_file, 'r')
    intr_ext    = yaml.safe_load(intr_ext)

    K   = np.array(intr_ext['camera_matrix']['data']).reshape(3, 3)
    d   = np.array(intr_ext['distortion_coefficients']['data']) # k1, k2, k3, p1, p2

    image_points, _ = cv2.projectPoints(pc_np[:, :3].astype(np.float64), 
        ext_homo_mat[:3, :3], ext_homo_mat[:3, 3], K, d)
    image_points = np.swapaxes(image_points, 0, 1).astype(np.int32).squeeze()
    valid_points_mask = get_pointsinfov_mask(
        (ext_homo_mat[:3, :3]@pc_np[:, :3].T).T+ext_homo_mat[:3, 3])

    return image_points, valid_points_mask

def project_3dto2d_bbox(tred_annotation, calib_ext_file, calib_intr_file):
    """
    wcs_mat - 4x4 homogeneous matrix
    """

    all_image_points = np.empty((0,8,2), dtype=np.int32)
    all_points_fov_mask = np.empty( (0, 8), dtype=np.bool)
    all_valid_obj_idx   = np.empty( (0, 1), dtype=np.int32)
    for annotation_idx, annotation in enumerate(tred_annotation["3dbbox"]):
        tred_corners = np.zeros((8, 3), dtype=np.float32)

        cX, cY, cZ, l, w, h = annotation['cX'], annotation['cY'], annotation['cZ'],\
                annotation['l'], annotation['w'], annotation['h']
        if "r" in annotation.keys():
            r, p, y = annotation['r'], annotation['p'], annotation['y']
            tred_orien_transform = R.from_euler("xyz", [r, p, y], degrees=False).as_matrix()
        else:
            qx, qy, qz, qw = annotation['qx'], annotation['qy'], annotation['qz'], annotation["qw"]
            tred_orien_transform = R.from_quat([qx, qy, qz, qw]).as_matrix()
        
        #Compute corners in axis aligned coordinates
        x_sign = np.array([-1, -1, 1, 1])
        y_sign = np.array([1, -1, -1, 1])
        z_sign = np.array([-1, -1, -1, -1])

        for cnum in np.arange(0, 8, 1):
            sign_idx = cnum % 4
            x_offset = x_sign[sign_idx] * l/2
            y_offset = y_sign[sign_idx] * w/2
            z_offset = z_sign[sign_idx] * h/2

            if cnum >=4:
                z_offset *= -1
            tred_corners[cnum] = np.array([
                x_offset, y_offset, z_offset])

        tred_corners = (tred_orien_transform@tred_corners.T).T
        tred_corners += np.array([cX, cY, cZ])

        #Compute ego lidar to ego camera coordinate systems (Extrinsic)
        calib_ext = open(calib_ext_file, 'r')
        calib_ext = yaml.safe_load(calib_ext)['extrinsic_matrix']
        ext_homo_mat    = np.array(calib_ext['data']).reshape(
            calib_ext['rows'], calib_ext['cols']
        )

        #Load projection, rectification, distortion camera matrices
        intr_ext    = open(calib_intr_file, 'r')
        intr_ext    = yaml.safe_load(intr_ext)

        K   = np.array(intr_ext['camera_matrix']['data']).reshape(3, 3)
        d   = np.array(intr_ext['distortion_coefficients']['data'])
        P   = np.array(intr_ext['projection_matrix']['data']).reshape(3, 4)
        Re  = np.array(intr_ext['rectification_matrix']['data']).reshape(3, 3)

        image_points, _ = cv2.projectPoints(tred_corners[:, :3],
            ext_homo_mat[:3, :3], ext_homo_mat[:3, 3], K, d)
        image_points = np.swapaxes(image_points, 0, 1).astype(np.int32)
        valid_points_mask = get_pointsinfov_mask((ext_homo_mat[:3, :3]@tred_corners.T).T+ext_homo_mat[:3, 3])

        if annotation["classId"] in BBOX_CLASS_VIZ_LIST:
            all_image_points = np.vstack(
                (all_image_points, image_points)
            )
            all_points_fov_mask = np.vstack(
                (all_points_fov_mask, valid_points_mask)
            )
            all_valid_obj_idx = np.vstack(
                (all_valid_obj_idx, annotation_idx)
            )
    return all_image_points, all_points_fov_mask, all_valid_obj_idx

def draw_2d_bbox(image, bbox_coords, color=(0,0,255), thickness=2):
    # Expects nx4 array of minxy, maxxy
    for bbox in bbox_coords:
        if np.sum(bbox==0)==4:
            continue
        
        bbox = bbox.astype(np.int)
        tl_corner = (bbox[0], bbox[1])
        tr_corner = (bbox[0], bbox[3])
        bl_corner = (bbox[2], bbox[1])
        br_corner = (bbox[2], bbox[3])
        corners = [tl_corner, tr_corner, br_corner, bl_corner]
        
        for i in range(0, 4):
            image = cv2.line(image, corners[i], corners[(i+1)%4], color, thickness)

    return image

def draw_bbox(image, bbox_2d_pts, valid_points, color=(0,0,255), thickness=2):
    #Draws 4 rectangles Left, Right, Top, Bottom
    available_points = np.where(valid_points)[0]
    old_to_new_index_map = np.zeros((8,), dtype=np.int32) 
    old_to_new_index_map[available_points] = np.arange(0, available_points.shape[0])

    try:
        #Vert Beams
        for i in range(0, 4):
            if i in available_points and i+4 in available_points:
                si = old_to_new_index_map[i]
                ei = old_to_new_index_map[i+4]
                image = cv2.line(image, tuple(bbox_2d_pts[si]), tuple(bbox_2d_pts[ei]), color, thickness)

        # Horizontal Beams
        for i in range(0, 8, 2):
            if i in available_points and i+1 in available_points:
                si = old_to_new_index_map[i]
                ei = old_to_new_index_map[i+1]
                image = cv2.line(image, tuple(bbox_2d_pts[si]), tuple(bbox_2d_pts[ei]), color, thickness)

        # misc beams
        for i in range(0, 8, 4):
            if i in available_points and i+3 in available_points:
                si = old_to_new_index_map[i]
                ei = old_to_new_index_map[i+3]
                image = cv2.line(image, tuple(bbox_2d_pts[si]), tuple(bbox_2d_pts[ei]), color, thickness)

        # misc beams
        for i in range(1, 8, 4):
            if i in available_points and i+1 in available_points:
                si = old_to_new_index_map[i]
                ei = old_to_new_index_map[i+1]
                image = cv2.line(image, tuple(bbox_2d_pts[si]), tuple(bbox_2d_pts[ei]), color, thickness)
    except Exception as e:
        print(e)
        pdb.set_trace()
    #Draw
    # image = cv2.drawContours(image, np.array([contour_top]), 0, color, thickness)
    # image = cv2.drawContours(image, np.array([contour_bottom]), 0, color, thickness)

    return image

def get_pointsinfov_mask(points):
    """
    Assumes camera coordinate system input points
    """
    norm_p = np.linalg.norm(points, axis=-1, keepdims=True)

    forward_vec = np.array([0, 0, 1]).reshape(3, 1)
    norm_f = np.linalg.norm(forward_vec)
    norm_p[norm_p==0] = 1e-6 # Prevent divide by zero error

    angles_vec  = np.arccos( np.dot(points, forward_vec) / (norm_f*norm_p) )

    in_fov_mask = np.abs(angles_vec[:,0]) <= 0.785398

    return in_fov_mask

# def matplot_3d_points(points, title="coda 3d plot"):
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], cmap='Greens')
#     ax.set_title(title)
#     plt.show()
#     return ax
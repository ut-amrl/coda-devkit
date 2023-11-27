"""
Copyright (c) 2023 Arthur King Zhang arthurz@cs.utexas.edu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following condition
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Softwar
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import cv2
import yaml
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from liegroups import SE3, SO3

from helpers.constants import BBOX_CLASS_TO_ID, NONRIGID_CLASS_IDS, BBOX_CLASS_VIZ_LIST, SEM_ID_TO_COLOR, BBOX_ID_TO_COLOR

def find_closest_pose(pose_np, target_ts):
    """
    Interpolates between two closest poses to the given target timestamp in the pose_np array.

    Parameters:
        pose_np (numpy.ndarray): An array representing a sequence of poses.
            Each row should contain pose information with at least two rows
            The first column should represent the timestamps (time values) of the poses,
            and the rest of the columns should contain the pose data (e.g., position and orientation).
            The timestamps in the first column are assumed to be sorted in ascending order.

        target_ts (float): The target timestamp for which the closest pose is to be found.
            This is the timestamp value at which you want to find the closest pose in the pose_np array.

    Returns:
        numpy.ndarray: Interpolated pose at the target_ts using the provided poses.

    Notes:
        This function uses binary search to locate the index of the pose in pose_np with a timestamp closest to target_ts.
        It then calculates the two nearest poses based on this index (curr_ts_idx and next_ts_idx).
        If the timestamps of these two nearest poses are not identical, it uses inter_pose function to interpolate a pose at target_ts.
        Otherwise, if the timestamps are identical, it returns the pose at curr_ts_idx directly.
    """
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

def densify_poses_between_ts(pose_np, ts_np):
    """
    Densify poses between specified timestamps in `pose_np` array.

    Parameters:
        pose_np : numpy.ndarray
            Array of poses with timestamps.
        ts_np : numpy.ndarray or array-like
            Target timestamps for densification.

    Returns:
        numpy.ndarray
            Densified poses with modified timestamps to match `ts_np`.

    .. note::
        This function loops through each timestamp in `ts_np` and finds the closest pose in `pose_np` to that timestamp
        using `find_closest_pose` function. The pose timestamp is then modified to be the target timestamp (`ts`),
        effectively densifying the poses at the desired points in time. The resulting poses are stored in the `out_pose_np` array,
        which is returned after all timestamps have been processed.
        The `find_closest_pose` function is used to find the closest pose to a given target timestamp in `pose_np`.
        Ensure that the `pose_np` array contains timestamps sorted in ascending order for accurate results.
    """
    out_pose_np = np.empty((0, pose_np.shape[1]), dtype=np.float64)
    ts_np = ts_np.reshape(-1,)
    for ts in ts_np:
        closest_pose = find_closest_pose(pose_np, ts)
        # Modify the pose timestamp to be the target timestamp
        closest_pose[0] = ts
        out_pose_np = np.vstack((out_pose_np, closest_pose))

    return out_pose_np

def xyz_quaternion_to_SE3(x, y, z, qw, qx, qy, qz) -> SE3:
    """Convert the [x, y, z, qw, qx, qy, qz] to an SE(3) transformation matrix"""
    r = SO3.from_quaternion([qw, qx, qy, qz], "wxyz")
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = r.as_matrix()
    transformation_matrix[:3, 3] = [x, y, z]
    return SE3.from_matrix(transformation_matrix, normalize=True)

def SE3_to_xyz_quaternion(transformation_matrix: np.ndarray) -> np.ndarray:
    """Convert the SE(3) transformation matrix to [x, y, z, qw, qx, qy, qz]"""
    r = SO3.from_matrix(transformation_matrix[:3, :3])
    t = transformation_matrix[:3, 3]
    return np.concatenate([t, r.to_quaternion("wxyz")])

def inter_pose(posea, poseb, sensor_ts):
    """
    Interpolate pose between `posea` and `poseb` at `sensor_ts`.

    Parameters:
        posea : numpy.ndarray
            First pose with timestamp.
        poseb : numpy.ndarray
            Second pose with timestamp.
        sensor_ts : float
            Sensor timestamp for interpolation.

    Returns:
        numpy.ndarray
            Interpolated pose at the sensor timestamp.
    """
    tsa = posea[0]
    tsb = poseb[0]
    if tsa==tsb or sensor_ts<=tsa:
        return posea
    elif sensor_ts>=tsb:
        return poseb
    
    posea_SE3 = xyz_quaternion_to_SE3(*posea[1:])
    poseb_SE3 = xyz_quaternion_to_SE3(*poseb[1:])
    pose_change = posea_SE3.inv().as_matrix() @ poseb_SE3.as_matrix()
    xi = SE3.from_matrix(pose_change, True).log()
    
    tparam = abs(sensor_ts - tsa) / (tsb - tsa)
    interpolated_pose = (
        posea_SE3.as_matrix() @ SE3.exp(tparam * xi).as_matrix()
    )
    new_pose = SE3_to_xyz_quaternion(interpolated_pose)
    new_pose = np.concatenate((sensor_ts, new_pose), axis=None)

    return new_pose

def bbox_transform(annotation, trans, use_quat=False):
    """
    Transform the bounding box coordinates of an annotation using the given transformation matrix.

    Parameters:
        annotation : dict
            Dictionary containing bounding box information. (Described in DATA_REPORT.md)
        trans : numpy.ndarray
            4x4 transformation matrix.
        use_quat : bool
            determines whether the orientation is represented using Euler angles or quaternion 
    Returns:
        dict
            Updated annotation dictionary with transformed bounding box coordinates.

    Notes:
        The function takes the annotation dictionary and applies the transformation matrix `trans`
        to update the bounding box coordinates accordingly.
    """
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

def pose_to_homo(pose):
    """
    Convert pose information to a 4x4 homogeneous transformation matrix.

    Parameters:
        pose : numpy.ndarray
            Array representing the pose with timestamp and pose data.

    Returns:
        numpy.ndarray
            4x4 homogeneous transformation matrix.

    Notes:
        The function extracts the translation and quaternion rotation information from the input pose array.
        It constructs the 3x3 rotation matrix from the quaternion and combines it with the translation to form
        the 4x4 homogeneous transformation matrix.
    """
    trans = pose[1:4]
    quat = np.array([pose[5], pose[6], pose[7], pose[4]])
    rot_mat = R.from_quat(quat).as_matrix()

    homo_mat = np.eye(4, dtype=np.float64)
    homo_mat[:3, :3] = rot_mat
    homo_mat[:3, 3] = trans
    return homo_mat

def get_points_in_bboxes(pc, anno_filepath, verbose=True):
    """
    Identify points from a point cloud (pc) that fall within bounding boxes defined in the annotation file.

    Parameters:
        pc : numpy.ndarray
            Point cloud as a 2D array with (x, y, z) coordinates in columns.
        anno_filepath : str
            File path to the annotation JSON file containing bounding box information.
        verbose : bool, optional
            Whether to print messages about file not found. Defaults to True.

    Returns:
        numpy.ndarray
            Boolean mask representing points within bounding boxes (True) and points outside (False).

    Notes:
        The function reads bounding box information from the annotation file and processes each bounding box.
        It checks each point in the point cloud (pc) and determines whether it falls within any bounding box.
        The function returns a boolean mask indicating points within bounding boxes (True) and points outside (False).
    """
    if not os.path.exists(anno_filepath):
        if verbose:
            print("File %s not found, not printing"%anno_filepath)
        return

    anno_file   = open(anno_filepath, 'r')
    anno_json   = json.load(anno_file)

    output_mask = np.zeros((pc.shape[0], ), dtype=bool)
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
    """
    Project 3D points from lidar space to 2D image space using camera calibration information.

    Parameters:
        pc_np : numpy.ndarray
            Point cloud as a 2D array with (x, y, z) coordinates in columns.
        calib_ext_file : str or numpy.ndarray
            File path or 4x4 homogeneous matrix representing the extrinsic calibration information.
        calib_intr_file : str
            File path to the camera calibration YAML file containing intrinsic matrix and distortion coefficients.
        wcs_pose : numpy.ndarray, optional
            4x4 homogeneous matrix representing the World Coordinate System (WCS) pose.
            If provided, the point cloud (pc_np) is first transformed from WCS to ego lidar space. Defaults to None.

    Returns:
        numpy.ndarray
            2D image points (x, y) obtained by projecting 3D points onto the image plane.
        numpy.ndarray
            Boolean mask indicating valid points within the camera field of view (True) and points outside (False).

    Notes:
        The function first transforms the point cloud from WCS to ego lidar space (if wcs_pose is provided).
        It then projects the 3D points onto the image plane using camera calibration matrices.
        The function returns 2D image points and a boolean mask indicating valid points within the camera field of view.
    """

    #Compute ego lidar to ego camera coordinate systems (Extrinsic)
    if isinstance(calib_ext_file, str):
        calib_ext = open(calib_ext_file, 'r')
        calib_ext = yaml.safe_load(calib_ext)['extrinsic_matrix']
        ext_homo_mat    = np.array(calib_ext['data']).reshape(
            calib_ext['rows'], calib_ext['cols']
        )
    else: # Overloaded function to also accept matrix inputs
        ext_homo_mat = calib_ext_file
    if wcs_pose is not None:
        #Transform PC from WCS to ego lidar
        pc_np_homo = np.hstack((pc_np, np.ones(pc_np.shape[0], 1)))
        pc_np = (np.linalg.inv(wcs_pose) @ pc_np_homo.T).T[:, :3]

    #Load projection, rectification, distortion camera matrices
    intr_ext    = open(calib_intr_file, 'r')
    intr_ext    = yaml.safe_load(intr_ext)

    K   = np.array(intr_ext['camera_matrix']['data']).reshape(3, 3)
    d   = np.array(intr_ext['distortion_coefficients']['data']) # k1, k2, k3, p1, p2

    image_points = projectPointsWithDist(pc_np[:, :3].astype(np.float64), ext_homo_mat[:3, :3], 
        ext_homo_mat[:3, 3], K, d, use_dist=False)

    # bin_homo_os1 = np.hstack((bin_np, np.ones( (bin_np.shape[0], 1) ) ))
    #     bin_homo_cam = (os1_to_cam @ bin_homo_os1.T).T

    valid_points_mask = get_pointsinfov_mask(
        (ext_homo_mat[:3, :3]@pc_np[:, :3].T).T+ext_homo_mat[:3, 3])

    return image_points, valid_points_mask

def get_safe_projs(rmat, tvec, distCoeffs, obj_pts):
    """
    Efficiently check safe points for projection from point cloud to image
    TODO: Vectorize this for more efficiency (Slow on large point clouds)

    Parameters:
        rmat : numpy.ndarray
            Rotation matrix (3x3) representing camera extrinsic calibration.
        tvec : numpy.ndarray
            Translation vector (3x1) representing camera extrinsic calibration.
        distCoeffs : numpy.ndarray
            Distortion coefficients of the camera.
        obj_pts : numpy.ndarray
            3D points (n x 3) to be projected.

    Returns:
        numpy.ndarray
            Indices of points (in obj_pts) that are safe to project.

    Notes:
        The function vectorizes the process to efficiently check if the 3D points (obj_pts) are safe for projection.
        It first computes the location of the points relative to the camera using the rotation matrix (rmat) and
        translation vector (tvec). Then, it calculates the homogenous coordinates (x and y) and distortion terms
        for vectorized polynomial evaluation. Finally, it determines the bounds on x and y coordinates using polynomial roots
        and returns the indices of safe points that can be safely projected onto the image plane.
    
    """
    # Define a list of booleans to denote if a variable is safe
    obj_pts_safe = np.array([True] * len(obj_pts))
    
    # First step is to get the location of the points relative to the camera.
    obj_pts_rel_cam = np.squeeze([[np.matmul(rmat, pt) + tvec] for pt in obj_pts], axis=1)

    # Define the homogenous coordiantes
    x_homo_vals = (obj_pts_rel_cam[:, 0] / obj_pts_rel_cam[:, 2]).astype(complex)
    y_homo_vals = (obj_pts_rel_cam[:, 1] / obj_pts_rel_cam[:, 2]).astype(complex)

    # Define the distortion terms, and vectorize calculating of powers of x_homo_vals
    #   and y_homo_vals
    k1, k2, p1, p2, k3 = distCoeffs[0]
    x_homo_vals_2 = np.power(x_homo_vals, 2)
    y_homo_vals_2 = np.power(y_homo_vals, 2)
    x_homo_vals_4 = np.power(x_homo_vals, 4)
    y_homo_vals_4 = np.power(y_homo_vals, 4)
    x_homo_vals_6 = np.power(x_homo_vals, 6)
    y_homo_vals_6 = np.power(y_homo_vals, 6)

    # Find the bounds on the x_homo coordinate to ensure it is closer than the
    #   inflection point of x_proj as a function of x_homo
    x_homo_min = np.full(x_homo_vals.shape, np.inf)
    x_homo_max = np.full(x_homo_vals.shape, -np.inf)
    for i in range(len(y_homo_vals)):
        # Expanded projection function polynomial coefficients
        x_proj_coeffs = np.array([k3,
                                  0,
                                  k2 + 3*k3*y_homo_vals_2[i],
                                  0,
                                  k1 + 2*k2*y_homo_vals_2[i] + 3*k3*y_homo_vals_4[i],
                                  3*p2,
                                  1 + k1 * y_homo_vals_2[i] + k2 * y_homo_vals_4[i] + k3*y_homo_vals_6[i] + 2*p1*y_homo_vals[i],
                                  p2*y_homo_vals_2[i]])

        # Projection function derivative polynomial coefficients
        x_proj_der_coeffs = np.polyder(x_proj_coeffs)    
        
        # Find the root of the derivative
        roots = np.roots(x_proj_der_coeffs)
        
        # Get the real roots
        # Approximation of real[np.where(np.isreal(roots))]
        real_roots = np.real(roots[np.where(np.abs(np.imag(roots)) < 1e-10)])
        
        for real_root in real_roots:
            x_homo_min[i] = np.minimum(x_homo_min[i], real_root)
            x_homo_max[i] = np.maximum(x_homo_max[i], real_root)
        
    # Check that the x_homo values are within the bounds
    obj_pts_safe *= np.where(x_homo_vals > x_homo_min, True, False)
    obj_pts_safe *= np.where(x_homo_vals < x_homo_max, True, False)

    # Find the bounds on the y_homo coordinate to ensure it is closer than the
    #   inflection point of y_proj as a function of y_homo
    y_homo_min = np.full(y_homo_vals.shape, np.inf)
    y_homo_max = np.full(y_homo_vals.shape, -np.inf)
    for i in range(len(x_homo_vals)):
        # Expanded projection function polynomial coefficients
        y_proj_coeffs = np.array([k3,
                                  0,
                                  k2 + 3*k3*x_homo_vals_2[i],
                                  0,
                                  k1 + 2*k2*x_homo_vals_2[i] + 3*k3*x_homo_vals_4[i],
                                  3*p1,
                                  1 + k1 * x_homo_vals_2[i] + k2 * x_homo_vals_4[i] + k3*x_homo_vals_6[i] + 2*p2*x_homo_vals[i],
                                  p1*x_homo_vals_2[i]])

        # Projection function derivative polynomial coefficients
        y_proj_der_coeffs = np.polyder(y_proj_coeffs)    
        
        # Find the root of the derivative
        roots = np.roots(y_proj_der_coeffs)
        
        # Get the real roots
        # Approximation of real[np.where(np.isreal(roots))]
        real_roots = np.real(roots[np.where(np.abs(np.imag(roots)) < 1e-10)])

        for real_root in real_roots:
            y_homo_min[i] = np.minimum(y_homo_min[i], real_root)
            y_homo_max[i] = np.maximum(y_homo_max[i], real_root)
        
    # Check that the x_homo values are within the bounds
    obj_pts_safe *= np.where(y_homo_vals > y_homo_min, True, False)
    obj_pts_safe *= np.where(y_homo_vals < y_homo_max, True, False)
    # import pdb; pdb.set_trace()
    # Return the indices where obj_pts is safe to project
    return np.where(obj_pts_safe == True)[0]

def projectPointsWithDist(points_3d, R, T, K, d, use_dist=True):
    """
    Custom point projection function with handling for OpenCV distortion issue for out of bound points.
    https://github.com/opencv/opencv/issues/17768

    Parameters:
        points_3d : numpy.ndarray
            3D points (n x 3) to be projected.
        R : numpy.ndarray
            Rotation matrix (3x3) representing camera extrinsic calibration.
        T : numpy.ndarray
            Translation vector (3x1) representing camera extrinsic calibration.
        K : numpy.ndarray
            Camera intrinsic matrix (3x3).
        d : numpy.ndarray
            Distortion coefficients of the camera.
        use_dist : bool, optional
            Whether to use distortion in projection. Defaults to True.

    Returns:
        numpy.ndarray
            Projected 2D image points (n x 2).

    Notes:
        The function handles the distortion issue present in OpenCV for out-of-bounds points during projection.
        It first projects the points with distortion using cv2.projectPoints. If use_dist is True, it checks if the
        projected points are out of bounds and replaces them with corresponding undistorted points. This ensures
        that all points are safely projected onto the image plane, even when some points fall outside the bounds.
        The safe_image_points_mask is used to identify safe points to project with distortion, and unsafe points
        are replaced with undistorted points.
    """
    assert points_3d.shape[1]==3, "Error points_3d input is not dimension 3, it is %i"%points_3d.shape[1]
    points_3d = points_3d[:, :3]
    d = np.array(d).reshape(1, -1)
    image_points, _ = cv2.projectPoints(points_3d, R, T, K, d)
    
    MAX_INT32 = np.iinfo(np.int32).max
    MIN_INT32 = np.iinfo(np.int32).min
    image_points = np.clip(image_points, MIN_INT32, MAX_INT32)
    image_points = np.swapaxes(image_points, 0, 1).astype(np.int32)

    if use_dist:
        image_points_nodist, _ = cv2.projectPoints(points_3d, R, T, K, np.zeros((5,)))
        image_points_nodist     = np.swapaxes(image_points_nodist, 0, 1) #.astype(np.int32)
        _, num_points, _ = image_points_nodist.shape

        points_3d = points_3d[np.all(points_3d!=0, axis=-1)]
        safe_image_points_mask = get_safe_projs(R, T, d, points_3d)
        
        safe_image_points = np.zeros((1, num_points, 2)) 
        safe_image_points[0, safe_image_points_mask, :] = image_points[0, safe_image_points_mask, :]
        unsafe_image_points_mask = np.delete(np.arange(0, num_points), safe_image_points_mask)
        safe_image_points[0, unsafe_image_points_mask, :] = image_points_nodist[0, unsafe_image_points_mask, :]
        image_points = safe_image_points
    else:
        image_points, _ = cv2.projectPoints(points_3d, R, T, K, np.zeros((5,)))
        image_points     = np.swapaxes(image_points, 0, 1).squeeze()

    # Cast to image points to pixel ints
    image_points = np.clip(image_points, MIN_INT32, MAX_INT32)
    image_points = image_points.astype(np.int32)

    return image_points

def get_3dbbox_corners(bbox_dict):
    """
    Converts 3D bounding box annotation from CODa format to 3D corners.

    Parameters:
        bbox_dict : dict
            Dictionary containing 3D bounding box annotation in CODa format.

    Returns:
        numpy.ndarray
            Array containing 3D corners of the bounding box.

    Notes:
        The function takes the bounding box annotation in CODa format and computes the 3D corners of the bounding box.
        It extracts the center coordinates (cX, cY, cZ) and dimensions (l, w, h) from the dictionary.
        If rotation angles (r, p, y) are provided, it uses Euler angles to construct the orientation transformation matrix.
        Otherwise, it uses the quaternion (qx, qy, qz, qw) to construct the orientation transformation matrix.
        The corners are then computed in axis-aligned coordinates and transformed to the 3D bounding box's local coordinate system
        using the orientation transformation matrix and translation vector (cX, cY, cZ).
    """
    tred_corners = np.zeros((8, 3), dtype=np.float32)
    cX, cY, cZ, l, w, h = bbox_dict['cX'], bbox_dict['cY'], bbox_dict['cZ'],\
                bbox_dict['l'], bbox_dict['w'], bbox_dict['h']
    if "r" in bbox_dict.keys():
        r, p, y = bbox_dict['r'], bbox_dict['p'], bbox_dict['y']
        tred_orien_transform = R.from_euler("xyz", [r, p, y], degrees=False).as_matrix()
    else:
        qx, qy, qz, qw = bbox_dict['qx'], bbox_dict['qy'], bbox_dict['qz'], bbox_dict["qw"]
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
    
    Tbox = np.eye(4)
    Tbox[:3, :3] =  tred_orien_transform
    Tbox[:3, 3] = np.array([cX, cY, cZ])

    tred_corners_homo = np.hstack((tred_corners, np.ones((tred_corners.shape[0], 1) )))
    tred_corners = (Tbox @ tred_corners_homo.T).T
    tred_corners = tred_corners[:, :3]

    return tred_corners
    

def project_3dto2d_bbox(tred_annotation, calib_ext_file, calib_intr_file, check_img=False):
    """
    wcs_mat - 4x4 homogeneous matrix

    Return point projected to 2d image plane and mask for points in camera field of view
    """

    all_image_points = np.empty((0,8,2), dtype=np.int32)
    all_points_fov_mask = np.empty( (0, 8), dtype=bool)
    all_valid_obj_idx   = np.empty( (0, 1), dtype=np.int32)
    for annotation_idx, annotation in enumerate(tred_annotation["3dbbox"]):
        tred_corners = get_3dbbox_corners(annotation)

        #Compute ego lidar to ego camera coordinate systems (Extrinsic)
        calib_ext = open(calib_ext_file, 'r')
        calib_ext = yaml.safe_load(calib_ext)['extrinsic_matrix']
        ext_homo_mat    = np.array(calib_ext['data']).reshape(
            calib_ext['rows'], calib_ext['cols']
        )

        #Load projection, rectification, distortion camera matrices
        intr_ext    = open(calib_intr_file, 'r')
        intr_ext    = yaml.safe_load(intr_ext)

        img_w = intr_ext['image_width']
        img_h = intr_ext['image_height']
        K   = np.array(intr_ext['camera_matrix']['data']).reshape(3, 3)
        d   = np.array(intr_ext['distortion_coefficients']['data'])
        P   = np.array(intr_ext['projection_matrix']['data']).reshape(3, 4)
        Re  = np.array(intr_ext['rectification_matrix']['data']).reshape(3, 3)

        image_points = projectPointsWithDist(tred_corners[:, :3], ext_homo_mat[:3, :3], ext_homo_mat[:3, 3], K, d)
       
        if check_img:
            valid_points_mask = np.logical_and(
                np.logical_and(image_points[:, :, 0] >= 0, image_points[:, :, 0] < img_w), 
                np.logical_and(image_points[:, :, 1] >= 0, image_points[:, :, 1] < img_h)
            )
        else:
            valid_points_mask = get_pointsinfov_mask((ext_homo_mat[:3, :3]@tred_corners.T).T+ext_homo_mat[:3, 3])
            valid_points_mask = valid_points_mask.reshape(1, -1)
            # import pdb; pdb.set_trace()
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

def draw_2d_bbox(image, bbox_coords, color=(0,0,255), thickness=-1):
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

def draw_2d_sem(image, valid_pts, pts_labels):
    for pt_idx, pt in enumerate(valid_pts):
        pt_label = pts_labels[pt_idx]

        pt_color = SEM_ID_TO_COLOR[pt_label]
        image = cv2.circle(image, (pt[0], pt[1]), radius=1, color=pt_color)
    return image

def redistribute_rgb(r, g, b):
    threshold = 255.999
    m = max(r, g, b)
    if m <= threshold:
        return int(r), int(g), int(b)
    total = r + g + b
    if total >= 3 * threshold:
        return int(threshold), int(threshold), int(threshold)
    x = (3 * threshold - total) / (3 * m - total)
    gray = threshold - x * m
    return int(gray + x * r), int(gray + x * g), int(gray + x * b)

def draw_inst_label(image, bbox_2d_pts, valid_point_mask, bbox_dict, img_w=1223, img_h=1023):
    instanceId = bbox_dict['instanceId'].replace(':', ' ')
    obj_id = BBOX_CLASS_TO_ID[bbox_dict["classId"]]
    obj_color = BBOX_ID_TO_COLOR[obj_id]

    text = instanceId
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.9
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, fontScale, thickness)
    text_w, text_h = text_size

    available_points = np.where(valid_point_mask)[0]
    # old_to_new_index_map = np.zeros((8,), dtype=np.int32)
    # old_to_new_index_map[available_points] = np.arange(0, available_points.shape[0])
    # try:
    # print(available_points)
    # try:
    #     valid_points = bbox_2d_pts[available_points, :]
    # except:
        # import pdb; pdb.set_trace()
    if instanceId=="Tree 1":
        import pdb; pdb.set_trace()
    
    valid_points = bbox_2d_pts
    valid_points[:,0] = np.clip(valid_points[:,0], a_min=text_h, a_max=img_h-text_h)
    valid_points[:,1] = np.clip(valid_points[:,1], a_min=text_w, a_max=img_w-text_w)
    x_anchor, y_anchor = min(valid_points[:,0]), min(valid_points[:,1])

    org = (x_anchor, y_anchor)

    label_image = cv2.rectangle(image, org, (org[0] + text_w, org[1] - text_h), (255, 255, 255), -1)
    label_image = cv2.putText(label_image, text, org, font, fontScale, obj_color, thickness, cv2.LINE_AA, False)
    # cv2.imwrite("test.png", label_image)
    # import pdb; pdb.set_trace()
    return label_image

def draw_bbox(image, bbox_2d_pts, valid_points, color=(0,0,255), bbox_dict=None, thickness=4, img_w=1223, img_h=1023):
    def draw_instlabel(orig_image, bbox_dict, x_anchor, y_anchor):
        instanceId = bbox_dict['instanceId'].replace(':', ' ')
        obj_id = BBOX_CLASS_TO_ID[bbox_dict["classId"]]
        obj_color = BBOX_ID_TO_COLOR[obj_id]

        text = instanceId
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.6
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, fontScale, thickness)
        text_w, text_h = text_size

        org = (x_anchor, y_anchor+text_h)

        label_image = cv2.rectangle(orig_image, org, (org[0] + text_w, org[1] - text_h), (255, 255, 255), -1)
        label_image = cv2.putText(label_image, text, org, font, fontScale, obj_color, thickness, cv2.LINE_AA, False)
        return label_image

    #Draws 4 rectangles Left, Right, Top, Bottom
    available_points = np.where(valid_points)[0]
    # available_points = np.arange(8, dtype=np.int32)
    old_to_new_index_map = np.zeros((8,), dtype=np.int32)
    old_to_new_index_map[available_points] = np.arange(0, available_points.shape[0])
    brighter_color = tuple([c*1.8 for c in color])
    color = (redistribute_rgb(brighter_color[0], brighter_color[1], brighter_color[2]))

    instlabel_exists = bbox_dict==None
    labelanchor_x, labelanchor_y = 10000, 10000
    try:
        #Vert Beams
        for i in range(0, 4):
            if i in available_points and i+4 in available_points:
                si = old_to_new_index_map[i]
                ei = old_to_new_index_map[i+4]

                p1 = tuple(bbox_2d_pts[si])
                p2 = tuple(bbox_2d_pts[ei])
                _, p1, p2 = cv2.clipLine((0,0,img_w,img_h), p1, p2)
                
                labelanchor_x = min(labelanchor_x, p2[0])
                labelanchor_y = min(labelanchor_y, p2[1])
               
                image = cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)
        # Horizontal Beams
        for i in range(0, 8, 2):
            if i in available_points and i+1 in available_points:
                si = old_to_new_index_map[i]
                ei = old_to_new_index_map[i+1]

                p1 = tuple(bbox_2d_pts[si])
                p2 = tuple(bbox_2d_pts[ei])
                _, p1, p2 = cv2.clipLine((0,0,img_w,img_h), p1, p2)
                image = cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)

        # misc beams
        for i in range(0, 8, 4):
            if i in available_points and i+3 in available_points:
                si = old_to_new_index_map[i]
                ei = old_to_new_index_map[i+3]

                p1 = tuple(bbox_2d_pts[si])
                p2 = tuple(bbox_2d_pts[ei])
                _, p1, p2 = cv2.clipLine((0,0,img_w,img_h), p1, p2)
                image = cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)

        # misc beams
        for i in range(1, 8, 4):
            if i in available_points and i+1 in available_points:
                si = old_to_new_index_map[i]
                ei = old_to_new_index_map[i+1]

                p1 = tuple(bbox_2d_pts[si])
                p2 = tuple(bbox_2d_pts[ei])
                _, p1, p2 = cv2.clipLine((0,0,img_w,img_h), p1, p2)
                image = cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)

        if not instlabel_exists:
            image = draw_instlabel(image, bbox_dict, labelanchor_x, labelanchor_y)
            instlabel_exists = True
    except Exception as e:
        print(e)

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

    in_fov_mask = np.abs(angles_vec[:,0]) <= 1.57 #0.785398

    return in_fov_mask
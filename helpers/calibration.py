import os
import yaml
import numpy as np

def load_extrinsic_matrix(extrinsic_file):
    """
    Load extrinsic calibration from file and convert it to a 4x4 homogeneous matrix.

    Parameters:
        calib_ext_file : str
            File path to the extrinsic calibration YAML file. See DATA_REPORT.md for expected structure

    Returns:
        numpy.ndarray
            4x4 homogeneous matrix representing the extrinsic calibration.

    Notes:
        The function reads the YAML file at `calib_ext_file` and extracts the extrinsic matrix information.
        If the matrix is represented with 'R' (rotation) and 'T' (translation) keys, it constructs the 4x4 matrix.
        Otherwise, it constructs the matrix directly from the 'data' field in the YAML file.
    """
    calib_ext = open(extrinsic_file, 'r')
    calib_ext = yaml.safe_load(calib_ext)['extrinsic_matrix']
    if "R" in calib_ext.keys() and "T" in calib_ext.keys():
        ext_homo_mat = np.eye(4)
        ext_homo_mat[:3, :3] = np.array(calib_ext['R']['data']).reshape(
            calib_ext['R']['rows'], calib_ext['R']['cols']
        )
        ext_homo_mat[:3, 3] = np.array(calib_ext['T'])
    else:
        ext_homo_mat    = np.array(calib_ext['data']).reshape(
            calib_ext['rows'], calib_ext['cols']
        )
    return ext_homo_mat

def load_camera_params(intrinsic_file):
    with open(intrinsic_file, 'r') as f:
        params = yaml.safe_load(f)

        intrinsic_matrix = np.array(params['camera_matrix']['data']).reshape(
            params['camera_matrix']['rows'], params['camera_matrix']['cols']
        )

        distortion_coeffs = np.array(params['distortion_coefficients']['data'])

        image_width = params['image_width']
        image_height = params['image_height']

    return intrinsic_matrix, distortion_coeffs, (image_width, image_height)

def get_calibration_info(filepath):
    filename = filepath.split('/')[-1]
    filename_prefix = filename.split('.')[0]
    filename_split = filename_prefix.split('_')

    calibration_info = None
    src, tar = filename_split[1], filename_split[-1]
    if len(filename_split) > 3:
        #Sensor to Sensor transform
        extrinsic = yaml.safe_load(open(filepath, 'r'))
        calibration_info = extrinsic
    else:
        #Intrinsic transform
        intrinsic = yaml.safe_load(open(filepath, 'r'))
        calibration_info = intrinsic
    
    return calibration_info, src, tar

def load_cam_calib_from_path(calibrations_path):
    """
    Input:
        calibrations_path: path to calibration directory
    Output:
        cam_calibrations: dictionary of camera calibrations (keys below)
            "cam0_intrinsics"
            "cam1_intrinsics"
            "cam2_intrinsics"
            "cam3_intrinsics"
            "cam0_cam1"
            "cam4_intrinsics"
            "cam3_cam4"
    """
    calibration_fps = [os.path.join(calibrations_path, file) for file in os.listdir(calibrations_path) if file.endswith(".yaml")]

    cam_calibrations = {}
    for calibration_fp in calibration_fps:
        cal, src, tar = get_calibration_info(calibration_fp)

        if 'cam' in src:
            cal_id = "%s_%s"%(src, tar)
            cam_id = src[-1]

            if cal_id not in cam_calibrations.keys():
                cam_calibrations[cal_id] = {}
                cam_calibrations[cal_id]['cam_id'] = cam_id

            cam_calibrations[cal_id].update(cal)
    return cam_calibrations
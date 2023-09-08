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
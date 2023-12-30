import os
import os.path as osp
import sys
import pdb
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import ruamel.yaml

# For imports
sys.path.append(os.getcwd())

from helpers.calibration import load_extrinsic_matrix

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sequence', type=int, default=0, help="Sequence to process")


CAM0_FILENAME = "calib_cam0_intrinsics.yaml"
CAM1_FILENAME = "calib_cam1_intrinsics.yaml"
OS1_CAM0_FILENAME = "calib_os1_to_cam0.yaml"
OS1_CAM1_FILENAME = "calib_os1_to_cam1.yaml"
CAM0_CAM1_FILENAME = "calib_cam0_to_cam1.yaml"

def my_represent_float(self, data):
    if 0 < abs(data) < 1e-5:
        return self.represent_scalar(u'tag:yaml.org,2002:float', '{:.15f}'.format(data).rstrip('0').rstrip('.'))
    else:
        # Default representation for other cases (including 0.0, 1.0, etc.)
        return self.represent_scalar(u'tag:yaml.org,2002:float', repr(data))

yaml = ruamel.yaml.YAML(typ='safe', pure=True)
yaml.width = 2
yaml.representer.add_representer(float, my_represent_float)

def save_calibs(calib_dict, calib_dict_path):
    calib_dict_keys = calib_dict.keys()

    calib_yaml_dict = {}
    for key in calib_dict_keys:
        entry_name = key
        entry_data = calib_dict[key]
        entry_rows = entry_data.shape[0]
        entry_cols = entry_data.shape[1]
        calib_yaml_dict[entry_name] = {
            'rows': entry_rows,
            'cols': entry_cols,
            'data': entry_data.reshape(-1).tolist()
        }

    with open(calib_dict_path, 'w') as yaml_file:
        yaml.dump(calib_yaml_dict, yaml_file)

def compute_lidar_to_rect(T_lidar_to_cam_ext, R, P):
    T_lidar_cam = np.eye(4)
    T_lidar_cam[:3, :3] = T_lidar_to_cam_ext[:3, :3]
    T_lidar_cam[:3, 3] = T_lidar_to_cam_ext[:3, 3]
    T_canon = np.eye(4)
    T_canon[:3, :3] = R

    T_lidar_to_rect = P @ T_canon @ T_lidar_cam

    return T_lidar_to_rect

def process_single_trajectory(indir, outdir, trajectory):
    trajectory = str(trajectory)
    calib_dir = osp.join(indir, str(trajectory))
    assert osp.exists(indir), "Calibration directory does not exist %s"%calib_dir
    out_calib_dir = osp.join(outdir, trajectory)
    if not osp.exists(out_calib_dir):
        print("Output directory does not exist %s, creating" % out_calib_dir)
        os.makedirs(out_calib_dir)

    os1_cam0_path = osp.join(calib_dir, OS1_CAM0_FILENAME)
    cam0_cam1_path = osp.join(calib_dir, CAM0_CAM1_FILENAME)
    cam0_intr_path = osp.join(calib_dir, CAM0_FILENAME)
    cam1_intr_path = osp.join(calib_dir, CAM1_FILENAME)
    cam0_intr = yaml.load(open(cam0_intr_path, 'r'))
    cam1_intr = yaml.load(open(cam1_intr_path, 'r'))

    # Compute LiDAR to undistorted camera transform
    R1 = np.array(cam0_intr['rectification_matrix']['data']).reshape(
        cam0_intr['rectification_matrix']['rows'], cam0_intr['rectification_matrix']['cols']
    )
    R2 = np.array(cam1_intr['rectification_matrix']['data']).reshape(
        cam1_intr['rectification_matrix']['rows'], cam1_intr['rectification_matrix']['cols']
    )
    P1 = np.array(cam0_intr['projection_matrix']['data']).reshape(
        cam0_intr['projection_matrix']['rows'], cam0_intr['projection_matrix']['cols']
    )
    P2 = np.array(cam1_intr['projection_matrix']['data']).reshape(
        cam1_intr['projection_matrix']['rows'], cam1_intr['projection_matrix']['cols']
    )
    T_os1_to_cam0 = load_extrinsic_matrix(os1_cam0_path)
    T_cam0_to_cam1 = load_extrinsic_matrix(cam0_cam1_path)
    T_os1_to_cam1 = T_cam0_to_cam1 @ T_os1_to_cam0
    
    # Compute LiDAR to undistorted rectified camera transforms
    T_lidar_to_rect0 = compute_lidar_to_rect(T_os1_to_cam0, R1, P1)
    T_lidar_to_rect1 = compute_lidar_to_rect(T_os1_to_cam1, R2, P2)

    os1_to_cam0_dict = {
        "extrinsic_matrix": T_os1_to_cam0,
        "projection_matrix": T_lidar_to_rect0
    }
    os1_to_cam1_dict = {
        "extrinsic_matrix": T_os1_to_cam1,
        "projection_matrix": T_lidar_to_rect1
    }

    np.set_printoptions(suppress=True, precision=8)

    # Save ouster to camera calibrations to matrix
    out_calib0_path = osp.join(out_calib_dir, OS1_CAM0_FILENAME)
    save_calibs(os1_to_cam0_dict, out_calib0_path)
    
    out_calib1_path = osp.join(out_calib_dir, OS1_CAM1_FILENAME)
    save_calibs(os1_to_cam1_dict, out_calib1_path)

def main(args):
    indir = "/robodata/arthurz/Datasets/CODa_v2/calibrations"
    outdir = "/robodata/arthurz/Datasets/CODa_v2/calibrations"
    sequence = args.sequence
    process_single_trajectory(indir, outdir, sequence)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
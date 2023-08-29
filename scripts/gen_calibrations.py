import os
import os.path as osp
import sys
import pdb
import yaml
import numpy as np

# For imports
sys.path.append(os.getcwd())

from helpers.geometry import load_ext_calib_to_mat

OS1_CAM0_FILENAME = "calib_os1_to_cam0.yaml"
OS1_CAM1_FILENAME = "calib_os1_to_cam1.yaml"
CAM0_CAM1_FILENAME = "calib_cam0_to_cam1.yaml"

def write_yaml(homo_mat, outdir):
    data = homo_mat.reshape(-1,).tolist()
    # data_str = "[ %6.15f,\t %6.15f,\t %6.15f,\t %6.15f,\n \
    #             \t\t\t\t %6.15f,\t %6.15f,\t %6.15f,\t %6.15f,\n \
    #             \t\t\t\t %6.15f,\t %6.15f,\t %6.15f,\t %6.15f,\n \
    #             \t\t\t\t %6.15f,\t %6.15f,\t %6.15f,\t %6.15f ]"
    # data_str = data_str % tuple(data)
    # import pdb; pdb.set_trace()

    YAML_STR = "extrinsic_matrix:\n  cols: 4\n  rows: 4\n  data: [ %6.15f,  %6.15f,  %6.15f,  %6.15f,\n \
          %6.15f,  %6.15f,  %6.15f,  %6.15f,\n \
          %6.15f,  %6.15f,  %6.15f,  %6.15f,\n \
          %6.15f,  %6.15f,  %6.15f,  %6.15f ]\n"

    yaml_data_str = YAML_STR % tuple(data)

    # yaml_data = {
    #     'extrinsic_matrix': {
    #         'rows': 4,
    #         'cols': 4,
    #         'data': data
    #     }
    # }
    yaml_path = osp.join(outdir, OS1_CAM1_FILENAME)
    print("Writing trajectory %s os1 to cam1 calibration to file..."%yaml_path)
    # import pdb; pdb.set_trace()
    with open(yaml_path, 'w') as file:
        file.write(yaml_data_str)
        # yaml.safe_dump(yaml_data, file)
    file.close()

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

    os1_cam0_mat = load_ext_calib_to_mat(os1_cam0_path)
    cam0_cam1_mat = load_ext_calib_to_mat(cam0_cam1_path)

    os1_cam1_mat = cam0_cam1_mat @ os1_cam0_mat
    
    write_yaml(os1_cam1_mat, out_calib_dir)
    

def main():
    indir = "/robodata/arthurz/Datasets/CODa/calibrations"
    outdir = "/robodata/arthurz/Datasets/CODa_dev/calibrations"
    trajectory = 3
    process_single_trajectory(indir, outdir, trajectory)



if __name__ == '__main__':
    main()
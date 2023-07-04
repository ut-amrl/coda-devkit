# import pdb; pdb.set_trace()
import os
import sys
import argparse
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# sys.path.append(os.getcwd())
# from helpers.geometry import *

parser = argparse.ArgumentParser()
parser.add_argument('--traj', default="0",
                    help="Generated file for HitL")

# def pose_to_homo_mat(pose, np_n_of_val_scans):
#     trans    = pose[:, 1:3]
#     quat     = np.array([pose[:, 5], pose[:, 6], pose[:, 7], pose[:, 4]]).transpose()

#     # rot_mat  = R.from_quat(quat).as_matrix()
#     # rot_mat  = rot_mat[:, :2, :2]
#     rot_mat_euler = R.from_quat(quat).as_euler('zxy', degrees=False)
#     np_cos = np.cos(rot_mat_euler[:, 0]).reshape(-1, 1)
#     np_sin = np.sin(rot_mat_euler[:, 0]).reshape(-1, 1)
#     rot_mat  = np.hstack((np_cos, -np_sin, np_sin, np_cos)).reshape(-1, 2, 2)

#     d_zeros  = np.repeat([0, 0], len(rot_mat)).reshape(-1, 2)
#     d_zeros  = np.expand_dims(d_zeros, axis=1)
#     rot_mat  = np.concatenate((rot_mat, d_zeros), 1)

#     trans    = np.expand_dims(trans, axis=2)
#     d_ones   = np.repeat(1, len(trans)).reshape(-1, 1)
#     d_ones   = np.expand_dims(d_ones, axis=2)
#     trans    = np.concatenate((trans, d_ones), 1)

#     homo_mat = np.concatenate((rot_mat, trans), 2)
#     homo_mat = np.repeat(homo_mat, np_n_of_val_scans, axis=0)
#     return homo_mat  # (# of scans, 3, 3)

# def correct_obs(homo_mat, obs):
#     '''
#     correct observation x & y with homo matrix
#     '''
#     obs = obs.reshape(-1, 3, 1)     # (# of scans, 3, 1)
#     corrected_obs = np.matmul(homo_mat, obs)
#     corrected_obs = np.transpose(corrected_obs, [0, 2, 1])
#     corrected_obs = corrected_obs.reshape(-1, 3)
    
#     return corrected_obs # (# of scans, 3)

def main(args):
    traj  = args.traj
    fdir    = "/home/jihwan98/coda/HitL"
    fpath   = os.path.join(fdir, traj + ".bin")

    # reads in the generated file
    bin_np  = np.fromfile(fpath).reshape(-1, 16)
    pose_x, pose_y, obs_x, obs_y = bin_np[:, 0], bin_np[:, 1], bin_np[:, 3], bin_np[:, 4]

    print("Start plotting")
    fig = plt.figure()
    # plots the 2d observations and poses on a grid
    plt.scatter(obs_x,  obs_y,  marker=".", s=10, label='Observation')
    plt.scatter(pose_x, pose_y, marker=".", s=50, label='Pose Estimation')
    plt.grid(visible=True)
    plt.legend(loc="upper right")

    # save as png
    fname = os.path.join(fdir, traj + ".png")
    print("Saving image")
    plt.savefig(fname)
    
    return

if __name__ == "__main__":
    import time
    start_time = time.time()
    args = parser.parse_args()
    main(args)
    print("--- %s seconds ---" % (time.time() - start_time))

# import pdb; pdb.set_trace()
import os
import sys
import argparse
import numpy as np
import math
import time
from sklearn.neighbors import KDTree

parser = argparse.ArgumentParser()
parser.add_argument('--traj', default="/robodata/arthurz/Datasets/CODa",
                    help="CODa directory")

# # For imports
# sys.path.append(os.getcwd())

# from helpers.sensors import *

def read_bin(bin_path, keep_intensity=True):
    OS1_POINTCLOUD_SHAPE = [1024, 128, 3]
    num_points = OS1_POINTCLOUD_SHAPE[0]*OS1_POINTCLOUD_SHAPE[1]
    bin_np = np.fromfile(bin_path, dtype=np.float32).reshape(num_points, -1)

    # # write a single row to a open3d viewer
    # bin_np = bin_np.reshape(1028, 128, -1) # check if this 

    if not keep_intensity:
        bin_np = bin_np[:, :3]
    return bin_np

# https://stackoverflow.com/questions/5782658/extracting-yaw-from-a-quaternion
def calc_yaw(qw, qx, qy, qz):
    yaw = math.atan2(2.0*(qy*qz + qw*qx), qw*qw - qx*qx - qy*qy + qz*qz)
    return yaw

def get_poses(pose_path, HitL_np):

    pose_np = np.loadtxt(pose_path).reshape(-1, 8)
    np_tmp  = np.zeros((pose_np.shape[0], 3))

    cnt = 0
    for pose in pose_np:
        _, x, y, z, qw, qx, qy, qz = pose
        yaw = calc_yaw(qw, qx, qy, qz)
        np_tmp[cnt, :3] = np.array([x, y, yaw])
        cnt += 1

    HOR_RES = 1024

    HitL_np[:, :3] = np.repeat(np_tmp, repeats=HOR_RES, axis=0)

    return HitL_np

def get_normal(closest_points):
    
    """ test data 
        a = [-1, 1, 2, -4, 2, 2, -2, 1, 5, 0, 0, 0, 1, 0, 0, 0, 1, 0, -1, 1, 2, -4, 2, 2, -2, 1, 5]
        closest_points = np.array(a, dtype='f').reshape(3, 3, 3)
        result: (3, 9, 1)
    """

    p0 = closest_points[:, 0, :]
    p1 = closest_points[:, 1, :]
    p2 = closest_points[:, 2, :]
    v01 = np.subtract(p0, p1)
    v02 = np.subtract(p0, p2)
    nv  = np.cross(v01, v02)

    return nv[:, :2]

def cov_gen(x_sigma = 0.01, y_sigma = 0.01, th_sigma = 0.01):
    # covariance matrix - Normal Distribution
    x_unc  = np.random.normal(0, x_sigma,  1)
    y_unc  = np.random.normal(0, y_sigma,  1)
    th_unc = np.random.normal(0, th_sigma, 1)
    unc_lst = [x_unc, y_unc, th_unc]
    # (1, 9) Covar_xx ~ Covar_thth
    cov_arr = []
    for a in unc_lst:
        for b in unc_lst:
            cov_arr.append(a*b)
    cov_arr = np.array(cov_arr).reshape(1, 9)

    return cov_arr


def main(args):
    
    VER_RES, HOR_RES = 128, 1024

    trajectory = args.traj
    pose_path = f"/robodata/arthurz/Datasets/CODa/poses/dense/{trajectory}.txt"
    bin_dir   = f"/robodata/arthurz/Datasets/CODa/3d_raw/os1/{trajectory}/"
    
    # number of lidar frames
    n_of_bin_files = len([entry for entry in os.listdir(bin_dir) if os.path.isfile(os.path.join(bin_dir, entry))])
    print(f"n_of_bin_files: {n_of_bin_files}")
    HitL_np = np.zeros((n_of_bin_files * HOR_RES, 16)) # create an array of zeros

    # add pose terms
    print("Geting Pose") # ~3s
    HitL_np = get_poses(pose_path, HitL_np)
    print("Pose Done.\n")

    # add obs & normal terms
    print("Getting Obs, Normal")
    tmp_obs_normal = np.zeros((n_of_bin_files, 4))
    # for frame in range(n_of_bin_files):
    for frame in range(n_of_bin_files):
        if (frame % 1000) == 0:
            print(f"Processing frame: {frame}")

        bin_file  = f"3d_raw_os1_{trajectory}_{frame}.bin"
        bin_path  = os.path.join(bin_dir, bin_file)
        pc_np     = read_bin(bin_path, False).reshape(1024, 128, 3)
        # pc_np = pc_np[:3, :3, :] # for testing

        pc_np_flattened = pc_np.reshape(-1, 3)
        tree = KDTree(pc_np_flattened, leaf_size=2) # for an efficient closest points search

        # 1) filter Z Channel based on thresholds
        z_lower, z_upper = 2.0, 6.0
        z_mask = np.logical_or( (pc_np[:, :, 2] < z_lower), (pc_np[:, :, 2] > z_upper) )
        pc_np_filtered = np.copy(pc_np) # o.w. overwrite pc_np
        pc_np_filtered[z_mask, :] = np.inf # to help finding the min Euc. dist.

        # 2) find the min ||p_i||
        pc_np_norm = np.linalg.norm(pc_np_filtered, axis=-1)
        pc_np_min_idx = np.argmin(pc_np_norm, axis=-1) 
        dummy_idx = np.arange(len(pc_np_min_idx))
        # pc_np_min = pc_np[dummy_idx, pc_np_min_idx[dummy_idx], :]
        pc_np_min = pc_np[dummy_idx, pc_np_min_idx, :]

        # 3) find two nearest points
        _, ind = tree.query(pc_np_min, k=3)

        # closest_points = pc_np_flattened[ind[np.arange(1024), :]]
        closest_points = pc_np_flattened[ind]
        
        # 4) find the normal vector to the plane
        norm_xy = get_normal(closest_points)  # (1024, 2)

        HitL_np[1024*frame:1024*(frame+1), 3:5] = pc_np_min[:, :2]
        HitL_np[1024*frame:1024*(frame+1), 5:7] = norm_xy
    print("Obs and Normal Done.")

    # add covar terms
    print("Geting Covar") # ~ 5s
    tmp_np = np.zeros((n_of_bin_files, 9))
    for i in range(n_of_bin_files):
        if i == 0:
            tmp_np[i, :] = cov_gen(0, 0, 0)
        else:
            tmp_np[i, :] = cov_gen(0.01, 0.001, 0.01)
        
    HitL_np[:, 7:] = np.repeat(tmp_np, repeats=HOR_RES, axis=0)
    print("Covar Done.")
    
    # save as bin file
    HitL_np_flattened = HitL_np.reshape(-1, )
    save_dir  = os.path.join("./", "HitL")
    fpath_out = os.path.join(save_dir, trajectory + ".bin")
    HitL_np_flattened.tofile(fpath_out)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
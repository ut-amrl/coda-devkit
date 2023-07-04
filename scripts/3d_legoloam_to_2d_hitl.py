# import pdb; pdb.set_trace()
import os
import sys
import argparse
import numpy as np
import math
import time

from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import KDTree

parser = argparse.ArgumentParser()
parser.add_argument('--traj', default="/robodata/arthurz/Datasets/CODa",
                    help="CODa directory")

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
    yaw = np.arctan2(2.0*(qy*qz + qw*qx), qw*qw - qx*qx - qy*qy + qz*qz)
    return yaw

# def get_pose(pose_of_frame, n_of_val_scans):
#     _, x, y, z, qw, qx, qy, qz = pose_of_frame
#     yaw = calc_yaw(qw, qx, qy, qz)
#     np_tmp = np.array([x, y, yaw]).reshape(-1, 3)
#     return np.repeat(np_tmp, repeats=n_of_val_scans, axis=0)

def get_pose(pose_np, np_n_of_val_scans):
    _, x, y, z, qw, qx, qy, qz = pose_np.transpose()
    yaw = calc_yaw(qw, qx, qy, qz)
    np_tmp = np.concatenate([x.reshape(-1,1), y.reshape(-1,1), yaw.reshape(-1,1)], axis=1).reshape(-1, 3)
    return np.repeat(np_tmp, repeats=np_n_of_val_scans, axis=0)

def pose_to_homo_mat(pose):
    trans = pose[1:4]
    quat  = np.array([pose[5], pose[6], pose[7], pose[4]]) # qx, qy, qz, qw - order confirmed
    rot_mat  = R.from_quat(quat).as_matrix()
    homo_mat = np.eye(4, dtype=np.float64)
    homo_mat[:3, :3] = rot_mat
    homo_mat[:3, 3] = trans
    return homo_mat # (4, 4)

def correct_obs(homo_mat, obs):
    '''
    correct observation x & y with homo matrix
    '''
    obs = np.concatenate( (obs, np.ones((len(obs), 1))), axis=-1 ).transpose()
    corrected_obs = np.matmul(homo_mat, obs)[:3, :].transpose()
    return corrected_obs # (# of scans, 3)

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

def cov_gen(n_of_bin, np_n_of_val_scans, xxyy_sigma=0.0007, thth_sigma=0.000117, xyth_sigma=0.002, xyyx_sigma=0.000008):
    xxyy_unc = np.random.normal(0, xxyy_sigma, n_of_bin-1)
    thth_unc = np.random.normal(0, thth_sigma, n_of_bin-1)
    xyth_unc = np.random.normal(0, xyth_sigma, n_of_bin-1)
    xyyx_unc = np.random.normal(0, xyyx_sigma, n_of_bin-1)
    cov_mat  = np.concatenate([xxyy_unc, xyyx_unc, xyth_unc, xyyx_unc, xxyy_unc, xyth_unc, xyth_unc, xyth_unc, thth_unc])
    cov_mat  = cov_mat.reshape(9, -1).transpose()
    cov_mat  = np.concatenate([np.zeros(9).reshape(1, 9), cov_mat], axis=0)
    return np.repeat(cov_mat, repeats=np_n_of_val_scans, axis=0)


def main(args):
    
    trajectory = args.traj
    pose_path = f"/robodata/arthurz/Datasets/CODa_dev/poses/dense/{trajectory}.txt"
    bin_dir   = f"/robodata/arthurz/Datasets/CODa/3d_raw/os1/{trajectory}/"
    
    pose_np = np.loadtxt(pose_path).reshape(-1, 8)

    # number of lidar frames
    # n_of_bin_files = len([entry for entry in os.listdir(bin_dir) if os.path.isfile(os.path.join(bin_dir, entry))])
    print("\n\n" + "---"*15)
    print(f"\nNumber of lidar frames: {len(pose_np)}\n")
    
    HitL_np = np.array([])
    np_n_of_val_scans = np.zeros(len(pose_np), dtype=int)

    np_bin = np.zeros((len(pose_np)*1024, 16))

    local_time = time.time()
    # for frame in range(len(pose_np)):
    for frame in range(2000):
        pose_of_frame = pose_np[frame, :]

        if (frame % 100)==0 and (frame != 0):
            print(f"---Processing frame: {frame:4} || {time.time() - local_time} seconds ---")
            local_time = time.time()

        bin_file  = f"3d_raw_os1_{trajectory}_{frame}.bin"
        bin_path  = os.path.join(bin_dir, bin_file)
        pc_np_org = read_bin(bin_path, False)

        # pc_np     = pc_np_org.reshape(1024, 128, 3) # reshape incorrectly
        # axis 0 - hortizontal channel (1024), axis 1 - vertical channel (128)
        pc_np     = pc_np_org.reshape(128, 1024, 3)
        pc_np     = np.transpose(pc_np, [1, 0, 2]) # (1024, 128, 3)
        # TEST - np.equal(pc_np_org[::1024], pc_np[0])

        pc_np_flattened = pc_np.reshape(-1, 3)
        tree = KDTree(pc_np_flattened, leaf_size=2) # for an efficient closest points search

        # 1.1) filter Z Channel based on thresholds
        z_lower, z_upper = 4.0, 6.0
        z_mask = np.logical_or( (pc_np[:, :, 2] < z_lower), (pc_np[:, :, 2] > z_upper) )
        pc_np_filtered = np.copy(pc_np)    # copy. o.w. overwrite pc_np
        pc_np_filtered[z_mask, :] = np.inf # to help finding the min Euc. dist.

        # 1.2) filter based on range (x-y Euc. dist.) thresholds
        r_lower, r_upper = 15.0, 18.0
        pc_np_norm_2d  = np.linalg.norm(pc_np_filtered[:, :, :2], axis=-1) # x-y Euc. dist.    
        r_mask     = np.logical_or( (pc_np_norm_2d < r_lower), (pc_np_norm_2d > r_upper) )   
        pc_np_filtered[r_mask, :] = np.inf

        # 2) find the min ||p_i||
        pc_np_norm_3d  = np.linalg.norm(pc_np_filtered, axis=-1)
        pc_np_min_idx  = np.argmin(pc_np_norm_3d, axis=-1) 
        pc_np_min_mask = (pc_np_min_idx != 0)
        n_of_val_scans = np.count_nonzero(pc_np_min_idx)
        np_n_of_val_scans[frame] = n_of_val_scans
        dummy_idx     = np.arange(len(pc_np_min_idx))
        pc_np_min     = pc_np[dummy_idx, pc_np_min_idx, :] # pc_np_min = pc_np[dummy_idx, pc_np_min_idx[dummy_idx], :]
        pc_np_min     = pc_np_min[pc_np_min_mask, :]
        
        if ( n_of_val_scans != 0 ): # skips to write if no valid pc in the frame
            # 3) find two nearest points
            _, ind = tree.query(pc_np_min, k=3)

            # closest_points = pc_np_flattened[ind[np.arange(1024), :]]
            closest_points = pc_np_flattened[ind]

            # 4) find the normal vector to the plane
            norm_xy = get_normal(closest_points)  # (n_of_val_scans, 2)
            
            end   = np.sum(np_n_of_val_scans)
            start = end - np_n_of_val_scans[frame]
            np_bin[start:end, 3:5] = correct_obs(pose_to_homo_mat(pose_of_frame), pc_np_min)[:, :2] # add corrected obs
            np_bin[start:end, 5:7] = norm_xy
            
    np_bin = np_bin[~np.all(np_bin == 0, axis=1)]
    np_bin[:, :3] = get_pose(pose_np, np_n_of_val_scans) # add pose
    np_bin[:, 7:] = cov_gen(n_of_bin=len(pose_np), np_n_of_val_scans=np_n_of_val_scans)

    # save as bin file
    HitL_np_flattened = np_bin.reshape(-1, )
    # import pdb; pdb.set_trace()
    save_dir  = os.path.join("./", "HitL")
    fpath_out = os.path.join(save_dir, trajectory + ".bin")
    HitL_np_flattened.tofile(fpath_out)

if __name__ == "__main__":
    import time
    start_time = time.time()
    args = parser.parse_args()
    main(args)
    print("--- Final: %s seconds ---" % (time.time() - start_time))
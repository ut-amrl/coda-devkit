# import pdb; pdb.set_trace()
import os
from os.path import join
import sys
import argparse
import numpy as np
import time
import json

import open3d as o3d
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import normalize
from multiprocessing import Pool

# For imports
sys.path.append(os.getcwd())
from helpers.sensors import set_filename_dir

# parser = argparse.ArgumentParser()
# parser.add_argument('--traj', default="0",
#                     help="number of trajectory, e.g. 1")

def read_bin(bin_path, keep_intensity=False):
    OS1_POINTCLOUD_SHAPE = [1024, 128, 3]
    num_points = OS1_POINTCLOUD_SHAPE[0]*OS1_POINTCLOUD_SHAPE[1]
    bin_np = np.fromfile(bin_path, dtype=np.float32).reshape(num_points, -1)
    if not keep_intensity:
        bin_np = bin_np[:, :3]
    return bin_np

def calc_yaw(qw, qx, qy, qz):
    quat_np = np.stack((qx, qy, qz, qw), axis=-1)
    euler_rot = R.from_quat(quat_np).as_euler('xyz', degrees=False)
    yaw = euler_rot[:, -1]
    return yaw

def yaw_to_homo(pose_np, yaw):
    trans = pose_np[:, 1:4]
    rot_mat = R.from_euler('z', yaw, degrees=True).as_matrix()
    tmp = np.expand_dims(np.eye(4, dtype=np.float64), axis=0)
    homo_mat = np.repeat(tmp, len(trans), axis=0)
    homo_mat[:, :3, :3] = rot_mat
    # homo_mat[:, :3, 3 ] = trans
    return homo_mat

def apply_hom_mat(pose_np, homo_mat, non_origin):
    _, x, y, z, _, _, _, _ = pose_np.transpose()
    x, y, z, ones = [p.reshape(-1, 1) for p in [x, y, z, np.ones(len(x))]]
    xyz1 = np.expand_dims(np.hstack((x, y, z, ones)), -1)

    if non_origin:
        xyz1_center = np.copy(xyz1)
        xyz1_center[:, :2] = xyz1[:, :2] - xyz1[0, :2]
        xyz1_center_rotated = np.matmul(homo_mat, xyz1_center)[:, :3, 0]
        xyz1_center_rotated[:, :2] = xyz1_center_rotated[:, :2] + xyz1[0, :2].reshape(1, -1)
        corrected_pose_np = xyz1_center_rotated
    else:
        corrected_pose_np = np.matmul(homo_mat, xyz1)[:, :3, 0]
    return corrected_pose_np

def correct_pose(pose_np, trajectory):
    dir = './json'
    fpath = os.path.join(dir, 'pose_correction.json')
    f = open(fpath, "r")
    pose_correction = json.load(f)
    f.close()

    JSON_NAMES = ["start_arr", "end_arr", "yaw_arr"]
    start_arr, end_arr, yaw_arr = [], [], []

    if trajectory in pose_correction.keys():
        traj_dict = pose_correction[trajectory]
        start_arr, end_arr, yaw_arr = traj_dict[JSON_NAMES[0]], traj_dict[JSON_NAMES[1]], traj_dict[JSON_NAMES[2]]

    corrected_pose = np.copy(pose_np)
    
    # handles multiple rotation
    for i in range(len(start_arr)): 
        start, end, yaw = start_arr[i], end_arr[i], yaw_arr[i]
        non_origin = False
        if start != 0:
            non_origin = True
        homo_mat = yaw_to_homo(corrected_pose[start:end, :], yaw)
        corrected_pose[start:end, 1:4] = apply_hom_mat(corrected_pose[start:end, :], homo_mat, non_origin)
    return corrected_pose

def get_pose(pose_np, n_pcs):
    _, x, y, z, qw, qx, qy, qz = pose_np.transpose()
    yaw = calc_yaw(qw, qx, qy, qz)
    np_tmp = np.concatenate([x.reshape(-1,1), y.reshape(-1,1), yaw.reshape(-1,1)], axis=1).reshape(-1, 3)
    return np.repeat(np_tmp, repeats=n_pcs, axis=0)

def pose_to_homo_mat(pose):
    trans = pose[1:4]
    quat  = np.array([pose[5], pose[6], pose[7], pose[4]]) # qx, qy, qz, qw - order confirmed
    rot_mat  = R.from_quat(quat).as_matrix()
    homo_mat = np.eye(4, dtype=np.float64)
    homo_mat[:3, :3] = rot_mat
    homo_mat[:3, 3] = trans
    return homo_mat

def correct_obs(homo_mat, obs):
    # correct observation x & y with homo matrix
    obs = np.concatenate( (obs, np.ones((len(obs), 1))), axis=-1 ).transpose()
    corrected_obs = np.matmul(homo_mat, obs)[:3, :].transpose()
    return corrected_obs # (# of scans, 3)

def get_normal(pcs, knn=32):
    obs = o3d.geometry.PointCloud()
    obs.points = o3d.utility.Vector3dVector(pcs)
    search = o3d.geometry.KDTreeSearchParamKNN(knn=knn)
    obs.estimate_normals(search_param=search)
    obs.orient_normals_consistent_tangent_plane(k=knn)
    norm = np.asarray(obs.normals)
    norm = normalize(norm[:, :2])
    return norm

def cov_gen(n_of_bin, n_pcs, xxyy_sigma=0.0007, thth_sigma=0.000117, xyth_sigma=0.002, xyyx_sigma=0.000008):
    xxyy_unc = np.random.normal(0, xxyy_sigma, n_of_bin-1)
    thth_unc = np.random.normal(0, thth_sigma, n_of_bin-1)
    xyth_unc = np.random.normal(0, xyth_sigma, n_of_bin-1)
    xyyx_unc = np.random.normal(0, xyyx_sigma, n_of_bin-1)
    cov_mat  = np.concatenate([xxyy_unc, xyyx_unc, xyth_unc, xyyx_unc, xxyy_unc, xyth_unc, xyth_unc, xyth_unc, thth_unc])
    cov_mat  = cov_mat.reshape(9, -1).transpose()
    cov_mat  = np.concatenate([np.zeros(9).reshape(1, 9), cov_mat], axis=0)
    return np.repeat(cov_mat, repeats=n_pcs, axis=0)

def z_filter(pc_np):
    LIDAR_HEIGHT = 0.8 # meters
    ZBAND_HEIGHT = 0.5 # meters
    ZMIN_OFFSET  = 1.9 # meters
    zmin = ZMIN_OFFSET - LIDAR_HEIGHT
    zmax = zmin+ZBAND_HEIGHT
    z_mask = np.logical_and( (pc_np[:, 2] > zmin), (pc_np[:, 2] < zmax) )
    pc_np_filtered = np.copy(pc_np) # copy original pc. o.w. overwrite pc_np
    pc_np_filtered = pc_np_filtered[z_mask]
    return pc_np_filtered

def r_filter(pc_np_filtered):
    rmin, rmax = 10.0, 100.0
    pc_np_norm_2d  = np.linalg.norm(pc_np_filtered[:, :2], axis=-1) # x-y Euc. dist.    
    r_mask = np.logical_and( (pc_np_norm_2d > rmin), (pc_np_norm_2d < rmax) )   
    pc_np_filtered = pc_np_filtered[r_mask]
    return pc_np_filtered
    
def process_pc(args):
    pose_of_frame, frame, bin_path, outdir = args
    pc_np = read_bin(bin_path)
    
    pc_np = pc_np.reshape(128, 1024, 3)
    pc_np = np.transpose(pc_np, [1, 0, 2]) # result (1024, 128, 3)
    pc_np = pc_np[:, ::8, :] # downsample: 1024 -> 512 and 128 -> 32
    pc_np = pc_np.reshape(-1, 3)
    
    # 1.1) filter Z Channel based on thresholds
    pc_np_filtered = z_filter(pc_np)
    # 1.2) filter based on range (x-y Euc. dist.) thresholds
    # pc_np_filtered = r_filter(pc_np_filtered)
    
    obs_xy  = correct_obs(pose_to_homo_mat(pose_of_frame), pc_np_filtered)

    # Dump frame to npy file
    np_bin_path = join(outdir, "np_bin_%i.npy"%frame)
    np.save(np_bin_path, obs_xy)
    
    # print(f"Processed frame {str(frame)}")

def find_closest_frame(pose_np, timestamp_np, indir, trajectory):
    bin_path_list, frame_list = [], []
    for _, pose in enumerate(pose_np):
        pose_ts = pose[0]
        closest_lidar_frame = np.searchsorted(timestamp_np, pose_ts, side='left')
        frame_list.append(closest_lidar_frame)
        bin_path = set_filename_dir(indir, "3d_raw", "os1", trajectory, closest_lidar_frame, include_name=True)
        bin_path_list.append(bin_path)

    return bin_path_list, frame_list

def get_obs(frame_list, outdir):
    
    obs   = np.zeros((len(frame_list)*1024*10, 3))
    n_pcs = np.zeros(len(frame_list), dtype=int)

    start, end = 0, 0
    for i in range(len(frame_list)): # Load in pcs for each frame
        frame = frame_list[i]
        np_bin_path = join(outdir, "np_bin_%i.npy"%frame)

        # Load and assign processed point clouds to unified file
        np_bin_single = np.load(np_bin_path)
        n_pcs[i] = len(np_bin_single)
        end = start + n_pcs[i]

        obs[start:end] = np_bin_single

        # update start for next iteration
        start = end
    
    obs = obs[:np.sum(n_pcs)]

    return obs, n_pcs

def main(trajectory):
    trajectory = trajectory
    indir = "/home/jihwan98/coda/HitL/poses_cor"
    pose_path = os.path.join(indir, f"{trajectory}_cor.txt")
    indir = "/robodata/arthurz/Datasets/CODa_dev"
    # pose_path = f"{indir}/poses/{trajectory}.txt"
    ts_path   = f"{indir}/timestamps/{trajectory}_frame_to_ts.txt"
    outdir   = f"./cloud_to_laser/%s" % trajectory
    save_dir = os.path.join("./", "HitL", "Corrected_GDC")

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pose_np = np.loadtxt(pose_path).reshape(-1, 8)
    timestamp_np = np.loadtxt(ts_path).reshape(-1)

    bin_path_list, frame_list = find_closest_frame(pose_np, timestamp_np, indir, trajectory)

    # pose_np = correct_pose(pose_np, trajectory)
    
    pose_of_frame_list, outdir_list = [], []
    for frame in range(len(frame_list)):
        pose_of_frame_list.append(pose_np[frame, :])
        outdir_list.append(outdir)

    pool = Pool(processes=96)
    pool.map(process_pc, zip(pose_of_frame_list, frame_list, bin_path_list, outdir_list), chunksize=32)
    pool.close() 
    pool.join()
    
    # print("\nAggregating all pcs.\n")
    
    obs, n_pcs = get_obs(frame_list, outdir)

    np_bin = np.zeros((len(obs), 16))
    np_bin[:, :3]  = get_pose(pose_np, n_pcs)
    np_bin[:, 3:6] = obs
    # print("Normal vectors start.")
    # np_bin[:, 5:7] = get_normal(obs)
    # print("Normal vectors done.")
    np_bin[:, 7:]  = cov_gen(n_of_bin=len(pose_np), n_pcs=n_pcs)
    
    np_bin[:, :7] = np.around(np_bin[:, :7], decimals = 4)
    np_bin[:, 7:] = np.around(np_bin[:, 7:], decimals = 5)

    print("Saving text")
    fpath_out = os.path.join(save_dir, trajectory + ".txt")
    header = 'StarterMap\n1455656519.379815'
    np.savetxt(fpath_out, np_bin, delimiter=',', header=header, comments='')


if __name__ == "__main__":
    GDC  = [0,1,3,4,5,18,19]
    GUAD = [2,7,12,16,17,21]
    WCP  = [6,9,10,11,13,20,22]
    for i in GDC[:2]:
        print(f"Started Trajectory {i}")
        trajectory = str(i)
        main(trajectory)
        print("Done.\n")
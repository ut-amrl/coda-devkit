import sys, os
import numpy as np
import json
from numpy.linalg import inv
import open3d as o3d

from scipy.spatial.transform import Rotation as R

# For imports
sys.path.append(os.getcwd())
from scripts.pose_to_hitl import save_hitl_input

'''
def pose_to_homo_stacked(pose_np):

    homo_mat = np.tile(np.eye(4, dtype=np.float64), (len(pose_np), 1, 1))
    trans = pose_np[:, 1:4]
    quat = np.array([pose_np[:, 5], pose_np[:, 6], pose_np[:, 7], pose_np[:, 4]]).transpose()
    rot_mat = R.from_quat(quat).as_matrix()

    homo_mat[:, :3, :3] = rot_mat
    homo_mat[:, :3,  3] = trans
    return homo_mat

def ego_to_global(homo_mat, pose_np):
    _, x, y, z, _, _, _, _ = pose_np.transpose()
    x, y, z, ones = [p.reshape(-1, 1) for p in [x, y, z, np.ones(len(x))]]
    xyz1 = np.expand_dims(np.hstack((x, y, z, ones)), -1)
    pose_global = np.matmul(homo_mat, xyz1)[:, :3, 0]
    return pose_global

'''

def grab_json(trajectory):
    dir = './json'
    fpath = os.path.join(dir, 'pose_correction.json')
    f = open(fpath, "r")
    pose_correction = json.load(f) # dictionary
    f.close()

    JSON_NAMES = ["start_arr", "end_arr", "mode_arr", "angle_arr"]
    start_arr, end_arr, mode_arr, angle_arr = [], [], [], []

    if trajectory in pose_correction.keys():
        traj_dict = pose_correction[trajectory]
        start_arr, end_arr, mode_arr, angle_arr = traj_dict[JSON_NAMES[0]], traj_dict[JSON_NAMES[1]], traj_dict[JSON_NAMES[2]], traj_dict[JSON_NAMES[3]]
    return [start_arr, end_arr, mode_arr, angle_arr]

def pose_np_to_mat(pose_np):
    """
    INPUT
        pose_np  - (N, 8) Pose Numpy Arrays
    OUTPUT
        pose_mat - (N, 4, 4) Pose Matrix
    """
    pose_mat = np.tile(np.eye(4, dtype=np.float64), (len(pose_np), 1, 1)) # Initialize

    xyz  = pose_np[:, 1:4]
    quat = pose_np[:, 4:][:, [1, 2, 3, 0]] # (qw, qx, qy, qz) -> (qx, qy, qz, qw)  
    rot_mat = R.from_quat(quat).as_matrix()

    pose_mat[:, :3, 3]  = xyz
    pose_mat[:, :3, :3] = rot_mat
    return pose_mat

def pose_mat_to_np(pose_np, pose_mat):
    """
    INPUT
        pose_np  - (N, 8)    Pose Numpy Arrays
        pose_mat - (N, 4, 4) Pose Matrix
    OUTPUT
        pose_np_new - (N, 8) Pose Numpy Arrays after transformation
    """

    ts = pose_np[:, 0] # timestamp
    pose_np_new = np.zeros((len(pose_np), 8))
    
    xyz = pose_mat[:, :3, 3]      # (N, 3)   xyz matrix
    rot_mat = pose_mat[:, :3, :3] # (N, 3, ) rotaiton matrix
    r = R.from_matrix(rot_mat)
    quat = r.as_quat()[:, [3, 0, 1, 2]] # (N, 4) quaternion matrix | (qx, qy, qz, qw) ->(qw, qx, qy, qz)    

    pose_np_new[:, 0]   = ts
    pose_np_new[:, 1:4] = xyz
    pose_np_new[:, 4:]  = quat
    return pose_np_new

def get_rotation_mat(mode, angle):

    seq_dict = {"yaw": "z", "roll": "y", "pitch": "x"}
    seq = seq_dict[mode]
    r = R.from_euler(seq, angle, degrees=True)
    rot_mat = r.as_matrix() 
    return rot_mat

def apply_affine(pose_mat, start, end, mode, angle=0, trans=np.zeros((1, 3))):
    '''
    INPUT
        pose_mat - (N, 4, 4)
        start - starting idx
        end   - ending idx
        mode  - either yaw, roll or pitch
        angle - amount of rotation in [deg]
        trans - amount of translation in (x, y, z)
    OUTPUT
        pose_mat - (N, 4, 4)
    '''
    
    end = len(pose_mat) if end == -1 else end  # to differentiate global and local origin
    
    pose_mat_copy = pose_mat.copy()[start:end, :, :]
    
    rot_mat = get_rotation_mat(mode, angle)

    # Find affine matrix to multiply
    affine_mat = np.tile(np.eye(4, dtype=np.float64), (len(pose_mat_copy), 1, 1))
    affine_mat[:, :3, :3] = rot_mat
    affine_mat[:, :3,  3] = trans.reshape(1, -1)
    
    # Rotate about global origin (0, 0, 0) - cp_g = A * p_g
    if end == len(pose_mat):
        pose_mat = np.matmul(affine_mat, pose_mat)
        return pose_mat
    
    # Rotate about local origin - cp_g = (T_g_r)^-1 * A * T_g_r * p_g
    else:
        rot_mat = R.from_euler('z', 0, degrees=True).as_matrix() 
        trans = pose_mat_copy[0, :3, 3] # (x,y,z) starting point
        homo_mat = np.tile(np.eye(4, dtype=np.float64), (len(pose_mat_copy), 1, 1))
        homo_mat[:, :3, :3] = rot_mat
        homo_mat[:, :3,  3] = trans
        
        T_r_g = homo_mat      # Hmogenoues matrix from robot to global
        T_g_r = inv(homo_mat) # Inverse of homogenoues matrix from robot to global

        tmp = np.matmul(T_g_r, pose_mat_copy)
        tmp = np.matmul(affine_mat, tmp)
        tmp = np.matmul(T_r_g, tmp)
        pose_mat[start:end, :, :] = tmp
        return pose_mat

# Functions to save files
def saveas_txt(traj, pose_np):
    '''
    INPUT
        pose_np - (N, 8) standard pose format
    '''
    save_dir = f"./poses_cor/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f_out = os.path.join(save_dir, str(traj) + ".txt")
    fmt = ['%.6f'] + ['%.8f']*7
    np.savetxt(f_out, pose_np, delimiter=' ', fmt=fmt, comments='')
    print(f'\n[txt] Trajectory {traj} saved in {save_dir}')

def saveas_pcd(traj, route, pose_np, obs = False):
    save_dir = f"./poses_cor/"
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(pose_np)
    if obs == True:
        fpath = os.path.join(save_dir, f"{traj}_obs.pcd")
    else:
        fpath = os.path.join(save_dir, f"{traj}.pcd")
    o3d.io.write_point_cloud(fpath, pcd, write_ascii=False)
    print(f'\n[PCD] Trajectory {traj} saved in {save_dir}')


def main(trajectory, route):
    indir = "/home/arnavbagad/coda-devkit/UNB"

    pose_path = f"{indir}/{trajectory}.txt"
    pose_global = np.loadtxt(pose_path).reshape(-1, 8) # original poses from LeGO-LOAM
    print(trajectory)
    start_arr, end_arr, mode_arr, angle_arr = grab_json(trajectory)

    pose_mat = pose_np_to_mat(pose_global)

    # Step 1) Apply rotation about global origin
    start, end, mode, angle = start_arr.pop(0), end_arr.pop(0), mode_arr.pop(0), angle_arr.pop(0)
    pose_mat = apply_affine(pose_mat, start, end, mode, angle, trans = np.zeros((1, 3)))

    # Step 2) Apply rotation about local origin
    while len(start_arr) != 0:
        tart, end, mode, angle = start_arr.pop(0), end_arr.pop(0), mode_arr.pop(0), angle_arr.pop(0)        
        pose_mat = apply_affine(pose_mat, start, end, mode, angle, trans = np.zeros((1, 3)))

    pose_global = pose_mat_to_np(pose_global, pose_mat)


    print("\n\nSaving poses.")
    saveas_txt(trajectory, pose_global)
    saveas_pcd(trajectory, route, pose_global[:, 1:4])

    print("\n\nSaving observations.")
    obs_xyz, _ = save_hitl_input(trajectory, route, hitl_corrected=False)
    saveas_pcd(trajectory, route, obs_xyz, obs=True)

if __name__ == "__main__":
    GDC  = [0,1,3,4,5,18,19]
    GUAD = [2,7,12,16,17,21]
    WCP  = [6,9,10,11,13,20,22]
    UNB  = [8,82,83,14,141,142,143,15,151,152,153] 
    route = "UNB"
    for trajectory in [141]:
        main(str(trajectory), route)
    print("\n\nGenerating all corrected poses done.\n\n")
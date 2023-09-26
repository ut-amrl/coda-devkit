# import pdb; pdb.set_trace()
import os
from os.path import join
import sys
import argparse
import numpy as np
import time
import json

from scipy.spatial.transform import Rotation as R
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped

import open3d as o3d

# For imports
sys.path.append(os.getcwd())
from helpers.sensors import set_filename_dir, read_bin
from helpers.geometry import pose_to_homo, find_closest_pose, densify_poses_between_ts
from helpers.visualization import pub_pc_to_rviz, pub_pose

parser = argparse.ArgumentParser()
parser.add_argument('--traj', default="0",
                    help="number of trajectory, e.g. 1")
parser.add_argument('--pose_tag', default="0",
                    help="Identify tag to publish pose")
parser.add_argument('--pub', default=1,
                    help="Publish global poses to rviz for visualization, set to 0 to stop publishing")
parser.add_argument('--save_to_file', default="-1",
                    help="Saves entire map to PCD file format, set to -1 if you don't want to save. Otherwise, specify directory to save pcd file to")




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
        homo_mat = yaw_to_homo(corrected_pose[:, :], yaw)
        corrected_pose[:, 1:4] = apply_hom_mat(corrected_pose[:, :], homo_mat, non_origin)
        # import pdb; pdb.set_trace();
    return corrected_pose

def apply_manual_correction_obs(trajectory, pc_np_filtered, n_pcs, pose_np):
    
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
    
    rot_mat = None
    manual_corrected_pc_np = None
    manual_pose_np = None

    # handles multiple rotation
    for i in range(len(start_arr)): 
        start, end, yaw = start_arr[i], end_arr[i], yaw_arr[i]
        r = R.from_euler('z', yaw, degrees=True)
        rot_mat = r.as_matrix()
        # print(f"Yaw Rotation: {yaw}")
        if i == 0:
            manual_corrected_pc_np = rot_mat @ np.transpose(pc_np_filtered)
            manual_corrected_pc_np = np.transpose(manual_corrected_pc_np)
            manual_pose_np = rot_mat @ np.transpose(pose_np)
            manual_pose_np = np.transpose(manual_pose_np)
        else:
            start_sum = np.sum(n_pcs[:start])
            end_sum = np.sum(n_pcs[:end+1]) + 1
            # import pdb; pdb.set_trace()

            # observations
            offset = manual_pose_np[start, :2] # (x, y) of offset
            temp_np = manual_corrected_pc_np[start_sum:end_sum]
            temp_np[:, :2] -= offset
            temp_np = np.transpose(rot_mat @ np.transpose(temp_np)) 
            temp_np[:, :2] += offset
            manual_corrected_pc_np[start_sum:end_sum] = temp_np

            # poses
            offset = manual_pose_np[start, :2] # (x, y) of offset
            temp_np = manual_pose_np[start_sum:end_sum]
            temp_np[:, :2] -= offset
            temp_np = np.transpose(rot_mat @ np.transpose(temp_np)) 
            temp_np[:, :2] += offset
            manual_pose_np[start_sum:end_sum] = temp_np

    return manual_corrected_pc_np


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

def find_closest_frame(pose_np, timestamp_np, indir, trajectory):
    bin_path_list, frame_list = [], []
    for _, pose in enumerate(pose_np):
        pose_ts = pose[0]
        closest_lidar_frame = np.searchsorted(timestamp_np, pose_ts, side='left')
        frame_list.append(closest_lidar_frame)
        bin_path = set_filename_dir(indir, "3d_raw", "os1", trajectory, closest_lidar_frame, include_name=True)
        bin_path_list.append(bin_path)

    return bin_path_list, frame_list

def main(args):

    trajectory  = args.traj
    pub_rviz    = int(args.pub)
    save_pcd    = str(args.save_to_file)!="-1"
    pose_tag    = args.pose_tag

    # trajectory = traj
    indir = "/home/arnavbagad/coda-devkit/UNB"
    pose_path = f"{indir}/{trajectory}.txt"
    # pose_path = f"{trajectory}.txt"
    ts_path = f"/robodata/arthurz/Datasets/CODa_dev/timestamps/{trajectory}.txt"
    outdir  = f"./cloud_to_laser/%s" % trajectory

    if not os.path.exists(outdir):
        os.makedirs(outdir)
            
    if pub_rviz:
        # Initialize ros publishing
        rospy.init_node('bin_publisher', anonymous=True)
        pc_pub = rospy.Publisher('/coda/ouster/lidar_packets', PointCloud2, queue_size=10)
        pose_pub = rospy.Publisher(f'/coda/pose/{pose_tag}', PoseStamped, queue_size=10)
        rate = rospy.Rate(75)

    pose_np = np.loadtxt(pose_path).reshape(-1, 8)
    timestamp_np = np.loadtxt(ts_path).reshape(-1)
    
    print("\n" + "---"*15)
    print(f"\nNumber of lidar frames: {len(pose_np)}\n")

    bin_path_list, frame_list = find_closest_frame(pose_np, timestamp_np, indir, trajectory)

    obs_xyz, n_pcs = get_obs(frame_list, outdir)
    obs_xyz_man = apply_manual_correction_obs(trajectory, obs_xyz, n_pcs, pose_np[:, 1:4])

    # Iterate through all poses
    LIDAR_HEIGHT = 0.8 # meters
    ZBAND_HEIGHT = 0.5 # meters
    ZMIN_OFFSET = 1.9 # meters
    # ZMIN_OFFSET = 2.5 # meters

    xyz = np.empty((0,3))

    start = 0
    end = 0

    for pose_idx, pose in enumerate(pose_np):

        pose_ts = pose[0]
        closest_lidar_frame = np.searchsorted(timestamp_np, pose_ts, side='left')
        lidar_ts = timestamp_np[closest_lidar_frame]

        pose_np = pose.reshape(1, -1)
        pose = correct_pose(pose_np, trajectory)



        import pdb; pdb.set_trace()

        
        pose = pose.reshape(-1,)


        # print("pose ", pose_idx)

        end += n_pcs[pose_idx]
        trans_lidar_np = obs_xyz_man[start:end, :]
        trans_lidar_np = np.float32(trans_lidar_np)

        if save_pcd is not True and pub_rviz:
            print(f'Publishing pose {pose_idx}')
            pub_pc_to_rviz(trans_lidar_np, pc_pub, lidar_ts)
            pub_pose(pose_pub, pose, pose[0])
            # import pdb; pdb.set_trace()
        else:
            if len(xyz) == 0:
                xyz = trans_lidar_np
            else:
                xyz = np.vstack([xyz, trans_lidar_np])

        if pub_rviz:
            rate.sleep()
        
        start = end
    
    if save_pcd:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        fpath = f"./raw_map/{trajectory}.pcd"
        o3d.io.write_point_cloud(fpath, pcd, write_ascii=False)
        

if __name__ == "__main__":
    start_time = time.time()
    args = parser.parse_args()
    main(args)
    # # main(args)

    # GDC  = [0,1,3,4,5,18,19]

    # for i in [0]:
    #     args.traj = str(i)
    #     args.pub = '0'
    #     args.save_to_file = './raw_map'
    #     main(args)
    

    print("--- Final: %s seconds ---" % (time.time() - start_time))
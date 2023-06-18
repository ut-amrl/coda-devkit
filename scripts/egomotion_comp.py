import os
import os.path as osp
from os.path import join
import sys
import pdb
import json
import copy
import argparse

import numpy as np

# For imports
sys.path.append(os.getcwd())

from helpers.sensors import *
from helpers.geometry import inter_pose, pose_to_homo, find_closest_pose, densify_poses_between_ts
from helpers.visualization import project_3dpoint_image

from multiprocessing import Pool
import tqdm

parser = argparse.ArgumentParser()

"""
This script is used to perform egomotion compensation on lidar point clouds using
SLAM estimated position.
"""

def compensate_frame(pc_path, frame_ts, s0_pose, s2_pose):
    """
    s0_pose - closest known pose before frame ts
    s2_pose - closet known pose after frame ts
    frame_ts - ts of when lidar collection was started
    """
    assert osp.exists(pc_path), "Given frame path does not exist %s" % pc_path

    bin_np = read_bin(pc_path)
    bin_np_shape = bin_np.shape
    bin_np = bin_np.reshape(128, 1024, -1)
    
    comp_pc = np.zeros_like(bin_np)

    # Uncomment to transform all points to first frame
    A_s0_orig = pose_to_homo(inter_pose(s0_pose, s2_pose, frame_ts)) # lidar start pose
    # pose_to_homo(s0_pose)
    A_orig_s0   = np.linalg.inv(A_s0_orig)
    sim_rel_ts = np.linspace(0, 0.099, 1024)
    for col_idx in range(bin_np.shape[1]):
        col_rel_ts = sim_rel_ts[col_idx]
        col_ts = frame_ts + col_rel_ts #*1e-9 # tuning threshold to account for camera hardware delays

        s1_pose   = inter_pose(s0_pose, s2_pose, col_ts)
        A_s1_orig = pose_to_homo(s1_pose)

        A_s1_s0     = A_orig_s0 @ A_s1_orig
        pc_col = bin_np[:, col_idx, :3]
        pc_col_homo = np.hstack( (pc_col, np.ones((pc_col.shape[0], 1))) )
        comp_pc_col_homo = (A_s1_s0 @ pc_col_homo.T).T

        comp_pc[:, col_idx, :3] = comp_pc_col_homo[:, :3]

    # Uncomment to transform all points to last frame
    # col_ts = frame_ts + 0.099
    # s0_pose= inter_pose(start_pose, end_pose, frame_ts)
    # A_s1_orig = pose_to_homo(s1_pose)
    # A_orig_s1   = np.linalg.inv(A_s1_orig)

    # sim_rel_ts = np.linspace(0, 0.099, 1024)
    # # sim_rel_ts = np.roll(sim_rel_ts, 512, axis=0)
    # for col_idx in range(bin_np.shape[1]):
    #     col_rel_ts = sim_rel_ts[col_idx] # bin_np[0, col_idx, -1]
    #     col_ts = frame_ts + col_rel_ts #*1e-9

    #     s0_pose   = inter_pose(s0_pose, s2_pose, col_ts)
    #     A_s0_orig = pose_to_homo(s0_pose)

    #     A_s0_s1     = A_orig_s1 @ A_s0_orig
    #     pc_col = bin_np[:, col_idx, :3]
    #     pc_col_homo = np.hstack( (pc_col, np.ones((pc_col.shape[0], 1))) )
    #     comp_pc_col_homo = (A_s0_s1 @ pc_col_homo.T).T

    #     comp_pc[:, col_idx, :3] = comp_pc_col_homo[:, :3]

    # Assumming row index is the issue
    # col_ts = frame_ts+0.05
    # s1_pose   = inter_pose(s0_pose, s2_pose, col_ts)
    # A_s1_orig = pose_to_homo(s1_pose)
    # A_orig_s1   = np.linalg.inv(A_s1_orig)

    # sim_rel_ts = np.linspace(0, 0.1, 128)
    # sim_rel_ts = np.roll(sim_rel_ts, 512, axis=0)
    # for row_idx in range(bin_np.shape[0]):
    #     # import pdb; pdb.set_trace()
    #     col_rel_ts = sim_rel_ts[row_idx] # bin_np[0, col_idx, -1]
    #     col_ts = frame_ts + col_rel_ts #*1e-9

    #     s0_pose   = inter_pose(s0_pose, s2_pose, col_ts)
    #     A_s0_orig = pose_to_homo(s0_pose)

    #     A_s0_s1     = A_orig_s1 @ A_s0_orig
    #     pc_row = bin_np[row_idx, :, :3]
    #     pc_row_homo = np.hstack( (pc_row, np.ones((pc_row.shape[0], 1))) )
    #     comp_pc_row_homo = (A_s0_s1 @ pc_row_homo.T).T
    #     comp_pc[row_idx, :, :3] = comp_pc_row_homo[:, :3]

    return comp_pc.reshape(bin_np_shape)

def compensate_single_trajectory_frame(args):
    indir, outdir, trajectory, pc_path, offset, camid = args

    ts_path = join(indir, "timestamps", "%s_frame_to_ts.txt"%trajectory)
    pose_path = join(indir, "poses", "%s.txt"%trajectory)

    # Load timestamps and dense poses file
    frame_ts_np = np.fromfile(ts_path, sep=' ').reshape(-1, 1)
    pose_np     = np.fromfile(pose_path, sep=' ').reshape(-1, 8)

    # Load calibrations
    calibextr_path = join(indir, "calibrations", trajectory, "calib_os1_to_%s.yaml"%camid)
    calibintr_path = join(indir, "calibrations", trajectory, "calib_%s_intrinsics.yaml"%camid)

    pc_filename = pc_path.split("/")[-1]
    _, _, _, frame = get_filename_info(pc_filename)
    frame = int(frame)
    

    frame_ts = frame_ts_np[frame][0]
    start_pose  = find_closest_pose(pose_np, frame_ts)
    end_pose    = find_closest_pose(pose_np, frame_ts + 0.1)
    comp_pc = compensate_frame(pc_path, frame_ts, start_pose, end_pose)

    img_frame = frame + offset
    img_filename = set_filename_by_prefix("2d_raw", camid, trajectory, img_frame)
    img_path = join(indir, "2d_raw", camid, trajectory, img_filename)

    comp_pc = comp_pc[:, :3].astype(np.float64)
    img_np = cv2.imread(img_path)
    comp_img_np = project_3dpoint_image(img_np, comp_pc, calibextr_path, calibintr_path, colormap="camera")

    print("Writing compensated frame %i to %s" % (frame, join(outdir, "%i.jpg"%frame)))
    cv2.imwrite(join(outdir, "%i.jpg"%frame), comp_img_np)

def compensate_trajectory_frames(args):
    indir, outdir, trajectory, frames, offset_frames_list, camid = args
    trajectory = str(trajectory)

    pc_dir  = join(indir, "3d_raw", "os1",trajectory)

    pc_paths = [f.path for f in os.scandir(pc_dir) if f.is_file()]
    pc_paths.sort(
        key=lambda pcpath: int(pcpath.split("/")[-1].split(".")[0].split("_")[-1])
    )

    # ts_path = join(indir, "timestamps", "%s_frame_to_ts.txt"%trajectory)
    # pose_path = join(indir, "poses", "%s.txt"%trajectory)
    # ts_to_frame_path = os.path.join(indir, "timestamps", "%s_frame_to_ts.txt"%trajectory)
    # ts_to_poses_path = join(indir, "poses", "%s.txt"%trajectory)
    # frame_to_poses_np = np.loadtxt(ts_to_poses_path).reshape(-1, 8)
    # frame_to_ts_np = np.loadtxt(ts_to_frame_path)
    # dense_poses = densify_poses_between_ts(frame_to_poses_np, frame_to_ts_np)
    # import pdb; pdb.set_trace()
    
    indir_list, outdir_list, trajectory_list, pc_path_list, offset_list, cam_list = [], [], [], [], [], []
    # Compensate all point clouds in frame list. If empty do all
    for pc_path in pc_paths:
        pc_filename = pc_path.split("/")[-1]
        
        _, _, _, frame = get_filename_info(pc_filename)
        frame = int(frame)
        if len(frames)>0: # Skip frame if not specified
            if frame not in frames:
                continue
        elif len(frames)==len(indir_list):
            break

        offset = 0
        for offset_frames in offset_frames_list:
            bounds = offset_frames["bounds"]
            if len(bounds) > 0 and frame >=bounds[0] and frame <= bounds[1]:
                offset =  offset_frames["offset"]

        indir_list.append(indir)
        outdir_list.append(outdir)
        trajectory_list.append(trajectory)
        pc_path_list.append(pc_path)
        offset_list.append(offset)
        cam_list.append(camid)

    pool = Pool(processes=96)
    for _ in tqdm.tqdm(pool.imap_unordered(compensate_single_trajectory_frame, zip(indir_list, outdir_list, \
        trajectory_list, pc_path_list, offset_list, cam_list)), total=len(pc_paths)):
        pass
        
        # start_pose  = find_closest_pose(pose_np, frame_ts)
        # end_pose    = find_closest_pose(pose_np, frame_ts + 0.1)
        # comp_pc = compensate_frame(pc_path, frame_ts, start_pose, end_pose)

        # # Save comp_pc and regular pc as pcd files to compare
        # # orig_pc = read_bin(pc_path)

        # # Project point clouds to images to compare
        # img_frame = frame
        # for offset_frames in offset_frames_list:
        #     bounds = offset_frames["bounds"]
        #     offset = offset_frames["offset"]
        #     if frame >=bounds[0] and frame <= bounds[1]:
        #         img_frame = frame + offset
        # # if frame >=10:
        # #     img_frame = frame
        # img_filename = set_filename_by_prefix("2d_raw", "cam0", trajectory, img_frame)
        # img_path = join(indir, "2d_raw", "cam0", trajectory, img_filename)
        # # img_np = cv2.imread(img_path)
        
        # # orig_pc = orig_pc[:, :3].astype(np.float64)
        # # orig_img_np = project_3dpoint_image(img_np, orig_pc, calibextr_path, calibintr_path)
        # # cv2.imwrite("orig_%s.png"%(trajectory), orig_img_np)

        # comp_pc = comp_pc[:, :3].astype(np.float64)
        # # bin_to_ply(orig_pc, "orig%s_%i.pcd" % (trajectory, frame))
        # # bin_to_ply(comp_pc, "comp%s_%i.pcd" % (trajectory, frame))

        # img_np = cv2.imread(img_path)
        # comp_img_np = project_3dpoint_image(img_np, comp_pc, calibextr_path, calibintr_path)
        # # cv2.imwrite("comp_%s.png"%(trajectory), comp_img_np)
        
        # combined_img_np = np.concatenate((orig_img_np, comp_img_np), axis=1)
        # print("Writing compensated frame %i" % frame)
        # cv2.imwrite(join(outdir, "%i.jpg"%frame), comp_img_np)

def compensate_all_frames(indir, outdir, trajectory, offset_frames, skip_amount=1, camid="cam0"):
    ts_path = join(indir, "timestamps", "%s_frame_to_ts.txt"%trajectory)
    frame_ts_np = np.fromfile(ts_path, sep=' ').reshape(-1, 1)

    total_frames = frame_ts_np.shape[0]
    trajectory_frame_list = np.arange(0, total_frames, skip_amount)
    
    trajoutdir = join(outdir, str(trajectory))
    if not osp.exists(trajoutdir):
        os.makedirs(trajoutdir)
    compensate_trajectory_frames((indir, trajoutdir, trajectory, trajectory_frame_list, offset_frames, camid))

# def save_single_trajectory_frame(args):
#     indir, outdir, trajectory, pc_path = args

#     ts_path = join(indir, "timestamps", "%s_frame_to_ts.txt"%trajectory)
#     pose_path = join(indir, "poses", "%s.txt"%trajectory)

#     # Load timestamps and dense poses file
#     frame_ts_np = np.fromfile(ts_path, sep=' ').reshape(-1, 1)
#     pose_np     = np.fromfile(pose_path, sep=' ').reshape(-1, 8)

#     # Load calibrations
#     calibextr_path = join(indir, "calibrations", trajectory, "calib_os1_to_%s.yaml"%camid)
#     calibintr_path = join(indir, "calibrations", trajectory, "calib_%s_intrinsics.yaml"%camid)

#     pc_filename = pc_path.split("/")[-1]
#     _, _, _, frame = get_filename_info(pc_filename)
#     frame = int(frame)
#     print("Writing compensated frame %i to %s" % (frame, p) ) 

#     frame_ts = frame_ts_np[frame][0]
#     start_pose  = find_closest_pose(pose_np, frame_ts)
#     end_pose    = find_closest_pose(pose_np, frame_ts + 0.1)
#     comp_pc = compensate_frame(pc_path, frame_ts, start_pose, end_pose)

    

    # img_frame = frame + offset
    # img_filename = set_filename_by_prefix("2d_raw", camid, trajectory, img_frame)
    # img_path = join(indir, "2d_raw", camid, trajectory, img_filename)

    # comp_pc = comp_pc[:, :3].astype(np.float64)
    # img_np = cv2.imread(img_path)
    # comp_img_np = project_3dpoint_image(img_np, comp_pc, calibextr_path, calibintr_path, colormap="camera")

    # cv2.imwrite(join(outdir, "%i.jpg"%frame), comp_img_np)

def main(args):
    indir  = "/robodata/arthurz/Datasets/CODa"
    outdir = "/robodata/arthurz/Datasets/CODa_egocomp_full"
    camid="cam0"
    skip_amount = 1

    dummy_trajectory = {
        "trajectory": 0,
        "offset_frames": [
            {
                "bounds": [],
                "offset": 0
            }
        ]
    }
    trajectory_list = []
    for i in range(22):
        curr_traj_dict = copy.deepcopy(dummy_trajectory)
        curr_traj_dict["trajectory"] = i
        trajectory_list.append(curr_traj_dict)

    for trajectory_dict in trajectory_list:
        print("Compensating trajectory %i" % trajectory_dict["trajectory"])
        compensate_all_frames(indir, outdir, trajectory_dict["trajectory"], trajectory_dict["offset_frames"], skip_amount=skip_amount)

    ### Uncomment Below for testing

    # trajectory = 3
    # frame = 6950
    # offset_frames = [
    #     {
    #         "bounds": [10, 6750],
    #         "offset": 0
    #     }
    # ]

    # compensate_all_frames(indir, outdir, trajectory, offset_frames, skip_amount=skip_amount)

    # Test single frame
    # pc_path = join(indir, "3d_raw", "os1", str(trajectory), "3d_raw_os1_%i_%i.bin"%(trajectory, frame))
    # compensate_single_trajectory_frame((indir, ".", str(trajectory), pc_path, offset_frames[0]["offset"], camid))

    # compensate_trajectory_frames((indir, ".", trajectory, [6590], offset_frames)) #, 7485, 7800, 8850]))



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

# trajectory_list = [
#         {
#             "trajectory": 0,
#             "offset_frames": [
#                 {
#                     "bounds": [6590, 7700],
#                     "offset": -1
#                 }
#             ]
#         },
#         {
#             "trajectory": 1,
#             "offset_frames": [
#                 {
#                     "bounds": [6575, 8800],
#                     "offset": -1
#                 }
#             ]
#         },
#         {
#             "trajectory": 2,
#             "offset_frames": [
#                 {
#                     "bounds": [2800, 3330],
#                     "offset": -1
#                 }
#             ]
#         },
#         {
#             "trajectory": 3,
#             "offset_frames": [
#                 {
#                     "bounds": [5170, 8045],
#                     "offset": 1
#                 }
#             ]
#         },
#         {
#             "trajectory": 4,
#             "offset_frames": [
#                 {
#                     "bounds": [6901, 7775],
#                     "offset": -1
#                 }
#             ]
#         },
#         {
#             "trajectory": 5,
#             "offset_frames": [
#                 {
#                     "bounds": [],
#                     "offset": 0
#                 }
#             ]
#         },
#         {
#             "trajectory": 6,
#             "offset_frames": [
#                 {
#                     "bounds": [6600, 6750],
#                     "offset": -1
#                 }
#             ]
#         },
#         {
#             "trajectory": 7,
#             "offset_frames": [
#                 {
#                     "bounds": [14400, 16200],
#                     "offset": -1
#                 }
#             ]
#         },
#         {
#             "trajectory": 8,
#             "offset_frames": [
#                 {
#                     "bounds": [1000, 1650],
#                     "offset": -2
#                 }
#             ]
#         },
#         {
#             "trajectory": 9,
#             "offset_frames": [
#                 {
#                     "bounds": [1000, 1400],
#                     "offset": -2
#                 }
#             ]
#         },
#     ]
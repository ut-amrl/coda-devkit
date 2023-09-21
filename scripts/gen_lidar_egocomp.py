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
    return comp_pc.reshape(bin_np_shape)

def compensate_single_trajectory_frame(args):
    indir, outdir, trajectory, pc_path, offset, camid, modality = args

    ts_path = join(indir, "timestamps", "%s.txt"%trajectory)
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


    if modality=="3d":
        compc_path = set_filename_dir(outdir, TRED_COMP_DIR, "os1", trajectory, str(frame))
        flat_pc = comp_pc.reshape(-1, comp_pc.shape[-1])
        flat_pc = flat_pc.reshape(-1)
        print("Writing compensated point cloud to %s " % compc_path)
        flat_pc.tofile(compc_path)
    elif modality=="2d":
        img_frame = frame + offset
        if camid=="cam0" or camid=="cam1":
            img_path = set_filename_dir(indir, TWOD_RAW_DIR, camid, trajectory, img_frame, include_name=True)
        else:
            img_path = set_filename_dir(indir, TWOD_RAW_DIR, camid, trajectory, img_frame, include_name=True)

        comp_pc = comp_pc[:, :3].astype(np.float64)
        img_np = cv2.imread(img_path)
        comp_img_np = project_3dpoint_image(img_np, comp_pc, calibextr_path, calibintr_path, colormap="camera")

        text = "Traj %s Frame %s" % (str(trajectory), str(frame))
        org = (1224-350, 40)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 255, 255)
        thickness = 2
        comp_img_np = cv2.putText(comp_img_np, text, org, font, fontScale, 
                    color, thickness, cv2.LINE_AA, False)

        out_img_path = join(outdir, TWOD_PROJ_DIR, camid, trajectory, "%i.jpg"%frame)

        print("Writing compensated frame %i to %s" % (frame, out_img_path))
        import pdb; pdb.set_trace()
        cv2.imwrite(out_img_path, comp_img_np)
    else:
        print("Specified undefined modality %s, nothing saved" % modality)

def compensate_trajectory_frames(args):
    indir, outdir, trajectory, frames, offset_frames_list, camid, save_modality = args
    trajectory = str(trajectory)

    pc_dir  = join(indir, "3d_raw", "os1", trajectory)

    pc_paths = [f.path for f in os.scandir(pc_dir) if f.is_file()]
    pc_paths.sort(
        key=lambda pcpath: int(pcpath.split("/")[-1].split(".")[0].split("_")[-1])
    )
    indir_list, outdir_list, trajectory_list, pc_path_list, offset_list, cam_list = [], [], [], [], [], []
    save_mod_list = []

    if save_modality=="3d":
        savedir = set_filename_dir(outdir, TRED_COMP_DIR, "os1", trajectory, include_name=False)
        if not osp.exists(savedir):
            print("Making output directory %s for trajectory %s" % (savedir, trajectory))
            os.makedirs(savedir)
    else:
        savedir = set_filename_dir(outdir, TWOD_PROJ_DIR, camid, trajectory)
        if not osp.exists(savedir):
            print("Making output directory %s for trajectory %s" % (savedir, trajectory))
            os.makedirs(savedir)

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
        save_mod_list.append(save_modality)

    pool = Pool(processes=16)
    for _ in tqdm.tqdm(pool.imap_unordered(compensate_single_trajectory_frame, zip(indir_list, outdir_list, \
        trajectory_list, pc_path_list, offset_list, cam_list, save_mod_list)), total=len(pc_paths)):
        pass
        
def compensate_all_frames(indir, outdir, trajectory, offset_frames, skip_amount=1, camid="cam0", save_modality="2d"):
    ts_path = join(indir, "timestamps", "%s.txt"%trajectory)
    frame_ts_np = np.fromfile(ts_path, sep=' ').reshape(-1, 1)

    total_frames = frame_ts_np.shape[0]
    trajectory_frame_list = np.arange(0, total_frames, skip_amount)
    
    compensate_trajectory_frames((indir, outdir, trajectory, trajectory_frame_list, offset_frames, camid, save_modality))

def main(args):
    indir  = "/robodata/arthurz/Datasets/CODa_dev"
    outdir = "/robodata/arthurz/Datasets/CODa_egocomp_full"
    # outdir = "/robodata/arthurz/Datasets/CODa" # for writing point clouds
    camid="cam3"
    skip_amount = 1
    save_modality="2d"

    dummy_trajectory = {
        "trajectory": 0,
        "offset_frames": [
            {
                "bounds": [],
                "offset": 0
            }
        ]
    }

    # Uncomment Below to view all egocompensated trajectories
    # trajectory_list = []
    # for i in range(0, 23):
    #     curr_traj_dict = copy.deepcopy(dummy_trajectory)
    #     curr_traj_dict["trajectory"] = i
    #     trajectory_list.append(curr_traj_dict)

    # for trajectory_dict in trajectory_list:
    #     print("Compensating trajectory %i" % trajectory_dict["trajectory"])
    #     compensate_all_frames(indir, outdir, trajectory_dict["trajectory"], trajectory_dict["offset_frames"], skip_amount=skip_amount, save_modality=save_modality)

    ### Uncomment Below for testing
    outdir = "."
    trajectory = 0
    frame = 500
    offset_frames = [
        {
            "bounds": [10, 6750],
            "offset": 0
        }
    ]

    # Test single frame
    pc_path = join(indir, "3d_raw", "os1", str(trajectory), "3d_raw_os1_%i_%i.bin"%(trajectory, frame))
    outdir = join(".", "2d_raw", camid, str(trajectory))
    if not osp.exists(outdir):
        os.makedirs(outdir)
    compensate_single_trajectory_frame((indir, ".", str(trajectory), pc_path, offset_frames[0]["offset"], camid, "2d"))

    # compensate_trajectory_frames((indir, ".", trajectory, [6590], offset_frames)) #, 7485, 7800, 8850]))



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

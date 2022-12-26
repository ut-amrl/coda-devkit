import os
import sys
import pdb
import numpy as np

# For imports
sys.path.append(os.getcwd())

#CustomImports
from helpers.sensors import *


def main():
    bin_dir = "/home/arthur/AMRL/Datasets/CODa/3d_raw/os1/2"
    bin_file    = "3d_raw_os1_2_2780.bin"
    outdir  = "/home/arthur/AMRL/Benchmarks/test/CODa_pcd"
    # bin_dir = "/home/arthur/AMRL/Benchmarks/OpenPCDet/data/kitti/training/velodyne"
    # bin_file    = "002286.bin"
    # outdir  = "/home/arthur/AMRL/Benchmarks/test/KITTI_pcd"

    bin_path = os.path.join(bin_dir, bin_file)
    pc_np = read_bin(bin_path, True)

    #Downsample from 128 to 64 channels
    pc_ds_np    = pc_np[:, :4].reshape(1024, 128, 4)
    pc_ds_np    = pc_ds_np[:, np.arange(0, 128, 2), :]
    pc_intensity= pc_ds_np[:, :, -1].reshape(-1, 1)
    pc_ds_np    = pc_ds_np[:, :, :3].reshape(-1, 3)

    #Filter out points in same FOV Velodyne
    pc_dist     = np.linalg.norm(pc_ds_np[:, :3], axis=1)
    zero_mask   = pc_dist!=0
    pc_ds_np    = pc_ds_np[zero_mask]
    pc_intensity= pc_intensity[zero_mask]

    pc_angle    = np.arcsin(pc_ds_np[:, 2] / pc_dist[zero_mask])
    fov_mask    = np.abs(pc_angle) <= 0.2338741
    pc_ds_np    = pc_ds_np[fov_mask, :]
    pc_intensity= pc_intensity[fov_mask, :]
    pc_ds_np[:, 2] -= 1.2

    pcd_path = os.path.join(outdir, bin_file.replace(".bin", ".pcd"))
    bin_to_ply(pc_ds_np, pcd_path)

    pc_np = np.hstack((pc_ds_np, pc_intensity))
    pc_np.tofile(pcd_path.replace(".pcd", ".bin"))
    print("Wrote pcd file to ", pcd_path)

if __name__ == "__main__":
    main()
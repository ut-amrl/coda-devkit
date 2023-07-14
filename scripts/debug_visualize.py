#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--traj', default="0",
                    help="Generated file for HitL")

def main(args):
    traj  = args.traj
    fdir    = "/home/jihwan98/coda/HitL"
    fpath   = os.path.join(fdir, traj + "_vis" + ".bin")

    # reads in the generated file
    bin_np  = np.fromfile(fpath).reshape(-1, 8)
    pose_x, pose_y, obs_x, obs_y = bin_np[:, 0], bin_np[:, 1], bin_np[:, 3], bin_np[:, 4]

    print("Start plotting")
    fig = plt.figure()
    # plots the 2d observations and poses on a grid
    plt.scatter(obs_x,  obs_y,  marker=".", s=5,  label='Observation')
    plt.scatter(pose_x, pose_y, marker=".", s=25, label='Pose Estimation')
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
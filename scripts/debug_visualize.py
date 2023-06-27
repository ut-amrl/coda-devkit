import os
import sys
import argparse
import numpy as np
import math

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--traj', default="0",
                    help="Generated file for HitL")

def main(args):
    # reads in the generated file
    fdir  = "/home/jihwan98/coda/HitL"
    traj  = args.traj
    fpath = os.path.join(fdir, traj + ".bin")
    bin_np = np.fromfile(fpath).reshape(-1, 16)
    pose_x, pose_y, obs_x, obs_y = bin_np[:, 0], bin_np[:, 1], bin_np[:, 2], bin_np[:, 3]

    # TODO: convert obs based on pose' orientation
    obs_x_moving = np.add(pose_x, obs_x)
    obs_y_moving = np.add(pose_y, obs_y)

    print("Start plotting")
    fig = plt.figure()
    # plots the 2d observations and poses on a grid

    # import pdb; pdb.set_trace()

    plt.scatter(obs_x_moving[0::1024],  obs_y_moving[0::1024],  marker=".", s=10, label='Observation')
    plt.scatter(pose_x,        pose_y,        marker=".", s=50, label='Pose Estimation')
    plt.grid(visible=True)
    plt.legend(loc="upper right")

    # save as png
    fname = os.path.join(fdir, traj + ".png")
    print("Saving image")
    plt.savefig(fname)
    
    return

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
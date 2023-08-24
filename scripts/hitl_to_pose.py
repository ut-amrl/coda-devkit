import os
from os.path import join
import sys
import argparse
import numpy as np
import time

from scipy.spatial.transform import Rotation as R

# parser = argparse.ArgumentParser()
# parser.add_argument('--traj', default=0,
#                     help="number of trajectory, e.g. 1")

def yaw_to_quat(yaw):
    angle = R.from_euler('z', yaw, degrees=False)
    quat_np = angle.as_quat()
    quat_np = quat_np[:, [3, 0, 1, 2]] # (qx, qy, qz, qw) ->( qw, qx, qy, qz)
    return quat_np

def get_Z(pose_np):
    pose_Transpose = pose_np.T[3]
    return pose_Transpose

def create_Output(pose_np, xy_np, quat_np):
    timestamp_np = pose_np[:, 0].reshape(-1, 1)
    z_np = pose_np[:, 3].reshape(-1, 1)
    final_np = np.hstack((timestamp_np, xy_np, z_np, quat_np))
    return final_np

def main():
    # trajectory = args.traj
    base = str(0)
    comp = str(1)
    dir = f"HitL/hitl_results"
    hitl_path = os.path.join(dir, f"hitl_results_trajectory_{base}_{comp}.txt")
    pose_dir = f"/robodata/arthurz/Datasets/CODa_dev/poses"
    base_path = os.path.join(pose_dir, f"{base}.txt")
    comp_path = os.path.join(pose_dir, f"{comp}.txt")

    hitl_np = np.loadtxt(hitl_path, delimiter=" ").reshape(-1, 3)
    base_np = np.loadtxt(base_path).reshape(-1, 8)
    comp_np = np.loadtxt(comp_path).reshape(-1, 8)
    pose_np = np.vstack((base_np, comp_np))

    # first create xy_np
    xy_np = hitl_np[:, :2]

    # then create the quat_np
    yaw_Val = hitl_np[:, 2].reshape(-1, 1)
    quat_np = yaw_to_quat(yaw_Val)

    # then combine
    output_np = create_Output(pose_np, xy_np, quat_np)

    base_np = output_np[:len(base_np)]
    base_np = np.around(base_np[:, 0], decimals = 6)
    base_np = np.around(base_np[:, 1:], decimals = 8)
    comp_np = output_np[len(base_np):]
    comp_np = np.around(comp_np[:, 0], decimals = 6)
    comp_np = np.around(comp_np[:, 1:], decimals = 8)

    save_dir = os.path.join("HitL/poses_cor")
    base_out = os.path.join(save_dir, base + "_cor.txt")
    comp_out = os.path.join(save_dir, comp + "_cor.txt")
    np.savetxt(base_out, base_np, delimiter=' ', comments='')
    np.savetxt(comp_out, comp_np, delimiter=' ', comments='')

if __name__ == "__main__":
    start_time = time.time()
    # args = parser.parse_args()
    main()
    print("--- Final: %s seconds ---" % (time.time() - start_time))
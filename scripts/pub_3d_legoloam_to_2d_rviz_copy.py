# import pdb; pdb.set_trace()
import os
from os.path import join
import sys
import argparse
import numpy as np
import time
import json

from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import KDTree

import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped

from multiprocessing import Pool
import tqdm

# For imports
sys.path.append(os.getcwd())
from helpers.sensors import set_filename_dir, read_bin
from helpers.geometry import pose_to_homo, find_closest_pose, densify_poses_between_ts
from helpers.visualization import pub_pc_to_rviz, pub_pose

parser = argparse.ArgumentParser()
parser.add_argument('--traj', default="0",
                    help="number of trajectory, e.g. 1")
parser.add_argument('--option', default="hitl",
                    help="hitl for hitl SLAM and vis for visualization ")

def main(args):
    global option
    trajectory, option = args.traj, args.option
    indir = "/robodata/arthurz/Datasets/CODa_dev"
    pose_path = f"{indir}/poses/{trajectory}.txt"
    # pose_path = f"{trajectory}.txt"
    ts_path = f"{indir}/timestamps/{trajectory}_frame_to_ts.txt"
    bin_dir   = f"{indir}/3d_comp/os1/{trajectory}/"
    outdir    = f"./cloud_to_laser/%s" % args.traj

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Initialize ros publishing
    rospy.init_node('bin_publisher', anonymous=True)
    pc_pub = rospy.Publisher('/coda/ouster/lidar_packets', PointCloud2, queue_size=10)
    pose_pub = rospy.Publisher('/coda/pose', PoseStamped, queue_size=10)

    pose_np = np.loadtxt(pose_path).reshape(-1, 8)
    timestamp_np = np.loadtxt(ts_path).reshape(-1)
    
    # number of lidar frames
    # n_of_bin_files = 1000
    n_of_bin_files = len([entry for entry in os.listdir(bin_dir) if os.path.isfile(os.path.join(bin_dir, entry))])
    print("\n" + "---"*15)
    print(f"\nNumber of lidar frames: {len(pose_np)}\n")

    # Iterate through all poses
    LIDAR_HEIGHT = 0.8 # meters
    ZBAND_HEIGHT = 0.5 # meters
    ZMIN_OFFSET = 1.9 # meters

    rate = rospy.Rate(100)

    dense_pose_np = densify_poses_between_ts(pose_np, timestamp_np)
    last_pose = None

    waypointsfulljson = {
        "0": {
	    "S0": [0],
	    "S1": [1800],
	    "S2": [2964],
	    "S3": [3526],
	    "D1": [4510],
	    "S1": [6591],
	    "S0": [8200]
        },
        "1": {
	    "S0": [0],
	    "S1": [1612],
	    "D1": [3676],
	    "S3": [5012],
	    "S2": [5750],
	    "S1": [6736],
	    "S0": [8646]
        },
        "2": {
	    "S0": [0],
	    "S1": [1649],
	    "G1": [4005],
	    "G2": [5330],
	    "G3": [6547],
	    "G4": [8167],
	    "G5": [9330],
	    "G2": [11057],
	    "G1": [12700],
	    "S1": [15044],
	    "S0": [16613]
        },
        "3": {
	    "S0": [0],
	    "S1": [1694],
	    "S2": [2535],
	    "S3": [2908],
	    "D1": [5361],
	    "S1": [7499],
	    "D1": [10060],
	    "S3": [11211],
	    "S2": [11843],
	    "S1": [12791],
	    "S0": [14675]
        },
        "4": {
	    "S0": [0],
	    "S1": [1545],
	    "D1": [4240],
	    "S3": [5278],
	    "S2": [5811],
	    "S1": [6844],
	    "S0": [8210]
        },
        "5": {
	    "S0": [0],
	    "S1": [1557],
	    "S2": [2383],
	    "S3": [2759],
	    "D1": [3746],
	    "S1": [5570],
	    "S0": [7211]
        },
        "6": {
	    "S0": [0],
	    "S1": [1579],
	    "S2": [2433],
	    "S3": [3329],
	    "W7": [4395],
	    "W6": [6516],
	    "W5": [8335],
	    "W7": [9517],
	    "S2": [11028],
	    "S1": [11895],
	    "S0": [13570]
        },
        "7": {
	    "S0": [0],
	    "S1": [1534],
	    "G1": [3722],
	    "G2": [5006],
	    "G3": [6271],
	    "G4": [7924],
	    "G5": [9050],
	    "G2": [10693],
	    "G1": [12225],
	    "S1": [14556],
	    "S0": [16117]
        },
        "8": {
	    "U2": [7264],
	    "U1": [8332],
	    "G2": [10100]
        },
        "9": {
	    "S0": [0],
	    "S1": [1530],
	    "S2": [2456],
	    "W7": [3750],
	    "W6": [5714],
	    "W5": [7332],
	    "W7": [8450],
	    "S2": [9863],
	    "S1": [10720],
	    "S0": [12258]
        },
        "10": {
	    "S0": [0],
	    "S1": [1589],
	    "S2": [2709],
	    "W7": [4094],
	    "W5": [5190],
	    "W6": [6860],
	    "W7": [8866],
	    "S2": [10445],
	    "S1": [11718],
	    "S0": [13300]
        },
        "11": {
	    "S0": [0],
	    "S1": [1532],
	    "S2": [2572],
	    "S3": [3804],
	    "W7": [5322],
	    "W6": [7750],
	    "W5": [10174],
	    "W7": [11425],
	    "S3": [12473],
	    "S2": [13800],
	    "S1": [14759],
	    "S0": [16451]
        },
        "12": {
	    "S0": [0],
	    "S1": [1613],
	    "G1": [3900],
	    "G2": [5249],
	    "G5": [6940],
	    "G4": [8201],
	    "G3": [11882],
	    "G2": [13336],
	    "G1": [15056],
	    "S1": [18044],
	    "S0": [19664],
        },
        "13": {
	    "S0": [0],
	    "S1": [1629],
	    "S2": [2565],
	    "W7": [4241],
	    "W5": [5467],
	    "W6": [7630],
	    "W7": [10294],
	    "S2": [12680],
	    "S1": [13678],
	    "S0": [15260]
        },
        "14": {
	    "G2": [0],
	    "U1": [1310],
	    "U1": [10520],
	    "G2": [11981]
        },
        "15": {
	    "G2": [0],
	    "U1": [1354],
	    "U1": [9906],
	    "G2": [11233]
        },
        "16": {
	    "S0": [0],
	    "S1": [1539],
	    "G1": [4375],
	    "G2": [5304],
	    "G3": [7150],
	    "G4": [8673],
	    "G5": [9750],
	    "G2": [11648],
	    "G1": [13047],
	    "S1": [15366],
	    "S0": [17054]
        },
        "17": {
	    "S0": [0],
	    "S1": [1610],
	    "G1": [4085],
	    "G2": [5819],
	    "G5": [7690],
	    "G4": [9160],
	    "G3": [11300],
	    "G2": [13200],
	    "G1": [15100],
	    "S1": [17664],
	    "S0": [19500]
        },
        "18": {
	    "S0": [0],
	    "S1": [1543],
	    "D1": [3769],
	    "S3": [5660],
	    "S2": [6150],
	    "S1": [7090],
	    "S0": [8759]
        },
        "19": {
	    "S0": [0],
	    "S1": [1861],
	    "S2": [7474],
	    "S3": [7859],
	    "D1": [9667],
	    "S1": [12628],
	    "S0": [14383]
        },
        "20": {
	    "S0": [0],
	    "S1": [1656],
	    "S2": [2530],
	    "S3": [2994],
	    "W7": [4020],
	    "W5": [5118],
	    "W6": [7062],
	    "W7": [9310],
	    "S3": [10742],
	    "S2": [11358],
	    "S1": [12328],
	    "S0": [13951]
        },
        "21": {
	    "S0": [0],
	    "S1": [1784],
	    "G1": [4300],
	    "G2": [5680],
	    "G5": [10272],
	    "G4": [11562],
	    "G3": [13888],
	    "G2": [15646],
	    "G1": [17124],
	    "S1": [19598],
	    "S0": [21368]
        },
        "22": {
	    "S0": [0],
	    "S1": [1560],
	    "S2": [2363],
	    "S3": [4299],
	    "W7": [5463],
	    "W5": [6659],
	    "W6": [8377],
	    "W7": [10475],
	    "S3": [11917],
	    "S2": [12830],
	    "S1": [13720],
	    "S0": [15497]
        }
    }
    # waypointsjson = json.dumps(waypointsfulljson)
    # waypointsjson2 = json.load(waypointsjson) # Uncomment once we load from file

    waypointsjson2 = waypointsfulljson
    waypointswithtype = waypointsjson2[args.traj]
    waypoints = waypointswithtype.values()
    waypoint_timestamps = []
    for waypoint_frame in waypoints:
        waypoint_ts = timestamp_np[waypoint_frame]
        waypoint_timestamps.append(waypoint_ts)

    for pose_idx, pose in enumerate(pose_np):
        pose_ts = pose[0]
        # if pose_idx < 1000:
        #     continue
        print("pose ", pose_idx)
        closest_lidar_frame = np.searchsorted(timestamp_np, pose_ts, side='left')
        lidar_ts = timestamp_np[closest_lidar_frame]
        bin_path = set_filename_dir(indir, "3d_raw", "os1", trajectory, closest_lidar_frame, include_name=True)
        lidar_np = read_bin(bin_path, keep_intensity=False)

        if waypoint_timestamps[0] < pose_ts:
            closest_pose = find_closest_pose(pose_np, waypoint_timestamps[0])
            pub_waypoint_to_rviz(marker_publisher, closest_pose, waypoint_timestamps[0])
            waypoint_timestamps.pop(0)
    
        # Filter all point between zmin and zmax, downsample angular to 1/4 original size
        lidar_np = lidar_np.reshape(128, 1024, -1)
        zmin = ZMIN_OFFSET - LIDAR_HEIGHT
        zmax = zmin+ZBAND_HEIGHT
        z_mask = np.logical_and(lidar_np[:, :, 2] > zmin, lidar_np[:, :, 2] < zmax)
        lidar_np = lidar_np[z_mask].reshape(-1, 3).astype(np.float32)

        LtoG                = pose_to_homo(pose) # lidar to global frame
        homo_lidar_np       = np.hstack((lidar_np, np.ones((lidar_np.shape[0], 1))))
        trans_homo_lidar_np = (LtoG @ homo_lidar_np.T).T
        trans_lidar_np      = trans_homo_lidar_np[:, :3] #.reshape(128, 1024, -1)
        trans_lidar_np      = trans_lidar_np.reshape(-1, 3).astype(np.float32)

        pub_pc_to_rviz(trans_lidar_np, pc_pub, lidar_ts, point_type="xyz")
        pub_pose(pose_pub, pose, pose[0])
        if last_pose is None:
            last_pose = pose
        else:
            print("check is last poses are similar")
            # print("last pose ", last_pose[1:])
            # print("last pose ", pose[1:])
            print(np.allclose(last_pose[1:], pose[1:], rtol=1e-5) )
            last_pose = pose
        # import pdb; pdb.set_trace()
        # rate.sleep()

if __name__ == "__main__":
    start_time = time.time()
    args = parser.parse_args()
    main(args)
    print("--- Final: %s seconds ---" % (time.time() - start_time))
    
    if (args.option == "vis"):
        os.system(f"scripts/debug_visualize.py --traj {args.traj}")

# python -W ignore scripts/3d_legoloam_to_2d_hitl.py --traj 0 --option hitl
# python -W ignore scripts/3d_legoloam_to_2d_hitl.py --traj 0 --option vis
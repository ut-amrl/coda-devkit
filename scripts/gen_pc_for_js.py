import json
import numpy as np
import pickle

import os
import sys
sys.path.append(os.getcwd())

from helpers.constants import *

def read_bin_file(file_path):
    """Read a .bin file as a point cloud."""
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
        points = data.reshape(-1, 4)  # Assuming 4 floats per point (x, y, z, intensity)
    return points[:, :3]

def downsample_point_cloud(points, vfactor, hfactor):
    """Downsample a point cloud by a factor."""
    _, V = points.shape
    points = points.reshape(128, 1024, -1)
    points = points[::vfactor, ::hfactor, :V]
    points = points.reshape(-1, V)

    return points

def save_bin_file(points, output_path):
    """Save a point cloud as a binary file."""
    with open(output_path, 'wb') as fp:
        fp.write(points.flatten().tobytes())
        print('Done writing pc into a binary file: ', output_path)

def read_bbox_file(file_path):
    anno_file   = open(file_path, 'r')
    anno_json   = json.load(anno_file)

    bbox_annotations = np.empty((len(anno_json["3dbbox"]), 11), dtype=np.float32) # x, y, z, l, w, h, r, p, y
    for idx, annotation in enumerate(anno_json["3dbbox"]):
        classId             = BBOX_CLASS_TO_ID[annotation['classId']]
        instanceId          = annotation["instanceId"].split(':')[-1]
        px, py, pz          = annotation["cX"], annotation["cY"], annotation["cZ"]
        l, w, h             = annotation["l"], annotation["w"], annotation["h"]
        r, p, y             = annotation["r"], annotation["p"], annotation["y"]

        bbox_annotations[idx] = np.array([px, py, pz, l, w, h, r, p, y, classId, instanceId], dtype=np.float32)

    return bbox_annotations

def save_bbox_file(bbox_annotations, output_path):
    """Save a bbox numpy array x, y, z, l, w, h, r, p, y as a binary file."""
    with open(output_path, 'wb') as fp:
        fp.write(bbox_annotations.flatten().tobytes())
        print('Done writing bbox into a binary file: ', output_path)

if __name__ == "__main__":
    traj = 20
    start_frame = 3100 
    num_frames = 50
    skip_interval = 5

    for cf in range(start_frame, start_frame+num_frames*skip_interval, skip_interval):
        # Define input and output paths
        pc_path = f'/robodata/arthurz/Datasets/CODa_dev/3d_raw/os1/{traj}/3d_raw_os1_{traj}_{cf}.bin'
        pc_outpath = f'ds/{cf}.bin'

        # Read the point cloud
        point_cloud = read_bin_file(pc_path)

        # Define downsampling factor (e.g., keep every 10th point)
        vdownsample_factor = 4
        hdownsample_factor = 4

        # Downsample the point cloud
        downsampled_cloud = downsample_point_cloud(point_cloud, vdownsample_factor, hdownsample_factor)

        # Change to float16 to save space, increases runtime
        downsampled_cloud = downsampled_cloud.astype(np.float32)

        # Save the downsampled point cloud
        save_bin_file(downsampled_cloud, pc_outpath)

        # Save corresponding 3D bounding box annotation
        bbox_path = f'/robodata/arthurz/Datasets/CODa_dev/3d_bbox/os1/{traj}/3d_bbox_os1_{traj}_{cf}.json'
        bbox_outpath = f'ds/{cf}bbox.bin'

        bbox_np = read_bbox_file(bbox_path)

        save_bbox_file(bbox_np, bbox_outpath)
    
    # # Save to .pcd file
    # import open3d as o3d

    # with open(pc_outpath, 'rb') as f:
    #     reload_ds_cloud = np.fromfile(f, dtype=np.float16)
    #     reload_ds_cloud = reload_ds_cloud.reshape(-1, 3)

    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(reload_ds_cloud[:, :3])
    #     o3d.io.write_point_cloud("ds/3500.pcd", pcd)
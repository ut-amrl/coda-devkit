import os
import folium
from geopy import distance
from IPython.display import Image, display
import numpy as np
import json

from scipy.spatial.transform import Rotation as R
import numpy as np
from sklearn.neighbors import KDTree

TRAJECTORY  = {'AHGGDC': [0, 1, 3, 4, 5, 18, 19],
               'AHGGuad': [2, 7, 12, 16, 17, 21],
               'AHGWCP': [6, 9, 10, 11, 13, 20, 22],
               'AHGUNB': [8, 14, 15]}

JSON_NAMES = ["start_arr", "end_arr", "yaw_arr"]

COLORS = ['red', 'green', 'blue', 'orange', 'pink', 'purple', 'black']

def convert_xyz_to_latlon(x, y, z):
    # Define the origin coordinates (latitude, longitude) and altitude
    # origin = (30.28805556, -97.7375, 0)  # Ut Austin Start Coordinates
    # origin = (30.288114, -97.737699, 0)  # AHG original
    origin = (30.288130, -97.73762, 0)     # Very Good
    trans = np.array([[-1, 0], [0, -1]])
    trans_xy = trans @ np.array([x, y]).reshape(2, 1)
    offset = np.linalg.norm(trans_xy)
    bearing =  np.arctan2(trans_xy[1].item(), trans_xy[0].item())

    dest = distance.distance(meters=offset).destination(origin[:2], bearing=90 - bearing * 180 / np.pi)

    # Return the latitude, longitude, and altitude
    return dest[0], dest[1], origin[2] + z

def add_marker(x, y, z, m, color):
    latitude, longitude, altitude = convert_xyz_to_latlon(x, y, z)
    # folium.Marker(location=[latitude, longitude]).add_to(m)
    folium.CircleMarker(location=[latitude, longitude],
                        radius=1, weight=1, color=color).add_to(m)
    return m

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

def correct_pose(pose_np, start_arr, end_arr, yaw_arr):
    corrected_pose = np.copy(pose_np)
    # handles multiple rotation
    for i in range(len(start_arr)): 
        start, end, yaw = start_arr[i], end_arr[i], yaw_arr[i]
        non_origin = False
        if start != 0:
            non_origin = True
        homo_mat = yaw_to_homo(corrected_pose[start:end, :], yaw)
        corrected_pose[start:end, 1:4] = apply_hom_mat(corrected_pose[start:end, :], homo_mat, non_origin)
    return corrected_pose

def find_overlapping_pc(pose_np, pc):
    tree = KDTree(pose_np, leaf_size=2) # for an efficient closest points search
    _, ind = tree.query(pc.reshape(1, -1), k=10)
    print(ind)

def main():
    TRAJ_name = list(TRAJECTORY.keys())[1]
    trajectory_list = TRAJECTORY[TRAJ_name]
    # trajectory_list = [7]

    outdir = './json'
    fpath = os.path.join(outdir, 'pose_correction.json')
    
    f = open(fpath, "r")
    pose_correction = json.load(f)
    f.close()

    latitude, longitude, altitude = convert_xyz_to_latlon(0, 0, 0)
    m = folium.Map(location=[latitude, longitude], zoom_start=35)
    # for trajectory in trajectory_list:
    for i in range(len(trajectory_list)):
        trajectory = str(trajectory_list[i])
        print("---"*10 + f"\nTrajectory {trajectory}")
        pose_path = f"/robodata/arthurz/Datasets/CODa_dev/poses/dense/{trajectory}.txt"
        pose_np = np.loadtxt(pose_path).reshape(-1, 8)
        
        start_arr, end_arr, yaw_arr = [], [], []

        if trajectory in pose_correction.keys():
            traj_dict = pose_correction[trajectory]
            start_arr, end_arr, yaw_arr = traj_dict[JSON_NAMES[0]], traj_dict[JSON_NAMES[1]], traj_dict[JSON_NAMES[2]]
        
        # for pose in pose_np:
        # # for pose in pose_np:
        #     _, x, y, z, _, _, _, _ = pose
        #     m = add_marker(x, y, z, m, COLORS[0])
        # print(f"Before color: {COLORS[0]}")
        
        corrected_pose_np = correct_pose(pose_np, start_arr, end_arr, yaw_arr)
        # import pdb; pdb.set_trace()

        # find_overlapping_pc(corrected_pose_np[:, 1:3], corrected_pose_np[5100, 1:3])

        for pose in corrected_pose_np:
            # if cnt > start_arr[1]:
            #     _, x, y, z, _, _, _, _ = pose
            #     m = add_marker(x, y, z, m, COLORS[0])
            # else:
            _, x, y, z, _, _, _, _ = pose
            m = add_marker(x, y, z, m, COLORS[i])
            
        print(f"After  color: {COLORS[i]}")

    map_filename = "./map_image.html"
    m.fit_bounds(m.get_bounds())
    m.save(map_filename)
    print("-"*20 + "\nMap saved as HTML:", map_filename)

if __name__ == '__main__':
    main()

    # if not os.path.exists(outdir):
    #     os.makedirs(outdir)
    #     print("json directory created")

    # if not os.path.exists(fpath):
    #     json_object = json.dumps({})
    #     with open(fpath, "w") as f:
    #         f.write(json_object)
    #     f.close()
    #     print("empty json file created")

        # traj_dict = {JSON_NAMES[0]: [start], JSON_NAMES[1]: [end], JSON_NAMES[2]: [yaw]}
        # pose_correction[int(trajectory)] = traj_dict
        # json_object = json.dumps(pose_correction, indent=4)


        # with open(fpath, "w+") as f:
    #     f.write(json_object)
    # f.close()
    # print("\njson file successfully saved")
import open3d as o3d
import matplotlib
import numpy as np

import os
import sys
import json

# For imports
sys.path.append(os.getcwd())
from helpers.geometry import get_3dbbox_corners
from helpers.constants import BBOX_ID_TO_COLOR, BBOX_CLASS_TO_ID, SEM_ID_TO_COLOR
from helpers.visualization import apply_semantic_cmap

def draw_open3d_box(vis, bbox_path):
    bbox_file   = open(bbox_path, 'r')
    bbox_json   = json.load(bbox_file)

    for bbox_dict in bbox_json["3dbbox"]:
        bbox_corners = get_3dbbox_corners(bbox_dict)

        bbox_class_id = bbox_dict["classId"]
        center = np.array([ bbox_dict['cX'], bbox_dict['cY'], bbox_dict['cZ'] ])
        lwh = np.array([ bbox_dict['l'], bbox_dict['w'], bbox_dict['h'] ])
        axis_angles = np.array([ bbox_dict['r'], bbox_dict['p'], bbox_dict['y'] ])
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
        box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
        lines = np.asarray(line_set.lines)
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

        line_set.lines = o3d.utility.Vector2iVector(lines)

        class_color = np.array([color / 255 for color in BBOX_ID_TO_COLOR[BBOX_CLASS_TO_ID[bbox_class_id]] ])
        line_set.paint_uniform_color( class_color )

        vis.add_geometry(line_set)
    
    return vis

def draw_open3d_pc(vis, bin_path, sem_path=None):
    num_points = 1024*128
    bin_np = np.fromfile(bin_path, dtype=np.float32).reshape(num_points, -1)
    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(bin_np[:, :3])
    vis.add_geometry(pts)

    if sem_path is not None:
        color_map = apply_semantic_cmap(sem_path) / 255.0 # normalize colormap
        pts.colors = o3d.utility.Vector3dVector(color_map)
    else:
        pts.colors = o3d.utility.Vector3dVector(np.ones((bin_np.shape[0], 3)))
    return vis

def main():

    # Read annotation file
    # bbox_path = "/media/arthur/ExtremePro/CODa/3d_bbox/os1/0/3d_bbox_os1_0_4781.json"
    # sem_path = "/media/arthur/ExtremePro/CODa/3d_semantic/os1/0/3d_semantic_os1_0_4781.bin"
    # bin_path  = "/media/arthur/ExtremePro/CODa/3d_raw/os1/0/3d_raw_os1_0_4781.bin"
    indir="/home/arthur/Downloads/CODa"
    traj=2
    frame=6880
    bbox_path = "%s/3d_bbox_os1_%i_%i.json"%(indir, traj, frame)
    sem_path = "%s/3d_semantic_os1_%i_%i.bin"%(indir, traj, frame)
    bin_path  = "%s/3d_raw_os1_%i_%i.bin" % (indir, traj, frame)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.ones(3) * 255
    vis = draw_open3d_box(vis, bbox_path)
    vis = draw_open3d_pc(vis, bin_path, sem_path)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()

# num_points = 1000
# point_cloud_np = np.random.rand(num_points, 3)
# # Load point cloud
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(point_cloud_np)

# # Define the corners of the bounding box
# corners = np.array([
#     [-1, -1, -1],
#     [1, -1, -1],
#     [-1, 1, -1],
#     [-1, -1, 1],
#     [1, 1, -1],
#     [1, -1, 1],
#     [-1, 1, 1],
#     [1, 1, 1]
# ])

# # Create a non-axis aligned bounding box
# bbox = o3d.geometry.OrientedBoundingBox()
# bbox.create_from_points(o3d.utility.Vector3dVector(corners))

# # Create a color map for the points
# colors = o3d.utility.Vector3dVector([
#     [1.0, 0.0, 0.0],  # Red
#     [0.0, 1.0, 0.0],  # Green
#     [0.0, 0.0, 1.0]   # Blue
# ])

# # Color code the points within the bounding box
# indices = bbox.get_point_indices_within_bounding_box(point_cloud.points)
# point_cloud.colors = o3d.utility.Vector3dVector([colors[0]] * len(point_cloud.points))
# # point_cloud.colors[indices] = colors[1]

# # Visualize the point cloud with the bounding box
# o3d.visualization.draw_geometries([point_cloud, bbox])
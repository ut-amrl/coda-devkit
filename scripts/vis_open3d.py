import open3d as o3d
import numpy as np

num_points = 1000
point_cloud_np = np.random.rand(num_points, 3)
# Load point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(point_cloud_np)

# Define the corners of the bounding box
corners = np.array([
    [-1, -1, -1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [1, 1, -1],
    [1, -1, 1],
    [-1, 1, 1],
    [1, 1, 1]
])

# Create a non-axis aligned bounding box
bbox = o3d.geometry.OrientedBoundingBox()
bbox.create_from_points(o3d.utility.Vector3dVector(corners))

# Create a color map for the points
colors = o3d.utility.Vector3dVector([
    [1.0, 0.0, 0.0],  # Red
    [0.0, 1.0, 0.0],  # Green
    [0.0, 0.0, 1.0]   # Blue
])

# Color code the points within the bounding box
indices = bbox.get_point_indices_within_bounding_box(point_cloud.points)
point_cloud.colors = o3d.utility.Vector3dVector([colors[0]] * len(point_cloud.points))
point_cloud.colors[indices] = colors[1]

# Visualize the point cloud with the bounding box
o3d.visualization.draw_geometries([point_cloud, bbox])
import open3d as o3d
import matplotlib
import numpy as np
import argparse

import os
import sys
import json

# For imports
sys.path.append(os.getcwd())
from helpers.geometry import get_3dbbox_corners
from helpers.constants import BBOX_ID_TO_COLOR, BBOX_CLASS_TO_ID, SEM_ID_TO_COLOR
from helpers.visualization import apply_semantic_cmap

parser = argparse.ArgumentParser()
parser.add_argument('--indir', default="/home/arthur/Downloads/CODa",
                    help="Directory path with files downloaded from CODa")
parser.add_argument('--traj', default="0",
                    help="number of trajectory, e.g. 1")
parser.add_argument('--frame', default="0",
                    help="number of frame, e.g. 42")

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle

def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), 
                    center=cylinder_segment.get_center())
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)

def draw_open3d_box(vis, bbox_path):
    bbox_file   = open(bbox_path, 'r')
    bbox_json   = json.load(bbox_file)

    line_sets = []
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

        corner_remap = [3, 0, 2, 1, 7, 4, 6, 5]
        new_bbox_corners = []
        for corner in corner_remap:
            new_bbox_corners.append(bbox_corners[corner].tolist())
        new_bbox_corners = np.array(new_bbox_corners)

        lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
        # Workaround for open3d line width setting
        line_mesh1 = LineMesh(new_bbox_corners, lines, class_color, radius=0.05)
        line_mesh1_geoms = line_mesh1.cylinder_segments

        for line_mesh in line_mesh1_geoms:
            vis.add_geometry(line_mesh)
        # vis.add_geometry(line_set)

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

def main(args):
    """
    Expects the semantic segmentation, bounding box annotation, and point cloud files to 
    be in the same directory. This is meant to be used to evaluation individual frames
    from CODa. For those interested in viewing all annotations in a video format, use the 
    vis_annos_rviz.py script.
    """
    indir=args.indir
    traj=int(args.traj)
    frame=int(args.frame)
    bbox_path = "%s/3d_bbox_os1_%i_%i.json"%(indir, traj, frame)
    sem_path = "%s/3d_semantic_os1_%i_%i.bin"%(indir, traj, frame)
    bin_path  = "%s/3d_raw_os1_%i_%i.bin" % (indir, traj, frame)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().line_width = 10
    vis.get_render_option().background_color = np.ones(3) * 255
    vis = draw_open3d_box(vis, bbox_path)
    vis = draw_open3d_pc(vis, bin_path, sem_path)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

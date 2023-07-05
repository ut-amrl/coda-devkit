import os
from os.path import join
import sys
import pdb
import json

import rospy
from sensor_msgs.msg import PointCloud2, Image

import cv2
import numpy as np

# For imports
sys.path.append(os.getcwd())

from helpers.sensors import get_filename_info, set_filename_by_prefix, read_bin, read_sem_label, set_filename_dir
from helpers.visualization import *
from helpers.metadata import read_metadata_anno
from helpers.constants import METADATA_DIR, CALIBRATION_DIR, TWOD_RAW_DIR, TRED_RAW_DIR, SEM_POINT_SIZE

import argparse
from multiprocessing import Pool
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--mod', default="3d_bbox", help="Visualize annotation type")
parser.add_argument('--gen_all', default=True, help="Generate all annotations")
parser.add_argument('--cfg_file', default='config/checker_annotation.yaml', help="Config file to use for input files")
parser.add_argument('--cam_id', default="cam0", help="Camera id to generate calibration for")

def generate_point_calibration_img(image_np, bin_np, calib_ext_file, calib_intr_file):
    image_pts, pts_mask = project_3dto2d_points(bin_np, calib_ext_file, calib_intr_file)
    in_bounds = np.logical_and(
            np.logical_and(image_pts[:, 0]>=0, image_pts[:, 0]<1224),
            np.logical_and(image_pts[:, 1]>=0, image_pts[:, 1]<1024)
        )
    valid_point_mask = in_bounds & pts_mask
    valid_points = image_pts[valid_point_mask, :]

    for pt in valid_points:
        image_np = cv2.circle(image_np, (pt[0], pt[1]), radius=SEM_POINT_SIZE, color=(0, 0, 255), thickness=-1)
    return image_np
    
def dump_calibration_img(outdir, traj, frame, cam_id, pt_image_np):
    #Assumes images are rectified now
    out_image_dir = join(outdir, TWOD_RAW_DIR, cam_id, traj)
    if not os.path.exists(out_image_dir):
        print("Output image directory %s  does not exist, creating now..." % out_image_dir)
        os.makedirs(out_image_dir)
    out_image_file = set_filename_by_prefix("2d_rect", cam_id, traj, frame)
    out_image_path = join(out_image_dir, out_image_file)

    print("Saving %s projection for traj %s frame %s at %s..." % (cam_id, traj, frame, out_image_path))
    cv2.imwrite(out_image_path, pt_image_np)

def generate_calibration_summary(args):
    cfg, traj = args

    # assumes images are not rectified by default
    TRAJ_SUB_KEY = "[TRAJECTORY]"
    CAM_SUB_KEY = "[CAM]"
    cam_list = ["cam0"]

    traj = str(traj)
    calibextr_path = join(  cfg['indir'], 
        cfg['calibextr_subdir'].replace(TRAJ_SUB_KEY, traj) )
    calibintr_path = join(  cfg['indir'], 
        cfg['calibintr_subdir'].replace(TRAJ_SUB_KEY, traj) )
    tred_path = join( cfg['indir'],
        cfg['tred_subdir'].replace(TRAJ_SUB_KEY, traj)
    )

    twod_paths = []
    for cam_id in cam_list:
        twod_paths.append(join( cfg['indir'],
            cfg['cam_subdir'].replace(CAM_SUB_KEY, cam_id).replace(TRAJ_SUB_KEY, traj)
        ))

    ts_path = join( cfg['indir'],
        cfg['ts_subdir'].replace(TRAJ_SUB_KEY, traj)
    )

    # Project 3D LiDAR to Images
    frame_to_ts_np = np.loadtxt(ts_path)
    for frame, ts in enumerate(frame_to_ts_np):
        if frame % cfg['fr']==0:
            for cam_list_idx, cam_id in enumerate(cam_list):
                twod_img_file   = set_filename_by_prefix(TWOD_RAW_DIR, cam_id, traj, frame)
                twod_img_path = os.path.join(twod_paths[cam_list_idx], twod_img_file)
                image_np = cv2.imread(twod_img_path)

                tred_bin_file = set_filename_by_prefix(TRED_COMP_DIR, "os1", traj, frame)
                tred_bin_path = join(tred_path, tred_bin_file)
                bin_np = read_bin(tred_bin_path)
                pt_image_np = project_3dpoint_image(image_np, bin_np, calibextr_path, calibintr_path)

                dump_calibration_img(cfg['outdir'], traj, frame, cam_id, pt_image_np)

def generate_single_anno_file(args):
    indir, outdir, modality, sensor, traj, frame, cam_list = args
    traj = str(traj)

    # Assumes images are not rectified by default
    TRAJ_SUB_KEY = "[TRAJECTORY]"
    CAM_SUB_KEY = "[CAM]"

    # Set IO paths
    calibextr_path  = join( indir, CALIBRATION_DIR, traj, "calib_os1_to_%s.yaml"%cam_list[0])
    calibintr_path  = join( indir, CALIBRATION_DIR, traj, "calib_%s_intrinsics.yaml"%cam_list[0])
    tred_anno_path  = set_filename_dir(indir, modality, sensor, traj, frame)
    twod_img_path   = set_filename_dir(indir, TWOD_RAW_DIR, cam_list[0], traj, frame)
    
    # Project 3D anno to 2D
    image_np = cv2.imread(twod_img_path)
    if "3d_bbox"==modality:
        tred_anno_dict = json.load(open(tred_anno_path))
        tred_anno_image = project_3dbbox_image(tred_anno_dict, calibextr_path, calibintr_path, image_np)
    elif "3d_semantic"==modality:
        pc_path = set_filename_dir(indir, TRED_RAW_DIR, sensor, traj, frame)
        pc_np   = read_bin(pc_path, keep_intensity=False)
        tred_anno_image = project_3dpoint_image(image_np, pc_np, calibextr_path, calibintr_path, tred_anno_path)

    dump_calibration_img(outdir, traj, frame, cam_list[0], tred_anno_image)

def generate_annotation_visualization(args):
    cfg, traj, frame = args

    # assumes images are not rectified by default
    TRAJ_SUB_KEY = "[TRAJECTORY]"
    CAM_SUB_KEY = "[CAM]"
    cam_list = ["cam0"]

    traj = str(traj)
    calibextr_path = join(  cfg['indir'], 
        cfg['calibextr_subdir'].replace(TRAJ_SUB_KEY, traj) )
    calibintr_path = join(  cfg['indir'], 
        cfg['calibintr_subdir'].replace(TRAJ_SUB_KEY, traj) )
    bbox_path = join( cfg['indir'],
        cfg['bbox_subdir'].replace(TRAJ_SUB_KEY, traj)
    )

    twod_paths = []
    for cam_id in cam_list:
        twod_paths.append(join( cfg['indir'],
            cfg['cam_subdir'].replace(CAM_SUB_KEY, cam_id).replace(TRAJ_SUB_KEY, traj)
        ))

    # Project 3D annotations to images
    for cam_list_idx, cam_id in enumerate(cam_list):
        twod_img_file   = set_filename_by_prefix(TWOD_RAW_DIR, cam_id, traj, frame)
        twod_img_path = os.path.join(twod_paths[cam_list_idx], twod_img_file)
        image_np = cv2.imread(twod_img_path)

        tred_bbox_file = set_filename_by_prefix(TRED_BBOX_LABEL_DIR, "os1", traj, frame)
        tred_bbox_path = join(bbox_path, tred_bbox_file)
        bbox_anno_dict = json.load(open(tred_bbox_path))
        tred_bbox_image = project_3dbbox_image(bbox_anno_dict, calibextr_path, calibintr_path, image_np)

        dump_calibration_img(cfg['outdir'], traj, frame, cam_id, tred_bbox_image)

def main(args):
    """
    indir - CODa directory (assumes 3d_labels exists)
    outdir - directory to save bbox projections to

    This script can be used to project the point cloud to corresponding images. 
    """
    cfg_path    = args.cfg_file
    gen_all     = args.gen_all
    modality    = args.mod
    cam_id      = args.cam_id

    checker_fp = os.path.join(os.getcwd(), cfg_path)
    with open(checker_fp, 'r') as checker_file:
        checker_cfg = yaml.safe_load(checker_file)
        indir       = checker_cfg["indir"]
        outdir       = checker_cfg["outdir"]

        if gen_all:
            for trajectory in np.arange(23):
                # Read all 3d bbox files from coda metadata
                metadata_path = join(indir, METADATA_DIR, "%i.json"%trajectory)
                if not os.path.exists(metadata_path):
                    print("Trajectory %i does not contain metadata file %s" % (trajectory, metadata_path))
                    continue
                anno_subpaths = read_metadata_anno(metadata_path, modality=modality)
                num_annos = len(anno_subpaths)
                if num_annos==0:
                    continue
                
                _, sensor_name, _, frame = get_filename_info(anno_subpaths[0].split("/")[-1])
                indir_list = [indir] * num_annos
                outdir_list = [outdir] * num_annos
                modality_list = [modality] * num_annos
                sensor_list = [sensor_name] * num_annos
                traj_list = [trajectory] * num_annos
                frame_list = []
                cam_list = [[cam_id]] * num_annos
                for anno_path in anno_subpaths:
                    modality, sensor_name, _, frame = get_filename_info(anno_path.split("/")[-1])

                    # Make trajectory ehre
                    output_img_dir = join(outdir, TWOD_RAW_DIR, cam_id, str(trajectory))
                    if not os.path.exists(output_img_dir):
                        print("Output image dir for %s does not exist, creating..."%output_img_dir)
                        os.makedirs(output_img_dir)

                    #Uncomment below for testing
                    # if not (trajectory==9 and int(frame) == 1529):
                    #     continue
                    frame_list.append(frame)

                    #Uncomment below for testing
                    # generate_single_anno_file((indir, ".", modality, sensor_name, trajectory, frame, [cam_id]))
                    # import pdb; pdb.set_trace()

                pool = Pool(processes=checker_cfg['num_workers'])
                for _ in tqdm.tqdm(pool.imap_unordered(generate_single_anno_file, \
                    zip(indir_list, outdir_list, modality_list, sensor_list, traj_list, frame_list, cam_list)), total=num_annos):
                    pass

                # pool = Pool(processes=checker_cfg['num_workers'])
                # for _ in tqdm.tqdm(pool.imap_unordered(generate_calibration_summary, \
                #     zip(indir_list, outdir_list, modality_list, sensor_list, traj_list, frame_list)), total=num_annos):
                #     pass

    # indir   = "/robodata/arthurz/Datasets/CODa"
    # trajectory = int(args.traj)
    # start_frame = int(args.frame)
    # outdir  = args.outdir
    # assert os.path.exists(outdir), "Out directory is not empty and does not exist %s " % args.outdir

    # use_wcs = False
    # use_sem = False

    # #Project 3d bbox annotations to 2d
    # calib_ext_file = join(indir, "calibrations", str(trajectory), "calib_os1_to_cam0.yaml")
    # calib_intr_file= join(indir, "calibrations", str(trajectory), "calib_cam0_intrinsics.yaml")
    # tred_bin_dir   = join(indir, "3d_raw", "os1", str(trajectory))
    # sem_tred_anno_dir   = join(indir, "3d_semantic", "os1", str(trajectory))
    # tred_anno_dir  = join(indir, "3d_bbox", "os1", str(trajectory))
    # # tred_anno_dir = "/robodata/arthurz/Benchmarks/unsupda/ST3D/tools/preds/%i" % trajectory
    # twod_anno_dir  = join(indir, "2d_rect", "cam0", str(trajectory))

    # #Locate closest pose from frame
    # ts_to_frame_path = join(indir, "timestamps", "%s_frame_to_ts.txt"%trajectory)
    # ts_to_poses_path = join(indir, "poses", "%s.txt"%trajectory)
    # frame_to_poses_np = np.loadtxt(ts_to_poses_path).reshape(-1, 8)
    # frame_to_ts_np = np.loadtxt(ts_to_frame_path)

    # # frame_order  = np.argsort([ int(get_filename_info(file)[-1]) for file in os.listdir(tred_anno_dir) ])
    # # tred_anno_files = np.array(os.listdir(tred_anno_dir))[frame_order]
    
    # frame_order  = np.argsort([ int(get_filename_info(file)[-1]) for file in os.listdir(tred_bin_dir) ])
    # bin_files   = np.array(os.listdir(tred_bin_dir))[frame_order]

    # image_pub   = rospy.Publisher('/coda/stereo/%s'%"cam0", Image, queue_size=10)
    # pc_pub      = rospy.Publisher('/coda/ouster/lidar_packets', PointCloud2, queue_size=10)
    # dense_poses = densify_poses_between_ts(frame_to_poses_np, frame_to_ts_np)

    # for (idx, bin_file) in enumerate(bin_files):
    #     if idx < start_frame:
    #         continue

    #     modality, sensor_name, trajectory, frame = get_filename_info(bin_file)

    #     bin_path = join(tred_bin_dir, bin_file)
    #     bin_np = read_bin(bin_path, False)

    #     wcs_pose = None
    #     if use_wcs:
    #         wcs_pose = pose_to_homo(dense_poses[idx])
    #         bin_np_homo = np.hstack((bin_np, np.ones( (bin_np.shape[0], 1) ) ))
    #         bin_np      = (wcs_pose @ bin_np_homo.T).T[:, :3]

    #     twod_img_file   = set_filename_by_prefix("2d_rect", "cam0", trajectory, str(int(frame)))
    #     twod_img_path = join(twod_anno_dir, twod_img_file)
    #     image = cv2.imread(twod_img_path)

    #     if use_sem:
    #         sem_tred_label_file = set_filename_by_prefix("3d_semantic", "os1", trajectory, frame)
    #         sem_tred_label_path = join(sem_tred_anno_dir, sem_tred_label_file)
    #         if not os.path.exists(sem_tred_label_path):
    #             print("3D Label File Does Not Exist %s " % sem_tred_label_path)
    #     else:
    #         tred_label_file = set_filename_by_prefix("3d_bbox", "os1", trajectory, frame)
    #         tred_label_path = join(tred_anno_dir, tred_label_file)
    #         assert os.path.exists(tred_label_path), "3D Label File Does Not Exist %s " % tred_label_path
        
    #     if use_sem:
    #         valid_points, valid_points_mask = project_3dsem_image(bin_np, calib_ext_file, calib_intr_file, wcs_pose)

    #         if os.path.exists(sem_tred_label_path):
    #             sem_labels_np = read_sem_label(sem_tred_label_path)
    #             valid_sem_labels_np = sem_labels_np[valid_points_mask]

    #             twod_sem_image = draw_2d_sem(image, valid_points, valid_sem_labels_np)
    #         else:
    #             for pt in valid_points:
    #                 twod_sem_image = cv2.circle(image, (pt[0], pt[1]), radius=1, color=(0, 0, 255))
            
    #         if outdir!='.':
    #             image_file = set_filename_by_prefix("2d_rect", "cam0", str(trajectory), str(frame))
    #             image_path = join(outdir, image_file)

    #             cv2.imwrite(image_path, twod_sem_image)
    #         else:
    #             cv2.imwrite("testsegmentation.png", twod_sem_image)
    #     else:
    #         anno_dict       = json.load(open(tred_label_path))
    #         tred_bbox_image = project_3dbbox_image(anno_dict, calib_ext_file, calib_intr_file, image, )
            
    #         if outdir!='.':
    #             image_file = set_filename_by_prefix("2d_rect", "cam0", str(trajectory), str(frame))
    #             image_path = join(outdir, image_file)

    #             cv2.imwrite(image_path, tred_bbox_image)
    #         else:
    #             cv2.imwrite("testbboxcalibration.png", tred_bbox_image)
            
    #         # Uncomment Below to Visualize 2D
    #         bbox_coords = project_3dto2d_bbox_image(anno_dict, calib_ext_file, calib_intr_file)
    #         image = cv2.imread(twod_img_path)
    #         twod_bbox_image = draw_2d_bbox(image, bbox_coords)
    #         cv2.imwrite("test2dbboxcalibration.png", twod_bbox_image)
        # import pdb; pdb.set_trace()

        # cv2.imshow('img', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
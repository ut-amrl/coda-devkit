
import os
import sys
import pdb
import yaml
import time
import numpy as np
from helpers.sensors import get_filename_info, set_filename_by_prefix

#ROS Imports
import rospy
from sensor_msgs.msg import PointCloud2, Image

# For imports
sys.path.append(os.getcwd())

#CustomImports
from helpers.visualization import *
from helpers.sensors import read_bin, read_sem
from helpers.constants import *

def save_bboxmask_and_pose(object_mask_dir, trajectory, frame, bin_np, threed_label_file, pose_np):
    output_mask = get_points_in_bboxes(bin_np, threed_label_file)
    bin_np      = bin_np[output_mask, :]

    mask_outdir = os.path.join(object_mask_dir, "%s"%trajectory)
    if not os.path.exists(mask_outdir):
        print("Creating mask out directory at %s", mask_outdir)
        os.makedirs(mask_outdir)
    mask_filepath = os.path.join(mask_outdir, "%s.npy"%frame)
    pose_filepath = os.path.join(mask_outdir, "%s_pose"%frame)
    np.save(mask_filepath, output_mask)
    np.save(pose_filepath, pose_np)

def main():

    settings_fp = os.path.join(os.getcwd(), "config/visualize.yaml")
    with open(settings_fp, 'r') as settings_file:
        settings = yaml.safe_load(settings_file)
        viz_2dbbox      = settings['viz_2dbbox']
        viz_img         = settings['viz_img']
        viz_pc          = settings['viz_pc']
        viz_3dbbox      = settings['viz_3dbbox']
        viz_pose        = settings['viz_pose']
        viz_2d_terrain  = settings['viz_2dterrain']

        root_dir        = settings['root_dir']
        img_root_dir    = os.path.join(root_dir, DIR_2D_RAW)
        bin_root_dir    = os.path.join(root_dir, DIR_3D_RAW, 'os1')
        pose_root_dir   = os.path.join(root_dir, "poses")
        bbox3d_root_dir = os.path.join(root_dir, DIR_BBOX_LABEL, 'os1')
        sem3d_root_dir  = os.path.join(root_dir, DIR_SEMANTIC_LABEL, 'os1')
        ts_root_dir     = os.path.join(root_dir, "timestamps")
        calib_dir       = os.path.join(root_dir, "calibrations")

        trajectory_frames   = settings['trajectory_frames']
        ds_rate         = settings['downsample_rate']
        save_object_masks= settings['save_object_masks']
        object_mask_dir = settings['object_mask_dir']
        use_wcs         = settings['use_wcs']

        if viz_img:
            assert os.path.exists(img_root_dir), "%s does not exist, provide valid img_root_dir."
        if viz_pc:
            assert os.path.exists(bin_root_dir), "%s does not exist, provide valid bin_root_dir."
        if viz_pose:
            assert os.path.exists(pose_root_dir), "%s does not exist, provide valid pose_root_dir."
        if viz_pc:
            assert os.path.exists(ts_root_dir), "%s does not exist, provide valid ts_root_dir."
        if viz_3dbbox:
            assert os.path.exists(bbox3d_root_dir), "%s does not exist, provide valid 3d bbox dir."
        if viz_2d_terrain:
            assert os.path.exists(sem3d_root_dir), "%s does not exist, provide valid 3d semantic dir."

    assert len(sys.argv)>=2, "Specify the trajectory number you wish to visualize\n"
    trajectory = sys.argv[1]
    
    bin_dir = os.path.join(bin_root_dir, str(trajectory))
    cam_list= sorted([cam for cam in os.listdir(img_root_dir) if os.path.isdir(os.path.join(img_root_dir, cam))])
    cam_dirs= [ os.path.join(img_root_dir, cam, trajectory) for cam in cam_list]
    assert os.path.exists(bin_dir), "%s does not exist, generate the trajectory's .bin files first\n"

    #Initialize ros point cloud publisher
    rospy.init_node('bin_publisher', anonymous=True)
    pc_pub = rospy.Publisher('/coda/bin/points', PointCloud2, queue_size=10)
    img_pubs = [rospy.Publisher('/coda/stereo/%s'%cam, Image, queue_size=10) for cam in cam_list ]
    pose_pub = rospy.Publisher('/coda/pose', PoseStamped, queue_size=10)
    marker_pub = rospy.Publisher('/coda/bbox', MarkerArray, queue_size=10)
    pub_rate = rospy.Rate(5) # Publish at 10 hz

    frame_to_ts_file = os.path.join(ts_root_dir, "%s_frame_to_ts.txt"%trajectory)
    frame_ts_np = np.fromfile(frame_to_ts_file, sep=' ').reshape(-1, 1)
    if viz_pose:
        pose_file   = os.path.join(pose_root_dir, "%s.txt"%trajectory)
        pose_np     = np.fromfile(pose_file, sep=' ').reshape(-1, 8)

    trajectory_frame_enum = enumerate(range(trajectory_frames[0], trajectory_frames[1]))
    if trajectory_frames[0]==-1:
        
        trajectory_frames = np.array([ get_filename_info(file)[-1] for file in os.listdir(bin_dir) ])
        trajectory_frames = trajectory_frames[np.argsort([int(frame) for frame in trajectory_frames])]
        trajectory_frame_enum = enumerate(trajectory_frames)
        
    for (frame_idx, frame) in trajectory_frame_enum:
        frame_time = rospy.get_rostime()

        if frame_idx%ds_rate==0:
            print("Visualizing frame ", frame)

            #Publish pose
            ts  = frame_ts_np[int(frame)][0]
            if viz_pose:
                pose = find_closest_pose(pose_np, ts)
                pub_pose(pose_pub, pose, frame, frame_time)

            bbox_filename = set_filename_by_prefix(DIR_BBOX_LABEL, "os1",
                    str(trajectory), frame)
            sem_filename    = set_filename_by_prefix(DIR_SEMANTIC_LABEL, "os1",
                    str(trajectory), frame)
            threed_bbox_file    = os.path.join(bbox3d_root_dir, str(trajectory), bbox_filename)
            threed_sem_file     = os.path.join(sem3d_root_dir, str(trajectory), sem_filename)

            bin_file = set_filename_by_prefix(DIR_3D_RAW, "os1", trajectory, frame)
            bin_path = os.path.join(bin_dir, bin_file)

            #Publish images
            if viz_img:
                for (cam_idx, cam_dir) in enumerate(cam_dirs):

                    cam_name = cam_list[cam_idx]
                    img_file = set_filename_by_prefix(DIR_2D_RAW, cam_name, trajectory, frame)
                    img_path = os.path.join(cam_dir, img_file)
                    calib_ext_file = os.path.join(calib_dir, str(trajectory), "calib_os1_to_cam0.yaml")
                    calib_intr_file= os.path.join(calib_dir, str(trajectory), "calib_cam0_intrinsics.yaml")

                    image = cv2.imread(img_path)
                    is_image_changed    = False
                    if viz_2dbbox and cam_name=="cam0" and os.path.exists(threed_bbox_file):
                        anno_dict       = json.load(open(threed_bbox_file))

                        bbox_pts, bbox_mask, bbox_idxs = project_3dto2d_bbox(anno_dict, calib_ext_file, calib_intr_file)

                        for obj_idx in range(0, bbox_pts.shape[0]):
                            in_bounds = np.logical_and(
                                np.logical_and(bbox_pts[obj_idx, :, 0]>=0, bbox_pts[obj_idx, :, 0]<1224),
                                np.logical_and(bbox_pts[obj_idx, :, 1]>=0, bbox_pts[obj_idx, :, 1]<1024)
                            )

                            valid_point_mask = bbox_mask[obj_idx] & in_bounds
                            valid_points = bbox_pts[obj_idx, valid_point_mask, :]
                            if valid_points.shape[0]==0:
                                continue

                            bbox_idx = bbox_idxs[obj_idx][0]
                            obj_id = BBOX_CLASS_TO_ID[anno_dict["3dbbox"][bbox_idx]["classId"]]
                            obj_color = BBOX_ID_TO_COLOR[obj_id]
                            image = draw_bbox(image, valid_points, valid_point_mask, color=obj_color)
                            is_image_changed = True
                    
                    #Project Terrain Segmentation to 2D
                    if viz_2d_terrain and cam_name=="cam0" and os.path.exists(threed_sem_file):
                        bin_points  = read_bin(bin_path, False).astype(np.float64)
                        sem_labels  = read_sem(threed_sem_file)

                        sem_pts, sem_mask = project_3dto2d_points(bin_points, calib_ext_file, calib_intr_file)
                        in_bounds = np.logical_and(
                                np.logical_and(sem_pts[:, 0]>=0, sem_pts[:, 0]<1224),
                                np.logical_and(sem_pts[:, 1]>=0, sem_pts[:, 1]<1024)
                            )
                        valid_point_mask = in_bounds & sem_mask
                        valid_points = sem_pts[valid_point_mask, :]

                        sem_labels = sem_labels[valid_point_mask, :].reshape(-1,)
                        for pt_idx, pt in enumerate(valid_points):
                            bgr_color = (
                                SEM_ID_TO_COLOR[sem_labels[pt_idx]][2],
                                SEM_ID_TO_COLOR[sem_labels[pt_idx]][1],
                                SEM_ID_TO_COLOR[sem_labels[pt_idx]][0]
                            )
                            image = cv2.circle(image, (pt[0], pt[1]), radius=2, 
                                color=bgr_color, thickness=-1)
                        
                        is_image_changed = True

                    if is_image_changed:
                        img_label_dir = cam_dir.replace(DIR_2D_RAW, DIR_2D_SEMANTIC)
                        img_label_file = set_filename_by_prefix(DIR_2D_SEMANTIC, cam_name, trajectory, frame)
                        img_path = os.path.join(img_label_dir, img_label_file)
                        if not os.path.exists(img_label_dir):
                            print("Creating image semantic directory at %s..."%img_label_dir)
                            os.makedirs(img_label_dir)
                        cv2.imwrite(img_path, image)

                    pub_img(img_pubs[cam_idx], img_path)
                    # import pdb; pdb.set_trace()
                    
            #Publish point cloud
            if viz_pc:
                bin_np = read_bin(bin_path, True)
                intensity_np = bin_np[:, 3].reshape(-1, 1)
                bin_np = bin_np[:, :3]

                if use_wcs:
                    pose_mat = np.eye(4)
                    pose_mat[:3, :3] = R.from_quat([pose[5], pose[6], pose[7], pose[4]]).as_matrix()
                    pose_mat[:3, 3] = pose[1:4]
                    wcs_bin_np = (pose_mat @ np.hstack((bin_np[:, :3], np.ones((bin_np.shape[0], 1)))).T).T
                    bin_np = wcs_bin_np[:, :3].astype(np.float32)

                if viz_3dbbox and save_object_masks and os.path.exists(threed_bbox_file):
                    # Only return points contained within bboxes
                    output_mask = get_points_in_bboxes(bin_np, threed_bbox_file)
                    bin_np      = bin_np[output_mask, :]
                    save_bboxmask_and_pose(object_mask_dir, trajectory, frame, bin_np, threed_bbox_file, pose_np)
                
                full_bin_np = np.hstack( (bin_np, intensity_np) )
                pub_pc_to_rviz(full_bin_np, pc_pub, frame_time, frame_id="os_sensor")
            
            #Publish bboxes
            if viz_3dbbox:
                print("threed ", threed_bbox_file)
                # pdb.set_trace()
                if os.path.exists(threed_bbox_file):
                    pub_3dbbox_to_rviz(marker_pub, threed_bbox_file, frame_time, verbose=True)
                    time.sleep(0.1)
                    # pdb.set_trace()
                else:
                    print("No annotations available for frame %s, skipping..." % str(frame) )

            pub_rate.sleep()

if __name__ == "__main__":
    main()

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
from helpers.geometry import inter_pose
from helpers.constants import OS1_POINTCLOUD_SHAPE

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
        viz_anno    = settings['viz_anno']
        viz_img     = settings['viz_img']
        viz_pc      = settings['viz_pc']
        viz_obj     = settings['viz_obj']
        viz_pose    = settings['viz_pose']

        img_root_dir    = settings['img_root_dir']
        bin_root_dir    = settings['bin_root_dir']
        pose_root_dir   = settings['pose_root_dir']
        ts_root_dir     = settings['ts_root_dir']
        bbox3d_root_dir = settings['bbox3d_root_dir']
        trajectory_frames   = settings['trajectory_frames']
        ds_rate         = settings['downsample_rate']
        save_object_mask= settings['save_object_masks']
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
        if viz_anno:
            assert os.path.exists(bbox3d_root_dir), "%s does not exist, provide valid ts_root_dir."

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

            label_filename = set_filename_by_prefix("3d_label", "os1",
                    str(trajectory), frame)
            threed_label_file = os.path.join(bbox3d_root_dir, str(trajectory), 
                label_filename)

            #Publish images
            if viz_img:
                for (cam_idx, cam_dir) in enumerate(cam_dirs):

                    cam_name = cam_list[cam_idx]
                    img_file = set_filename_by_prefix("2d_raw", cam_name, trajectory, frame)
                    img_path = os.path.join(cam_dir, img_file)
                    # print(cam_name)
                    if viz_anno and cam_name=="cam0" and os.path.exists(threed_label_file):
                        anno_dict       = json.load(open(threed_label_file))
                        image = cv2.imread(img_path)
                        indir="/home/arthur/AMRL/Datasets/CODa"
                        calib_ext_file = os.path.join(indir, "calibrations", str(trajectory), "calib_os1_to_cam0.yaml")
                        calib_intr_file= os.path.join(indir, "calibrations", str(trajectory), "calib_cam0_intrinsics.yaml")
                        bbox_pts, bbox_mask, bbox_idxs = project_3dto2d_bbox(anno_dict, pose, calib_ext_file, calib_intr_file)

                        for obj_idx in range(0, bbox_pts.shape[0]):
                            in_bounds = np.logical_and(
                                np.logical_and(bbox_pts[obj_idx, :, 0]>=0, bbox_pts[obj_idx, :, 0]<1224),
                                np.logical_and(bbox_pts[obj_idx, :, 1]>=0, bbox_pts[obj_idx, :, 1]<1024)
                            )
                            # pdb.set_trace()
                            valid_point_mask = bbox_mask[obj_idx] & in_bounds
                            valid_points = bbox_pts[obj_idx, valid_point_mask, :]
                            if valid_points.shape[0]==0:
                                continue

                            bbox_idx = bbox_idxs[obj_idx][0]
                            obj_id = CLASS_TO_ID[anno_dict["3dannotations"][bbox_idx]["classId"]]
                            obj_color = ID_TO_COLOR[obj_id]
                            image = draw_bbox(image, valid_points, valid_point_mask, color=obj_color)

                        img_label_dir = cam_dir.replace("raw", "label")
                        img_label_file = set_filename_by_prefix("2d_label", cam_name, trajectory, frame)
                        if not os.path.exists(img_label_dir):
                            os.makedirs(img_label_dir)

                        img_path = os.path.join(img_label_dir, img_label_file)
                        cv2.imwrite(img_path, image)

                    pub_img(img_pubs[cam_idx], img_path)

            #Publish point cloud
            if viz_pc:
                bin_file = set_filename_by_prefix("3d_raw", "os1", trajectory, frame)
                bin_path = os.path.join(bin_dir, bin_file)

                bin_np = np.fromfile(bin_path, dtype=np.float32)

                if use_wcs:
                    bin_np = bin_np.reshape(-1, 3)
                    pose_mat = np.eye(4)
                    pose_mat[:3, :3] = R.from_quat([pose[5], pose[6], pose[7], pose[4]]).as_matrix()
                    pose_mat[:3, 3] = pose[1:4]
                    wcs_bin_np = (pose_mat @ np.hstack((bin_np, np.ones((bin_np.shape[0], 1)))).T).T
                    bin_np = wcs_bin_np[:, :3].astype(np.float32)

                if viz_obj and save_object_mask and os.path.exists(threed_label_file):
                    # Only return points contained within bboxes
                    bin_np = bin_np.reshape(-1, 3)
                    output_mask = get_points_in_bboxes(bin_np, threed_label_file)
                    bin_np      = bin_np[output_mask, :]
                    save_bboxmask_and_pose(object_mask_dir, trajectory, frame, bin_np, threed_label_file, pose_np)
                else:
                    bin_np = bin_np.reshape(OS1_POINTCLOUD_SHAPE)

                pub_pc_to_rviz(bin_np, pc_pub, frame_time, frame_id="os_sensor")
            #Publish bboxes
            if viz_anno:
                if os.path.exists(threed_label_file):
                    pub_3dbbox_to_rviz(marker_pub, threed_label_file, frame_time, verbose=True)
                    time.sleep(1)
                    # pdb.set_trace()
                else:
                    print("No annotations available for frame %s, skipping..." % str(frame) )
            
            pub_rate.sleep()

if __name__ == "__main__":
    main()
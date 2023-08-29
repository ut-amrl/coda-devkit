# Getting Started

The CODa package provides tools for converting your own rosbag files into individual files organized using the CODa file structure. For instance, the script will synchronize any number of topics provided, assign these synchronized topics to a unified frame number, and save each topic (LiDAR, camera, audio, inertial) to its own data file. Furthermore, the script will also republish specified topics in the bag file over ROS to work in conjunction with packages with ROS interfaces. Beyond this, the package also provides visualization, LiDAR camera calibration, and annotation workflow tools.

## CODA File Structure

We describe the CODa file structure in the [data report](DATA_REPORT.md) in detail.

## API Features

### Ground Truth Visualization 
- Visualize 3D annotations on LiDAR point clouds in open3d. See vis_open3d.py
- Generate 3D bounding box and semantic segmentation annotations on 2D images. See check_lidar_annos.py
- Visualize reconstructed point cloud using pseudo-ground truth LiDAR poses. See pub_3d_legoloam_to_2d_rviz.py

### Calibration Visualization
- Visualize RGB image next to corresponding stereo depth image. See viz_stereo_rgb.py
- Visualize stereo depth cloud and closest LiDAR point cloud on rviz. See check_stereo_pc.py
- Generate 3D point cloud projection to camera image visualizations. See gen_lidar_egocomp.py

### Data Processing
- Convert rosbag file to CODa file structure. See decode_multiday.py

## API Documentation

### Visualize on Open3D (vis_open3d.py)


### Visualize Offline on 2D Images (check_lidar_annos.py)

This script can be used to project the point cloud annotations to RGB images for offline visualization when open3d is not available. This script requires settings the config/checker_annotation.yaml file to work properly. You will only need to configure this file once.

```
indir - Set this to the absolute filepath to where CODa is downloaded. If you downloaded the dataset using pip, this can be determined by running `echo $PATH`. 
outdir - Directory where the annotated images will be written to.
```

The script accepts arguments to visualize either 3D bounding box or 3D semantic segmentation annotations. It can also be used to view

Example:
```
python check_lidar_annos.py --mod 3d_bbox --cfg_file config/checker_annotation.yaml
```


## Convert Rosbag to CODa Dataset

Modify the `config/bagdecoder.yaml` file before using coda to converting your rosbags.

Example: *`python scripts/decode_multiday.py --config config/bagdecoder.yaml`*

#### Settings Description

- `repository_root` - This is the parent folder that contains all of your bag files. We expect this directory to contain subdirectories that contain your bag files. In the example below, you provide the path to the `bags` directory.
- `dataset_output_root` - This specifies the output directory that CODa will be generated in. This must be a valid directory regardless of whether or not you configure the tool to generate data.
- `bag_date` - This is the name of the folder containing the bag files that you want to convert. This does not have to be a date, but we will use dates as the subdirectory names in our provided data.
- `gen_data` - Boolean specifying whether or not to generate the CODa dataset. If you set this to `false`, the script will still publish processed topics over ROS.
- `sync_topics` - List of topics to synchronize. The script will synchronize all topics in this list and assign them to a synchronized frame number. The topics in this list must be present in the bag file.
- `sensor_topics` - List of sensor_topics to save to the CODa dataset. The topics in this list may be duplicated in the `sync_topics` list, but topics in the `sync_topics` list will still be synchronized regardless.
- `bags_to_process` - List of bag filenames to process. If this list is empty, the script will process all bag files in the `bag_date` directory.
- `bags_to_traj_ids` - List of trajectory IDs to assign to each bag file. Each entry in `bags_to_process` should match to the ID at the same position in the list as itself. If this list is empty, the script will assign trajectory IDs in the order that the bag files are processed beginning from 0 and increasing by one.

<p align="center">
  <img src="./bag_decoder.png" width="70%">
</p>

## Checking LiDAR Camera Calibrations

<p align="center">
  <img src="./check_calibrations.png" width="70%">
</p>

Example: *`python scripts/check_calibrations.py`*

Those interested in checking the accuracy of our LiDAR camera calibrations can be use the `scripts/check_calibration.py` script. This script will load the annotated LiDAR and image data from CODa and project 3D bounding boxes from LiDAR annotations as 6DOF bounding boxes on the 2D image. This simultaneously checks both the quality of the LiDAR and FLIR camera synchronization and the quality of the annotations. The necessary configuration variables are variables `indir` and `trajectory` in the python script. 

## Visualize CODa Dataset

<p align="center">
  <img src="./visualize_data.png" width="70%">
</p>

Example: *`python scripts/visualize_data.py`*

The CODa can be visualized using `scripts/visualize_data.py`. To modify the visualization topics for your use case, modify `config/visualize.yaml` before running the script. The visualization script will publish the specified topics over ROS and can be viewed using RViz. 

#### Settings Description

- `viz_img` - Boolean specifying whether or not to visualize the FLIR camera images. By default it will visualize the left camera.
- `viz_pc`  - Boolean specifying whether or not to visualize the LiDAR point clouds. By default it will visualize the `os1` point cloud.
- `viz_3danno` - Boolean specifying whether or not to visualize the 3D bounding boxes. By default it will visualize bounding boxes for the `os1` point cloud. If the bounding boxes do not exist for the frame, this variable does nothing.
- `viz_pose`    - Boolean specifying whether or not to visualize the pose. By default it will visualize the pose of the `os1` sensor.
- `viz_2danno`  - Boolean specifying whether or not to visualize the 2D bounding boxes. By default it will visualize bounding boxes for the left FLIR camera. If the bounding boxes do not exist for the frame, this variable does nothing.
- `use_wcs`     - Boolean specifying whether or not to use the world coordinate system. If this is set to `true`, the point clouds will be w.r.t the world reference frame. If this is set to `false`, the visualization will be in the egocentric LiDAR reference frame.
- `save_object_masks`   - Boolean specifying whether or not to save the object masks for points contained in bounding boxes. If this is set to `true`, the object masks will be saved to the `object_mask_dir` directory. The masks will be saved as a 2D numpy array with the frame number as the filename.
- `object_mask_dir` - Directory to save the object masks to. This directory must exist before running the script.
- `img_root_dir`    - Directory containing the image directories. This will publish all images in the subdirectories over ROS. This directory must exist before running the script.
- `bin_root_dir`    - Directory containing the point clouds collected by a specific sensor. This will publish all point clouds to that specific sensor over ROS. This directory must exist before running the script.
- `pose_root_dir`   - Directory containing the ground truth pose. This can be any valid pose in the CODa pose file format.
- `ts_root_dir`     - Directory containing the timestamps for the synchronized sensors. This directory must exist before running the script.
- `bbox3d_root_dir` - Directory containing the 3D bounding boxes annotation files for a specific sensor. This directory must exist before running the script.
- `trajectory_frames`   - List containing the start and end frame to visualize for the specified trajectory. If the contents of this two element array are -1, the script will visualize all frames for the specified trajectory.
- `downsample_rate`     - Integer specifying the downsample rate for the frames to be visualized. If this is set to 2, only every other frame's sensor data will be published.

## Annotatate CODa Dataset

<p align="center">
  <img src="./annotation.png" width="70%">
</p>

Example: *`python scripts/encode_annotations.py`* *`python scripts/decode_annotations.py`*

Those interested in contributing to the dataset may be interested in the annotation encoding and decoding tools that we provide. These tools convert sensor data in the CODa to sensor formats required by 3D annotation vendors and open source tools. The `scripts/encode_annotations.py` script converts the CODa annotations to the format required by annotation tools/services. The settings for this can be found in `config/encoder.yaml`. After annotating the sensor data, these annotations can be converted back to the CODa annotation format using the `scripts/decode_annotations.py` script. The settings for this can be found in `config/annotationdecoder.yaml`. We currently support the following annotation tools/services:

1. Sagemaker Ground Truth
2. DeepenAI
3. ScaleAI
4. ewannotate (ROS based open source tool)

```
###########################################################################
#                 UT Austin Campus Object Dataset (UT CODa)               #
#                                                                         #
#   Principal Investigators                                               #
#   Arthur Zhang                        Dr. Joydeep Biswas                #
#   University of Texas at Austin       University of Texas at Austin     #
#   2317 Speedway, Austin, TX 78712     2317 Speedway, Austin, TX 78712   #
#   arthurz@cs.utexas.edu               joydeepb@cs.utexas.edu            #
#                                                                         #
#   Co-Investigators                                                      #
#       Chaitanya Eranki        (chai77@cs.utexas.edu)                    #
#       Christina Zhang         (yymzhang@cs.utexas.edu)                  #
#       Raymond Hong            (raymond22@cs.utexas.edu)                 #
#       Pranav Kalyani          (pranavkalyani@cs.utexas.edu)             #
#       Lochana Kalyanaraman    (lochanakalyan@cs.utexas.edu)             #
#       Arsh Gamare             (arsh.gamare@utexas.edu)                  #
#       Maria Esteva            (maria@tacc.utexas.edu)                   #
#                                                                         #
#   Data Collection Period/Location                                       #
#       2023-01-16  to  2023-02-10                                        #
#       University of Texas at Austin (30.2849° N, 97.7341° W)            #
#                                                                         #
#   Supported By:                                                         #
#       Autonomous Mobile Robotics Laboratory (AMRL)                      #
#       University of Texas at Austin                                     #
#       Texas Advanced Computing Center                                   #
#                                                                         #
###########################################################################
```

## Terms of Use

UT CODa is available for non-commerical under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (“CC BY-NC-SA 4.0”). The CC BY-NC-SA 4.0 may be accessed at https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode When You download or use the Datasets from the Websites or elsewhere, You are agreeing to comply with the terms of CC BY-NC-SA 4.0 as applicable, and also agreeing to the Dataset Terms. Where these Dataset Terms conflict with the terms of CC BY-NC-SA 4.0, these Dataset Terms shall prevail.

# Citation
If you use our dataset of the tools, we would appreciate if you cite our paper \TODO arthur

```
@inproceedings{zhang2023ijjr,
  author = {A. Zhang, C. Eranki, C. Zhang, R. Hong, P. Kalyani, L. Kalyanaraman, A. Gamare, M. Esteva, J. Biswas},
  title = {{Towards Robust 3D Robot Perception in Urban Environments: The UT Campus Object Dataset}},
  booktitle = {},
  year = {2023}
}
```

# Data Format Description

The data for UT CODa is divided into the following folders.


```
- CODa
    - 2d_raw
        - {cam0/cam1}
            - {SEQUENCE}
                - 2d_raw_{cam0/cam1}_{SEQUENCE}_{FRAME}.png
        - {cam2/cam3/cam4}
            - {SEQUENCE}
                - 2d_raw_{cam2/cam3/cam4}_{SEQUENCE}_{TIMESTAMP}.png
    - 2d_rect
        - {cam0/cam1}
            - {SEQUENCE}
                - 2d_rect_{cam0/cam1}_{SEQUENCE}_{FRAME}.jpg
    - 3d_bbox
        - os1
            - 3d_bbox_os1_{SEQUENCE}_{FRAME}.json
    - 3d_comp
        - os1
            - {SEQUENCE}
                - 3d_comp_os1_{SEQUENCE}_{FRAME}.bin
    - 3d_raw
        - os1
            - {SEQUENCE}
                - 3d_raw_os1_{SEQUENCE}_{FRAME}.bin
        - {cam2/cam3}
            - {SEQUENCE}
                - 3d_raw_{cam2/cam3}_{SEQUENCE}_{TIMESTAMP}.bin
    - 3d_semantic
        - os1
            - {SEQUENCE}
                - 3d_semantic_os1_{SEQUENCE}_{FRAME}.bin
    - calibrations
        - {SEQUENCE}
            - calib_cam0_intrinsics.yaml
            - calib_cam0_to_cam1.yaml
            - calib_cam1_intrinsics.yaml
            - calib_os1_to_base.yaml
            - calib_os1_to_cam0.yaml
            - calib_os1_to_cam1.yaml
            - calib_os1_to_vnav.yaml
    - metadata 
    - metadata_md
    - metadata_small
    - poses
        - dense
        - gps
        - gpsodom
        - imu
        - inekfodom
        - mag
    - timestamps
```

## Sensor Name to Sensor Hardware Abbreviations
```
os1   -> Ouster OS1-128 3D LiDAR (Synchronized with cam0 and cam1 @ 10Hz)
cam0  -> Left Teledyne FLIR Blackfly S RGB Camera  (Synchronized with OS1 @ 10Hz)
cam1  -> Right Teledyne FLIR Blackfly S RGB Camera (Synchronized with OS1@ 10Hz)
cam2  -> Microsoft Azure Kinect RGBD Camera (Unsynchronized @ 5 Hz)
cam3  -> Left Stereolabs ZED 2i Stereo Camera (Unsynchronized @ 5 Hz)
cam4  -> Right Stereolabs ZED 2i Stereo Camera (Unsynchronized @ 5 Hz)
vnav  -> Vectornav VN-310 Dual GNSS/INS (Unsynchronized @ 40 Hz)
```

## Calibration Files (With Examples)

**calib_cam0_to_cam1.yaml** - Transformation matrix from cam0 to cam1
```
extrinsic_matrix:
  R:
   rows: 3
   cols: 3
   data: [ 0.999607939093204,      -0.00728019495660303,        0.0270363988583955,
       0.00661202516157305,         0.999672512844786,         0.024721411485882,
       -0.0272075214802657,       -0.0245329538373475,         0.999328717165135 ]
  T: [ -0.197673440251558,         0.00128769601558891,         0.00253652872125049 ]

```

**calib_os1_to_{base/cam0/cam1/vnav}.yaml** - Transformation matrix from os1 to robot base, cam0, cam1, or vnav
```
extrinsic_matrix:
  rows: 4
  cols: 4
  data: [
    -0.0050614, -0.9999872, 0.0000000, 0.03,
  -0.1556502,  0.0007878, -0.9878119, -0.05,
   0.9877993, -0.0049997, -0.1556522, 0,
    0, 0, 0, 1 ]
```

**calib_cam{0/1}_intrinsics.yaml** - Camera intrinsics for cam0/cam1
```
image_width: 1224
image_height: 1024
camera_name: narrow_stereo/left
camera_matrix:
  rows: 3
  cols: 3
  data: [730.271578753826,                         0,           610.90462936767,
                         0,          729.707285068689,          537.715474717007,
                         0,                         0,                         1   ]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [-0.0559502131995934, 0.123761456061624, 0.00114530935813615, -0.00367111451580028, -0.0636070725936968]
rectification_matrix:
  rows: 3
  cols: 3
  data: [0.99977534231419884, -0.015206958507487951, 0.014765273904612077, 
    0.015024146363298435, 0.99981006328490463, 0.012414200751121740,
       -0.014951251672715080, -0.012189576169272843, 0.99981391984020351 ]
projection_matrix:
  rows: 3
  cols: 4
  data: [ 730.93414758424547, 0., 606.62505340576172, 0., 
    0., 730.93414758424547, 531.60715866088867, 0., 
    0., 0., 1., 0.     ]
```

## Poses

Each line will contain 8 numbers, the first number being the timestamp for the current measurement. The last 7 numbers will be a the x, y, z translation and quarternion qw, qx, qy, qz denoting the rigid body transformation to the origin for the current trajectory. We will release updated global poses using the center of all trajectories as the origin in a future dataset update. Poses in the `poses` directory are generated using LeGO-LOAM at a different frequency than the os1 measurements. Most users will want to use poses from the `poses/dense` directory there is a one to one correspondence between each line in this pose file to each os1 measurement. We ensure this by linearly interpolating between our known SLAM poses.

Example Pose File: (ts x y z qw qx qy qz)
```
1673884185.768105 -0.00037586 -0.00013268 0.00001081 0.99988589 0.00019951 0.01510509 -0.00010335
1673884191.568186 0.30363009 -0.01651052 -0.00002372 0.99133541 -0.00306328 0.01296573 -0.13067749
```

## Poses Subdirectories

`gps` - GPS data estimated by the Vectornav GPS/IMU (vnav). Because of poor GPS connectivity, we will only valid GPS measurements for sequence 3. GPS measurements will be in a .txt file in the following format:
```
ts latitude longitude altitude
```

`gpsodom` - Odometry estimated by the Vectornav GPS/IMU (vnav). Because of poor GPS connectivity, the x y z and linear acceleration values in the odometry from the vnav are all zeros. However, the other values are still valid.
```
ts x y z qw qx qy qz twist_lin_x twist_lin_y twist_lin_z twist_ang_x twist_ang_y twist_ang_z
```

`imu` - Inertial information consisting of linear acceleration (accx, accy, accz), angular velocity (angx, angy, angz), and orientation (qw, qx, qy, qz).
```
ts accx accy accz angx angy angz qw qx qy qz
```

`inekfodom` - Odometry estimated from [Husky Invariant Extended Kalman Filter (InEKF) Library](https://github.com/UMich-CURLY/husky_inekf) using IMU and wheel velocities. File format is identical to `gpsodom` files.
```
ts x y z qw qx qy qz twist_lin_x twist_lin_y twist_lin_z twist_ang_x twist_ang_y twist_ang_z
```

`mag` - Magnetometer data from Vectornav GPS/IMU (vnav).
```
ts magx magy magz
```

## Timestamps and Frames

`timestamps` - Timestamps for each synchronized frame for the os1, cam0, and cam1. The line index in the file is the frame number and the value on each line is the frame's timestamp. The os1, cam0, and cam1 sensors are hardware synchronized. At the start of each lidar sweep, the os1 electrically begins image capture for cam0 and cam1. We treat each revolution as a single frame and use the timestamp from the os1 as the timestamp for each frame. Note that because the os1 begins the lidar sweep from the back of the robot, there may exist small point cloud image misalignments in the sensor data. Each line in the 
```
ts0
...
tsn
```

## Metadata Format and Usage

We provide a metadata file for users to use for their train, validation, and test splits on the UT CODa benchmarks. Each metadata file contains a list of the 
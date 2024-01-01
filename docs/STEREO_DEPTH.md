# Pseudo Ground Truth Depth Maps

We release pseudo ground truth depth maps from the ZED 2I stereo camera for the following sequences in CODa: 0, 1, 10, 11, 12.
These depth maps are generated using the ZED 2I's built-in depth estimation algorithm. The depth maps are generated ~ 7.5 Hz and are saved as 16-bit PNG images. The depth maps are saved in the `3d_raw` subdirectory and corresponding stereo rgb images are stored in the `2d_raw/cam3` and `2d_raw/cam4` subdirectories. 

## Downloading RGBD Images

The RGBD images can be downloaded manually using the following link from [TACC](https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/depthonly). Assuming you have set the `CODA_ROOT_DIR` environment variable, you can download and extract the RGBD images for sequence 0 using the following commands:

```
wget https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/depthonly/0.zip
unzip 0.zip -d $CODA_ROOT_DIR/
```

Each zip file contains the RGBD images for a sequence. We recommend unzipping the files to the same directory as the rest of the dataset. Once unzipped, the RGBD images will follow the directory structure below:

```
CODa
|_ 3d_raw
    |_ cam3
        |_ {sequence}
            |_ 3d_raw_cam3_{sequence}_{frame}.png
            ...
|_ 2d_raw
    |_ cam3
        |_ {sequence}
            |_ 2d_raw_cam3_{sequence}_{frame}.png
            ...
    |_ cam4
        |_ {sequence}
            |_ 2d_raw_cam4_{sequence}_{frame}.png
            ...
```

To visualize the RGBD images alongside the LiDAR point clouds, you can run the following script to publish the RGBD images and LiDAR point clouds from sequence 0 to Rviz. It should look similar to the video shown below:

```
python scripts/check_stereo_pc.py --sequence 0
```

![Rviz RGBD](./depthcheck.gif)

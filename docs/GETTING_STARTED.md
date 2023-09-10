# Getting Started

Please follow these steps to familiarize yourself with CODa.

- Understand CODa's file structure and contents by reading the [data report](DATA_REPORT.md)
- Set up the [conda environment](../README.md)
- Read this document!

# Development Kit Features

The CODa development kit provides visualization tools and scripts to convert your dataset to the CODa format. 
We support the following features. Currently, we only provide the stereo RGB, LiDAR, IMU, and locally 
consistent ground truth poses. We plan to release the stereo depth and and globally consistent ground truth
poses for download in the future.

|       Feature                           |       Script        |       Purpose                             |
|       --------                          |       --------      |       --------                            |
| CODa Downloader                         |   download_split.py | Download Dataset                          |
| Annotation Visualizer (ROS)             | vis_annos_rviz.py   | View annotations on RVIZ (Recommended)    |
| Annotation Visualizer (Open3D)          | vis_annos_open3d.py | View annotations on Open3D                |
| Annotation Visualizer (On Image)        | vis_annos_rgb.py    | View annotations on 2D Images (Offline)   |
| Depth LiDAR Visualizer (ROS)            | check_stereo_pc.py  | View depth and LiDAR together             |
| Depth RGB Visualizer (On Image)         | check_stereo_rgb.py | View depth and RGB together               |
| ROSBag to CODa Converter (ROS)          | decode_multiday.py  | Converts ROSbags to CODa file format      |

# Downloading the Dataset 

You can download the dataset programmatically by sequence or split. We recommend downloading by sequence for 
visualization adn by split for benchmarks and experiments. Run the following script from the root directory
of the coda-devkit to download sequence 0 (17GB).

```
python scripts/download_split.py -d ./data -t sequence -se 0
```

For a detailed explanation, run the same script with the help flag.
```
python scripts/download_split.py -h

usage: download_split.py [-h] -d DOWNLOAD_PARENT_DIR -t TYPE [-sp SPLIT] [-se SEQUENCE]

optional arguments:
  -h, --help            show this help message and exit
  -d DOWNLOAD_PARENT_DIR, --download_parent_dir DOWNLOAD_PARENT_DIR
                        Parent directory to download CODa split to
  -t TYPE, --type TYPE  Download by sequence (recommended for visualization) or by split (recommended for experiments) Options: ['sequence', 'split']
  -sp SPLIT, --split SPLIT
                        CODa split to download. Only applies when type=split Options: ['tiny', 'small', 'medium', 'full']
  -se SEQUENCE, --sequence SEQUENCE
                        CODa sequence to download. Only applies when type=sequence Options: [0 - 22]
```

## Post Download Steps

After a successful download, the script will print out the environment variable to add. It is <b>ESSENTIAL</b> 
that you do this step, or the devkit will not function properly.

# Visualizing a Sequence on Rviz

![Sequence 0 Clip](CODaComp1000Trim.gif)

We recommend using Rviz for visualization. Rviz allows you to install the dataset on a remote server and visualize
it on a local machine. This is important because CODa is quite large (~1.5TB). We refer the read to the 
[ROS Multiple Machine Tutorial](http://wiki.ros.org/ROS/Tutorials/MultipleMachines) for how to set this up.

After setting up your machine, run the following command to see the visualization shown above.

```
python scripts/vis_annos_rviz.py -s 0 -f 0 -c classId
```

For a detailed explanation, run the same script with the help flag.
```
python scripts/vis_annos_rviz.py -h

usage: vis_annos_rviz.py [-h] [-s SEQUENCE] [-f START_FRAME] [-c COLOR_TYPE]

CODa rviz visualizer

optional arguments:
  -h, --help            show this help message and exit
  -s SEQUENCE, --sequence SEQUENCE
                        Sequence number (Default 0)
  -f START_FRAME, --start_frame START_FRAME
                        Frame to start at (Default 0)
  -c COLOR_TYPE, --color_type COLOR_TYPE
                        Color map to use for coloring boxes Options: [isOccluded, classId] (Default classId)
```

## All Other Visualizations

We provide command line help instructions for using the other visualization tools. By default, they assume
you would like to visualize sequence 0.

# Converting ROSBags to CODa Format

<p align="center">
  <img src="./bag_decoder.png" width="70%">
</p>

We recommend converting custom ROSBags to the CODa format to reuse the visualization scripts that we provide.

1. Check that the parent folder for `repository_root` matches the directory structure above.
2. Modify `config/bagdecoder.yaml` settings to match as follows:

- `repository_root` - This is the parent folder that contains all of your bag files. We expect this directory to contain subdirectories that contain your bag files. In the example below, you provide the path to the `bags` directory.
- `dataset_output_root` - This specifies the output directory that CODa will be generated in. This must be a valid directory regardless of whether or not you configure the tool to generate data.
- `bag_date` - This is the name of the folder containing the bag files that you want to convert. This does not have to be a date, but we will use dates as the subdirectory names in our provided data.
- `gen_data` - Boolean specifying whether or not to generate the CODa dataset. If you set this to `false`, the script will still publish processed topics over ROS.
- `sync_topics` - List of topics to synchronize. The script will synchronize all topics in this list and assign them to a synchronized frame number. The topics in this list must be present in the bag file.
- `sensor_topics` - List of sensor_topics to save to the CODa dataset. The topics in this list may be duplicated in the `sync_topics` list, but topics in the `sync_topics` list will still be synchronized regardless.
- `bags_to_process` - List of bag filenames to process. If this list is empty, the script will process all bag files in the `bag_date` directory.
- `bags_to_traj_ids` - List of trajectory IDs to assign to each bag file. Each entry in `bags_to_process` should match to the ID at the same position in the list as itself. If this list is empty, the script will assign trajectory IDs in the order that the bag files are processed beginning from 0 and increasing by one.

3. Run the bagdecoder script. We provide an example below:

<b>Decode a single bag using the config file </b>
```
python scripts/decode_multiday.py -c config/bagdecoder.yaml
```

<b>Decode all bags in `repository_root`</b>
```
python scripts/decode_multiday.py -c config/bagdecoder.yaml -a True
```

For a detailed explanation, run the same script with the help flag.
```
python scripts/decode_multiday.py -h

usage: decode_multiday.py [-h] [-c CONFIG] [-a ALL_DAYS]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        decode config file (see config/decode.yaml for example)
  -a ALL_DAYS, --all_days ALL_DAYS
                        decode all sudirs or the one specified in .yaml
```
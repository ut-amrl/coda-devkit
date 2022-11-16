# Campus Scale Object Perception Dataset

Planned Release: February 2023

## Helpful Scripts

### Decoding Raw Bag Files

We provide a tool that can be used to convert data within raw bag files into individual files for improved accessibility. For instance, the script will synchronize any number of topics provided, assign these topics to a synchronized frame number, and save each topic (LiDAR, camera, audio, inertial) to its own data file. Furthermore, the script will also republish specified topics in the bag file over ROS to work in conjunction with packages (such as LeGO-LOAM) that ingest topics over ROS. 

TODO: introduce what settings are available
To configure this tool for your use case, please be aware that you will need to set the following settings from the yaml file that you specify in the command line. By default, we include one called `config/decoder.yaml`. If you do not specify a yaml file at the command line, this file will be used.

`repository_root` - This is the parent folder that contains all of your bag files. We do not support searching recursively through subdirectories currently. 

`dataset_output_root` - This specifies the output directory that processed sensor data will be stored in. This must be a valid directory regardless of whether or not you configure the tool to generate data.

To use this tool, simply run the following command from this repository's root directory:

`python3 scripts/decode_single.py --config your_yaml_filepath_here` 

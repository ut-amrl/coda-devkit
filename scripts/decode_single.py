import os
import sys
import pdb
import numpy as np

# For imports
sys.path.append(os.getcwd())

#CustomImports
from helpers.bagdecoder import BagDecoder


def main():
    # User defined filepaths
    BAG_ROOT_DIR    = "/home/arthur/AMRL/Datasets/UTPeDa/101222"

    # bin_header = {
    #     "rate": FRAME_SCALE,
    #     "scene": SCENE_ENV
    # }

    # manifest_header = {
    #     "prefix": "s3://scand-trial1-sagemaker/artifacts/",
    #     "rate": FRAME_SCALE, # Measured in terms of Data Hz / rate Ex: Lidar @ 10Hz, rate=10 = Lidar @ 1Hz
    #     "scene": SCENE_ENV 
    # }

    # image_header = {
    #     "rate": FRAME_SCALE,
    #     "scene": SCENE_ENV
    # }

    bag_decoder = BagDecoder(BAG_ROOT_DIR)
    bag_decoder.convert_bag()
    

if __name__ == "__main__":
    main()
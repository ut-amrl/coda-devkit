import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import yaml
import numpy as np
import argparse
from helpers.bagdecoder import BagDecoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config/bagdecoder_lidarimuonly.yaml", help="decode config file (see config/decode.yaml for example)")
    args = parser.parse_args()
    bag_decoder = BagDecoder(args.config)
    bag_decoder.convert_bag()

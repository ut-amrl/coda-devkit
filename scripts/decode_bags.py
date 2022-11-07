import os
import sys
import pdb
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', default="config/decode.yaml",
                    help="decode config file (see config/decode.yaml for example)")

# For imports
sys.path.append(os.getcwd())

#CustomImports
from helpers.bagdecoder import BagDecoder


def main(args):
    bag_decoder = BagDecoder(args.config)
    bag_decoder.convert_bag()
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
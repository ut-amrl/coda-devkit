import os
import sys
import pdb
import numpy as np

# For imports
sys.path.append(os.getcwd())

#CustomImports
from helpers.bagdecoder import BagDecoder


def main():
    bag_decoder = BagDecoder()
    bag_decoder.convert_bag()
    

if __name__ == "__main__":
    main()
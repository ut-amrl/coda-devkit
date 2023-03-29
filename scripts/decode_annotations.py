import os
import sys
import pdb
import numpy as np

# For imports
sys.path.append(os.getcwd())

#CustomImports
from helpers.annotationdecoder import AnnotationDecoder


def main():
    annot_decoder = AnnotationDecoder()
    annot_decoder.decode_annotations()

if __name__ == "__main__":
    main()
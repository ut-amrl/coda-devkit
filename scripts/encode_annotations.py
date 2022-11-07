import os
import sys
import pdb
import numpy as np

# For imports
sys.path.append(os.getcwd())

#CustomImports
from helpers.annotationencoder import AnnotationEncoder


def main():
    manifest_gen = AnnotationEncoder()
    manifest_gen.encode_annotations()
    

if __name__ == "__main__":
    main()
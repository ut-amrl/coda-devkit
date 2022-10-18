import os
import sys
import pdb
import numpy as np

# For imports
sys.path.append(os.getcwd())

#CustomImports
from helpers.manifest import ManifestGenerator


def main():
    manifest_gen = ManifestGenerator()
    manifest_gen.create_manifest()
    

if __name__ == "__main__":
    main()
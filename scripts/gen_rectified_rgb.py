import os
from os.path import join

import yaml
import argparse
import tqdm
from multiprocessing import Pool

import cv2
import numpy as np

from helpers.calibration import load_cam_calib_from_path
from helpers.sensors import set_filename_by_prefix, get_filename_info
from helpers.geometry import rectify_image
from helpers.constants import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='.', help="Output root directory for rectified images")
    parser.add_argument('--seq', type=int, default=None, help="Sequence to rectify, default is all")
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    return args

def rectify_sequence(args):
    """
    Rectifies all images for a single sequence
    """
    indir, outdir, seq, cam_calibration = args
    cams = ['cam0', 'cam1']
    for cam in cams:
        raw_dir = join(indir, TWOD_RAW_DIR, cam)
        rect_dir = join(outdir, TWOD_RECT_DIR, cam)
        if not os.path.exists(rect_dir):
            print("Creating %s rectified dir at %s " % (cam, rect_dir))
            os.makedirs(rect_dir, exist_ok=True)

        raw_img_dir = join(raw_dir, str(seq))
        raw_img_dir_files = os.listdir(raw_img_dir)
        raw_img_dir_files = sorted(raw_img_dir_files, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        
        rect_img_dir = join(rect_dir, str(seq))
        if not os.path.exists(rect_img_dir):
            os.makedirs(rect_img_dir, exist_ok=True)

        for img_file in raw_img_dir_files:
            _, _, _, frame = get_filename_info(img_file)
            img_path = os.path.join(raw_img_dir, img_file)
            rect_img_file   = set_filename_by_prefix(TWOD_RECT_DIR, cam, "png", seq, frame)

            if os.path.isfile(img_path):
                intrinsics = cam_calibration['%s_intrinsics'%cam]

                rectified_img_np = rectify_image(img_path, intrinsics)
                rect_img_path = join(rect_img_dir, rect_img_file)
                success = cv2.imwrite(rect_img_path, rectified_img_np)
                if not success:
                    print("Error writing rectified image to %s " % (rect_img_path))
    print(f'Finished rectifying sequence {seq} to {rect_img_dir}')

def rectify_sequences(rootdir, outdir, num_workers=1, seq=None):
    cam0_dir = os.path.join(rootdir, TWOD_RAW_DIR, 'cam0')
    assert os.path.isdir(cam0_dir), "Camera 0 directory not found at %s " % cam0_dir
    if seq is None:
        sequences = sorted([int(seq) for seq in next(os.walk(cam0_dir))[1]])
    else:
        sequences = [seq]

    indirs = [rootdir]*len(sequences)
    outdirs = [outdir]*len(sequences)
    cam_calibrations = []
    for seq in sequences:
        #Load calibrations if they exist
        calibrations_path = os.path.join(rootdir, CALIBRATION_DIR, str(seq))
        if os.path.exists(calibrations_path):
            print("Calibrations directory exists %s, loading calibrations" % calibrations_path)
            cam_calibrations.append(load_cam_calib_from_path(calibrations_path))

        assert cam_calibrations!=None, "No camera calibrations found for seq %i" % seq

    pool = Pool(processes=num_workers)
    for _ in tqdm.tqdm(pool.imap_unordered(
        rectify_sequence, zip(indirs, outdirs, sequences, cam_calibrations)), total=len(sequences)
    ):
        pass

def main(args):
    #0 Setup input and output directories
    rootdir = os.getenv(ENV_CODA_ROOT_DIR)
    seq = args.seq
    num_workers = args.workers
    outdir = args.outdir
    if not os.path.exists(outdir):
        print(f'Creating output directory {outdir}')
        os.makedirs(outdir)

    #1 Build list of frames to rectify
    rectify_sequences(rootdir, outdir, num_workers, seq=seq)
        

if __name__ == "__main__":
    args = parse_args()
    main(args)
import fractions
import os
import pdb
import yaml

from helpers.constants import *

class ManifestGenerator(object):
    """
    """
    def __init__(self):
        self._settings_fp = os.path.join(os.getcwd(), "config/manifest.yaml")

        with open(self._settings_fp, 'r') as settings_file:
            settings = yaml.safe_load(settings_file)

            self._trajs         = settings['trajectories']
            self._traj_frames   = settings['trajectory_frames']
            self._ds_rate       = settings['downsample_rate']
            self._indir         = settings['dataset_input_root']
            self._outdir        = settings['manifest_output_root']

            # Sagemaker Specific
            self._prefix        = settings['s3_prefix']

        #Directory checks
        assert os.path.isdir(self._indir), '%s does not exist' % self._indir
        if not os.path.exists(self._outdir):
            print("Output directory does not exist, creating at %s " % self._outdir)
            os.mkdir(self._outdir)

    def create_manifest(self):
        """
        """
        prefix_text = PREFIX_TEXT % self._prefix

        for traj in self._trajs:
            traj_frames = self._traj_frames[traj]

            for frame_seq in traj_frames:
                start, end = frame_seq[0], frame_seq[1]

                self.load_frames(traj, start, end)
                pdb.set_trace()

    def load_frames(self, traj, start, end):
        # Load pose estimate

        # Load point cloud 

        # Load stereo camera images
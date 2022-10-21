import pdb
import numpy as np

from scipy.spatial.transform import Rotation as R

def inter_pose(posea, poseb, sensor_ts):
    tsa = posea[0]
    tsb = poseb[0]
    quata = posea[4:]
    quatb = poseb[4:]
    transa  = posea[1:4]
    transb  = poseb[1:4]

    inter_trans = transa + (transb-transa) / (tsb - tsa) * 0.5
    theta = np.arccos(np.dot(quata, quatb))
    inter_quat  = np.sin(0.5*theta)/np.sin(theta)*quata +\
        np.sin(0.5*theta) / np.sin(theta) * quatb

    new_pose = np.concatenate((sensor_ts, inter_trans, inter_quat), axis=None)

    return new_pose

def wcs_mat(angles):
    """
    assumes angles order is zyx degrees
    """
    r = R.from_euler('zyx', angles, degrees=True)
    return r

def oxts_to_homo(pose):
    trans = pose[1:4]
    quat = np.array([pose[7], pose[4], pose[5], pose[6]])
    rot_mat = R.from_quat(quat).as_matrix()
    
    temp_r = wcs_mat([0, 180, 0]).as_matrix()
    # pdb.set_trace()

    homo_mat = np.eye(4, dtype=np.float32)
    homo_mat[:3, :3] = temp_r@rot_mat
    homo_mat[:3, 3] = trans
    return homo_mat
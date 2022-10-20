import pdb
import numpy as np

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
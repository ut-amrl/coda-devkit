import numpy as np
import os
import sys

sys.path.append(os.getcwd())
from scripts.pose_to_hitl import get_normal 

GDC  = [0,1,3,4,5,18,19]
GUAD = [2,7,12,16,17,21]
WCP  = [6,9,10,11,13,20,22]
UNB  = [8,14,15]

#save_dir = os.path.join("./", "HitL")
n = None
np_bin = np.array([])

trajs = [14,15]

print("\nPutting two txt files together.")
for traj in trajs:
    dir = "/home/arnavbagad/coda-devkit/UNB/"
    print(traj)
    fpath = os.path.join(dir, f"{traj}.txt")
    np_txt = np.loadtxt(fpath, comments=None, delimiter=' ',  skiprows=2)
    if len(np_bin) == 0:
        np_bin = np_txt
    else:
        np_bin = np.vstack((np_bin, np_txt))
    if n == None:
        n = len(np_bin)

print("Compute Normal.")
np_bin[:, 5:7] = get_normal(np_bin[:, 3:6])

print("Save txt.")
fpath_out = os.path.join("/home/arnavbagad/coda-devkit/Corrected_UNB/" + f"_{trajs[0]}_{trajs[1]}.txt")
header = 'StarterMap\n1455656519.379815'
np.savetxt(fpath_out, np_bin, delimiter=',', header=header, comments='')

# fpath_out = os.path.join(save_dir, "0" + ".txt")
# header = 'StarterMap\n1455656519.379815'
# np.savetxt(fpath_out, np_bin[:n], delimiter=',', header=header, comments='')

# fpath_out = os.path.join(save_dir, "1" + ".txt")
# header = 'StarterMap\n1455656519.379815'
# np.savetxt(fpath_out, np_bin[n:], delimiter=',', header=header, comments='')

print("Global Map done.")
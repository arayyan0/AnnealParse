import glob as glob
from spin_lib import AnnealedSpinConfiguration
from common import pi
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools as it
from shutil import copyfile
import sys

# generate filenames
versions = 7
# which = '18_p0.4_dense'
which='denser24/618'
run = '8'

a = 0.195
parray = np.linspace(0.15,0.17,20+1)
print(parray)

for p in parray:
    version_Elist = []

    for version in range(1,versions+1):
        file = f'out/{which}/jobrun_{run}/v_{version}/p_{p:.3f}_a_{a:.3f}_.out'
        spinstuff = AnnealedSpinConfiguration(file)
        version_Elist.append(spinstuff.MCEnergyDensity)
    which_version = version_Elist.index(min(version_Elist))+1
    print(f"{p:.3f}", which_version, max(version_Elist)- min(version_Elist), [x - min(version_Elist) for x in version_Elist])
    # print(f"{p:.3f}", which_version, min(version_Elist))

    if not os.path.exists(f"out/{which}/jobrun_{run}/v_0"):
        os.makedirs(f"out/{which}/jobrun_{run}/v_0")

    copyfile(f"out/{which}/jobrun_{run}/v_{which_version}/p_{p:.3f}_a_{a:.3f}_.out",
             f"out/{which}/jobrun_{run}/v_0/p_{p:.3f}_a_{a:.3f}_.out")

# p = 0.4
# aarray = np.linspace(0.150,0.175,25+1)
# print(aarray)
#
# for a in aarray:
#     version_Elist = []
#
#     for version in range(1,versions+1):
#         file = f'out/{which}/jobrun_{run}/v_{version}/p_{p:.3f}_a_{a:.3f}_.out'
#         spinstuff = AnnealedSpinConfiguration(file)
#         version_Elist.append(spinstuff.MCEnergyDensity)
#     which_version = version_Elist.index(min(version_Elist))+1
#     print(f"{a:.3f}", which_version, max(version_Elist)- min(version_Elist), [x - min(version_Elist) for x in version_Elist])
#     # print(f"{a:.3f}", which_version, min(version_Elist))
#
#     if not os.path.exists(f"out/{which}/jobrun_{run}/v_0"):
#         os.makedirs(f"out/{which}/jobrun_{run}/v_0")
#
#     copyfile(f"out/{which}/jobrun_{run}/v_{which_version}/p_{p:.3f}_a_{a:.3f}_.out",
#              f"out/{which}/jobrun_{run}/v_0/p_{p:.3f}_a_{a:.3f}_.out")

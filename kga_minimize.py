import glob as glob
from spin_lib import AnnealedSpinConfiguration
from common import pi
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools as it
from shutil import copyfile

# generate filenames
run = 5
versions = 5

a = 0.00
parray = np.linspace(0.01,0.10,9+1)
# parray = np.linspace(0.07,0.24,17+1)
# parray = np.linspace(0.22,0.49,27+1)
print(parray)

for p in parray:
    version_Elist = []

    for version in range(1,versions+1):
        file = f'out/jobrun_{run}/v_{version}/p_{p:.3f}_a_{a:.3f}_.out'
        spinstuff = AnnealedSpinConfiguration(file)
        version_Elist.append(spinstuff.MCEnergyDensity)
    which_version = version_Elist.index(min(version_Elist))+1
    print(f"{p:.3f}", which_version, max(version_Elist)- min(version_Elist), [x - min(version_Elist) for x in version_Elist])

    if not os.path.exists(f"out/jobrun_{run}/v_0"):
        os.makedirs(f"out/jobrun_{run}/v_0")

    copyfile(f"out/jobrun_{run}/v_{which_version}/p_{p:.3f}_a_{a:.3f}_.out",
             f"out/jobrun_{run}/v_0/p_{p:.3f}_a_{a:.3f}_.out")

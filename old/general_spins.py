#file   general_spins.py
#author Ahmed Rayyan
#date   August 30, 2020
#brief  spin analysis of general spin_configuration

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from lib_new_parse import LocalRotation, WhichUnitCell, SpinConfiguration
import os
import sys

import warnings
plt.rcParams['figure.facecolor'] = 'black'
warnings.filterwarnings('ignore')

# plt.style.use('dark_background')
xyz_to_abc = LocalRotation(np.array([1,1,1])/sqrt(3))

type = 1
file = f"outmy/H_spin_wave/1.out"
# file = '/Users/ahmed/Documents/University/PhD/Research/KSL_classical/cmc/code/simulating/sim/another_smectic.out'
# file = 'out/zz_rh2_0.013_0.000_1.out'
print(file)
assert(os.path.exists(file)),"Your data directory does not exist."
###############################################################################
with open(file, 'r') as f:
    file_data = f.readlines()

hc_or_kek = int(file_data[2])
l1, l2, s = [int(x) for x in file_data[4].split()]
sites = l1*l2*s

T1, T2, sublattice_vectors, _ = WhichUnitCell(hc_or_kek, type, s)

flat_spin_loc, flat_spin_config = np.empty((sites, 2)), np.empty((sites, 3))
for i, line in enumerate(file_data[32:]):
    x, y, sub, Sx, Sy, Sz = line.split()
    x, y, sub = list(map(int, [x,y,sub]))
    rot_spin = xyz_to_abc.dot(np.array([float(Sx), float(Sy), float(Sz)]))
    flat_spin_loc[i, :] = x*T1 + y*T2 + sublattice_vectors[sub]
    flat_spin_config[i, :] = rot_spin

spinstuff = SpinConfiguration(flat_spin_loc, flat_spin_config, [hc_or_kek, s, l1, l2, type])

fig = spinstuff.PlotSSF()
# fig.show()
fig.axes[0].set_facecolor('black')
# plt.savefig(f"outmy/H_spin_wave/ssf_1.pdf")
plt.close()

fig = spinstuff.PlotSpins()
# fig.show()
# plt.savefig(f"outmy/H_spin_wave/spins_1s.pdf")
plt.close()

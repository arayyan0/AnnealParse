#file: spin_waves.py
#author: Ahmed Rayyan
#date: April 23, 2020
#brief: calculates spin wave given a spin configuration

from lib_new_parse import WhichOrder, ReciprocalLattice, LSWT, FindReciprocalVectors, IndexToPosition, EquivalentBravais, LocalRotation
import numpy as np
from math import isclose
import matplotlib.pyplot as plt

# -----------------------load up the spin configuration-------------------------#
order = "zz"

p = 0.04
file = "out/%s_%.2f_.out"%(order,p)
print(file)
with open(file, 'r') as f:
    file_data = f.readlines()

#extract Hamiltonian parameters
Kx, Ky, Kz = [float(x) for x in file_data[15].split()]
Gx, Gy, Gz = [float(x) for x in file_data[17].split()]
Gp = float(file_data[19].split()[0])
J1 = float(file_data[21].split()[0])
#
# extract cluster information
l1, l2, s = [int(x) for x in file_data[4].split()]
sites = l1*l2*s

# extract spin information
spins = np.empty((sites, 3), dtype=float)
for i, line in enumerate(file_data[32:]):
    n1, n2, sub, Sx, Sy, Sz = line.split()
    spins[i, :] = np.array([float(Sx), float(Sy), float(Sz)])
print(spins)
#
#create LSWT object
lswtea = LSWT([[Kx,Ky,Kz], [Gx, Gy, Gz], Gp, J1],
              order,
              spins)
#
print(lswtea.ClusterEnergy/sites)
# #-------------------------------create kspace----------------------------------#
lattice_characteristics = ReciprocalLattice()
# kp=np.array([0,0])
# kp = lattice_characteristics.MakeKPath(["G","M1","Gp1","Gp2","M2","G"],100)
kp = lattice_characteristics.MakeKPath(["G","M1","K","G"],200)
# print(kp[0][0])
# #---------------------fill the second quantized Hamiltonian--------------------#
lswtea.ObtainMagnonSpectrumAndDiagonalizer(kp[0])
# lswtea.ObtainMagnonSpectrumAndDiagonalizer([np.array([0,0])])
fig = lswtea.PlotMagnonDispersions(*kp)
fig.suptitle(r"$\phi/\pi = %.2f$, %s order"%(p, order))
# fig.axes[0].set_ylim(0,0.16)
fig.tight_layout(rect=[0,0.03,1,0.95])
fig.show()
plt.close()

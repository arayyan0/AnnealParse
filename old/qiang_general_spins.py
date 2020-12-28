#file   qiang_general_spins.py
#author Ahmed Rayyan
#date   October 29, 2020
#brief  spin analysis of general spin_configuration in Qiang's notation

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import matplotlib.colors as clr
from math import sqrt
from lib_new_parse import LocalRotation, FindReciprocalVectors, NewMakeGrid, pi, my_cmap
import os
import sys

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times New Roman']})
rc('text', usetex=False)

import warnings
warnings.filterwarnings('ignore')

a1 = np.array([+1/2,sqrt(3)/2])
a2 = np.array([-1/2,sqrt(3)/2])

dz = (a1+a2)/3

# plt.style.use('dark_background')
xyz_to_abc = LocalRotation(np.array([1,1,1])/sqrt(3))

###############################################################################
Lx_list=[36,54,72,90]
Lx = Lx_list[3]
Ly = 8
N_sites = Lx*Ly
file = f"/Users/ahmed/Downloads/ZZ-Smectic-Plot-Ly08/avgSpin_Phi02250_g0.80_Ly08Lx{Lx}.out"
print(file)
# assert(os.path.exists(file)),"Your data directory does not exist."
###############################################################################

with open(file, 'r') as f:
    file_data = f.readlines()

spin_config = []
spin_loc = []

for line in file_data[1:]:
    n, sx, sy, sz = line.split()

    n = int(n)
    j = n%(Ly)
    i = int((n - j)/Ly) #Qiang's notation: Ly*i + j = n
    bravais_vec = int(j/2)*(a1+a2) + int(i/2)*(a1-a2)

    jp = n%(2*Ly)

    if ((jp >= 0) and (jp <= Ly-1)):
        if (n%2 == 0): #A site index
            location = bravais_vec

        elif (n%2 == 1): #B site index
            location = bravais_vec + dz

    elif ((jp >= Ly) and (jp <= 2*Ly-1)):
        if (n%2 == 0): #B site index
            location = dz-a2 + bravais_vec
        elif (n%2 == 1): #A site index
            location = dz-a2+2*dz + bravais_vec

    spin_loc.append(location)
    spin = xyz_to_abc.dot(np.array(list(map(float, [sx,sy,sz]))))
    spin_config.append(spin)

spin_config = np.array(spin_config)
spin_loc = np.array(spin_loc)
###############################################################################
B1, B2 = FindReciprocalVectors(a1-a2, a1+a2)
# may be modified later...i dont need meshgrid
KX, KY, gggg = NewMakeGrid(B1, B2, int(Lx/2), int(Ly/2), 2)
kx, ky = [np.reshape(x, -1) for x in [KX, KY]]
k = np.stack((kx, ky)).T

SdotS_mat = np.einsum("ij,kj", spin_config, spin_config)
s_k = np.empty(len(k))
for i, kv in enumerate(k):
    phase_i = np.exp(1j * np.einsum('i,ji', kv, spin_loc))
    phase_j = np.exp(-1j * np.einsum('i,ji', kv, spin_loc))
    phase_mat = np.einsum('i,j->ij', phase_i, phase_j)
    s_k[i] = np.abs((SdotS_mat * phase_mat).sum())/N_sites/N_sites

s_k = np.reshape(s_k, KX.shape)

scale = 2*pi

fig, ax = plt.subplots()
ax.scatter(KX/scale, KY/scale, marker="+", color='gray', s=2)
c = ax.scatter(KX/scale, KY/scale, c=s_k, cmap='afmhot', edgecolors="none", s=30,alpha=0.5)
cbar = fig.colorbar(c, fraction=0.05)
cbar.set_label(r'$\frac{s_\vec{k}}{N}$',labelpad=14, rotation=0, usetex=False,fontsize=15)
ax.axis("equal")

# ax.axis("off")
ax.set_facecolor('black')
#
b1, b2 = FindReciprocalVectors(a1, a2)
bz2 = ptch.RegularPolygon(
    (0, 0), 6, np.linalg.norm(
        (2 * b1 + b2) / 3)/scale, pi / 6, fill=False,color='white')
bz3 = ptch.RegularPolygon((0, 0), 6, np.linalg.norm(b1)/scale, 0, fill=False,\
                          color='white')
ax.add_patch(bz2)
ax.add_patch(bz3)
ax.set_xlim(-6.5/scale, 6.5/scale)
ax.set_ylim(-7.5/scale, 7.5/scale)
ax.set_xlabel(r'$k_x/2\pi$', fontsize=10, usetex=True)
ax.set_ylabel(r'$k_y/2\pi\qquad$',rotation=0,fontsize=10, usetex=True)

plt.savefig("smectic_SSF.pdf")
plt.show()
plt.close()
###############################################################################
theta = np.arccos(np.clip(spin_config[:, 2], -1, 1)) / np.pi * 180
norm = clr.Normalize(vmin=0,vmax=180)

cm = my_cmap

fig, ax = plt.subplots()
ax.quiver(spin_loc[:,0], spin_loc[:,1], spin_config[:,0], spin_config[:,1], theta,
          cmap=cm,
          norm=norm,
          scale=50,
          minlength=1,
          headwidth=5,
          pivot='mid')
ax.axis("off")
ax.axis("equal")
ax.set_facecolor('whitesmoke')

sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
cb = plt.colorbar(
    sm,
    fraction=0.008,
    pad=0,
    orientation='vertical')
cb.set_ticks([0,90,180])
cb.ax.set_yticklabels(['0$^\circ$', '90$^\circ$', '180$^\circ$'])

cb.set_label(r'$\theta_{\mathbf{c}^*}$', labelpad=1, rotation=0, usetex=True)

for x in range(-1,int(Lx/2)):
    for y in range(-1,int(Ly/2)):
        center1 = x * (a1-a2) + y * (a1+a2)
        if (y!=-1):
            hex1 = ptch.RegularPolygon(
                center1+a1-dz, 6, 1 / sqrt(3), 0, fill=False, linewidth=0.05,color='gray')
            ax.add_patch(hex1)
        if (x!=int(Lx/2)):
            hex2 = ptch.RegularPolygon(
            center1 + a1+a1-dz, 6, 1 / sqrt(3), 0, fill=False, linewidth=0.05,color='gray')
            ax.add_patch(hex2)

# plt.savefig("smectic_spin.pdf")
plt.show()
plt.close()

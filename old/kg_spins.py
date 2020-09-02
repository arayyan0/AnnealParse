#file   kg_parse_energy.py
#author Ahmed Rayyan
#date   January 14 2020
#brief  parses through data for analysis visualization

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from math import sqrt, cos
import sys

from lib_parse import pi, FindReciprocalVectors, MakeGrid, XYZtoABC, WhichUnitCell

project = "kg"
gk = 0.500
gg = 0.500
ak = 0.000
ag = 0.000
l = 6
version = 5

parameter_dir = "/gk_%.3f_gg_%.3f_ak_%.3f_ag_%.3f/l_%i/v_%i"%(gk, gg, ak, ag, l, version)
data_dir = "../../raw_data/" + project + parameter_dir
print(data_dir)
assert(os.path.exists(data_dir)),"Your data_dir does not exist. Go make it or point to its proper location."

file_list = glob.glob(data_dir+"/*")
file_list.sort()

for file in file_list:
    p = float(file.split("_")[-2])
    print(p)

    with open(file, 'r') as f:
        file_data = f.readlines()

    sublattice = int(file_data[2])
    sites = int(float(file_data[4]))
    length = int(sqrt(sites/sublattice))
    a1, a2, sublattice_vectors = WhichUnitCell(sublattice)

    flat_spin_loc, flat_spin_config = np.empty((sites, 2)), np.empty((sites, 3))
    for i, line in enumerate(file_data[32:]):
        x, y, sub, vx, vy, vz = line.split()
        x, y, sub = list(map(int, [x,y,sub]))
        rot_spin = XYZtoABC(np.array([float(vx), float(vy), float(vz)]))
        flat_spin_loc[i, :] = x*a1 + y*a2 + sublattice_vectors[sub]
        flat_spin_config[i, :] = rot_spin
    #--------------------calculating the static structure factor--------------------
    dot_mat = np.einsum("ij,kj", flat_spin_config, flat_spin_config)

    b1, b2 = FindReciprocalVectors(a1, a2);
    KX, KY, gggg = MakeGrid(b1, b2,12) #may be modified later...i dont need meshgrid
    kx, ky = [np.reshape(x, -1) for x in [KX, KY]]
    k = np.stack((kx,ky)).T

    bz1 = ptch.RegularPolygon((0,0), 6, np.linalg.norm(2*b1+b2)/3, pi/6, fill = False)
    bz2 = ptch.RegularPolygon((0,0), 6, np.linalg.norm(b1+b2), 0, fill = False)
    # gggg.axes[0].add_patch(bz1)
    # gggg.axes[0].add_patch(bz2)
    # gggg.axes[0].set_xlim(-6.5, 6.5)
    # gggg.axes[0].set_ylim(-8, 8)
    # gggg.show()

    s_k = np.empty(len(k))
    for i, kv in enumerate(k):
        phase_i = np.exp(1j * np.einsum('i,ji', kv, flat_spin_loc))
        phase_j = np.exp(-1j * np.einsum('i,ji', kv, flat_spin_loc))
        phase_mat = np.einsum('i,j->ij', phase_i, phase_j)
        s_k[i] = (dot_mat * phase_mat).sum()/sites
    s_k = np.reshape(s_k, KX.shape)

    fig, ax = plt.subplots()
    c = ax.pcolormesh(KX, KY, s_k, cmap='viridis')
    ax.axis("equal")
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('$s_k$', labelpad=10)
    ax.axis("off")

    fig.axes[0].add_patch(bz1)
    fig.axes[0].add_patch(bz2)
    fig.axes[0].set_xlim(-6.5, 6.5)
    fig.suptitle("$(\phi/\pi,\, g_K,\, g_\Gamma,\, a_K,\, a_\Gamma) = (%.3f, %.2f, %.2f, %.2f, %.2f)$"%(p, gk, gg, ak, ag))
    plot_dir = "../../plot_data/" + project + parameter_dir
    fig.savefig(plot_dir + "/ssf_p_%.3f-%i.pdf"%(p,version))
    # fig.show()
    # plt.close()
#-----------------------------visualizing the spins-----------------------------
    oneD = np.array(range(0, length))
    n1, n2 = np.meshgrid(oneD,oneD)

    RX_list = np.empty((sublattice), dtype=np.ndarray)
    RY_list = np.empty((sublattice), dtype=np.ndarray)

    for i in range(sublattice):
        RX_list[i] = sublattice_vectors[i,0]+ n1*a1[0] + n2*a2[0]
        RY_list[i] = sublattice_vectors[i,1]+ n1*a1[1] + n2*a2[1]

    mat_spin_config = np.reshape(flat_spin_config, (length, length, sublattice,3))

    color = np.array(["r","k"])
    fig, ax = plt.subplots()
    for i in range(sublattice):
        ax.quiver(RX_list[i], RY_list[i], mat_spin_config[:,:,i,0], mat_spin_config[:,:,i,1], color=color[i], scale=45,headwidth=5,headlength=8)

    for x in range(length):
        for y in range(length):
            center1 = a1 + x*a1 + y*a2
            hex1 = ptch.RegularPolygon(center1, 6, 1/sqrt(3), 0, fill = False, linewidth=0.2)
            ax.add_patch(hex1)
    ax.axis("off")

    ax.plot([0, a1[0], 0, a2[0]], [0, a1[1], 0, a2[1]])
    ax.plot([a1[0],a1[0]+a2[0],a2[0],a1[0]+a2[0] ], [a1[1],a1[1]+a2[1], a2[1],a1[1]+a2[1]])
    #
    ax.set(xlim=(-5, 5), ylim=(0, 10))
    ax.set_aspect('equal')
    fig.suptitle("$(\phi/\pi,\, g_K,\, g_\Gamma,\, a_K,\, a_\Gamma) = (%.3f, %.2f, %.2f, %.2f, %.2f)$"%(p, gk, gg, ak, ag))
    fig.savefig(plot_dir + "/spins_p_%.3f-%i.pdf"%(p,version))
    # fig.show()
    # plt.close()

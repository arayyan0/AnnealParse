#file   which_order_spins.py
#author Ahmed Rayyan
#date   February 24, 2020
#brief  calculates SSF/spin configuration for which_order_g_dom

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import glob
from math import sqrt
from lib_parse import XYZtoABC, FindReciprocalVectors, pi, NewMakeGrid
from lib_which_order import WhichUnitCell
import warnings
warnings.filterwarnings('ignore')

# G = 0*b1 + 0*b2
# M1 = 0.5*b1 + 0*b2
# M2 = 0.5*b1 + 0.5*b2
# M3 = 0*b1 + 0.5*b2
# K = 2*b1/3 + b2/3
# Kp = b1/3 + 2*b2/3
# M231 = 2/3*M1
# M232 = 2/3*M2
# M233 = 2/3*M3
# K2 = K/2
# Kp2 = Kp/2
# K34 = 3*K/4
# Kp34 = 3*Kp/4
#
# sympoint = np.reshape(np.concatenate((G,M1,M2,M3,K,Kp,M231,M232,M233,K2,Kp2,K34,Kp34)),(13,2))
orders = [4,6,12,18,30,50]
#------------------------Accessing spins in the file----------------------------
project = "which_order_g_dom"
g = 0.5
for order in orders:
    # for version in range(1,10+1):
    version = 0
    parameter_dir = "/g_%.3f/o_%i/v_%i"%(g, order, version)
    data_dir = "../../raw_data/" + project + parameter_dir
    assert(os.path.exists(data_dir)),"Your raw data directory does not exist. Go make it or point to its proper location."

    file_list = glob.glob(data_dir + "/*")
    file_list.sort()

    for file in file_list:
        #----------------------Read the spins in from file--------------------------
        p = float(file.split("_")[-4])
        a = float(file.split("_")[-2])

        with open(file, 'r') as f:
            file_data = f.readlines()

        type = int(file_data[2])
        l1, l2, s = [int(x) for x in file_data[4].split()]
        sites = l1*l2*s

        T1, T2, sublattice_vectors, color = WhichUnitCell(type, s)

        flat_spin_loc, flat_spin_config = np.empty((sites, 2)), np.empty((sites, 3))
        for i, line in enumerate(file_data[32:]):
            x, y, sub, Sx, Sy, Sz = line.split()
            x, y, sub = list(map(int, [x,y,sub]))
            rot_spin = XYZtoABC(np.array([float(Sx), float(Sy), float(Sz)]))
            flat_spin_loc[i, :] = x*T1 + y*T2 + sublattice_vectors[sub]
            flat_spin_config[i, :] = rot_spin

        #--------------------------Creating my k-space grid-----------------------------
        a1, a2 = np.array([1/2, sqrt(3)/2]), np.array([-1/2, sqrt(3)/2])
        b1, b2 = FindReciprocalVectors(a1, a2)

        KX, KY, gggg = NewMakeGrid(b1, b2, 12, 12) #may be modified later...i dont need meshgrid
        kx, ky = [np.reshape(x, -1) for x in [KX, KY]]
        k = np.stack((kx,ky)).T

        #keep vectors only within second brillouin zone
        # klist = list(k)
        # for i, kv in enumerate(klist):
        #     if np.linalg.norm(kv) > np.linalg.norm(b1):
        #         klist.pop(i)
        # k = np.array(klist)

        #------------------------------Calculate SSF--------------------------------
        dot_mat = np.einsum("ij,kj", flat_spin_config, flat_spin_config)

        s_k = np.empty(len(k))
        for i, kv in enumerate(k):
            phase_i = np.exp(1j * np.einsum('i,ji', kv, flat_spin_loc))
            phase_j = np.exp(-1j * np.einsum('i,ji', kv, flat_spin_loc))
            phase_mat = np.einsum('i,j->ij', phase_i, phase_j)
            s_k[i] = (dot_mat * phase_mat).sum()/sites

        #------------------------------Plot SSF--------------------------------
        s_k = np.reshape(s_k, KX.shape)
        fig, ax = plt.subplots()
        c = ax.pcolormesh(KX, KY, s_k, cmap='viridis')#, edgecolors="face")
        ax.axis("equal")
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('$s_k$', labelpad=10)
        # ax.axis("off")

        bz2 = ptch.RegularPolygon((0,0), 6, np.linalg.norm((2*b1+b2)/3), pi/6, fill = False)
        bz3 = ptch.RegularPolygon((0,0), 6, np.linalg.norm(b1), 0, fill = False)
        bz4 = ptch.RegularPolygon((0,0), 60, np.linalg.norm(b1), 0, fill = False)
        fig.axes[0].add_patch(bz2)
        fig.axes[0].add_patch(bz3)
        fig.axes[0].add_patch(bz4)
        fig.axes[0].set_xlim(-6.5, 6.5)
        fig.axes[0].set_ylim(-7.5, 7.5)
        # fig.axes[0].scatter(sympoint.T[0],sympoint.T[1],color="w",s=0.8)

        fig.suptitle(r"($\phi/\pi,\, g, a, o, v)= (%.3f, %.3f, %.3f, %i, %i)$"%(p,g,a,order,version))
        plot_dir = "../../plot_data/" + project + parameter_dir
        assert(os.path.exists(data_dir)),"Your plotting directory does not exist. Go make it or point to its proper location."
        fig.savefig(plot_dir + "/ssf_p_%.3f_a_%.3f-%i.pdf"%(p,a,version))
        # fig.show()
        plt.close()
        #----------------------Spit out spin configuration--------------------------
        oneD1 = np.array(range(0, l1))
        oneD2 = np.array(range(0, l2))
        n1, n2 = np.meshgrid(oneD1,oneD2)

        RX_list = np.empty((s), dtype=np.ndarray)
        RY_list = np.empty((s), dtype=np.ndarray)

        for i in range(s):
            RX_list[i] = sublattice_vectors[i,0]+ n1*T1[0] + n2*T2[0]
            RY_list[i] = sublattice_vectors[i,1]+ n1*T1[1] + n2*T2[1]

        mat_spin_config = np.reshape(flat_spin_config, (l1, l2, s,3))

        fig, ax = plt.subplots()
        for i in range(s):
            ax.quiver(RX_list[i], RY_list[i], mat_spin_config[:,:,i,0], mat_spin_config[:,:,i,1], color=color[i], scale=30,minlength=2)#, scale=None,headwidth=1,headlength=1)

        for x in range(l1):
            for y in range(l2):
                center1 = a2 + x*T1 + y*T2
                hex1 = ptch.RegularPolygon(center1, 6, 1/sqrt(3), 0, fill = False, linewidth=0.2)
                ax.add_patch(hex1)
        ax.axis("off")

        ax.plot([0, T1[0], 0, T2[0]], [0, T1[1], 0, T2[1]])
        ax.plot([T1[0],T1[0]+T2[0],T2[0],T1[0]+T2[0] ], [T1[1],T1[1]+T2[1], T2[1],T1[1]+T2[1]])

        ax.set(xlim=(-5, 5), ylim=(0, 10))
        ax.set_aspect('equal')
        fig.suptitle(r"($\phi/\pi,\, g, a, o, v)= (%.3f, %.3f, %.3f, %i, %i)$"%(p,g,a,order,version))
        fig.savefig(plot_dir + "/spins_p_%.3f_a_%.3f-%i.pdf"%(p,a,version))
        # fig.show()
        plt.close()

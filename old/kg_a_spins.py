#file   kkj_parse.py
#author Ahmed Rayyan
#date   November 20, 2019
#brief  parses through kkj data for analysis visualization
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from math import sqrt, tan
import sys

from lib_parse import pi, Phase, FindReciprocalVectors, MakeGrid, XYZtoABC, WhichUnitCell, ThreeToOneIndex
project = "kg_a"

l=6
p=0.25
# version=1
# for p in [0.01,0.08,0.15,0.25,0.352,0.49,0.5]:
for version in range(1,11):
    parameter_dir = "/l_%i/p_%.3f/v_%i"%(l,p,version)
    data_dir = "../../raw_data/" + project + parameter_dir
    assert(os.path.exists(data_dir)),"Your data_dir does not exist. Go make it or point to its proper location."
    print(data_dir)

    #extracting out the spins and projecting onto honeycomb plane
    file_list = glob.glob(data_dir + "/*")
    file_list.sort()

    for file in file_list:
        g = float(file.split("_")[-4])
        a = float(file.split("_")[-2])

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

    #--------------------------------calculating SSF--------------------------------
        oneD = np.array(range(0, length))

        dot_mat = np.einsum("ij,kj", flat_spin_config, flat_spin_config)

        b1, b2 = FindReciprocalVectors(a1, a2);
        KX, KY, gggg = MakeGrid(b1, b2, 12) #may be modified later...i dont need meshgrid
        kx, ky = [np.reshape(x, -1) for x in [KX, KY]]
        k = np.stack((kx,ky)).T

        bz2 = ptch.RegularPolygon((0,0), 6, np.linalg.norm((2*b1+b2)/3), pi/6, fill = False)
        bz3 = ptch.RegularPolygon((0,0), 6, np.linalg.norm(b1+b2), 0, fill = False)
    #
    #     gggg.axes[0].add_patch(bz2)
    #     gggg.axes[0].add_patch(bz3)
    #     gggg.axes[0].set_xlim(-6.5, 6.5)
    #     gggg.axes[0].set_ylim(-7.5, 7.5)
    #     gggg.show()
    # #
        s_k = np.empty(len(k))
        for i, kv in enumerate(k):
            phase_i = np.exp(1j * np.einsum('i,ji', kv, flat_spin_loc))
            phase_j = np.exp(-1j * np.einsum('i,ji', kv, flat_spin_loc))
            phase_mat = np.einsum('i,j->ij', phase_i, phase_j)
            s_k[i] = (dot_mat * phase_mat).sum()/sites
        s_k = np.reshape(s_k, KX.shape)
    #
        fig, ax = plt.subplots()
        c = ax.pcolormesh(KX, KY, s_k, cmap='viridis')
        ax.axis("equal")
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('$s_k$', labelpad=10)
        ax.axis("off")

        fig.axes[0].add_patch(bz2)
        fig.axes[0].add_patch(bz3)

        G = 0*b1 + 0*b2
        M1 = 0.5*b1 + 0*b2
        M2 = 0.5*b1 + 0.5*b2
        M3 = 0*b1 + 0.5*b2
        K = 2*b1/3 + b2/3
        Kp = b1/3 + 2*b2/3
        M231 = 2/3*M1
        M232 = 2/3*M2
        M233 = 2/3*M3
        K2 = K/2
        Kp2 = Kp/2
        K34 = 3*K/4
        Kp34 = 3*Kp/4

        sympoint = np.reshape(np.concatenate((G,M1,M2,M3,K,Kp,M231,M232,M233,K2,Kp2,K34,Kp34)),(13,2))
        fig.axes[0].scatter(sympoint.T[0],sympoint.T[1],color="w",s=0.8)

        fig.suptitle(r"($\phi/\pi,\, g, a)= (%.3f, %.3f, %.3f)$"%(p,g,a))
        plot_dir = "../../plot_data/" + project + parameter_dir
        # fig.savefig(plot_dir + "/ssf_g_%.3f_a_%.3f-%i.pdf"%(g,a,version))
        fig.show()
        plt.close()
    # #-----------------------------visualizing the spins-----------------------------
        oneD = np.array(range(0, length))
        n1, n2 = np.meshgrid(oneD,oneD)

        RX_list = np.empty((sublattice), dtype=np.ndarray)
        RY_list = np.empty((sublattice), dtype=np.ndarray)

        for i in range(sublattice):
            RX_list[i] = sublattice_vectors[i,0]+ n1*a1[0] + n2*a2[0]
            RY_list[i] = sublattice_vectors[i,1]+ n1*a1[1] + n2*a2[1]

        mat_spin_config = np.zeros((length, length, sublattice,3))
        for j in range(length):
            for i in range(length):
                for s in range(2):
                    z= ThreeToOneIndex(i, j, s, length, sublattice)
                    mat_spin_config[i,j,s,:] =flat_spin_config[z]

        color = np.array(["r","k","g","b","c","y"])
        fig, ax = plt.subplots()
        for i in range(sublattice):
            ax.quiver(RX_list[i], RY_list[i], mat_spin_config[:,:,i,0].T, mat_spin_config[:,:,i,1].T, color=color[i], scale=45,headwidth=5,headlength=8)

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
        fig.suptitle(r"($\phi/\pi,\, g, a)= (%.3f, %.3f, %.3f)$"%(p,g,a))
        plot_dir = "../../plot_data/" + project + parameter_dir
        # fig.savefig(plot_dir + "/spins_g_%.3f_a_%.3f-%i.pdf"%(g,a,version))
        fig.show()
        plt.close()

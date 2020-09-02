#file   kga_spins.py
#author Ahmed Rayyan
#date   March 15, 2020
#brief  spin analysis of KG + A phase diagram

import glob
import os
from lib_new_parse import WhichUnitCell, LocalRotation, FindReciprocalVectors, NewMakeGrid, pi, a1, a2
import numpy as np
from math import sqrt
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import multiprocessing as mp
from joblib import Parallel, delayed
import warnings

# for a in [-0.4,-0.3,-0.2,-0.1,0,-0.5]:
#     for v in [0,1,2,3,4,5,6]:
#         data_dir = "../../raw_data/kg/gk=gg_ak=ag/%i_%i_%i/c_%i/Tfp_%i/g_%.3f_a_%.3f/v_%i/"%(s, l1, l2,clus,Tfp,g,a,v)
#         print(data_dir)
#         assert(os.path.exists(data_dir)),"Your data directory does not exist."
#
#         plot_dir = "../../plot_data/kg/gk=gg_ak=ag/%i_%i_%i/c_%i/Tfp_%i/g_%.3f_a_%.3f/v_%i/"%(s, l1, l2,clus,Tfp,g,a,v)
#         if not os.path.exists(plot_dir):
#             os.makedirs(plot_dir)
#
#         file_list = glob.glob(data_dir+"/*")
#         file_list.sort()
#
#         for file in file_list:
#             #----------------------Read the spins in from file--------------------------
#             p = float(file.split("_")[-2])
#             with open(file, 'r') as f:
#                 file_data = f.readlines()
#             type = int(file_data[2])
#             l1, l2, s = [int(x) for x in file_data[4].split()]
#             sites = l1*l2*s
#             T1, T2, sublattice_vectors, color = WhichUnitCell(type, s, clus)
#
#             flat_spin_loc, flat_spin_config = np.empty((sites, 2)), np.empty((sites, 3))
#             for i, line in enumerate(file_data[32:]):
#                 x, y, sub, Sx, Sy, Sz = line.split()
#                 x, y, sub = list(map(int, [x,y,sub]))
#                 rot_spin = xyz_to_abc.dot(np.array([float(Sx), float(Sy), float(Sz)]))
#                 flat_spin_loc[i, :] = x*T1 + y*T2 + sublattice_vectors[sub]
#                 flat_spin_config[i, :] = rot_spin
#             #--------------------------Creating my k-space grid-------------------------
#             a1, a2 = np.array([1/2, sqrt(3)/2]), np.array([-1/2, sqrt(3)/2])
#             # b1, b2 = FindReciprocalVectors(a1, a2)
#             # B1, B2 = FindReciprocalVectors(T1, T2)
#             #
#             # KX, KY, gggg = NewMakeGrid(B1, B2, l1, l2) #may be modified later...i dont need meshgrid
#             # kx, ky = [np.reshape(x, -1) for x in [KX, KY]]
#             # k = np.stack((kx,ky)).T
#
#             # #keep vectors only within second brillouin zone
#             # klist = list(k)
#             # for i, kv in enumerate(klist):
#             #     if np.linalg.norm(kv) > np.linalg.norm(b1):
#             #         klist.pop(i)
#             # k = np.array(klist)
#             #------------------------------Calculate SSF--------------------------------
#             # dot_mat = np.einsum("ij,kj", flat_spin_config, flat_spin_config)
#             #
#             # s_k = np.empty(len(k))
#             # for i, kv in enumerate(k):
#             #     phase_i = np.exp(1j * np.einsum('i,ji', kv, flat_spin_loc))
#             #     phase_j = np.exp(-1j * np.einsum('i,ji', kv, flat_spin_loc))
#             #     phase_mat = np.einsum('i,j->ij', phase_i, phase_j)
#             #     s_k[i] = (dot_mat * phase_mat).sum()/sites
#             #---------------------------------Plot SSF----------------------------------
#             # s_k = np.reshape(s_k, KX.shape)
#             # fig, ax = plt.subplots()
#             # c = ax.scatter(KX, KY, c=s_k, cmap='viridis', edgecolors="none")
#             # cbar = fig.colorbar(c)
#             # cbar.set_label('$s_k$', labelpad=10)
#             # ax.axis("equal")
#             # ax.axis("off")
#             #
#             # bz2 = ptch.RegularPolygon((0,0), 6, np.linalg.norm((2*b1+b2)/3), pi/6, fill = False)
#             # bz3 = ptch.RegularPolygon((0,0), 6, np.linalg.norm(b1), 0, fill = False)
#             # bz4 = ptch.RegularPolygon((0,0), 60, np.linalg.norm(b1), 0, fill = False)
#             # fig.axes[0].add_patch(bz2)
#             # fig.axes[0].add_patch(bz3)
#             # fig.axes[0].add_patch(bz4)
#             # fig.axes[0].set_xlim(-6.5, 6.5)
#             # fig.axes[0].set_ylim(-7.5, 7.5)
#             #
#             # fig.savefig(plot_dir + "ssf_p_%.3f.pdf"%p)
#             # fig.show()
#             # plt.close()
#             #----------------------Spit out spin configuration--------------------------
#             oneD1 = np.array(range(0, l1))
#             oneD2 = np.array(range(0, l2))
#             n1, n2 = np.meshgrid(oneD1,oneD2)
#
#             RX_list = np.empty((s), dtype=np.ndarray)
#             RY_list = np.empty((s), dtype=np.ndarray)
#
#             for i in range(s):
#                 RX_list[i] = sublattice_vectors[i,0]+ n1*T1[0] + n2*T2[0]
#                 RY_list[i] = sublattice_vectors[i,1]+ n1*T1[1] + n2*T2[1]
#
#             mat_spin_config = np.reshape(flat_spin_config, (l1, l2, s,3))
#
#             fig, ax = plt.subplots()
#             mandem = np.arccos(np.clip(mat_spin_config[:,:,:,2],-1,1))/np.pi*180
#             norm = clr.Normalize()
#             norm.autoscale(mandem)
#             cm = plt.cm.coolwarm
#             for i in range(s):
#                 c = ax.quiver(RX_list[i], RY_list[i], mat_spin_config[:,:,i,0], mat_spin_config[:,:,i,1], mandem[:,:,i], cmap=cm, norm=norm, scale=15,minlength=3)#, scale=None,headwidth=1,headlength=1)
#             sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
#             plt.colorbar(sm)
#
#             for x in range(l1):
#                 for y in range(l2):
#                     center1 = x*T1 + y*T2
#                     center2 = a1+ x*T1 + y*T2
#                     hex1 = ptch.RegularPolygon(center1, 6, 1/sqrt(3), 0, fill = False, linewidth=0.2)
#                     ax.add_patch(hex1)
#                     hex2 = ptch.RegularPolygon(center2, 6, 1/sqrt(3), 0, fill = False, linewidth=0.2)
#                     ax.add_patch(hex2)
#             ax.axis("off")
#
#             ax.plot([0, T1[0], 0, T2[0]], [0, T1[1], 0, T2[1]])
#             ax.plot([T1[0],T1[0]+T2[0],T2[0],T1[0]+T2[0] ], [T1[1],T1[1]+T2[1], T2[1],T1[1]+T2[1]])
#             ax.set_aspect('equal')
#             # fig.savefig(plot_dir + "spin_p_%.3f.pdf"%p)
#             fig.show()
#             plt.close()

def func(v):
    plt.style.use('dark_background')
    warnings.filterwarnings('ignore')
    xyz_to_abc = LocalRotation(np.array([1,1,1])/sqrt(3))

    s = 2
    l1 = 3
    l2 = 3
    Tfp= 300
    clus = 2
    g = 0.5

    for p in [0.25]:
        data_dir = "../../raw_data/kg/gk=gg_ak=ag/%i_%i_%i/c_%i/Tfp_%i/g_%.3f_p_%.3f/v_%i/"%(s, l1, l2,clus,Tfp,g,p,v)
        print(data_dir)
        assert(os.path.exists(data_dir)),"Your data directory does not exist."

        plot_dir = "../../plot_data/kg/gk=gg_ak=ag/%i_%i_%i/c_%i/Tfp_%i/g_%.3f_p_%.3f/v_%i/"%(s, l1, l2,clus, Tfp,g,p,v)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        file_list = glob.glob(data_dir+"/*")
        file_list.sort()
        for file in file_list:
            #----------------------Read the spins in from file--------------------------
            a = float(file.split("_")[-2])
            with open(file, 'r') as f:
                file_data = f.readlines()
            # print(a)
            type = int(file_data[2])
            l1, l2, s = [int(x) for x in file_data[4].split()]
            sites = l1*l2*s

            T1, T2, sublattice_vectors, _ = WhichUnitCell(type, s, clus)

            flat_spin_loc, flat_spin_config = np.empty((sites, 2)), np.empty((sites, 3))
            for i, line in enumerate(file_data[32:]):
                x, y, sub, Sx, Sy, Sz = line.split()
                x, y, sub = list(map(int, [x,y,sub]))
                rot_spin = xyz_to_abc.dot(np.array([float(Sx), float(Sy), float(Sz)]))
                flat_spin_loc[i, :] = x*T1 + y*T2 + sublattice_vectors[sub]
                flat_spin_config[i, :] = rot_spin

            a1, a2 = np.array([1/2, sqrt(3)/2]), np.array([-1/2, sqrt(3)/2])
            #--------------------------Creating my k-space grid-------------------------
            b1, b2 = FindReciprocalVectors(a1, a2)
            B1, B2 = FindReciprocalVectors(T1, T2)
            # KX, KY, gggg = NewMakeGrid(B1, B2, l1, l2,2) #may be modified later...i dont need meshgrid
            KX, KY, gggg = NewMakeGrid(B1, B2, l1, l2,2) #may be modified later...i dont need meshgrid
            kx, ky = [np.reshape(x, -1) for x in [KX, KY]]
            k = np.stack((kx,ky)).T

            ## # keep vectors only within second brillouin zone
            ## klist = list(k)
            ## print()
            ## for i, kv in enumerate(klist):
            ##     if np.linalg.norm(kv) > np.linalg.norm(b1):
            ##         klist.pop(i)
            ## k = np.array(klist)
            # ------------------------------Calculate SSF--------------------------------
            dot_mat = np.einsum("ij,kj", flat_spin_config, flat_spin_config)

            s_k = np.empty(len(k))
            for i, kv in enumerate(k):
                phase_i = np.exp(1j * np.einsum('i,ji', kv, flat_spin_loc))
                phase_j = np.exp(-1j * np.einsum('i,ji', kv, flat_spin_loc))
                phase_mat = np.einsum('i,j->ij', phase_i, phase_j)
                s_k[i] = (dot_mat * phase_mat).sum()/sites

            # ---------------------------------Plot SSF----------------------------------
            # print(s_k)
            s_k = np.reshape(s_k, KX.shape)
            fig, ax = plt.subplots()
            c = ax.scatter(KX, KY, c=s_k, cmap='copper', edgecolors="none")
            cbar = fig.colorbar(c)
            cbar.set_label('$s_k$', labelpad=10)
            ax.axis("equal")
            ax.axis("off")

            bz2 = ptch.RegularPolygon((0,0), 6, np.linalg.norm((2*b1+b2)/3), pi/6, fill = False)
            bz3 = ptch.RegularPolygon((0,0), 6, np.linalg.norm(b1), 0, fill = False)
            fig.axes[0].add_patch(bz2)
            fig.axes[0].add_patch(bz3)
            fig.axes[0].set_xlim(-6.5, 6.5)
            fig.axes[0].set_ylim(-7.5, 7.5)

            # fig.savefig(plot_dir + "ssf_a_%.3f.pdf"%a)
            plt.show()
            plt.close()
            # # ----------------------Spit out spin configuration--------------------------
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
            # plt.figure(figsize=(8,8))
            mandem = np.arccos(np.clip(mat_spin_config[:,:,:,2],-1,1))/np.pi*180
            norm = clr.Normalize()
            norm.autoscale(mandem)
            cm = plt.cm.coolwarm
            for i in range(s):
                c = ax.quiver(RX_list[i], RY_list[i], mat_spin_config[:,:,i,0], mat_spin_config[:,:,i,1], mandem[:,:,i], cmap=cm, norm=norm, scale=50,minlength=1.5)#, scale=None,headwidth=1,headlength=1)

                # c = ax.quiver(RX_list[i], RY_list[i], mat_spin_config[:,:,i,0], mat_spin_config[:,:,i,1], mandem[:,:,i], cmap=cm, norm=norm, scale=15,minlength=2)#, scale=None,headwidth=1,headlength=1)
            sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
            cb = plt.colorbar(sm,fraction=0.08,pad=0.15,orientation='horizontal')
            cb.ax.set_title(r'$\;\qquad\theta_c$')

            for x in range(l1):
                for y in range(l2):
                    center1 = x*T1 + y*T2
                    hex1 = ptch.RegularPolygon(center1, 6, 1/sqrt(3), 0, fill = False, linewidth=0.2)
                    ax.add_patch(hex1)
                    hex2 = ptch.RegularPolygon(center1+a1, 6, 1/sqrt(3), 0, fill = False, linewidth=0.2)
                    ax.add_patch(hex2)
            ax.axis("off")

            ax.plot([0, T1[0], 0, T2[0]], [0, T1[1], 0, T2[1]])
            ax.plot([T1[0],T1[0]+T2[0],T2[0],T1[0]+T2[0] ], [T1[1],T1[1]+T2[1], T2[1],T1[1]+T2[1]])
            # ax.set_aspect('equal')
            # fig.savefig(plot_dir + "spin_a_%.3f.pdf"%a)
            fig.show()
            plt.close()

if __name__ == "__main__":

    vlist =  np.array(range(81))
    do_eet = Parallel(n_jobs=mp.cpu_count())(delayed(func)(i) for i in vlist)

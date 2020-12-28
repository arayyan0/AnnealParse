#file   kga_spins.py
#author Ahmed Rayyan
#date   March 15, 2020
#brief  spin analysis of KG + A phase diagram

import glob
import os
from lib_new_parse import WhichUnitCell, FindReciprocalVectors, LocalRotation, pi, SpinConfiguration, LSWT, MakeFirstBZ,ReciprocalLattice, LSWT, AddBZ
import numpy as np
from math import sqrt
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import multiprocessing as mp
from joblib import Parallel, delayed
import warnings

def func(a):
    plt.style.use('dark_background')
    # warnings.filterwarnings('ignore')
    xyz_to_abc = LocalRotation(np.array([1,1,1])/sqrt(3))

    p = 0.05

    order = "ZZ"
    run = 2
    version = 0
    cluster="rh2"

    # alst = np.linspace(-0.4,0,80+1)
    # alst = np.linspace(0.005,0.805,80+1)
    # for a in alst:
    data_folder = f"out/p_{p:.3f}_r_{run}/v_{version}/"
    file = data_folder + f"a_{a:.3f}_.out"
    print(file)
    # assert(os.path.exists(data_dir)),"Your data directory does not exist."

    plot_dir = f"out/p_{p:.3f}_r_{run}/v_{version}/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    #----------------------Read the spins in from file--------------------------
    with open(file, 'r') as f:
        file_data = f.readlines()

    Kx, Ky, Kz = [float(x) for x in file_data[15].split()]
    Gx, Gy, Gz = [float(x) for x in file_data[17].split()]
    Gp = float(file_data[19].split()[0])
    J1 = float(file_data[21].split()[0])

    # print(a)
    hc_or_kek, type = [int(x) for x in file_data[2].split()]
    s, l1, l2 = [int(x) for x in file_data[4].split()]
    sites = l1*l2*s
    #----------------------PLOT SSF AND SPINS-------------------------
    # T1, T2, sublattice_vectors, _ = WhichUnitCell(hc_or_kek, s, type)
    # flat_spin_loc, flat_spin_config = np.empty((sites, 2)), np.empty((sites, 3))
    # for i, line in enumerate(file_data[32:]):
    #     x, y, sub, Sx, Sy, Sz = line.split()
    #     x, y, sub = list(map(int, [x,y,sub]))
    #     rot_spin = xyz_to_abc.dot(np.array([float(Sx), float(Sy), float(Sz)]))
    #     flat_spin_loc[i, :] = x*T1 + y*T2 + sublattice_vectors[sub]
    #     flat_spin_config[i, :] = rot_spin
    #
    # spinstuff = SpinConfiguration(flat_spin_loc, flat_spin_config, [hc_or_kek, s, l1, l2, type])
    #
    # fig = spinstuff.PlotSSF()
    # # fig.show()
    # fig.savefig(plot_dir + f"ssf_a_{a:.3f}.pdf")
    # plt.close()
    #
    # fig = spinstuff.PlotSpins()
    # # fig.show()
    # fig.savefig(plot_dir + f"spin_a_{a:.3f}.pdf")
    # plt.close()
    #----------------------do LSWT--------------------------
    spins = np.empty((sites, 3), dtype=np.longdouble)
    for i, line in enumerate(file_data[32:]):
        n1, n2, sub, Sx, Sy, Sz = line.split()
        spins[i, :] = np.array([np.longdouble(Sx), np.longdouble(Sy), np.longdouble(Sz)])

    #create LSWT object
    lswtea = LSWT([np.array([Kx,Ky,Kz]), np.array([Gx, Gy, Gz]), Gp, J1],
                  [order,cluster],
                  spins)
    print(f"Cluster energy: {lswtea.ClusterEnergy/sites}")

        #---------------------full k grid (moment, lowest_band_plotting)--------------------------
    n=3
    B1, B2 = FindReciprocalVectors(lswtea.T1, lswtea.T2)
    KX, KY, fig = MakeFirstBZ(B1, B2, n*2*3, n*2*3,3,6)
    kx, ky = [np.reshape(x, -1) for x in [KX, KY]]
    k = np.stack((kx, ky)).T

    # lift=0.0
    # lswtea.ObtainMagnonSpectrumAndDiagonalizer(k, 0, lift)
    # lswtea.MagnonWFProperties()
    # print(f"Moment reduced to: {lswtea.ReductionOfMoment}")
    # return [lswtea.ReductionOfMoment, a]

    lift=0.00
    lswtea.ObtainMagnonSpectrumAndDiagonalizer(k, 0, lift)
    fig = lswtea.PlotLowestBand(KX, KY)
    fig.savefig(plot_dir + f"lb_a_{a:.3f}.pdf")
    # plt.show()
    plt.close()
    # print(f"Moment reduced to: {lswtea.ReductionOfMoment}")
    # return [lswtea.ReductionOfMoment, a]

        #---------------------k path (band structure)--------------------------
    lift=0
    lattice_characteristics = ReciprocalLattice()
    kp = lattice_characteristics.MakeKPath(["X","G","M2","Gp1","M1","G"],50)
    i=0
    lswtea.ObtainMagnonSpectrumAndDiagonalizer(kp[0], i*2*np.pi/3, lift)
    fig = lswtea.PlotMagnonKPath(*kp)
    # fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.savefig(plot_dir + f"lswt_a_{a:.3f}.pdf")

if __name__ == "__main__":

    #parameter parallelization
    alist =  np.linspace(0.01,0.11,10+1)
    # alist =  np.linspace(-0.1,0.0,10+1)
    # alist = np.array([0.01])
    neg_list = np.array(Parallel(n_jobs=mp.cpu_count())(delayed(func)(a) for a in alist))
    #
    # idx = np.argsort(unsort_alist)
    # sort_alist = unsort_alist[idx]
    # moment_list = moment_list[idx]
    #
    #
    # plt.plot(sort_alist,moment_list)
    # plt.show()
    # plt.close()

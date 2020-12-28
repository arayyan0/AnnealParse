# file: spin_waves.py
#author: Ahmed Rayyan
#date: April 23, 2020
#brief: calculates spin wave given a spin configuration

from lib_new_parse import ReciprocalLattice, LSWT, FindReciprocalVectors, MakeFirstBZ, AddBZ
import numpy as np
from math import isclose
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
# import warnings
# warnings.filterwarnings('ignore')

# for jjj in range(1,10+1):
# j=1
# -----------------------load up the spin configuration-------------------------#
# order = "6"
# cluster_type = "rh"
# cluster_number = "2"
# cluster = cluster_type+cluster_number
# file = f"outmy/{order}_4.out"
# file=f"D1_spin_wave/{jjj}.out"
# p=0
# file = f"out/{order}_{cluster}_{v}.out"
# order="6"
# cluster="rh2"
# file="_6_2.out"

# p = 0.400
# g = 0.100
# v = 1
# file = f"out/{order}_{cluster}_{p:.3f}_{g:.3f}_{v}.out"
# p=0.05
# file = f"out/{order}_{p:.2f}_.out"
# file = "out/exc.out"
# order="18"
# cluster="rh1"
# for j in range(1,11):
order = "18"
for j in range(1,10+1):
    # j=2
    cluster = "rh1"
    folder = "outmy/H_spin_wave"
    # folder = "outJGR/"
    file = f"{folder}/{j}.out"
    # file = f"{folder}/{order}_{j}."
    print(file)
    with open(file, 'r') as f:
        file_data = f.readlines()

    #extract Hamiltonian parameters
    Kx, Ky, Kz = [float(x) for x in file_data[15].split()]
    Gx, Gy, Gz = [float(x) for x in file_data[17].split()]
    # print([g/np.abs(k) for k,g in zip([Kx, Ky, Kz],[Gx, Gy, Gz])])

    Gp = float(file_data[19].split()[0])
    J1 = float(file_data[21].split()[0])

    #
    # extract cluster information
    l1, l2, s = [int(x) for x in file_data[4].split()]
    sites = l1*l2*s
    # print(l1,l2,s)

    # print(file_data[32:])

    # extract spin information
    spins = np.empty((sites, 3), dtype=np.longdouble)
    for i, line in enumerate(file_data[32:]):
        n1, n2, sub, Sx, Sy, Sz = line.split()
        spins[i, :] = np.array([np.longdouble(Sx), np.longdouble(Sy), np.longdouble(Sz)])

    #create LSWT object
    lswtea = LSWT([np.array([Kx,Ky,Kz]), np.array([Gx, Gy, Gz]), Gp, J1],
                  [order,cluster],
                  spins)
    #
    print(f"Cluster energy: {lswtea.ClusterEnergy/sites}")

    # # # #-------------------create kgrid and calculate...------------------------#
    n=1
    B1, B2 = FindReciprocalVectors(lswtea.T1, lswtea.T2)
    KX, KY, fig = MakeFirstBZ(B1, B2, n*2*3, n*2*3,3,3)
    kx, ky = [np.reshape(x, -1) for x in [KX, KY]]
    k = np.stack((kx, ky)).T

    # plt.show()
    # plt.close()

    lift=0
    lswtea.ObtainMagnonSpectrumAndDiagonalizer(k, 0, lift)
    lswtea.MagnonWFProperties()
    # plt.close()

    print(f"Moment reduced to: {lswtea.ReductionOfMoment}")

    fig = lswtea.PlotLowestBand(KX, KY)
    plt.title(f"Moment reduced to: {lswtea.ReductionOfMoment}")
    plt.savefig(f"{folder}/lb_{j}.pdf")

    # plt.show()
    plt.close()

    # # # #-------------------------------create kpath-----------------------------#
    lattice_characteristics = ReciprocalLattice()
    # kp = lattice_characteristics.MakeKPath(["G","M1","Gp1","Gp2","M2","G"],100)
    # kp = lattice_characteristics.MakeKPath(["G","M1","K","G","M2","Kp","G","M3","Kpp","G"],100)
    # kp = lattice_characteristics.MakeKPath(["G","M1","K","G"],100)
    # kp = lattice_characteristics.MakeKPath(["G","M2","Kpp"],10)
    kp = lattice_characteristics.MakeKPath(["X","G","M2","Gp1","M1","G"],50)
    # kp = [np.array([[0.2,0.4],[0.1,0.5]])]

    i=0
    lswtea.ObtainMagnonSpectrumAndDiagonalizer(kp[0], i*2*np.pi/3, 0.001)
    fig = lswtea.PlotMagnonKPath(*kp)
    # fig.suptitle(rf"$\phi/\pi = {p:.3f}$, {order} order")
    # # fig.axes[0].set_ylim(0,0.01)
    # fig.tight_layout(rect=[0,0.03,1,0.95])
    plt.savefig(f"{folder}/bands_{j}.pdf")
    # # # plt.savefig(f"D1_spin_wave/{jjj}.pdf")
    # plt.show()
    plt.close()

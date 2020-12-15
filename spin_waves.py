#file: spin_waves.py
#author: Ahmed Rayyan
#date: April 23, 2020
#brief: calculates spin wave given a spin configuration

from lib_new_parse import WhichOrder, ReciprocalLattice, LSWT, FindReciprocalVectors, IndexToPosition, EquivalentBravais, LocalRotation
import numpy as np
from math import isclose
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# for jjj in range(1,10+1):
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
for j in range(1,11):
    # j=2
    order = "18"
    cluster = "rh1"
    folder = "H_spin_wave"
    file = f"{folder}/{j}.out"
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
    spins = np.empty((sites, 3), dtype=np.double)
    for i, line in enumerate(file_data[32:]):
        n1, n2, sub, Sx, Sy, Sz = line.split()
        spins[i, :] = np.array([float(Sx), float(Sy), float(Sz)])

    #create LSWT object
    lswtea = LSWT([[Kx,Ky,Kz], [Gx, Gy, Gz], Gp, J1],
                  [order,cluster],
                  spins)
    #
    print(lswtea.ClusterEnergy/sites)
    # # #-------------------------------create kspace----------------------------------#
    lattice_characteristics = ReciprocalLattice()
    # kp=[[np.array([-3.14159,  1.8138]),-np.array([-3.14159,  1.8138])]]

    # kp = lattice_characteristics.MakeKPath(["G","M1","Gp1","Gp2","M2","G"],100)
    # kp = lattice_characteristics.MakeKPath(["G","M1","K","G","M2","Kp","G","M3","Kpp","G"],100)
    # kp = lattice_characteristics.MakeKPath(["G","M1","K","G"],100)
    # kp = lattice_characteristics.MakeKPath(["-M1","G","M1"],10)

    # with np.printoptions(threshold=np.inf):
        # print(kp[0])
    # print(len(kp[0]))
    kp = lattice_characteristics.MakeKPath(["X","G","M2","Gp1","M1","G"],100)
    # print(kp)
    # print(kp[0][0])
    # #---------------------fill the second quantized Hamiltonian--------------------#
    # for i in range(3):
    i=0
    lswtea.ObtainMagnonSpectrumAndDiagonalizer(kp[0],i*2*np.pi/3,0)
    fig = lswtea.PlotMagnonDispersions(*kp)
    # fig.suptitle(rf"$\phi/\pi = {p:.3f}$, {order} order")
    # fig.axes[0].set_ylim(25,60)
    fig.tight_layout(rect=[0,0.03,1,0.95])
    plt.savefig(f"{folder}/{j}_lswt.pdf")
    # plt.savefig(f"D1_spin_wave/{jjj}.pdf")
    # plt.show()
    # plt.close()
    # lswtea.ObtainMagnonSpectrumAndDiagonalizer([np.array([-3.14159,  1.8138]),-np.array([-3.14159,  1.8138])],0)

    # lswtea.ObtainMagnonSpectrumAndDiagonalizer(kp[0],0)

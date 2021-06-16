import numpy as np
import matplotlib.pyplot as plt
from spin_lib import AnnealedSpinConfigurationTriangular
import os
import glob as glob

which = "ising_l=6,12,18"

for number in range(1,3+1):
    run = f'{number}'

    for version in range(1,5+1):
        data_folder = f'out/{which}/jobrun_{run}/v_{version}/'
        print(data_folder)
        plot_folder = data_folder + f'plots/'
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

        file_lst = glob.glob(data_folder+'*.out')

        Tlst = []
        specificheatlst=[]

        for file in file_lst:
            split_file = file.replace('/','_').split('_')
            T = float(split_file[-2])
            spinstuff = AnnealedSpinConfigurationTriangular(file)
            sh = spinstuff.SpecificHeat

            Tlst.append(T)
            specificheatlst.append(sh)

        Tarray = np.array(Tlst)
        SpecificHeatarray = np.array(specificheatlst)

        idx = np.argsort(Tarray)
        Tarray = Tarray[idx]
        SpecificHeatarray = SpecificHeatarray[idx]

        print(np.array([Tarray, SpecificHeatarray]).T)

        fig,ax = plt.subplots()
        ax.scatter(Tarray, SpecificHeatarray, marker = '.', facecolors='none',edgecolors='green')

        ax.set_xlabel('Temperature')
        ax.set_ylabel(r'$C_v$')
        ax.set_title(f'L={number*6}')

        fig.tight_layout()
        plt.savefig(plot_folder+f'specificHeat.pdf')
        plt.show()
        plt.close()

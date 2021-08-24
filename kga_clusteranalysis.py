import glob as glob
import matplotlib.pyplot as plt
import numpy as np
import os
from spin_lib import AnnealedSpinConfiguration
import sys

colors = ['red', 'salmon', 'dodgerblue', 'cyan', 'purple', 'fuchsia']

clusters = [
    [2,2,2,1],
    [2,2,3,1],
    [2,2,1,3],
    [2,2,8,1],
    [2,4,1,4],
    [2,4,3,2],
]
colors = ['salmon', 'dodgerblue', 'cyan', 'green', 'lime', 'fuchsia']
labels = ['ZZy', '6y', '6z', '16y', '16z', '24z']

fig, ax = plt.subplots()

data_folder = 'out/16-siteregion-2/'
scan_which = 'p'
y_fixed = float(sys.argv[1])

clusterEnergies = []
for [type, s, l1, l2], color, label in zip(clusters, colors, labels):
    cluster_folder = data_folder+ f'{s}_{l1}_{l2}/'
    print(cluster_folder)

    if  scan_which == 'p':
        plot_folder = cluster_folder+'plots/'+f'a_{y_fixed}/'
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        file_list = glob.glob(cluster_folder+f'p_*_a_{y_fixed:.3f}_.out')
        # print(cluster_folder+f'p_{y_fixed:.3f}_a_*_.out')
    elif scan_which == 'a':
        plot_folder = cluster_folder+'plots/'+f'p_{y_fixed}/'
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        file_list = glob.glob(cluster_folder+f'p_{y_fixed:.3f}_a_*_.out')
        # print(cluster_folder+f'p_{y_fixed:.3f}_a_*_.out')
    xlist = []
    Elist = []
    # print(file_list)
    for file in file_list:
        split_file = file.replace("/","_").split("_")
        p, a = [float(split_file[x]) for x in [-4, -2]]

        if  scan_which == 'p':
            xlist.append(p)
        elif  scan_which == 'a':
            xlist.append(a)

        spinstuff = AnnealedSpinConfiguration(file)
        Elist.append(spinstuff.MCEnergyDensity)

        cb_options=[0.04,'vertical','binary']
        usetex= False
        ssffig = spinstuff.CalculateAndPlotSSF(cb_options, usetex)
        plt.savefig(plot_folder+f'spins_p_{p:.3f}_a_{a:.3f}.pdf')
        ssffig.tight_layout(rect=[0,0.03,1,0.95])
        plt.close()

    xarray, Earray = map(np.array, [xlist, Elist])
    idx = np.argsort(xarray)
    xarray = xarray[idx]
    Earray = Earray[idx]

    clusterEnergies.append(Earray)

    ax.plot(xarray, Earray, '-o', c=color, label=label)
    # ax.scatter(xarray, Earray, c=color, label=label)

plt.legend()
plt.show()
plt.close()

clusterEnergies.append(xarray)

clusterEnergies = np.array(clusterEnergies)
print(clusterEnergies.T)
min = np.amin(clusterEnergies, axis = 0)
print(min)

#file   z_energy.py
#author Ahmed Rayyan
#date   June 19, 2020
#brief  finds minimum energy within z phase line

import glob
import os
from lib_new_parse import ExtractEnergyFromFile, PhiSweep
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.filterwarnings('ignore')

s=4
l1=12
l2=6

z=-0.5

for v in [0,1,2,3,4]:
    data_dir = "../../raw_data/z/%i_%i_%i/z_%.3f/v_%i/"%(s,l1,l2,z,v)
    print(data_dir)
    assert(os.path.exists(data_dir)),"Your data directory does not exist."

    plot_dir = "../../plot_data/z/%i_%i_%i/z_%.3f/v_%i/"%(s,l1,l2,z,v)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    file_list = glob.glob(data_dir+"/*")
    file_list.sort()

    p_list, e_list = [], []
    for file in file_list:
        #----------------------Read the spins in from file--------------------------
        p = float(file.split("_")[-2])
        with open(file, 'r') as f:
            file_data = f.readlines()

        e = ExtractEnergyFromFile(file)
        p_list.append(p)
        e_list.append(e)

    idx = np.argsort(p_list)
    p_list = np.array(p_list)[idx]
    e_list = np.array(e_list)[idx]

    sweep = PhiSweep(p_list,e_list)
    print(p_list)
    fig = sweep.PlotLabeledSweep()
    fig.savefig(plot_dir+ "/pd.pdf")
    # plt.show()

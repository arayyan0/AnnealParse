#file   kgh_energy.py
#author Ahmed Rayyan
#date   June 19, 2020
#brief  finds minimum energy within KG + h phase diagram

import glob
import os
from lib_new_parse import ExtractEnergyFromFile, FieldSweep
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.filterwarnings('ignore')

s=4
l1=15
l2=9

hth = 5
gp = -0.02
p = 0.148
a = 0

for v in [0,1,2,3,4,5,6]:
    data_dir = "../../raw_data/kgh/%i_%i_%i/hth_%i_gp_%.3f/p_%.3f_a_%.3f/v_%i/"%(s, l1, l2, hth, gp, p, a,v)
    print(data_dir)
    assert(os.path.exists(data_dir)),"Your data directory does not exist."

    plot_dir = "../../plot_data/kgh/%i_%i_%i/hth_%i_gp_%.3f/p_%.3f_a_%.3f/v_%i/"%(s, l1, l2, hth, gp, p, a,v)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    file_list = glob.glob(data_dir+"/*")
    file_list.sort()

    h_list, e_list = [], []
    for file in file_list:
        #----------------------Read the spins in from file--------------------------
        h = float(file.split("_")[-2])
        with open(file, 'r') as f:
            file_data = f.readlines()

        e = ExtractEnergyFromFile(file)
        h_list.append(h)
        e_list.append(e)

    idx = np.argsort(h_list)
    h_list = np.array(h_list)[idx]
    e_list = np.array(e_list)[idx]

    sweep = FieldSweep(h_list,e_list)
    print(e_list)
    fig = sweep.PlotLabeledSweep()
    fig.savefig(plot_dir+ "/pd.pdf")
    # plt.show()

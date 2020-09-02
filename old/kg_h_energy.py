#file   kg_h_parse_energy.py
#author Ahmed Rayyan
#date   December 18, 2019
#brief  parses through data for analysis visualization
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from lib_parse import KG_HSweep

project = "kg_h"
j = 0.000
gp = 0.000
ak = 0.000
ag = 0.000
ht = 0.000
hp = 0.000

parameter_dir = "/gp_%.3f_j_%.3f/ak_%.3f_ag_%.3f/ht_%.3f_hp_%.3f"%(gp, j, ak, ag, ht, hp)
data_dir = "../../raw_data/" + project + parameter_dir
assert(os.path.exists(data_dir)),"Your data_dir does not exist. Go make it or point to its proper location."

which_p = 0.121

file_list = glob.glob(data_dir+"/p_%.3f*"%which_p)
file_list.sort()

sweep_list = []
h_list, e_list = [], []

for file in file_list:
    h = float(file.split("_")[-2])
    with open(file, 'r') as f:
        file_data = f.readlines()
    e = float(file_data[30])

    h_list.append(h)
    e_list.append(e)

idx = np.argsort(h_list)
h_list = np.array(h_list)[idx]
e_list = np.array(e_list)[idx]

sweep_list.append(KG_HSweep(h_list, e_list))

plot_dir = "../../plot_data/" + project + parameter_dir
#
for sweep in sweep_list:
    fig = sweep.PlotSweep()
    fig.suptitle(r"$(\phi/\pi,\,J,\, \Gamma',\, h_\theta,\, h_\phi) = (%.3f,%.1f,%.1f,%.0f,%.0f)$"%(which_p,j,gp,ht,hp))
    fig.savefig(plot_dir + "/p-%.3f.pdf"%which_p)
    fig.show()
    plt.close()

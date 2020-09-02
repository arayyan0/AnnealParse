#file   kkj_energy.py
#author Ahmed Rayyan
#date   November 20, 2019
#brief  parses through kkj data for analysis visualization
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys

from lib_parse import ExtractEnergyFromFile, KKJSweep

project = "kkj"
g = 0.000
gp = 0.000
h = 0.000
ht = 0.000
hp = 0.000

version = 7

parameter_dir = "/g_%.3f_gp_%.3f/h_%.3f/ht_%.3f_hp_%.3f/v_%i"%(g, gp, h, ht, hp, version)
data_dir = "../../raw_data/" + project + parameter_dir
assert(os.path.exists(data_dir)),"Your data_dir does not exist. Go make it or point to its proper location."

file_list = glob.glob(data_dir+"/*")
p_list, e_list = [], []

for file in file_list:
    p = float(file.split("_")[-2])
    p_list.append(p)

    e = ExtractEnergyFromFile(file)
    e_list.append(e)

idx = np.argsort(p_list)
new_p_list = np.array(p_list)[idx]
e_list = np.array(e_list)[idx]

sweep = KKJSweep(new_p_list, e_list)
fig = sweep.PlotSweep()
fig.suptitle(r"Kekule-Kitaev-Heisenberg: J' = - sin $\theta,\;$ K' = cos $\theta$")
fig.show()
plt.close()
# plot_dir = "../../plot_data/" + project + parameter_dir
# fig.savefig(plot_dir + "/kkj_sweep_zoomed.pdf")

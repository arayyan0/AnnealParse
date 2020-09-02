#file   kg_parse_energy.py
#author Ahmed Rayyan
#date   January 14 2020
#brief  parses through data for analysis visualization
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from lib_parse import KG_Sweep

project = "kg"
gk = 0.500
gg = 0.500
ak = 0.000
ag = 0.000
l = 6
version = 5

parameter_dir = "/gk_%.3f_gg_%.3f_ak_%.3f_ag_%.3f/l_%i/v_%i"%(gk, gg, ak, ag, l, version)
data_dir = "../../raw_data/" + project + parameter_dir
print(data_dir)
assert(os.path.exists(data_dir)),"Your data_dir does not exist. Go make it or point to its proper location."

file_list = glob.glob(data_dir+"/*")
file_list.sort()

p_list = []
e_list = []

for file in file_list:
    p = float(file.split("_")[-2])
    with open(file, 'r') as f:
        file_data = f.readlines()
    e = float(file_data[30])

    p_list.append(p)
    e_list.append(e)

idx = np.argsort(p_list)
p_list = np.array(p_list)[idx]
e_list = np.array(e_list)[idx]

sweep = KG_Sweep(p_list, e_list)
fig = sweep.PlotSweep()
fig.suptitle("$(g_K,\, g_\Gamma,\, a_K,\, a_\Gamma) = (%.2f, %.2f, %.2f, %.2f)$"%(gk, gg, ak, ag))
plot_dir = "../../plot_data/" + project + parameter_dir
fig.savefig(plot_dir + "/kg-%i.pdf"%version)
fig.show()
plt.close()

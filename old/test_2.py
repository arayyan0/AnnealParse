#file   test.py
#author Ahmed Rayyan
#date   November 20, 2019
#brief  parses through kkj data for energy landscape

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import shutil

from lib_parse import ExtractEnergyFromFile, KG_ASweep

project = "kg_a"
# for p in [0,0.01,0.08,0.15,0.25,0.352,0.49,0.5]:
p = 0.15
g = 0.5
l = 6

parameter_dir = "/l_%i/p_%.3f"%(l,p)
data_dir = "../../raw_data/" + project + parameter_dir
assert(os.path.exists(data_dir)),"Your data_dir does not exist. Go make it or point to its proper location."

version_list = glob.glob(data_dir+"/*")
version_list.sort()

number_of_versions = len(version_list)
number_of_datapoints = len(glob.glob(version_list[0]+"/*"))

data = np.zeros((number_of_datapoints, number_of_versions+1))

for version in version_list:
    i = int(version.split("_")[-1])
    file_list = glob.glob(version+"/*")
    file_list.sort()
    for j, file in enumerate(file_list):
        data[j,0] =  float(file.split("_")[-2])
        # data[j,i] =  round(ExtractEnergyFromFile(file),3)
        data[j,i] =  round(ExtractEnergyFromFile(file),14)

idx = np.argsort(data[:,0])
data[:,0] = np.array(data[:,0])[idx]
data[:,1:] = np.array(data[:,1:])[idx]

minimum = data.min(axis=1)
x = (data.T - minimum).T
#
# ##------------------------moving ssfs----------------------------------
# for i,j in zip(np.where(x == 0)[0], np.where(x==0)[1]):
#     shutil.copy("../../plot_data/" + project + parameter_dir + "/v_%i/ssf_g_%.3f_a_%.3f-%i.pdf"%(j,g,data[i,0],j), "../../plot_data/" + project + parameter_dir + "/total")
#     shutil.copy("../../plot_data/" + project + parameter_dir + "/v_%i/spins_g_%.3f_a_%.3f-%i.pdf"%(j,g,data[i,0],j), "../../plot_data/" + project + parameter_dir + "/total")
#
# ##-------------------------plotting the best one-----------------------
sweep = KG_ASweep("g", g, data[:,0], minimum)
fig = sweep.PlotSweep()
fig.suptitle(r"($\phi/\pi,\, g)= (%.3f, %.3f)$"%(p,g))
# fig.savefig("../../plot_data/" + project + parameter_dir + "/total/kg_a.pdf")
fig.show()
plt.close()

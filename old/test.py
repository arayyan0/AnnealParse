#file   test.py
#author Ahmed Rayyan
#date   November 20, 2019
#brief  parses through kkj data for energy landscape

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import shutil

from lib_parse import ExtractEnergyFromFile, KG_Sweep

project = "kg"
gk = 0.500
gg = 0.500
ak = 0.000
ag = 0.000
l = 6

parameter_dir = "/gk_%.3f_gg_%.3f_ak_%.3f_ag_%.3f/l_%i"%(gk, gg, ak, ag, l)
data_dir = "../../raw_data/" + project + parameter_dir
print(data_dir)
assert(os.path.exists(data_dir)),"Your data_dir does not exist. Go make it or point to its proper location."

version_list = glob.glob(data_dir+"/*")
version_list.sort()

number_of_versions = len(version_list)
number_of_datapoints = len(glob.glob(version_list[0]+"/*"))

data = np.zeros((number_of_datapoints, number_of_versions+1))

for i in range(1,number_of_versions+1):
    file_list = glob.glob(version_list[i-1]+"/*")
    file_list.sort()
    for j, file in enumerate(file_list):
        data[j,0] =  float(file.split("_")[-2])
        data[j,i] =  ExtractEnergyFromFile(file)

minimum = data.min(axis=1)
x = (data.T - minimum).T

##------------------------moving ssfs----------------------------------
for i,j in zip(np.where(x == 0)[0], np.where(x==0)[1]):
    print(data[i,0],j)
    shutil.copy("../../plot_data/" + project + parameter_dir + "/v_%i/ssf_p_%.3f-%i.pdf"%(j,data[i,0],j), "../../plot_data/" + project + parameter_dir + "/total")
    shutil.copy("../../plot_data/" + project + parameter_dir + "/v_%i/spins_p_%.3f-%i.pdf"%(j,data[i,0],j), "../../plot_data/" + project + parameter_dir + "/total")

##-------------------------plotting the best one-----------------------
sweep = KG_Sweep(data[:,0], minimum)
fig = sweep.PlotSweep()
fig.suptitle(r"K$\Gamma$ model: K = - sin $\phi,\;$ $\Gamma$ = cos $\phi$")
fig.savefig("../../plot_data/" + project + parameter_dir + "/total/kg.pdf")
fig.show()
plt.close()

#file  kga_finite.py
#author Ahmed Rayyan
#date   March 15, 2020
#brief  explores finite size effects in the KG + a phase PhaseDiagram

import os
from lib_new_parse import ExtractEnergyFromFile
import matplotlib.pyplot as plt
import numpy as np
# plt.style.use('dark_background')

s = 4
l2 = 2
Tfp = 300
c = 2

v = 0

g = 0.5
p = 0.12
a = 0.00

l1list = 2*np.array([1,2,3,4,5,6,7,8,9,10,11,12])
elist = []
for l1 in l1list:
    data_dir = "../../raw_data/kg/gk=gg_ak=ag/%i_%i_%i/c_%i/Tfp_%i/g_%.3f_p_%.3f/v_%i/a_%.3f_.out"%(s,l1,l2, c,Tfp, g,p,v,a)
    print(data_dir)
    assert(os.path.exists(data_dir)),"Your data directory does not exist."

    # plot_dir = "../../plot_data/kg/gk=gg_ak=ag/%i_%i_%i/c_%i/Tfp_%i/g_%.3f_p_%.3f/v_%i/"%(s, l1, l2,c, Tfp,g,p,v)
    # if not os.path.exists(plot_dir):
    #     os.makedirs(plot_dir)

    e = ExtractEnergyFromFile(data_dir)
    elist.append(e)

    print(l1)
    print(e)

lowest_min = np.min(elist)
position = np.where(elist==lowest_min)
print(l1list[position])
print(np.array(elist)[position])

plt.scatter(1./(l1list*l2*s),elist)
plt.show()
plt.close()

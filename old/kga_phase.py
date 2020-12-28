#file   kga_energy.py
#author Ahmed Rayyan
#date   June 29, 2020
#updated June 29, 2020
#brief  constructing KG + A phase diagram

import glob as glb
from lib_new_parse import ExtractEnergyFromFolder, AnisotropySweep, pi, PhaseDiagram
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('dark_background')

s = 4
l1 = 12
l2 = 6
c = 2
Tfp = 200
g = 0.5

v = 0
data_dir = "../../raw_data/kg/gk=gg_ak=ag/%i_%i_%i/c_%i/Tfp_%i/"%(s, l1, l2,c, Tfp)
assert(os.path.exists(data_dir)),"Your data directory does not exist."

plot_dir = "../../plot_data/kg/gk=gg_ak=ag/%i_%i_%i/c_%i/Tfp_%i/"%(s, l1, l2,c, Tfp)

folder_list = glb.glob(data_dir+"g_0.500_p_*/v_0/")
folder_list.sort()

p_list = []
sweep_list = []

for folder in folder_list:
    g = float(folder.replace("/","_").split("_")[-6])
    p = float(folder.replace("/","_").split("_")[-4])
    a_list, e_list = ExtractEnergyFromFolder(folder, -2)
    sweep = AnisotropySweep("g",g,a_list,e_list)

    p_list.append(p)
    sweep_list.append(sweep)

phase_space = PhaseDiagram(p_list, sweep_list)
fig = phase_space.PlotPeaks(0.00000,0.00000)
ax = fig.axes[0]
ax.set_xlim(0,1)
ax.set_ylim(-0.5,1)

ax.set_xlabel(r"$\phi/\pi$", rotation="horizontal", fontsize=12, labelpad=12)
ax.set_ylabel(r"$\alpha$", rotation="horizontal", fontsize=12, labelpad=16)
ax.legend(loc="lower right", markerscale=1, fontsize=12)

fig.tight_layout(rect=[0,0.03,1,0.95])

# fig.savefig(plot_dir+ "/phase_diagrammm_v_%i.pdf"%v)
plt.show()
plt.close()

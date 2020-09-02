#file   anis_parse_energy.py
#author Ahmed Rayyan
#date   December 17, 2019
#brief  parses through data for analysis visualization
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.tri as tri
import sys

from lib_parse import ExtractEnergyFromFile, KG_ASweep, pi, Couplings, RotateC3

project = "kg_a"
l=12
# for p in [0, 0.01,0.08,0.15,0.25,0.352,0.49,0.5]:
p=0
version = 1
parameter_dir = "/l_%i/p_%.3f/v_%i/"%(l,p,version)
data_dir = "../../raw_data/" + project + parameter_dir
assert(os.path.exists(data_dir)),"Your data_dir does not exist. Go make it or point to its proper location."
K0 = math.sin(p*pi)
G0 = math.cos(p*pi)

sweep_list = []
fixed_var = "g"
for fixed_val in np.linspace(0, 0.5, 10+1):
# fixed_val = 0.25
    file_list = glob.glob(data_dir+"*%s_%.3f*"%(fixed_var,fixed_val))
    file_list.sort()

    swept_var_list, e_list = [], []
    for file in file_list:
        if fixed_var == "g":
            swept_var = float(file.split("_")[-2])
        elif fixed_var == "a":
            swept_var = float(file.split("_")[-4])

        swept_var_list.append(swept_var)

        e = ExtractEnergyFromFile(file)
        e_list.append(e)

    idx = np.argsort(swept_var_list)
    swept_var_list = np.array(swept_var_list)[idx]
    e_list = np.array(e_list)[idx]

    sweep_list.append(KG_ASweep(fixed_var, fixed_val, swept_var_list, e_list))
for sweep in sweep_list:
    fig = sweep.PlotSweep()
    # fig.suptitle(r"($\phi/\pi,\, %s)= (%.3f, %.3f)$"%(fixed_var, p, sweep.FixedVal))
    plot_dir = "../../plot_data/" + project + parameter_dir
    # fig.savefig(plot_dir + "/%s_%.3f-%i.pdf"%(fixed_var, fixed_val, version))
    # fig.show()
    plt.close()
# -----------------------Anisotropic Phase Diagram--------------------
Gx_list, Gy_list, Gz_list, full_e_list = [], [], [], []
for sweep in sweep_list:
    for a, e in zip(sweep.SweptParList,sweep.EList):
        Gx, Gy, Gz = Couplings(+1, 1/3, sweep.FixedVal, a)
        full_e_list.append(e/3)
        Gx_list.append(Gx)
        Gy_list.append(Gy)
        Gz_list.append(Gz)

a_lst, b_lst, c_lst, v_lst = RotateC3(Gx_list, Gy_list, Gz_list, full_e_list)
# a_lst, b_lst, c_lst, v_lst = np.array(Gx_list), np.array(Gy_list), np.array(Gz_list), np.array(full_e_list)

energy_x = 0.5 * ( 2.*b_lst+c_lst )
energy_y = 0.5*np.sqrt(3) * c_lst
T = tri.Triangulation(energy_x,energy_y)

fig, ax1 = plt.subplots()
c = ax1.tripcolor(energy_x,energy_y,T.triangles,v_lst,cmap="viridis",shading="flat")
cbar = plt.colorbar(c,fraction=0.046, pad=0.04)
cbar.ax.set_title(r'$E_0/N$')
ax1.set_aspect("equal")
ax1.axis("off")

# corner_labels = [(r"$|K_x|=1$", (-0.03,-0.02)), (r"$|K_y|=1$", (1.00/3-0.04,-0.02)), (r"$|K_z|=1$",(0.5/3-0.04,np.sqrt(3)/2/3+0.01))]
corner_labels = [(r"$\Gamma_x=1$", (-0.03,-0.02)), (r"$\Gamma_y=1$", (1.00/3-0.04,-0.02)), (r"$\Gamma_z=1$",(0.5/3-0.04,np.sqrt(3)/2/3+0.01))]
for corner_label in corner_labels:
    ax1.annotate(*corner_label, size="large")

ax1.set_xlim(-0.03, 1.2/2.5)
ax1.set_ylim(-0.03, 0.3)

# fig.suptitle(r"$(J,\, K,\, \Gamma',\, h,\, h_\theta,\, h_\phi) = (%.2f,%.2f,%.2f,%.2f,%.0f,%.0f)$"%(j,k,gp,h,ht,hp))
fig.suptitle(r"$\Gamma_x + \Gamma_y + \Gamma_z = 3$")
# fig.suptitle(r"$|K_x| + |K_y| + |K_z| = 1$")
plot_dir = "../../plot_data/" + project + parameter_dir
fig.savefig(plot_dir+ "/phase_triangle-%i.pdf"%(version))

fig.show()
plt.close()

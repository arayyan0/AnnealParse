#file   kga_energy.py
#author Ahmed Rayyan
#date   March 15, 2020
#updated June 19, 2020
#brief  energy analysis of KG + A phase diagram

import glob
import os
from lib_new_parse import ExtractEnergyFromFile, PhiSweep, AnisotropySweep, pi, PhaseDiagram
import matplotlib.pyplot as plt
import warnings
import numpy as np
from math import tan
import pickle
from scipy.signal import peak_prominences
warnings.filterwarnings('ignore')

plt.style.use('dark_background')

s = 4
l1 = 10
l2 = 2
Tfp = 200
c = 2
g = 0.5

# for v in [0]:
#     a_peak_list_list, p_peak_list_list, f_peak_list_list = [], [], []
#     for a in [-0.5,-0.4,-0.3,-0.2,-0.1,0]:
#
#         data_dir = "../../raw_data/kg/gk=gg_ak=ag/%i_%i_%i/c_%i/Tfp_%i/g_%.3f_a_%.3f/v_%i/"%(s, l1, l2, c, Tfp,g,a,v)
#         assert(os.path.exists(data_dir)),"Your data directory does not exist."
#
#         plot_dir = "../../plot_data/kg/gk=gg_ak=ag/%i_%i_%i/c_%i/Tfp_%i/g_%.3f_a_%.3f/v_%i/"%(s, l1, l2, c, Tfp,g,a,v)
#         if not os.path.exists(plot_dir):
#             os.makedirs(plot_dir)
#
#         file_list = glob.glob(data_dir+"/*")
#         file_list.sort()
#
#         p_list, e_list = [], []
#         for file in file_list:
#             #----------------------Read the spins in from file--------------------------
#             p = float(file.split("_")[-2])
#             with open(file, 'r') as f:
#                 file_data = f.readlines()
#             e = ExtractEnergyFromFile(file)
#             p_list.append(p)
#             e_list.append(e)
#
#         idx = np.argsort(p_list)
#         p_list = np.array(p_list)[idx]
#         e_list = np.array(e_list)[idx]
#
#         sweep = PhiSweep(p_list,e_list)
#         # --------------- plot 1D phase diagram --------------
#         # fig = sweep.PlotLabeledSweep()
#         # fig.savefig(plot_dir+ "/pd.pdf")
#         # plt.show()
#         # plt.close()
#
#         # --------------- add peaks --------------
#         p_peak_list, f_peak_list, f_prominences = sweep.PseudoSusceptibilityPeaks(0.04)
#
#         a_peak_list_list.append(a*np.ones(len(p_peak_list)))
#         p_peak_list_list.append(p_peak_list)
#         f_peak_list_list.append(f_peak_list)
#
#     pickled_data_dir = "../../pickled_data/kg/gk=gg_ak=ag/%i_%i_%i/c_%i/Tfp_%i/g_%.3f/v_%i/"%(s, l1, l2,c,Tfp,g,v)
#     if not os.path.exists(pickled_data_dir):
#         os.makedirs(pickled_data_dir)
#
#     with open(pickled_data_dir+"/chi_p_peaks.pickle", "wb") as newf:
#         pickle.dump([a_peak_list_list,p_peak_list_list,f_peak_list_list], newf)

for v in [0,1,2,3,4]:#,1,2,3,4,5,6,7]:
    p_peak_list_list, a_peak_list_list, f_peak_list_list = [], [], []
    sweeplst = []
    plst=[]
    # for p in [0,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.2,0.225,0.25,0.275,0.3,
    #           0.35,0.4,0.45,0.48,0.5,0.525,0.55,0.6,0.65,0.7,0.75,0.775,0.8,0.825,
    #           0.85,0.875,0.9,0.925,0.950,0.975,1]:
    for p in [0.08,0.176]:
        data_dir = "../../raw_data/kg/gk=gg_ak=ag/%i_%i_%i/c_%i/Tfp_%i/g_%.3f_p_%.3f/v_%i/"%(s, l1, l2,c,Tfp,g,p,v)
        print(data_dir)
        assert(os.path.exists(data_dir)),"Your data directory does not exist."

        plot_dir = "../../plot_data/kg/gk=gg_ak=ag/%i_%i_%i/c_%i/Tfp_%i/g_%.3f_p_%.3f/v_%i/"%(s, l1, l2,c, Tfp,g,p,v)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        file_list = glob.glob(data_dir+"/*")
        file_list.sort()

        a_list, e_list = [], []
        for file in file_list:
            a = float(file.split("_")[-2])
            with open(file, 'r') as f:
                file_data = f.readlines()
            e = ExtractEnergyFromFile(file)
            a_list.append(a)
            e_list.append(e)

        idx = np.argsort(a_list)
        a_list = np.array(a_list)[idx]
        e_list = np.array(e_list)[idx]

        sweep = AnisotropySweep("g",g,a_list,e_list)
        plst.append(p)
        sweeplst.append(sweep)

        # --------------- plot 1D phase diagram --------------
        fig = sweep.PlotLabeledSweep()
        fig.suptitle(r"$\phi/\pi=%.3f \quad \Gamma/|K| = %.3f$"%(p,np.tan(pi*p)))
        fig.savefig(plot_dir+ "/pd.pdf")
        plt.show()
        plt.close()

        # # --------------- add peaks --------------
        # a_peak_list, f_peak_list, f_prominences = sweep.PseudoSusceptibilityPeaks(0.07)
        #
        # p_peak_list_list.append(p*np.ones(len(a_peak_list)))
        # a_peak_list_list.append(a_peak_list)
        # f_peak_list_list.append(f_peak_list)

    # if v == 0:
    # #     #-------------------------pickling peak data
    #     pickled_data_dir = "../../pickled_data/kg/gk=gg_ak=ag/%i_%i_%i/c_%i/Tfp_%i/g_%.3f/v_%i/"%(s, l1, l2,c,Tfp,g,v)
    #     if not os.path.exists(pickled_data_dir):
    #         os.makedirs(pickled_data_dir)

        # print(a_peak_list_list)
        # with open(pickled_data_dir+"/chi_a_peaks.pickle", "wb") as newf:
        #     pickle.dump([p_peak_list_list,a_peak_list_list,f_peak_list_list,plst,sweeplst], newf)

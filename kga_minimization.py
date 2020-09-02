#file   kga_minimization.py
#author Ahmed Rayyan
#date   March 16, 2020
#brief  finds minimum energy within KG + A phase diagram

import os
import numpy as np
from lib_new_parse import ExtractEnergyFromFile
from shutil import copyfile

s = 2
l1 = 3
l2 = 3
Tfp = 300
g = 0.5
clust = 2
versions = 80
#--------------
# a = -0.5
# data_dir = "../../raw_data/kg/gk=gg_ak=ag/%i_%i_%i/c_%i/Tfp_%i/g_%.3f_a_%.3f"%(s,l1,l2,c,Tfp,g,a)
# print(data_dir)
# assert(os.path.exists(data_dir)),"Your data directory does not exist."
#
# plist = np.linspace(0, 0.5, 25+1)
#
# for p in plist:
#     version_Elist = []
#     for v in range(1,versions+1):
#         file = data_dir + "/v_%i/p_%.3f_.out"%(v, p)
#         e = ExtractEnergyFromFile(file)
#         version_Elist.append(e)
#     which_version = version_Elist.index(min(version_Elist))+1
#     print("%.2f"%p, which_version, max(version_Elist)- min(version_Elist), [x - min(version_Elist) for x in version_Elist])
#     if not os.path.exists(data_dir+"/v_0"):
#         os.makedirs(data_dir+"/v_0")
#     copyfile(data_dir + "/v_%i/p_%.3f_.out"%(which_version, p), data_dir+"/v_0/p_%.3f_.out"%p)
#--------------
for p in [0.25]:
    data_dir = "../../raw_data/kg/gk=gg_ak=ag/%i_%i_%i/c_%i/Tfp_%i/g_%.3f_p_%.3f"%(s,l1,l2,clust,Tfp,g,p)
    print(data_dir)
    assert(os.path.exists(data_dir)),"Your data directory does not exist."

    # alist = np.linspace(-0.5, 1, 30+1)
    # alist = np.linspace(-0.03,0.21,80+1)
    # alist = np.linspace(0,0.4,20+1)
    # alist = np.array([0, 0.1,0.45,0.6])
    alist = np.array([0.999])

    for a in alist:
        version_Elist = []
        for v in range(1,versions+1):
            try:
                file = data_dir + "/v_%i/a_%.3f_.out"%(v, a)
                try:
                    e = ExtractEnergyFromFile(file)
                except:
                    print("file empty. deleted.")
                    e = np.nan
                    os.remove(file)
            except:
                print("file does not exist")
                e=np.nan
                continue
            version_Elist.append(e)
        which_version = version_Elist.index(min(version_Elist))+1
        print("%.3f"%a, which_version, max(version_Elist)- min(version_Elist), [x - min(version_Elist) for x in version_Elist])
        if not os.path.exists(data_dir+"/v_0"):
            os.makedirs(data_dir+"/v_0")
        try:
            copyfile(data_dir + "/v_%i/a_%.3f_.out"%(which_version, a), data_dir+"/v_0/a_%.3f_.out"%a)
        except:
            print("file does not exist")

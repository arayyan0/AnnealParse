#file   which_order_minimization.py
#author Ahmed Rayyan
#date   February 26, 2019
#brief  see Feb 26 2020 notes

import os
import numpy as np
from lib_parse import ExtractEnergyFromFile
from shutil import copyfile

project = "which_order_g_dom"
g = 0.5
orders = [4,6,12,18,30,50]
#Part I
for order in orders:
    data_dir = "../../raw_data/" + project + "/g_%.3f"%g
    if not os.path.exists(data_dir + "/o_%i/v_0"%order):
        os.makedirs(data_dir + "/o_%i/v_0"%order)

    alist = np.linspace(0,  0.4, 10+1)
    plist = np.linspace(0, 0.26, 13+1)

    for p in plist:
        for a in alist:
            version_Elist = []
            for version in range(1,10+1):
                file= data_dir+"/o_%i/v_%i/p_%.3f_a_%.3f_.out"%(order,version,p,a)
                e = ExtractEnergyFromFile(file)
                version_Elist.append(e)
            which_version = version_Elist.index(min(version_Elist))+1
            copyfile(file, data_dir+"/o_%i/v_0/p_%.3f_a_%.3f_.out"%(order,p,a))
#Part II
if not os.path.exists(data_dir + "/total"):
    os.makedirs(data_dir + "/total")
for p in plist:
    for a in alist:
        order_Elist = []
        for order in orders:
            file = data_dir + "/o_%i/v_0/p_%.3f_a_%.3f_.out"%(order,p,a)
            e = ExtractEnergyFromFile(file)
            order_Elist.append(e)
        print("p=%.3f, a=%.3f"%(p,a))
        print(order_Elist)
        m=min(order_Elist)
        indices = [i for i, x in enumerate(order_Elist) if x == m]
        which_orders = [orders[i] for i in indices]
        print(which_orders)
        print(m)
        print("......")
#Part III

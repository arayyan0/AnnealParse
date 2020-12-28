#file   kgh_minimization.py
#author Ahmed Rayyan
#date   June 19, 2020
#brief  finds optimum spin config within several runs of KG + h phase diagram

import os
import numpy as np
from lib_new_parse import ExtractEnergyFromFile
from shutil import copyfile

s=4
l1=15
l2=9

hth = 5
gp = -0.02
p = 0.148
a = 0

versions = 7
hlist = np.linspace(0, 1.5, 30+1)

data_dir = "../../raw_data/kgh/%i_%i_%i/hth_%i_gp_%.3f/p_%.3f_a_%.3f"%(s, l1, l2, hth, gp, p, a)
print(data_dir)
assert(os.path.exists(data_dir)),"Your data directory does not exist."

for h in hlist:
    version_Elist = []
    for v in range(1,versions+1):
        try:
            file = data_dir + "/v_%i/h_%.3f_.out"%(v, h)
            try:
                e = ExtractEnergyFromFile(file)
            except:
                print("file empty. deleted.")
                e=np.nan
                os.remove(file)
        except:
            print("file does not exist.")
            e=np.nan
            continue
        version_Elist.append(e)
    which_version = version_Elist.index(min(version_Elist))+1
    print("%.3f"%h, which_version, max(version_Elist)- min(version_Elist), [x - min(version_Elist) for x in version_Elist])
    if not os.path.exists(data_dir+"/v_0"):
        os.makedirs(data_dir+"/v_0")
    try:
        copyfile(data_dir + "/v_%i/h_%.3f_.out"%(which_version, h), data_dir+"/v_0/h_%.3f_.out"%h)
    except:
        print("file does not exist")

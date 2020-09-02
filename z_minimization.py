#file   z_minimization.py
#author Ahmed Rayyan
#date   June 19, 2020
#brief  finds optimum spin config within several runs of z sweep

import os
import numpy as np
from lib_new_parse import ExtractEnergyFromFile
from shutil import copyfile

s=4
l1=12
l2=6
z=-0.5

versions = 4
plist = np.linspace(0, 0.5, 25+1)

data_dir = "../../raw_data/z/%i_%i_%i/z_%.3f"%(s,l1,l2,z)
print(data_dir)
assert(os.path.exists(data_dir)),"Your data directory does not exist."

for p in plist:
    version_Elist = []
    for v in range(1,versions+1):
        try:
            file = data_dir + "/v_%i/p_%.3f_.out"%(v, p)
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
    print("%.3f"%p, which_version, max(version_Elist)- min(version_Elist), [x - min(version_Elist) for x in version_Elist])
    if not os.path.exists(data_dir+"/v_0"):
        os.makedirs(data_dir+"/v_0")
    try:
        copyfile(data_dir + "/v_%i/p_%.3f_.out"%(which_version, p), data_dir+"/v_0/p_%.3f_.out"%p)
    except:
        print("file does not exist")

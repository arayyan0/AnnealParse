import numpy as np
import matplotlib.pyplot as plt
from lswt_lib import LSWT
from common import SymmetryPoints, pi
import sys

argv = sys.argv
input_data_filename = argv[1]
output_data_filename = argv[2]

##----Extract file information
lswtea = LSWT(input_data_filename)
print(lswtea.LSWTEnergyDensity)

# k = [np.array([-0.1,-0.4]),np.array([0.1, 0.4])]

# lswtea.ObtainMagnonSpectrumAndDiagonalizer(k, 0, 0)

##----Calculate moment reduction
n=1
lswtea.CalculateMagnonProperties(2*3*n, 2*3*n)
print(lswtea.ReducedMoment)
##----Plot the lowest magnon band
# n=3
# fig = lswtea.PlotLowestBand(n*2*3,n*2*3,2,1, 0)
# plt.savefig(f'lmb_{output_data_filename}.pdf')
# plt.show()
# plt.close()

##----Calculate magnon band structure
# kp = SymmetryPoints().MakeKPath(["G","M1","K","G"],50)
# kp = SymmetryPoints().MakeKPath(["X","G","M2","Gp1","M1","G"],50)
# fig = lswtea.PlotMagnonKPath(kp)
# # # fig.axes[0].set_ylim(0,0.01)
# fig.tight_layout(rect=[0,0.03,1,0.95])
# plt.savefig(f'bands_{output_data_filename}.pdf')
# plt.show()
# plt.close()

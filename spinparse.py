# import numpy as np
# import matplotlib as mpl
# mpl.use('tkagg')
import matplotlib.pyplot as plt
from newspin_lib import MonteCarloOutput, MuonSimulation
import sys

##-------------------------------------------------Select input/output filenames
argv = sys.argv
input_data_filename = argv[1]
output_data_filename = argv[2]
print(input_data_filename)

##------------------------------------------------------Extract file information
spinstuff = MonteCarloOutput(input_data_filename)

##---------------------------------------------------------------Return MC energy
print(spinstuff.MCEnergyPerSite)

##----------------------------------------------------Plot 3D spin configuration
# fig = spinstuff.Plot3D()
# plt.savefig(f'3d_spins_{output_data_filename}.pdf')
# plt.show()
# plt.close()
##----------------------------------------------Plot Az layer spin configuration
plane = -1 #-1 for Az plane with most moments, >=0 otherwise

if spinstuff.NumSites > 500:
    quiver_options = [11*35, 2.5, 2]       #[scale, minlength, headwidth]
else:
    quiver_options = [0.5*35, 2.5, 2]
cm = 'seismic'
cb_options = [0.04, 'horizontal', cm,'x'] #[fraction, orientation, colormap, axis]
usetex=False
for plane in range(len(spinstuff.LayerNumber)):
    spinstuff.Plot2DPlane(plane, quiver_options, cb_options, usetex)
    plt.savefig(f'2d_spins_{output_data_filename}_plane_{plane}.pdf')
    plt.show()
    plt.close()

##---------------------------------------------------------------muSR simulation
# muon_simulation = MuonSimulation(input_data_filename)
# fig = muon_simulation.PlotFieldDistribution()
# plt.savefig(f'field_{output_data_filename}.pdf')
# plt.show()
# plt.close()

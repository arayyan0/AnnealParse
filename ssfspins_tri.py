import numpy as np
import matplotlib.pyplot as plt
from spin_lib import AnnealedSpinConfigurationTriangular
import sys

##--------Select input/output filenames
argv = sys.argv
input_data_filename = argv[1]
output_data_filename = argv[2]
print(input_data_filename)

##--------Extract file information
spinstuff = AnnealedSpinConfigurationTriangular(input_data_filename)

##--------Return MC energy
print(spinstuff.MCEnergyDensity)


##--------Plot spin configuration
quiver_options = [0.4*35, 2.5, 3.5]       #[scale, minlength, headwidth]

cm = 'gnuplot'
cb_options = [0.04, 'horizontal', cm,'x']    #[fraction, orientation, colormap]
usetex=False
fig = spinstuff.PlotSpins(quiver_options, cb_options, usetex)
plt.savefig(f'spins_{output_data_filename}.pdf')
plt.close()

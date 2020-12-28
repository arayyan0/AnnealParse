import numpy as np
import matplotlib.pyplot as plt
from spin_lib import AnnealedSpinConfiguration
import sys

##--------Select input/output filenames
argv = sys.argv
input_data_filename = argv[1]
output_data_filename = argv[2]

##--------Extract file information
spinstuff = AnnealedSpinConfiguration(input_data_filename)

##--------Return MC energy
print(spinstuff.MCEnergyDensity)

##----Construct and Plot SSF
fig = spinstuff.CalculateAndPlotSSF()
fig.axes[0].set_facecolor('black')
plt.savefig(f'ssf_{output_data_filename}.pdf')
plt.show()
plt.close()

##--------Plot spin configuration
# # Good for 4-site cluster
# quiver_options = [7.5, 1, 4]       #[scale, minlength, headwidth]

# # Good for 18-site cluster
quiver_options = [20, 1.5, 4]       #[scale, minlength, headwidth]

cb_options = [0.04, 'vertical']    #[fraction, orientation]
fig = spinstuff.PlotSpins(quiver_options, cb_options)
# plt.savefig(f'spins_{output_data_filename}.pdf')
plt.show()
plt.close()

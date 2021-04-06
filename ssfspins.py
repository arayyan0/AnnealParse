import numpy as np
import matplotlib.pyplot as plt
from spin_lib import AnnealedSpinConfiguration
import sys

##--------Select input/output filenames
argv = sys.argv
input_data_filename = argv[1]
output_data_filename = argv[2]
print(input_data_filename)

##--------Extract file information
spinstuff = AnnealedSpinConfiguration(input_data_filename)

##--------Return MC energy
print(spinstuff.MCEnergyDensity)

# ##--------Construct and Plot SSF
cb_options =[0.04, 'vertical', 'terrain']    #[fraction, orientation, colormap]
fig = spinstuff.CalculateAndPlotSSF(cb_options)
fig.axes[0].set_facecolor('black')
plt.savefig(f'ssf_{output_data_filename}.pdf')
plt.show()
plt.close()
#
##--------Plot spin configuration
# quiver_options = [10, 1, 4]       #[scale, minlength, headwidth] 4-site
quiver_options = [12, 3, 4]       #[scale, minlength, headwidth] 18-site

cb_options = [0.04, 'horizontal', 'winter']    #[fraction, orientation, colormap]
signstructure = True
fig = spinstuff.PlotSpins(quiver_options, cb_options, signstructure)
plt.savefig(f'spins_{output_data_filename}.pdf')
plt.show()
plt.close()
#
# np.set_printoptions(precision=6,suppress=True)
# spinstuff.ExtractMomentsAndPositions()
# print(spinstuff.SpinsABC)

##--------Plot spin configuration
spinstuff.ExtractMomentsAndPositions()

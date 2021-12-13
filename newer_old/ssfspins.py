import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
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
cb_options =[0.04, 'vertical', 'binary']    #[fraction, orientation, colormap]
usetex=True
fig = spinstuff.CalculateAndPlotSSF(cb_options,usetex)
# fig.axes[0].set_facecolor('black')
plt.savefig(f'ssf_{output_data_filename}.pdf')
plt.show()
plt.close()
#
##--------Plot spin configuration
# quiver_options = [10, 1, 4]       #[scale, minlength, headwidth] 4-site
quiver_options = [1.2*35, 0.5, 3.5]       #[scale, minlength, headwidth] 18-site

# for cm in ['viridis', 'plasma','inferno','magma','cividis']:
cm = 'gnuplot'
cb_options = [0.04, 'horizontal', cm,'x']    #[fraction, orientation, colormap]
plaquettes = False
signstructure = False
usetex=False
fig = spinstuff.PlotSpins(quiver_options, cb_options, plaquettes,signstructure,usetex)
plt.savefig(f'spins_{output_data_filename}.pdf')
plt.savefig(f'{cm}.pdf')
plt.show()
plt.close()
#
# np.set_printoptions(precision=6,suppress=True)
# spinstuff.ExtractMomentsAndPositions()
# print(spinstuff.SpinsABC)

##--------Plot spin configuration
# spinstuff.ExtractMomentsAndPositions()

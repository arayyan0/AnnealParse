import numpy as np
import matplotlib
matplotlib.use('tkagg')
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

if spinstuff.Sites > 500:
    quiver_options = [1.9*35, 2.5, 2]       #[scale, minlength, headwidth]
else:
    quiver_options = [0.5*35, 2.5, 2]       #[scale, minlength, headwidth]

cm = 'seismic'
cb_options = [0.04, 'horizontal', cm,'x']    #[fraction, orientation, colormap]
usetex=False
fig = spinstuff.PlotSpins(quiver_options, cb_options, usetex)
plt.savefig(f'spins_{output_data_filename}.pdf')
# plt.show()
plt.close()

##----------Plot Dipolar FIeld

# fig = spinstuff.PlotDipolarField()
# plt.savefig(f'dipolar_{output_data_filename}.pdf')
# plt.show()
# plt.close()

# ##--------Construct and Plot SSF
cb_options =[0.04, 'vertical', 'gist_yarg']    #[fraction, orientation, colormap]
usetex=False
fig = spinstuff.CalculateAndPlotSSF(cb_options,usetex)
# fig.axes[0].set_facecolor('black')
plt.savefig(f'ssf_{output_data_filename}.pdf')
# plt.show()
plt.close()

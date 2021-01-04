import glob as glob
from lswt_lib import LSWT
from common import pi, SymmetryPoints, PsiSweep
import matplotlib.pyplot as plt
import numpy as np
import os

# generate filenames
run = 4

for version in range(0,1):
    data_folder = f'out/jobrun_{run}/v_{version}/'
    plot_folder = data_folder+f'plots/'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    file_lst = glob.glob(data_folder+'*.out')

    plst, alst, elst, momentlst = [], [], [], []

    for file in file_lst:
        split_file = file.replace("/","_").split("_")
        p, a = float(split_file[-4]), float(split_file[-2])
        # print(p,a)
        # spinstuff = AnnealedSpinConfiguration()
        spinstuff = LSWT(file)  #use when spin cluster is ONE magnetic unit cell

        # plst.append(p)
        # alst.append(a)
        # elst.append(spinstuff.MCEnergyDensity)

        # n=3
        # spinstuff.CalculateMagnonProperties(2*3*n, 2*3*n)
        # momentlst.append(spinstuff.ReducedMoment)
        # print(spinstuff.ReducedMoment)
        lift=0.000
        n=2
        fig = spinstuff.PlotLowestBand(n*2*3,n*2*3,2,1, lift)
        plt.savefig(plot_folder+f'lmb_p_{p:.3f}_a_{a:.3f}.pdf')
        # plt.show()
        plt.close()

        fig = spinstuff.CalculateAndPlotSSF()
        fig.axes[0].set_facecolor('black')
        plt.savefig(plot_folder+f'ssf_p_{p:.3f}_a_{a:.3f}.pdf')
        # plt.show()
        plt.close()

        quiver_options = [20, 1.5, 4]       #[scale, minlength, headwidth] 18-site
        cb_options = [0.04, 'vertical', 'coolwarm']    #[fraction, orientation, colormap]
        fig = spinstuff.PlotSpins(quiver_options, cb_options)
        plt.savefig(plot_folder+f'spins_p_{p:.3f}_a_{a:.3f}.pdf')
        # plt.show()
        plt.close()

        kp = SymmetryPoints().MakeKPath(["X","G","M2","Gp1","M1","G"],50)
        fig = spinstuff.PlotMagnonKPath(kp,lift)
        # # fig.axes[0].set_ylim(0,0.01)
        fig.tight_layout(rect=[0,0.03,1,0.95])
        plt.savefig(plot_folder+f'bands_p_{p:.3f}_a_{a:.3f}.pdf')
        # plt.show()
        plt.close()


    # parray, aarray, earray, momentlst = map(np.array, [plst,alst,elst,momentlst])
    #########-------------------MUST CHANGE WHEN GOING ALONG A......
    # idx = np.argsort(parray)
    # parray, aarray, earray, momentlst = parray[idx], aarray[idx], earray[idx], momentlst[idx]
    #
    # plt.plot(parray[2:], momentlst[2:])
    # plt.show()
    # plt.close()
    #
    # gih = PsiSweep(parray, earray)
    # fig = gih.PlotLabeledSweep()
    # fig.tight_layout(rect=[0,0.03,1,0.95])
    # plt.show()
    # plt.close()

    # fig = AnisotropySweep('a', a[0], parray, earray).PlotLabeledSweep()

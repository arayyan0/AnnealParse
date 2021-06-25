import numpy as np
import matplotlib.pyplot as plt
from spin_lib import AnnealedSpinConfigurationTriangular
import os
import glob as glob

L=36
which = f'jtau_ising_defect_l={L}'

for number in range(7,7+1):
    run = f'{number}'

    for version in range(1,1+1):
        data_folder = f'out/{which}/jobrun_{run}/v_{version}/'
        print(data_folder)
        plot_folder = data_folder + f'plots/'
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

        file_lst = glob.glob(data_folder+'*.out')

        Tlst = []
        fluclist = []
        OPlist = []
        binderlist = []

        for file in file_lst:
            split_file = file.replace('/','_').split('_')
            T = float(split_file[-2])
            spinstuff = AnnealedSpinConfigurationTriangular(file)

            Tlst.append(T)

            fluclist.append([spinstuff.SpecificHeat, spinstuff.SpecificHeatError,
                spinstuff.FMSusceptibility, spinstuff.PerpSusceptibility,
                spinstuff.ParSusceptibility, spinstuff.CombinedSusceptibility
            ])

            OPlist.append([spinstuff.FMNorm,spinstuff.PerpNorm,spinstuff.ParNorm,spinstuff.CombinedNorm])

            binderlist.append([spinstuff.FMBinder, spinstuff.PerpBinder, spinstuff.ParBinder,spinstuff.CombinedBinder])

            if (number == 7):
                # quiver_options = [0.6*35, 1.5, 3.5]       #[scale, minlength, headwidth]
                quiver_options = [1.4*35, 1.5, 3.5]       #[scale, minlength, headwidth]
                cm = 'gnuplot'
                cb_options = [0.04, 'horizontal', cm,'x']    #[fraction, orientation, colormap]
                usetex=False
                fig = spinstuff.PlotSpins(quiver_options, cb_options, usetex)
                plt.savefig(plot_folder+f'spins_temp_{T}.pdf')
                plt.close()

        Tarray = np.array(Tlst)
        flucarray = np.array(fluclist)
        OParray = np.array(OPlist)
        binderarray = np.array(binderlist)

        idx = np.argsort(Tarray)
        Tarray = Tarray[idx]
        flucarray = flucarray[idx]
        OParray = OParray[idx]
        binderarray  = binderarray[idx]

        # print(flucarray.shape)
        # print(OParray.shape)
        # print(binderarray.shape)

        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)

        colors = ['blue', 'orange', 'green','purple']
        scale = 4
        OPscale = scale
        # ax1.plot(Tarray,OParray[:,0]*OPscale, '-o',label=fr'$\langle |m| \rangle\times{OPscale}$',color=colors[0])
        ax1.plot(Tarray,OParray[:,1], '-o', label=fr'$\langle |m^\bot| \rangle$',color=colors[1])
        ax1.plot(Tarray,OParray[:,2]*OPscale,'-o', label=fr'$\langle |m^\parallel| \rangle\times{OPscale}$',color=colors[2])
        ax1.plot(Tarray,OParray[:,3],'-o', label=fr'$\langle |M| \rangle\times{OPscale}$',color=colors[3])

        ax1.axhline(0,ls='--',color='gray')
        ax1.axhline(1,ls='--',color='gray')
        ax1.set_ylim(0,1)
        ax1.legend()

        flucscale = scale
        # ax3.plot(Tarray,flucarray[:,0], '-o',label=fr'$c_V$',color='green')
        # ax3.plot(Tarray,flucarray[:,2]*flucscale, '-o', label=rf'$\chi\times{flucscale}$',color=colors[0])
        ax3.plot(Tarray,flucarray[:,3], '-o', label=rf'$\chi^\bot$',color=colors[1])
        ax3.plot(Tarray,flucarray[:,4]*flucscale, '-o', label=rf'$\chi^\parallel\times{flucscale}$',color=colors[2])
        ax3.plot(Tarray,flucarray[:,5], '-o', label=rf'$X\times{flucscale}$',color=colors[3])
        ax3.axhline(0,ls='--',color='gray')
        ax3.legend()

        ax2.plot(Tarray,flucarray[:,0], '-o', label=r'$c_V$',color='red')
        ax2.legend()

        ax3.set_xlim(0,np.max(Tarray))
        ax3.set_xlabel('T')

        fig.tight_layout()
        plt.savefig(plot_folder+f'temp_obsv.pdf')
        plt.show()
        plt.close()

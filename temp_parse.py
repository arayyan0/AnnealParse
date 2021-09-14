import numpy as np
import matplotlib.pyplot as plt
from spin_lib import AnnealedSpinConfigurationTriangular
import os
import glob as glob

def PlotSpins(spinstuff,T,plot_folder):
    # quiver_options = [0.6*35, 1.5, 3.5]       #[scale, minlength, headwidth]
    quiver_options = [1.6*35, 1.5, 3.5]       #[scale, minlength, headwidth]
    cm = 'seismic'
    cb_options = [0.04, 'horizontal', cm,'x']    #[fraction, orientation, colormap]
    usetex=False
    fig = spinstuff.PlotSpins(quiver_options, cb_options, usetex)
    plt.savefig(plot_folder+f'spins_temp_{T}.pdf')
    plt.close()

def PlotSSF(spinstuff,T,plot_folder):
    cb_options =[0.04, 'vertical', 'gist_yarg']    #[fraction, orientation, colormap]
    usetex=False
    fig = spinstuff.CalculateAndPlotSSF(cb_options,usetex)
    # fig.axes[0].set_facecolor('black')
    plt.savefig(plot_folder+f'ssf_{T}.pdf')
    # plt.show()
    plt.close()

def PlotSweep(Tlst, fluclist,quadlist,octolist, spinstuff):
    Tarray = np.array(Tlst)
    flucarray = np.array(fluclist)
    quadarray = np.array([np.array(xi) for xi in quadlist ])
    octoarray = np.array([np.array(xi) for xi in octolist ])
    # print(quadarray)

    idx = np.argsort(Tarray)
    Tarray = Tarray[idx]
    flucarray = flucarray[idx]
    quadarray = quadarray[idx, :]
    octoarray = octoarray[idx, :]

    num_q = quadarray.shape[1]

    fig,(ax1,ax2, ax3) = plt.subplots(3,1,sharex=True)

    ax1.plot(Tarray,flucarray, '-o',color='purple')

    scale_q, scale_o = 4, 1
    colors = ['red', 'green', 'blue', 'black', 'orange', 'teal']
    for i in range(num_q):
        ax2.plot(Tarray,quadarray[:,i]*scale_q,
                    '-o',
                    # s=10,
                    color=colors[i],
                    # facecolors='none',
                    # linewidth=2,
                    label=spinstuff.QLabels[i])
        ax3.plot(Tarray,octoarray[:,i]*scale_o,
                    '-o',
                    # s=10,
                    color=colors[i],
                    # facecolors='none',
                    # linewidth=2,
                    label=spinstuff.QLabels[i])
    ax2.axhline(0,ls='--',color='gray')
    ax2.axhline(1,ls='--',color='gray')
    ax2.set_ylim(0,1)

    ax3.axhline(0,ls='--',color='gray')
    ax3.axhline(1,ls='--',color='gray')
    ax3.set_ylim(0,1)

    ax3.legend()

    ax1.set_ylabel(r'$C$'                                      , rotation=0, labelpad=30)
    ax2.set_ylabel(r'$%i \times m^{Quad}_\mathbf{k}$'%(scale_q), rotation=0, labelpad=30)
    ax3.set_ylabel(r'$%i \times m^{Octo}_\mathbf{k}$'%(scale_o), rotation=0, labelpad=30)

    ax3.set_xlabel(r'$T$')

    ax1.grid(axis='x')
    ax2.grid(axis='x')
    ax3.grid(axis='x')

    fig.tight_layout()
    plt.savefig(plot_folder+f'temp_obsv.pdf')
    plt.show()
    plt.close()


if __name__ == '__main__':
    # L=36
    which = f'refined-JO-dom-0908'
    # which = f'aQeffect-0908'
    scale = 3.5
    n = 2
    v = 2
    for number in range(1,1+1):
        run = f'{number}'

        for version in range(2,2+1):
            data_folder = f'out/{which}/jobrun_{run}/v_{version}/'
            print(data_folder)
            plot_folder = data_folder + f'plots/'
            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder)

            file_lst = glob.glob(data_folder+'*.out')

            Tlst = []
            fluclist = []
            quadlist = []
            octolist = []

            for file in file_lst:
                split_file = file.replace('/','_').split('_')
                T = float(split_file[-2])
                spinstuff = AnnealedSpinConfigurationTriangular(file)

                Tlst.append(T)
                fluclist.append(spinstuff.SpecificHeat)
                quadlist.append(spinstuff.QuadBar)
                octolist.append(spinstuff.OctoBar)

                # PlotSpins(spinstuff,T,plot_folder)
                # PlotSSF(spinstuff,T,plot_folder)

            PlotSweep(Tlst, fluclist,quadlist,octolist, spinstuff)

import glob as glob
from lswt_lib import LSWT
from spin_lib import AnnealedSpinConfiguration
from common import pi, SymmetryPoints, AnisotropySweep
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import pickle

# which = 'a=0'
# which = 'a=0.2'
# which = 'p=0.4'
which='denser24/zz'
# generate filenames
for number in range(8,8+1):
    run = f'{number}'
    # run = 'f'
    for version in range(0,0+1):
        data_folder = f'out/{which}/jobrun_{run}/v_{version}/'
        print(data_folder)
        plot_folder = data_folder+f'plots/'
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

        file_lst = glob.glob(data_folder+'*.out')

        plst, alst, elst = [], [], []
        momentlst, swecorrlst, magnongaplst, swelst = [], [], [], []
        Exlst, Eylst, Ezlst = [], [], []

        for file in file_lst:
            split_file = file.replace("/","_").split("_")
            p, a = float(split_file[-4]), float(split_file[-2])
            print(p, a)
            # spinstuff = AnnealedSpinConfiguration(file)
            spinstuff = LSWT(file)  #use when spin cluster is ONE magnetic unit cell

            plst.append(p)
            alst.append(a)
            elst.append(spinstuff.MCEnergyDensity)

            Exlst.append(spinstuff.Ex)
            Eylst.append(spinstuff.Ey)
            Ezlst.append(spinstuff.Ez)
            print(spinstuff.Ex, spinstuff.Ey,spinstuff.Ez)

            # n=3
            # spinstuff.CalculateMagnonProperties(2*3*n, 2*3*n)
            # momentlst.append(spinstuff.ReducedMoment)
            # swecorrlst.append(spinstuff.SWEnergyCorrection)
            # swelst.append(spinstuff.SWEnergy)
            # magnongaplst.append(spinstuff.MagnonGap)

            # lift=0.000
            # n=2
            # fig = spinstuff.PlotLowestBand(n*2*3,n*2*3,2,2, lift)
            # plt.savefig(plot_folder+f'lmb_p_{p:.3f}_a_{a:.3f}.pdf')
            # plt.show()
            # plt.close()

            cb_options =[0.04, 'vertical', 'gnuplot']
            usetex=False
            setticks=[False,1]
            fig = spinstuff.CalculateAndPlotSSF(cb_options, usetex,setticks)
            fig.axes[0].set_facecolor('black')
            plt.savefig(plot_folder+f'ssf_p_{p:.3f}_a_{a:.3f}.pdf')
            # plt.show()
            plt.close()
            #
            quiver_options = [20, 1.0, 4]       #[scale, minlength, headwidth] 18-site
            cb_options = [0.04, 'vertical', 'seismic']    #[fraction, orientation, colormap]
            signstructure =False
            usetex = False
            fig = spinstuff.PlotSpins(quiver_options, cb_options, signstructure, usetex)
            plt.savefig(plot_folder+f'spins_p_{p:.3f}_a_{a:.3f}.pdf')
            fig.tight_layout(rect=[0,0.03,1,0.95])
            # plt.show()
            plt.close()

            # kp = SymmetryPoints().MakeKPath(["X","G","M2","Gp1","M1","G"],25)
            # fig = spinstuff.PlotMagnonKPath(kp,lift)
            # # fig.axes[0].set_ylim(0,0.01)
            # fig.tight_layout(rect=[0,0.03,1,0.95])
            # plt.savefig(plot_folder+f'bands_p_{p:.3f}_a_{a:.3f}.pdf')
            # plt.show()
            # plt.close()

        parray, aarray, earray = map(np.array,[plst, alst, elst])

        #
        # momentlst, swecorrlst, magnongaplst, swelst = map(np.array,[momentlst, swecorrlst,magnongaplst,swelst])

        Exarray, Eyarray, Ezarray = map(np.array, [Exlst,Eylst,Ezlst])

        #---------------------------sweeping over a
        # idx = np.argsort(aarray)
        # parray, aarray = parray[idx], aarray[idx]
        # #
        # earray = earray[idx]
        # fig = AnisotropySweep('p', parray[0], aarray, earray).PlotLabeledSweep()
        # plt.savefig(plot_folder+f'energy.pdf')
        # # plt.show()
        # plt.close()
        #
        # Exarray, Eyarray, Ezarray = Exarray[idx], Eyarray[idx], Ezarray[idx]
        # Delta = (Exarray+Eyarray)/2 - Ezarray
        # fig, [ax1, ax2]=plt.subplots(2,1, sharex=True)
        # ax1.scatter(aarray, Delta/np.abs(earray), marker=".", facecolors='none', edgecolors='red')
        # ax2.scatter(aarray,np.gradient(Delta,aarray,edge_order=2),marker=".", facecolors='none', edgecolors='red')
        # ax1.grid(True)
        # ax2.grid(True)
        # print(repr(aarray), repr(Delta/np.abs(earray)), repr(np.gradient(Delta,aarray,edge_order=2)))
        # ax2.set_xlabel(r'$g$')
        # ax1.set_ylabel(r'$\Delta/|E|$')
        # ax2.set_ylabel(r'$\partial\Delta$')
        # fig.tight_layout(rect=[0,0.03,1,0.95])
        # plt.savefig(plot_folder+f'bond_energy.pdf')
        # # plt.show()
        # plt.close()
        #
        #
        # momentlst, swecorrlst = momentlst[idx], swecorrlst[idx]
        # magnongaplst = magnongaplst[idx]
        # swelst = swelst[idx]
        # fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True)
        #
        # ax1.scatter(aarray, swelst, marker=".", facecolors='none', edgecolors='r')
        # ax1.set_ylabel(r'$E_sw$')
        #
        # ax2.scatter(aarray, magnongaplst, marker=".", facecolors='none', edgecolors='black')
        # ax2.set_ylabel(r'$\Delta_0$')
        # # ax2.set_ylim(0,np.max(magnongaplst))
        #
        # ax3.scatter(aarray, momentlst, marker=".", facecolors='none', edgecolors='b')
        # ax3.set_ylabel(r'$m_{||}/S$')
        # ax3.set_ylim(-1,1)
        # #
        # ax3.set_xlabel(r'$g$')
        # fig.savefig(plot_folder+f'properties.pdf')
        # plt.show()
        # plt.close()
    # # ---------------------------sweeping over p
        idx = np.argsort(parray)
        parray, aarray = parray[idx], aarray[idx]

        earray = earray[idx]
        fig = AnisotropySweep('a', aarray[0], parray, earray).PlotLabeledSweep()
        fig.tight_layout(rect=[0,0.03,1,0.95])
        plt.savefig(plot_folder+f'energy.pdf')
        plt.show()
        plt.close()

        Exarray, Eyarray, Ezarray = Exarray[idx], Eyarray[idx], Ezarray[idx]
        Delta = (Exarray+Eyarray)/2 - Ezarray

        print(repr(parray[46:len(parray)-1]))
        print(repr((Delta/np.abs(earray))[46:len(parray)-1]))
        # print(repr(np.gradient(Delta,parray,edge_order=2)))

        fig, [ax1, ax2]=plt.subplots(2,1, sharex=True)
        ax1.scatter(parray, Delta/np.abs(earray), marker=".", facecolors='none', edgecolors='red')
        ax2.scatter(parray,np.gradient(Delta,parray,edge_order=2),marker=".", facecolors='none', edgecolors='red')
        ax1.grid(True)
        ax2.grid(True)
        # ax1.set_ylim(-0.025,0.025)
        # ax2.set_ylim(-2,2)
        ax2.set_xlabel(r'$\psi/\pi$')
        ax1.set_ylabel(r'$\Delta/|E|$')
        ax2.set_ylabel(r'$\partial\Delta$')
        fig.tight_layout(rect=[0,0.03,1,0.95])
        plt.savefig(plot_folder+f'bond_energy.pdf')
        plt.show()
        plt.close()

        # momentlst, swecorrlst = momentlst[idx], swecorrlst[idx]
        # magnongaplst = magnongaplst[idx]
        # swelst = swelst[idx]
        # # print(momentlst)
        # fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True)
        #
        # ax1.scatter(parray, swelst, marker=".", facecolors='none', edgecolors='r')
        # ax1.set_ylabel(r'$E_{sw}$')
        #
        # ax2.scatter(parray, magnongaplst, marker=".", facecolors='none', edgecolors='black')
        # ax2.set_ylabel(r'$\Delta_0$')
        # ax2.axhline(y=0, color='0.75', linestyle=':')
        #
        # ax3.scatter(parray, momentlst, marker=".", facecolors='none', edgecolors='b')
        # ax3.set_ylabel(r'$m_{||}/S$')
        # ax3.set_ylim(-0.5,1)
        # ax3.axhline(y=0, color='0.75', linestyle=':')
        #
        # ax3.set_xlabel(r'$\psi/\pi$')
        # fig.savefig(plot_folder+f'properties.pdf')
        # # plt.show()
        # plt.close()

    # stuff = [
    #     parray,
    #     momentlst,
    #     magnongaplst
    # ]
    #
    # with open(plot_folder+'lswt_properties.out', 'wb') as f:
    #     pickle.dump(stuff, f)

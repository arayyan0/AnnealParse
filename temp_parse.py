import numpy as np
import matplotlib.pyplot as plt
from spin_lib import AnnealedSpinConfigurationTriangular
import os
import glob as glob

which = 'jtau_l=36'
# which = 'jtau_d=-2_l=8'
# which = 'jtau_d=-1.8_i=-0.2_l=8'

# finite_size_cv_lst = []
# finite_size_m_lst = []

for number in range(1,1+1):
    run = f'{number}'

    for version in range(1,1+1):
        data_folder = f'out/{which}/jobrun_{run}/v_{version}/'
        print(data_folder)
        plot_folder = data_folder + f'plots/'
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

        file_lst = glob.glob(data_folder+'*.out')

        Tlst = []
        specificheatlst=[]
        FMnormlst =[]
        perpnormlst =[]
        parnormlst  =[]

        for file in file_lst:
            split_file = file.replace('/','_').split('_')
            T = float(split_file[-2])
            spinstuff = AnnealedSpinConfigurationTriangular(file)

            Tlst.append(T)
            specificheatlst.append(spinstuff.SpecificHeat)
            FMnormlst.append(spinstuff.FMNorm)
            perpnormlst.append(spinstuff.PerpNorm)
            parnormlst.append(spinstuff.ParNorm)

            # # quiver_options = [0.6*35, 1.5, 3.5]       #[scale, minlength, headwidth]
            # quiver_options = [1.4*35, 1.5, 3.5]       #[scale, minlength, headwidth]
            # cm = 'gnuplot'
            # cb_options = [0.04, 'horizontal', cm,'x']    #[fraction, orientation, colormap]
            # usetex=False
            # fig = spinstuff.PlotSpins(quiver_options, cb_options, usetex)
            # plt.savefig(plot_folder+f'spins_temp_{T}.pdf')
            # plt.close()

        Tarray = np.array(Tlst)
        SpecificHeatarray = np.array(specificheatlst)
        FMnormarray = np.array(FMnormlst)
        perpnormarray = np.array(perpnormlst)
        parnormarray = np.array(parnormlst)

        idx = np.argsort(Tarray)
        Tarray = Tarray[idx]
        SpecificHeatarray = SpecificHeatarray[idx]
        FMnormarray = FMnormarray[idx]
        perpnormarray=perpnormarray[idx]
        parnormarray=parnormarray[idx]

        print(parnormarray)

        # print(np.array([Tarray, SpecificHeatarray]).T)

        fig,ax = plt.subplots()

        ax_right = ax.twinx()

        scale = 10
        ax.plot(Tarray, SpecificHeatarray, '-o',color='green')
        ax_right.plot(Tarray, FMnormarray*scale, '-o',color='red',label=fr'$\langle |m| \rangle\times{scale}$')
        ax_right.plot(Tarray, perpnormarray, '-o',color='orange',label=fr'$\langle |m^\bot| \rangle$')
        ax_right.plot(Tarray, parnormarray**scale, '-o',color='pink',label=fr'$\langle |m^\parallel| \rangle\times{scale}$')

        ax_right.axhline(0,ls='--',color='gray')
        ax_right.axhline(1,ls='--',color='gray')

        ax_right.set_ylim(0,1)
        ax.set_xlim(0,np.max(Tarray))

        ax.set_xlabel('Temperature')
        ax.set_ylabel(r'$C_v$',color='green')
        ax.set_title(f'L={number*36}')
        ax.tick_params(axis="y", colors='green')

        ax_right.legend()

        # ax_right.set_ylabel(r'$\langle m\rangle$')
        # ax_right.tick_params(axis="y", colors='blue')

        fig.tight_layout()
        plt.savefig(plot_folder+f'temp_obsv.pdf')
        plt.show()
        plt.close()

    # finite_size_cv_lst.append(SpecificHeatarray)
    # finite_size_m_lst.append(Magnetizationarray)

# print(finite_size_lst)
# fig, (ax1,ax2) =plt.subplots(2,1,sharex=True)
#
# colors = ['red','green','blue']
#
# for array,color in zip(finite_size_cv_lst,colors):
#     ax1.plot(Tarray, array, '-o', color=color)
#
# for array,color in zip(finite_size_m_lst,colors):
#     ax2.plot(Tarray, array, '-o', color=color)
# #
# ax1.set_xlabel('Temperature')
# ax1.set_ylabel(r'$C_v$')
# ax2.set_ylabel(r'$m$')
# ax1.set_title(f'L=12,24,36')
# #
# fig.tight_layout()
# plt.show()
# plt.close()

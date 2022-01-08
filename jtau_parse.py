import numpy as np
import matplotlib.pyplot as plt
from newspin_lib import MonteCarloOutput
from common import FreeEnergyDerivatives, pi
import os
import glob as glob
import pandas as pd
import sys
#
def PlotSpins(spinstuff,plot_folder,s):
    plane = -1 #-1 for Az plane with most moments, >=0 otherwise
    if spinstuff.NumSites > 500:
        quiver_options = [11*35, 2.5, 2]       #[scale, minlength, headwidth]
    else:
        quiver_options = [0.5*35, 2.5, 2]
    cm = 'seismic'
    cb_options = [0.04, 'horizontal', cm,'x'] #[fraction, orientation, colormap, axis]
    usetex=False
    for plane in range(len(spinstuff.LayerNumber)):
        fig = spinstuff.Plot2DPlane(plane, quiver_options, cb_options, usetex)
        plt.savefig(plot_folder+f'2d_spins_'+s+f'plane_{plane}.pdf')
        # plt.show()
        plt.close()
#
def PlotSSF(spinstuff,plot_folder,s):
    plane = -1 #-1 for Az plane with most moments, >=0 otherwise
    cb_options =[0.04, 'vertical', 'viridis']    #[fraction, orientation, colormap]
    usetex=False
    for plane in range(len(spinstuff.LayerNumber)):
        fig = spinstuff.Plot2DSSF(plane,cb_options,usetex)
        fig.axes[0].set_facecolor('black')
        plt.savefig(plot_folder+f'2d_ssf_'+s+f'_plane_{plane}.pdf')
        # plt.show()
        plt.close()

def is_not_unique(params):
    bool_array = []
    num_params = params.shape[1]
    for i in range(num_params):
        tol = 10**-6
        a = params[:,i]
        bool_array.append( ~(np.abs(a[0] - a) < tol).all() )
    return bool_array

def ToJKGGp(paramsl):
    t,p,jb = paramsl[:2+1]
    vec = np.array([
                    np.cos(t*pi),
                    jb,
                    np.sin(t*pi)*np.cos(p*pi),
                    np.sin(t*pi)*np.sin(p*pi),
    ])
    rot = np.array([
                    [0,2/3,4/3,-4/3],
                    [0,np.sqrt(2)/3,-np.sqrt(2)/3,np.sqrt(2)/3],
                    [1,0,-1,0],
                    [1,1/3,2/3,4/3]
    ])
    invrot = np.linalg.inv(rot)
    return (invrot @ vec).tolist()

if __name__ == '__main__':
    which = f'12.16.2021_jbn0_luc_phiscan'
    ########------------model-sepcific parameters------------########
    paramslabel    = [      't',     'p',   'jb' , 'h']
    paramsTeXlabel = [r'\theta', r'\phi', r'J_B' ,r'h']
    isangle = [True,True,False,False]

    #run = 1-7
    clusters = [
                [1,1,3,2,1],
                [1,1,4,1,1],
                [1,0,1,6,1],
                [1,1,1,2,1],
                [1,2,1,1,1],
                [1,1,1,1,1]
               ]
    cluster_colors = ['r','g','b','black','grey','purple']
    cluster_labels = ['24',
                      '16',
                      '12',
                      '8',
                      '6',
                      '4'
                     ]


    # run = 5
    # clusters = [ [1,1,7,2,1], [1,1,5,2,1], [1,1,3,2,1] ]
    # cluster_colors = ['r','g','b']
    # cluster_labels = ['56',
    #                   '40',
    #                   '24',
    #                  ]

    # run = 9
    # clusters = [ [1,1,1,2,1] ]
    # cluster_colors = ['r','g','b']
    # cluster_labels = ['56',
    #                   '40',
    #                   '24',
    #                  ]

    # run = 10
    # clusters = [ [1,1,1,2,1], [1,1,3,2,1], [1,1,5,2,1] ]
    # cluster_colors = ['r','g','b']
    # cluster_labels = ['8',
    #                   '24',
    #                   '40',
    #                  ]

    # run = 6
    # clusters = [ [1,1,3,2,1], [1,1,4,1,1], [1,1,1,2,1] ]
    # cluster_colors = ['r','g','b']
    # cluster_labels = ['24',
    #                   '16',
    #                   '8',
    #                  ]

    # the 8 M/2 clusters
    # run =
    # clusters = [ [1,1,n,2,1] for n in range(8,0,-1) ]
    # cluster_colors = ['r','g','b','black','grey','purple','pink','cyan',]
    # cluster_labels = ['8',
    #                   '7',
    #                   '6',
    #                   '5',
    #                   '4',
    #                   '3',
    #                   '2',
    #                   '1'
    #                  ]
    ########-------------------------------------------------########
    for number in range(1,4+1):
        cluster_sweeps = []
        for lat, s, l1, l2, l3 in clusters:
            for v in range(1,1+1):
                run = f'{number}'
                data_folder = f'out/{which}/jobrun_{run}/lat_{lat}_s_{s}_l1_{l1}_l2_{l2}_l3_{l3}/v_{v}/'
                print(data_folder)
                plot_folder = data_folder + f'plots/'
                if not os.path.exists(plot_folder):
                    os.makedirs(plot_folder)

                file_lst = sorted(glob.glob(data_folder+'*.out'))

                params = []
                JKGGPparams = []
                elst = []
                for file in file_lst:
                    split_file = file.replace('/','_').split('_')
                    ########------------model-sepcific parameters------------########
                    paramsl = list(map(float, [split_file[-8],split_file[-6],split_file[-4],split_file[-2]]))
                    params.append(paramsl)
                    # print(paramsl)
                    JKGGPparams.append(ToJKGGp(paramsl))
                    # print(ToJKGGp(paramsl))
                    s = ''
                    for k in range(len(paramslabel)):
                        s = s + paramslabel[k] + '_' + f'{paramsl[k]:.6f}_'
                    ########-------------------------------------------------########
                    spinstuff = MonteCarloOutput(file)
                    elst.append(spinstuff.MCEnergyPerSite)

                    PlotSpins(spinstuff,plot_folder,s)
                    PlotSSF(spinstuff,plot_folder,s)
                params = np.array(params)
                earr = np.array(elst)
                JKGGPparams = np.array(JKGGPparams)

                # print(params)
                # print(earr)
                # print(JKGGPparams)

                #identify swept parameters
                which_params = is_not_unique(params)
                # print(which_params)
                #
                # sort data
                which_parameter_to_sort = 1
                idx = np.argsort(params[:, which_parameter_to_sort])
                params = params[idx,:]
                earr = earr[idx]
                JKGGPparams = JKGGPparams[idx]
                ########----------------------------------------------------########

                # put params and energy in a pandas dataframe. this will be especially
                # useful for multiple swept parameters, but recheck the code to see
                # if things are running properly
                # print(params.shape, earr.shape)
                data = np.column_stack((params,earr))
                df = pd.DataFrame(data=data, index=None, columns=paramslabel+['energy'])
                # print(df)

                # for each swept parameter
                for i, boolean in enumerate(which_params):
                    # if constant array, we wont be plotting the energy over it
                    if ~boolean:
                        # print(f'{paramslabel[i]} is not swept over')
                        continue

                    # figure out the factor that multiplies the parameter
                    if isangle[i]:
                        xlabel = paramsTeXlabel[i] + r'/\pi'
                        factor = pi
                    else:
                        xlabel = paramsTeXlabel[i]
                        factor = 1

                    # compute derivatives and reorder colors
                    derivs = FreeEnergyDerivatives(df[paramslabel[i]], df['energy'], factor)
                    ########--------------model-sepcific sorting----------------########
                    colors, color_order = derivs.Colors, [0,2,1]
                    colors = [colors[color_index] for color_index in color_order]

                    # figure out the y-labels
                    ylabel = [
                    r"$\frac{E_0}{N}$",
                    r"$-\frac{1}{N}\frac{\mathrm{d}^2E_0}{\mathrm{d}%s^2}\quad$" % (paramsTeXlabel[i]),
                    r"$-\frac{1}{N}\frac{\mathrm{d}E_0}{\mathrm{d}%s}$" % (paramsTeXlabel[i]),
                    ]

                    # plot the derivs with proper labels
                    fig = derivs.PlotSweep()
                    for j, [ax, color] in enumerate(zip(fig.axes, colors)):
                        ax.set_ylabel(     ylabel[j], rotation = "horizontal",
                                         fontsize=12,             labelpad=20,
                                         color=color)
                    fig.axes[1].set_xlabel(r"$%s$ " % xlabel )
                    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

                    # save and show the plot after creating filename
                    s = ''
                    for k, boolean in enumerate(which_params):
                        if ~boolean:
                            s = s + paramslabel[k] + '_' + f'{df[paramslabel[k]][0]:.6f}_'
                    # print(s)
                    plt.savefig(plot_folder + 'energy_' + s + '.pdf')
                    # plt.show()
                    plt.close()

                    fig, ax = plt.subplots()
                    colors = ['violet','green','red','blue']
                    labels = ['$J$', '$K$', '$\Gamma$', r"$\Gamma'$"]
                    linestyles = ['-','--','-.',':']
                    for ii, [color,label,linestyle] in enumerate(zip(colors,labels,linestyles)):
                        ax.plot(df[paramslabel[i]],JKGGPparams[:,ii],c=color,label=label,linestyle=linestyle)
                    ax.plot(df[paramslabel[i]],JKGGPparams[:,2]+2*JKGGPparams[:,3],c='grey',label=r"$\Gamma+2\Gamma'$")
                    ax.set_xlabel(r"$%s$ " % xlabel )
                    ax.axhline(color='gray',ls="--")
                    plt.legend()
                    plt.savefig(plot_folder + 'JKGGp_' + s + '.pdf')
                    # plt.show()
                    plt.close()
            cluster_sweeps.append(earr)
        fig, ax = plt.subplots()
        for i, [sweep,color,label] in enumerate(zip(cluster_sweeps,cluster_colors,cluster_labels)):
            ax.plot(df['p'], sweep, color = color, label = label, marker='x')
        plt.legend()
        plt.savefig(f'out/{which}/jobrun_{run}/LUCcomparison.pdf')
        plt.show()
        plt.close()

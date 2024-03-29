import numpy as np

# import matplotlib as mpl
# mpl.use('tkagg')

import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.patches as ptch
import matplotlib.pylab as pl
from matplotlib.colors import TwoSlopeNorm
#from functools import cached_property

from common import pi, sqrt3, gen_eps, a1, a2, AddBZ, FindReciprocalVectors, \
LocalRotation, IndexToPosition, KMeshForPlotting, PlotLineBetweenTwoPoints,\
KMeshForIntegration



def WhichUnitCell(hc_or_kek: int, type: int, sublattice: int):
    '''
    Selects the unit cell used in the Monte Carlo simulation.

    Input
    hc_or_kek          (int): standard honeycomb (0) or Kekule arrangement (1)
    type               (int): orientation 1 (1) or 2 (2) of unit cell, refer to notes
    sublattice         (int): rhombus (2) or rectangular (4) unit cell

    Output
    A1,A2    (numpy.ndarray): two primitive unit cell vectors of shape (2,)
    sub_list (numpy.ndarray): location of each sublattice, shape (sublattice, 2)
    '''
    # standard honeycomb pattern
    if hc_or_kek == 0:
        ## rhombus cluster 1, oriented along z bond
        if sublattice == 2 and type == 1:
            A1, A2 = a1, a2
            sub_list = np.array([
                (1.0*A1 + 1.0*A2)/3.0,
                (2.0*A1 + 2.0*A2)/3.0
            ])

        ## rhombus cluster 2, oriented along x bond
        elif sublattice == 2 and type == 2:
            A1, A2 = a1 - a2, a1
            sub_list = np.array([
                (1.0*A1 + 1.0*A2)/3.0,
                (2.0*A1 + 2.0*A2)/3.0
            ])

        elif sublattice == 2 and type == 3:
            A1, A2 = a2, a2-a1
            sub_list = np.array([
                (1.0*A1 + 1.0*A2)/3.0,
                (2.0*A1 + 2.0*A2)/3.0
            ])

        # rectangular cluster 1, oriented along y bond
        elif sublattice == 4 and type == 1:
            A1, A2 = a1, a1 + 2 * (a2 - a1)
            sub_list = np.array([
                (3.0*A1 + 1.0*A2)/6.0,
                (6.0*A1 + 2.0*A2)/6.0,
                (6.0*A1 + 4.0*A2)/6.0,
                (3.0*A1 + 5.0*A2)/6.0,
            ])

        # rectangular cluster 2, oriented along z bond
        elif sublattice == 4 and type == 2:
            A1, A2 = a1 - a2, a1 + a2
            sub_list = np.array([
                (3.0*A1 + 1.0*A2)/6.0,
                (6.0*A1 + 2.0*A2)/6.0,
                (6.0*A1 + 4.0*A2)/6.0,
                (3.0*A1 + 5.0*A2)/6.0,
            ])

        elif sublattice == 4 and type == 3:
            A1, A2 = a2, a2-2*a1
            sub_list = np.array([
                (3.0*A1 + 1.0*A2)/6.0,
                (6.0*A1 + 2.0*A2)/6.0,
                (6.0*A1 + 4.0*A2)/6.0,
                (3.0*A1 + 5.0*A2)/6.0,
            ])

        elif sublattice == 6:
            A1, A2 = 2*a1 - a2, 2*a2 - a1
            sub_list = np.array([
                (1.0*A1+1.0*A2)/6.0,
                (3.0*A1+1.0*A2)/6.0,
                (5.0*A1+3.0*A2)/6.0,
                (5.0*A1+5.0*A2)/6.0,
                (3.0*A1+5.0*A2)/6.0,
                (1.0*A1+3.0*A2)/6.0,
            ])

    # Kekule cluster
    elif hc_or_kek == 1:
        if sublattice == 6:
            A1, A2 = 2 * a1 - a2, 2 * a2 - a1
            sub_list = np.array([
                (1.0*A1+1.0*A2)/6.0,
                (3.0*A1+1.0*A2)/6.0,
                (5.0*A1+3.0*A2)/6.0,
                (5.0*A1+5.0*A2)/6.0,
                (3.0*A1+5.0*A2)/6.0,
                (1.0*A1+3.0*A2)/6.0,
            ])
    return A1, A2, sub_list



class AnnealedSpinConfiguration:
    '''
    Class associated with the raw output file of the C++ simulated annealing library.
    Contains all relevant information of the simulation, including cluster information,
    simulation parameters, Hamiltonian parameters, resulting energy per site,
    spin position indices, and spin components.
    Class that further processes the raw data in ParseSAFile. Here we may calculate
    the static strucutre factor and plot it over the 1st and 2nd Brillouin zones.
    The spins are also rotated into the abc* direction so they may be plotted
    over the honeycomb plane.

    Note: does not include the Zeeman field magnitude h or the field direction
          characterized by (h_theta, h_phi). Please restrict to parsing h = 0
          simulations at this time.

    Attributes
    S, L1, L2                   (int): sublattice within unit cell, and number of
                                       times it is translated in the A1/A2 directions
    Sites                       (int): number of sites in the cluster
    A1, A2            (numpy.ndarray): unit cell vectors of the cluster,
                                       both of shape (2,)
    SublatticeVectors (numpy.ndarray): position of sites within the unit cell
                                       of shape (Sites, 2)
    SimulationParameters       (list): numbers specifiying initial and final temp.
                                       as well as the number of Metropolis and
                                       determinstic flips
    HamiltonianParameters      (list): parameters of the (anisotropic) Kitaev,
                                       (anisotropic) Gamma, Gamma', and 1st NN
                                       Heisenberg interactions
    MCEnergyDensity    (numpy.double): final energy of the spin configuration
    SpinsXYZ          (numpy.ndarray): the spin components in the global xyz basis,
                                       of shape (l1*l2*s,3)
    SpinLocations     (numpy.ndarray): spin locations, shape (Sites, 2)
    SpinsABC          (numpy.ndarray): spins written in the abc* basis, shape (Sites, 3)
    '''
    def __init__(self, filename):
        '''
        Construct ParseSAFile instance.

        Parameters
        filename (string): filename of the simulated annealing raw data file.
        '''
        self.Filename = filename

        with open(self.Filename, 'r') as f:
            file_data = f.readlines()

        hc_or_kek, self.Type = [int(x) for x in file_data[2].split()]
        self.S, self.L1, self.L2 = [int(x) for x in file_data[4].split()]
        self.Sites = self.S*self.L1*self.L2
        self.A1, self.A2, self.SublatticeVectors = \
                                        WhichUnitCell(hc_or_kek, self.Type, self.S)

        T_i = np.double(file_data[6])             #initial annealing temperature
        T_f = np.double(file_data[8])               #final annealing temperature
        SA_sweeps, therm_sweeps, measuring_sweeps, sampling_time = \
         list(map(int, file_data[10].replace("/",' ').split()))
        det_aligns = int(file_data[12])          #number of deterministic aligns
        self.SimulationParameters = [T_i, T_f, SA_sweeps, therm_sweeps,
                                     measuring_sweeps, sampling_time, det_aligns]

        Kx, Ky, Kz = [float(x) for x in file_data[15].split()]
        Gx, Gy, Gz = [float(x) for x in file_data[17].split()]
        Gp = float(file_data[19].split()[0])
        J1 = float(file_data[21].split()[0])
        self.HamiltonianParameters = [np.array([Kx, Ky, Kz]), np.array([Gx, Gy, Gz]), Gp, J1]

        self.MCEnergyDensity = np.double(file_data[28])

    def ExtractMomentsAndPositions(self):
        '''
        Extracts the moments and locations from the file. Separated from __init__
        to save time if one just needs the energy of the spin configuration.

        Parameters
        filename (string): filename of the simulated annealing raw data file.
        '''
        with open(self.Filename, 'r') as f:
            file_data = f.readlines()

        sign = 1 #to flip sign of spins
        self.SpinLocations = np.array(np.empty((self.Sites, 2)))
        self.SpinsXYZ = np.empty((self.Sites, 3))
        for i, line in enumerate(file_data[30:]):
            n1, n2, sub, Sx, Sy, Sz = line.split()
            self.SpinsXYZ[i] = sign*np.array(list(map(np.longdouble,[Sx,Sy,Sz])))
            self.SpinLocations[i] = IndexToPosition(self.A1, self.A2, self.SublatticeVectors,
                                                      map(int,[n1,n2,sub]))

        xyz_to_abc = LocalRotation(np.array([1,1,1])/sqrt3)
        self.SpinsABC = np.einsum('ab,cb->ca', xyz_to_abc, self.SpinsXYZ)


    def CalculateAndPlotSSF(self, cb_options,usetex):
        '''
        Calculates the static structure factor (SSF) defined as
                s_k/N = 1/N^2 sum_{ij} S_i . S_j exp(-i k.(r_i - r_j) )
        over a Gamma centered k-mesh of the 1st and 2nd crystal BZ. Kmesh consists
        of the accessible momentum points of the cluster, ie.
                      k = m_1 B1 /L1 + m_2 B2 / L2
        where m_i are integers.

        Returns
        fig (numpy.ndarray): figure of the SSF and accessible momentum points
        '''
        self.ExtractMomentsAndPositions()

        B1, B2 = FindReciprocalVectors(self.A1, self.A2)

        addbz=True
        addPoints=False

        KX, KY, fig = KMeshForPlotting(B1, B2, self.L1, self.L2, 3, 3, addbz, addPoints, usetex)
        kx, ky = [np.reshape(x, -1) for x in [KX, KY]]
        k = np.stack((kx, ky)).T
        ax = fig.axes[0]
        scale = 2*pi

        SdotS_mat = np.einsum("ij,kj", self.SpinsXYZ, self.SpinsXYZ)
        s_kflat = np.zeros(len(k), dtype=np.cdouble)
        for i, kv in enumerate(k):
            phase_i = np.exp(1j * np.einsum('i,ji', kv, self.SpinLocations))
            phase_j = np.exp(-1j * np.einsum('i,ji', kv, self.SpinLocations))
            phase_mat = np.einsum('i,j->ij', phase_i, phase_j)
            s_kflat[i] = (SdotS_mat * phase_mat).sum()/ self.Sites**2
        #     ax.annotate(f'$\quad${np.real(s_kflat[i]):.6f}',
        #                 kv/scale,
        #                 fontsize=4,
        #                 color = 'lightgray')
        # print(self.SpinsXYZ)

        ssf = np.reshape(s_kflat, KX.shape)/np.max(s_kflat)

        ssf_complex_flag = not np.all(np.imag(ssf) < gen_eps)

        if ssf_complex_flag:
            print("Warning: SSF has complex values. Please recheck calculation.")

        #note: we are plotting the real part, since the SSF should be real.

        fraction, orientation, colormap = cb_options
        c = ax.scatter(KX/scale, KY/scale, c=np.real(ssf), cmap=colormap, edgecolors="none",zorder=0)
        cbar = fig.colorbar(c, fraction=fraction, orientation=orientation)
        cbar.ax.set_title(r'{\rm Relative} $s_\mathbf{k}$',usetex=usetex,fontsize=9,y=-3)

        ticks = np.linspace(0,1,4+1)
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels([f'${val:.2f}$' for val in ticks],usetex=usetex,fontsize=9)
        # cbar.ax.set_xticklabels([f'${val:.2f}$' for val in ticks],usetex=usetex,fontsize=9)

        return fig

    def PlotSpins(self, quiver_options, cb_options,plaquettes,signstructure,usetex):
        '''
        Plots the spins (in the ABC basis) over the honeycomb plane, with color
        indicating the angle made with the vector c* = (1,1,1)/sqrt(3) in
        Cartesian basis

        Returns
        fig (numpy.ndarray): figure of the spin configuration over the , as well
                             as the unit cell used to construct the cluster
        '''
        self.ExtractMomentsAndPositions()

        sss, minlength, headwidth = quiver_options
        fraction, orientation, cm, tickaxis = cb_options

        sign = 1 #do not change here: change in ExtractMomentsAndPositions()

        thetac = np.arccos(np.clip(sign*self.SpinsABC[:, 2], -1, 1)) / pi * 180
        norm = clr.Normalize(vmin=0,vmax=180)

        # print("xyz:")
        # thetaz = (np.arccos(self.SpinsXYZ[:, 2])) / pi
        # phixy = (np.arctan2(self.SpinsXYZ[:, 1],self.SpinsXYZ[:, 0]))/ pi
        # print(self.SpinsXYZ)
        # # print(np.stack((phixy,thetaz)).T)
        #
        # print("abc:")
        # thetacc = (np.arccos(self.SpinsABC[:, 2])) / pi
        # phiab = (np.arctan2(self.SpinsABC[:, 1],self.SpinsABC[:, 0]))/ pi
        # print(self.SpinsABC)
        # print(np.stack((phiab,thetacc)).T)

        # cm = pl.cm.RdBu
        # my_cmap = cm(np.arange(cm.N))
        # my_cmap[:,-1] = np.linspace(0, 1, cm.N)
        # cm = clr.ListedColormap(my_cmap)

        fig, ax = plt.subplots()

        for x in range(0,self.L1+1):
            for y in range(0,self.L2+1):
                #used for rhom unit cell
                center1 = x * self.A1 + y * self.A2
                hex2 = ptch.RegularPolygon(
                center1+a1, 6, 1 / sqrt3, 0, fill=False, linewidth=0.0005,color='gray')
                ax.add_patch(hex2)

                hex3 = ptch.RegularPolygon(
                center1, 6, 1 / sqrt3, 0, fill=False, linewidth=0.0005,color='gray')
                ax.add_patch(hex3)

                #used for rect unit cell
                # if x != 0:
                #     hex4 = ptch.RegularPolygon(
                #     x * self.A1 + y*self.A2, 6, 1 / sqrt3, 0, fill=False, linewidth=0.0005,color='gray')
                #     ax.add_patch(hex4)
                # if y != 0:
                #     hex5 = ptch.RegularPolygon(
                #     x * self.A1 + y*self.A2-a2, 6, 1 / sqrt3, 0, fill=False, linewidth=0.0005,color='gray')
                #     ax.add_patch(hex5)

                #used for 6-site unit cell
                # center1 = x * self.A1 + y * self.A2 + 2*(self.A1+self.A2)/3
                # hex2 = ptch.RegularPolygon(
                # center1+a1, 6, 1 / sqrt3, 0, fill=False, linewidth=0.0005,color='gray')
                # ax.add_patch(hex2)
                #
                # hex2 = ptch.RegularPolygon(
                # center1+a2, 6, 1 / sqrt3, 0, fill=False, linewidth=0.0005,color='gray')
                # ax.add_patch(hex2)
                #
                # hex3 = ptch.RegularPolygon(
                # center1, 6, 1 / sqrt3, 0, fill=False, linewidth=0.0005,color='gray')
                # ax.add_patch(hex3)

        ax.quiver(self.SpinLocations[:,0], self.SpinLocations[:,1],
                  sign*self.SpinsABC[:,0]     , sign*self.SpinsABC[:,1]     ,
                  thetac, alpha=1,cmap=cm, norm=norm, pivot = 'mid',
                  scale=sss,
                  minlength=minlength,
                  headwidth=headwidth,
                  linewidth=0.1,
                  ec='black')

        ax.axis("off")
        # ax.set_facecolor('black')
        ax.axis("equal") #zooms in on arrows

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        cb = plt.colorbar(sm,
            fraction=fraction,
            # pad=0,
            orientation=orientation)
        cb.ax.set_title(r'$\theta_{[111]}\quad \left(^\circ\right)$', usetex=usetex)

        ticks = np.linspace(0,180,6+1)
        cb.set_ticks(ticks)
        if tickaxis=='x':
            cb.ax.set_xticklabels([f'${val:.0f}$' for val in ticks],usetex=usetex)
        elif tickaxis== 'y':
            cb.ax.set_yticklabels([f'${val:.0f}$' for val in ticks],usetex=usetex)


        #overlaying plaquette values
        size=6
        if plaquettes == True and self.S == 2:
            plabels, ____ = self.CalculatePlaquetteVariables()
            i=0
            for py in range(0,3):
                for px in range(0,3):
                    pos = 2*(self.A1 + self.A2)/3 + (-self.A1) + px*self.A1 + py*self.A2
                    ax.annotate(plabels[i],
                                pos,
                                fontsize=size,
                    )
                    i += 1
            print(plabels)
            ax.axis("equal") #shows all signs


        #overlaying plaquette sgn structure
        size=20
        if signstructure == True and self.S == 2:
            plabels=self.CalculateSignStructure()
            i=0
            for py in range(0,3):
                for px in range(0,3):
                    pos = 2*(self.A1 + self.A2)/3 + (-self.A1) + px*self.A1 + py*self.A2
                    ax.annotate(plabels[i],
                                pos,
                                fontsize=sss,
                    )
                    i += 1
            print(plabels)
            ax.axis("equal") #shows all signs
        elif signstructure == True and self.S == 4:
            plabels=self.CalculateSignStructure()
            i=0
            for py in range(0,1+1):
                for px in range(0,3):
                    pos = px*self.A1 + py*a1+2*a1/3
                    ax.annotate(plabels[i],
                                pos,
                                fontsize=sss,
                    )
                    i += 1
            print(plabels)
            ax.axis("equal") #shows all signs


        #adding unit cell used for the calculation
        # g = np.zeros(2)
        # PlotLineBetweenTwoPoints(ax, g, self.A1)
        # PlotLineBetweenTwoPoints(ax, g, self.A2)
        # PlotLineBetweenTwoPoints(ax, self.A1, self.A1+self.A2)
        # PlotLineBetweenTwoPoints(ax, self.A2, self.A1+self.A2)
        return fig

    def CalculatePlaquetteVariables(self):
        #currently only works for rh2 18-site cluster
        if self.Type == 2 and self.S == 2 and self.L1 == 3 and self.L2 == 3:
            self.ExtractMomentsAndPositions()

            spins = self.SpinsXYZ
            p1a = (2**6)*spins[0,2]*spins[1,1]*spins[6,0]*spins[11,2]*spins[10,1]*spins[5,0]
            p2a = (2**6)*spins[2,2]*spins[3,1]*spins[8,0]*spins[7,2]*spins[6,1]*spins[1,0]
            p3a = (2**6)*spins[4,2]*spins[5,1]*spins[10,0]*spins[9,2]*spins[8,1]*spins[3,0]

            p4a = (2**6)*spins[6,2]*spins[7,1]*spins[12,0]*spins[17,2]*spins[16,1]*spins[11,0]
            p5a = (2**6)*spins[8,2]*spins[9,1]*spins[14,0]*spins[13,2]*spins[12,1]*spins[7,0]
            p6a = (2**6)*spins[10,2]*spins[11,1]*spins[16,0]*spins[15,2]*spins[14,1]*spins[9,0]

            p7a = (2**6)*spins[12,2]*spins[13,1]*spins[0,0]*spins[5,2]*spins[4,1]*spins[17,0]
            p8a = (2**6)*spins[14,2]*spins[15,1]*spins[2,0]*spins[1,2]*spins[0,1]*spins[13,0]
            p9a = (2**6)*spins[16,2]*spins[17,1]*spins[4,0]*spins[3,2]*spins[2,1]*spins[15,0]

            valuearray = [p1a,p2a,p3a,p4a,p5a,p6a,p7a,p8a,p9a]
            labelarray = [f'{i:.8f}' for i in valuearray]

            # print(sum(np.array([p1a,p2a,p3a,p4a,p5a,p6a,p7a,p8a,p9a])/9))
            #
            # print("A:",(p1a+p5a+p9a)/3)
            # print("B:",(p2a+p6a+p7a)/3)
            # print("C:",(p3a+p4a+p8a)/3)
            return labelarray, valuearray
        else:
            print("you must expand CalculateSignStructure to include this cluster shape")

    def CalculateSignStructure(self):
        #currently only works for rh2 18-site cluster
        if self.Type == 2 and self.S == 2 and self.L1 == 3 and self.L2 == 3:
            self.ExtractMomentsAndPositions()

            #second index tracks z,y,x
            ispins=self.SpinsXYZ
            for i in range(0, 18, 2): #flip sign of spins on B sublattice
                ispins[i,:] = - ispins[i,:]
            signs = np.sign(ispins)
            p1a = np.array([signs[0,2],signs[1,1],signs[6,0],signs[11,2],signs[10,1],signs[5,0]])
            p2a = np.array([signs[2,2],signs[3,1],signs[8,0],signs[7,2],signs[6,1],signs[1,0]])
            p3a = np.array([signs[4,2],signs[5,1],signs[10,0],signs[9,2],signs[8,1],signs[3,0]])

            p4a = np.array([signs[6,2],signs[7,1],signs[12,0],signs[17,2],signs[16,1],signs[11,0]])
            p5a = np.array([signs[8,2],signs[9,1],signs[14,0],signs[13,2],signs[12,1],signs[7,0]])
            p6a = np.array([signs[10,2],signs[11,1],signs[16,0],signs[15,2],signs[14,1],signs[9,0]])

            p7a = np.array([signs[12,2],signs[13,1],signs[0,0],signs[5,2],signs[4,1],signs[17,0]])
            p8a = np.array([signs[14,2],signs[15,1],signs[2,0],signs[1,2],signs[0,1],signs[13,0]])
            p9a = np.array([signs[16,2],signs[17,1],signs[4,0],signs[3,2],signs[2,1],signs[15,0]])

            plabel = []
            for pa in [p1a,p2a,p3a,p4a,p5a,p6a,p7a,p8a,p9a]:
                p = np.int(np.prod(pa))
                if p == -1:
                    plabel.append('?')
                if p == 1 and pa[0] == 1:
                    plabel.append('+')
                if p == 1 and pa[0] == -1:
                    plabel.append('-')
            return plabel


        elif self.Type == 2 and self.S == 4 and self.L1 == 3 and self.L2 == 1:
            self.ExtractMomentsAndPositions()
            for i in range(0, 12, 2): #flip sign of spins on B sublattice
                self.SpinsXYZ[i] = - self.SpinsXYZ[i]

            signs = np.sign(self.SpinsXYZ)

            p1a = np.array([signs[0,2],signs[1,1],signs[2,0],signs[3,2],signs[10,1],signs[9,0]])
            p2a = np.array([signs[4,2],signs[5,1],signs[6,0],signs[7,2],signs[2,1],signs[1,0]])
            p3a = np.array([signs[8,2],signs[9,1],signs[10,0],signs[11,2],signs[6,1],signs[5,0]])

            p4a = np.array([signs[2,2],signs[7,1],signs[4,0],signs[1,2],signs[0,1],signs[3,0]])
            p5a = np.array([signs[6,2],signs[11,1],signs[8,0],signs[5,2],signs[4,1],signs[7,0]])
            p6a = np.array([signs[10,2],signs[3,1],signs[0,0],signs[9,2],signs[8,1],signs[11,0]])

            plabel = []
            for pa in [p1a,p2a,p3a,p4a,p5a,p6a]:
                p = np.int(np.prod(pa))
                if p == -1:
                    plabel.append('?')
                if p == 1 and pa[0] == 1:
                    plabel.append('+')
                if p == 1 and pa[0] == -1:
                    plabel.append('-')
            return plabel

        else:
            print("you must expand CalculateSignStructure to include this cluster shape")

################################################################################
################################################################################
################################################################################

def WhichTriangularUnitCell(type: int, sublattice: int):
    '''
    Selects the unit cell used in the Monte Carlo simulation.

    Input
    type               (int): orientation 1 (1) or 2 (2) of unit cell, refer to notes
    sublattice         (int): rhombic (1) or rectangular (2) unit cell

    Output
    A1,A2    (numpy.ndarray): two primitive unit cell vectors of shape (2,)
    sub_list (numpy.ndarray): location of each sublattice, shape (sublattice, 2)
    '''
    ## rhombus cluster 2
    if sublattice == 1 and type == 2:
        A1, A2 = a1-a2, a1
        sub_list = np.array([
            (A1+A2)/3,
        ])

    return A1, A2, sub_list

class AnnealedSpinConfigurationTriangular:
    '''
    Class associated with the raw output file of the C++ simulated annealing library.
    Contains all relevant information of the simulation, including cluster information,
    simulation parameters, Hamiltonian parameters, resulting energy per site,
    spin position indices, and spin components.
    Class that further processes the raw data in ParseSAFile. Here we may calculate
    the static strucutre factor and plot it over the 1st and 2nd Brillouin zones.
    The spins are also rotated into the abc* direction so they may be plotted
    over the honeycomb plane.

    Note: does not include the Zeeman field magnitude h or the field direction
          characterized by (h_theta, h_phi). Please restrict to parsing h = 0
          simulations at this time.

    Attributes
    S, L1, L2                   (int): sublattice within unit cell, and number of
                                       times it is translated in the A1/A2 directions
    Sites                       (int): number of sites in the cluster
    A1, A2            (numpy.ndarray): unit cell vectors of the cluster,
                                       both of shape (2,)
    SublatticeVectors (numpy.ndarray): position of sites within the unit cell
                                       of shape (Sites, 2)
    SimulationParameters       (list): numbers specifiying initial and final temp.
                                       as well as the number of Metropolis and
                                       determinstic flips
    HamiltonianParameters      (list): parameters of the (anisotropic) Kitaev,
                                       (anisotropic) Gamma, Gamma', and 1st NN
                                       Heisenberg interactions
    MCEnergyDensity    (numpy.double): final energy of the spin configuration
    SpinsXYZ          (numpy.ndarray): the spin components in the global xyz basis,
                                       of shape (l1*l2*s,3)
    SpinLocations     (numpy.ndarray): spin locations, shape (Sites, 2)
    SpinsABC          (numpy.ndarray): spins written in the abc* basis, shape (Sites, 3)
    '''
    def __init__(self, filename):
        '''
        Construct ParseSAFile instance.

        Parameters
        filename (string): filename of the simulated annealing raw data file.
        '''
        self.Filename = filename
        # print(filename)

        with open(self.Filename, 'r') as f:
            file_data = f.readlines()

        self.Type = int(file_data[2])
        self.S, self.L1, self.L2 = [int(x) for x in file_data[4].split()]
        self.Sites = self.S*self.L1*self.L2
        self.A1, self.A2, self.SublatticeVectors = \
                                        WhichTriangularUnitCell(self.Type, self.S)

        T_i = np.double(file_data[6])             #initial annealing temperature
        T_f = np.double(file_data[8])               #final annealing temperature

        SA_sweeps, therm_sweeps, measuring_sweeps, sampling_time = \
         list(map(int, file_data[10].replace("/",' ').split()))

        # met_sweeps = list(map(int, file_data[10].split()))              #number of metropolis trials
        det_aligns = int(file_data[12])          #number of deterministic aligns
        self.SimulationParameters = [T_i, T_f, SA_sweeps, therm_sweeps,
                                     measuring_sweeps, sampling_time, det_aligns]

        Jtau = float(file_data[15].split()[0])
        lambd = float(file_data[17].split()[0])
        self.JQuad, self.JOcto = list(map(np.double,file_data[19].replace('/',' ').split()))
        hfield = float(file_data[21].split()[0])
        # hdirection = np.array(list(map(float,file_data[22].split())))
        self.DefectQuad, self.DefectOcto, self.Lengthscale, self.NumDefects = list(map(np.double, file_data[25].replace('/',' ').split()))


        self.MCEnergyDensity = np.double(file_data[28])
        self.ExtractMomentsAndPositions()
        #
        # num_q = 16
        # octo = []
        # quad = []
        # for i in range(num_q):
        #     split = file_data[28+self.Sites+4+i].replace('\n',' ').split(' ')
        #     quad.append(np.double(split[-3]))
        #     octo.append(np.double(split[-2]))
        #
        #
        # if int(measuring_sweeps) != 0:
        #     self.E, self.E2 = list(map(np.longdouble, file_data[28+self.Sites+4+num_q+2].split()))
        #     self.SpecificHeat = self.Sites/(T_f)**2 * (self.E2 - self.E**2)
        #     # print(self.E, self.E2, self.E2 - self.E**2, self.Sites/(T_f)**2, self.SpecificHeat)
        #
        #     octobar = []
        #     quadbar = []
        #     for i in range(num_q):
        #         split = file_data[28+self.Sites+4+num_q+2+3+i].replace('\n',' ').split(' ')
        #         quadbar.append(np.double(split[-3]))
        #         octobar.append(np.double(split[-2]))
        #
        #     self.OctoBar = self.AnalyzeQPoints(octobar)
        #     self.QuadBar = self.AnalyzeQPoints(quadbar)
        #
        #     self.QLabels = [r'$\Gamma$', r"$\Gamma'$", r'$K$',r'$K/2$',r'$M$',r'$2M/3$']

    def AnalyzeQPoints(self, allqlist):
        '''
        q points are currently
        G,
        Gpx, Gpy, Gpz,
        Kx, Ky, Kz,
        Kx/2, Ky/2, Kz/2,
        Mx, My, Mz,
        2Mx/3, 2My/3, 2Mz/3
        '''

        g = allqlist[0]
        gpmax       = max(allqlist[1:1+3])
        kmax        = max(allqlist[4:4+3])
        kdiv2max    = max(allqlist[7:7+3])
        mmax        = max(allqlist[10:10+3])
        twothirdmmax = max(allqlist[13:13+3])

        return [g, gpmax, kmax, kdiv2max, mmax, twothirdmmax]


    def ExtractMomentsAndPositions(self):
        '''
        Extracts the moments and locations from the file. Separated from __init__
        to save time if one just needs the energy of the spin configuration.

        Parameters
        filename (string): filename of the simulated annealing raw data file.
        '''
        with open(self.Filename, 'r') as f:
            file_data = f.readlines()

        sign = 1 #to flip sign of spins

        self.SpinLocations = np.array(np.empty((self.Sites, 2)))
        self.SpinsXYZ = np.empty((self.Sites, 3))
        self.DipoleMoment = np.empty((self.Sites, 3))
        for i, line in enumerate(file_data[30:30+self.Sites]):
            n1, n2, sub, Sx, Sy, Sz = line.split()
            # print(n1, n2, sub, Sx, Sy, Sz)
            self.SpinsXYZ[i] = sign*np.array(
            [np.longdouble(Sx),-np.longdouble(Sz),np.longdouble(Sy)]
            )
            self.SpinLocations[i] = IndexToPosition(self.A1, self.A2, self.SublatticeVectors,
                                                      map(int,[n1,n2,sub]))
            self.DipoleMoment[i] = np.array([0,0,np.longdouble(Sy)])

        r_outofplane = 0 #moments are restricted to the single plane! UNLIKE muon
        self.SpinLocations3D = np.pad(self.SpinLocations, ((0,0),(0,1)),
                                      mode='constant', constant_values=r_outofplane)

    def PlotSpins(self, quiver_options, cb_options,usetex):
        '''
        Plots the spins (in the ABC basis) over the honeycomb plane, with color
        indicating the angle made with the vector c* = (1,1,1)/sqrt(3) in
        Cartesian basis

        Returns
        fig (numpy.ndarray): figure of the spin configuration over the , as well
                             as the unit cell used to construct the cluster
        '''

        sss, minlength, headwidth = quiver_options
        fraction, orientation, cm, tickaxis = cb_options

        sign = 1 #do not change here: change in ExtractMomentsAndPositions()

        # print(self.SpinLocations)
        # print(self.SpinsXYZ)

        thetac = np.arccos(np.clip(sign*self.SpinsXYZ[:, 2], -1, 1)) / pi * 180
        norm = clr.Normalize(vmin=0,vmax=180)

        fig, ax = plt.subplots()

        for x in range(0,self.L1+1):
            for y in range(-1,self.L2):
        #         #used for rhom unit cell
                center1 = x * self.A1 + y * self.A2
                hex2 = ptch.RegularPolygon(
                center1+(a1+a2)/3, 3, 1 / sqrt3, 0, fill=False, linewidth=0.005,color='gray')
                ax.add_patch(hex2)
        #
        #         hex3 = ptch.RegularPolygon(
        #         center1, 6, 1 / sqrt3, 0, fill=False, linewidth=0.0005,color='gray')
        #         ax.add_patch(hex3)

        ax.quiver(self.SpinLocations[:,0], self.SpinLocations[:,1],
                  sign*self.SpinsXYZ[:,0]     , sign*self.SpinsXYZ[:,1]     ,
                  thetac, alpha=1,cmap=cm, norm=norm, pivot = 'mid',
                  scale=25,
                  # headlength = 2,
                  # headaxislength = 2,
                  headwidth = 6,
                  minlength=minlength,
                  # headaxislength=0.1
                  # minshaft=0.7,
                  linewidth=0.2,
                  width=0.0025,
                  ec='black')

        if (self.NumDefects == 1) and ((self.DefectQuad!=0) or (self.DefectOcto!=0)):
            defe = ptch.Circle(3*self.L1/6 *self.A1 + 3*self.L2/6 *self.A2 +(self.A1+self.A2)/3,
                        radius=self.Lengthscale*2, fill=True, alpha=0.1, linewidth=1.5,color='black')
            ax.add_patch(defe)

        plt.title(f'$J^Q$ = {self.JQuad:.2f}, $J^O$ = {self.JOcto:.2f},\
                    $a^Q$ = {self.DefectQuad:.2f}, $a^O$ = {self.DefectOcto:.2f},\
                    $L$ = {self.Lengthscale:.2f}  ')

        # ax.quiver(self.SpinLocations[:,0], self.SpinLocations[:,1],
        #           sign*self.SpinsXYZ[:,0]     , sign*self.SpinsXYZ[:,1]     ,
        #           thetac, alpha=1,cmap=cm, norm=norm, pivot = 'mid',
        #           scale=sss,
        #           minlength=minlength,
        #           headwidth=headwidth,
        #           linewidth=0.1,
        #           ec='black')

        ax.axis("off")
        # ax.set_facecolor('black')
        ax.axis("equal") #zooms in on arrows

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        cb = plt.colorbar(sm,
            fraction=fraction,
            # pad=0,
            orientation=orientation)
        cb.ax.set_title(r'$\theta_{[111]}\quad \left(^\circ\right)$', usetex=usetex)

        ticks = np.linspace(0,180,6+1)
        cb.set_ticks(ticks)
        if tickaxis=='x':
            cb.ax.set_xticklabels([f'${val:.0f}$' for val in ticks],usetex=usetex)
        elif tickaxis== 'y':
            cb.ax.set_yticklabels([f'${val:.0f}$' for val in ticks],usetex=usetex)

        #adding unit cell used for the calculation
        g = np.zeros(2)
        PlotLineBetweenTwoPoints(ax, g, self.A1)
        PlotLineBetweenTwoPoints(ax, g, self.A2)
        PlotLineBetweenTwoPoints(ax, self.A1, self.A1+self.A2)
        PlotLineBetweenTwoPoints(ax, self.A2, self.A1+self.A2)
        return fig

    def CalculateAndPlotSSF(self, cb_options,usetex):
        '''
        Calculates the static structure factor (SSF) defined as
                s_k/N = 1/N^2 sum_{ij} S_i . S_j exp(-i k.(r_i - r_j) )
        over a Gamma centered k-mesh of the 1st and 2nd crystal BZ. Kmesh consists
        of the accessible momentum points of the cluster, ie.
                      k = m_1 B1 /L1 + m_2 B2 / L2
        where m_i are integers.

        Returns
        fig (numpy.ndarray): figure of the SSF and accessible momentum points
        '''

        B1, B2 = FindReciprocalVectors(self.A1, self.A2)

        addbz=True
        addPoints=False

        KX, KY, fig = KMeshForPlotting(B1, B2, self.L1, self.L2, 3, 3, addbz, addPoints, usetex)
        kx, ky = [np.reshape(x, -1) for x in [KX, KY]]
        k = np.stack((kx, ky)).T
        ax = fig.axes[0]
        scale = 2*pi

        SdotS_mat = np.einsum("ij,kj", self.SpinsXYZ, self.SpinsXYZ)
        s_kflat = np.zeros(len(k), dtype=np.cdouble)
        for i, kv in enumerate(k):
            phase_i = np.exp(1j * np.einsum('i,ji', kv, self.SpinLocations))
            phase_j = np.exp(-1j * np.einsum('i,ji', kv, self.SpinLocations))
            phase_mat = np.einsum('i,j->ij', phase_i, phase_j)
            s_kflat[i] = (SdotS_mat * phase_mat).sum()/ self.Sites**2
        #     ax.annotate(f'$\quad${np.real(s_kflat[i]):.6f}',
        #                 kv/scale,
        #                 fontsize=4,
        #                 color = 'lightgray')
        # print(self.SpinsXYZ)

        ssf = np.reshape(s_kflat, KX.shape)

        ssf_complex_flag = not np.all(np.imag(ssf) < gen_eps)

        if ssf_complex_flag:
            print("Warning: SSF has complex values. Please recheck calculation.")

        ssf = np.sqrt(np.real(ssf))


        #note: we are plotting the real part, since the SSF should be real.

        fraction, orientation, colormap = cb_options
        c = ax.scatter(KX/scale, KY/scale, c=ssf, cmap=colormap, edgecolors="none",zorder=0)
        cbar = fig.colorbar(c, fraction=fraction, orientation=orientation)
        cbar.ax.set_title(r'{\rm Relative} $s_\mathbf{k}$',usetex=usetex,fontsize=9,y=-3)

        ticks = np.linspace(0,1,4+1)
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels([f'${val:.2f}$' for val in ticks],usetex=usetex,fontsize=9)
        # cbar.ax.set_xticklabels([f'${val:.2f}$' for val in ticks],usetex=usetex,fontsize=9)

        return fig

    def CalculateMuonProperties(self):
        line_density = 4
        # create meshgrid

        # oneD1, oneD2 = [ np.arange(0, LL, 1/line_density) for LL in [3, 3] ]
        oneD1, oneD2 = [ np.arange(0, LL, 1/line_density) for LL in [self.L1, self.L2] ]
        n1, n2 = np.meshgrid(oneD1, oneD2)  # grid points indexing a1/a2 direction
        self.MuonX, self.MuonY = (n1 * self.A1[i] + n2 * self.A2[i] + self.SublatticeVectors[0][i] for i in [0,1])  # bends meshgrid into shape of BZ
        x, y = [np.reshape(z, -1) for z in [self.MuonX, self.MuonY]]
        r_muon_list = np.stack((x, y)).T

        #creates list of dipolar fields at each muon point, B = (x, y, z) spatial basis (same as a1, a2)
        r_muon_outofplane = 0/5.6 # 1 = 5.6 Angstroms
        self.MuonPositions = np.pad(r_muon_list, ((0,0),(0,1)),
                                    mode='constant', constant_values=r_muon_outofplane)
        self.BField = np.zeros(np.shape(self.MuonPositions))
        for muon_site, r_muon in enumerate(self.MuonPositions):
            self.BField[muon_site,:] = TotDipolarField(r_muon, self.SpinLocations3D, self.DipoleMoment)
        self.BField_y = self.BField[:,2]
        self.BField_norm = np.linalg.norm(self.BField,axis=1)

        # print(self.MuonPositions.shape)
        # print(self.BField_norm.shape)

    def PlotDipolarField(self):
        '''
        create meshgrid and plot the dipolar fields over it for plotting
        '''
        #selects whether to plot Bz component or the norm, plus reshapes for pcolormesh
        norm_or_y = True
        if norm_or_y: #plot norm if True
            plot = self.BField_norm
            title = r"$|\vec{B}_{dip}|$ (G)"
        else:         #plot outofplane if False
            plot = self.BField_y
            title = r"$B^y_{dip}$ (G)"
        nx = self.MuonX.shape[0]
        ny = self.MuonY.shape[0]
        plot_mat  = np.resize(plot, (nx, ny) )

        #a flag that centers the colormap around zero if it is crossed
        crosseszero = False
        innn = np.nanmin(plot_mat)
        axxx = np.nanmax(plot_mat)
        if (innn < 0) and (axxx>0):
            crosseszero = True
            divnorm=TwoSlopeNorm(vmin=innn, vmax=axxx,vcenter=0.0)

        #create plot
        fig, ax = plt.subplots()
        if crosseszero:
            c = ax.pcolormesh(self.MuonX, self.MuonY, plot_mat, cmap = 'seismic', shading = 'nearest', norm=divnorm)#, ec = 'k', lw=0.1)
        else:
            c = ax.pcolormesh(self.MuonX, self.MuonY, plot_mat, cmap = 'Reds', shading = 'nearest')
        cb = plt.colorbar(c, ax=ax)

        cb.set_label(title)
        ax.axis("off")
        ax.axis("equal") #zooms in on arrows
        plt.title(f'$J^Q$ = {self.JQuad:.2f}, $J^O$ = {self.JOcto:.2f}, $a^Q$ = {self.DefectQuad:.2f}, $a^O$ = {self.DefectOcto:.2f}, $L$ = {self.Lengthscale:.2f}  ')

        #add the triangular lattice
        for x in range(0,self.L1+1):
            for y in range(-1,self.L2):
        #         #used for rhom unit cell
                center1 = x * self.A1 + y * self.A2
                hex2 = ptch.RegularPolygon(
                center1+(a1+a2)/3, 3, 1 / sqrt3, 0, fill=False, linewidth=0.005,color='gray')
                ax.add_patch(hex2)

        #add the defect range
        defe = ptch.Circle(3*self.L1/6 *self.A1 + 3*self.L2/6 *self.A2 +(self.A1+self.A2)/3,
                    radius=self.Lengthscale*2, fill=True, alpha=0.1, linewidth=1.5,color='black')
        ax.add_patch(defe)

        return fig

    # def FilterMuonPositions(self):
    #     Rd  = 2*self.Lengthscale
    #     eps = 1/5.6
    #
    #     rdefect_outofplane = 0
    #     pos_d = 3*self.L1/6 *self.A1 + 3*self.L2/6 *self.A2 +(self.A1+self.A2)/3
    #     pos_dd = np.concatenate((pos_d, np.array([rdefect_outofplane])))
    #
    #     bool = np.linalg.norm(self.MuonPositions - pos_dd, axis=1) <= Rd
    #     filtered_positions = self.MuonPositions[bool]
    #     filtered_fields = self.BField_norm[bool]
    #
    #     print(filtered_positions)
    #
    #     print(filtered_positions.shape)
    #     print(filtered_fields.shape)

        # for i in range(self.SpinLocations3D.shape[0]):
        #     bool = np.linalg.norm(filtered_positions - self.SpinLocations3D[i,:], axis=1) >= eps
        #     filtered_positions = filtered_positions[bool]
        #     filtered_fields = filtered_fields[bool]
        #
        # print(filtered_positions.shape)
        # print(filtered_fields.shape)
        # boolArr = np.linalg.norm(self. - pos_d) > Rd

    def PlotMuonProbability(self):
        Rd  = 2*self.Lengthscale
        eps = 1/5.6

        rdefect_outofplane = 0
        pos_d = 3*self.L1/6 *self.A1 + 3*self.L2/6 *self.A2 +(self.A1+self.A2)/3
        pos_dd = np.concatenate((pos_d, np.array([rdefect_outofplane])))

        bool = np.linalg.norm(self.MuonPositions - pos_dd, axis=1) <= Rd
        filtered_positions = self.MuonPositions[bool]
        filtered_fields = self.BField_norm[bool]

        print(filtered_positions.shape)
        print(filtered_fields.shape)

        for i in range(self.SpinLocations3D.shape[0]):
            bool = np.linalg.norm(filtered_positions - self.SpinLocations3D[i,:], axis=1) >= eps
            filtered_positions = filtered_positions[bool]
            filtered_fields = filtered_fields[bool]

        print(filtered_positions.shape)
        print(filtered_fields.shape)

        bp_min=0.0
        bp_max=1000
        Nbp=10000
        bp_array = np.linspace(bp_min,bp_max,Nbp+1)
        print(bp_array)
        #
        eps = (bp_max - bp_min)/Nbp
        p_array = np.zeros(bp_array.shape, dtype=float)
        for bp_index, bp in enumerate(bp_array):
            for Bdip in filtered_fields:
                p_array[bp_index] += Lorentzian(bp - Bdip, eps)
        p_array = p_array/self.MuonPositions.shape[0]
        # print(p_array)

        fig, ax = plt.subplots()
        # ax.semilogy(bp_array, p_array)
        ax.plot(bp_array, p_array)

        ax.set_xlabel(r"$|\vec{B}_{dip}|$ (G)")
        ax.set_ylabel(r"$p(B')$")

def DipolarField(R_i, R_i_norm, m_i):
    r_i = R_i/R_i_norm
    return (3*r_i.dot(m_i)*r_i - m_i) / (R_i_norm**3)

def TotDipolarField(r_muon, r_i_list, moment_list):

    outer_radius = 100   #to speed up calculation but make sure it includes relevant!!
    inner_radius = 1e-4  #to prevent on-site contributions

    z = 0
    for r_i, m_i in zip(r_i_list, moment_list):
        l_i = np.linalg.norm(r_muon - r_i)
        if (l_i < outer_radius) and (l_i > inner_radius):
            z += DipolarField(r_muon - r_i, l_i, m_i)
        else: # zerto if on-site or outside outer_radius
            z += 0

    coefficient  = 52 #sets units
    return coefficient*z

def Lorentzian(x, eps):
    return eps/2 / (x**2 + (eps/2)**2) / pi

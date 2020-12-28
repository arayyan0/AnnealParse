import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.patches as ptch

from common import pi, sqrt3, gen_eps, a1, a2, AddBZ, FindReciprocalVectors, \
LocalRotation, IndexToPosition, KMeshForPlotting, PlotLineBetweenTwoPoints

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
                (A1 + A2) / 3,
                2 * (A1 + A2) / 3
            ])

        ## rhombus cluster 2, oriented along x bond
        elif sublattice == 2 and type == 2:
            A1, A2 = a1 - a2, a1
            sub_list = np.array([
                (A1 + A2) / 3,
                2 * (A1 + A2) / 3
            ])

        # rectangular cluster 1, oriented along y bond
        elif sublattice == 4 and type == 1:
            A1, A2 = a1, a1 + 2 * (a2 - a1)
            sub_list = np.array([
                (a1 + a2) / 3,
                2 * (a1 + a2) / 3,
                (a1 + a2) / 3 + a2,
                2 * (a1 + a2) / 3 + a2 - a1,
            ])

        # rectangular cluster 2, oriented along z bond
        elif sublattice == 4 and type == 2:
            A1, A2 = a1 - a2, a1 + a2
            sub_list = np.array([
                (a1 + a2) / 3 + (a1 + a2) / 3 - a2,
                2 * (a1 + a2) / 3 + 2 * ((a1 + a2) / 3 - a2),
                (a1 + a2) / 3 + (a1 + a2) / 3 - a2 + a1,
                2 * (a1 + a2) / 3 + 2 * ((a1 + a2) / 3 - a2) + a2,
            ])

    # Kekule cluster
    elif hc_or_kek == 1:
        if sublattice == 6:
            A1, A2 = 2 * a1 - a2, 2 * a2 - a1
            sub_list = np.array([
                (A1 + A2) / 2 + A1 / 3,
                (A1 + A2) / 2 - A2 / 3,
                (A1 + A2) / 2 + (A1 + A2) / 3,
                (A1 + A2) / 2 - A1 / 3,
                (A1 + A2) / 2 + A2 / 3,
                (A1 + A2) / 2 - (A1 + A2) / 3
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
        with open(filename, 'r') as f:
            file_data = f.readlines()

        hc_or_kek, type = [int(x) for x in file_data[2].split()]
        self.S, self.L1, self.L2 = [int(x) for x in file_data[4].split()]
        self.Sites = self.S*self.L1*self.L2
        self.A1, self.A2, self.SublatticeVectors = \
                                        WhichUnitCell(hc_or_kek, type, self.S)

        T_i = np.double(file_data[6])             #initial annealing temperature
        T_f = np.double(file_data[8])               #final annealing temperature
        met_flips = int(file_data[10])              #number of metropolis trials
        det_aligns = int(file_data[12])          #number of deterministic aligns
        self.SimulationParameters = [T_i, T_f, met_flips, det_aligns]

        Kx, Ky, Kz = [float(x) for x in file_data[15].split()]
        Gx, Gy, Gz = [float(x) for x in file_data[17].split()]
        Gp = float(file_data[19].split()[0])
        J1 = float(file_data[21].split()[0])
        self.HamiltonianParameters = [np.array([Kx, Ky, Kz]), np.array([Gx, Gy, Gz]), Gp, J1]

        self.MCEnergyDensity = np.double(file_data[30])

        #could be slow for large clusters, and not necessary if i just want the 
        #energy. consider saving filedata as an attribute and extracting from it
        #through some method self.ReadSpins(), etc.
        self.SpinLocations = np.array(np.empty((self.Sites, 2)))
        self.SpinsXYZ = np.empty((self.Sites, 3))
        for i, line in enumerate(file_data[32:]):
            n1, n2, sub, Sx, Sy, Sz = line.split()
            self.SpinsXYZ[i] = np.array(list(map(np.longdouble,[Sx,Sy,Sz])))
            self.SpinLocations[i] = IndexToPosition(self.A1, self.A2, self.SublatticeVectors,
                                                      map(int,[n1,n2,sub]))

    def CalculateAndPlotSSF(self):
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
        KX, KY, fig = KMeshForPlotting(B1, B2, self.L1, self.L2, 1, 1, True, True)
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
            ax.annotate(f'$\quad${np.real(s_kflat[i]):.3f}',
                        kv/scale,
                        fontsize=4,
                        color = 'lightgray')

        ssf = np.reshape(s_kflat, KX.shape)

        ssf_complex_flag = not np.all(np.imag(ssf) < gen_eps)

        if ssf_complex_flag:
            print("Warning: SSF has complex values. Please recheck calculation.")

        #note: we are plotting the real part, since the SSF should be real.
        c = ax.scatter(KX/scale, KY/scale, c=np.real(ssf), cmap='afmhot', edgecolors="none")
        cbar = fig.colorbar(c, fraction=0.05)
        cbar.ax.set_title(r'$s_\vec{k}/N$')

        AddBZ(ax, scale)
        return fig

    def PlotSpins(self, quiver_options, cb_options):
        '''
        Plots the spins (in the ABC basis) over the honeycomb plane, with color
        indicating the angle made with the vector c* = (1,1,1)/sqrt(3) in
        Cartesian basis

        Returns
        fig (numpy.ndarray): figure of the spin configuration over the , as well
                             as the unit cell used to construct the cluster
        '''
        xyz_to_abc = LocalRotation(np.array([1,1,1])/sqrt3)
        self.SpinsABC = np.einsum('ab,cb->ca', xyz_to_abc, self.SpinsXYZ)

        sss, minlength, headwidth = quiver_options
        fraction, orientation = cb_options

        theta = np.arccos(np.clip(self.SpinsABC[:, 2], -1, 1)) / pi * 180
        norm = clr.Normalize(vmin=0,vmax=180)
        cm = 'viridis'

        fig, ax = plt.subplots()
        ax.quiver(self.SpinLocations[:,0], self.SpinLocations[:,1],
                  self.SpinsABC[:,0]     , self.SpinsABC[:,1]     ,
                  theta, cmap=cm, norm=norm, pivot = 'mid',
                  scale=sss,
                  minlength=minlength,
                  headwidth=headwidth)
        ax.axis("off")
        ax.axis("equal")
        # ax.set_facecolor('black')

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        cb = plt.colorbar(sm,
            fraction=fraction,
            # pad=0,
            orientation=orientation)
        cb.ax.set_title(r'$\theta_{\mathbf{c}^*}$')

        #adding honeycomb plane
        for x in range(-1,self.L1+1):
            for y in range(-1,self.L2+1):
                center1 = x * self.A1 + y * self.A2
                hex2 = ptch.RegularPolygon(
                center1+a1, 6, 1 / sqrt3, 0, fill=False, linewidth=0.2,color='black')
                ax.add_patch(hex2)

        #adding unit cell used for the calculation
        g = np.zeros(2)
        PlotLineBetweenTwoPoints(ax, g, self.A1)
        PlotLineBetweenTwoPoints(ax, g, self.A2)
        PlotLineBetweenTwoPoints(ax, self.A1, self.A1+self.A2)
        PlotLineBetweenTwoPoints(ax, self.A2, self.A1+self.A2)
        return fig

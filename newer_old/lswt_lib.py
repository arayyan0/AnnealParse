import numpy as np
from scipy import linalg as LA
import itertools as it
import matplotlib.pyplot as plt
import time

from spin_lib import AnnealedSpinConfiguration
from common import a1, a2, b1, b2, pi, gen_eps, LocalRotation, FindReciprocalVectors,\
IndexToPosition, KMeshForIntegration, KMeshForPlotting, PlotLineBetweenTwoPoints,\
AddBZ, RotateIn2D, EZhangBZ


class LSWT(AnnealedSpinConfiguration):
    '''
    A class which performs the linear spin wave theory on an ordered spin
    configuration.

    Note: Only works if the cluster used contains ONE magnetic unit cell.

    Attributes
    S                       (float): the length of the polarized moment s.t. S^2 = |S_i|^2
    Hx, Hy, Hz      (numpy.ndarray): x, y, and z bond Hamiltonians of shape (3,3)
    T1, T2          (numpy.ndarray): magnetic unit cell vectors of shape(2,)
    NumUniqueBonds            (int): number of unique bonds in the magnetic unit cell
    BondIndices     (numpy.ndarray): array of site indices (s1, s2) of shape
                                     (2*NumUniqueBonds, 2)
    BondType        (numpy.ndarray): bond type 'x','y','z', of shape (2*NumUniqueBonds,)
    BondDiffs       (numpy.ndarray): relative position between two sites, of shape
                                     (2*NumUniqueBonds,2)
    BondTransH      (numpy.ndarray): the bond Hamiltonians in the rotated local basis,
                                     array is of shape (2*NumUniqueBonds, 3, 3)
    AElements       (numpy.ndarray): elements of transverse density-like matrix in
                                     real space, of shape (2*NumUniqueBonds,)
    BElements       (numpy.ndarray): elements of transverse pairing-like matrix in
                                     real space, of shape (2*NumUniqueBonds,)
    HzzElements     (numpy.ndarray): elements of longitudinal density-like matrix in
                                     real space, of shape (2*NumUniqueBonds,)
    LSWTEnergyDensity   (np.double): energy per site as calculated within LSWT
    LocalMagneticField  (np.double): local magnetic field to occupy diagonal of BdG
                                     Hamiltonian
    PseudoMatrixG   (numpy.ndarray): G matrix = diag(1, 1, ... -1, -1), of shape
                                     (2*Sites, 2*Sites)
    Dispersions     (numpy.ndarray): positive magnon frequencies of shape
                                     (len(klst), 2*Sites, 2*Sites)
    Diagonalizer    (numpy.ndarray): diagonalizer of the Hamiltonian of shape
                                     (len(klst), 2*Sites, 2*Sites)
    CholeskyFailure          (bool): whether the Cholesky decomposition failed at
                                     some k point (True) or not (False)
    ReducedMoment           (float): the reduced moment as defined in
    BosonWF         (numpy.ndarray): magnon wave function, shape (2*Sites, Sites)
    '''
    def __init__(self, filename):
        '''
        Purpose
        Initializes LSWT instance inherited from the AnnealedSpinConfiguration class.

        Parameters
        filename
        '''
        super().__init__(filename)
        self.CreateHamiltonians()

        self.MomentSize = 1/2

        self.T1, self.T2 = self.L1*self.A1, self.L2*self.A2

        self.NumUniqueBonds = int(self.Sites*3/2) #easily demonstrated for the hc lattice
        self.BondIndices = np.empty((2*self.NumUniqueBonds, 2), dtype=int)
        self.ExtractMomentsAndPositions()

        self.BondDiffs   = np.empty((2*self.NumUniqueBonds, 2), dtype=np.double)
        self.BondType    = np.empty((2*self.NumUniqueBonds,  ), dtype=np.string_)
        self.BondTransH  = np.empty((2*self.NumUniqueBonds, 3, 3), dtype=np.double)
        self.DetermineBondCharacteristics()

        self.Construct1DArrays()

        self.LocalMagneticField = np.array([ np.sum(
                                             self.HzzElements[3*i:3*(i+1)]
                                             ) for i in range(self.Sites) ])

        self.CalculateBondEnergies()

        self.EquilibriumCheck = np.array([ np.sum(
                                             self.FElements[3*i:3*(i+1)]
                                             ) for i in range(self.Sites) ])
        # print("equilibrium check: ", np.sum(self.EquilibriumCheck))

        self.LSWTEnergyDensity = np.sum(self.LocalMagneticField)/2/self.Sites
        energy_flag = np.abs(self.MCEnergyDensity - self.LSWTEnergyDensity) > gen_eps
        if energy_flag:
            print("Warning: Your calculated classical energy density does not the match\n\
                   the Monte Carlo result. This MUST be fixed before moving on.")

        ones = np.ones(self.Sites)
        zeros = np.zeros((self.Sites,self.Sites))
        self.PseudoMatrixG = np.diag(np.concatenate((ones, -ones)))
        self.PermMatrixP = np.block([[zeros,np.diag(ones)],[np.diag(ones),zeros]])
        self.ProjectorMatrix = np.block([[zeros,zeros],[zeros,np.diag(ones)]])

    def CreateHamiltonians(self):
        '''
        Creates the x, y, and z bond Hamiltonians.
        '''
        [Kx, Ky, Kz], [Gx, Gy, Gz], Gp, J1 = self.HamiltonianParameters
        self.Hx = np.array([[Kx + J1, Gp, Gp],
                            [Gp, J1, Gx],
                            [Gp, Gx, J1]])
        self.Hy = np.array([[J1, Gp, Gy],
                            [Gp, Ky + J1, Gp],
                            [Gy, Gp, J1]])
        self.Hz = np.array([[J1, Gz, Gp],
                            [Gz, J1, Gp],
                            [Gp, Gp, J1 + Kz]])


    def DetermineBondCharacteristics(self):
        '''
        Determines whether the sublattice indices s1 and s2 refer to
        nearest-neighbour bonds, and if so, whether they are of type x, y, or z.
        At the same time, it calculates the bond Hamiltonian in the local basis
                R(i).H(i,j).(R(j).T) where R(l).S(l) = (0,0,1)
        and determines the relative distance between the two sites
                                r(s1) - r(s2)
        which points from site s2 to site s1.
        '''
        Ainv = np.linalg.inv(np.array([self.T1, self.T2]).T)

        bond_counter = 0
        lst = it.product(range(self.Sites), range(self.Sites))
        for [s1, s2] in lst:
            d = self.SpinLocations[s1] - self.SpinLocations[s2]
            for whichd, H, type in zip([dx, dy, dz], [self.Hx, self.Hy, self.Hz], ['x','y','z']):
                #checks if d = +/- d(x,y,z) + R, where R is (magnetic) Bravais unit vectors
                nplus, nminus = [Ainv @ (d-sign*whichd) for sign in [+1, -1]]
                bool1, bool2 = [
                                (abs(n[0] - round(n[0])) < gen_eps) and \
                                (abs(n[1] - round(n[1])) < gen_eps)
                                for n in [nplus, nminus]
                               ]

                if bool1:
                    self.BondIndices[bond_counter,:] = [s1, s2]
                    self.BondDiffs[bond_counter] = +1*whichd
                    self.BondType[bond_counter] = type
                    Ri, Rj = map(LocalRotation,[self.SpinsXYZ[s1],self.SpinsXYZ[s2]])
                    self.BondTransH[bond_counter] = Ri @ H @ Rj.T
                    bond_counter += 1

                elif bool2:
                    self.BondIndices[bond_counter,:] = [s1, s2]
                    self.BondDiffs[bond_counter] = -1*whichd
                    self.BondType[bond_counter] = type
                    Ri, Rj = map(LocalRotation,[self.SpinsXYZ[s1],self.SpinsXYZ[s2]])
                    self.BondTransH[bond_counter] = Ri @ H @ Rj.T
                    bond_counter += 1

    def Construct1DArrays(self):
        '''
        Constructs the 1D arrays AElements, BElements, and HzzElements
        that are used to construct the BdG Hamiltonian. For each bond, the element
        of either the A or B matrix is given by
                        A = ( Hxx + Hyy - i (Hxy - Hyx) )/2
                        B = ( Hxx - Hyy + i (Hxy + Hyx) )/2
        '''
        self.AElements = 0.5*(
                              (self.BondTransH[:, 0, 0] + self.BondTransH[:, 1, 1])
                       - 1j * (self.BondTransH[:, 0, 1] - self.BondTransH[:, 1, 0])
                       )
        self.BElements = 0.5*(
                              (self.BondTransH[:, 0, 0] - self.BondTransH[:, 1, 1])
                       + 1j * (self.BondTransH[:, 0, 1] + self.BondTransH[:, 1, 0])
                       )
        self.HzzElements = self.BondTransH[:, 2, 2]

        self.FElements = self.BondTransH[:, 0, 2] + 1j*self.BondTransH[:, 1, 2]

    def CalculateBondEnergies(self):
        '''
        Calculates energy contribution from the x, y, and z bonds.
        '''
        [Kx, Ky, Kz], [Gx, Gy, Gz], Gp, J1= self.HamiltonianParameters
        Exarray, Eyarray, Ezarray = [], [], []
        for i in range(2*self.NumUniqueBonds):
            bond_type = self.BondType[i]
            e = self.HzzElements[i]
            if bond_type == b'x':
                Exarray.append(e)
            elif bond_type ==b'y':
                Eyarray.append(e)
            elif bond_type==b'z':
                Ezarray.append(e)

        self.Ex,self.Ey,self.Ez = [np.sum(Earray)/2/self.Sites for Earray in [Exarray, Eyarray, Ezarray]]


    def ObtainMagnonSpectrumAndDiagonalizer(self, klst, angle, offset):
        '''
        For each momentum vector in klst, fill the corresponding BdG matrix and
        obtain the spectrum + diagonalizer via Colpa's algorithim. Will fail
        at k-points where the spectrum goes to zero.

        Diagonalizer and Dispersions are defined such that
                np.conj(Diagonalizer.T) H Diagonalizer = diag(Dispersions)

        Parameters:
        klst (list-like): list of momentum vectors
        angle    (float): rotate momentum points by an angle to cut through different
                          paths of reciprocal space
        offset   (float): adding a small scalar to the diagonal of the BdG Hamiltonian
                          to lift the spectrum, in the case of zero-eigenvalues
        '''
        R = RotateIn2D(angle)
        Adiag = np.diag(self.LocalMagneticField)
        self.Diagonalizer = np.empty((len(klst), 2*self.Sites,2*self.Sites), dtype=np.clongdouble)
        self.Dispersions = np.empty((len(klst), 2*self.Sites), dtype=np.longdouble)
        cholesky_fail = []
        for i, kp in enumerate(klst): # i indexes the k point
            k = np.dot(R, np.array(kp))
            phase = np.exp(-1j * np.dot(self.BondDiffs, k.T))
            a_matrix_elements = self.AElements*phase
            b_matrix_elements = self.BElements*phase
            e_matrix_elements = self.AElements*phase.conj()

            Ak, Bk, Ek = np.zeros((3, self.Sites, self.Sites), dtype=complex)
            for j in range(2*self.NumUniqueBonds): #j indexes the bond.
                s1, s2 = self.BondIndices[j]
                Ak[s1, s2] += a_matrix_elements[j] #- self.HzzElements[j]
                Bk[s1, s2] += b_matrix_elements[j]
                Ek[s1, s2] += e_matrix_elements[j]
            Ak = Ak - Adiag + offset * np.eye(*Ak.shape)
            Ek = Ek - Adiag + offset * np.eye(*Ak.shape)

            Mk = np.block([
                [Ak           , Bk  ],
                [Bk.T.conj(), Ek.T]
            ])/2


            try:
                Hk = LA.cholesky(Mk)
            except LA.LinAlgError:
                cholesky_fail.append(True)
                print(f"Cholesky decomposition has failed at k = {k}.")
            else:
                cholesky_fail.append(False)
                tobediagonalized = Hk @ self.PseudoMatrixG @ Hk.T.conj()
                Lpk, Upk = np.linalg.eigh(tobediagonalized)

                idx = np.argsort(Lpk)[::-1]
                Lk, Uk = Lpk[idx], Upk[:,idx]

                Lk[:self.Sites] = Lk[:self.Sites][::-1]
                Uk[:,:self.Sites] = np.flip(Uk[:, :self.Sites], axis=1)
                # print(Lk)

                Wk = self.PseudoMatrixG @ np.diag(Lk)
                # print(Lk, Wk)

                # print(Lk)
                # print(np.diag(Wk))
                self.Dispersions[i] = np.diag(Wk)
                self.Diagonalizer[i] = LA.inv(Hk) @ Uk @ np.sqrt(Wk)


                # print(Lk)
                # print(np.allclose(self.Diagonalizer[i].T.conj() @ Mk @ self.Diagonalizer[i], Wk))
                # print(np.allclose(self.Diagonalizer[i].T.conj() @ self.PseudoMatrixG @ self.Diagonalizer[i], self.PseudoMatrixG))
                # print(np.allclose(self.Diagonalizer[i] @ np.diag(Lk) @ LA.inv(self.Diagonalizer[i]) , self.PseudoMatrixG @ Mk))
                # print(self.Diagonalizer[i,:,0])
                # print(self.Diagonalizer[i,:,0+self.Sites])
                # print(self.Dispersions[i])
                # print(self.Diagonalizer[i,:,0])
                # print(self.Diagonalizer[i,:,self.Sites])
                # print(self.Diagonalizer[i,:,0])
                # print((self.PermMatrixP @ self.Diagonalizer[i] @ self.PermMatrixP)[0,:])

                # print(self.Diagonalizer[i,0,:self.Sites])
                # print(self.Diagonalizer[i,self.Sites:,self.Sites:])
                # print("----------------")
        self.CholeskyFailure = np.any(np.array(cholesky_fail,dtype=bool))

    def PlotMagnonKPath(self, sympoints,lift):
        '''
        Obtain the spectrum along the given kpath.

        Parameters
        sympoints (common.SymmetryPoints): kpath created by SymmetryPoints.MakeKPath()

        Returns
        fig           (matplotlib.Figure): plot of magnon dispersion along kpath
        '''
        kpath, tick_mark, sym_labels, _ = sympoints
        n_rot=0
        self.ObtainMagnonSpectrumAndDiagonalizer(kpath, n_rot*2*pi/3, lift)
        n = len(kpath)
        fig, ax = plt.subplots()
        ax.plot(range(len(kpath)), self.Dispersions[:,:self.Sites])

        ax.axhline(y=0, color='0.75', linestyle=':')
        ax.set_ylabel(r'$\frac{\omega_{\mathbf{k}}^s}{J}$', rotation=0,labelpad=10)
        plt.xticks(tick_mark, sym_labels, usetex=True)
        if self.CholeskyFailure:
            plt.title("warning: Cholesky decomposition failed")

        plt.vlines(tick_mark,0,np.max(self.Dispersions),linestyles=':',color='0.75')
        return fig

    def CalculateMagnonProperties(self, L1, L2):
        '''
        Obtains the magnon wave function over the unit cell of the "first BZ"
                        k = n1 B1 /L1 + n2 B2 /L2, ni=0,1,2,...Li-1
        and then calculated the reduced moment according to...
                          (MUST INCLUDE AT LATER POINT)
        Parameters
        L1, L2  (ints): density of mesh in B1 and B2 directions.
        '''
        B1, B2 = FindReciprocalVectors(self.T1, self.T2)
        KX, KY = KMeshForIntegration(B1, B2, L1, L2)
        # KX, KY = KMeshForIntegration(B1, B2, 10, 10)
        kx, ky = [np.reshape(x, -1) for x in [KX, KY]]
        k = np.stack((kx, ky)).T

        # k = EZhangBZ(B1, B2, L1, L2)
        # print(len(k))

        self.ObtainMagnonSpectrumAndDiagonalizer(k, 0, 0)


        if self.CholeskyFailure:
            self.ReducedMoment = np.nan
            self.MagnonGap = np.nan
            self.SWEnergyCorrection = np.nan
            self.SWEnergy= np.nan

        else:
            total_moment_lst = []
            energy_lst = []
            for j in range(self.Sites):
                energy = 0
                total_moment = 0
                for i, kv in enumerate(k):
                    u = self.Diagonalizer[i, :self.Sites, j]
                    v = self.Diagonalizer[i, self.Sites:, j]
                    total_moment += v.conj() @ v
                    energy += self.Dispersions[i, j]
                energy_lst.append(energy)
                total_moment_lst.append(total_moment)
            # print("energy branches:", self.S*np.array(energy_lst)/len(k)/self.Ssites)
            # print("moment reduction:", np.array(total_moment_lst)/len(k)/self.S)

            self.ReducedMoment = 1-np.sum(np.array(total_moment_lst)/len(k)/self.Sites/self.MomentSize)

            self.SWEnergyCorrection = np.sum(self.MomentSize*np.array(energy_lst)/len(k)/self.Sites)
            self.SWEnergy = self.MCEnergyDensity*self.MomentSize*(self.MomentSize+1) + self.SWEnergyCorrection

            self.MagnonGap = np.min(self.Dispersions[:,0])

    def PlotLowestBand(self, n1, n2, m1, m2, lift):
        '''
        Plots the lowest (physical) magnon band over the reciprocal space.

        Parameters
        n1, n2     (int): density of mesh in B1 and B2 directions.
        m1, m2     (int): copies in B1 and B2 directions.
        lift     (float): adding a small scalar to the diagonal of the BdG Hamiltonian
                          to lift the spectrum, in the case of zero-eigenvalues
        '''
        B1, B2 = FindReciprocalVectors(self.T1, self.T2)
        KX, KY, fig = KMeshForPlotting(B1, B2, n1*self.L1, n2*self.L2, m1, m2, True, False)
        ax = fig.axes[0]

        kx, ky = [np.reshape(x, -1) for x in [KX, KY]]
        k = np.stack((kx, ky)).T

        self.ObtainMagnonSpectrumAndDiagonalizer(k, 0, lift)
        lowest_band = np.reshape(self.Dispersions[:,0], KX.shape)
        if self.CholeskyFailure:
            plt.title("beware: cholesky decomposition failed at some isolated points")
        scale = 2*np.pi
        try:
            c = ax.pcolormesh(KX/scale, KY/scale, np.real(lowest_band), cmap='afmhot')
            cbar = fig.colorbar(c, fraction=0.05)
            cbar.ax.set_title(r'$\omega_{\vec{k},0}$')
        except:
            c = ax.pcolormesh(KX/scale, KY/scale, np.zeros(lowest_band.shape), cmap='afmhot')
        else:
            g = np.zeros((2,))
            PlotLineBetweenTwoPoints(ax, g, B1/scale)
            PlotLineBetweenTwoPoints(ax, g, B2/scale)
            ax.annotate('B1', B1/scale,fontsize=10,color="white" )
            ax.annotate('B2', B2/scale,fontsize=10,color="white" )

            ax.axis("equal")
            ax.set_aspect('equal')

            AddBZ(ax, scale)
        return fig

#------------------------Some standard global variables------------------------#
# vectors along each bond of the honeycomb lattice
dz = -(a1 + a2) / 3
dx, dy = [a1 + dz, a2 + dz]

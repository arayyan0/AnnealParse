# file   lib_new_parse.py
# author Ahmed Rayyan
# date   April 23, 2020
# brief  Collection of functions for parsing SA data

from math import sqrt
import numpy as np
import scipy.linalg as la
from scipy.signal import find_peaks
import itertools as it
import time
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.patches as ptch
import glob as glb

from matplotlib.colors import LinearSegmentedColormap
basic_cols=['#ff0000', '#aeaeae', '#0051ff']
my_cmap=LinearSegmentedColormap.from_list('mycmap', basic_cols)

np.set_printoptions(precision=5, suppress=True)

pi = np.pi

a1, a2 = np.array([1 / 2, sqrt(3) / 2]), np.array([-1 / 2, sqrt(3) / 2])
dz = -(a1 + a2) / 3
dx, dy = [a1 + dz, a2 + dz]


def WhichUnitCell(type, sublattice, version):
    # selects the unit cell information:
    #    translational vectors (ie. geometry)
    #    location of sublattices
    #    colors for each sublattice
    if type == 0:
        if sublattice == 2:
            if version == 1:
                # Rhombic cluster 1 (encloses z bonds)
                T1, T2 = a1, a2
                sub_list = np.array([
                    (T1 + T2) / 3,
                    2 * (T1 + T2) / 3
                ])
            if version == 2:
                # Rhombic cluster 2 (encloses x bonds)
                T1, T2 = a1 - a2, a1
                sub_list = np.array([
                    (T1 + T2) / 3,
                    2 * (T1 + T2) / 3
                ])
            color = np.array(["r", "c"])
        if sublattice == 4:
            if version == 1:
                # Rectangular cluster 1 (encloses y bonds)
                T1, T2 = a1, a1 + 2 * (a2 - a1)
                sub_list = np.array([
                    (a1 + a2) / 3,
                    2 * (a1 + a2) / 3,
                    (a1 + a2) / 3 + a2,
                    2 * (a1 + a2) / 3 + a2 - a1,
                ])
            if version == 2:
                # Rectangular cluster 2 (encloses z bonds)
                T1, T2 = a1 - a2, a1 + a2
                sub_list = np.array([
                    (a1 + a2) / 3 + (a1 + a2) / 3 - a2,
                    2 * (a1 + a2) / 3 + 2 * ((a1 + a2) / 3 - a2),
                    (a1 + a2) / 3 + (a1 + a2) / 3 - a2 + a1,
                    2 * (a1 + a2) / 3 + 2 * ((a1 + a2) / 3 - a2) + a2,
                ])
            color = np.array(["tab:red", "tab:cyan", "tab:orange", "tab:blue"])
    elif type == 1:
        if sublattice == 6:
            # Kekule cluster
            T1, T2 = 2 * a1 - a2, 2 * a2 - a1
            sub_list = np.array([
                (T1 + T2) / 2 + T1 / 3,
                (T1 + T2) / 2 - T2 / 3,
                (T1 + T2) / 2 + (T1 + T2) / 3,
                (T1 + T2) / 2 - T1 / 3,
                (T1 + T2) / 2 + T2 / 3,
                (T1 + T2) / 2 - (T1 + T2) / 3
            ])
            color = np.array(["r", "b", "g", "c", "y", "p"])
    return T1, T2, sub_list, color


def WhichOrder(order,cluster):
    #################################FM/AFM####################################
    if order == "FM" and cluster == "rh2":
        #note: specifically, l1 = 1, l2 = 1
        T1, T2 = (a1 - a2), a1
        A, B = (2*a1-a2)/3, 2*(2*a1-a2)/3
        sub_list = np.array([A,
                             B])
    #################################ZZ####################################
    # elif order == "ZZ" and cluster == "rh1":
    #     #note: specifically, l1 = 2, l2 = 1, M1 configuration
    #     T1, T2 = 2*a1, a2
    #     A, B = (a1 + a2) / 3, 2 * (a1 + a2) / 3
    #     sub_list = np.array([A,
    #                          B,
    #                          A + a1,
    #                          B + a1])

    elif order == "ZZ" and cluster == "rh2":
        #note: specifically, l1 = 1, l2 = 2, M2 configuration
        T1, T2 = a1 - a2, 2*a1
        A, B = (2*a1-a2)/3, 2*(2*a1-a2)/3
        sub_list = np.array([A,
                             B,
                             A + a1,
                             B + a1])
    #
    # elif order == "ZZ" and cluster == "rh1":
    #     #note: specifically, l1 = 1, l2 = 2, M3 configuration
    #     T1, T2 = a1, 2*a2
    #     A, B = (a1 + a2) / 3, 2 * (a1 + a2) / 3
    #     sub_list = np.array([A,
    #                          B,
    #                          A + a2,
    #                          B + a2])
    #################################6sites####################################
    #
    # elif order == "6" and cluster == "rh1":
    #     #note: specifically, l1 = 3, l2 = 1
    #     T1, T2 = 3*a1, a2
    #     A, B = (a1+a2)/3, 2*(a1+a2)/3
    #     sub_list = np.array([A,
    #                          B,
    #                          A+a1,
    #                          B+a1,
    #                          A+2*a1,
    #                          B+2*a1])

    elif order == "6" and cluster == "rh2":
        #note: specifically, l1 = 1, l2 = 3
        T1, T2 = a1 - a2, 3*a1
        A, B = (2*a1-a2)/3, 2*(2*a1-a2)/3
        sub_list = np.array([A,
                             B,
                             A+a1,
                             B+a1,
                             A+2*a1,
                             B+2*a1])

    # elif order == "6" and cluster == "rh1":
    #     #note: specifically, l1 = 1, l2 = 3
    #     T1, T2 = a1, 3*a2
    #     A, B = (a1+a2)/3, 2*(a1+a2)/3
    #     sub_list = np.array([A,
    #                          B,
    #                          A+a2,
    #                          B+a2,
    #                          A+2*a2,
    #                          B+2*a2])
    #################################18sites####################################
    elif order == "18" and cluster == "rh1":
        T1, T2 = 3*a1, 3*a2
        A, B = (a1+a2)/3, 2*(a1+a2)/3
        sub_list = np.array([A + 0*a1 + 0*a2,
                             B + 0*a1 + 0*a2,
                             A + 1*a1 + 0*a2,
                             B + 1*a1 + 0*a2,
                             A + 2*a1 + 0*a2,
                             B + 2*a1 + 0*a2,
                             A + 0*a1 + 1*a2,
                             B + 0*a1 + 1*a2,
                             A + 1*a1 + 1*a2,
                             B + 1*a1 + 1*a2,
                             A + 2*a1 + 1*a2,
                             B + 2*a1 + 1*a2,
                             A + 0*a1 + 2*a2,
                             B + 0*a1 + 2*a2,
                             A + 1*a1 + 2*a2,
                             B + 1*a1 + 2*a2,
                             A + 2*a1 + 2*a2,
                             B + 2*a1 + 2*a2])
    #################################other####################################

    # elif order == "zz" and cluster == "rh2":
    #     #note: specifically, l1 = 2, l2 = 1
    #     T1, T2 = 2*(a1 - a2), a1
    #     A, B = (2*a1-a2)/3, 2*(2*a1-a2)/3
    #     sub_list = np.array([A,
    #                          B,
    #                          A + T1/2,
    #                          B + T1/2])

    # elif order == "zz" and cluster == "re2":
    #     #note: specifically, l1 = 1, l2 = 1
    #     T1, T2 = a1 - a2, a1 + a2
    #     A, B = (a1 + a2) / 3 - dy, 2 * (a1 + a2) / 3 - 2 * dy
    #     sub_list = np.array([A,
    #                          B,
    #                          A + a1,
    #                          B + a2])
    # elif order == "12":
    #     T1, T2 = 3 * (a1 - a2), a1 + a2
    #     A, B = (a1 + a2) / 3 - dy, 2 * (a1 + a2) / 3 - 2 * dy
    #     sub_list = np.array([A,
    #                          B,
    #                          A + a1,
    #                          B + a2,
    #                          A + T1 / 3,
    #                          B + T1 / 3,
    #                          A + a1 + T1 / 3,
    #                          B + a2 + T1 / 3,
    #                          A + 2 * T1 / 3,
    #                          B + 2 * T1 / 3,
    #                          A + a1 + 2 * T1 / 3,
    #                          B + a2 + 2 * T1 / 3])

    else:
        print("put new cluster in WhichOrder function")
    return T1, T2, sub_list


def FindReciprocalVectors(a1, a2):
    z = np.zeros((2, 2))
    A = np.block([[np.array([a1, a2]), z], [z, np.array([a1, a2])]])
    b = np.array([2 * pi, 0, 0, 2 * pi])
    x = la.solve(A, b)
    b1, b2 = x[:2], x[2:4]
    return b1, b2
#


def LocalRotation(spin):
    # given spin, will return matrix U such that (0,0,1) = U.(x,y,z)
    # for (Sx, Sy, Sz) = (1, 1, 1)/sqrt(3), U is xyz to abc* change of basis
    Sx, Sy, Sz = spin
    Sperp = la.norm(spin[:2])
    if Sperp != 0:
        U = np.array([
            [Sz * Sx / Sperp, Sz * Sy / Sperp, -Sperp],
            [-Sy / Sperp, Sx / Sperp, 0],
            [Sx, Sy, Sz]
        ])
    else:
        U = np.array([
            [Sz * np.sign(Sx), Sz * np.sign(Sy), -Sperp],
            [-np.sign(Sy), np.sign(Sx), 0],
            [Sx, Sy, Sz]
        ])
    return U


def IndexToPosition(T1, T2, sv, indices):
    n1, n2, s = indices
    z = n1 * T1 + n2 * T2 + sv[s]
    return z


def EquivalentBravais(T1, T2, x, y):
    # Takes two vectors x and y and sees if they can be written as a linear combination
    # of two Bravais vectors T1 and T2.
    # n: array [n1, n2] such that x-y = n1*T1 + n2*T2
    # bool: variable which, if True, means that it n1, n2 are INTEGER. if not,
    # they are REAL.
    T = np.array([T1, T2]).T
    n = la.solve(T, x - y)

    eps = 10**(-10)  # careful with this, as always
    bool = (abs(n[0] - round(n[0])) < eps) and (abs(n[1] - round(n[1])) < eps)
    return n, bool

# ------------------------------------------------------------------------------


def Path(A, B, n):
    # Function...
    # creates a list running from vector A to vector B
    # ie. 2d version of linspace along the line B-A

    # Input...
    # A, B. arrays. initial and final vectors in list
    # n. integer. initial number of points in list (recalculated)

    # Output...
    # z, array. list of vectors along line B-A
    # n, integer. new number of points in list
    n = int(la.norm(B - A) * n)  # takes into account length of difference
    run = B[0] - A[0]
    if run != 0:  # if line is not vertical (slope well-defined)
        x = np.linspace(A[0], B[0], n)
        m = (B[1] - A[1]) / run
        b = B[1] - m * B[0]
        y = m * x + b
    else:  # if line is vertical (slope ill-defined)
        y = np.linspace(A[1], B[1], n)
        x = np.array([B[0]] * n)
    z = np.stack((x, y)).T
    return z, n


def NewMakeGrid(b1, b2, L1, L2, n):
    oneD1 = np.array(range(-n * L1, n * L1+1))  # 1d list of points
    oneD2 = np.array(range(-n * L2, n * L2+1))  # 1d list of points
    n1, n2 = np.meshgrid(oneD1, oneD2)  # grid points indexing G1/G2 direction
    KX = n1 * b1[0] / L1 + n2 * b2[0] / L2  # bends meshgrid into shape of BZ
    KY = n1 * b1[1] / L1 + n2 * b2[1] / L2

    fig, ax = plt.subplots()
    ax.plot(KX, KY, '+',c='grey')
    # ax.plot([0, b1[0]], [0, b1[1]], color='black', linestyle='-', linewidth=2)
    # ax.plot([0, b2[0]], [0, b2[1]], color='black', linestyle='-', linewidth=2)
    ax.set_aspect('equal')
    # ax.set_ylim("off")
    # ax.axis("equal")
    # ax.axis(xticks="off")
    # ax.set_xlabel(r'$k_x$')
    # ax.set_ylabel(r'$k_y$')
    ax.set_facecolor('black')

    return KX, KY, fig
# ------------------------------------------------------------------------------


def ExtractEnergyFromFile(file):
    with open(file, 'r') as f:
        file_data = f.readlines()
    # depends on file MAY FAIL, can be fixed by passing position as arg
    e = float(file_data[30])
    return e


def ExtractEnergyFromFolder(folder, position):
    # position is index of the parameter value in the file
    x_list, e_list = [], []

    file_list = glb.glob(folder + "*")
    for file in file_list:
        x = float(file.split("_")[position])
        with open(file, 'r') as f:
            file_data = f.readlines()
        e = ExtractEnergyFromFile(file)
        x_list.append(x)
        e_list.append(e)

    idx = np.argsort(x_list)
    x_list = np.array(x_list)[idx]
    e_list = np.array(e_list)[idx]
    return x_list, e_list
# ------------------------------------------------------------------------------


class ReciprocalLattice:
    def __init__(self):
        self.b1, self.b2 = FindReciprocalVectors(a1, a2)

        list = [(0, 0), (1 / 2, 0), (1 / 2, 1 / 2), (0, 1 / 2),
                (2 / 3, 1 / 3), (1 / 3, 2 / 3), (-1 / 3 , 1 / 3), (1, 0), (1, 1),(-1/2,1/2)]
        self.G, self.M1, M2, M3, self.K, self.Kp, self.Kpp, Gp1, Gp2, X = [IndexToPosition(
            self.b1, self.b2, [0], [*i, 0]) for i in list]

        # print(self.M1)
        # print(M2)
        # print(M3)

        self.Sym = {
            "G": self.G,
            "M1": self.M1,
            "-M1": -self.M1,
            "K": self.K,
            "M2": M2,
            "Kp": self.Kp,
            "M3": M3,
            "Gp1": Gp1,
            "Gp2": Gp2,
            "Kpp": self.Kpp,
            "X":X}
        self.SymTeX = {
            "G": r"$\Gamma$",
            "M1": "$M_1$",
            "-M1": "$-M_1$",
            "K": "$K$",
            "M2": "$M_2$",
            "Kp": "$K'$",
            "M3": "$M_3$",
            "Gp1": r"$\Gamma'_1$",
            "Gp2": "$\Gamma'_2$",
            "Kpp": r"$K''$",
            "X": r"$X$"}
        # print(self.M1, self.K)

    def MakeKPath(self, sym_points, n):
        sym_labels = list(map(self.SymTeX.get, sym_points))
        sym_values = list(map(self.Sym.get, sym_points))
        shifted_sym_values = sym_values[1:] + sym_values[:1]
        pairs = list(zip(sym_values, shifted_sym_values))
        del pairs[-1]

        kpath, tick_mark, l = [], [0], 0
        for i, pair in enumerate(pairs):
            path, length = Path(*pair, n)
            kpath.append(path)
            l = l + length
            tick_mark.append(l)
        return np.concatenate(np.array(kpath,dtype=object)), tick_mark, sym_labels
# ------------------------------------------------------------------------------


class LSWT:
    # class that performs the linear spin wave analysis
    # takes in the SA result on a SINGLE magnetic unit cell
    def __init__(self, h_parameters, c_parameters, s_config):
        self.CreateHamiltonians(h_parameters)

        self.Spins = s_config
        self.Sites = s_config.shape[0]

        self.Order, self.Cluster = c_parameters
        self.T1, self.T2, self.SublatticeVectors = WhichOrder(self.Order, self.Cluster)


        self.BondIndices, self.BondHamiltonians, self.BondDiffs, self.BondType = [], [], [], []
        self.Construct1DArrays()

        self.ClusterEnergy = sum(self.HzzMatrixElements)/2

    def CreateHamiltonians(self, h_parameters):
        [Kx, Ky, Kz], [Gx, Gy, Gz], Gp, J1 = h_parameters
        self.Hx = np.array([[Kx + J1, Gp, Gp],
                            [Gp, J1, Gx],
                            [Gp, Gx, J1]])
        self.Hy = np.array([[J1, Gp, Gy],
                            [Gp, Ky + J1, Gp],
                            [Gy, Gp, J1]])
        self.Hz = np.array([[J1, Gz, Gp],
                            [Gz, J1, Gp],
                            [Gp, Gp, J1 + Kz]])
        # print(self.Hx)

    def Construct1DArrays(self):
        def determine_bond_characteristics(s1, s2):
            d = self.SublatticeVectors[s1] - self.SublatticeVectors[s2]
            for whichd, H, type in zip([dx, dy, dz], [self.Hx, self.Hy, self.Hz], ['x','y','z']):
                [_, bool1], [_, bool2] = [EquivalentBravais(self.T1, self.T2,
                                                            d - (pmd), np.array([0, 0]))
                                          for pmd in [whichd, -whichd]]
                if bool1:
                    self.BondIndices.append([s1, s2])
                    self.BondHamiltonians.append(H)
                    self.BondDiffs.append(+whichd)
                    self.BondType.append(type)
                elif bool2:
                    self.BondIndices.append([s1, s2])
                    self.BondHamiltonians.append(H)
                    self.BondDiffs.append(-whichd)
                    self.BondType.append(type)
            return 0

        lst = it.product(range(self.Sites), range(self.Sites))
        crap = [determine_bond_characteristics(s1, s2) for [s1, s2] in lst]
        self.BondIndices, self.BondHamiltonians, self.BondDiffs = [
            np.array(x) for x in [self.BondIndices, self.BondHamiltonians, self.BondDiffs]]

        # for i in range(len(self.BondIndices)):
        #     print(self.BondIndices[i], self.BondDiffs[i],self.BondType[i])
        #     print("..........")

        self.bond_spins = np.array([[self.Spins[i], self.Spins[j]]
                               for i, j in self.BondIndices[:, :]])
        self.bond_rotations = np.array([ [LocalRotation(Si), LocalRotation(Sj)]
                                    for Si, Sj in self.bond_spins[:, :]])

        self.transformed_bond_hamiltonians = np.einsum('abc,acd,aed->abe', self.bond_rotations[:, 0, :],
                                                  self.BondHamiltonians, self.bond_rotations[:, 1, :])

        # new_energies = np.einsum('ab,abc,ac->a',bond_spins[:,0],transformed_bond_hamiltonians,bond_spins[:,1])
        # print(np.sum(new_energies)/self.Sites)


        def parse_hamiltonian(H):
            A = ( H[0, 0] + H[1, 1] + 1j * (H[1, 0] - H[0, 1]) )/2
            B = ( H[0, 0] - H[1, 1] + 1j * (H[1, 0] + H[0, 1]) )/2
            return np.array([A, B])

        self.ABElements = np.array(
            [parse_hamiltonian(H) for H in self.transformed_bond_hamiltonians])
        # print(self.ABElements[:,1])
        self.HzzMatrixElements = self.transformed_bond_hamiltonians[:, 2, 2]

        # for i in range(len(self.HzzMatrixElements)):
            # print(i, self.BondIndices[i], self.HzzMatrixElements[i])

    def ObtainMagnonSpectrumAndDiagonalizer(self, klst,angle,offset):

        def R2(theta):
            s, c = np.sin(theta), np.cos(theta)
            return np.array([[c, -s], [s, c]])

        R = R2(angle)

        self.klst = np.einsum('ab,cb->ca', R, np.array(klst))

        self.phase_matrix_elements = np.exp(-1j * np.dot(self.BondDiffs, self.klst.T))
        self.other_phase_matrix_elements = np.exp(-1j * np.dot(self.BondDiffs, -self.klst.T))
        # print(self.phase_matrix_elements.shape)

        # ikk = 155
        # jkk = 3
        # r = self.BondDiffs[jkk,:]
        # k = self.klst[ikk,:]
        # print(f"swag{self.phase_matrix_elements[jkk,ikk] - np.exp(1j*k.dot(r))}")
        # print(f"ga{self.ABElements[jkk,0]*self.phase_matrix_elements[jkk,ikk] }")

        a_matrix_elements = np.einsum(
            'a,ac->ac', self.ABElements[:, 0], self.phase_matrix_elements)
        b_matrix_elements = np.einsum(
            'a,ac->ac', self.ABElements[:, 1], self.phase_matrix_elements)
        c_matrix_elements = np.einsum(
            'a,ac->ac', np.conj(self.ABElements[:, 1]), self.phase_matrix_elements)
        d_matrix_elements = np.einsum(
            'a,ac->ac', np.conj(self.ABElements[:, 0]), self.phase_matrix_elements)

        e_matrix_elements = np.einsum(
            'a,ac->ac', self.ABElements[:, 0], self.other_phase_matrix_elements)

        self.Ak, self.Bk, self.Ck, self.Dk, self.Ek = np.zeros(
            (5, len(klst), self.Sites, self.Sites), dtype=complex)

        self.Hk = np.zeros(
            (len(klst),
             2 * self.Sites,
             2 * self.Sites),
            dtype=complex)

        def fill_matrix(i, j):
            # self.BondIndices is of shape (number of bonds, 2 (two site indices) )
            # j indexes the bond
            # i indexes the k point
            s1, s2 = self.BondIndices[j, :]
            self.Ak[i, s1, s2] += a_matrix_elements[j, i] #- self.HzzMatrixElements[j]
            self.Bk[i, s1, s2] += b_matrix_elements[j, i]
            self.Ck[i, s1, s2] += c_matrix_elements[j, i]
            self.Dk[i, s1, s2] += d_matrix_elements[j, i] #- self.HzzMatrixElements[j]
            self.Ek[i, s1, s2] += e_matrix_elements[j, i] #- self.HzzMatrixElements[j]
            return 0

        #shape of self.HzzMatrixElements is (number of bonds, )
        # print(self.HzzMatrixElements)
        Adiag = np.diag([sum(self.HzzMatrixElements[i:i + 3])
                         for i in range(self.Sites)])
        # print([sum(self.HzzMatrixElements[i:i + 3])
        #                  for i in range(self.Sites)])
        # print(Adiag)

        ones = np.array([1 for i in range(self.Sites)])
        G = np.diag(np.concatenate((ones, -ones)))
        # print(klst)
        #
        def diagonalize_given_k(i):
            crap = [
                fill_matrix(
                    i, j) for j in range(
                    self.BondIndices.shape[0])]
            # print(self.Ak[i])
            # print(self.Bk[i])
            # print(self.Ck[i])
            # print(self.Dk[i])

            self.Ak[i] = self.Ak[i] - Adiag + offset * np.eye(*self.Ak[i].shape)# - Adiag
            self.Dk[i] = self.Dk[i] - Adiag + offset * np.eye(*self.Ak[i].shape)# - Adiag
            self.Ek[i] = self.Ek[i] - Adiag + offset * np.eye(*self.Ak[i].shape)# - Adiag
            # print(self.Ak[i])

            eps =  10E-12
            A_is_hermit = np.all(np.abs(self.Ak[i] - np.conj(self.Ak[i].T)) <= eps)
            B_is_C_hermit = np.all(np.abs(self.Bk[i] - np.conj(self.Ck[i].T)) <= eps)
            D_is_hermit = np.all(np.abs(self.Dk[i] - np.conj(self.Dk[i].T)) <= eps)

            diag_cond = np.all(np.abs(self.Dk[i] - self.Ek[i].T )<= eps)

            if not A_is_hermit:
                print("A != A dagger")
            if not B_is_C_hermit:
                print("B != C dagger")
            if not D_is_hermit:
                print("D != D dagger")
            if not diag_cond:
                print("check the diagonals!")

            # self.Hk[i] = np.block([
            #     [self.Ak[i], self.Bk[i]],
            #     [self.Ck[i], self.Dk[i]]
            # ])
            self.Hk[i] = np.block([
                [self.Ak[i]           , self.Bk[i]],
                [np.conj(self.Bk[i].T), self.Ek[i].T]
            ])

            # Method 1: straightforward diagonalization of G Hk
            newmat = np.matmul(G, self.Hk[i])

            dispersion, vectors = np.linalg.eig(newmat)
            dispersion = -np.sort(-dispersion)
            # print(np.imag(dispersion))

            # if not is_normal:
                # print("H not normal")

            # Method 2: straightforward diagonalization of Hk
            # dispersion, vectors = np.linalg.eigh(self.Hk[i])
            # dispersion = np.sort(dispersion)


            # Method 3: straightforward diagonalization of G Hk
            # try:
                # Mk = la.cholesky(self.Hk[i])
                # print("cholesky decomposition passed.")
            # except BaseException:
                # print("cholesky decomposition failed.")
                # Mk  = np.empty(self.Hk[i].shape)
                # tempval, tempvec = la.eigh(self.Hk[i])
                # sqtempdiag = np.diag(tempval)
                # Mk = np.einsum(
                    # 'ab,bc,dc->ad',
                    # tempvec,
                    # np.sqrt(
                        # self.Hk[i]),
                    # np.conjugate(tempvec))
            #     # print(tempval)

            # eigval, eigvec = la.eigh(
                # np.einsum('ab,bc,dc->ad', Mk, G, np.conjugate(Mk)))
            # idx = eigval.argsort()[::-1]
            # dispersion, Uk = eigval[idx], eigvec[:, idx]
            # # print(dispersion)

            # Tk = la.solve_triangular(
            #     Mk,
            #     np.einsum(
            #         'ab,bc->ac',
            #         Uk,
            #         np.sqrt(
            #             np.einsum(
            #                 'ab,bc->ac',
            #                 G,
            #                 np.diag(dispersion)))))

            self.Dispersions[i, :] = dispersion[:self.Sites]/2
            # self.Tk[i, :, :] = Tk[:self.Sites, :self.Sites]
            return 0
        #
        self.Dispersions = np.empty((len(klst), self.Sites), dtype=np.double)
        # self.Tk = np.empty((len(klst), self.Sites, self.Sites), dtype=complex)
        crap = [diagonalize_given_k(i) for i in range(len(klst))]

        self.Dispersions[np.abs(self.Dispersions) < 10E-12] = 0

        if not np.all(self.Dispersions >= 0):
            print("Warning: non-trivial negative eigenvalues.")

    def PlotMagnonDispersions(self, kpath, tick_mark, sym_labels):
        n = len(kpath)
        fig, ax = plt.subplots()
        for i in range(3):
            ax.plot(range(len(kpath)), self.Dispersions)
        if not np.all(self.Dispersions >= 0):
            plt.title("Warning: non-trivial negative eigenvalues.")

        ax.axhline(y=0, color='0.75', linestyle=':')
        ax.set_ylabel(r'$\omega_{n\mathbf{k}}$', usetex=True)
        plt.xticks(tick_mark, sym_labels, usetex=True)
        return fig
# ------------------------------------------------------------------------------


class FreeEnergyDerivatives:
    Colors = ["blue", "magenta", "green"] #nondark background
    # Colors = ["turquoise", "limegreen", "orange"] #dark background
    # Colors = ["turquoise", "limegreen", "orange", "red"] #dark background

    def __init__(self, x_list, y_list, factor):
        self.XList = x_list
        self.YList = y_list
        self.Factor = factor

    def PseudoMagnetization(self):
        m = -np.gradient(self.YList, self.XList, edge_order=2) / self.Factor
        return m

    def PseudoSusceptibility(self):
        m = self.PseudoMagnetization()
        chi = np.gradient(m, self.XList, edge_order=2) / self.Factor
        return chi

    def ThirdDerivative(self):
        chi = self.PseudoSusceptibility()
        f = np.gradient(chi, self.XList, edge_order=2) / self.Factor
        return f

    def PlotSweep(self):
        m = self.PseudoMagnetization()
        chi = self.PseudoSusceptibility()
        # f = self.ThirdDerivative()

        functions = [self.YList, chi, m]
        # functions = [self.YList, chi, m, f]
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        fig.subplots_adjust(top=1.5)
        axes = [ax1, ax2, ax2.twinx()]
        # axes = [ax1, ax2, ax1.twinx(), ax2.twinx()]

        for function, ax, color in zip(functions, axes, self.Colors):
            ax.scatter(
                self.XList,
                function,
                marker="o",
                clip_on=False,
                s=20,
                facecolors='none',
                edgecolors=color,
                linewidth=1.5)
            ax.tick_params(axis="y", colors=color)
        axes[2].axhline(c='gray',ls="-.")
        # ax2.axhline(color=self.Colors[1], ls="-.")
        ax2.set_ylim([-0.25,1.25])
        axes[2].set_ylim([-0.25,1.25])

        # ax2.set_ylim([-10,10])

        ax1.grid(True, axis='x')
        ax2.grid(True, axis='x')

        plt.xlim(min(self.XList), max(self.XList))

        return fig

    def PseudoSusceptibilityPeaks(self, prom):
        f = self.PseudoSusceptibility()

        x_peak_list, f_peak_list = [], []
        f_peaks, f_prominences = find_peaks(f, prominence=prom)

        for f_peak_index in f_peaks:
            x_peak_list.append(self.XList[f_peak_index])
            f_peak_list.append(f[f_peak_index])

        return x_peak_list, f_peak_list, f_prominences["prominences"]


class PhiSweep(FreeEnergyDerivatives):
    ELabel = r"$\frac{E_0}{N}$"
    MLabel = r"-$\frac{1}{N}\frac{\mathrm{d}E_0}{\mathrm{d}\phi}$"
    ChiLabel = r"-$\frac{1}{N}\frac{\mathrm{d}^2E_0}{\mathrm{d}\phi^2}\qquad$"

    def __init__(self, p_list, e_list):
        super().__init__(p_list, e_list, np.pi)

    def PlotLabeledSweep(self):
        fig = self.PlotSweep()
        for ax, color, label in zip(fig.axes, self.Colors,
                                    [self.ELabel, self.ChiLabel, self.MLabel]):
            ax.set_ylabel(
                label,
                rotation="horizontal",
                fontsize=12,
                labelpad=15,
                color=color)
        fig.axes[1].set_xlabel(r"$\phi/\pi$")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig


class AnisotropySweep(FreeEnergyDerivatives):
    ELabel = r"$\frac{E_0}{N}$"

    def __init__(self, fixed_var, fixed_val, swept_par_list, e_list):
        super().__init__(swept_par_list, e_list, 1)

        if (fixed_var == "?"):
            self.SweptVar = "g"
        elif (fixed_var == "g"):
            self.SweptVar = "?"

        self.SweptParList = swept_par_list

        self.MLabel = r"$-\frac{1}{N}\frac{\mathrm{d}E_0}{\mathrm{d}%s}$" % (
            self.SweptVar)
        self.ChiLabel = r"$-\frac{1}{N}\frac{\mathrm{d}^2E_0}{\mathrm{d}%s^2}\quad$" % (
            self.SweptVar)
        # self.TDLabel = r"-$\frac{1}{N}\frac{\mathrm{d}^3E_0}{\mathrm{d}%s^3}\quad$" % (
            # self.SweptVar)

    def PlotLabeledSweep(self):
        fig = self.PlotSweep()
        for ax, color, label in zip(fig.axes, self.Colors,
                                    [self.ELabel, self.ChiLabel, self.MLabel]):#, self.TDLabel]):
            ax.set_ylabel(
                label,
                rotation="horizontal",
                fontsize=12,
                labelpad=10,
                color=color)
        fig.axes[1].set_xlabel(r"$%s$" % self.SweptVar)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig


class FieldSweep(FreeEnergyDerivatives):
    ELabel = r"$\frac{E_0}{N}$"
    MLabel = r"-$\frac{1}{N}\frac{\mathrm{d}E_0}{\mathrm{d}h}$"
    ChiLabel = r"-$\frac{1}{N}\frac{\mathrm{d}^2E_0}{\mathrm{d}h^2}\quad$"

    def __init__(self, h_list, e_list):
        super().__init__(h_list, e_list, 1)

    def PlotLabeledSweep(self):
        fig = self.PlotSweep()
        for ax, color, label in zip(fig.axes, self.Colors,
                                    [self.ELabel, self.ChiLabel, self.MLabel]):
            ax.set_ylabel(
                label,
                rotation="horizontal",
                fontsize=12,
                labelpad=15,
                color=color)
        fig.axes[1].set_xlabel(r"$h$")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig


class PhaseDiagram:
    def __init__(self, x_list, sweep_list):
        self.XList = np.array(x_list)
        # assumes that y_list is the same for each sweep!
        self.YList = np.array(sweep_list[0].SweptParList)
        self.X, self.Y = np.meshgrid(self.XList, self.YList, indexing="ij")
        self.Z = np.empty(self.X.shape)

        for index, sweep in enumerate(sweep_list):
            self.Z[index, :] = sweep.YList

        self.Chixx, self.Chiyx, self.Chixy, self.Chiyy = self.SusceptibilityGrids([
                                                                                  pi, 1])

    def PlotEnergy(self):
        fig, ax = plt.subplots()
        c = ax.pcolormesh(self.X, self.Y, self.Z, cmap='magma')
        ax.contour(self.X, self.Y, self.Z, 15, colors='k')
        fig.colorbar(c, ax=ax)
        ax.grid(True)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def SusceptibilityGrids(self, factor_list):
        m_list = np.gradient(self.Z, self.XList, self.YList, edge_order=2)
        mx, my = [-m_list[i] / factor_list[i] for i in range(2)]

        chix_list = np.gradient(mx, self.XList, self.YList, edge_order=2)
        chiy_list = np.gradient(my, self.XList, self.YList, edge_order=2)
        chixx, chiyx = [chix_list[i] / factor_list[i] for i in range(2)]
        chixy, chiyy = [chiy_list[i] / factor_list[i] for i in range(2)]
        return chixx, chiyx, chixy, chiyy

    def PlotMainSusceptibility(self):
        chiz = np.sqrt(self.Chixx**2 + self.Chiyy**2)

        fig, ax = plt.subplots()
        c = ax.pcolormesh(self.X, self.Y, chiz, cmap='inferno')
        cb = plt.colorbar(c, fraction=0.02)
        cb.ax.set_title(r'$\sqrt{\chi_\phi^2 + \chi_\alpha^2}\quad\;$', fontsize=7)
        ax.set_xticks(np.linspace(0, 1, 10 + 1), minor=True)
        # ax.xaxis.grid(True,which='major')
        # ax.xaxis.grid(True,which='minor')
        # ax.set_yticks([-0.5,-0.25,0,0.25,0.5,0.75,1], minor=False)
        # ax.yaxis.grid(True,which='major')
        # fig.tight_layout(rect=[0,0.03,1,0.95])
        return fig

    def PlotAuxSusceptibilities(self):
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        for ax, chi in zip([ax1, ax2, ax3, ax4], [
                           self.Chixx, self.Chiyx, self.Chixy, self.Chiyy]):
            pl = ax.pcolormesh(self.X, self.Y, chi, cmap='inferno')
            fig.colorbar(pl, ax=ax)
            ax.grid(True)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def HessianGrids(self, factor_list):
        chixx, chiyx, chixy, chiyy = self.SusceptibilityGrids(factor_list)
        dethess = chixx * chiyy - chixy * chiyx
        trhess = chixx + chiyy
        return dethess, trhess

    def PlotHessian(self):
        dethess, trhess = self.HessianGrids([pi, 1])

        # plotting trhess is like chiz in PlotMainSusceptibility, ignore
        fig, ax = plt.subplots()
        for ax, hess in zip([ax], [dethess]):
            pl = ax.pcolormesh(self.X, self.Y, hess, cmap='magma')
            fig.colorbar(pl, ax=ax)
            ax.grid(True)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def ChiPeaksAlongX(self, min_prom):
        peak_list = []
        for i in range(self.YList.shape[0]):
            y_coord = self.YList[i]
            f = self.Chixx[:, i]
            x_peaks, f_char = find_peaks(f, prominence=min_prom)

            for i, x_peak_index in enumerate(x_peaks):
                x_coord = self.XList[x_peak_index]
                peak_height = f[x_peak_index]
                prom = f_char["prominences"][i]
                peak_list.append([x_coord, y_coord, peak_height, prom])
        return peak_list

    def ChiPeaksAlongY(self, min_prom):
        peak_list = []
        for i in range(self.XList.shape[0]):
            x_coord = self.XList[i]
            f = self.Chiyy[i, :]
            y_peaks, f_char = find_peaks(f, prominence=min_prom)

            for i, y_peak_index in enumerate(y_peaks):
                y_coord = self.YList[y_peak_index]
                peak_height = f[y_peak_index]
                prom = f_char["prominences"][i]
                peak_list.append([x_coord, y_coord, peak_height, prom])
        return peak_list

    def PlotPeaks(self, min_prom_x, min_prom_y):
        peak_x, peak_y = np.array(
            self.ChiPeaksAlongX(min_prom_x)), np.array(
            self.ChiPeaksAlongY(min_prom_y))

        peak_x_heights = peak_x[:, 2]
        peak_y_heights = peak_y[:, 2]

        min_val_x, max_val_x = min(peak_x_heights), max(peak_x_heights)
        min_val_y, max_val_y = min(peak_y_heights), max(peak_y_heights)
        min_val = min([min_val_x, min_val_y])
        max_val = min([max_val_x, max_val_y])
        norm = clr.Normalize(vmin=min_val, vmax=max_val)

        fig = self.PlotMainSusceptibility()
        ax = fig.axes[0]
        ax.axhline(ls="--", color="darkgray")
        # fig, ax = plt.subplots() #replaces previous 3 lines for dots only
        ax.scatter(peak_x[:, 0], peak_x[:, 1], norm(peak_x[:, 2]) * 40,
                   label=r'$\chi_\phi$', marker="o", facecolors="gold", edgecolors="gold")
        ax.scatter(peak_y[:, 0], peak_y[:, 1], norm(peak_y[:, 2]) * 40,
                   label=r'$\chi_\alpha$', marker="o", facecolors="red", edgecolors="red")

        return fig
# ------------------------------------------------------------------------------


class SpinConfiguration:
    def __init__(self, flat_spin_loc, flat_spin_config, cluster_info):
        self.SpinLocations = flat_spin_loc  # array shape: (sites, 2)
        self.SpinsABC = flat_spin_config  # array shape: (sites, 3)
        self.Type, self.S, self.L1, self.L2, self.Cluster = cluster_info

        self.T1, self.T2, self.SublatticeVectors, _ = WhichUnitCell(
            self.Type, self.S, self.Cluster)

    def PlotSSF(self):
        scale=2*pi
        B1, B2 = FindReciprocalVectors(self.T1, self.T2)
        # may be modified later...i dont need meshgrid
        KX, KY, gggg = NewMakeGrid(B1, B2, self.L1, self.L2, 2)
        kx, ky = [np.reshape(x, -1) for x in [KX, KY]]
        k = np.stack((kx, ky)).T

        SdotS_mat = np.einsum("ij,kj", self.SpinsABC, self.SpinsABC)
        s_kflat = np.empty(len(k))

        fig, ax = plt.subplots()

        for i, kv in enumerate(k):
            phase_i = np.exp(1j * np.einsum('i,ji', kv, self.SpinLocations))
            phase_j = np.exp(-1j * np.einsum('i,ji', kv, self.SpinLocations))
            phase_mat = np.einsum('i,j->ij', phase_i, phase_j)
            s_kflat[i] = np.abs((SdotS_mat * phase_mat).sum() / \
                self.SpinLocations.shape[0]/self.SpinLocations.shape[0])
            ax.annotate(f'$\quad${s_kflat[i]:.6f}', kv/scale,fontsize=2 )

        s_k = np.reshape(s_kflat, KX.shape)
        c = ax.scatter(KX/scale, KY/scale, c=s_k, cmap='afmhot', edgecolors="none")
        ax.scatter(KX/scale, KY/scale, marker="+", color='gray', s=2)

        cbar = fig.colorbar(c, fraction=0.05)
        cbar.set_label(r'$s_\vec{k}/N$',labelpad=9, rotation=0)
        ax.axis("equal")
        # ax.axis("off")
        ax.set_facecolor('black')

        b1, b2 = FindReciprocalVectors(a1, a2)
        bz2 = ptch.RegularPolygon(
            (0, 0), 6, np.linalg.norm(
                (2 * b1 + b2) / 3)/2/pi, pi / 6, fill=False,color='gray')
        bz3 = ptch.RegularPolygon((0, 0), 6, np.linalg.norm(b1)/2/pi, 0, fill=False,\
                                  color='gray')
        ax.add_patch(bz2)
        ax.add_patch(bz3)
        ax.set_xlim(-6.5/scale, 6.5/scale)
        ax.set_ylim(-7.5/scale, 7.5/scale)
        ax.set_xlabel(r'$k_x/2\pi$')
        ax.set_ylabel(r'$k_y/2\pi$',rotation=0)
        return fig

    def PlotSpins(self):
        oneD1 = np.array(range(0, self.L1))
        oneD2 = np.array(range(0, self.L2))
        n1, n2 = np.meshgrid(oneD1, oneD2)

        RX_list = np.empty((self.S), dtype=np.ndarray)
        RY_list = np.empty((self.S), dtype=np.ndarray)

        for i in range(self.S):
            RX_list[i] = self.SublatticeVectors[i, 0] + \
                n1 * self.T1[0] + n2 * self.T2[0]
            RY_list[i] = self.SublatticeVectors[i, 1] + \
                n1 * self.T1[1] + n2 * self.T2[1]
        #
        mat_spin_config = np.reshape(
            self.SpinsABC, (self.L1, self.L2, self.S, 3))
        #
        fig, ax = plt.subplots()
        for x in range(self.L1):
            for y in range(self.L2):
                center1 = x * self.T1 + y * self.T2
                hex1 = ptch.RegularPolygon(
                    center1, 6, 1 / sqrt(3), 0, fill=False, linewidth=0.1)
                ax.add_patch(hex1)
                hex2 = ptch.RegularPolygon(
                    center1 + a1, 6, 1 / sqrt(3), 0, fill=False, linewidth=0.1)
                ax.add_patch(hex2)
        # # plt.figure(figsize=(8,8))
        mandem = np.arccos(
            np.clip(mat_spin_config[:, :, :, 2], -1, 1)) / np.pi * 180
        norm = clr.Normalize()
        norm.autoscale(mandem)
        cm = my_cmap
        for i in range(self.S):
            # c = ax.quiver(RX_list[i], RY_list[i], mat_spin_config[:,:,i,0], mat_spin_config[:,:,i,1], mandem[:,:,i], cmap=cm, norm=norm, scale=50,minlength=1.5)#, scale=None,headwidth=1,headlength=1)
            #
            c = ax.quiver(RX_list[i],
                          RY_list[i],
                          mat_spin_config[:,
                                          :,
                                          i,
                                          0],
                          mat_spin_config[:,
                                          :,
                                          i,
                                          1],
                          mandem[:,
                                 :,
                                 i],
                          cmap=cm,
                          norm=norm,
                          scale=12,
                          headwidth=5,
                          minlength=1,
                          pivot='mid')  # , scale=None,headwidth=1,headlength=1)
        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        cb = plt.colorbar(
            sm,
            fraction=0.033,
            pad=0.00,
            orientation='vertical')

        cb.set_label(r'$\;\qquad\theta_{\mathbf{c}^*}$', labelpad=1, rotation=0)
        # cb.ax.set_title(r'$\;\qquad\theta_{\mathbf{c}^*}$')
        #
        ax.axis("off")
        plt.axis('scaled')

        #
        # ax.plot([0, self.T1[0], 0, self.T2[0]], [0, self.T1[1], 0, self.T2[1]])
        # ax.plot([self.T1[0], self.T1[0] +
        #          self.T2[0], self.T2[0], self.T1[0] +
        #          self.T2[0]], [self.T1[1], self.T1[1] +
        #                        self.T2[1], self.T2[1], self.T1[1] +
        #                        self.T2[1]])
        return fig
#

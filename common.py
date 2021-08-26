import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from itertools import product


def AddBZ(ax, scale: float, usetex):
    '''
    Adds the first, second, and sqrt(3) x sqrt(3) Brillouin zones to the reciprocal
    lattice figure. Also fixes the x and y limits and axis labels manually

    Parameters
    ax    (matplotlib.Axes): axes of reciprocal lattice figure, matplotlib.axes object
    scale           (float): so that I can plot k/scale instead of k.

    Returns
    ax    (matplotlib.Axes): modified axes with BZs, matplotlib.axes object
    '''
    #first crystal BZ
    bz2 = ptch.RegularPolygon((0, 0), 6, np.linalg.norm((2 * b1 + b2) / 3)/scale, pi / 6, fill=False,color='r')
    ax.add_patch(bz2)
    #second crystal BZ
    bz3 = ptch.RegularPolygon((0, 0), 6, np.linalg.norm(b1)/scale, 0, fill=False,color='g')
    ax.add_patch(bz3)
    # #sqrt(3) x sqrt(3) reduced 1st BZ
    # bz4 = ptch.RegularPolygon((0, 0), 6, np.linalg.norm(b1 + b2)/3/scale, 0, fill=False,color='b')
    # ax.add_patch(bz4)

    ax.set_xlim(-7.5/scale, 7.5/scale)
    ax.set_ylim(-7.5/scale, 7.5/scale)

    if abs(scale - 2*pi) <= 10**(-8):
        denom = f'/$2\pi$'
    else:
        denom = f'/{scale:.3f}'

    ax.set_xlabel(r'$k_x$'+denom, usetex=usetex)
    ax.set_ylabel(r'$k_y$'+denom, usetex=usetex)

    ticks = np.linspace(-1,1,4+1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'${val:.1f}$' for val in ticks],usetex=usetex)
    ax.set_yticklabels([f'${val:.1f}$' for val in ticks],usetex=usetex)

    return ax

def FindReciprocalVectors(A1, A2):
    '''
    Returns B1, B2 such that Ai.Bj = 2*pi KroneckerDelta(i,j)

    Parameters:
    A1, A2 (numpy.ndarray): two vectors of shape (2,)

    Returns:
    B1, B2 (numpy.ndarray): two vectors of shape (2,)
    '''
    z = np.zeros((2, 2))
    A = np.block([[np.array([A1, A2]), z], [z, np.array([A1, A2])]])
    const = np.array([2 * pi, 0, 0, 2 * pi])
    x = np.linalg.solve(A, const)
    B1, B2 = x[:2], x[2:4]
    return B1, B2

def RotateIn2D(theta: float):
    '''
    Rotation matrix such that takes a vector v = (x,y) and rotates it
    counterclockwise an angle theta (in radians) from the x-axis.

    Parameters:
    theta     (float): angle which to rotate vector by in radians.

    Returns:
    R (numpy.ndarray): rotation matrix of shape (2,2)
    '''
    s, c = np.sin(theta), np.cos(theta)
    R = np.array([[c, -s], [s, c]])
    return R

def LocalRotation(spin):
    '''
    Returns rotation matrix U such that U(spin) @ spin = (0,0,1)
    For spin = (1, 1, 1)/sqrt(3), U is the xyz to abc* change of basis matrix

    Parameters
    spin (numpy.ndarray): the spin moment of shape (3,)

    Returns
    U    (numpy.ndarray): the rotation matrix of shape (3,3)
    '''
    Sx, Sy, Sz = spin
    Sperp = np.linalg.norm(spin[:2])
    if Sperp > gen_eps:
        U = np.array([
            [Sz * Sx / Sperp, Sz * Sy / Sperp, -Sperp],
            [-Sy / Sperp, Sx / Sperp, 0],
            [Sx, Sy, Sz]
        ])
    else: #handles north pole degeneracy
        U = np.array([
            [Sz * np.sign(Sx), Sz * np.sign(Sy), -Sperp],
            [-np.sign(Sy), np.sign(Sx), 0],
            [Sx, Sy, Sz]
        ])
    return U

def IndexToPosition(A1, A2, sv, indices):
    '''
    Converts spin position indices in Monte Carlo file (n1, n2, s) to the actual
    position
                            z = n1*A1 + n2*A2 + sv[s]
    where A1 and A2 are the unit cell vectors of the cluster and sv is the list of
    sublattice positions within the unit cell

    Parameters
    A1, A2 (numpy.ndarray): two unit cell vectors of shape (2,)
    sv         (list-like): list of sublattice positions within unit cell
    indices    (list-like): list of position indices (ints) in MC spin file

    Returns
    z      (numpy.ndarray): position of spin of shape (2,-)
    '''
    n1, n2, s = indices
    z = n1 * A1 + n2 * A2 + sv[s] # just in case s is turned into a float
    return z

def KMeshForPlotting(B1, B2, L1: int, L2: int, m1: int, m2: int, addbz: bool, addPoints: bool,usetex):
    '''
    Creates two dimensional Gamma-centered K-mesh. Note that meshgrids are
    'x,y' indexed, so the shape you would expect is flipped around (see np.meshgrid
    documentation for more details). Do not use for integration, as it may overcount
    values in the first Brillouin zone.

    Parameters
    B1, B2  (numpy.ndarray): two vectors of shape (2,)
    L1, L2            (int): density of mesh in B1 and B2 directions.
    m1, m2            (int): copies in B1 and B2 directions.
    addbz            (bool): whether to add BZs (True) or not (False). If False,
                            you might want to fix the x and y limits and axis
                            labels manually.
    addPoints:       (bool): whether to plot the momentum points (True) or not (False)

    Returns
    KX      (numpy.ndarray): numpy.meshgrid of x coordinates of shape (2*m2*L2+1,2*m1*L1+1)
    KY      (numpy.ndarray): numpy.meshgrid of y coordinates of shape (2*m2*L2+1,2*m1*L1+1)
    fig (matplotlib.Figure): plot of meshgrid
    '''
    oneD1, oneD2 = [np.array(range(-m * l, m * l+1))/l for l,m in zip([L1, L2],[m1,m2])]
    n1, n2 = np.meshgrid(oneD1, oneD2)  # grid points indexing G1/G2 direction
    KX, KY = [n1 * B1[i] + n2 * B2[i] for i in [0,1]]  # bends meshgrid into shape of BZ

    g = np.zeros(2)
    scale = 2*pi
    fig, ax = plt.subplots()
    if addPoints: #whether to plot BZ or not
        ax.plot(KX/scale, KY/scale, '+', c='lightgrey',zorder=0)
    # PlotLineBetweenTwoPoints(ax, g/scale, B1/scale)
    # PlotLineBetweenTwoPoints(ax, g/scale, B2/scale)

    ax.set_aspect('equal')
    # ax.axis("equal")
    # ax.set_facecolor('white')
    if addbz: #whether to plot BZ or not
        AddBZ(ax, scale,usetex)

    return KX, KY, fig

def KMeshForIntegration(B1, B2, L1: int, L2: int):
    '''
    Creates two dimensional K-mesh for integration along the span{B1,B2}.
    Note that meshgrids are 'x,y' indexed, so the shape you would expect is
    flipped around (see np.meshgrid documentation for more details).

    Parameters
    B1, B2 (numpy.ndarray): two vectors of shape (2,)
    L1, L2        (int): density of mesh in B1 and B2 directions.

    Returns
    KX     (numpy.ndarray):  numpy.meshgrid of x coordinates of shape (2*n*L2+1,2*n*L1+1)
    KY     (numpy.ndarray):  numpy.meshgrid of y coordinates of shape (2*n*L2+1,2*n*L1+1)
    '''
    # oneD1, oneD2 = (np.array(range(0, l))/l for l in [L1, L2])
    # n1, n2 = np.meshgrid(oneD1, oneD2)  # grid points indexing G1/G2 direction
    # KX, KY = (n1 * B1[i] + n2 * B2[i] for i in [0,1])  # bends meshgrid into shape of BZ
    oneD1, oneD2 = (np.arange(-l/2, l/2)/l for l in [L1, L2])
    n1, n2 = np.meshgrid(oneD1, oneD2)  # grid points indexing G1/G2 direction
    KX, KY = (n1 * B1[i] + n2 * B2[i] for i in [0,1])  # bends meshgrid into shape of BZ
    return KX, KY

def EZhangBZ(B1, B2, L1: int, L2: int):
    '''
    Emily's implementation of the first BZ momenta.

    Parameters
    B1, B2 (numpy.ndarray): two vectors of shape (2,)
    L1, L2           (int): density of mesh in B1 and B2 directions.

    Returns
    ks               (list:  list of first BZ momenta of length L1*L2
    '''
    k = lambda k1, k2: (k1/L1)*B1 + (k2/L2)*B2
    s1, s2 = [np.arange(0, l) for l in [L1, L2]]
    ks = [ k(k1, k2) for k1, k2 in product(s1, s2) ]
    return ks

def PlotLineBetweenTwoPoints(ax, A, B):
    '''
    Adds a line between two points A  and B (in Cartesian xy basis) to a figure

    Parameters
    ax (matplotlib.Axes): axes of figure
    A    (numpy.ndarray): starting point A = (a1, a2),  shape (2,)
    B    (numpy.ndarray): finishing point B = (b1, b2), shape (2,)

    Returns
    ax (matplotlib.Axes): modified axes
    '''
    points = np.array([A,B])
    ax.plot(points[:,0], points[:,1], color='black', linestyle='-', linewidth=2)
    return ax

def IsItBravais(B1, B2, d):
    '''
    Checks if a vector d can be written as an integer linear combination of
    two other vectors B1 and B2:
                      d = n1 B1 + n2 B2, ni are integers

    Note: Note that the matrix B will need to be inverted, so avoid throwing
    this function in a loop that must be performed many times. A better solution
    would be to precompute the matrix B^(-1), calculate n, feed that into a function
    and return the bool. Check the

    Parameters
    B1, B2 (numpy.ndarray): two lattice vectors of shape (2,)
    d      (numpy.ndarray): vector of shape (2,)

    Returns
    n      (numpy.ndarray): [n1, n2] in equation above, shape (2,)
    bool: checks whether both n1 and n2 are integers (True) or not (False)å
    '''
    B = np.array([B1, B2]).T
    n = np.linalg.solve(B, d)

    bool = (abs(n[0] - round(n[0])) < gen_eps) and (abs(n[1] - round(n[1])) < gen_eps)
    return n, bool

def Path(A, B, n: int):
    '''
    A 2d version of linspace along the line B-A, where A,B are two-dimensional vectors.
    Used to construct KPath between high symmetry points

    Parameters
    A  (numpy.ndarray): initial vector of shape (2,)
    B  (numpy.ndarray): final vector of shape (2,)
    n            (int): initial number of points to sample line, which will be
                        modified based on the length of B-A

    Returns
    z  (numpy.ndarray): the list of vectors along path, shape (n,2)
    '''
    n = int(np.linalg.norm(B - A) * n)  # takes into account length of difference
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
    return z


class SymmetryPoints:
    '''
    A class containing the symmetry points of the reciprocal space of
    the triangular Bravais lattice with primitive unit vectors
                 a1, a2 = ( 1, sqrt(3) )/2, ( 1, -sqrt(3) )/2
    Symmetry points are thus given in the basis
               b1, b2 = 2 pi (1, 1/sqrt(3)), 2 pi (1, -1/sqrt(3))

    Attributes
    Sym    (dict): contains string label (keys) and reciprocal vector (values)
    SymTeX (dict): contains string label (keys) and LaTeX labels (values)
    '''
    def __init__(self):
        '''
        Constructor of the SymmetryPoints class.
        '''
        list = [(0, 0), (1/2, 0), (1/2, 1/2), (0, 1/2), (2/3, 1/3), (1/3, 2/3),
                    (-1/3, 1/3),     (1, 0),   (1, 1), (-1/2,1/2)
        ]
        G, M1, M2, M3, K, Kp, Kpp, Gp1, Gp2, X = \
                           [IndexToPosition(b1, b2, [0], [*i, 0]) for i in list]

        self.Sym = {
            "G": G,
            "M1": M1,
            "-M1": -M1,
            "K": K,
            "M2": M2,
            "Kp": Kp,
            "M3": M3,
            "Gp1": Gp1,
            "Gp2": Gp2,
            "Kpp": Kpp,
            "X":X
            }

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
            "X": r"$X$"
            }

    def MakeKPath(self, sym_points, n: int):
        '''
        Creates kpath, or array of two-dimensional vectors which cut a path
        connecting the chosen symmetry points.

        Parameters
        sym_points     (list): list of labels (string) of chosen symmetry points
        n               (int): initial density of points along path between two
                               symmetry points

        Returns
        path  (numpy.ndarray): array of vectors along path of shape(..., 2)
        tick_mark      (list): list of ticks (ints) at which to place LaTeX label
                               in the kpath plot
        sym_labels     (list): list of LaTeX labels (string)
        '''
        sym_labels = list(map(self.SymTeX.get, sym_points))
        sym_values = list(map(self.Sym.get, sym_points))
        shifted_sym_values = sym_values[1:] + sym_values[:1]
        pairs = list(zip(sym_values, shifted_sym_values))
        del pairs[-1]

        kpath, tick_mark, l = [], [0], 0
        for i, pair in enumerate(pairs):
            path = Path(*pair, n)
            kpath.append(path)
            l = l + path.shape[0]
            tick_mark.append(l)
        path = np.concatenate(np.array(kpath,dtype=object))
        return path, tick_mark, sym_labels, sym_values

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

        functions = [self.YList, m, chi]
        # functions = [self.YList, chi, m, f]
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        # fig.subplots_adjust(top=1.5)
        axes = [ax1, ax1.twinx(), ax2]
        # axes = [ax1, ax2, ax1.twinx(), ax2.twinx()]

        for function, ax, color in zip(functions, axes, self.Colors):
            print(function, ax, color)
            ax.scatter(
                self.XList,
                function,
                marker=".",
                # clip_on=False,
                s=20,
                facecolors='none',
                edgecolors=color,
                linewidth=1.5)
            ax.tick_params(axis="y", colors=color)
        # axes[2].axhline(c='gray',ls="-.")
        # ax2.axhline(color=self.Colors[1], ls="-.")
        # ax2.set_ylim([-0.25,1.25])
        # axes[2].set_ylim([-0.25,1.25])

        # ax2.set_ylim([-10,10])

        ax1.grid(True, axis='x')
        ax2.grid(True, axis='x')

        plt.xlim(min(self.XList), max(self.XList))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig

    def PseudoSusceptibilityPeaks(self, prom):
        f = self.PseudoSusceptibility()

        x_peak_list, f_peak_list = [], []
        f_peaks, f_prominences = find_peaks(f, prominence=prom)

        for f_peak_index in f_peaks:
            x_peak_list.append(self.XList[f_peak_index])
            f_peak_list.append(f[f_peak_index])

        return x_peak_list, f_peak_list, f_prominences["prominences"]

class AnisotropySweep(FreeEnergyDerivatives):
    ELabel = r"$\frac{E_0}{N}$"

    def __init__(self, fixed_var, fixed_val, swept_par_list, e_list):

        if (fixed_var == "a"):
            self.SweptVar = "\psi"
            factor = pi
        elif (fixed_var == "p"):
            self.SweptVar = "g"
            factor = 1

        super().__init__(swept_par_list, e_list, factor)

        self.SweptParList = swept_par_list

        self.MLabel = r"$-\frac{1}{N}\frac{\mathrm{d}E_0}{\mathrm{d}%s}$" % (
            self.SweptVar)
        self.ChiLabel = r"$-\frac{1}{N}\frac{\mathrm{d}^2E_0}{\mathrm{d}%s^2}\quad$" % (
            self.SweptVar)
        # self.TDLabel = r"-$\frac{1}{N}\frac{\mathrm{d}^3E_0}{\mathrm{d}%s^3}\quad$" % (
            # self.SweptVar)

    def PlotLabeledSweep(self):
        fig = self.PlotSweep()
        for ax, color, label in zip(fig.axes, ["blue", "green", "magenta"],
                                    [self.ELabel, self.ChiLabel, self.MLabel]):#, self.TDLabel]):
            ax.set_ylabel(
                label,
                rotation="horizontal",
                fontsize=12,
                labelpad=10,
                color=color)
        if self.SweptVar == "g":
            fig.axes[1].set_xlabel(r"$%s$" % self.SweptVar)
        elif self.SweptVar == "\psi":
            fig.axes[1].set_xlabel(r"$%s/\pi$" % self.SweptVar)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

#------------------------Some standard global variables------------------------#
# constants
pi = np.pi
sqrt3 = np.sqrt(3)
gen_eps = 10E-12 #general tolerance used to test floats for equivalence to zero

# Bravais primitive vectors of triangular lattice
a1, a2 = np.array([1 / 2, sqrt3 / 2]), np.array([-1 / 2, sqrt3 / 2])
# Reciprocal lattice vectors
b1, b2 = FindReciprocalVectors(a1, a2)

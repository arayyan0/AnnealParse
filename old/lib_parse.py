#file   lib_parse.py
#author Ahmed Rayyan
#date   November 20, 2019
#brief  modules that parse C++ data for visualization
import numpy as np
import cmath
import matplotlib.pyplot as plt
from math import sqrt

pi = np.pi

xyz_to_abc_matrix = np.array([
                               [1/sqrt(6) , 1/sqrt(6), -sqrt(2/3)],
                               [-1/sqrt(2), 1/sqrt(2), 0         ],
                               [1/sqrt(3) , 1/sqrt(3), 1/sqrt(3) ]
                              ])

def WhichUnitCell(sublattice):
    a1, a2 =  np.array([1/2, sqrt(3)/2]), np.array([-1/2, sqrt(3)/2])
    if sublattice == 2:
        T1, T2 = a1, a2
        sub_list = np.array([
                             (T1+T2)/3,
                             2*(T1+T2)/3
                            ])
    if sublattice == 6:
        T1, T2 = 2*a1 -a2, 2*a2-a1
        sub_list = np.array([
                             (T1+T2)/2 + T1/3,
                             (T1+T2)/2 - T2/3,
                             (T1+T2)/2 + (T1+T2)/3,
                             (T1+T2)/2 - T1/3,
                             (T1+T2)/2 + T2/3,
                             (T1+T2)/2 - (T1+T2)/3
                            ])
    return T1, T2, sub_list

def Phase(k, x):
    z = np.asscalar(np.real_if_close(cmath.rect(1,k.dot(x))))
    return z

def Couplings(sign, scale, g, a):
    Jx = sign*scale*2*(1-a)*(1-g)/3;
    Jy = sign*scale*2*(1-a)*g/3;
    Jz = sign*scale*(1+2*a)/3
    return Jx, Jy, Jz

def FindReciprocalVectors(a1, a2):
    z = np.zeros((2,2))
    A = np.block([[np.array([a1,a2]),z],[z,np.array([a1,a2])]])
    b = np.array([2*pi,0,0,2*pi])
    x = np.linalg.tensorsolve(A,b)
    b1,b2 = x[:2], x[2:4]
    return b1, b2

def MakeGrid(b1, b2, N):
    oneD = np.array(range(-2*N,2*N)) #1d list of points
    n1, n2 = np.meshgrid(oneD,oneD) #grid points indexing G1/G2 direction

    KX = n1*b1[0]/N + n2*b2[0]/N #bends meshgrid into shape of BZ
    KY = n1*b1[1]/N + n2*b2[1]/N

    fig, ax = plt.subplots()
    ax.plot(KX, KY,'+')
    ax.plot([0, b1[0]], [0, b1[1]], color='black', linestyle='-', linewidth=2)
    ax.plot([0, b2[0]], [0, b2[1]], color='black', linestyle='-', linewidth=2)
    ax.set_aspect('equal')
    return KX, KY, fig

def XYZtoABC(v):
    new_v = xyz_to_abc_matrix.dot(v)
    return new_v

def RotateC3(x_lst, y_lst, z_lst, w_lst):
    a_lst = np.array(x_lst + y_lst + z_lst)
    b_lst = np.array(y_lst + z_lst + x_lst)
    c_lst = np.array(z_lst + x_lst + y_lst)
    v_lst = np.array(w_lst + w_lst + w_lst)
    return a_lst, b_lst, c_lst, v_lst

def ThreeToOneIndex(nx, ny, s, length, sublattice):
    i = (ny*length + nx)*int(sublattice) + s
    return i

def ExtractEnergyFromFile(file):
    with open(file, 'r') as f:
        file_data = f.readlines()
    e = float(file_data[30])
    return e

def NewMakeGrid(b1, b2, L1, L2):
    oneD1 = np.array(range(-3*L1,3*L1)) #1d list of points
    oneD2 = np.array(range(-3*L2,3*L2)) #1d list of points
    n1, n2 = np.meshgrid(oneD1,oneD2) #grid points indexing G1/G2 direction
    KX = n1*b1[0]/L1 + n2*b2[0]/L2 #bends meshgrid into shape of BZ
    KY = n1*b1[1]/L1 + n2*b2[1]/L2

    fig, ax = plt.subplots()
    ax.plot(KX, KY,'+')
    ax.plot([0, b1[0]], [0, b1[1]], color='black', linestyle='-', linewidth=2)
    ax.plot([0, b2[0]], [0, b2[1]], color='black', linestyle='-', linewidth=2)
    ax.set_aspect('equal')
    return KX, KY, fig

class KG_ASweep:
    ELabel = r"$\frac{E_0}{N}$"

    def __init__(self, fixed_var, fixed_val, swept_par_list, e_list):
        self.FixedVar = fixed_var
        self.FixedVal = fixed_val
        self.SweptParList = swept_par_list
        self.EList = e_list

        if (fixed_var == "g"):
            self.SweptVar = "a"
        elif (fixed_var == "a"):
            self.SweptVar = "g"

        self.MLabel = r"-$\frac{1}{N}\frac{\mathrm{d}E_0}{\mathrm{d}%s}$"%(self.SweptVar)
        self.ChiLabel = r"-$\frac{1}{N}\frac{\mathrm{d}^2E_0}{\mathrm{d}%s^2}\quad$"%(self.SweptVar)

    def Magnetization(self):
        # print(self.EList, self.SweptParList)
        m = - np.gradient(self.EList, self.SweptParList, edge_order=2)
        return m

    def Susceptibility(self):
        m = self.Magnetization()
        chi = np.gradient(m, self.SweptParList, edge_order=2)
        return chi

    def PlotSweep(self):
        m = self.Magnetization()
        chi = self.Susceptibility()

        functions = [self.EList, m, chi]
        labels = [self.ELabel, self.MLabel, self.ChiLabel]

        fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)
        fig.subplots_adjust(top=1.5)

        axes = [ax1, ax1.twinx(), ax2]
        colors = ["blue", "orange", "green"]

        for function, ax, color, label in zip(functions, axes, colors, labels):
            ax.plot(self.SweptParList, function, marker="o", color = color, clip_on=False)
            ax.set_ylabel(label, rotation="horizontal", fontsize=12, labelpad=15, color=color)
            ax.tick_params(axis="y", colors=color)
            ax.axhline(color=color, ls="-.")

        ax1.grid(True, axis='x')
        ax2.grid(True, axis='x')
        ax2.set_xlabel(r"$%s$"%self.SweptVar)

        plt.xlim(min(self.SweptParList),max(self.SweptParList))

        fig.tight_layout(rect=[0,0.03,1,0.95])

        return fig

class KKJSweep:
    ELabel = r"$\frac{E_0}{N}$"
    MLabel = r"-$\frac{1}{N}\frac{\mathrm{d}E_0}{\mathrm{d}\theta}$"
    ChiLabel = r"-$\frac{1}{N}\frac{\mathrm{d}^2E_0}{\mathrm{d}\theta^2}\qquad$"

    def __init__(self, p_list, e_list):
        self.PhiList = p_list
        self.EList = e_list

    def Magnetization(self):
        m = - np.gradient(self.EList, self.PhiList, edge_order=2)/pi
        return m

    def Susceptibility(self):
        m = self.Magnetization()
        chi = np.gradient(m, self.PhiList, edge_order=2)/pi
        return chi

    def PlotSweep(self):
        m = self.Magnetization()
        chi = self.Susceptibility()

        functions = [self.EList, m, chi]
        labels = [self.ELabel, self.MLabel, self.ChiLabel]

        fig, (ax1a, ax2) = plt.subplots(2,1,sharex=True)
        fig.subplots_adjust(top=1.5)

        ax1b = ax1a.twinx()
        axes = [ax1a, ax1b, ax2]
        colors = ["blue", "orange", "green"]

        for function, ax, color, label in zip(functions, axes, colors, labels):
            ax.plot(self.PhiList, function, marker="o", color = color, clip_on=False)
            ax.set_ylabel(label, rotation="horizontal", fontsize=12, labelpad=15, color=color)
            ax.tick_params(axis="y", colors=color)
            ax.grid(True, axis='x')
            # ax.axhline(color=color, ls="-.")

        ax2.set_xlabel(r"$\theta/\pi$")
        ax2.set_ylim((0,5))

        plt.xlim(min(self.PhiList),max(self.PhiList))

        fig.tight_layout(rect=[0,0.03,1,0.95])

        return fig

class KG_HSweep:
    ELabel = r"$\frac{E_0}{N}$"
    MLabel = r"-$\frac{1}{N}\frac{\mathrm{d}E_0}{\mathrm{d}h}$"
    ChiLabel = r"-$\frac{1}{N}\frac{\mathrm{d}^2E_0}{\mathrm{d}h^2}\qquad$"

    def __init__(self, h_list, e_list):
        self.HList = h_list
        self.EList = e_list

    def Magnetization(self):
        m = - np.gradient(self.EList, self.HList, edge_order=2)
        return m

    def Susceptibility(self):
        m = self.Magnetization()
        chi = np.gradient(m, self.HList, edge_order=2)
        return chi

    def PlotSweep(self):
        m = self.Magnetization()
        chi = self.Susceptibility()

        functions = [self.EList, m, chi]
        labels = [self.ELabel, self.MLabel, self.ChiLabel]

        fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)
        fig.subplots_adjust(top=1.5)

        axes = [ax1, ax1.twinx(), ax2]
        colors = ["blue", "orange", "green"]
    #
        for function, ax, color, label in zip(functions, axes, colors, labels):
            ax.plot(self.HList, function, marker="o", color = color, clip_on=False)
            ax.set_ylabel(label, rotation="horizontal", fontsize=12, labelpad=15, color=color)
            ax.tick_params(axis="y", colors=color)
            ax.axhline(color=color, ls="-.")

        ax1.grid(True, axis='x')
        ax2.grid(True, axis='x')
        ax2.set_xlabel("h")

        plt.xlim(min(self.HList),max(self.HList))

        fig.tight_layout(rect=[0,0.03,1,0.95])

        return fig

class KG_Sweep:
    ELabel = r"$\frac{E_0}{N}$"
    MLabel = r"-$\frac{1}{N}\frac{\mathrm{d}E_0}{\mathrm{d}\phi}$"
    ChiLabel = r"-$\frac{1}{N}\frac{\mathrm{d}^2E_0}{\mathrm{d}\phi^2}\qquad$"

    def __init__(self, p_list, e_list):
        self.PList = p_list
        self.EList = e_list

    def Magnetization(self):
        m = - np.gradient(self.EList, self.PList, edge_order=2)/pi
        return m

    def Susceptibility(self):
        m = self.Magnetization()
        chi = np.gradient(m, self.PList, edge_order=2)/pi
        return chi

    def PlotSweep(self):
        m = self.Magnetization()
        chi = self.Susceptibility()

        functions = [self.EList, m, chi]
        labels = [self.ELabel, self.MLabel, self.ChiLabel]

        fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)
        fig.subplots_adjust(top=1.5)

        axes = [ax1, ax1.twinx(), ax2]
        colors = ["blue", "orange", "green"]
    #
        for function, ax, color, label in zip(functions, axes, colors, labels):
            ax.plot(self.PList, function, marker="o", color = color, clip_on=False)
            ax.set_ylabel(label, rotation="horizontal", fontsize=12, labelpad=15, color=color)
            ax.tick_params(axis="y", colors=color)
            ax.axhline(color=color, ls="-.")

        # ax2.set_ylim([-10,10])

        ax1.grid(True, axis='x')
        ax2.grid(True, axis='x')
        ax2.set_xlabel(r"$\phi/\pi$")

        plt.xlim(min(self.PList),max(self.PList))

        fig.tight_layout(rect=[0,0.03,1,0.95])

        return fig

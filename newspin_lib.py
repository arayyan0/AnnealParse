import numpy as np
import matplotlib.colors as clr
import matplotlib.patches as ptch
import matplotlib.pyplot as plt
import re

from common import pi, PlotLineBetweenTwoPoints,sqrt3,a1,a2,FindReciprocalVectors,\
                   KMeshForPlotting,gen_eps

class MonteCarloOutput:
    '''
    Parses Monte Carlo output for cluster information, simulation information,
    spin orientation matrix, energy, spin moments, and spin locations
    '''
    def __init__(self, filename):
        self.Filename = filename
        with open(self.Filename, 'r') as f:
            self.FileData = f.readlines()
        kw_list = [
              "Which lattice?",
              "Linear dimensions",
              "Translation vectors",
              "Sublattice vectors",
              "Number of Deterministic sweeps",
              "Spin basis",
              "Energy per site",
              "Spin configuration"
        ]
        kw_index_list  = self.ExtractQuantities(kw_list)

        kw_dict = dict(zip(kw_list, kw_index_list))

        self.WhichLattice = int(self.FileData[kw_dict["Which lattice?"]])
        self.L1, self.L2, self.L3 = [int(x) for x in self.FileData[kw_dict["Linear dimensions"]].split()]
        self.T1, self.T2, self.T3 = np.array(list(
            map(np.double, [x.split() for x in self.FileData[kw_dict["Translation vectors"]:kw_dict["Translation vectors"]+3]])
        )).transpose()
        self.SublatticeVectors = np.array(list(
            map(np.double, [x.split() for x in self.FileData[kw_dict["Sublattice vectors"]:kw_dict["Sublattice vectors"]+3]])
        )).transpose()
        self.TotNumSublattices = self.SublatticeVectors.shape[0]
        self.NumSites = self.L1*self.L2*self.L3*self.TotNumSublattices
        self.DetSweeps = int(self.FileData[kw_dict["Number of Deterministic sweeps"]])
        self.ChangeBasis = np.array(list(map(np.double, [x.split() for x in self.FileData[kw_dict["Spin basis"]:kw_dict["Spin basis"]+3]])))
        self.MCEnergyPerSite = np.double(self.FileData[kw_dict["Energy per site"]])

        self.SpinConfigIndices = np.array(list(map(np.double, [x.split() for x in self.FileData[kw_dict["Spin configuration"]:kw_dict["Spin configuration"]+self.NumSites]])))[:,:4]
        self.SpinPositions = np.empty((self.NumSites, 3), dtype = np.double)
        for i, a in enumerate(self.SpinConfigIndices):
            self.SpinPositions[i] = a[0]*self.T1+a[1]*self.T2+a[2]*self.T3+self.SublatticeVectors[int(a[3])]

        #extract spins (in pseudospin and mind basis)
        self.SpinConfigSpinBasis = np.array(list(
            map(np.double, [x.split() for x in self.FileData[kw_dict["Spin configuration"]:kw_dict["Spin configuration"]+self.NumSites]])
        ))[:,4:]
        self.SpinConfigMindBasis = (self.ChangeBasis @ self.SpinConfigSpinBasis.T).T
        self.LayerPositions, self.LayerSpins, self.LayerNumber = self.SortPlanes()

        # extract which lattice: 0 for tri, 1 for hc, 2 for FCC
        if ((self.WhichLattice == 0) or (self.WhichLattice == 1)):
            self.Dimensions=2
        elif self.WhichLattice == 2:
            self.Dimensions=3

        ##############---
        ##############---hardcoding which model and its parameters here.
        ##############---
        # self.WhichModel = "kga"
        self.WhichModel = "multipolar"
        kw_ham_list = self.HamiltonianKeywords()
        kw_ham_index_list  = self.ExtractQuantities(kw_ham_list)
        kw_ham_dict = dict(zip(kw_ham_list, kw_ham_index_list))
        if self.WhichModel == "multipolar":
            #extract the jtau/jb interactions
            self.JTau, self.JB = np.array(list(map( np.double, self.FileData[kw_ham_dict["bond-dep"]].replace("/"," ").replace("\n","").split(" ") )))
            #extract the quad/octo interactions
            self.JQuad, self.JOcto = np.array(list(map( np.double, self.FileData[kw_ham_dict["bond-indep"]].replace("/"," ").replace("\n","").split(" ") )))
            #extract the defect properties
            self.DefectQuad, self.DefectOcto, self.DefectLengthScale, self.NumDefects = np.array([0,0,0,0])
        elif self.WhichModel == "kga":
            self.Kx, self.Ky, self.Kz = np.array(list(map( np.double, self.FileData[kw_ham_dict["Kx Ky Kz"]].replace("/"," ").replace("\n","").split(" ") )))

    def ExtractQuantities(self, kw_list):
        kw_index_list = []

        #extract which lattice: 0 for tri, 1 for hc, 2 for FCC
        for kw in kw_list:
            for line_number, file_line in enumerate(self.FileData):
                z = re.match(kw, file_line)
                if z:
                    kw_index_list.append(line_number+1)
        return kw_index_list

    def HamiltonianKeywords(self):
        if self.WhichModel == "multipolar":
            kw_list = [
                "bond-dep",
                "bond-indep"
            ]
        elif self.WhichModel == "kga":
            kw_list = [
                "Kx Ky Kz",
                "Gx Gy Gz",
                "Gp",
                "J1"
            ]
        kw_list += ["hMag","hDir"]
        return kw_list

    def SortPlanes(self):
        z_planes  = np.unique(self.SpinPositions[:,2])

        layer_positions = []
        layer_spins = []
        layer_number = []

        for z in z_planes:
            mask = np.isclose(self.SpinPositions[:,2], z)
            big_mask = np.array([mask, mask, mask]).T
            layer_positions.append(np.reshape(self.SpinPositions[big_mask],(-1,3)))
            layer_spins.append(np.reshape(self.SpinConfigMindBasis[big_mask],(-1,3)))
            layer_number.append(mask.sum())

        return layer_positions, layer_spins, layer_number

    def Plot3D(self):
        #need to add unit
        #need to add bonds (?)
        #other general aesthetics (colors to represent in/out of Az, equal axes, etc)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.quiver(self.SpinPositions[:,0],self.SpinPositions[:,1],self.SpinPositions[:,2],
                  self.SpinConfigMindBasis[:,0],self.SpinConfigMindBasis[:,1],self.SpinConfigMindBasis[:,2],
                  pivot = 'middle',
                  length = 0.5,
                  ec='black'
                  )
        ax.set_xlabel('$\hat{A}_x$')
        ax.set_ylabel('$\hat{A}_y$')
        ax.set_zlabel('$\hat{A}_z$')

        if self.Dimensions == 2:
            ax.set_zlim3d(-1,1)
        return fig

    def Plot2DSSF(self, whichplane, cb_options,usetex):
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
        if whichplane == -1:
            layer_index = np.argmax(self.LayerNumber)
        else:
            layer_index = whichplane

        spins, locs = self.LayerSpins[layer_index], self.LayerPositions[layer_index]

        B1, B2 = FindReciprocalVectors(self.T1[:2], self.T2[:2])

        addbz=True
        addPoints=False

        KX, KY, fig = KMeshForPlotting(B1, B2, self.L1, self.L2, 3, 3, addbz, addPoints, usetex)
        kx, ky = [np.reshape(x, -1) for x in [KX, KY]]
        k = np.stack((kx, ky)).T
        ax = fig.axes[0]
        scale = 2*pi

        SdotS_mat = np.einsum("ij,kj", spins, spins)
        s_kflat = np.zeros(len(k), dtype=np.cdouble)
        for i, kv in enumerate(k):
            phase_i = np.exp(1j * np.einsum('i,ji', kv, locs[:,:2]))
            phase_j = np.exp(-1j * np.einsum('i,ji', kv, locs[:,:2]))
            phase_mat = np.einsum('i,j->ij', phase_i, phase_j)
            s_kflat[i] = (SdotS_mat * phase_mat).sum()/ self.NumSites**2
        #     ax.annotate(f'$\quad${np.real(s_kflat[i]):.6f}',
        #                 kv/scale,
        #                 fontsize=4,
        #                 color = 'lightgray')
        # print(self.SpinsXYZ)

        ssf = np.reshape(s_kflat, KX.shape)

        ssf_complex_flag = np.any(np.imag(ssf) > gen_eps)
        if ssf_complex_flag:
            print("Warning: SSF has complex values. Please recheck calculation.")

        ssf = np.sqrt(np.real_if_close(np.multiply(np.conj(ssf), ssf)))

        #note: we are plotting the real part, since the SSF should be real.

        fraction, orientation, colormap = cb_options
        c = ax.scatter(KX/scale, KY/scale, c=ssf, cmap=colormap, edgecolors="none",zorder=0)
        cbar = fig.colorbar(c, fraction=fraction, orientation=orientation)
        cbar.set_label(r'$\sqrt{|s_\mathbf{k}|^2}$',rotation=0,labelpad=10)

        # ticks = np.linspace(0,1,4+1)
        # cbar.set_ticks(ticks)
        # cbar.ax.set_yticklabels([f'${val:.2f}$' for val in ticks],usetex=usetex,fontsize=9)
        # cbar.ax.set_xticklabels([f'${val:.2f}$' for val in ticks],usetex=usetex,fontsize=9)

        return fig

    def Plot2DPlane(self, whichplane, quiver_options, cb_options, usetex):

        if whichplane == -1:
            layer_index = np.argmax(self.LayerNumber)
        else:
            layer_index = whichplane

        spins, locs = self.LayerSpins[layer_index], self.LayerPositions[layer_index]

        sss, minlength, headwidth = quiver_options
        fraction, orientation, cm, tickaxis = cb_options

        thetac = np.arccos(np.clip(spins[:, 2], -1, 1)) / pi * 180
        norm = clr.Normalize(vmin=0,vmax=180)

        fig, ax = plt.subplots()

        ax.quiver(locs[:,0], locs[:,1],
                  spins[:,0], spins[:,1],
                  thetac, alpha=1,cmap=cm, norm=norm, pivot = 'mid',
                  scale=35,
                  # headlength = 2,
                  # headaxislength = 2,
                  headwidth = 6,
                  minlength=minlength,
                  # headaxislength=0.1
                  # minshaft=0.7,
                  linewidth=0.2,
                  width=0.0025,
                  ec='black')
        ax.axis("off")
        # ax.set_facecolor('black')
        ax.axis("equal") #zooms in on arrows
        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        cb = plt.colorbar(sm,
                          fraction=fraction,
                          # pad=0,
                          orientation=orientation)
        cb.ax.set_title(r'$\theta_{\hat{A}_z}\quad \left(^\circ\right)$', usetex=usetex)
        ticks = np.linspace(0,180,6+1)
        cb.set_ticks(ticks)
        if tickaxis=='x':
            cb.ax.set_xticklabels([f'${val:.0f}$' for val in ticks],usetex=usetex)
        elif tickaxis== 'y':
            cb.ax.set_yticklabels([f'${val:.0f}$' for val in ticks],usetex=usetex)

        if self.WhichModel == "multipolar":
            if (self.NumDefects == 1) and ((self.DefectQuad!=0) or (self.DefectOcto!=0)):
                defe = ptch.Circle(3*self.L1/6 *self.T1 + 3*self.L2/6 *self.T2 +(self.T1+self.T2)/3,
                radius=self.DefectLengthScale*2, fill=True, alpha=0.1, linewidth=1.5,color='black')
                ax.add_patch(defe)

            plt.title(r'$J^{\tau}$' + f' = {self.JTau:.6f}, $J^Q$ = {self.JQuad:.6f}, $J^O$ = {self.JOcto:.6f}, $J^B$ = {self.JB:.6f}')

        if self.Dimensions==2:
            if self.WhichLattice == 0:
                shape = 3
                length=1/sqrt3
                rotate=0
                offset = 2*(2*a1-a2)/3
            elif self.WhichLattice == 1:
                shape = 6
                length = 1/sqrt3
                rotate=0
                offset = a1
            for x in range(0,self.L1):
                for y in range(0,self.L2):
                    for z in range(0, self.L3):
                        #used for rhom unit cell
                        center = (x * self.T1 + y * self.T2 + z*self.T3)[:2]
                        # print(center)
                        hex2 = ptch.RegularPolygon(
                        center+offset, shape, length, rotate, fill=False, linewidth=0.005,color='gray')
                        ax.add_patch(hex2)

            #adding unit cell used for the calculation
            g = np.zeros(3)
            PlotLineBetweenTwoPoints(ax, g, self.T1)
            PlotLineBetweenTwoPoints(ax, g, self.T2)
            PlotLineBetweenTwoPoints(ax, self.T1, self.T1+self.T2)
            PlotLineBetweenTwoPoints(ax, self.T2, self.T1+self.T2)
        return fig


class MuonSimulation(MonteCarloOutput):

    def __init__(self, filename):
        super().__init__(filename)
        if self.WhichModel == 'multipole':
            self.FilterPositions();
            self.GenerateStoppingSites();
            self.CalculateDipolarFields();

    def FilterPositions(self):
        tuningofradius=1
        defect_position = 3*self.L1/6 *self.T1 + 3*self.L2/6 *self.T2 +(self.T1+self.T2)/3,
        cutoff=self.DefectLengthScale*2*tuningofradius

        muonsiteslst=[]
        for siteposition in self.SpinPositions:
            d = np.linalg.norm(siteposition - defect_position)
            if d < cutoff:
                muonsiteslst.append(siteposition)
        print(len(muonsiteslst))

        self.FilteredSites = np.array(muonsiteslst)

        # print(self.SpinPositions.shape)
        # print(muonsites.shape)
    def GenerateStoppingSites(self):
        d_osmium = 8/np.sqrt(2)
        d_oxygen = 2 / d_osmium ; #in units where bond length = 5.6 Angstroms = 1 (on triangular/DP)
        r_oxygen = 1 / d_osmium ;
        R1= np.array([ [ 0, 1, 0],
                       [-1, 0, 0],
                       [ 0, 0, 1] ])
        R = np.array([ [ 1, 1,-2]/np.sqrt(6),
                       [-1, 1, 0]/np.sqrt(2),
                       [ 1, 1, 1]/np.sqrt(3) ])
        x, y, z = (R1 @ R).T

        far_or_close = +1 #+1 (-1) for the far (close) pole of the oxygen sphere along the bond

        o1 = (d_oxygen + far_or_close*r_oxygen)*x
        o2 = (d_oxygen + far_or_close*r_oxygen)*y
        o3 = (d_oxygen + far_or_close*r_oxygen)*z
        o4 = -o1
        o5 = -o2
        o6 = -o3

        oxygen = [o1,o2,o3,o4,o5,o6]
        stoppingsiteslst = []
        for site in self.FilteredSites:
            for o in oxygen:
                stoppingsiteslst.append(site+o)
        self.StoppingSites = np.array(stoppingsiteslst)

    def CalculateDipolarFields(self):
        g_factor = 1/2 #for J=2 moments
        direction111  = np.array([0,0,1])
        projection = [direction111.dot(spin) for spin in self.SpinConfigMindBasis]

        magmom = np.array([x*direction111 for x in projection])

        self.MagneticMoment = g_factor*4*magmom
        self.BField = np.zeros(np.shape(self.StoppingSites))
        for stopping_site, s_site_position in enumerate(self.StoppingSites):
            self.BField[stopping_site,:] = TotDipolarField( s_site_position, self.SpinPositions, self.MagneticMoment)

        self.BField_norm = np.linalg.norm(self.BField,axis=1)

    def PlotFieldDistribution(self):
        fig, ax = plt.subplots()
        plt.hist(self.BField_norm, bins=60,range=(0,200))
        plt.title(f'$J^Q$ = {self.JQuad:.2f}, $J^O$ = {self.JOcto:.2f}, $J^B$ = {self.JB:.2f}')
        ax.set_xlabel(r"$|\vec{B}_{dip}|$ (G)")
        ax.set_ylabel(r"$count$",rotation=0)
        return fig


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

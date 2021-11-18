import numpy as np
import matplotlib.colors as clr
import matplotlib.patches as ptch
import matplotlib.pyplot as plt

from common import pi, PlotLineBetweenTwoPoints,sqrt3,a1,a2

class MonteCarloOutput:
    '''
    Parses Monte Carlo output for spins and their locations
    '''
    def __init__(self, filename):
        self.Filename = filename

        with open(self.Filename, 'r') as f:
            file_data = f.readlines()

        #extract which lattice: 0 for tri, 1 for hc, 2 for FCC
        self.WhichLattice = int(file_data[2])

        if ((self.WhichLattice == 0) or (self.WhichLattice == 1)):
            self.Dimensions=2
        elif self.WhichLattice == 2:
            self.Dimensions=3

        #extract linear dimensions
        self.L1, self.L2, self.L3 = [int(x) for x in file_data[4].split()]

        #extract conventional cell vectors
        self.T1, self.T2, self.T3 = np.array(list(map(np.double, [x.split() for x in file_data[6:6+3]]))).transpose()

        #extract sublattice vectors
        self.SublatticeVectors = np.array(list(map(np.double, [x.split() for x in file_data[10:10+3]]))).transpose()
        self.TotNumSublattices = self.SublatticeVectors.shape[0]
        self.NumSites = self.L1*self.L2*self.L3*self.TotNumSublattices

        #extract the deterministic sweeps: if 0, then it's finite T. else, it's SA
        self.DetSweeps = [int(x) for x in file_data[20].split()]

        #WARNING: FROM HERE ON, THE NUMBERS ONLY WORK WITH MULTIPOLE HAMILTONIAN (w/o defects).
        #         NEED A CLEVER WAY TO PARSE WHEN THERE ARE MULTIPLE HAMILTONIA
        #         ALTERNATIVELY, PUT C++ OUTPUT INTO HD5 FILE. WILL DO THIS WHEN
        #         I DO OTHER HAMILTONIANS

        #extract the pseudospin to mind basis transformation
        self.ChangeBasis = np.array(list(map(np.double, [x.split() for x in file_data[31:31+3]])))

        #extract the jtau/jb interactions
        self.JTau, self.JB = np.array(list(map( np.double, file_data[23].replace("/"," ").replace("\n","").split(" ") )))

        #extract the quad/octo interactions
        self.JQuad, self.JOcto = np.array(list(map( np.double, file_data[25].replace("/"," ").replace("\n","").split(" ") )))

        #extract the defect properties
        self.DefectQuad, self.DefectOcto, self.DefectLengthScale, self.NumDefects = np.array([0,0,0,0])

        #extract the energy
        self.MCEnergyPerSite = np.double(file_data[36])

        #extract spin positions (in mind basis)
        self.SpinConfigIndices = np.array(list(map(np.double, [x.split() for x in file_data[38:38+self.NumSites]])))[:,:4]

        self.SpinPositions = np.empty((self.NumSites, 3), dtype = np.double)
        for i, a in enumerate(self.SpinConfigIndices):
            self.SpinPositions[i] = a[0]*self.T1+a[1]*self.T2+a[2]*self.T3+self.SublatticeVectors[int(a[3])]
        # print(self.SpinPositions)

        #extract spins (in pseudospin and mind basis)
        self.SpinConfigSpinBasis = np.array(list(map(np.double, [x.split() for x in file_data[38:38+self.NumSites]])))[:,4:]
        self.SpinConfigMindBasis = (self.ChangeBasis @ self.SpinConfigSpinBasis.T).T

        self.LayerPositions, self.LayerSpins, self.LayerNumber = self.SortPlanes()

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

        if (self.NumDefects == 1) and ((self.DefectQuad!=0) or (self.DefectOcto!=0)):
            defe = ptch.Circle(3*self.L1/6 *self.T1 + 3*self.L2/6 *self.T2 +(self.T1+self.T2)/3,
            radius=self.DefectLengthScale*2, fill=True, alpha=0.1, linewidth=1.5,color='black')
            ax.add_patch(defe)

        plt.title(f'$J^\tau$ = {self.JTau:.2f}, $J^Q$ = {self.JQuad:.2f}, $J^O$ = {self.JOcto:.2f}, $J^B$ = {self.JB:.2f}')

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

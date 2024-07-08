import numpy as np
import time as Time
from tqdm import tqdm, trange
from scipy.special import gamma, factorial
from scipy.spatial.transform import Rotation as R
import logging

class transport:
    def __init__(self, boundaryType, maxMove, timeStep, pressure_pa, temperature, cellSize, celllength, chamberSize, DXsec, logname):
        self.boundaryType = boundaryType # 'wafer' 'SMD'
        self.pressure = pressure_pa
        self.T = temperature
        self.N_A = 6.02214076*10**23
        self.R = 8.31446261815324
        self.q = 1.60217663*10**-19
        self.Al_m = 44.803928e-27
        # self.IonMass = 39.938/(self.N_A*1000) # Argon?
        self.atomMass = 1.66054e-27
        self.kB = 1.380649e-23
        self.Al_atom = 26.98
        self.Ar_atom = 39.95
        self.Cm_Ar = (2*self.kB*self.T/(self.Ar_atom*self.atomMass) )**0.5 # (2kT/m)**0.5 39.95 for the Ar
        self.mp = 27 # Al
        self.mg = 40 # Ar
        self.epsilion = 8.85*10**(-12)
        self.ng_pa = self.pressure/(self.R*self.T)*self.N_A
        self.cellSize_x = cellSize[0]
        self.cellSize_y = cellSize[1]
        self.cellSize_z = cellSize[2]
        self.celllength = celllength
        self.tstep = timeStep
        self.maxMove = maxMove
        self.chamberX = chamberSize[0]
        self.chamberY = chamberSize[1]
        self.chamberZ = chamberSize[2]
        self.cellBinsX = np.linspace(self.chamberX[0][0], self.chamberX[0][1], self.cellSize_x)
        self.cellBinsY = np.linspace(self.chamberY[0][0], self.chamberY[0][1], self.cellSize_y)
        self.cellBinsZ = np.linspace(self.chamberZ[0][0], self.chamberZ[0][1], self.cellSize_z)
        self.Fn = 1e5
        self.sigmaTCR = 
        self.DXsec = DXsec
        self.depo_pos = []
        # self.colltype_list = []
        self.ionPos_list = []

        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        self.fh = logging.FileHandler(filename='./logfiles/{}.log'.format(logname), mode='w')
        self.fh.setLevel(logging.INFO)
        self.formatter = logging.Formatter(
                    fmt='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )
        self.fh.setFormatter(self.formatter)
        self.log.addHandler(self.fh)
        self.log.info('-------Start--------')
    

    def MaxwellMat(self, N):
        coschi = 2*np.random.rand(N) - 1
        sinchi = np.sqrt(1 - coschi**2)

        Random1 = np.random.rand(N)
        Random2 = np.random.rand(N)
        Random3 = np.random.rand(N)

        u = self.Cm_Ar*np.sqrt(-np.log(Random1))*(np.cos(2*np.pi*Random2))*sinchi

        w = self.Cm_Ar*np.sqrt(-np.log(Random1))*(np.sin(2*np.pi*Random2))*sinchi

        v = self.Cm_Ar*np.sqrt(-np.log(Random3))*coschi
        
        velosity_matrix = np.array([u, w, v]).T
        return velosity_matrix

    def boundary(self, posvel):

        if self.boundaryType == 'wafer':
            pos_radius = np.linalg.norm(posvel[:, :2], axis=1)
            indices = pos_radius >= self.chamberX
        elif self.boundaryType == 'SMD':
            indices = np.logical_or(posvel[:, 0] >= self.cellSize_x * self.celllength, posvel[:, 0] <= -self.cellSize_x * self.celllength)
            indices |= np.logical_or(posvel[:, 1] >= self.cellSize_y * self.celllength, posvel[:, 1] <= -self.cellSize_y * self.celllength)
        else:
            print('no boundary')

        if np.any(indices):
            posvel = posvel[~indices]

        depo_indices = np.logical_or(posvel[:,2] > self.cellSize_z * self.celllength, posvel[:,2] < 0)

        if np.any(depo_indices):
            self.depo_pos.append(posvel[depo_indices])
            posvel = posvel[~depo_indices]

        return posvel
    
    # def depo_position(self, posvel):
    #     self.depo_pos.append(posvel)

    def getAcc_sparse(self, posvel, tstep):

        i = np.floor((posvel[:, 0]/self.celllength) + 0.5).astype(int)
        j = np.floor((posvel[:, 1]/self.celllength) + 0.5).astype(int)
        k = np.floor((posvel[:, 2]/self.celllength) + 0.5).astype(int)

        nC, edges = np.histogramdd(posvel[:, :3], bins = (self.cellBinsX, self.cellBinsY, self.cellBinsZ))

        posvelBoundary = self.boundary(posvel)
        posvelAcc = np.zeros_like(posvelBoundary)
        posvelAcc[:, :3] = posvelBoundary[:, 3:] * tstep + posvelBoundary[:, :3]
        posvelAcc[:, 3:] = posvelBoundary[:, 3:]

        return posvelAcc, posvelBoundary, nC
    
    def diVr_func(self, d_refi, eVr, wi):
        kb = 1.380649e-23
        Tref = 650
        diVr = d_refi * np.sqrt(((kb*Tref)/(eVr*self.q))**(wi-1/2)*gamma(5/2 - wi))
        return diVr

    def TotXsec(self, d_refi, eVr, wi):
        return np.pi * self.diVr_func(d_refi, eVr, wi)**2

    def setXsec(self, energy_range):
        energy = np.linspace(energy_range[0], energy_range[1], energy_range[2])
        self.Xsec = energy
        totXsection = self.TotXsec((4.614 + 4.151)/2*1e-10, energy, 0.7205)
        totXsection[0] = 0
        self.Xsectot = totXsection
        return self.Xsectot

    def rotate_matrix(self, phi, theta):
        phi_array = np.expand_dims(phi, axis=(0, 1))
        theta_array = np.expand_dims(theta, axis=(0, 1))
        cos_phi = np.cos(phi_array)
        sin_phi = np.sin(phi_array)
        cos_theta = np.cos(theta_array)
        sin_theta = np.sin(theta_array)
        matrix_i = np.concatenate((cos_theta*cos_phi, cos_theta*sin_phi, -sin_theta), axis=1)
        matrix_j = np.concatenate((matrix_i, np.concatenate((-sin_phi, cos_phi, np.zeros_like(phi_array)), axis=1)), axis=0)
        matrix = np.concatenate((matrix_j, np.concatenate((sin_theta*cos_phi, sin_theta*sin_phi, cos_theta), axis=1)), axis=0)

        return matrix.T
    
    def DCS_pdf(self, energy):
        randChoice = np.random.randint(int(1e6), size=(energy.shape[0]))
        chi = self.DXsec[randChoice]
        return chi

    def newVel_gpu(self, cp, cg):
        N = cp.shape[0]
        # cg = self.MaxwellMat(N)
        cm = (self.Al_atom*cp + self.Ar_atom*cg)/(self.Al_atom+self.Ar_atom)
        cr = cp - cg
        cr_mag = np.linalg.norm(cr, axis=1)
        eps = np.random.rand(N)*np.pi*2
        coschi = np.random.rand(N)*2 - 1
        sinchi = np.sqrt(1 - coschi*coschi)
        postCollision_cr = cr_mag*np.array([coschi, sinchi*np.cos(eps),  sinchi*np.sin(eps)]).T
        cp_p = cm + self.Ar_atom/(self.Al_atom + self.Ar_atom)*postCollision_cr
        cg_p = cm - self.Al_atom/(self.Al_atom + self.Ar_atom)*postCollision_cr
        return cp_p, cg_p

    def addPtToList(self, pt):
        self.ionPos_list.append(pt)

    def collision(self, prob, KE, vMag, posvelAcc):
        rand_val = np.random.rand(prob.shape[0])
        indices = np.nonzero(rand_val < prob)[0]
        if indices.size != 0:
            v3 = self.newVel_gpu(posvelAcc[indices][:, 3:], vMag[indices])
            posvelAcc[:, 3:][indices] = v3
            self.addPtToList(posvelAcc[indices][:, :3])
            return posvelAcc, indices.shape[0]
        else:
            return posvelAcc, 0
        
    def collProb(self, n, KE, delx, nC):

        selectedPairs = 0.5*nC*(nC-1)*self.Fn*self.sigmaTCR*self.tstep
        Xsec_interp = np.interp(KE, self.Xsec[:], self.Xsectot[:])
        sigTot = Xsec_interp
        return 1 - np.exp(-n * sigTot * delx)
    
    def runE(self, p0, v0, time):
        PosVel = np.concatenate((p0, v0), axis=1)
        tmax = time
        tstep = self.tstep
        t = 0

        with tqdm(total=100, desc='running', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
            i = 0
            while t < tmax:
                posvelAcc, posvelBoundary, nC = self.getAcc_sparse(PosVel, tstep)
                if posvelAcc.shape[0] < int(1e5):
                    break
                delx = np.linalg.norm(posvelAcc[:, :3] - posvelBoundary[:, :3], axis=1)
                vMag = np.linalg.norm(posvelBoundary[:, 3:], axis=1)
                vMax = vMag.max()
                KE = 0.5*self.Al_m*vMag**2/self.q
                prob = self.collProb(self.ng_pa, KE, delx, nC)
                # posvelCopy = np.copy(posvelAcc)
                posvelAcc, collsionNum = self.collision(prob, KE, vMag, posvelAcc)
                # rotateTure = np.allclose(posvelCopy, posvelAcc)
                t += self.tstep
                PosVel = posvelAcc
                if int(t/tmax*100) > i:
                    Time.sleep(0.01)
                    pbar.update(1)
                    i += 1

                if vMax*tstep < self.maxMove*self.celllength:                    
                    tstep *= 2
                elif vMax*tstep > self.maxMove*2*self.celllength:
                    tstep /= 2

                self.log.info('runStep:{}, timeStep:{}, vMaxMove:{:.3f}, collsion:{}, particleIn:{}'\
                        .format(i, tstep, vMax*tstep/self.celllength, collsionNum, PosVel.shape[0]))
        return self.ionPos_list, self.depo_pos
    

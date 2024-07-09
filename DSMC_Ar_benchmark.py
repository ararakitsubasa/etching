import numpy as np
import time as Time
from tqdm import tqdm, trange
from scipy.special import gamma, factorial, erf
from scipy.spatial.transform import Rotation as R

import logging

class transport:
    def __init__(self, mirror, maxMove, timeStep, pressure_pa, temperature, cellSize, celllength, chamberSize, DXsec, logname):
        self.symmetry = mirror
        self.pressure = pressure_pa
        self.T = temperature
        self.N_A = 6.02214076*10**23
        self.R = 8.31446261815324
        self.q = 1.60217663*10**-19
        self.Al_m = 44.803928e-27
        self.IonMass = 39.938/(self.N_A*1000)
        self.atomMass = 1.66054e-27
        self.kB = 1.380649e-23
        self.mp = 27 # Al
        self.mg = 40 # Ar
        self.Al_atom = 26.98
        self.Ar_atom = 39.95
        self.Ar_radius = 188e-12
        self.Al_radius = 184e-12
        self.sigmaT = np.pi*(self.Ar_radius + self.Al_radius)**2/4
        self.Cm_Ar = (2*self.kB*self.T/(self.Ar_atom*self.atomMass) )**0.5 # (2kT/m)**0.5 39.95 for the Ar
        self.epsilion = 8.85*10**(-12)
        self.vg = np.sqrt(2*self.kB*self.T/self.IonMass)
        self.ng_pa = self.pressure/(self.R*self.T)*self.N_A
        self.cellSizeX = cellSize[0]
        self.cellSizeY = cellSize[1]
        self.cellSizeZ = cellSize[2]
        self.celllength = celllength
        self.tstep = timeStep
        self.maxMove = maxMove
        self.chamberX = chamberSize[0]
        self.chamberY = chamberSize[1]
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
      
    def boundary(self, pos, vel, i, j, k, weights_arr):
        # print(pos)
        pos_cp = np.asarray(pos)
        vel_cp = np.asarray(vel)
        weights_arr_cp = np.asarray(weights_arr)
        i_cp = np.asarray(i)
        j_cp = np.asarray(j)
        k_cp = np.asarray(k)
        cellSize_x_cp = np.asarray(self.cellSizeX) 
        cellSize_y_cp = np.asarray(self.cellSizeY) 
        cellSize_z_cp = np.asarray(self.cellSizeZ) 

        if self.symmetry == True:
            indiceXMax = i_cp >= cellSize_x_cp
            indiceXMin = i_cp < 0

            # 使用布尔索引进行调整
            i_cp[indiceXMax] -= cellSize_x_cp 
            pos_cp[indiceXMax, 0] -= self.celllength * self.cellSizeX

            i_cp[indiceXMin] += cellSize_x_cp
            pos_cp[indiceXMin, 0] += self.celllength * self.cellSizeX

            # 检查并调整 j_cp 和对应的 pos_cp
            indiceYMax = j_cp >= cellSize_y_cp
            indiceYMin = j_cp < 0

            # 使用布尔索引进行调整
            j_cp[indiceYMax] -= cellSize_y_cp
            pos_cp[indiceYMax, 1] -= self.celllength * self.cellSizeY

            j_cp[indiceYMin] += cellSize_y_cp
            pos_cp[indiceYMin, 1] += self.celllength * self.cellSizeY

            # 检查并调整 j_cp 和对应的 pos_cp
            indiceZMax = k_cp >= cellSize_z_cp
            indiceZMin = k_cp < 0

            # 使用布尔索引进行调整
            k_cp[indiceZMax] -= cellSize_z_cp
            pos_cp[indiceZMax, 2] -= self.celllength * self.cellSizeZ

            k_cp[indiceZMin] += cellSize_z_cp
            pos_cp[indiceZMin, 2] += self.celllength * self.cellSizeZ

        indices = np.logical_or(i_cp >= cellSize_x_cp, i_cp < 0)
        indices |= np.logical_or(j_cp >= cellSize_y_cp, j_cp < 0)
        indices |= np.logical_or(k_cp >= cellSize_z_cp, k_cp < 0)

        if np.any(indices):
            pos_cp = pos_cp[~indices]
            vel_cp = vel_cp[~indices]
            weights_arr_cp = weights_arr_cp[~indices]
            i_cp = i_cp[~indices]
            j_cp = j_cp[~indices]
            k_cp = k_cp[~indices]

        return pos_cp, vel_cp, i_cp, j_cp, k_cp, weights_arr_cp
    
    # def depo_position(self, posvel):
    #     self.depo_pos.append(posvel)

    def getAcc_sparse(self, posvel, tstep):

        posvelBoundary = self.boundary(posvel)
        posvelAcc = np.zeros_like(posvelBoundary)
        posvelAcc[:, :3] = posvelBoundary[:, 3:] * tstep + posvelBoundary[:, :3]
        posvelAcc[:, 3:] = posvelBoundary[:, 3:]

        return posvelAcc, posvelBoundary
    
    def getAcc_depo(self, pos, vel, boxsize, tStep, film, weights_arr, depoStep):
        dx = boxsize

        pos_cp = pos
        vel_cp = vel

        tStep_cp = tStep

        i = np.floor((pos_cp[:, 0]/dx) + 0.5).astype(int)
        j = np.floor((pos_cp[:, 1]/dx) + 0.5).astype(int)
        k = np.floor((pos_cp[:, 2]/dx) + 0.5).astype(int)

        # pos, vel, i, j, k, cellSize_x, cellSize_y, cellSize_z,
        pos_cp, Nvel_cp, i, j, k, weights_arr = self.boundary(pos_cp, vel_cp, i, j, k, weights_arr)
        # print(pos_cp)
        film_depo, pos_cp, Nvel_cp, weights_arr_depo, depo_count, film_max = self.depo_film(film, pos_cp, Nvel_cp, i, j, k, weights_arr, depoStep)

        Npos2_cp = Nvel_cp * tStep_cp + pos_cp

        return np.array([pos_cp, Nvel_cp]), np.array([Npos2_cp, Nvel_cp]), film_depo, weights_arr_depo, depo_count, film_max


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

    def newVel_gpu(self, v, vMag):
        # print(v.shape)
        energy = 0.5*self.Al_m*vMag**2/self.q
        pN = energy.shape[0]
        vMagnew = vMag
        vfilp = np.ones(v.shape[0])
        filp_indice = np.array(v[:, 0] < 0)
        vfilp[filp_indice] = -1
        theta0 = np.arccos(v[:, 2]/vMag)*vfilp
        phi0= np.arctan(v[:, 1]/v[:, 0])
        chi = self.DCS_pdf(energy)
        phi = 2*np.pi*np.random.rand(pN)
        rotateMat = np.array([np.sin(chi)*np.cos(phi), np.sin(chi)*np.sin(phi), np.cos(chi)])
        Vrotate = np.multiply(rotateMat, vMagnew).T
        rz = R.from_matrix(self.rotate_matrix(phi0, theta0))
        v_rotate = rz.apply(Vrotate)
        v_rotate = (self.mp*v + self.mg*v_rotate)/(self.mp+self.mg)
        return v_rotate 

    def newVel_gpu_cr(self, cp):
        N = cp.shape[0]
        cg = self.MaxwellMat(N)
        cm = (self.Al_atom*cp + self.Ar_atom*cg)/(self.Al_atom+self.Ar_atom)
        cr = cp - cg
        cr_mag = np.linalg.norm(cr, axis=1)
        eps = np.random.rand(N)*np.pi*2
        coschi = np.random.rand(N)*2 - 1
        sinchi = np.sqrt(1 - coschi*coschi)
        # postCollision_cr = cr_mag*np.array([coschi, sinchi*np.cos(eps),  sinchi*np.sin(eps)]).T
        postCollision_cr = np.array([coschi*cr_mag, sinchi*np.cos(eps)*cr_mag,  sinchi*np.sin(eps)*cr_mag]).T
        cp_p = cm + self.Ar_atom/(self.Al_atom + self.Ar_atom)*postCollision_cr
        # cg_p = cm - self.Al_atom/(self.Al_atom + self.Ar_atom)*postCollision_cr
        return cp_p
    
    def addPtToList(self, pt):
        self.ionPos_list.append(pt)

    def collision(self, prob, posvelAcc):
        rand_val = np.random.rand(prob.shape[0])
        indices = np.nonzero(rand_val < prob)[0]
        if indices.size != 0:
            # v3 = self.newVel_gpu(posvelAcc[indices][:, 3:], vMag[indices])
            v3 = self.newVel_gpu_cr(posvelAcc[indices][:, 3:])
            posvelAcc[:, 3:][indices] = v3
            self.addPtToList(posvelAcc[indices][:, :3])
            return posvelAcc, indices.shape[0]
        else:
            return posvelAcc, 0
        
    def relVel(self, vmag):
        x = vmag/self.vg
        return self.vg*((x + 0.5/x)*erf(x) + 1/np.sqrt(np.pi)*np.exp(-x*x))

    def collProb(self, n, KE, delx):
        # Xsec_interp = np.interp(KE, self.Xsec[:], self.Xsectot[:])
        sigTot = self.sigmaT
        return 1 - np.exp(-n * sigTot * delx)
    
    def runE(self, p0, v0, time):
        PosVel = np.concatenate((p0, v0), axis=1)
        tmax = time
        tstep = self.tstep
        t = 0

        with tqdm(total=100, desc='running', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
            i = 0
            while t < tmax:
                posvelAcc, posvelBoundary = self.getAcc_sparse(PosVel, tstep)
                if posvelAcc.shape[0] < int(1e5):
                    break
                # delx = np.linalg.norm(posvelAcc[:, :3] - posvelBoundary[:, :3], axis=1)
                vMag = np.linalg.norm(posvelBoundary[:, 3:], axis=1)
                delx = self.relVel(vMag)*tstep
                vMax = vMag.max()
                KE = 0.5*self.Al_m*vMag**2/self.q
                prob = self.collProb(self.ng_pa, KE, delx)
                # posvelCopy = np.copy(posvelAcc)
                posvelAcc, collsionNum = self.collision(prob, posvelAcc)
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
    

import numpy as np
import time as Time
from tqdm import tqdm, trange
from scipy.special import gamma, factorial

class transport:
    def __init__(self, timeStep, pressure_pa, temperature, cellSize, celllength, chamberSize, DXsec):
        self.pressure = pressure_pa
        self.T = temperature
        self.N_A = 6.02214076*10**23
        self.R = 8.31446261815324
        self.q = 1.60217663*10**-19
        self.Al_m = 44.803928e-27
        self.IonMass = 39.938/(self.N_A*1000)
        self.epsilion = 8.85*10**(-12)
        self.ng_pa = self.pressure/(self.R*self.T)*self.N_A
        self.cellSize_x = cellSize[0]
        self.cellSize_y = cellSize[1]
        self.cellSize_z = cellSize[2]
        self.celllength = celllength
        self.tstep = timeStep
        self.chamberX = chamberSize[0]
        self.chamberY = chamberSize[1]
        self.DXsec = DXsec
        self.depo_pos = np.zeros((1,6))

    
    def boundary(self, pos, vel):
        pos_cp = np.asarray(pos)
        vel_cp = np.asarray(vel)

        pos_radius = np.linalg.norm(np.array([pos_cp[:, 0], pos_cp[:, 1]]), axis=0)

        indices = np.array(pos_radius >= self.chamberX)

        if np.any(indices):
            pos_cp = pos_cp[~indices]
            vel_cp = vel_cp[~indices]

        depo_indices = np.logical_or(pos_cp[:,2] > self.cellSize_z * self.celllength, pos_cp[:,2] < 0)

        if np.any(depo_indices):
            self.depo_position(pos_cp[depo_indices], vel_cp[depo_indices])
            pos_cp = pos_cp[~depo_indices]
            vel_cp = vel_cp[~depo_indices]

        return pos_cp, vel_cp
    
    def depo_position(self, pos, vel):
        PosVel = np.concatenate((pos, vel), axis=1)
        self.depo_pos = np.vstack((self.depo_pos, PosVel))

    def getAcc_sparse(self, pos, vel):

        pos_cp = pos
        vel_cp = vel
        tStep_cp = self.tstep

        pos_cp, Nvel_cp = self.boundary(pos_cp, vel_cp)

        Nvel2_cp = Nvel_cp
        cpos2_cp = Nvel_cp * tStep_cp + pos_cp

        return np.array([pos_cp, Nvel_cp]), np.array([cpos2_cp, Nvel2_cp])
    
    def diVr_func(self, d_refi, eVr, wi):
        kb = 1.380649e-23
        Tref = 300
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

        energy = 0.5*self.Al_m*vMag**2/self.q
        vMagnew = vMag
        theta0 = np.arccos(v[:, 2]/vMag)
        phi0= np.arctan(v[:, 0]/v[:, 1])
        chi = self.DCS_pdf(energy)
        # chi = np.arccos(1-2*r/(1+8*(energy/27.21)*(1-r)))
        phi = 2*np.pi*np.random.rand()
        m1m2 = self.rotate_matrix(phi0, theta0)
        rotateAngle = np.array([np.sin(chi)*np.cos(phi), np.sin(chi)*np.sin(phi), np.cos(chi)])
        dot_products = np.einsum('...ij,...i->...j', m1m2, rotateAngle.T)

        newVel = dot_products * vMagnew[:, np.newaxis]

        return newVel

    def addPtToList(self, pt, colltype, colltype_list, ionPos_list):
        colltype_list = np.hstack((colltype_list, colltype))
        indice = np.nonzero(colltype == 0)
        ionPos_list = np.concatenate((ionPos_list, pt[indice[0]]))

        return colltype_list, ionPos_list

    def collision(self, prob, colltype_list, elist, KE, vMag, p2, v2):
        rand_val = np.random.rand(prob.shape[0])
        indices = np.nonzero(rand_val < prob)
        if indices[0].size == 0:
            return colltype_list, elist, v2
        else:
            colltype = np.zeros(KE[indices[0]].size)
            v3 = self.newVel_gpu(v2[indices[0]], vMag[indices[0]])
            v2[indices[0]] = v3
            collList, elist = self.addPtToList(p2[indices[0]], colltype, colltype_list, elist)
            return collList, elist, v2
        
    def collProb(self, n, KE, delx):
        Xsec_interp = np.interp(KE, self.Xsec[:], self.Xsectot[:])
        sigTot = Xsec_interp
        return 1 - np.exp(-n * sigTot * delx)
    
    def runE(self, p0, v0, time):
        # q = -1.60217663*10**-19
        # m = 9.1093837*10**-31
        tmax = time
        # tstep = 10**-11
        t = 0
        p1 = p0
        v1 = v0
        collList = np.array([])
        elist = np.array([[0, 0, 0]])
        with tqdm(total=100, desc='running', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
            i = 0
            while t < tmax:
                p2v2 = self.getAcc_sparse(p1, v1)
                p2 = p2v2[1][0]
                if p2.shape[0] == 0:
                    break
                v2 = p2v2[1][1]
                p1 = p2v2[0][0]
                v1 = p2v2[0][1]
                # delx = np.linalg.norm(p1 - p2, axis=1)
                # vMag = np.linalg.norm(v1, axis=1)
                # KE = 0.5*self.Al_m*vMag**2/self.q
                # prob = self.collProb(self.ng_pa, KE, delx)
                # collList, elist, v2 = self.collision(prob, collList, elist, KE, vMag, p2, v2)
                t += self.tstep
                p1 = p2
                v1 = v2
                i += 1
                if i % (int((tmax/self.tstep)/100)) == 0:
                    Time.sleep(0.01)
                    # 更新发呆进度
                    pbar.update(1)
        return collList, elist, self.depo_pos[1:]
    

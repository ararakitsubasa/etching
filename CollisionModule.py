import numpy as np
from scipy.spatial.transform import Rotation as R
import time as Time
from tqdm import tqdm, trange
from scipy.special import gamma, factorial

class transport:
    def __init__(self, timeStep, pressure_pa, temperature, cellSize, celllength, chamberSize):
        self.pressure = pressure_pa
        self.T = temperature
        self.N_A = 6.02214076*10**23
        self.R = 8.31446261815324
        self.q = -1.60217663*10**-19
        self.m = 9.1093837*10**-31
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

    
    def boundary(self, pos, vel, i, j, k):
        pos_cp = np.asarray(pos)
        vel_cp = np.asarray(vel)
        i_cp = np.asarray(i)
        j_cp = np.asarray(j)
        k_cp = np.asarray(k)
        cellSize_x_cp = np.asarray(self.cellSize_x - 1)
        cellSize_y_cp = np.asarray(self.cellSize_y - 1)
        cellSize_z_cp = np.asarray(self.cellSize_z - 1)

        indices = np.logical_or(i_cp >= cellSize_x_cp, i_cp <= 0)
        indices |= np.logical_or(j_cp >= cellSize_y_cp, j_cp <= 0)
        indices |= np.logical_or(k_cp >= cellSize_z_cp, k_cp < 0)

        if np.any(indices):
            pos_cp = pos_cp[~indices]
            vel_cp = vel_cp[~indices]
            i_cp = i_cp[~indices]
            j_cp = j_cp[~indices]
            k_cp = k_cp[~indices]

        return pos_cp, vel_cp, i_cp, j_cp, k_cp
    
    def getAcc_sparse(self, pos, vel):

        dx = self.celllength # 0.3/cellsize
        start_x = self.chamberX
        start_y = self.chamberY
        pos_cp = pos

        vel_cp = vel
        tStep_cp = self.tstep

        i = np.floor((pos_cp[:, 0] - start_x) / dx).astype(int)
        j = np.floor((pos_cp[:, 1] - start_y) / dx).astype(int)
        k = np.floor(pos_cp[:, 2] / dx).astype(int)

        pos_cp, Nvel_cp, i, j, k = self.boundary(pos_cp, vel_cp, i, j, k)

        Nvel2_cp = Nvel_cp
        cpos2_cp = Nvel_cp * tStep_cp + pos_cp

        return np.array([pos_cp, Nvel_cp]), np.array([cpos2_cp, Nvel2_cp])
    
    def diVr_func(d_refi, eVr, wi):
        kb = 1.380649e-23
        Tref = 300
        ev = 1.62e-19
        diVr = d_refi * np.sqrt(((kb*Tref)/(eVr*ev))**(wi-1/2)*gamma(5/2 - wi))
        return diVr

    def TotXsec(d_refi, eVr, wi):
        return np.pi * diVr_func(d_refi, eVr, wi)**2

    def setXsec(self):
        self.Xsec = np.array([[0, 7.79, 0, 0],
                [1, 1.43, 0, 0],
                [2, 3.52, 0, 0],
                [3, 5.5, 0, 0],
                [4, 7.18, 0, 0],
                [5, 9.10, 0, 0],
                [6, 11.2, 0, 0],
                [7, 13.65, 0, 0],
                [8, 16.1, 0, 0],
                [9, 18.25, 0, 0],
                [10, 20.4, 0, 0],
                [11, 21.8, 0, 0],
                [12, 23.2, 0.0164, 0],
                [13, 23.4, 0.0544, 0],
                [14, 23.6, 0.11, 0],
                [15, 23.8, 0.1622, 0],
                [16, 22.98, 0.2033, 0.0202],
                [17, 22.16, 0.2374, 0.134],
                [18, 21.34, 0.2603, 0.294],
                [19, 20.52, 0.2766, 0.46],
                [20, 19.7, 0.292, 0.627],
                [25, 15.5, 0.3267, 1.3],
                [30, 12.5, 0.3483, 1.8],
                [35, 10.815, 0.3678, 2.175],
                [40, 9.13, 0.385, 2.39],
                [45, 8.255, 0.3919, 2.49],
                [50, 7.38, 0.3988, 2.53],
                [60, 6.34, 0.3963, 2.66],
                [70, 5.765, 0.3841, 2.77],
                [80, 5.19, 0.3641, 2.84],
                [90, 4.905, 0.3452, 2.86],
                [100, 4.62, 0.3263, 2.85],
                [150, 3.7, 0.2457, 2.68],
                [200, 3.13, 0.2012, 2.39],
                [250, 2.77, 0.1762, 2.17],
                [300, 2.5, 0.1511, 1.98],
                [350, 2.32, 0.1352, 1.81],
                [400, 2.14, 0.1225, 1.68],
                [450, 2.015, 0.1134, 1.57],
                [500, 1.89, 0.1044, 1.46],
                [600, 1.71, 0.0941, 1.3],
                [700, 1.58, 0.0839, 1.16],
                [800, 1.45, 0.0736, 1.06],
                [900, 1.355, 0.0678, 0.988],
                [1000, 1.26, 0.0621, 0.916],
                [2000, 1.12, 0.0552, 0.8142],
                [10000, 0, 0, 0]], dtype=float)

        self.Xsectot = self.Xsec[:, 1] + self.Xsec[:, 2] + self.Xsec[:, 3]
        return self.Xsec, self.Xsectot

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
    
    def DecideEvent(self, energy):
        r = np.random.rand(energy.size)
        energy_np = energy
        XsecEla = np.interp(energy_np, self.Xsec[:-2, 0], self.Xsec[:-2, 1]) * 10**-20
        XsecExc = np.interp(energy_np, self.Xsec[:-2, 0], self.Xsec[:-2, 2]) * 10**-20
        sigTot = np.interp(energy_np, self.Xsec[:-2, 0], self.Xsectot[:-2]) * 10**-20

        colltype = np.zeros(energy.size)

        Indice =  np.logical_and(r < XsecEla / sigTot, energy > 1)
        colltype[Indice] = 1
        Indice |= np.logical_and(r < (XsecEla + XsecExc) / sigTot, energy > 12)

        maskE = np.isin(np.where(Indice == True)[0], np.where(colltype == 0)[0])
        exiteE = np.where(Indice == True)[0][maskE]

        colltype[exiteE] = 2
        Indice |= np.logical_and(r > (XsecEla + XsecExc), energy > 16)

        maskI = np.isin(np.where(Indice == True)[0], np.where(colltype == 0)[0])
        IonE = np.where(Indice == True)[0][maskI]

        colltype[IonE] = 3

        return colltype


    def newVel_gpu(self, v, colltype, vMag):
        # m = 9.11*10**-31
        # q = 1.6*10**-19
        energy = 0.5*self.m*vMag**2/-self.q

        a = np.where(colltype == 1)
        energy[a] -= 1

        b = np.where(colltype == 2)
        energy[b] -= 12

        c = np.where(colltype == 3)
        energy[c] -= 16

        vMagnew = (2*energy*-self.q/self.m)**0.5
        theta0 = np.arccos(v[:, 2]/vMag)
        phi0= np.arctan(v[:, 1]/v[:, 0])
        r = np.random.rand()
        chi = np.arccos(1-2*r/(1+8*(energy/27.21)*(1-r)))
        phi = 2*np.pi*np.random.rand()
        m1m2 = self.rotate_matrix(phi0, theta0)
        rotateAngle = np.array([np.sin(chi)*np.cos(phi), np.sin(chi)*np.sin(phi), np.cos(chi)])
        dot_products = np.einsum('...ij,...i->...j', m1m2, rotateAngle.T)

        newVel = dot_products * vMagnew[:, np.newaxis]

        return newVel

    def addPtToList(self, pt, colltype, colltype_list, ionPos_list):
        colltype_list = np.hstack((colltype_list, colltype))
        indice = np.nonzero(colltype == 3)
        ionPos_list = np.concatenate((ionPos_list, pt[indice[0]]))

        return colltype_list, ionPos_list

    def collision(self, prob, colltype_list, elist, KE, vMag, p2, v2):
        rand_val = np.random.rand(prob.shape[0])
        indices = np.nonzero(rand_val < prob)

        if indices[0].size == 0:
            return colltype_list, elist, v2
        else:
            colltype = self.DecideEvent(KE[indices[0]])
            v3 = self.newVel_gpu(v2[indices[0]], colltype, vMag[indices[0]])
            v2[indices[0]] = v3
            collList, elist = self.addPtToList(p2[indices[0]], colltype, colltype_list, elist)
            return collList, elist, v2
        
    def collProb(self, n, KE, delx):
        Xsec_interp = np.interp(KE, self.Xsec[:-2, 0], self.Xsectot[:-2])
        sigTot = Xsec_interp * 10**-20
        return 1 - np.exp(-n * sigTot * delx)
    
    def runE(self, p0, v0, time):
        # q = -1.60217663*10**-19
        # m = 9.1093837*10**-31
        tmax = time
        # tstep = 10**-11
        t = 0
        p1 = p0
        v1 = v0
        cellSize = 50
        collList = np.array([])
        elist = np.array([[0, 0, 0]])
        cell = 0.1/(cellSize)
        with tqdm(total=100, desc='running', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
            i = 0
            while t < tmax:
                p2v2 = self.getAcc_sparse(p1, v1, cell, self.B_field, self.sheath_cell, self.tstep)
                p2 = p2v2[1][0]
                if p2.shape[0] == 0:
                    break
                v2 = p2v2[1][1]
                p1 = p2v2[0][0]
                v1 = p2v2[0][1]
                delx = np.linalg.norm(p1 - p2, axis=1)
                vMag = np.linalg.norm(v1, axis=1)
                KE = -0.5*self.m*vMag**2/self.q
                prob = self.collProb(self.ng_pa, KE, delx)
                collList, elist, v2 = self.collision(prob, collList, elist, KE, vMag, p2, v2)
                t += self.tstep
                p1 = p2
                v1 = v2
                i += 1
                if i % (int((tmax/self.tstep)/100)) == 0:
                    Time.sleep(0.01)
                    # 更新发呆进度
                    pbar.update(1)
        return collList, elist
    

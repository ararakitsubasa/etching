import numpy as np
import time as Time
from tqdm import tqdm, trange
from scipy.special import gamma, factorial, erf
from scipy.spatial.transform import Rotation as R

import logging

class DSMC_Bird:
    def __init__(self, mirror, maxMove, timeStep, pressure_pa, temperature, \
                cellSize, celllength, chamberSize, \
                numberDensity1, numberDensity2, nParticle, \
                logname):
        
        self.symmetry = mirror

        # physics constant
        self.N_A = 6.02214076*10**23
        self.R = 8.31446261815324
        self.q = 1.60217663*10**-19
        self.kB = 1.380649e-23
        self.epsilion = 8.85*10**(-12)
        self.atomMass = 1.66054e-27

        self.Al_m = 44.803928e-27
        self.Ar_m = 39.938/(self.N_A*1000)
        self.Ar_atom = 39.95
        self.Al_atom = 26.98
        self.Ar_dref = 4.614e-10
        self.Al_dref = 4.151e-10
        self.Ar_Dof = 0
        self.Al_Dof = 0
        self.Ar_Ei = 0
        self.Al_Ei = 0
        self.Ar_Omega = 0.721
        self.Al_Omega = 0.72

        self.Cm_Ar = (2*self.kB*self.T/(self.Ar_atom*self.atomMass) )**0.5 # (2kT/m)**0.5 39.95 for the Ar

        self.ng_pa = self.pressure/(self.R*self.T)*self.N_A

        # mesh 
        self.cellSizeX = cellSize[0]
        self.cellSizeY = cellSize[1]
        self.cellSizeZ = cellSize[2]
        self.celllength = celllength
        self.meshVolumes = self.celllength**3
        self.tstep = timeStep
        self.maxMove = maxMove
        self.chamberX = chamberSize[0]
        self.chamberY = chamberSize[1]
        self.chamberZ = chamberSize[2]
        self.cellArrayX = np.arange(self.cellSizeX)*self.celllength
        self.cellArrayY = np.arange(self.cellSizeY)*self.celllength
        self.cellArrayZ = np.arange(self.cellSizeZ)*self.celllength

        # scalar
        self.Tref = temperature
        self.pressure = pressure_pa
        self.numberDensity1 = numberDensity1
        self.numberDensity2 = numberDensity2
        self.nParticle = nParticle

        # Scaler field
        self.rhoN = np.zeros((self.cellSizeX, self.cellSizeY, self.cellSizeZ))
        self.rhoM = np.zeros((self.cellSizeX, self.cellSizeY, self.cellSizeZ))
        self.dsmcrhoN = np.zeros((self.cellSizeX, self.cellSizeY, self.cellSizeZ))
        self.linearKE = np.zeros((self.cellSizeX, self.cellSizeY, self.cellSizeZ))
        self.internalE = np.zeros((self.cellSizeX, self.cellSizeY, self.cellSizeZ))
        self.iDof = np.zeros((self.cellSizeX, self.cellSizeY, self.cellSizeZ))
        self.momentum = np.zeros((self.cellSizeX, self.cellSizeY, self.cellSizeZ))

        self.sigmaTCRMax = np.zeros((self.cellSizeX, self.cellSizeY, self.cellSizeZ))
        self.sigmaTCRMax += np.pi*self.Ar_dref**2 * np.sqrt(2*self.R*self.Tref/self.Ar_atom)
        self.collisionSelectedRemainder = np.ones((self.cellSizeX, self.cellSizeY, self.cellSizeZ))*np.random.rand()
        # vector field


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
    
    def resetFields(self):
        self.rhoN = np.zeros((self.cellSizeX, self.cellSizeY, self.cellSizeZ))
        self.rhoM = np.zeros((self.cellSizeX, self.cellSizeY, self.cellSizeZ))
        self.dsmcrhoN = np.zeros((self.cellSizeX, self.cellSizeY, self.cellSizeZ))
        self.linearKE = np.zeros((self.cellSizeX, self.cellSizeY, self.cellSizeZ))
        self.internalE = np.zeros((self.cellSizeX, self.cellSizeY, self.cellSizeZ))
        self.iDof = np.zeros((self.cellSizeX, self.cellSizeY, self.cellSizeZ))
        self.momentum = np.zeros((self.cellSizeX, self.cellSizeY, self.cellSizeZ))

    def constProp(self, parcel):
        if parcel[6] == 1:
            return self.Ar_m, self.Ar_Dof, self.Ar_dref, self.Ar_Ei, self.Ar_Omega
        if parcel[6] == 2:
            return self.Al_m, self.Al_Dof, self.Al_dref, self.Al_Ei, self.Al_Omega
        

    # VHS
    def sigmaTcR(self, p, q):
        pmass, pdof, pdref, pei, pomega = self.constProp(p)
        qmass, qdof, qdref, qei, qomega = self.constProp(q)

        dpq = (pdref + qdref)/2 
        omegapq = (pomega + qomega)/2 
        mr = (pmass*qmass)/(pmass+qmass)
        cr = np.linalg.norm(p[3:6] - q[3:6])
        sigmaTPQ = np.pi*dpq*dpq * np.sqrt(((2*self.kB*self.Tref)/(mr*cr*cr))**(omegapq-1/2)*gamma(2.5 - omegapq))
        return sigmaTPQ*cr

    def calculateFields(self, parcel):
        # particle data struction np.array([posX, posY, posZ, velX, velY, velZ, typeID])
        n = np.histogramdd(parcel[:, :3], bins=(self.cellArrayX, self.cellArrayY, self.cellArrayZ))[0]
        self.rhoN += n
        n1 = np.histogramdd(parcel[np.where(parcel[:, 6] == 1)][:, :3], bins=(self.cellArrayX, self.cellArrayY, self.cellArrayZ))[0]
        n2 = np.histogramdd(parcel[np.where(parcel[:, 6] == 2)][:, :3], bins=(self.cellArrayX, self.cellArrayY, self.cellArrayZ))[0]
        self.rhoM += n1*self.Ar_m
        self.rhoM += n2*self.Al_m
        self.dsmcrhoN += n
        for i in range(parcel.shape[0]):
            idx = np.floor((parcel[i][0]/self.celllength)).astype(int)
            idy = np.floor((parcel[i][1]/self.celllength)).astype(int)
            idz = np.floor((parcel[i][2]/self.celllength)).astype(int)
            self.linearKE[idx, idy, idz] += 0.5*self.constProp(parcel[i])[0]*np.linalg.norm(parcel[i, 3:6])**2
            self.internalE[idx, idy, idz] += self.constProp(parcel[i])[3]
            self.momentum[idx, idy, idz] += self.constProp(parcel[i])[0]*parcel[i, 3:6]

        self.rhoN *= self.nParticle/self.meshVolumes
        self.rhoM *= self.nParticle/self.meshVolumes
        self.linearKE *= self.nParticle/self.meshVolumes
        self.internalE *= self.nParticle/self.meshVolumes
        self.iDof *= self.nParticle/self.meshVolumes
        self.momentum *= self.nParticle/self.meshVolumes

    def collide(self, p, q):
        pmass = self.constProp(p)[0]
        qmass = self.constProp(q)[0]
        Ucm = (pmass*p[3:6] + qmass*q[3:6])/(pmass + qmass)
        cR = np.linalg.norm(p[3:6] - q[3:6])
        cosTheta = 2*np.random.rand() - 1
        sinTheta = np.sqrt(1- cosTheta*cosTheta)
        phi = 2*np.pi*np.random.rand()
        postCollisionRelU = cR*np.array([cosTheta, sinTheta*np.cos(phi), sinTheta*np.sin(phi)]).T
        Up = Ucm + postCollisionRelU*qmass/(qmass + pmass)
        Uq = Ucm + postCollisionRelU*pmass/(qmass + pmass)
        return Up, Uq

    def collisions(self, parcel, cell): # cell[x, y, z, centerPos]
        nC = parcel.shape[0]
        subcells = [[],[],[],[],[],[],[],[]]
        whichsubcell = []
        for i in range(nC):
            relPos = parcel[i, :3] - cell[3]
            subcell = int(np.floor(relPos[0]+1) + 2*np.floor(relPos[1]+1) + 4*np.floor(relPos[2]+1))
            subcells[subcell].append(i)
            whichsubcell.append(subcell)
        selectedPairs = self.collisionSelectedRemainder[cell[0], cell[1], cell[2]] + \
            0.5*nC*(nC-1)*self.nParticle*self.sigmaTCRMax[cell[0], cell[1], cell[2]]*self.tstep/self.meshVolumes
        nCandidates = int(selectedPairs)
        self.collisionSelectedRemainder[cell[0], cell[1], cell[2]] = selectedPairs - nCandidates

        for c in range(nCandidates):
            candidateP = np.random.randint(nC)
            subcellPs = subcells[whichsubcell[candidateP]]
            nSC = len(subcellPs)
            candidateQ = -1
            if (nSC > 1):
                candidateQ = np.random.randint(nSC)
                while(candidateP == candidateQ):
                    candidateQ = np.random.randint(nSC)
            else:
                candidateQ = np.random.randint(nC)
                while(candidateP == candidateQ):
                    candidateQ = np.random.randint(nC)              
            sigmaTcR = self.sigmaTcR(parcel[candidateP], parcel[candidateQ])
            if (sigmaTcR > self.sigmaTCRMax[cell[0], cell[1], cell[2]]):
                self.sigmaTCRMax[cell[0], cell[1], cell[2]] = sigmaTcR
            if ((sigmaTcR/self.sigmaTCRMax[cell[0], cell[1], cell[2]]) > np.random.rand()):
                parcel[candidateP, 3:6], parcel[candidateQ, 3:6] = self.collide(parcel[candidateP], parcel[candidateQ])

    def boundary(self, posvel, i, j, k):
        # print(pos)
        pos_cp = np.asarray(posvel)

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
            i_cp = i_cp[~indices]
            j_cp = j_cp[~indices]
            k_cp = k_cp[~indices]

        return pos_cp

    def move(self, posvel):

        i = np.floor((posvel[:, 0]/self.celllength)).astype(int)
        j = np.floor((posvel[:, 1]/self.celllength)).astype(int)
        k = np.floor((posvel[:, 2]/self.celllength)).astype(int)

        posvelBoundary = self.boundary(posvel, i, j, k)
        n = np.histogramdd(posvelBoundary[:, :3], bins=(self.cellArrayX, self.cellArrayY, self.cellArrayZ))[0]
        
        posvelAcc = np.zeros_like(posvelBoundary)
        posvelAcc[:, :3] = posvelBoundary[:, 3:6] * self.tstep + posvelBoundary[:, :3]
        posvelAcc[:, 3:6] = posvelBoundary[:, 3:6]

        return posvelAcc, posvelBoundary, n

    def inflow():
        return 0

    def evolve(self):
        self.resetFields()
        self.inflow()
        self.move()



    def runE(self, parcel, time):

        tmax = time
        t = 0
        with tqdm(total=100, desc='running', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
            i = 0
            while t < tmax:
                self.evolve()
                t += self.tstep

                if int(t/tmax*100) > i:
                    Time.sleep(0.01)
                    pbar.update(1)
                    i += 1

                self.log.info('runStep:{}, timeStep:{}'\
                        .format(i, t))
        return parcel
    

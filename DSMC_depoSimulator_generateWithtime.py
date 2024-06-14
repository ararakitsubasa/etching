import numpy as np
# import cupy as cp
from scipy.spatial import KDTree
import time as Time
from tqdm import tqdm, trange
import logging
from Collision import transport

class depo(transport):
    def __init__(self, mirror, pressure_pa, temperature, chamberSize, DXsec,
                 param, TS, N, sub_xy, film, n, cellSize, celllength, kdtreeN, 
                 tstep, thickness,substrateTop, logname):
        super().__init__(tstep, pressure_pa, temperature, cellSize, celllength, chamberSize, DXsec)
        self.symmetry = mirror
        self.depoThick = thickness
        self.param = param # n beta
        self.TS = TS
        self.kdtreeN = kdtreeN
        self.cellSizeX = cellSize[0]
        self.cellSizeY = cellSize[1]
        self.cellSizeZ = cellSize[2]
        self.celllength = celllength
        self.timeStep = tstep
        self.sub_x = sub_xy[0]
        self.sub_y = sub_xy[1]
        self.substrate = film
        self.n = n
        self.N = N
        self.T = 300
        self.Cm = (2*1.380649e-23*self.T/(27*1.66e-27) )**0.5 # (2kT/m)**0.5 27 for the Al

        self.substrateTop = substrateTop
        self.indepoThick = substrateTop
        self.surface_depo_mirror = np.zeros((self.cellSizeX+20, self.cellSizeY+20, self.cellSizeZ))

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



    def max_velocity_u(self, random1, random2):
        return self.Cm*np.sqrt(-np.log(random1))*(np.cos(2*np.pi*random2))**self.n

    def max_velocity_w(self, random1, random2):
        return self.Cm*np.sqrt(-np.log(random1))*(np.sin(2*np.pi*random2))**self.n

    def max_velocity_v(self, random3):
        return -self.Cm*np.sqrt(-np.log(random3))

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

    def depo_film(self, film, pos, vel, i, j, k, weights_arr, depoStep):

        try:
            indice_inject = np.array(film[i, j, k] > 5)
        except IndexError:
            print('get i out:{}'.format(i.max()))
            print(i.max())
            print('get j out:{}'.format(j.max()))
            print(j.max())
            print('get k out:{}'.format(k.max()))
            print(k.max())

        pos_1 = pos[indice_inject]
        # print(pos_1.shape[0])

        surface_depo = np.logical_and(film >= 0, film < 1) # depo
        self.surface_depo_mirror[10:10+self.cellSizeX, 10:10+self.cellSizeY, :] = surface_depo
        self.surface_depo_mirror[:10, 10:10+self.cellSizeY, :] = surface_depo[-10:, :, :]
        self.surface_depo_mirror[-10:, 10:10+self.cellSizeY, :] = surface_depo[:10, :, :]
        self.surface_depo_mirror[10:10+self.cellSizeX, :10, :] = surface_depo[:, -10:, :]
        self.surface_depo_mirror[10:10+self.cellSizeX:, -10:, :] = surface_depo[:, :10, :]
        self.surface_depo_mirror[:10, :10, :] = surface_depo[-10:, -10:, :]
        self.surface_depo_mirror[:10, -10:, :] = surface_depo[-10:, :10, :]
        self.surface_depo_mirror[-10:, :10, :] = surface_depo[:10, -10:, :]
        self.surface_depo_mirror[-10:, -10:, :] = surface_depo[:10, :10, :]

        surface_tree = KDTree(np.argwhere(self.surface_depo_mirror == True)*self.celllength)

        pos_1[:, 0] += 10*self.celllength
        pos_1[:, 1] += 10*self.celllength

        dd, ii = surface_tree.query(pos_1, k=self.kdtreeN, workers=10)

        surface_indice = np.argwhere(self.surface_depo_mirror == True)

        ddsum = np.sum(dd, axis=1)

        # kdi order
        for kdi in range(self.kdtreeN):
            i1 = surface_indice[ii][:,kdi,0] #[particle, order, xyz]
            j1 = surface_indice[ii][:,kdi,1]
            k1 = surface_indice[ii][:,kdi,2]
            i1 -= 10
            j1 -= 10
            indiceXMax = i1 >= self.cellSizeX
            indiceXMin = i1 < 0
            i1[indiceXMax] -= self.cellSizeX
            i1[indiceXMin] += self.cellSizeX

            indiceYMax = j1 >= self.cellSizeY
            indiceYMin = j1 < 0
            j1[indiceYMax] -= self.cellSizeY
            j1[indiceYMin] += self.cellSizeY

            # deposit the particle injected into the film
            film[i1,j1,k1] += weights_arr[indice_inject]*dd[:,kdi]/ddsum

        # delete the particle injected into the film
        if np.any(indice_inject):
            pos = pos[~indice_inject]
            vel = vel[~indice_inject]
            i = i[~indice_inject]
            j = j[~indice_inject]
            k = k[~indice_inject]
            weights_arr = weights_arr[~indice_inject]

        film_indepo_indice = np.logical_or(film == 10, film == 20)
        film_indepo = film[~film_indepo_indice]
        film_max = film_indepo.max()
        surface_film = np.logical_and(film >= 1, film < 2)
        film[surface_film] = 20

        return film, pos, vel, weights_arr, pos_1.shape[0], film_max

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

    def runDepo(self, p0, v0, time, film, weights_arr, depoStep, stepSize):

        tmax = time
        tstep = self.timeStep
        t = 0
        # p1 = p0
        # v1 = v0
        inputCount = int(p0.shape[0]/(tmax/tstep))
        film_1 = self.substrate
        # weights_arr_1 = weights_arr
        cell = self.celllength
        collList = np.array([])
        elist = np.array([[0, 0, 0]])

        p1 = p0[inputCount*int(t/tstep):inputCount*(int(t/tstep)+1)]
        v1 = v0[inputCount*int(t/tstep):inputCount*(int(t/tstep)+1)]
        weights_arr_1 = weights_arr[inputCount*int(t/tstep):inputCount*(int(t/tstep)+1)]
        with tqdm(total=100, desc='running', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
            i = 0
            while t < tmax:
                p2v2 = self.getAcc_depo(p1, v1, cell, tstep, film_1, weights_arr_1, depoStep)
                p2 = p2v2[1][0]
                if p2.shape[0] == 0:
                    print('p20')
                    break
                v2 = p2v2[1][1]
                p1 = p2v2[0][0]
                v1 = p2v2[0][1]
                film_1 = p2v2[2]
                if np.any(film_1[:, :, self.depoThick]) != 0:
                    print('depo finish')
                    break
                if np.any(film_1[:, :, self.indepoThick + stepSize]) != 0:
                    self.indepoThick = filmThickness
                    print('depo finish at: {}'.format(self.indepoThick))
                    break
                weights_arr_1 = p2v2[3]
                depo_count = p2v2[4]
                film_max = p2v2[5]
                delx = np.linalg.norm(p1 - p2, axis=1)
                vMag = np.linalg.norm(v1, axis=1)
                vMax = vMag.max()
                KE = 0.5*self.Al_m*vMag**2/self.q
                prob = self.collProb(self.ng_pa, KE, delx)
                collList, elist, v2 = self.collision(prob, collList, elist, KE, vMag, p2, v2)
                t += tstep
                p1 = p2
                v1 = v2

                p1 = np.vstack((p1, p0[inputCount*int(t/tstep):inputCount*(int(t/tstep)+1)]))
                v1 = np.vstack((v1, v0[inputCount*int(t/tstep):inputCount*(int(t/tstep)+1)]))
                weights_arr_1 = np.concatenate((weights_arr_1, weights_arr[inputCount*int(t/tstep):inputCount*(int(t/tstep)+1)]), axis=0)
                if int(t/tmax*100) > i:
                    Time.sleep(0.01)
                    pbar.update(1)
                    i += 1
                vzMax = np.abs(v1[:,2]).max()

                for thick in range(film.shape[2]):
                    if np.sum(film_1[:, :, thick]) == 0:
                        filmThickness = thick
                        break
                # if vMax*tstep < 0.1 and i > 2:
                # if vzMax*tstep < 0.3*self.celllength:                    
                #     tstep *= 2
                # elif vzMax*tstep > 1*self.celllength:
                #     tstep /= 2

                self.log.info('runStep:{}, timeStep:{}, depo_count:{}, vMaxMove:{:.3f}, vzMax:{:.3f}, filmMax:{:.3f}, thickness:{},  input_count:{}'\
                              .format(i, tstep, depo_count, vMax*tstep/self.celllength, vzMax*tstep/self.celllength, film_max, filmThickness, p1.shape[0]))
        # del self.log, self.fh
        self.substrate = film_1
        return film_1, collList, elist, filmThickness
    
    def stepRundepo(self, step, randomSeed, tmax, velosityDist, weights):

        for i in range(step):
            np.random.seed(randomSeed+i)
            position_matrix = np.array([np.random.rand(self.N)*self.cellSizeX, np.random.rand(self.N)*self.cellSizeY, np.random.uniform(0, 10, self.N)+ self.cellSizeZ - 10]).T
            position_matrix *= self.celllength
            result =  self.runDepo(position_matrix, velosityDist, tmax, self.substrate, weights, depoStep=i+1)
        # del self.log, self.fh
        return result
    
    def ThicknessDepo(self, step, seed, tmax, velosity_matrix, weight):
        weights = np.ones(velosity_matrix.shape[0])*weight
        for tk in range(50):
            depoFilm = self.stepRundepo(step, seed+tk, tmax, velosity_matrix, weights)
            if np.any(depoFilm[0][:, :, self.depoThick]) != 0:
                break
        del self.log, self.fh
        return depoFilm
    

    def run_afterCollision(self, step, seed, tmax, velosity_matrix, weight):
        weights = np.ones(velosity_matrix.shape[0])*weight
        depoFilm = self.stepRundepo(step, seed, tmax, velosity_matrix, weights)
        del self.log, self.fh
        return depoFilm
    
    def runDepoition(self, step, seed, tmax, N, weight, xyP):
        weights = np.ones(N)*weight
        Random1 = np.random.rand(N)
        Random2 = np.random.rand(N)
        Random3 = np.random.rand(N)
        velosity_matrix = np.array([self.max_velocity_u(Random1, Random2)*xyP, \
                                    self.max_velocity_w(Random1, Random2)*xyP, \
                                        self.max_velocity_v(Random3)]).T
        weights = np.ones(velosity_matrix.shape[0])*weight
        for tk in range(50):
            depoFilm = self.stepRundepo(step, seed+tk, tmax, velosity_matrix, weights)
            if np.any(depoFilm[0][:, :, self.depoThick]) != 0:
                break
        del self.log, self.fh
        return depoFilm

    def depo_position_increase(self, stepSize, randomSeed, tmax, N, weight):
        np.random.seed(randomSeed)
        for i in range(9):
            weights = np.ones(N)*weight
            Random1 = np.random.rand(N)
            Random2 = np.random.rand(N)
            Random3 = np.random.rand(N)
            velosity_matrix = np.array([self.max_velocity_u(Random1, Random2), \
                                        self.max_velocity_w(Random1, Random2), \
                                            self.max_velocity_v(Random3)]).T
            weights = np.ones(velosity_matrix.shape[0])*weight
            position_matrix = np.array([np.random.rand(self.N)*self.cellSizeX, \
                                        np.random.rand(self.N)*self.cellSizeY, \
                                        np.random.uniform(0, self.cellSizeZ-self.indepoThick - stepSize*(i+1), self.N)+ self.indepoThick + stepSize]).T
            position_matrix *= self.celllength
            result =  self.runDepo(position_matrix, velosity_matrix, tmax, self.substrate, weights, depoStep=i+1, stepSize=stepSize)
                
        del self.log, self.fh
        return result
    
    
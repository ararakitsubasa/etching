import numpy as np
# import cupy as cp
from scipy.spatial import KDTree
import time as Time
from tqdm import tqdm, trange
import logging
from Collision import transport
from surface_normalize import surface_normal

class etching(transport, surface_normal):
    def __init__(self, pressure_pa, temperature, chamberSize, DXsec, #transport
                 center_with_direction, range3D, InOrOut, yield_hist, #surface_normal
                 param, TS, N, sub_xy, film, n, cellSize, celllength, kdtreeN, tstep, logname):
        super().__init__(tstep, pressure_pa, temperature, cellSize, celllength, chamberSize, DXsec)
        surface_normal.__init__(self, center_with_direction, range3D, InOrOut,celllength, yield_hist)
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

        indiceXMax = i_cp >= cellSize_x_cp
        indiceXMin = i_cp < 0
        if np.any(indiceXMax):
            i_cp[indiceXMax] -= cellSize_x_cp 
            pos_cp[indiceXMax,0] -= self.celllength*self.cellSizeX
        if np.any(indiceXMin):
            i_cp[indiceXMin] += cellSize_x_cp
            pos_cp[indiceXMin,0] += self.celllength*self.cellSizeX

        indiceYMax = j_cp >= cellSize_y_cp
        indiceYMin = j_cp < 0
        if np.any(indiceYMax):
            j_cp[indiceYMax] -= cellSize_y_cp
            pos_cp[indiceYMax,1] -= self.celllength*self.cellSizeY
        if np.any(indiceYMin):
            j_cp[indiceYMin] += cellSize_y_cp
            pos_cp[indiceYMin,1] += self.celllength*self.cellSizeY

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

    def etching_film(self, film, pos, vel, i, j, k, weights_arr, depoStep, planes):

        try:
            # indice_inject = np.array(film[i, j, k] > 5)np.logical_and(film < 0, film > -90)
            indice_inject = np.logical_and(film[i, j, k] < 0, film[i, j, k] > -90)
            # indice_inject = np.array(film[i, j, k] < 0) #etching
        except IndexError:
            print('get i out:{}'.format(i.max()))
            print(i.max())
            print('get j out:{}'.format(j.max()))
            print(j.max())
            print('get k out:{}'.format(k.max()))
            print(k.max())

        pos_1 = pos[indice_inject]
        vel_1 = vel[indice_inject]
        # print(pos_1.shape[0])
        get_theta = self.get_inject_theta(planes, pos_1, vel_1)
        # get_theta = -get_theta + np.pi
        # cut_theta_low = np.where(get_theta >= 0, get_theta, np.abs(get_theta))
        # surface_true = np.where(cut_theta_low < np.pi/2, cut_theta_low, -cut_theta_low + np.pi)
        # print(surface_true.shape)
        # etch_yield = self.get_yield(surface_true)
        etch_yield = self.get_yield(get_theta)
        # print(etch_yield)
        # print(etch_yield.shape)
        # surface_depo = np.logical_and(film >= 0, film < 1) # depo
        surface_depo = np.logical_and(film < 0, film > -90) #etching
        # surface_depo = np.logical_and(film > 0, film < 2000) #etching
        surface_tree = KDTree(np.argwhere(surface_depo == True)*self.celllength)

        dd, ii = surface_tree.query(pos_1, k=self.kdtreeN, workers=1)

        surface_indice = np.argwhere(surface_depo == True)

        ddsum = np.sum(dd, axis=1)

        # kdi order
        for kdi in range(self.kdtreeN):
            i1 = surface_indice[ii][:,kdi,0] #[particle, order, xyz]
            j1 = surface_indice[ii][:,kdi,1]
            k1 = surface_indice[ii][:,kdi,2]
            # print(weights_arr[indice_inject].shape)
            # deposit the particle injected into the film
            film[i1,j1,k1] -= weights_arr[indice_inject]*etch_yield*dd[:,kdi]/ddsum

        # delete the particle injected into the film
        if np.any(indice_inject):
            pos = pos[~indice_inject]
            vel = vel[~indice_inject]
            i = i[~indice_inject]
            j = j[~indice_inject]
            k = k[~indice_inject]
            weights_arr = weights_arr[~indice_inject]

        film_indepo_indice = np.logical_or(film == -10, film == 100)
        film_indepo_indice |= np.array(film == -50)
        film_indepo = film[~film_indepo_indice]
        if film_indepo.shape[0] != 0:
            film_max = film_indepo.min()
        else:
            film_max = 0
        # surface_film = np.logical_and(film >= 1, film < 2) #depo
        # film[surface_film] = 20
        surface_film = np.logical_and(film > -12, film < -11)
        # film[surface_film] = int(100*depoStep)
        film[surface_film] = 0

        return film, pos, vel, weights_arr, pos_1.shape[0], film_max, np.sum(surface_film), etch_yield

    def getAcc_depo(self, pos, vel, boxsize, tStep, film, weights_arr, depoStep, planes):
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
        film_depo, pos_cp, Nvel_cp, weights_arr_depo, depo_count, film_max, surface_true, etch_yield =\
              self.etching_film(film, pos_cp, Nvel_cp, i, j, k, weights_arr, depoStep, planes)

        Npos2_cp = Nvel_cp * tStep_cp + pos_cp

        return np.array([pos_cp, Nvel_cp]), np.array([Npos2_cp, Nvel_cp]), film_depo, weights_arr_depo, depo_count, film_max, surface_true, etch_yield

    def runEtch(self, p0, v0, time, film, weights_arr, depoStep):

        tmax = time
        tstep = self.timeStep
        t = 0
        p1 = p0
        v1 = v0
        film_1 = self.substrate
        weights_arr_1 = weights_arr

        cell = self.celllength
        planes = self.get_pointcloud(film)
        count_etching = 0
        collList = np.array([])
        elist = np.array([[0, 0, 0]])
        with tqdm(total=100, desc='running', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
            i = 0
            while t < tmax:
                p2v2 = self.getAcc_depo(p1, v1, cell, tstep, film_1, weights_arr_1, depoStep, planes)

                surface_true = p2v2[6]
                count_etching += surface_true
                if count_etching >= 200:
                    count_etching = 0
                    planes = self.get_pointcloud(film_1)
                if surface_true <= 10 and i > 40:
                    break

                p2 = p2v2[1][0]
                if p2.shape[0] == 0:
                    break
                v2 = p2v2[1][1]
                p1 = p2v2[0][0]
                v1 = p2v2[0][1]
                film_1 = p2v2[2]
                weights_arr_1 = p2v2[3]
                depo_count = p2v2[4]
                film_min = p2v2[5]
                etch_yield = p2v2[7]
                if etch_yield.shape[0] != 0:
                    etch_yield_large = np.sum(etch_yield >= 0.5)
                    etch_yield_max = etch_yield_large/etch_yield.shape[0]
                else:
                    etch_yield_max = 0
                    etch_yield_min = 0
                delx = np.linalg.norm(p1 - p2, axis=1)
                vMag = np.linalg.norm(v1, axis=1)
                vMax = vMag.max()
                KE = 0.5*self.Al_m*vMag**2/self.q
                prob = self.collProb(self.ng_pa, KE, delx)
                collList, elist, v2 = self.collision(prob, collList, elist, KE, vMag, p2, v2)
                t += tstep
                p1 = p2
                v1 = v2
                if int(t/tmax*100) > i:
                    Time.sleep(0.01)
                    pbar.update(1)
                    i += 1
                vzMax = np.abs(v1[:,2]).max()
                # if vMax*tstep < 0.1 and i > 2:
                if vzMax*tstep < 0.3*self.celllength:                    
                    tstep *= 2
                elif vzMax*tstep > 1*self.celllength:
                    tstep /= 2

                self.log.info('runStep:{}, timeStep:{}, depo_count:{}, vMaxMove:{:.3f}, vzMax:{:.3f}, filmMax:{:.3f}, etching:{}, etch_yield_max:{:2.2%}, etch_yield_min:{:2.2%}'\
                              .format(i, tstep, depo_count, vMax*tstep/self.celllength, vzMax*tstep/self.celllength, film_min, surface_true, etch_yield_max, 1 - etch_yield_max))
        # del self.log, self.fh

        return film, collList, elist
    
    def stepRunEtch(self, step, randomSeed, tmax, velosityDist, weights):

        for i in range(step):
            np.random.seed(randomSeed+i)
            position_matrix = np.array([np.random.rand(self.N)*self.cellSizeX, np.random.rand(self.N)*self.cellSizeY, np.random.uniform(0, 10, self.N)+ self.cellSizeZ - 10]).T
            position_matrix *= self.celllength
            result =  self.runEtch(position_matrix, velosityDist, tmax, self.substrate, weights, depoStep=i+1)
        del self.log, self.fh
        return result
    
    def run_afterCollision(self, step, seed, tmax, velosity_matrix, weight):
        weights = np.ones(velosity_matrix.shape[0])*weight
        depoFilm = self.stepRunEtch(step, seed, tmax, velosity_matrix, weights)
        return depoFilm
    

    def runEtching(self, step, seed, tmax, N, weight):
        # filmMac = self.target_substrate(Ero_dist_x, Ero_dist_y, self.sub_x, self.sub_y)
        # velosity_matrix = self.velocity_dist(Ero_dist_x, filmMac)
        # N = int(631394)
        weights = np.ones(N)*weight
        Random1 = np.random.rand(N)
        Random2 = np.random.rand(N)
        Random3 = np.random.rand(N)
        velosity_matrix = np.array([self.max_velocity_u(Random1, Random2), self.max_velocity_w(Random1, Random2), self.max_velocity_v(Random3)]).T
        depoFilm = self.stepRunEtch(step, seed, tmax, velosity_matrix, weights)

        return depoFilm

    def runEtching2(self, step, seed, tmax, N, weight):
        # filmMac = self.target_substrate(Ero_dist_x, Ero_dist_y, self.sub_x, self.sub_y)
        # velosity_matrix = self.velocity_dist(Ero_dist_x, filmMac)
        # N = int(631394)
        weights = np.ones(N)*weight
        Random1 = np.random.rand(N)
        Random2 = np.random.rand(N)
        Random3 = np.random.rand(N)
        velosity_matrix = np.array([np.zeros(N), np.zeros(N), self.max_velocity_v(Random3)]).T
        depoFilm = self.stepRunEtch(step, seed, tmax, velosity_matrix, weights)

        return depoFilm
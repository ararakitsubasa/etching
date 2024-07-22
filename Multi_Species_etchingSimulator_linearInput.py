import numpy as np
# import cupy as cp
from scipy.spatial import KDTree
import time as Time
from tqdm import tqdm, trange
import logging
from Collision import transport
from surface_normalize import surface_normal

class etching(transport, surface_normal):
    def __init__(self, mirror, pressure_pa, temperature, chamberSize, DXsec, #transport
                 center_with_direction, range3D, InOrOut, yield_hist, #surface_normal
                 param, TS, N, sub_xy, film, n, cellSize, celllength, kdtreeN,
                 tstep, thickness, substrateTop, posGeneratorType, logname):
        super().__init__(tstep, pressure_pa, temperature, cellSize, celllength, chamberSize, DXsec)
        surface_normal.__init__(self, center_with_direction, range3D, InOrOut,celllength, yield_hist)
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

        self.posGeneratorType = posGeneratorType
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

    # particle data struction np.array([posX, posY, posZ, velX, velY, velZ, i, j, k, typeID])
    def boundary(self, parcel):

        if self.symmetry == True:
            indiceXMax = parcel[:, 6] >= self.cellSizeX
            indiceXMin = parcel[:, 6] < 0

            # 使用布尔索引进行调整
            parcel[indiceXMax, 6] -= self.cellSizeX
            parcel[indiceXMax, 0] -= self.celllength * self.cellSizeX

            parcel[indiceXMin, 6] += self.cellSizeX
            parcel[indiceXMin, 0] += self.celllength * self.cellSizeX

            # 检查并调整 j_cp 和对应的 pos_cp
            indiceYMax = parcel[:, 7] >= self.cellSizeY
            indiceYMin = parcel[:, 7] < 0

            # 使用布尔索引进行调整
            parcel[indiceYMax, 7] -= self.cellSizeY
            parcel[indiceYMax, 1] -= self.celllength * self.cellSizeY

            parcel[indiceYMin, 7] += self.cellSizeY
            parcel[indiceYMin, 1] += self.celllength * self.cellSizeY
        
        indices = np.logical_or(parcel[:, 6] >= self.cellSizeX, parcel[:, 6] < 0)
        indices |= np.logical_or(parcel[:, 7] >= self.cellSizeY, parcel[:, 7] < 0)
        indices |= np.logical_or(parcel[:, 8] >= self.cellSizeZ, parcel[:, 8] < 0)

        if np.any(indices):
            parcel = parcel[~indices]
        return parcel

    def reaction(self, parcel, theta):
        reactionWeight = np.zeros(parcel.shape[0])
        react1 = parcel[:, 9] == 1
        react2 = parcel[:, 9] == 2
        reactive_yield1 = self.get_yield1(theta[react1])  
        reactive_yield2 = self.get_yield2(theta[react2])
        reactionWeight[react1] = reactive_yield1
        reactionWeight[react2] = reactive_yield2
        return reactionWeight, parcel



    def etching_film(self, film, parcel, planes):

        i = parcel[:, 6]
        j = parcel[:, 7]
        k = parcel[:, 8]

        indice_inject = np.logical_and(film[i, j, k] < 0, film[i, j, k] > -90)

        pos_1 = parcel[indice_inject, :3]
        vel_1 = parcel[indice_inject, 3:6]
        # print(pos_1.shape[0])
        get_theta = self.get_inject_theta(planes, pos_1, vel_1)
        # etch_yield = self.get_yield(get_theta)

        surface_depo = np.logical_and(film < 0, film > -90) #etching
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

        reactionWeight, parcel = self.reaction(parcel[indice_inject], get_theta)

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
            # film[i1,j1,k1] -= weights_arr[indice_inject]*etch_yield*dd[:,kdi]/ddsum
            film[i1,j1,k1] -= reactionWeight*dd[:,kdi]/ddsum


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

    def runEtch(self, v0, time, film, weights_arr, depoStep, emptyZ):

        tmax = time
        tstep = self.timeStep
        t = 0
        inputCount = int(v0.shape[0]/(tmax/tstep))
        film_1 = self.substrate
        cell = self.celllength
        planes = self.get_pointcloud(film)
        count_etching = 0
        collList = np.array([])
        elist = np.array([[0, 0, 0]])

        filmThickness = self.substrateTop

        if self.posGeneratorType == 'full':
            self.log.info('using posGenerator_full')
            posGenerator = self.posGenerator_full
        elif self.posGeneratorType == 'top':
            self.log.info('using posGenerator_top')
            posGenerator = self.posGenerator_top
        else:
            self.log.info('using posGenerator')
            posGenerator = self.posGenerator 

        p1 = posGenerator(inputCount, filmThickness, emptyZ)
        v1 = v0[inputCount*int(t/tstep):inputCount*(int(t/tstep)+1)]
        weights_arr_1 = weights_arr[inputCount*int(t/tstep):inputCount*(int(t/tstep)+1)]

        with tqdm(total=100, desc='running', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
            i = 0
            while t < tmax:
                p2v2 = self.getAcc_depo(p1, v1, cell, tstep, film_1, weights_arr_1, depoStep, planes)

                surface_true = p2v2[6]
                count_etching += surface_true
                if count_etching >= 200:
                    count_etching = 0
                    planes = self.get_pointcloud(film_1)
                # if surface_true <= 10 and i > 40:
                #     break

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
                pGenerate = posGenerator(inputCount, filmThickness, emptyZ)
                p1 = np.vstack((p1, pGenerate))
                v1 = np.vstack((v1, v0[inputCount*int(t/tstep):inputCount*(int(t/tstep)+1)]))
                weights_arr_1 = np.concatenate((weights_arr_1, weights_arr[inputCount*int(t/tstep):inputCount*(int(t/tstep)+1)]), axis=0)
                if int(t/tmax*100) > i:
                    Time.sleep(0.01)
                    pbar.update(1)
                    i += 1
                vzMax = np.abs(v1[:,2]).max()
                # if vMax*tstep < 0.1 and i > 2:
                # if vzMax*tstep < 0.3*self.celllength:                    
                #     tstep *= 2
                # elif vzMax*tstep > 1*self.celllength:
                #     tstep /= 2

                self.log.info('runStep:{}, timeStep:{}, depo_count:{}, vMaxMove:{:.3f}, vzMax:{:.3f}, filmMax:{:.3f}, etching:{}, etch_yield_max:{:2.2%}, etch_yield_min:{:2.2%}, input_count:{}'\
                              .format(i, tstep, depo_count, vMax*tstep/self.celllength, vzMax*tstep/self.celllength, film_min, surface_true, etch_yield_max, 1 - etch_yield_max, p1.shape[0]))
        # del self.log, self.fh

        return film, collList, elist
    
    def posGenerator(self, IN, thickness, emptyZ):
        position_matrix = np.array([np.random.rand(IN)*self.cellSizeX, \
                                    np.random.rand(IN)*self.cellSizeY, \
                                    np.random.uniform(0, emptyZ, IN)+ thickness + emptyZ]).T
        position_matrix *= self.celllength
        return position_matrix
    
    def posGenerator_full(self, IN, thickness, emptyZ):
        position_matrix = np.array([np.random.rand(IN)*self.cellSizeX, \
                                    np.random.rand(IN)*self.cellSizeY, \
                                    np.random.uniform(0, self.cellSizeZ-thickness-emptyZ, IN)+ thickness + emptyZ]).T
        position_matrix *= self.celllength
        return position_matrix

    def posGenerator_top(self, IN, thickness, emptyZ):
        position_matrix = np.array([np.random.rand(IN)*self.cellSizeX, \
                                    np.random.rand(IN)*self.cellSizeY, \
                                    np.random.uniform(0, emptyZ, IN) + self.cellSizeZ - emptyZ]).T
        position_matrix *= self.celllength
        return position_matrix
      
    def depo_position_increase(self, randomSeed, velosity_matrix, tmax, weight, Zgap):
        np.random.seed(randomSeed)
        weights = np.ones(velosity_matrix.shape[0])*weight
        result =  self.runEtch(velosity_matrix, tmax, self.substrate, weights, depoStep=1, emptyZ=Zgap)
        del self.log, self.fh
        return result
    

    def depo_position_increase_cosVel(self, randomSeed, N, tmax, weight, Zgap):
        np.random.seed(randomSeed)
        Random1 = np.random.rand(N)
        Random2 = np.random.rand(N)
        Random3 = np.random.rand(N)
        velosity_matrix = np.array([self.max_velocity_u(Random1, Random2), \
                                    self.max_velocity_w(Random1, Random2), \
                                        self.max_velocity_v(Random3)]).T
        weights = np.ones(velosity_matrix.shape[0])*weight
        result =  self.runEtch(velosity_matrix, tmax, self.substrate, weights, depoStep=1, emptyZ=Zgap)            
        del self.log, self.fh
        return result
    
    def rfunc_2(self, x): #Release factor function
        # print("-------rfunc------")
        # print(x)
        y = np.cos(x) ** self.n 
        return y

    def depo_position_increase_cosVel_NoMaxwell(self, randomSeed, N, tmax, weight, Zgap):
        np.random.seed(randomSeed)
        theta_bins_size = 100
        theta_bins = np.linspace(-np.pi/2, np.pi/2, theta_bins_size)
        theta_hist_x = theta_bins + np.pi/((theta_bins_size-1)*2)
        theta_hist_x = theta_hist_x[:-1]

        theta_hist_y = self.rfunc_2(theta_hist_x)
        theta_hist_y *= 1e5
        theta_sample = np.array([])

        for i in range(theta_bins.shape[0] - 1):
            theta_sample = np.concatenate(( theta_sample, np.random.uniform(theta_bins[i], theta_bins[i+1], int(theta_hist_y[i]))))
        np.random.shuffle(theta_sample)
        theta_sample = theta_sample[:N]

        self.log.info('theta_sample.shape:{}'.format(theta_sample.shape[0]))
        phi = np.random.rand(theta_sample.shape[0])*2*np.pi
        vel_x = np.cos(phi)*np.sin(theta_sample)*1e3
        vel_y = np.sin(phi)*np.sin(theta_sample)*1e3
        vel_z = np.cos(theta_sample)*1e3
        velosity_matrix = np.array([vel_x, vel_y, -vel_z]).T
        weights = np.ones(velosity_matrix.shape[0])*weight
        result =  self.runEtch(velosity_matrix, tmax, self.substrate, weights, depoStep=1, emptyZ=Zgap)

        del self.log, self.fh
        return result
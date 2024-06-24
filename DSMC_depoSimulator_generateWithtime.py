import numpy as np
# import cupy as cp
from scipy.spatial import KDTree
import time as Time
from tqdm import tqdm, trange
import logging
from Collision import transport

class depo(transport):
    def __init__(self, mirror, collision, pressure_pa, temperature, chamberSize, DXsec,
                 param, TS, N, sub_xy, film, n, cellSize, celllength, kdtreeN, 
                 tstep, thickness,substrateTop, posGeneratorType, logname):
        super().__init__(tstep, pressure_pa, temperature, cellSize, celllength, chamberSize, DXsec)
        self.symmetry = mirror
        self.collider = collision
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
        self.filmDensity = np.copy(film)
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
        self.filmDensity[surface_film] = film[surface_film]
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

    def runDepo(self, v0, time, film, weights_arr, depoStep, emptyZ):

        tmax = time
        tstep = self.timeStep
        t = 0
        # p1 = p0
        # v1 = v0
        vAllparticle = v0.shape[0]
        v0Max = np.average(v0[:,2])
        depoTot = 0
        inputCount = int(v0.shape[0]/(tmax/tstep))
        film_1 = self.substrate
        # weights_arr_1 = weights_arr
        cell = self.celllength
        collList = np.array([])
        elist = np.array([[0, 0, 0]])
        filmThickness = self.substrateTop

        if self.posGeneratorType == 'full':
            self.log.info('using posGenerator_full')
            posGenerator = self.posGenerator_full
        elif self.posGeneratorType == 'top':
            self.log.info('using posGenerator_top')
            posGenerator = self.posGenerator_top
        elif self.posGeneratorType == 'vacuum':
            self.log.info('using posGenerator_vaccum')
            posGenerator = self.posGenerator_vacuum
        else:
            self.log.info('using posGenerator')
            posGenerator = self.posGenerator 

        p1 = posGenerator(inputCount, filmThickness, emptyZ)
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

                weights_arr_1 = p2v2[3]
                depo_count = p2v2[4]
                depoTot += depo_count
                film_max = p2v2[5]
                vMag = np.linalg.norm(v1, axis=1)
                vMax = vMag.max()
                if self.collider == True:
                    delx = np.linalg.norm(p1 - p2, axis=1)
                    KE = 0.5*self.Al_m*vMag**2/self.q
                    prob = self.collProb(self.ng_pa, KE, delx)
                    self.collList, self.elist, v2 = self.collision(prob, self.collList, self.elist, KE, vMag, p2, v2)
                t += tstep
                p1 = p2
                v1 = v2

                for thick in range(film.shape[2]):
                    if np.sum(film_1[:, :, thick]) == 0:
                        filmThickness = thick
                        break

                pGenerate = posGenerator(inputCount, filmThickness, emptyZ)
                p1 = np.vstack((p1, pGenerate))
                v1 = np.vstack((v1, v0[inputCount*int(t/tstep):inputCount*(int(t/tstep)+1)]))
                weights_arr_1 = np.concatenate((weights_arr_1, weights_arr[inputCount*int(t/tstep):inputCount*(int(t/tstep)+1)]), axis=0)
                if int(t/tmax*100) > i:
                    Time.sleep(0.01)
                    pbar.update(1)
                    i += 1
                vzMax = np.abs(v1[:,2]).max()

                vInhigh = np.sum(np.array(v2[:,2] > v0Max ))/v2.shape[0]
                vInlow = np.sum(np.array(v2[:,2] < v0Max ))/v2.shape[0]

                self.log.info('runStep:{}, timeStep:{}, inDepo:{}, DepoTot:{}, vMaxMove:{:.3f}, vzMax:{:.3f}, filmMax:{:.3f}, thickness:{},  ParticleIn:{}, ParticleAll:{:2.2%}, inhigh:{:2.2%}, inlow:{:2.2%}'\
                        .format(i, tstep, depo_count, depoTot, vMax*tstep/self.celllength, vzMax*tstep/self.celllength, film_max, filmThickness, p1.shape[0], (p1.shape[0] + depoTot)/vAllparticle, vInhigh, vInlow))
        # del self.log, self.fh
        self.substrate = film_1
        return film_1, collList, elist, filmThickness, self.filmDensity
    
    
    # def posGenerator(self, IN, thickness, emptyZ):
    #     position_matrix = np.array([np.random.rand(IN)*self.cellSizeX, \
    #                                 np.random.rand(IN)*self.cellSizeY, \
    #                                 np.random.uniform(0, self.cellSizeZ-thickness-emptyZ, IN)+ thickness + emptyZ]).T
    #     position_matrix *= self.celllength
    #     return position_matrix
    
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

    def posGenerator_vacuum(self, IN, thickness, emptyZ): # never use

        thick = 4
        # 计算每个位置从顶部开始第一次出现非零的索引
        # 因为我们从底部向上扫描，所以我们使用[::-1]反转
        film_reversed = self.substrate[:, :, ::-1]
        non_vacuum_top = np.argmax(film_reversed != 0, axis=2)

        # 计算在原始 film 中的实际索引
        non_vacuum_top = self.substrate.shape[2] - 1 - non_vacuum_top

        # 广播 non_vacuum_top 以匹配 film 的形状
        non_vacuum_top_broadcast = non_vacuum_top
        non_vacuum_top_broadcast += emptyZ

        position_matrix = np.array([np.random.rand(IN)*self.substrate.shape[0], \
                                    np.random.rand(IN)*self.substrate.shape[1], \
                                    np.random.uniform(0, thick, IN)]).T

        i = np.floor((position_matrix[:, 0])).astype(int)
        j = np.floor((position_matrix[:, 1])).astype(int)

        position_matrix[:,2] += non_vacuum_top_broadcast[i, j]

        position_matrix *= self.celllength
        return position_matrix

    def depo_position_increase(self, randomSeed, velosity_matrix, tmax, weight, Zgap):
        np.random.seed(randomSeed)
        energy = np.linalg.norm(velosity_matrix, axis=1)
        velosity_matrix[:,0] = np.divide(velosity_matrix[:,0], energy)
        velosity_matrix[:,1] = np.divide(velosity_matrix[:,1], energy)
        velosity_matrix[:,2] = np.divide(velosity_matrix[:,2], energy)
        for i in range(10):
            weights = np.ones(velosity_matrix.shape[0])*weight
            result =  self.runDepo(velosity_matrix, tmax, self.substrate, weights, depoStep=1, emptyZ=Zgap)
            if np.any(result[0][:, :, self.depoThick]) != 0:
                break  
        del self.log, self.fh
        return result
    

    def depo_position_increase_cosVel(self, randomSeed, N, tmax, weight, Zgap):
        np.random.seed(randomSeed)
        for i in range(9):
            Random1 = np.random.rand(N)
            Random2 = np.random.rand(N)
            Random3 = np.random.rand(N)
            velosity_matrix = np.array([self.max_velocity_u(Random1, Random2), \
                                        self.max_velocity_w(Random1, Random2), \
                                            self.max_velocity_v(Random3)]).T
            energy = np.linalg.norm(velosity_matrix, axis=1)
            velosity_matrix[:,0] = np.divide(velosity_matrix[:,0], energy)
            velosity_matrix[:,1] = np.divide(velosity_matrix[:,1], energy)
            velosity_matrix[:,2] = np.divide(velosity_matrix[:,2], energy)
            weights = np.ones(velosity_matrix.shape[0])*weight
            result =  self.runDepo(velosity_matrix, tmax, self.substrate, weights, depoStep=1, emptyZ=Zgap)
            if np.any(result[0][:, :, self.depoThick]) != 0:
                break             
        del self.log, self.fh
        return result
    
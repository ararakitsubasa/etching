import numpy as np
# import cupy as cp
from scipy.spatial import cKDTree
import time as Time
from tqdm import tqdm, trange
import logging
from Collision import transport
from surface_normalize_sf import surface_normal
from numba import jit, prange

#solid = film[i, j, k, 10][Si, SiF1, SiF2, SiF3, SiO SiO2, SiOF, SiOF2, SiO2F, SiO2F2]
#react_t g[Cu] s  [1,         2]
#react_t g[Cu] s  [Cu,       Si]

# react_table = np.array([[[0.700, 0, 1], [0.300, 0, 1]],
#                         [[0.800, -1, 0], [0.075, 0, -1]]])

#solid = film[i, j, k, 10][Si, SiF1, SiF2, SiF3, SiO SiO2, SiOF, SiOF2, SiO2F, SiO2F2]
#react_t g[F, O, ion] s  [1,          2,           3,          4,       5 ,   6,    7,    8,   9,  10]
#react_t g[F, O, ion] s  [Si,       SiF1,       SiF2,       SiF3,      SiO, SiO2, SiOF, SiOF2, SiO2F,SiO2F2]

# react_table3 = np.array([[[0.9, 2], [0.9, 3], [0.9, 4], [0.9, -4], [0.5, 7], [0.0, 0], [0.5, 8], [0.0, 0], [0.6, 10], [0.0, 0]],
#                         [[0.5, 5], [0.0, 0], [0.0, 0], [0.0, 0], [0.5, 6], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
#                         [[0.27, -1], [0.27, -2], [0.27, -3], [0.27, -4], [0.27, -5], [0.27, -6], [0.27, -7], [0.27, -8], [0.27, -9], [0.27, -10]]])

# react_table3 = np.array([[[0.1, 2], [0.1, 3], [0.1, 4], [0.1, -4], [0.5, 7], [0.0, 0], [0.5, 8], [0.0, 0], [0.6, 10], [0.0, 0]],
#                         [[0.5, 5], [0.0, 0], [0.0, 0], [0.0, 0], [0.5, 6], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
#                         [[0.27, -1], [0.27, -2], [0.27, -3], [0.27, -4], [0.27, -5], [0.27, -6], [0.27, -7], [0.27, -8], [0.27, -9], [0.27, -10]]])


# # print(react_table3.shape)

# react_table = np.zeros((3, 10, 11))

# for i in range(react_table3.shape[0]):
#     for j in range(react_table3.shape[1]):
#         for k in range(react_table3.shape[2]):
#             react_table[i, j, 0] = react_table3[i, j, 0]
#             react_table[i, j, j+1] = -1
#             react_chem =  int(np.abs(react_table3[i, j, 1]))
#             if react_table3[i, j, 1] > 0:
#                 react_plus_min = 1
#             elif react_table3[i, j, 1] < 0:
#                 react_plus_min = -1
#             elif react_table3[i, j, 1] == 0:
#                 react_plus_min = 0
#             react_table[i, j, react_chem] = react_plus_min


react_table = np.array([[[0.3, -1, 0, 0], [0.0, 0,  0, 0], [0.0, 0, 0, 0]],
                        [[0.8, -1, 1, 0], [0.0, 0,  0, 0], [0.0, 0, 0, 0]],
                        [[1.0,  0, 0, 0], [1.0, 0, -2, 0], [1.0, 0, 0, 0]]])


# react_table = np.array([[[0.3, 1, 0, 0], [1.0, 0,  0, 0], [0.0, 0, 0, 0]],
#                         [[0.8, -1, 0, 0], [0.0, 0,  0, 0], [0.0, 0, 0, 0]],
#                         [[1.0,  0, 0, 0], [1.0, 0, -2, 0], [1.0, 0, 0, 0]]])

# react_table[0, 3, 4] = -2
# etching act on film, depo need output
@jit(nopython=True)
def reaction_yield(parcel, film, film_vaccum, theta, update_film):

    num_parcels = parcel.shape[0]
    num_reactions = react_table.shape[1]
    choice = np.random.rand(parcel.shape[0], react_table.shape[1])
    reactList = np.ones(parcel.shape[0])*-1
    for i in range(num_parcels):
        for j in range(num_reactions):
            if film[i, j] <= 0:
                choice[i, j] = 1

    depo_parcel = np.zeros(parcel.shape[0])
    for i in range(parcel.shape[0]):
        acceptList = np.zeros(react_table.shape[1], dtype=np.bool_)
        for j in range(film.shape[1]):
            react_rate = react_table[int(parcel[i, -1]), j, 0]
            if react_rate > choice[i, j]:
                acceptList[j] = True
        react_choice_indices = np.where(acceptList)[0]
        if react_choice_indices.size > 0:
            react_choice = np.random.choice(react_choice_indices)
            reactList[i] = react_choice
            if np.sum(react_table[int(parcel[i, -1]), react_choice, 1:]) > 0:
                # print('deposition')
                depo_parcel[i] = 1
            if np.sum(react_table[int(parcel[i, -1]), react_choice, 1:]) < 0:
                depo_parcel[i] = -1
            if np.sum(react_table[int(parcel[i, -1]), react_choice, 1:]) == 0:
                depo_parcel[i] = -2
    for i in range(parcel.shape[0]):
        react_add = react_table[int(parcel[i, -1]), int(reactList[i]), 1:]
        if depo_parcel[i] == -1: # etching
            film[i, :] += react_add
            if np.all(film[i, :]) == 0:
                update_film[int(parcel[i, 6]), int(parcel[i, 7]), int(parcel[i, 8])] = True

        if depo_parcel[i] == 1: # depo
            if np.sum(react_add + film[i, :]) > 10:
                film_vaccum[i, :] += react_add
                update_film[int(parcel[i, 6]), int(parcel[i, 7]), int(parcel[i, 8])] = True  

            else:
                film[i, :] += react_add
                if np.sum(film[i, :]) == 10:
                        update_film[int(parcel[i, 6]), int(parcel[i, 7]), int(parcel[i, 8])] = True                

        if reactList[i] == -1:
            parcel[i,3:6] = SpecularReflect(parcel[i,3:6], theta[i])
            # print('reflection')
            # parcel[i,3:6] = DiffusionReflect(parcel[i,3:6], theta[i])

    return film, film_vaccum, parcel, update_film, reactList, depo_parcel

@jit(nopython=True)
def SpecularReflect(vel, normal):
    return vel - 2*vel@normal*normal

kB = 1.380649e-23
T = 100

@jit(nopython=True)
def DiffusionReflect(vel, normal):
    mass = 27*1.66e-27
    Ut = vel - vel@normal*normal
    tw1 = Ut/np.linalg.norm(Ut)
    tw2 = np.cross(tw1, normal)
    # U = np.sqrt(kB*T/particleMass[i])*(np.random.randn()*tw1 + np.random.randn()*tw2 - np.sqrt(-2*np.log((1-np.random.rand())))*normal)
    U = np.sqrt(kB*T/mass)*(np.random.randn()*tw1 + np.random.randn()*tw2 - np.sqrt(-2*np.log((1-np.random.rand())))*normal)
    UN = U / np.linalg.norm(U)
        # UN[i] = U
    return UN

@jit(nopython=True)
def reemission_multi(vel, normal):
    vels = np.zeros_like(vel)
    for i in range(vels.shape[0]):
        vels[i] = DiffusionReflect(vel[i], normal[i])
        # vels[i] = SpecularReflect(vel[i], normal[i])
    return vels

def removeFloat(film):  # fast scanZ
    
    # 获取当前平面的非零元素布尔索引
    current_plane = film != 0

    # 创建一个全是False的布尔数组来存储邻居的检查结果
    neighbors = np.zeros_like(film, dtype=bool)

    # 检查各个方向的邻居是否为零
    neighbors[1:, :, :] |= film[:-1, :, :] != 0  # 上面的邻居不为0
    neighbors[:-1, :, :] |= film[1:, :, :] != 0  # 下面的邻居不为0
    neighbors[:, 1:, :] |= film[:, :-1, :] != 0  # 左边的邻居不为0
    neighbors[:, :-1, :] |= film[:, 1:, :] != 0  # 右边的邻居不为0
    neighbors[:, :, 1:] |= film[:, :, :-1] != 0  # 前面的邻居不为0
    neighbors[:, :, :-1] |= film[:, :, 1:] != 0  # 后面的邻居不为0

    # 孤立单元格的条件是当前平面元素不为0且所有方向的邻居都为0
    condition = current_plane & ~neighbors

    # 将孤立的单元格设为0
    film[condition] = 0
    
    return film

@jit(nopython=True)
def boundaryNumba(parcel, cellSizeX, cellSizeY, cellSizeZ, celllength):
    # Adjust X dimension
    indiceXMax = parcel[:, 6] >= cellSizeX
    indiceXMin = parcel[:, 6] < 0

    parcel[indiceXMax, 6] -= cellSizeX
    parcel[indiceXMax, 0] -= celllength * cellSizeX

    parcel[indiceXMin, 6] += cellSizeX
    parcel[indiceXMin, 0] += celllength * cellSizeX

    # Adjust Y dimension
    indiceYMax = parcel[:, 7] >= cellSizeY
    indiceYMin = parcel[:, 7] < 0

    parcel[indiceYMax, 7] -= cellSizeY
    parcel[indiceYMax, 1] -= celllength * cellSizeY

    parcel[indiceYMin, 7] += cellSizeY
    parcel[indiceYMin, 1] += celllength * cellSizeY

    # Check if any particles are outside bounds in any direction
    indices = (parcel[:, 6] >= cellSizeX) | (parcel[:, 6] < 0) | \
              (parcel[:, 7] >= cellSizeY) | (parcel[:, 7] < 0) | \
              (parcel[:, 8] >= cellSizeZ) | (parcel[:, 8] < 0)

    # Remove particles outside the boundary
    return parcel[~indices]

@jit(nopython=True)
def update_parcel(parcel, celllength, tStep):
    # 预计算 1/celllength，避免重复计算
    inv_celllength = 1.0 / celllength

    # 更新位置：parcel[:, :3] 为位置，parcel[:, 3:6] 为速度
    parcel[:, :3] += parcel[:, 3:6] * tStep

    # 计算新的 ijk 值并将其直接赋值到 parcel 的第 6、7、8 列
    # ijk = np.rint((parcel[:, :3] * inv_celllength) + 0.5).astype(np.int32)
    ijk = np.rint(parcel[:, :3] * inv_celllength).astype(np.int32)
    parcel[:, 6:9] = ijk

    return parcel

class etching(surface_normal):
    def __init__(self, mirror, inputMethod, pressure_pa, temperature, chamberSize,depoThick, #transport
                 center_with_direction, range3D, InOrOut, yield_hist, #surface_normal
                 reaction_type, #reaction 
                 param, N, sub_xy, film, n, cellSize, celllength, kdtreeN,
                 tstep, substrateTop, posGeneratorType, logname):
        # super().__init__(tstep, pressure_pa, temperature, cellSize, celllength, chamberSize)
        surface_normal.__init__(self, center_with_direction, range3D, InOrOut,celllength, yield_hist)
        self.symmetry = mirror
        self.param = param # n beta
        self.kdtreeN = kdtreeN
        self.cellSizeX = cellSize[0]
        self.cellSizeY = cellSize[1]
        self.cellSizeZ = cellSize[2]
        self.celllength = celllength
        self.timeStep = tstep
        self.sub_x = sub_xy[0]
        self.sub_y = sub_xy[1]
        # self.substrate = film
        self.depoThick = depoThick
        self.inputMethod = inputMethod
        self.n = n
        self.N = N
        self.T = 300
        self.Cm = (2*1.380649e-23*self.T/(27*1.66e-27) )**0.5 # (2kT/m)**0.5 27 for the Al

        self.film = film

        self.reaction_type = reaction_type
        self.posGeneratorType = posGeneratorType
        self.substrateTop = substrateTop
        self.indepoThick = substrateTop
        self.surface_depo_mirror = np.zeros((self.cellSizeX+20, self.cellSizeY+20, self.cellSizeZ))
        self.surface_etching_mirror = np.zeros((self.cellSizeX+20, self.cellSizeY+20, self.cellSizeZ))
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
    def boundary(self):

        if self.symmetry == True:
            indiceXMax = self.parcel[:, 6] >= self.cellSizeX
            indiceXMin = self.parcel[:, 6] < 0

            # 使用布尔索引进行调整
            self.parcel[indiceXMax, 6] -= self.cellSizeX
            self.parcel[indiceXMax, 0] -= self.celllength * self.cellSizeX

            self.parcel[indiceXMin, 6] += self.cellSizeX
            self.parcel[indiceXMin, 0] += self.celllength * self.cellSizeX

            # 检查并调整 j_cp 和对应的 pos_cp
            indiceYMax = self.parcel[:, 7] >= self.cellSizeY
            indiceYMin = self.parcel[:, 7] < 0

            # 使用布尔索引进行调整
            self.parcel[indiceYMax, 7] -= self.cellSizeY
            self.parcel[indiceYMax, 1] -= self.celllength * self.cellSizeY

            self.parcel[indiceYMin, 7] += self.cellSizeY
            self.parcel[indiceYMin, 1] += self.celllength * self.cellSizeY
        
        indices = np.logical_or(self.parcel[:, 6] >= self.cellSizeX, self.parcel[:, 6] < 0)
        indices |= np.logical_or(self.parcel[:, 7] >= self.cellSizeY, self.parcel[:, 7] < 0)
        indices |= np.logical_or(self.parcel[:, 8] >= self.cellSizeZ, self.parcel[:, 8] < 0)

        if np.any(indices):
            self.parcel = self.parcel[~indices]

    def removeFloat(self):  # fast scanZ
        sumFilm = np.sum(self.film, axis=-1)
        # 获取当前平面的非零元素布尔索引
        current_plane = sumFilm != 0

        # 创建一个全是False的布尔数组来存储邻居的检查结果
        neighbors = np.zeros_like(sumFilm, dtype=bool)

        # 检查各个方向的邻居是否为零
        neighbors[1:, :, :] |= sumFilm[:-1, :, :] != 0  # 上面的邻居不为0
        neighbors[:-1, :, :] |= sumFilm[1:, :, :] != 0  # 下面的邻居不为0
        neighbors[:, 1:, :] |= sumFilm[:, :-1, :] != 0  # 左边的邻居不为0
        neighbors[:, :-1, :] |= sumFilm[:, 1:, :] != 0  # 右边的邻居不为0
        neighbors[:, :, 1:] |= sumFilm[:, :, :-1] != 0  # 前面的邻居不为0
        neighbors[:, :, :-1] |= sumFilm[:, :, 1:] != 0  # 后面的邻居不为0

        # 孤立单元格的条件是当前平面元素不为0且所有方向的邻居都为0
        condition = current_plane & ~neighbors

        # 将孤立的单元格设为0
        self.film[condition, :] = 0
        
        # return film

    def get_indices(self):
        # 直接将切片操作和数据类型转换合并
        to_depo = self.parcel[:, 9] == 1
        to_etch = self.parcel[:, 9] == 0

        return self.parcel[to_depo, 6].astype(int), self.parcel[to_depo, 7].astype(int), self.parcel[to_depo, 8].astype(int), \
               self.parcel[to_etch, 6].astype(int), self.parcel[to_etch, 7].astype(int), self.parcel[to_etch, 8].astype(int)
    
    def etching_film(self):

        i_depo, j_depo, k_depo, i_etch, j_etch, k_etch  = self.get_indices()

        # indice_inject_depo = np.array(self.sumFilm[i_depo, j_depo, k_depo] >= 10) # depo
        indice_inject = np.array(self.sumFilm[i_etch, j_etch, k_etch] > 0 ) # ethicng

        reactListAll = np.ones(indice_inject.shape[0])*-2

        pos_1 = self.parcel[indice_inject, :3]
        vel_1 = self.parcel[indice_inject, 3:6]
        ijk_1 = self.parcel[indice_inject, 6:9]

        if np.any(indice_inject):
            # self.planes = self.get_pointcloud(sumFilm)
            self.indice_inject = indice_inject
            get_plane, get_theta, get_plane_vaccum = self.get_inject_normal(self.planes, self.planes_vaccum, pos_1, vel_1)

            self.film[get_plane[:,0], get_plane[:,1],get_plane[:,2]],\
            self.film[get_plane_vaccum[:,0], get_plane_vaccum[:,1], get_plane_vaccum[:,2]], \
            self.parcel[indice_inject,:], self.update_film,\
            reactList, depo_parcel = \
            reaction_yield(self.parcel[indice_inject], \
                           self.film[get_plane[:,0], get_plane[:,1],get_plane[:,2]], \
                           self.film[get_plane_vaccum[:,0], get_plane_vaccum[:,1], get_plane_vaccum[:,2]], \
                           get_theta, self.update_film)
            if np.any(self.update_film):
                # self.planes = self.update_pointcloud(self.planes, self.film, self.update_film)
                self.sumFilm = np.sum(self.film, axis=-1)
                self.planes, self.planes_vaccum = self.get_pointcloud(self.sumFilm)
            # self.reactList_debug = reactList
            reactListAll[indice_inject] = reactList
            if np.any(reactListAll != -1):
                indice_inject[np.where(reactListAll == -1)] = False
                self.parcel = self.parcel[~indice_inject]

            return np.sum(depo_parcel == self.depo_count_type) #, film_max, np.sum(surface_film)
        else:
            return 0

    def getAcc_depo(self, tStep):

        # pos, vel, i, j, k, cellSize_x, cellSize_y, cellSize_z,
        # self.boundary()
        self.parcel = boundaryNumba(self.parcel, self.cellSizeX, self.cellSizeY, self.cellSizeZ, self.celllength)
        # self.removeFloat()
        # print(pos_cp)
        depo_count = self.etching_film()
        self.parcel = update_parcel(self.parcel, self.celllength, tStep)
        # Npos2_cp = Nvel_cp * tStep_cp + pos_cp
        # self.parcel[:, :3] += self.parcel[:, 3:6] * tStep 
        # i = np.floor((self.parcel[:, 0]/self.celllength) + 0.5).astype(int)
        # j = np.floor((self.parcel[:, 1]/self.celllength) + 0.5).astype(int)
        # k = np.floor((self.parcel[:, 2]/self.celllength) + 0.5).astype(int)
        # self.parcel[:, 6] = i
        # self.parcel[:, 7] = j
        # self.parcel[:, 8] = k

        return depo_count #, film_max, surface_true

    # particle data struction np.array([posX, posY, posZ, velX, velY, velZ, i, j, k, depo_or_etching,typeID])
    def Parcelgen(self, pos, vel, depo_or_etching, typeID):

        # i = np.floor((pos[:, 0]/self.celllength) + 0.5).astype(int)
        # j = np.floor((pos[:, 1]/self.celllength) + 0.5).astype(int)
        # k = np.floor((pos[:, 2]/self.celllength) + 0.5).astype(int)
        i = np.floor((pos[:, 0]/self.celllength)).astype(int)
        j = np.floor((pos[:, 1]/self.celllength)).astype(int)
        k = np.floor((pos[:, 2]/self.celllength)).astype(int)
        parcelIn = np.zeros((pos.shape[0], 11))
        parcelIn[:, :3] = pos
        parcelIn[:, 3:6] = vel
        parcelIn[:, 6] = i
        parcelIn[:, 7] = j
        parcelIn[:, 8] = k
        parcelIn[:, 9] = 0 # depo: 1 etch: 0
        # parcelIn[:, 9] = depo_or_etching # depo: 1 etch: 0
        parcelIn[:, 10] = typeID
        self.parcel = np.concatenate((self.parcel, parcelIn))


    def runEtch(self, inputCount,runningCount, max_react_count, emptyZ):

        self.parcel = np.zeros((1, 11))
        # tmax = time
        tstep = self.timeStep
        t = 0
        # inputCount = int(v0.shape[0]/(tmax/tstep))
        self.sumFilm = np.sum(self.film, axis=-1)
        self.planes, self.planes_vaccum = self.get_pointcloud(self.sumFilm)
        count_reaction = 0
        inputAll = 0
        filmThickness = self.substrateTop

        if self.posGeneratorType == 'full':
            self.log.info('using posGenerator_full')
            posGenerator = self.posGenerator_full
        elif self.posGeneratorType == 'top':
            self.log.info('using posGenerator_top')
            posGenerator = self.posGenerator_top
        elif self.posGeneratorType == 'benchmark':
            self.log.info('using posGenerator_benchmark')
            posGenerator = self.posGenerator_benchmark
        else:
            self.log.info('using posGenerator')
            posGenerator = self.posGenerator 

        velGenerator = self.velGenerator_input_normal 
        p1 = posGenerator(inputCount, filmThickness, emptyZ)
        vel_type = velGenerator(inputCount)
        v1 = vel_type[:, :3]
        typeIDIn = vel_type[:, -1]
        self.Parcelgen(p1, v1, 1, typeIDIn)
        self.parcel = self.parcel[1:, :]

        self.update_film = np.zeros_like(self.sumFilm, dtype=np.bool_)

        with tqdm(total=100, desc='running', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
            previous_percentage = 0
            while self.parcel.shape[0] > 500:
                depo_count = self.getAcc_depo(tstep)
                # print('parcel', self.parcel.shape)
                t += tstep
                count_reaction += depo_count
                current_percentage = int(count_reaction / max_react_count * 100)  # 当前百分比
                if self.parcel.shape[0] < runningCount and current_percentage < 100:
                    inputAll += inputCount
                    p1 = posGenerator(inputCount, filmThickness, emptyZ)
                    vel_type = velGenerator(inputCount)
                    v1 = vel_type[:, :3]
                    typeIDIn = vel_type[:, -1]
                    self.Parcelgen(p1, v1, 1, typeIDIn)

                # if self.inputMethod == 'bunch':
                #     p1 = posGenerator(inputCount, filmThickness, emptyZ)
                #     v1 = v0[inputCount*int(t/tstep):inputCount*(int(t/tstep)+1)]
                #     if v1.shape[0] != 0:
                #         typeIDIn = typeID[inputCount*int(t/tstep):inputCount*(int(t/tstep)+1)]
                #         self.Parcelgen(p1, v1, typeIDIn)

                # planes = self.get_pointcloud(np.sum(self.film, axis=-1))

                # if int(t/tmax*100) > i:
                #     Time.sleep(0.01)
                #     pbar.update(1)
                #     i += 1
                # current_percentage = int(count_reaction / max_react_count * 100)  # 当前百分比
                if current_percentage > previous_percentage and current_percentage < 100:
                    update_value = current_percentage - previous_percentage  # 计算进度差值
                    pbar.update(update_value)
                    previous_percentage = current_percentage  # 更新上一次的百分比

                if current_percentage > 80:
                    print('depo finish')
                    break
                for thick in range(self.film.shape[2]):
                    if np.sum(self.film[int(self.cellSizeX/2),int(self.cellSizeY/2), thick, :]) == 0:
                        filmThickness = thick
                        break

                self.log.info('runStep:{}, timeStep:{},inputAll:{},  depo_count:{}, count_reaction:{}, filmThickness:{},  input_count:{}'\
                              .format(previous_percentage,tstep,inputAll, depo_count, count_reaction, filmThickness, self.parcel.shape[0]))
        # del self.log, self.fh

        return self.film, self.planes
    
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

    def velGenerator_input_normal(self, IN):

        velosity_matrix = np.random.default_rng().choice(self.vel_type_shuffle, IN)

        return velosity_matrix     
    # def posGenerator_benchmark(self, IN, thickness, emptyZ):
    #     position_matrix = np.array([np.ones(IN)*self.cellSizeX/2, \
    #                                 np.ones(IN)*self.cellSizeY/2, \
    #                                 np.ones(IN)*self.cellSizeZ - emptyZ]).T
    #     position_matrix *= self.celllength
    #     return position_matrix
    
    def posGenerator_benchmark(self, IN, thickness, emptyZ):
        position_matrix = np.array([np.random.rand(IN)*20 + self.cellSizeX/2, \
                                    np.random.rand(IN)*20 + self.cellSizeY/2, \
                                    np.ones(IN)*self.cellSizeZ - emptyZ]).T
        position_matrix *= self.celllength
        return position_matrix
    
    def depo_position_increase(self, randomSeed, velosity_matrix, tmax, weight, Zgap):
        np.random.seed(randomSeed)
        weights = np.ones(velosity_matrix.shape[0])*weight
        result =  self.runEtch(velosity_matrix, tmax, self.film, weights, depoStep=1, emptyZ=Zgap)
        del self.log, self.fh
        return result
    
        # def runEtch(self, v0, typeID, time, emptyZ):
    def depo_position_increase_cosVel_normal(self, randomSeed, N, tmax, Zgap):
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

            typeID = np.ones(N)
            # def runEtch(self, v0, typeID, time, emptyZ):
            result =  self.runEtch(velosity_matrix, typeID, tmax, emptyZ=Zgap)
            if np.any(result[0][:, :, self.depoThick]) != 0:
                break             
        del self.log, self.fh
        return result
    

        # def runEtch(self, inputCount, max_react_count, emptyZ):
    def inputParticle(self, randomSeed, vel_type_shuffle, inputCount,runningCount, max_react_count,depo_count_type, Zgap):
        self.depo_count_type = depo_count_type
        np.random.seed(randomSeed)
        self.vel_type_shuffle = vel_type_shuffle
        result =  self.runEtch(inputCount,runningCount, max_react_count, emptyZ=Zgap)
        # if np.any(result[0][:, :, self.depoThick]) != 0:
        #     break             
        # del self.log, self.fh 
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
    

if __name__ == "__main__":

    # film = np.zeros((20, 100, 140, 2))

    # bottom = 100
    # height = 104

    # density = 10

    # center = 50

    # film[:, :45, bottom:height, 1] = density
    # film[:, 55:, bottom:height, 1] = density
    # # film[:, :, 0:bottom, :] = 0
    # film[:, :, 0:bottom, 0] = density # bottom

    maskUp = 2.09
    maskBottom = 2.6
    maskDeep = 2.808

    diameter = 260
    maskUp_sim = maskUp/maskBottom*diameter
    maskDeep_sim = maskDeep/maskBottom*diameter
    print(maskUp_sim)
    print(maskDeep_sim)

    film = np.zeros((300, 300, 400, 3))

    bottom = 100
    height = bottom + int(maskDeep_sim)

    density = 10

    sphere = np.ones((300, 300, 400), dtype=bool)

    # diameter = 120

    center = 150
    for k in range(int(diameter/2 - int(maskUp_sim/2))):
        # print(diameter/2 - k)
        radius = diameter/2 - k
        # print('deep', int(bottom + maskDeep_sim/(diameter/2 - maskUp_sim/2)*k))
        bottom_step = int(bottom + maskDeep_sim/(diameter/2 - maskUp_sim/2)*k)
        for i in range(sphere.shape[0]):
            for j in range(sphere.shape[1]):
                if np.abs(i-center)*np.abs(i-center) + np.abs(j-center)*np.abs(j-center) < radius*radius:
                    sphere[i, j, bottom_step:bottom_step+int(maskDeep_sim/(diameter/2 - maskUp_sim/2))] = 0

    film[sphere, 2] = density
    film[:, :, height:, :] = 0
    film[:, :, 0:bottom, 0] = density # bottom
    film[:, :, 0:bottom, 1] = 0 # bottom
    film[:, :, 0:bottom, 2] = 0 # bottom


    etchfilm = film

    T = 300
    Cm = (2*1.380649e-23*T/(27*1.66e-27) )**0.5 # (2kT/m)**0.5 27 for the Al

    def max_velocity_u( random1, random2):
        return Cm*np.sqrt(-np.log(random1))*(np.cos(2*np.pi*random2))

    def max_velocity_w( random1, random2):
        return Cm*np.sqrt(-np.log(random1))*(np.sin(2*np.pi*random2))

    def max_velocity_v( random3):
        return -Cm*np.sqrt(-np.log(random3))

    N = int(1e7)
    velosity_matrix = np.zeros((N, 3))

    Random1 = np.random.rand(N)
    Random2 = np.random.rand(N)
    Random3 = np.random.rand(N)
    velosity_matrix = np.array([max_velocity_u(Random1, Random2), \
                                max_velocity_w(Random1, Random2), \
                                    max_velocity_v(Random3)]).T

    energy = np.linalg.norm(velosity_matrix, axis=1)
    velosity_matrix[:,0] = np.divide(velosity_matrix[:,0], energy)
    velosity_matrix[:,1] = np.divide(velosity_matrix[:,1], energy)
    velosity_matrix[:,2] = np.divide(velosity_matrix[:,2], energy)

    typeID = np.zeros(N)


    vel_type_shuffle = np.zeros((N, 4))
    vel_type_shuffle[:, :3] = velosity_matrix
    vel_type_shuffle[:, -1] = typeID

    np.random.shuffle(vel_type_shuffle)

    logname = 'Multi_species_benchmark_0729'
    testEtch = etching(mirror=True,inputMethod='bunch', pressure_pa=0.001, temperature=300, chamberSize=etchfilm.shape,
                                            depoThick=120, center_with_direction=np.array([[etchfilm.shape[0]/2,center,75]]), 
                                            range3D=np.array([[0, etchfilm.shape[0], 0, etchfilm.shape[1], 0, etchfilm.shape[2]]]), InOrOut=[1], yield_hist=np.array([None]),
                                            reaction_type=False, param = [1.6, -0.7], N = 300000, 
                                            sub_xy=[0,0], film=etchfilm, n=1, cellSize=etchfilm.shape, 
                                            celllength=1e-5, kdtreeN=5, tstep=1e-5,
                                            substrateTop=40,posGeneratorType='top', logname=logname)


    step1 = testEtch.inputParticle(125,vel_type_shuffle,int(1e4),int(5e5),int(1e5),-1, 5)

    np.save('./bosch_data_1022_timeit/bosch_sf_step_test_Ar', etchfilm)
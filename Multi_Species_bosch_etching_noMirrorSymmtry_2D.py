import numpy as np
# import cupy as cp
from scipy.spatial import KDTree
import time as Time
from tqdm import tqdm, trange
import logging
# from Collision import transport
from surface_normalize_sf_2D import surface_normal
from numba import jit, prange
import torch
#solid = film[i, j, k, 10][Si, SiF1, SiF2, SiF3, SiO SiO2, SiOF, SiOF2, SiO2F, SiO2F2]
#react_t g[Cu] s  [1,         2]
#react_t g[Cu] s  [Cu,       Si]

# react_table = np.array([[[0.700, 0, 1], [0.300, 0, 1]],
#                         [[0.800, -1, 0], [0.075, 0, -1]]])

#solid = film[i, j, k, 10][Si, SiF1, SiF2, SiF3, C4F8]
#react_t g[F, c4f8, ion] s  [1,          2,           3,          4,       5 ]
#react_t g[F, c4f8, ion] s  [Si,       SiF1,       SiF2,       SiF3,     C4F8]

# react_table3 = np.array([[[0.5, 2], [0.5, 3], [0.5, 4], [0.5, -4], [0.0, 0]],
#                          [[0.5, 5], [0.0, 0], [0.0, 0], [0.0,  0], [0.5, 5]],
#                          [[0.27, -1], [0.27, -2], [0.27, -3], [0.27, -4], [0.27, -5]]])


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


#solid = film[i, j, k, 2][Si, C4F8, mask]
#react_t g[F, c4f8, ion] s  [1,    2 , 3]
#react_t g[F, c4f8, ion] s  [Si, C4F8, mask]

# react_table = np.array([[[0.200, -1, 0], [0.0  , 0,  0]],
#                         [[0.800,  -1, 1], [0.0, 0,  0]],
#                         [[0.1 ,  -1, 0], [0.9  , 0, -1]]])

react_table = np.array([[[0.200, -1, 0, 0], [0.0, 0, 0, 0], [0.0, 0, 0, 0]],
                        [[0.800, -1, 1, 0], [0.0, 0, 0, 0], [0.0, 0, 0, 0]],
                        [[0.950, -1, 0, 0], [0.95, 0,-1, 0], [0.0, 0, 0, 0]]])

# react_table[0, 3, 4] = -2
# etching act on film, depo need output

# @jit(nopython=True, parallel=True)
# def reaction_yield(parcel, film, theta):
#     num_parcels = parcel.shape[0]
#     num_reactions = react_table.shape[1]

#     choice = np.random.rand(num_parcels, num_reactions)
#     reactList = np.ones(num_parcels) * -1
#     depo_parcel = np.zeros(num_parcels)

#     # 矢量化筛选：设置film <= 0的元素的choice为1
#     choice[film <= 0] = 1

#     # 并行处理parcel的外循环
#     for i in prange(num_parcels):  # 使用prange来并行化外循环
#         react_rates = react_table[int(parcel[i, -1]), :, 0]

#         # 矢量化处理 acceptList，获取可接受的反应
#         acceptList = react_rates > choice[i, :]

#         # 获取可以反应的下标
#         react_choice_indices = np.where(acceptList)[0]

#         # 如果有可以反应的反应，随机选择一个
#         if react_choice_indices.size > 0:
#             react_choice = np.random.choice(react_choice_indices)
#             reactList[i] = react_choice

#             # 判断是否有沉积或蚀刻反应
#             if np.sum(react_table[int(parcel[i, -1]), react_choice, 1:]) > 0:
#                 depo_parcel[i] = 1  # 沉积
#             else:
#                 depo_parcel[i] = -1  # 蚀刻

#     # 矢量化处理 film 和 parcel
#     deposit_mask = depo_parcel == -1
#     film[deposit_mask] += react_table[parcel[deposit_mask, -1].astype(int), reactList[deposit_mask].astype(int), 1:]

#     # 处理反射
#     reflect_mask = reactList == -1
#     parcel[reflect_mask, 3:6] = SpecularReflect(parcel[reflect_mask, 3:6], theta[reflect_mask])

#     return film, parcel, reactList, depo_parcel

# @jit(nopython=True)
# def SpecularReflect(vel, normal):
#     # 矢量化反射计算
#     return vel - 2 * np.einsum('ij,ij->i', vel, normal)[:, np.newaxis] * normal



@jit(nopython=True, parallel=True)
def reaction_yield(parcel, film, theta):
    # print('react parcel', parcel.shape)
    # print('react film', film.shape)
    # print('react theta', theta.shape)
    num_parcels = parcel.shape[0]
    num_reactions = react_table.shape[1]
    choice = np.random.rand(parcel.shape[0], react_table.shape[1])
    reactList = np.ones(parcel.shape[0])*-1
    for i in range(num_parcels):
        for j in range(num_reactions):
            if film[i, j] <= 0:
                choice[i, j] = 1
    depo_parcel = np.zeros(parcel.shape[0])
    for i in prange(parcel.shape[0]):
        acceptList = np.zeros(react_table.shape[1], dtype=np.bool_)
        for j in prange(film.shape[1]):
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
            if np.sum(react_table[int(parcel[i, -1]), react_choice, 1:]) <= 0:
                depo_parcel[i] = -1
    for i in prange(parcel.shape[0]):
        if depo_parcel[i] == -1:
            film[i, :] += 1 * react_table[int(parcel[i, -1]), int(reactList[i]), 1:]
            # print('chemistry',film[i])
        if reactList[i] == -1:
            parcel[i,2:4] = SpecularReflect(parcel[i,2:4], theta[i])
            # print('reflection')
            # parcel[i,3:6] = reemission(parcel[i,3:6], theta[i])

    return film, parcel, reactList, depo_parcel

@jit(nopython=True)
def SpecularReflect(vel, normal):
    return vel - 2*vel@normal*normal

kB = 1.380649e-23
T = 100

@jit(nopython=True)
def reemission(vel, normal):
    mass = 27*1.66e-27
    Ut = vel - vel@normal*normal
    tw1 = Ut/np.linalg.norm(Ut)
    tw2 = np.cross(tw1, normal)
    # U = np.sqrt(kB*T/particleMass[i])*(np.random.randn()*tw1 + np.random.randn()*tw2 - np.sqrt(-2*np.log((1-np.random.rand())))*normal)
    U = np.sqrt(kB*T/mass)*(np.random.randn()*tw1 + np.random.randn()*tw2 - np.sqrt(-2*np.log((1-np.random.rand())))*normal)
    UN = U / np.linalg.norm(U)
        # UN[i] = U
    return UN

def removeFloat(film):  # fast scanZ
    
    # 获取当前平面的非零元素布尔索引
    current_plane = film != 0

    # 创建一个全是False的布尔数组来存储邻居的检查结果
    neighbors = np.zeros_like(film, dtype=bool)

    # 检查各个方向的邻居是否为零
    neighbors[1:, :] |= film[:-1, :] != 0  # 上面的邻居不为0
    neighbors[:-1, :] |= film[1:, :] != 0  # 下面的邻居不为0
    neighbors[:, 1:] |= film[:, :-1] != 0  # 左边的邻居不为0
    neighbors[:, :-1] |= film[:, 1:] != 0  # 右边的邻居不为0

    # 孤立单元格的条件是当前平面元素不为0且所有方向的邻居都为0
    condition = current_plane & ~neighbors

    # 将孤立的单元格设为0
    film[condition] = 0
    
    return film

class etching(surface_normal):
    def __init__(self,inputMethod, depo_or_etching, etchingPoint,depoPoint,density, 
                 center_with_direction, range3D, InOrOut, yield_hist, #surface_normal
                 reaction_type, #reaction 
                 param, sub_xy, film, n, cellSize, celllength, kdtreeN,
                 tstep, substrateTop, posGeneratorType, logname):
        # super().__init__(tstep, pressure_pa, temperature, cellSize, celllength, chamberSize)
        surface_normal.__init__(self, center_with_direction, range3D, InOrOut,celllength, yield_hist)
        self.param = param # n beta
        self.kdtreeN = kdtreeN
        self.cellSizeX = cellSize[0]
        self.cellSizeY = cellSize[1]
        self.cellSizeZ = cellSize[1]
        self.celllength = celllength
        self.timeStep = tstep
        self.sub_x = sub_xy[0]
        self.sub_y = sub_xy[1]
        # self.substrate = film
        self.depo_or_etching = depo_or_etching
        self.depoPoint = depoPoint
        self.etchingPoint = etchingPoint
        self.density = density
        self.inputMethod = inputMethod
        self.n = n
        self.T = 300
        self.Cm = (2*1.380649e-23*self.T/(27*1.66e-27) )**0.5 # (2kT/m)**0.5 27 for the Al

        self.film = film

        self.reaction_type = reaction_type
        self.posGeneratorType = posGeneratorType
        self.substrateTop = substrateTop
        self.indepoThick = substrateTop
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
        return self.Cm*np.sqrt(-np.log(random1))*(np.cos(2*np.pi*random2))

    def max_velocity_w(self, random1, random2):
        return self.Cm*np.sqrt(-np.log(random1))*(np.sin(2*np.pi*random2))

    def max_velocity_v(self, random3):
        return -self.Cm*np.sqrt(-np.log(random3))

    # particle data struction np.array([posX, posY, posZ, velX, velY, velZ, i, j, k, typeID])
    # particle data struction np.array([posX, posY, velX, velY, i, j, typeID])
    def boundary(self):

        # if self.symmetry == True:
        indiceXMax = self.parcel[:, 4] >= self.cellSizeX
        indiceXMin = self.parcel[:, 4] < 0

        # 使用布尔索引进行调整
        self.parcel[indiceXMax, 4] -= self.cellSizeX
        self.parcel[indiceXMax, 0] -= self.celllength * self.cellSizeX

        self.parcel[indiceXMin, 4] += self.cellSizeX
        self.parcel[indiceXMin, 0] += self.celllength * self.cellSizeX

        indices = np.logical_or(self.parcel[:, 4] >= self.cellSizeX, self.parcel[:, 4] < 0)
        indices |= np.logical_or(self.parcel[:, 5] >= self.cellSizeZ, self.parcel[:, 5] < 0)

        if np.any(indices):
            self.parcel = self.parcel[~indices]

    def removeFloat(self):  # fast scanZ
        sumFilm = np.sum(self.film, axis=-1)
        # 获取当前平面的非零元素布尔索引
        current_plane = sumFilm != 0

        # 创建一个全是False的布尔数组来存储邻居的检查结果
        neighbors = np.zeros_like(sumFilm, dtype=bool)

        # 检查各个方向的邻居是否为零
        neighbors[1:, :] |= sumFilm[:-1, :] != 0  # 上面的邻居不为0
        neighbors[:-1, :] |= sumFilm[1:, :] != 0  # 下面的邻居不为0
        neighbors[:, 1:] |= sumFilm[:, :-1] != 0  # 左边的邻居不为0
        neighbors[:, :-1] |= sumFilm[:, 1:] != 0  # 右边的邻居不为0


        # 孤立单元格的条件是当前平面元素不为0且所有方向的邻居都为0
        condition = current_plane & ~neighbors

        # 将孤立的单元格设为0
        self.film[condition, :] = 0
        
        # return film

    def etching_film(self, planes):

        i = self.parcel[:, 4].astype(int)
        j = self.parcel[:, 5].astype(int)
        sumFilm = np.sum(self.film, axis=-1)
        indice_inject = np.array(sumFilm[i, j] >= 1) 
        # if indice_inject.size != 0:
        pos_1 = self.parcel[indice_inject, :2]
        vel_1 = self.parcel[indice_inject, 2:4]
        ij_1 = self.parcel[indice_inject, 4:6]
        # print('pos1 shape',pos_1.shape[0])
        # print('ijk_1',ijk_1.shape[0])
        # print('parcel_ijk', self.film[ijk_1[0], ijk_1[1],ijk_1[2]].shape)
        if pos_1.size != 0:
            get_plane, get_theta = self.get_inject_normal(planes, pos_1, vel_1)

            # print('get plane', get_plane.shape)
            # print('i[indice_inject]',i[indice_inject].shape)

            # print('get plane', get_plane[0])
            # print('i[indice_inject]',i[indice_inject][0])
            # print('j[indice_inject]',j[indice_inject][0])
            # print('k[indice_inject]',k[indice_inject][0])
            # etch_yield = self.get_yield(get_theta)
            # print('parcel_ijk', self.film[i[indice_inject], j[indice_inject],k[indice_inject]].shape)
            # print('get theta', get_theta.shape)
            # print('parcel to react', self.parcel[indice_inject].shape)
            self.film[get_plane[:,0], get_plane[:,1]],self.parcel[indice_inject,:], reactList, depo_parcel = \
                reaction_yield(self.parcel[indice_inject], self.film[get_plane[:,0], get_plane[:,1]], get_theta)
            # print('after react')
        # if np.any(depo_parcel == -1):
        #     self.parcel = self.parcel[~indice_inject[np.where(depo_parcel == -1)[0]]]
        # reflect_choice = np.where(reactList==-1)[0]
        # reflect_parcel = SpecularReflect(vel_1[reflect_choice], get_theta[reflect_choice])

        # define depo area 
            surface_depo = np.logical_and(sumFilm >= 0, sumFilm < 1) 

            # # mirror
            # self.surface_depo_mirror[10:10+self.cellSizeX, 10:10+self.cellSizeY, :] = surface_depo
            # self.surface_depo_mirror[:10, 10:10+self.cellSizeY, :] = surface_depo[-10:, :, :]
            # self.surface_depo_mirror[-10:, 10:10+self.cellSizeY, :] = surface_depo[:10, :, :]
            # self.surface_depo_mirror[10:10+self.cellSizeX, :10, :] = surface_depo[:, -10:, :]
            # self.surface_depo_mirror[10:10+self.cellSizeX:, -10:, :] = surface_depo[:, :10, :]
            # self.surface_depo_mirror[:10, :10, :] = surface_depo[-10:, -10:, :]
            # self.surface_depo_mirror[:10, -10:, :] = surface_depo[-10:, :10, :]
            # self.surface_depo_mirror[-10:, :10, :] = surface_depo[:10, -10:, :]
            # self.surface_depo_mirror[-10:, -10:, :] = surface_depo[:10, :10, :]
            # # mirror end

            # surface_tree = KDTree(np.argwhere(self.surface_depo_mirror == True)*self.celllength)
            surface_tree = KDTree(np.argwhere(surface_depo == True)*self.celllength)

            to_depo = np.where(depo_parcel > 0)[0]
            # pos_1[:, 0] += 10*self.celllength
            # pos_1[:, 1] += 10*self.celllength

            # depo for depo_parcel > 0
            dd, ii = surface_tree.query(pos_1[to_depo], k=self.kdtreeN, workers=10)

            # surface_indice = np.argwhere(self.surface_depo_mirror == True)
            surface_indice = np.argwhere(surface_depo == True)

            ddsum = np.sum(dd, axis=1)

            # kdi order
            for kdi in range(self.kdtreeN):
                i1 = surface_indice[ii][:,kdi,0] #[particle, order, xyz]
                j1 = surface_indice[ii][:,kdi,1]
                # i1 -= 10
                # j1 -= 10
                # indiceXMax = i1 >= self.cellSizeX
                # indiceXMin = i1 < 0
                # i1[indiceXMax] -= self.cellSizeX
                # i1[indiceXMin] += self.cellSizeX

                # indiceYMax = j1 >= self.cellSizeY
                # indiceYMin = j1 < 0
                # j1[indiceYMax] -= self.cellSizeY
                # j1[indiceYMin] += self.cellSizeY

        # delete the particle injected into the film
                self.film[i1,j1,1] += 0.2*dd[:,kdi]/ddsum

            if self.depo_or_etching == 'depo':
                surface_film = np.logical_and(self.film[:, :, 1] >= 1, self.film[:, :, 1] < 2)
                self.film[surface_film, 1] = self.density

            if np.any(np.where(reactList != -1)[0]):
                indice_inject[np.where(reactList == -1)[0]] = False
                self.parcel = self.parcel[~indice_inject]
        # delete the particle injected into the film
        # if np.any(indice_inject):
        #     self.parcel = self.parcel[~indice_inject]

            return pos_1.shape[0] #, film_max, np.sum(surface_film)
        else:
            return 0

    def getAcc_depo(self, tStep, planes):

        # pos, vel, i, j, k, cellSize_x, cellSize_y, cellSize_z,
        self.boundary()
        self.removeFloat()
        # print(pos_cp)
        depo_count = self.etching_film(planes)

        self.parcel[:, :2] += self.parcel[:, 2:4] * tStep 
        i = np.floor((self.parcel[:, 0]/self.celllength) + 0.5).astype(int)
        j = np.floor((self.parcel[:, 1]/self.celllength) + 0.5).astype(int)
        # k = np.floor((self.parcel[:, 2]/self.celllength) + 0.5).astype(int)
        self.parcel[:, 4] = i
        self.parcel[:, 5] = j
        # self.parcel[:, 8] = k

        # # 预计算 1/self.celllength，避免重复计算
        # inv_celllength = 1.0 / self.celllength

        # # 更新位置
        # self.parcel[:, :3] += self.parcel[:, 3:6] * tStep

        # # 使用 np.rint() 进行取整，然后整体转换为整数类型，减少 .astype() 调用
        # ijk = np.rint((self.parcel[:, :3] * inv_celllength) + 0.5).astype(int)

        # # 一次性赋值给 parcel 的第 6、7、8 列
        # self.parcel[:, 6:9] = ijk

        # cuda
        # parcel_tensor = torch.tensor(self.parcel, device='cuda')  # 将数据转换为PyTorch张量
        # celllength_tensor = torch.tensor(self.celllength, device='cuda')

        # parcel_tensor[:, :2] += parcel_tensor[:, 2:4] * tStep

        # parcel_div_celllength = (parcel_tensor[:, :2] / celllength_tensor) + 0.5
        # ij = torch.floor(parcel_div_celllength).to(torch.int32)

        # parcel_tensor[:, 4:6] = ij
        # self.parcel = parcel_tensor.cpu().numpy()

        return depo_count #, film_max, surface_true

    # particle data struction np.array([posX, posY, posZ, velX, velY, velZ, i, j, k, typeID])
    def Parcelgen(self, pos, vel, typeID):

        i = np.floor((pos[:, 0]/self.celllength) + 0.5).astype(int)
        j = np.floor((pos[:, 1]/self.celllength) + 0.5).astype(int)

        parcelIn = np.zeros((pos.shape[0], 7))
        parcelIn[:, :2] = pos
        parcelIn[:, 2:4] = vel
        parcelIn[:, 4] = i
        parcelIn[:, 5] = j

        parcelIn[:, 6] = typeID
        self.parcel = np.concatenate((self.parcel, parcelIn))


    def runEtch(self, velGeneratorType, typeID, inputCount,max_react_count, time, emptyZ):

        self.log.info('inputType:{}'.format(typeID))
        self.parcel = np.zeros((1, 7))
        tmax = time

        tstep = self.timeStep
        t = 0
        # inputCount = int(v0.shape[0]/(tmax/tstep))

        planes = self.get_pointcloud(np.sum(self.film, axis=-1))
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

        if velGeneratorType == 'maxwell':
            self.log.info('using velGenerator_maxwell')
            velGenerator = self.velGenerator_maxwell_normal
        elif velGeneratorType == 'updown':
            self.log.info('using velGenerator_updown')
            velGenerator = self.velGenerator_updown_normal

        p1 = posGenerator(inputCount, filmThickness, emptyZ)
        v1 = velGenerator(inputCount)
        typeIDIn = np.zeros(inputCount)
        typeIDIn[:] = typeID
        self.Parcelgen(p1, v1, typeIDIn)
        self.parcel = self.parcel[1:, :]

        with tqdm(total=100, desc='running', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
            i = 0
            # while inputAll < max_react_count:
            while t < tmax:
                depo_count = self.getAcc_depo(tstep, planes)
                # print('parcel', self.parcel.shape)
                count_reaction += depo_count
                inputAll += inputCount
                # if count_reaction > self.max_react_count:
                #     break
                t += tstep
                vzMax = np.max(self.parcel[:,3])
                vzMin = np.min(self.parcel[:,3])
                if self.inputMethod == 'bunch':
                    p1 = posGenerator(inputCount, filmThickness, emptyZ)
                    v1 = velGenerator(inputCount)
                    typeIDIn = np.zeros(inputCount)
                    typeIDIn[:] = typeID
                    self.Parcelgen(p1, v1, typeIDIn)

                planes = self.get_pointcloud(np.sum(self.film, axis=-1))

                if int(t/tmax*100) > i:
                    Time.sleep(0.01)
                    pbar.update(1)
                    self.log.info('runStep:{}, timeStep:{}, depo_count_step:{}, count_reaction_all:{},inputAll:{},vzMax:{:.3f},vzMax:{:.3f}, filmThickness:{},  input_count:{}'\
                                .format(i, tstep, depo_count, count_reaction, inputAll,  vzMax, vzMin,  filmThickness, self.parcel.shape[0]))
                    i += 1
                
                for thick in range(self.film.shape[1]):
                    if np.sum(self.film[int(self.cellSizeX/2), thick, :]) == 0:
                        filmThickness = thick
                        break
                    
                if self.depo_or_etching == 'depo':
                    if self.depoPoint[2] == filmThickness:
                        print('depo finish')
                        break
                elif self.depo_or_etching == 'etching':
                    if self.etchingPoint[2] == filmThickness:
                        print('etch finish')
                        break      

                # self.log.info('runStep:{}, timeStep:{}, depo_count_step:{}, count_reaction_all:{},vzMax:{:.3f},vzMax:{:.3f}, filmThickness:{},  input_count:{}'\
                #               .format(i, tstep, depo_count, count_reaction, vzMax, vzMin,  filmThickness, self.parcel.shape[0]))
        # del self.log, self.fh

        return self.film, filmThickness
    
    def posGenerator(self, IN, thickness, emptyZ):
        position_matrix = np.array([np.random.rand(IN)*self.cellSizeX, \
                                    np.random.uniform(0, emptyZ, IN)+ thickness + emptyZ]).T
        position_matrix *= self.celllength
        return position_matrix
    
    def posGenerator_full(self, IN, thickness, emptyZ):
        position_matrix = np.array([np.random.rand(IN)*self.cellSizeX, \
                                    np.random.uniform(0, self.cellSizeZ-thickness-emptyZ, IN)+ thickness + emptyZ]).T
        position_matrix *= self.celllength
        return position_matrix

    def posGenerator_top(self, IN, thickness, emptyZ):
        position_matrix = np.array([np.random.rand(IN)*self.cellSizeX, \
                                    np.random.uniform(0, emptyZ, IN) + self.cellSizeZ - emptyZ]).T
        position_matrix *= self.celllength
        return position_matrix
     
    # def posGenerator_benchmark(self, IN, thickness, emptyZ):
    #     position_matrix = np.array([np.ones(IN)*self.cellSizeX/2, \
    #                                 np.ones(IN)*self.cellSizeY/2, \
    #                                 np.ones(IN)*self.cellSizeZ - emptyZ]).T
    #     position_matrix *= self.celllength
    #     return position_matrix
    def posGenerator_benchmark(self, IN, thickness, emptyZ):
        position_matrix = np.array([np.ones(IN)*self.cellSizeX/2, \
                                    np.ones(IN)*self.cellSizeZ - emptyZ]).T
        position_matrix *= self.celllength
        return position_matrix
    
    def velGenerator_maxwell_normal(self, IN):
        Random1 = np.random.rand(IN)
        Random2 = np.random.rand(IN)
        Random3 = np.random.rand(IN)
        velosity_matrix = np.array([self.max_velocity_u(Random1, Random2), \
                                        self.max_velocity_v(Random3)]).T

        energy = np.linalg.norm(velosity_matrix, axis=1)
        velosity_matrix[:,0] = np.divide(velosity_matrix[:,0], energy)
        velosity_matrix[:,1] = np.divide(velosity_matrix[:,1], energy)

        return velosity_matrix
    
    def velGenerator_updown_normal(self, IN):
        velosity_matrix = np.zeros((IN, 2))
        velosity_matrix[:, 0] = np.random.randn(IN)*0.01
        # velosity_matrix[:, 1] = np.random.randn(IN)*0.01
        velosity_matrix[:, 1] = -1 

        return velosity_matrix


    # def posGenerator_benchmark(self, IN, thickness, emptyZ):
    #     position_matrix = np.array([np.random.rand(IN)*20 + self.cellSizeX/2, \
    #                                 np.random.rand(IN)*20 + self.cellSizeY/2, \
    #                                 np.ones(IN)*self.cellSizeZ - emptyZ]).T
    #     position_matrix *= self.celllength
    #     return position_matrix
    
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

            typeID = np.zeros(N)
            # def runEtch(self, v0, typeID, time, emptyZ):
            result =  self.runEtch(velosity_matrix, typeID, tmax, emptyZ=Zgap)
            if np.any(result[0][self.depoPoint]) != 0:
                break             
        del self.log, self.fh
        return result
    
        # def runEtch(self, velGeneratorType, typeID, inputCount, emptyZ):
    def inputParticle(self, velGeneratorType, typeID, inputCount, max_react_count, tmax, Zgap):

        result =  self.runEtch(velGeneratorType, typeID, inputCount, max_react_count,tmax, emptyZ=Zgap)
        # if np.any(result[0][:, :, self.depoThick]) != 0:
        #     break             
        # del self.log, self.fh 
        return result  



if __name__ == "__main__":
    import pyvista as pv
    import cProfile


    film = np.zeros((100, 200, 3))

    bottom = 100
    height = 140

    density = 10

    center = 50

    film[:, 0:bottom, 0] = density # bottom
    film[:20, bottom:height, 2] = density # bottom
    film[-20:, bottom:height, 2] = density # bottom

    etchfilm = film

    # print(typeID[:10])
    # print(velosity_matrix[0])
    # print(vel_type_shuffle[:10])

    # print(velosity_matrix[0])

    logname = 'Multi_species_benchmark_0924'
    testEtch = etching(inputMethod='bunch', depo_or_etching='etching', 
                    etchingPoint = np.array([center, center, 37]),depoPoint = np.array([center, center, 37]),
                    density=density, center_with_direction=np.array([[35,100,75]]), 
                    range3D=np.array([[0, 70, 0, 100, 0, 150]]), InOrOut=[1], yield_hist=np.array([None]),
                    reaction_type=False, param = [1.6, -0.7],
                    sub_xy=[0,0], film=etchfilm, n=1, cellSize=etchfilm.shape, 
                    celllength=1e-5, kdtreeN=5, tstep=1e-5,
                    substrateTop=bottom,posGeneratorType='top', logname=logname)
    

    step1 = testEtch.inputParticle('maxwell', 0, int(1e2),int(1e5), 10)

    film3D = np.zeros((1, 100, 200, 3))
    film3D[0] = etchfilm

    depo1 = torch.Tensor(film3D[:, :,:, 0]!=0).to_sparse()
    depo1 = depo1.indices().numpy().T

    substrute = torch.Tensor(film3D[:,:, :, 1]!=0).to_sparse()
    substrute = substrute.indices().numpy().T
    depomesh = pv.PolyData(depo1)
    depomesh["radius"] = np.ones(depo1.shape[0])*0.5
    geom = pv.Box()

    submesh = pv.PolyData(substrute)
    submesh["radius"] = np.ones(substrute.shape[0])*0.5

    # Progress bar is a new feature on master branch
    depoglyphed = depomesh.glyph(scale="radius", geom=geom) # progress_bar=True)
    subglyphed = submesh.glyph(scale="radius", geom=geom) # progress_bar=True)

    p = pv.Plotter()
    p.add_mesh(depoglyphed, color='cyan')
    p.add_mesh(subglyphed, color='dimgray')
    p.enable_eye_dome_lighting()
    p.show()
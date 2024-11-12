import numpy as np
# import cupy as cp
from scipy.spatial import cKDTree
import time as Time
from tqdm import tqdm
import logging
# from Collision import transport
from surface_normalize_bosch import surface_normal
from numba import jit, prange
# from boundary import boundary
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

#solid = film[i, j, k, 2][Si, C4F8, mask]
#react_t g[F, c4f8, ion] s  [1,    2 , 3]
#react_t g[F, c4f8, ion] s  [Si, C4F8, mask]

# react_table = np.array([[[0.200, -1, 0], [0.0  , 0,  0]],
#                         [[0.800,  -1, 1], [0.0, 0,  0]],
#                         [[0.1 ,  -1, 0], [0.9  , 0, -1]]])

react_table = np.array([[[0.1, -1, 0, 0], [0.0, 0,  0, 0], [0.0, 0, 0, 0]],
                        [[0.8, -1, 1, 0], [0.0, 0,  0, 0], [0.0, 0, 0, 0]],
                        [[1.0,  0, 0, 0], [1.0, 0, -2, 0], [1.0, 0, 0, 0]]])

# react_table[0, 3, 4] = -2
# etching act on film, depo need output

# react_type
#       Si c4f8 mask
# sf   ([[KD, x, x],
# c4f8   [+,  x, x],
# Ar     [+, KD, +]])

react_type_table = np.array([[2, 0, 0],
                           [1, 0, 0],
                           [4, 3, 1]])


# @jit(nopython=True, parallel=True)
# def reaction_yield(parcel, film, theta):
#     # print('react parcel', parcel.shape)
#     # print('react film', film.shape)
#     # print('react theta', theta.shape)
#     num_parcels = parcel.shape[0]
#     num_reactions = react_table.shape[1]
#     choice = np.random.rand(parcel.shape[0], react_table.shape[1])
#     reactList = np.ones(parcel.shape[0], dtype=np.int_)*-1
#     for i in range(num_parcels):
#         for j in range(num_reactions):
#             if film[i, j] <= 0:
#                 choice[i, j] = 1
#     depo_parcel = np.zeros(parcel.shape[0])
#     for i in prange(parcel.shape[0]):
#         acceptList = np.zeros(react_table.shape[1], dtype=np.bool_)
#         for j in prange(film.shape[1]):
#             react_rate = react_table[int(parcel[i, -1]), j, 0]
#             if react_rate > choice[i, j]:
#                 acceptList[j] = True
#         react_choice_indices = np.where(acceptList)[0]
#         if react_choice_indices.size > 0:
#             react_choice = np.random.choice(react_choice_indices)
#             reactList[i] = react_choice
#             react_type = react_type_table[int(parcel[i, -1]), react_choice]
#             if react_type == 2: # kdtree Si-SF
#                 depo_parcel[i] = 2
#             elif react_type == 3: # kdtree Ar-c4f8
#                 depo_parcel[i] = 3
#             elif react_type == 1: # +
#                 depo_parcel[i] = 1
#             elif react_type == 4: # Ar - Si
#                 depo_parcel[i] = 4
#             elif react_type == 0:  # no reaction
#                 depo_parcel[i] = 0
#     for i in prange(parcel.shape[0]):
#         if depo_parcel[i] == 1:
#             film[i, :] += react_table[int(parcel[i, -1]), int(reactList[i]), 1:]
#             # print('chemistry')
#         if reactList[i] == -1:
#             parcel[i,3:6] = SpecularReflect(parcel[i,3:6], theta[i])
#             # print('reflection')
#             # parcel[i,3:6] = reemission(parcel[i,3:6], theta[i])

#     return film, parcel, reactList, depo_parcel

@jit(nopython=True, parallel=True)
def reaction_yield(parcel, film, theta):
    num_parcels = parcel.shape[0]
    num_reactions = react_table.shape[1]
    choice = np.random.rand(num_parcels, num_reactions)
    reactList = np.ones(num_parcels, dtype=np.int_) * -1

    # 手动循环替代布尔索引
    for i in prange(film.shape[0]):
        for j in prange(film.shape[1]):
            if film[i, j] <= 0:
                choice[i, j] = 1

    depo_parcel = np.zeros(parcel.shape[0])

    for i in prange(parcel.shape[0]):
        acceptList = np.zeros(num_reactions, dtype=np.bool_)
        for j in prange(film.shape[1]):
            react_rate = react_table[int(parcel[i, -1]), j, 0]
            if react_rate > choice[i, j]:
                acceptList[j] = True

        react_choice_indices = np.where(acceptList)[0]
        if react_choice_indices.size > 0:
            react_choice = react_choice_indices[np.random.randint(react_choice_indices.size)]
            reactList[i] = react_choice
            react_type = react_type_table[int(parcel[i, -1]), react_choice]

            if react_type == 2:
                depo_parcel[i] = 2
            elif react_type == 3:
                depo_parcel[i] = 3
            elif react_type == 1:
                depo_parcel[i] = 1
            elif react_type == 4:
                depo_parcel[i] = 4
            elif react_type == 0:
                depo_parcel[i] = 0

    for i in prange(parcel.shape[0]):
        if depo_parcel[i] == 1:
            film[i, :] += react_table[int(parcel[i, -1]), int(reactList[i]), 1:]

        if reactList[i] == -1:
            parcel[i, 3:6] = SpecularReflect(parcel[i, 3:6], theta[i])
            # parcel[i, 3:6] = reemission(parcel[i, 3:6], theta[i])

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

@jit(nopython=True)
def boundary(parcel, cellSizeX, cellSizeY, cellSizeZ, celllength):
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
    ijk = np.rint(parcel[:, :3] * inv_celllength).astype(np.int32)
    parcel[:, 6:9] = ijk

    return parcel

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

class etching(surface_normal):
    def __init__(self,inputMethod, depo_or_etching, etchingPoint,depoPoint,density, 
                 center_with_direction, range3D, InOrOut, yield_hist,
                 maskTop, maskBottom, maskStep, maskCenter, backup,#surface_normal
                 mirrorGap, # mirror
                 reaction_type,  #reaction 
                 param, n, celllength, kdtreeN,filmKDTree,weight,
                 tstep, substrateTop, posGeneratorType, logname):
        # super().__init__(tstep, pressure_pa, temperature, cellSize, celllength, chamberSize)
        surface_normal.__init__(self, center_with_direction, range3D, InOrOut,celllength, tstep, yield_hist,\
                                maskTop, maskBottom, maskStep, maskCenter, backup)
        self.param = param # n beta
        self.kdtreeN = kdtreeN
        self.celllength = celllength
        self.timeStep = tstep
        # self.sub_x = sub_xy[0]
        # self.sub_y = sub_xy[1]
        # self.substrate = film
        self.depo_or_etching = depo_or_etching
        self.depoPoint = depoPoint
        self.etchingPoint = etchingPoint
        self.density = density
        self.inputMethod = inputMethod
        self.n = n
        self.T = 300
        self.Cm = (2*1.380649e-23*self.T/(27*1.66e-27) )**0.5 # (2kT/m)**0.5 27 for the Al

        # self.film = film
        self.filmKDTree = filmKDTree
        self.weight = weight
        # filmKDTree=np.array([[2, 0], [3, 1]])
        #       KDTree    [depo_parcel,  film]
        self.mirrorGap = mirrorGap
        # self.surface_depo_mirror = np.zeros((self.cellSizeX+int(self.mirrorGap*2), self.cellSizeY+int(self.mirrorGap*2), self.cellSizeZ))
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
    def boundary(self):

        # print('bf bound',self.parcel.flags.f_contiguous)
        # if self.symmetry == True:
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
        # print('af bound',self.parcel.flags.f_contiguous)
        if np.any(indices):
            self.parcel = self.parcel[~indices].copy(order='F')
            # self.parcel = np.delete(self.parcel, np.where(indices)[0], axis=0)
            # self.parcel = self.parcel.ravel(order='F')[~indices]
        # print('af bound',self.parcel.flags.f_contiguous)

    def removeFloat(self):  # fast scanZ
        filmC = self.film[:,:,:,0]
        # 获取当前平面的非零元素布尔索引
        current_plane = filmC != 0

        # 创建一个全是False的布尔数组来存储邻居的检查结果
        neighbors = np.zeros_like(filmC, dtype=bool)

        # 检查各个方向的邻居是否为零
        neighbors[1:, :, :] |= filmC[:-1, :, :] != 0  # 上面的邻居不为0
        neighbors[:-1, :, :] |= filmC[1:, :, :] != 0  # 下面的邻居不为0
        neighbors[:, 1:, :] |= filmC[:, :-1, :] != 0  # 左边的邻居不为0
        neighbors[:, :-1, :] |= filmC[:, 1:, :] != 0  # 右边的邻居不为0
        neighbors[:, :, 1:] |= filmC[:, :, :-1] != 0  # 前面的邻居不为0
        neighbors[:, :, :-1] |= filmC[:, :, 1:] != 0  # 后面的邻居不为0

        # 孤立单元格的条件是当前平面元素不为0且所有方向的邻居都为0
        condition = current_plane & ~neighbors

        # 将孤立的单元格设为0
        self.film[condition, :] = 0

    def removeFloatPolymer(self):  # fast scanZ
        filmC = self.film[:,:,:,0]
        # 获取当前平面的非零元素布尔索引
        current_plane = self.film[:,:,:,1] != 0

        # 创建一个全是False的布尔数组来存储邻居的检查结果
        neighbors = np.zeros_like(filmC, dtype=bool)

        # 检查各个方向的邻居是否为零
        neighbors[1:, :, :] |= filmC[:-1, :, :] != 0  # 上面的邻居不为0
        neighbors[:-1, :, :] |= filmC[1:, :, :] != 0  # 下面的邻居不为0
        neighbors[:, 1:, :] |= filmC[:, :-1, :] != 0  # 左边的邻居不为0
        neighbors[:, :-1, :] |= filmC[:, 1:, :] != 0  # 右边的邻居不为0
        neighbors[:, :, 1:] |= filmC[:, :, :-1] != 0  # 前面的邻居不为0
        neighbors[:, :, :-1] |= filmC[:, :, 1:] != 0  # 后面的邻居不为0

        # 孤立单元格的条件是当前平面元素不为0且所有方向的邻居都为0
        condition = current_plane & ~neighbors

        # 将孤立的单元格设为0
        self.film[condition, :] = 0

    def etching_film(self):
        i, j, k = self.get_indices()
        sumFilm = np.sum(self.film, axis=-1)
        indice_inject = np.array(sumFilm[i, j, k] != 0)
        reactListAll = np.ones(indice_inject.shape[0])*-2
        oscilationList = np.zeros_like(indice_inject, dtype=np.bool_)

        if np.any(indice_inject):
            pos_1, vel_1 = self.get_positions_and_velocities(indice_inject)
            get_plane, get_theta, ddshape, maxdd, ddi, dl1, oscilation_indice = self.calculate_injection(pos_1, vel_1, sumFilm)

            film_update_results = self.update_film(get_plane, get_theta, indice_inject, ddi, dl1, ddshape, maxdd)

            self.handle_surface_depo(film_update_results, pos_1, sumFilm, indice_inject, reactListAll, oscilationList, film_update_results['reactList'], oscilation_indice)

            return film_update_results['depo_count'], ddshape, maxdd, ddi, dl1
        else:
            return 0, 0, 0, 0, 0

    # def get_indices(self):
    #     i = self.parcel[:, 6].astype(int)
    #     j = self.parcel[:, 7].astype(int)
    #     k = self.parcel[:, 8].astype(int)
    #     return i, j, k

    # def get_positions_and_velocities(self, indice_inject):
    #     pos_1 = self.parcel[indice_inject, :3]
    #     vel_1 = self.parcel[indice_inject, 3:6]
    #     return pos_1, vel_1
    
    def get_indices(self):
        # 直接将切片操作和数据类型转换合并
        return self.parcel[:, 6].astype(int), self.parcel[:, 7].astype(int), self.parcel[:, 8].astype(int)

    def get_positions_and_velocities(self, indice_inject):
        # 直接返回切片
        return self.parcel[indice_inject, :3], self.parcel[indice_inject, 3:6]

    def calculate_injection(self, pos_1, vel_1, sumFilm):
        self.planes = self.get_pointcloud(sumFilm)
        get_plane, get_theta, ddshape, maxdd, ddi, dl1, pos1e4, vel1e4, oscilation_indice = self.get_inject_normal(self.planes, pos_1, vel_1)
        return get_plane, get_theta, ddshape, maxdd, ddi, dl1, oscilation_indice

    def update_film(self, get_plane, get_theta, indice_inject, ddi, dl1, ddshape, maxdd):
        self.film[get_plane[:,0], get_plane[:,1], get_plane[:,2]], self.parcel[indice_inject, :], reactList, depo_parcel = \
            reaction_yield(self.parcel[indice_inject], self.film[get_plane[:,0], get_plane[:,1], get_plane[:,2]], get_theta)

        results = {
            'reactList': reactList,
            'depo_parcel': depo_parcel,
            'depo_count': np.sum(depo_parcel == self.depo_count_type),
            'ddshape': ddshape,
            'maxdd': maxdd,
            'ddi': ddi,
            'dl1': dl1
        }
        return results

    def toKDtree(self):
        return cKDTree(np.argwhere(self.surface_depo_mirror == True) * self.celllength)

    def find_surface_vaccum(self, film): # fast scanZ
        film = torch.Tensor(film)
        xshape, yshape, zshape = film.shape
        self.zshape = zshape
        # 初始化一个全零的表面稀疏张量
        surface_sparse = torch.zeros((xshape, yshape, zshape))
        
        # 获取当前平面与前后平面的布尔索引
        current_plane = film == 0

        # 获取周围邻居的布尔索引
        neighbors = torch.zeros_like(film, dtype=torch.bool)
        
        neighbors[1:, :, :] |= film[:-1, :, :] != 0  # 上面
        neighbors[:-1, :, :] |= film[1:, :, :] != 0  # 下面
        neighbors[:, 1:, :] |= film[:, :-1, :] != 0  # 左边
        neighbors[:, :-1, :] |= film[:, 1:, :] != 0  # 右边
        neighbors[:, :, 1:] |= film[:, :, :-1] != 0  # 前面
        neighbors[:, :, :-1] |= film[:, :, 1:] != 0  # 后面
        
        # 获取满足条件的索引
        condition = current_plane & neighbors
        
        # 更新表面稀疏张量
        surface_sparse[condition] = 1
        
        return surface_sparse.to_sparse().indices().T.numpy()

    def reflect_pos(self, sumFilm, indice_reflect):
        surface_vaccum = self.find_surface_vaccum(sumFilm) 
        surface_vaccum_tree = cKDTree(surface_vaccum*self.celllength)

        # particle data struction np.array([posX, posY, posZ, velX, velY, velZ, i, j, k, typeID])
        dd, ii = surface_vaccum_tree.query(self.parcel[indice_reflect, :3] , k=1, workers=1)
        self.parcel[indice_reflect, :3] =  surface_vaccum[ii]*self.celllength
        self.parcel[indice_reflect, 6:9] =  surface_vaccum[ii]

    def handle_surface_depo(self, film_update_results, pos_1, sumFilm, indice_inject, reactListAll, oscilationList, reactList, oscilation_indice):
        depo_parcel = film_update_results['depo_parcel']
        # reactList = film_update_results['reactList']
        # self.reflect_pos(sumFilm, indice_inject[reactListAll == -1])

        for type in self.filmKDTree:
            # Check if the depo_parcel matches the current type
            if np.any(depo_parcel == type[0]):
                
                # Process surface deposition
                self.process_surface_depo(type)

                # Build surface KDTree
                # surface_tree = cKDTree(np.argwhere(self.surface_depo_mirror == True) * self.celllength)
                surface_tree = self.toKDtree()
                
                # Query the KDTree for neighbors
                ii, dd, surface_indice = self.query_surface_tree(surface_tree, pos_1, depo_parcel, type)

                # Distribute deposition
                self.distribute_depo(surface_indice, ii, dd, type)

                # Handle deposition or etching
                self.handle_deposition_or_etching(type)

        reactListAll[indice_inject] = reactList
        oscilationList[indice_inject] = oscilation_indice

        if np.any(reactListAll != -1):
            indice_inject[np.where(reactListAll == -1)] = False
            indice_inject[oscilationList == True] = False
            self.parcel = self.parcel[~indice_inject]
    # def check_depo_parcel(self, depo_parcel, type):
    #     # Check if any depo_parcel matches the type
    #     return np.any(depo_parcel == type[0])

    def process_surface_depo(self, type):
        # Generate surface deposition mask
        surface_depo = np.array(self.film[:, :, :, type[1]] > 0)
        self.update_surface_mirror(surface_depo)
        # return surface_depo

    def query_surface_tree(self, surface_tree, pos_1, depo_parcel, type):
        # Adjust positions for mirror and query nearest neighbors
        to_depo = np.where(depo_parcel == type[0])[0]
        pos_1[:, 0] += self.mirrorGap * self.celllength
        pos_1[:, 1] += self.mirrorGap * self.celllength
        dd, ii = surface_tree.query(pos_1[to_depo], k=self.kdtreeN, workers=32)
        surface_indice = np.argwhere(self.surface_depo_mirror == True)
        return ii, dd, surface_indice

    def handle_deposition_or_etching(self, type):
        if self.depo_or_etching == 'depo':
            surface_film = np.array(self.film[:, :, :, type[1]] >= 11)
            self.film[surface_film, type[1]] = self.density
        elif self.depo_or_etching == 'etching':
            surface_film = np.array(self.film[:,:,:,type[1]] < 9)
            self.film[surface_film, type[1]] = 0


    def update_surface_mirror(self, surface_depo):
        self.surface_depo_mirror[self.mirrorGap:self.mirrorGap+self.cellSizeX, self.mirrorGap:self.mirrorGap+self.cellSizeY, :] = surface_depo
        self.surface_depo_mirror[:self.mirrorGap, self.mirrorGap:self.mirrorGap+self.cellSizeY, :] = surface_depo[-self.mirrorGap:, :, :]
        self.surface_depo_mirror[-self.mirrorGap:, self.mirrorGap:self.mirrorGap+self.cellSizeY, :] = surface_depo[:self.mirrorGap, :, :]
        self.surface_depo_mirror[self.mirrorGap:self.mirrorGap+self.cellSizeX, :self.mirrorGap, :] = surface_depo[:, -self.mirrorGap:, :]
        self.surface_depo_mirror[self.mirrorGap:self.mirrorGap+self.cellSizeX:, -self.mirrorGap:, :] = surface_depo[:, :self.mirrorGap, :]
        self.surface_depo_mirror[:self.mirrorGap, :self.mirrorGap, :] = surface_depo[-self.mirrorGap:, -self.mirrorGap:, :]
        self.surface_depo_mirror[:self.mirrorGap, -self.mirrorGap:, :] = surface_depo[-self.mirrorGap:, :self.mirrorGap, :]
        self.surface_depo_mirror[-self.mirrorGap:, :self.mirrorGap, :] = surface_depo[:self.mirrorGap, -self.mirrorGap:, :]
        self.surface_depo_mirror[-self.mirrorGap:, -self.mirrorGap:, :] = surface_depo[:self.mirrorGap, :self.mirrorGap, :]

    def distribute_depo(self, surface_indice, ii, dd, type):
        ddsum = np.sum(dd, axis=1)

        for kdi in range(self.kdtreeN):
            i1 = surface_indice[ii][:, kdi, 0]
            j1 = surface_indice[ii][:, kdi, 1]
            k1 = surface_indice[ii][:, kdi, 2]
            
            # Apply mirror gap correction
            i1 -= self.mirrorGap
            j1 -= self.mirrorGap
            indiceXMax = i1 >= self.cellSizeX
            indiceXMin = i1 < 0
            i1[indiceXMax] -= self.cellSizeX
            i1[indiceXMin] += self.cellSizeX

            indiceYMax = j1 >= self.cellSizeY
            indiceYMin = j1 < 0
            j1[indiceYMax] -= self.cellSizeY
            j1[indiceYMin] += self.cellSizeY

            self.film[i1, j1, k1, type[1]] += self.weight * dd[:, kdi] / ddsum


    def correct_indices(self, i1, j1):
        i1[i1 >= self.cellSizeX] -= self.cellSizeX
        i1[i1 < 0] += self.cellSizeX
        j1[j1 >= self.cellSizeY] -= self.cellSizeY
        j1[j1 < 0] += self.cellSizeY
        return i1, j1

    def toboundary(self):
        self.parcel = boundary(self.parcel, self.cellSizeX, self.cellSizeY, self.cellSizeZ, self.celllength)
      
    def toupdate_parcel(self, tStep):
        self.parcel = update_parcel(self.parcel, self.celllength, tStep)

    def getAcc_depo(self, tStep):

        # pos, vel, i, j, k, cellSize_x, cellSize_y, cellSize_z,

        # print('bf bound',self.parcel.flags.f_contiguous)
        self.toboundary()
        # self.parcel = boundary(self.parcel, self.cellSizeX, self.cellSizeY, self.cellSizeZ, self.celllength)
        # print('af bound',self.parcel.flags.f_contiguous)
        self.removeFloat()
        self.removeFloatPolymer()
        # print(pos_cp)
        # print('bf bound',self.parcel.flags.f_contiguous)
        depo_count, ddshape, maxdd, ddi, dl1 = self.etching_film()
        # print('bf bound',self.parcel.flags.f_contiguous)
        # self.parcel = update_parcel(self.parcel, self.celllength, tStep)
        self.toupdate_parcel(tStep)
        # print('af bound',self.parcel.flags.f_contiguous)
        # 预计算 1/self.celllength，避免重复计算
        # inv_celllength = 1.0 / self.celllength

        # # 更新位置
        # self.parcel[:, :3] += self.parcel[:, 3:6] * tStep

        # # 使用 np.rint() 进行取整，然后整体转换为整数类型，减少 .astype() 调用
        # # ijk = np.rint((self.parcel[:, :3] * inv_celllength) + 0.5).astype(int)
        # ijk = np.rint((self.parcel[:, :3] * inv_celllength)).astype(int)
        # # 一次性赋值给 parcel 的第 6、7、8 列
        # self.parcel[:, 6:9] = ijk
        # print('af bound',self.parcel.flags.f_contiguous)
        return depo_count, ddshape, maxdd, ddi, dl1 #, film_max, surface_true

    # particle data struction np.array([posX, posY, posZ, velX, velY, velZ, i, j, k, typeID])
    def Parcelgen(self, pos, vel, typeID):

        # i = np.floor((pos[:, 0]/self.celllength) + 0.5).astype(int)
        # j = np.floor((pos[:, 1]/self.celllength) + 0.5).astype(int)
        # k = np.floor((pos[:, 2]/self.celllength) + 0.5).astype(int)
        i = np.floor((pos[:, 0]/self.celllength)).astype(int)
        j = np.floor((pos[:, 1]/self.celllength)).astype(int)
        k = np.floor((pos[:, 2]/self.celllength)).astype(int)
        # parcelIn = np.zeros((pos.shape[0], 10), order='F')
        parcelIn = np.zeros((pos.shape[0], 10))
        parcelIn[:, :3] = pos
        parcelIn[:, 3:6] = vel
        parcelIn[:, 6] = i
        parcelIn[:, 7] = j
        parcelIn[:, 8] = k
        parcelIn[:, 9] = typeID

        # print(self.parcel.flags.f_contiguous)
        self.parcel = np.concatenate((self.parcel, parcelIn))
        # print(self.parcel.flags)

    def runEtch(self, velGeneratorType, typeID, inputCount, runningCount, max_react_count, emptyZ, step):

        self.log.info('inputType:{}'.format(typeID))
        # if step == 0:
        #     self.parcel = np.zeros((1, 10))
        # tmax = time
        start_time = Time.time()
        tstep = self.timeStep
        t = 0
        # inputCount = int(v0.shape[0]/(tmax/tstep))

        self.planes = self.get_pointcloud(np.sum(self.film, axis=-1))
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
        # self.parcel = self.parcel[1:, :]
        ti = 0
        with tqdm(total=100, desc='particle input', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
            previous_percentage = 0  # 记录上一次的百分比
            while self.parcel.shape[0] > 500:
                # np.save('./bosch_data_1011_ratio08_trench_condition5_300wide/parcel4_{}'.format(ti), self.parcel)
                ti += 1
                # print('bbf bound',self.parcel.flags.f_contiguous)
                depo_count, ddshape, maxdd, ddi, dl1 = self.getAcc_depo(tstep)
                # print(self.parcel.flags.f_contiguous)
                # print('parcel', self.parcel.shape)
                # print('bf bound',self.parcel.flags.f_contiguous)
                count_reaction += depo_count
                # if count_reaction > self.max_react_count:
                #     break
                t += tstep
                if count_reaction > max_react_count:
                    end_time = Time.time()

                    # 计算运行时间并转换为分钟和秒
                    elapsed_time = end_time - start_time
                    minutes = int(elapsed_time // 60)
                    seconds = int(elapsed_time % 60)

                    # 输出运行时间
                    self.log.info(f"run time: {minutes} min {seconds} sec")
                    self.log.info('DataFind---step:{},inputType:{},count_reaction_all:{},inputAll:{}'.format(step,typeID,count_reaction,inputAll))
                    break

                if count_reaction > max_react_count/8 and depo_count < 1:
                    end_time = Time.time()

                    # 计算运行时间并转换为分钟和秒
                    elapsed_time = end_time - start_time
                    minutes = int(elapsed_time // 60)
                    seconds = int(elapsed_time % 60)

                    # 输出运行时间
                    self.log.info(f"stop by depo_count run time: {minutes} min {seconds} sec")
                    self.log.info('DataFind---step:{},inputType:{},count_reaction_all:{},inputAll:{}'.format(step,typeID,count_reaction,inputAll))
                    break

                vzMax = np.max(self.parcel[:,5])
                vzMin = np.min(self.parcel[:,5])
                # if self.inputMethod == 'bunch' and inputAll < max_react_count:
                if self.parcel.shape[0] < runningCount:
                    inputAll += inputCount
                    p1 = posGenerator(inputCount, filmThickness, emptyZ)
                    v1 = velGenerator(inputCount)
                    typeIDIn = np.zeros(inputCount)
                    typeIDIn[:] = typeID
                    self.Parcelgen(p1, v1, typeIDIn)

                # planes = self.get_pointcloud(np.sum(self.film, axis=-1))

                current_percentage = int(count_reaction / max_react_count * 100)  # 当前百分比
                if current_percentage > previous_percentage:
                    update_value = current_percentage - previous_percentage  # 计算进度差值
                    pbar.update(update_value)
                    previous_percentage = current_percentage  # 更新上一次的百分比
                self.log.info('particleIn:{}, timeStep:{}, depo_count_step:{}, count_reaction_all:{},inputAll:{},vzMax:{:.3f},vzMin:{:.3f}, filmThickness:{}, input_count:{}, ddi:{}, dl1:{}, ddshape:{}, maxdd:{}'\
                            .format(previous_percentage, tstep, depo_count, count_reaction, inputAll,  vzMax, vzMin,  filmThickness, self.parcel.shape[0], ddi, dl1, ddshape, maxdd))
            
                for thick in range(self.film.shape[2]):
                    if np.sum(self.film[int(self.cellSizeX/2),int(self.cellSizeY/2), thick, :]) == 0:
                        filmThickness = thick
                        break
                    
                # if self.depo_or_etching == 'depo':
                #     if self.depoPoint[2] == filmThickness:
                #         print('depo finish')
                #         break
                # elif self.depo_or_etching == 'etching':
                #     if self.etchingPoint[2] == filmThickness:
                #         print('etch finish')
                #         break      
                # print('af bound',self.parcel.flags.f_contiguous)
                # self.log.info('runStep:{}, timeStep:{}, depo_count_step:{}, count_reaction_all:{},vzMax:{:.3f},vzMax:{:.3f}, filmThickness:{},  input_count:{}'\
                #               .format(i, tstep, depo_count, count_reaction, vzMax, vzMin,  filmThickness, self.parcel.shape[0]))
        # del self.log, self.fh

        return self.film, filmThickness, self.parcel
    
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
     
    # def posGenerator_benchmark(self, IN, thickness, emptyZ):
    #     position_matrix = np.array([np.ones(IN)*self.cellSizeX/2, \
    #                                 np.ones(IN)*self.cellSizeY/2, \
    #                                 np.ones(IN)*self.cellSizeZ - emptyZ]).T
    #     position_matrix *= self.celllength
    #     return position_matrix
    def posGenerator_benchmark(self, IN, thickness, emptyZ):
        position_matrix = np.array([np.ones(IN)*self.cellSizeX/2, \
                                    np.ones(IN)*self.cellSizeY/2, \
                                    np.ones(IN)*self.cellSizeZ - emptyZ]).T
        position_matrix *= self.celllength
        return position_matrix
    
    def velGenerator_maxwell_normal(self, IN):
        Random1 = np.random.rand(IN)
        Random2 = np.random.rand(IN)
        Random3 = np.random.rand(IN)
        velosity_matrix = np.array([self.max_velocity_u(Random1, Random2), \
                                    self.max_velocity_w(Random1, Random2), \
                                        self.max_velocity_v(Random3)]).T

        energy = np.linalg.norm(velosity_matrix, axis=1)
        velosity_matrix[:,0] = np.divide(velosity_matrix[:,0], energy)
        velosity_matrix[:,1] = np.divide(velosity_matrix[:,1], energy)
        velosity_matrix[:,2] = np.divide(velosity_matrix[:,2], energy)

        return velosity_matrix
    
    def velGenerator_updown_normal(self, IN):
        velosity_matrix = np.zeros((IN, 3))
        velosity_matrix[:, 0] = np.random.randn(IN)*0.001
        velosity_matrix[:, 1] = np.random.randn(IN)*0.001
        velosity_matrix[:, 2] = -1 

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
    def inputParticle(self,film, parcel, velGeneratorType, typeID, inputCount, runningCount, max_react_count, depo_count_type, Zgap, step):
        self.depo_count_type = depo_count_type
        self.film = film
        self.parcel = parcel
        self.cellSizeX = self.film.shape[0]
        self.cellSizeY = self.film.shape[1]
        self.cellSizeZ = self.film.shape[2]
        self.surface_depo_mirror = np.zeros((self.cellSizeX+int(self.mirrorGap*2), self.cellSizeY+int(self.mirrorGap*2), self.cellSizeZ))
        print(self.surface_depo_mirror.shape)
        self.log.info('circle step:{}'.format(step))
        result =  self.runEtch(velGeneratorType, typeID, inputCount,runningCount, max_react_count, Zgap, step)
        # if np.any(result[0][:, :, self.depoThick]) != 0:
        #     break             
        # del self.log, self.fh 
        return result  



if __name__ == "__main__":
    import pyvista as pv
    import cProfile


    maskUp = 2.09
    maskBottom = 2.6
    maskDeep = 2.808

    diameter = 60
    maskUp_sim = maskUp/maskBottom*diameter
    maskDeep_sim = maskDeep/maskBottom*diameter

    film = np.zeros((100, 100, 200, 3))

    bottom = 100
    height = bottom + int(maskDeep_sim)

    density = 10

    sphere = np.ones((100, 100, 200), dtype=bool)

    diameter = 60

    center = 50
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

    # maskUp = 2.09
    # maskBottom = 2.6
    # maskDeep = 2.808

    # diameter = 120
    # maskUp_sim = maskUp/maskBottom*diameter
    # maskDeep_sim = maskDeep/maskBottom*diameter

    # film = np.zeros((200, 200, 250, 3))

    # bottom = 100
    # height = bottom + int(maskDeep_sim)

    # density = 10

    # sphere = np.ones((200, 200, 250), dtype=bool)

    # diameter = 120

    # center = 100
    # for k in range(int(diameter/2 - int(maskUp_sim/2))):
    #     # print(diameter/2 - k)
    #     radius = diameter/2 - k
    #     # print('deep', int(bottom + maskDeep_sim/(diameter/2 - maskUp_sim/2)*k))
    #     bottom_step = int(bottom + maskDeep_sim/(diameter/2 - maskUp_sim/2)*k)
    #     for i in range(sphere.shape[0]):
    #         for j in range(sphere.shape[1]):
    #             if np.abs(i-center)*np.abs(i-center) + np.abs(j-center)*np.abs(j-center) < radius*radius:
    #                 sphere[i, j, bottom_step:bottom_step+int(maskDeep_sim/(diameter/2 - maskUp_sim/2))] = 0

    # film[sphere, 2] = density
    # film[:, :, height:, :] = 0
    # film[:, :, 0:bottom, 0] = density # bottom
    # film[:, :, 0:bottom, 1] = 0 # bottom
    # film[:, :, 0:bottom, 2] = 0 # bottom


    etchfilm = film

    logname = 'Multi_species_benchmark_1021_hole_ratio01'

    testEtch = etching(inputMethod='bunch', depo_or_etching='etching', 
                    etchingPoint = np.array([center, center, bottom-30]),depoPoint = np.array([center, center, bottom-30]),
                    density=density, center_with_direction=np.array([[int(etchfilm.shape[0]/2),int(etchfilm.shape[1]/2),150]]), 
                    range3D=np.array([[0, etchfilm.shape[0], 0, etchfilm.shape[1], 0, etchfilm.shape[2]]]), InOrOut=[1], yield_hist=np.array([None]),
                    maskTop=40, maskBottom=98, maskStep=10, maskCenter=[int(etchfilm.shape[0]/2), int(etchfilm.shape[1]/2)],backup=False, 
                    mirrorGap=5,
                    reaction_type=False, param = [1.6, -0.7],n=1,
                    celllength=1e-5, kdtreeN=5, filmKDTree=np.array([[2, 0], [3, 1]]),weight=-0.2, tstep=1e-5,
                    substrateTop=bottom,posGeneratorType='top', logname=logname)
    

    cicle = 100
    celllength=1e-5
    parcel = np.array([[etchfilm.shape[0]*celllength, etchfilm.shape[0]*celllength, 199*celllength, 0, 0, 1, etchfilm.shape[0], etchfilm.shape[0], 199, 0]])
    # parcel = np.array([[etchfilm.shape[0]*celllength, etchfilm.shape[0]*celllength, 199*celllength, 0, 0, 1, etchfilm.shape[0], etchfilm.shape[0], 199, 0]], order='F')
    # for i in range(cicle):
    step1 = testEtch.inputParticle(etchfilm, parcel, 'maxwell', 0, int(1e4), int(5e5), int(1e5),2, 10, 1)

    np.save('./bosch_data_1022_timeit/bosch_sf_step_test_Ar', etchfilm)
    #                                               (velGeneratorType, typeID, inputCount, emptyZ=Zgap)
    # maxwell = 'maxwell'
    # cProfile.run('etching1 = testEtch.inputParticle(maxwell, 0, int(1e5),int(1e7), 10)', 'noMirror_cprofile')

    # etching1 = testEtch.inputParticle(125, velosity_matrix, typeID, 2e-3, 10)

    # sumFilm = np.sum(etching1[0], axis=-1)

    # # depo1 = torch.Tensor(np.logical_and(sumFilm[:60, :, :,]!=10, sumFilm[:60, :, :,]!=0)).to_sparse()
    # # depo1 = depo1.indices().numpy().T

    # # substrute = torch.Tensor(sumFilm[:60, :, :,]==10).to_sparse()
    # # substrute = substrute.indices().numpy().T
    # # depomesh = pv.PolyData(depo1)
    # # depomesh["radius"] = np.ones(depo1.shape[0])*0.5
    # # geom = pv.Box()

    # # submesh = pv.PolyData(substrute)
    # # submesh["radius"] = np.ones(substrute.shape[0])*0.5

    # # # Progress bar is a new feature on master branch
    # # depoglyphed = depomesh.glyph(scale="radius", geom=geom) # progress_bar=True)
    # # subglyphed = submesh.glyph(scale="radius", geom=geom) # progress_bar=True)

    # # p = pv.Plotter()
    # # # p.add_mesh(depoglyphed, color='cyan')
    # # p.add_mesh(subglyphed, color='dimgray')
    # # p.enable_eye_dome_lighting()
    # # p.show()


    # point_cloud = pv.PolyData(etching1[1][:, 3:])
    # vectors = etching1[1][:, :3]

    # point_cloud['vectors'] = vectors
    # arrows = point_cloud.glyph(
    #     orient='vectors',
    #     scale=1000,
    #     factor=2,
    # )

    # # Display the arrows
    # plotter = pv.Plotter()
    # plotter.add_mesh(point_cloud, color='maroon', point_size=5.0, render_points_as_spheres=True)
    # # plotter.add_mesh(arrows, color='lightblue')
    # # plotter.add_point_labels([point_cloud.center,], ['Center',],
    # #                          point_color='yellow', point_size=20)
    # plotter.show_grid()
    # plotter.show()
import numpy as np
# import cupy as cp
from scipy.spatial import cKDTree
# from scipy.spatial import KDTree
from pykdtree.kdtree import KDTree
import time as Time
from tqdm import tqdm
import logging
# from Collision import transport
from surface_normalize_etching_depo import surface_normal
from numba import jit, prange
# from boundary import boundary
# import torch
from scipy import interpolate

import sputter_angle_dist as sp_angle

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

# react_table = np.array([[[0.1, -1, 0, 0], [0.0, 0,  0, 0], [0.0, 0, 0, 0]],
#                         [[0.8, -1, 1, 0], [0.0, 0,  0, 0], [0.0, 0, 0, 0]],
#                         [[1.0,  0, 0, 0], [1.0, 0, -2, 0], [1.0, 0, 0, 0]]])

# # react_table[0, 3, 4] = -2
# # etching act on film, depo need output

# # react_type
# #       Si c4f8 mask
# # sf   ([[KD, x, x],
# # c4f8   [+,  x, x],
# # Ar     [+, KD, +]])

# react_type_table = np.array([[2, 0, 0],
#                            [1, 0, 0],
#                            [4, 3, 1]])

# react_table = np.array([[[1.0, 0, 1], [1.0, 0, 1]],
#                         [[1.00, -1, 0], [1.00, 0, -1]]])

react_table = np.array([[[0.0, 0, 1], [0.0, 0, 1]],
                        [[0.0, -1, 0], [1.0, 0, -1]]])

react_type_table = np.array([[2, 0],
                             [3, 0]])


# @jit(nopython=True, parallel=True)
# def reaction_yield(parcel, film, theta):
#     num_parcels = parcel.shape[0]
#     num_reactions = react_table.shape[1]
#     choice = np.random.rand(num_parcels, num_reactions)
#     reactList = np.ones(num_parcels, dtype=np.int_) * -1

#     # 手动循环替代布尔索引
#     for i in prange(film.shape[0]):
#         for j in prange(film.shape[1]):
#             if film[i, j] <= 0:
#                 choice[i, j] = 1

#     depo_parcel = np.zeros(parcel.shape[0])

#     for i in prange(parcel.shape[0]):
#         acceptList = np.zeros(num_reactions, dtype=np.bool_)
#         for j in prange(film.shape[1]):
#             react_rate = react_table[int(parcel[i, -1]), j, 0]
#             if react_rate > choice[i, j]:
#                 acceptList[j] = True

#         react_choice_indices = np.where(acceptList)[0]
#         if react_choice_indices.size > 0:
#             react_choice = react_choice_indices[np.random.randint(react_choice_indices.size)]
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

#         if reactList[i] == -1:
#             parcel[i, 3:6] = SpecularReflect(parcel[i, 3:6], theta[i])

#     return film, parcel, reactList, depo_parcel

def Rn_coeffcient(c1, c2, c3, c4, alpha):
    return c1 + c2*np.tanh(c3*alpha - c4)

rn_angle = np.arange(0, np.pi/2, 0.1)
# xnew = np.array([])
rn_prob = [Rn_coeffcient(0.9423, 0.9434, 2.342, 3.026, i) for i in rn_angle]
rn_prob /= rn_prob[-1]
# rn_func = interpolate.interp1d(rn_angle, rn_prob, kind='quadratic')

# react ratio have to be set to 0.0 for 100% react probability for the following react yield

@jit(nopython=True, parallel=True)
def reaction_yield(parcel, film, normal):
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
            if int(parcel[i, -1]) == 1:
                dot_product = np.dot(parcel[i, 3:6], normal[i])
                dot_product = np.abs(dot_product)
                angle_rad = np.arccos(dot_product)
                react_rate = np.interp(angle_rad, rn_angle, rn_prob)
            else:
                react_rate = react_table[int(parcel[i, -1]), j, 0]
            if react_rate < choice[i, j]:
                acceptList[j] = True

        react_choice_indices = np.where(acceptList)[0]
        if react_choice_indices.size > 0:
            react_choice = react_choice_indices[np.random.randint(react_choice_indices.size)]
            reactList[i] = react_choice
            react_type = react_type_table[int(parcel[i, -1]), react_choice]

            if react_type == 2: # kdtree Si-SF
                depo_parcel[i] = 2
            elif react_type == 3: # kdtree Ar-c4f8
                depo_parcel[i] = 3
            elif react_type == 1: # +
                depo_parcel[i] = 1
            elif react_type == 4: # Ar - Si
                depo_parcel[i] = 4
            elif react_type == 0:  # no reaction
                depo_parcel[i] = 0

    for i in prange(parcel.shape[0]):
        if depo_parcel[i] == 1:
            film[i, :] += react_table[int(parcel[i, -1]), int(reactList[i]), 1:]

        if reactList[i] == -1:
            parcel[i, 3:6] = SpecularReflect(parcel[i, 3:6], normal[i])

    return film, parcel, reactList, depo_parcel


@jit(nopython=True)
def SpecularReflect(vel, normal):
    return vel - 2*vel@normal*normal

# kB = 1.380649e-23
# T = 100

# @jit(nopython=True)
# def DiffusionReflect(vel, normal):
#     mass = 27*1.66e-27
#     Ut = vel - vel@normal*normal
#     tw1 = Ut/np.linalg.norm(Ut)
#     tw2 = np.cross(tw1, normal)
#     # U = np.sqrt(kB*T/particleMass[i])*(np.random.randn()*tw1 + np.random.randn()*tw2 - np.sqrt(-2*np.log((1-np.random.rand())))*normal)
#     U = np.sqrt(kB*T/mass)*(np.random.randn()*tw1 + np.random.randn()*tw2 - np.sqrt(-2*np.log((1-np.random.rand())))*normal)
#     UN = U / np.linalg.norm(U)
#         # UN[i] = U
#     return UN

# def angle_to_vel(vel, normal):
#     vels = np.zeros_like(vel)
#     for i in range(vels.shape[0]):
#         R, theta, phi = sp_angle.phi_theta_dist(resolu, theta1, Eth, E)
#         phitheta = sp_angle.accept_reject_phi_theta(R, theta, phi)


@jit(nopython=True)
def reemission_multi(vel, normal):
    vels = np.zeros_like(vel)
    for i in range(vels.shape[0]):
        # vels[i] = DiffusionReflect(vel[i], normal[i])
        vels[i] = SpecularReflect(vel[i], normal[i])
    return vels

@jit(nopython=True)
def redepo_Generator_numba(i, j, k, vel, normal):
    poses = np.array([i, j, k]).T
    vels = reemission_multi(vel, normal)
    typeID = np.zeros(vel.shape[0]) # 0 for Al depo
    for n in range(10):
        poses = np.concatenate((poses, poses))
        vels = np.concatenate((vels, vels))
        typeID = np.concatenate((typeID, typeID))
    return poses, vels, typeID

    # self.Parcelgen(poses, vels, typeID)

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
    def __init__(self,inputMethod, etchingPoint,depoPoint,density, 
                 center_with_direction, range3D, InOrOut, yield_hist,
                 maskTop, maskBottom, maskStep, maskCenter, backup,#surface_normal
                 mirrorGap, offset_distence, # mirror
                 reaction_type,  #reaction 
                 param, n, celllength, kdtreeN,filmKDTree,weightDepo,weightEtching,
                 tstep, substrateTop, posGeneratorType, logname):
        # super().__init__(tstep, pressure_pa, temperature, cellSize, celllength, chamberSize)
        surface_normal.__init__(self, center_with_direction, range3D, InOrOut,celllength, tstep, yield_hist,\
                                maskTop, maskBottom, maskStep, maskCenter, backup, density, mirrorGap, offset_distence)
        self.param = param # n beta
        self.kdtreeN = kdtreeN
        self.celllength = celllength
        self.timeStep = tstep
        # self.sub_x = sub_xy[0]
        # self.sub_y = sub_xy[1]
        # self.substrate = film
        # self.depo_or_etching = depo_or_etching
        self.depoPoint = depoPoint
        self.etchingPoint = etchingPoint
        self.density = density
        self.inputMethod = inputMethod
        self.n = n
        self.T = 300
        self.Cm = (2*1.380649e-23*self.T/(27*1.66e-27) )**0.5 # (2kT/m)**0.5 27 for the Al

        # self.film = film
        self.filmKDTree = filmKDTree
        self.weightDepo = weightDepo
        self.weightEtching = weightEtching
        # filmKDTree=np.array([[2, 0], [3, 1]])
        #       KDTree    [depo_parcel,  film]
        self.mirrorGap = mirrorGap
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
        current_plane = filmC >= 0

        # 创建一个全是False的布尔数组来存储邻居的检查结果
        neighbors = np.zeros_like(filmC, dtype=bool)

        # 检查各个方向的邻居是否为零
        neighbors[1:, :, :] |= filmC[:-1, :, :] >= 1  # 上面的邻居不为0
        neighbors[:-1, :, :] |= filmC[1:, :, :] >= 1  # 下面的邻居不为0
        neighbors[:, 1:, :] |= filmC[:, :-1, :] >= 1  # 左边的邻居不为0
        neighbors[:, :-1, :] |= filmC[:, 1:, :] >= 1  # 右边的邻居不为0
        neighbors[:, :, 1:] |= filmC[:, :, :-1] >= 1  # 前面的邻居不为0
        neighbors[:, :, :-1] |= filmC[:, :, 1:] >= 1  # 后面的邻居不为0

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

    # def scanDepoFloat(self, type): # fast scanZ
    #     film = torch.Tensor(self.film)
    #     sumFilm = torch.sum(film, axis=-1)
    #     # sumFilm = torch.Tensor(sumFilm)

    #     filmC = film[:,:,:,type[1]]
    #     # 初始化一个全零的表面稀疏张量
    #     surface_sparse = torch.zeros_like(sumFilm)
    #     surface_Float = torch.zeros_like(sumFilm)

    #     # 获取当前平面与前后平面的布尔索引
    #     current_plane = sumFilm == 0
    #     current_Float = torch.logical_and(film[:,:,:,type[1]] > 0, film[:,:,:,type[1]] < 1)
    #     # print(current_Float)
    #     # 获取周围邻居的布尔索引
    #     neighbors_plane = torch.zeros_like(filmC, dtype=torch.bool)
        
    #     neighbors_plane[1:, :, :] |= filmC[:-1, :, :] >= 9  # 上面
    #     neighbors_plane[:-1, :, :] |= filmC[1:, :, :] >= 9  # 下面
    #     neighbors_plane[:, 1:, :] |= filmC[:, :-1, :] >= 9  # 左边
    #     neighbors_plane[:, :-1, :] |= filmC[:, 1:, :] >= 9  # 右边
    #     neighbors_plane[:, :, 1:] |= filmC[:, :, :-1] >= 9  # 前面
    #     neighbors_plane[:, :, :-1] |= filmC[:, :, 1:] >= 9  # 后面

    #     # 获取周围邻居的布尔索引
    #     neighbors_float = torch.zeros_like(filmC, dtype=torch.bool)
        
    #     neighbors_float[1:, :, :] |= filmC[:-1, :, :] <= 0  # 上面
    #     neighbors_float[:-1, :, :] |= filmC[1:, :, :] <= 0  # 下面
    #     neighbors_float[:, 1:, :] |= filmC[:, :-1, :] <= 0  # 左边
    #     neighbors_float[:, :-1, :] |= filmC[:, 1:, :] <= 0  # 右边
    #     neighbors_float[:, :, 1:] |= filmC[:, :, :-1] <= 0  # 前面
    #     neighbors_float[:, :, :-1] |= filmC[:, :, 1:] <= 0  # 后面
        
    #     # 获取满足条件的索引
    #     condition = current_plane & neighbors_plane
    #     condition_float = current_Float & neighbors_float
    #     # 更新表面稀疏张量
    #     surface_sparse[condition] = 1
    #     surface_Float[condition_float] = 1

    #     points = surface_Float.to_sparse().indices().T

    #     return surface_sparse.numpy(), points

    def scanDepoFloat(self, type):
        # 将 self.film 转换为 numpy 数组
        film = np.array(self.film)
        sumFilm = np.sum(film, axis=-1)

        filmC = film[:, :, :, type[1]]

        # 初始化全零的表面稀疏数组
        surface_sparse = np.zeros_like(sumFilm)
        surface_Float = np.zeros_like(sumFilm)

        # 获取当前平面与前后平面的布尔索引
        current_plane = sumFilm == 0
        current_Float = (film[:, :, :, type[1]] > 0) & (film[:, :, :, type[1]] < 1)

        # 获取周围邻居的布尔索引
        neighbors_plane = np.zeros_like(filmC, dtype=bool)
        neighbors_float = np.zeros_like(filmC, dtype=bool)
        
        # 设置布尔掩码，检查邻居元素
        neighbors_plane[1:, :, :] |= filmC[:-1, :, :] >= 9  # 上面
        neighbors_plane[:-1, :, :] |= filmC[1:, :, :] >= 9  # 下面
        neighbors_plane[:, 1:, :] |= filmC[:, :-1, :] >= 9  # 左边
        neighbors_plane[:, :-1, :] |= filmC[:, 1:, :] >= 9  # 右边
        neighbors_plane[:, :, 1:] |= filmC[:, :, :-1] >= 9  # 前面
        neighbors_plane[:, :, :-1] |= filmC[:, :, 1:] >= 9  # 后面

        # 邻居小于等于0的情况
        neighbors_float[1:, :, :] |= filmC[:-1, :, :] <= 0
        neighbors_float[:-1, :, :] |= filmC[1:, :, :] <= 0
        neighbors_float[:, 1:, :] |= filmC[:, :-1, :] <= 0
        neighbors_float[:, :-1, :] |= filmC[:, 1:, :] <= 0
        neighbors_float[:, :, 1:] |= filmC[:, :, :-1] <= 0
        neighbors_float[:, :, :-1] |= filmC[:, :, 1:] <= 0

        # 获取满足条件的索引
        condition = current_plane & neighbors_plane
        condition_float = current_Float & neighbors_float

        # 更新表面稀疏张量
        surface_sparse[condition] = 1
        surface_Float[condition_float] = 1

        # 获取满足条件的点云
        points = np.array(np.nonzero(surface_Float)).T

        return surface_sparse, points


    def depoFloat(self, type):
        # plane_point, Float_point, points_float_depo = self.scanDepoFloat()
        plane_point, Float_point = self.scanDepoFloat(type)

        plane_tree = cKDTree(np.argwhere(plane_point == 1))

        dd, ii = plane_tree.query(Float_point, k=1, workers=1)

        surface_indice = np.argwhere(plane_point)
        i1 = surface_indice[ii][:,0] #[particle, order, xyz]
        j1 = surface_indice[ii][:,1]
        k1 = surface_indice[ii][:,2]

        # Float_point = Float_point.numpy()
        self.film[i1, j1, k1, type[1]] += self.film[Float_point[:, 0],Float_point[:, 1],Float_point[:, 2],type[1]]
        self.film[Float_point[:, 0],Float_point[:, 1],Float_point[:, 2],type[1]] = 0
        # self.film[Float_point[:, 0],Float_point[:, 1],Float_point[:, 2],1] = 0
        # return film


    def etching_film(self):
        i, j, k = self.get_indices()
        # self.sumFilm = np.sum(self.film, axis=-1)
        # indice_inject = np.array(sumFilm[i, j, k] != 0) # etching
        indice_inject = np.array(self.sumFilm[i, j, k] >= 1) # depo
        reactListAll = np.ones(indice_inject.shape[0])*-2
        oscilationList = np.zeros_like(indice_inject, dtype=np.bool_)

        if np.any(indice_inject):
            pos_1, vel_1, weight_1 = self.get_positions_velocities_weight(indice_inject)
            get_plane, etch_yield, get_theta, ddshape, maxdd, ddi, dl1, oscilation_indice = self.calculate_injection(pos_1, vel_1)

            film_update_results = self.update_film(get_plane, get_theta, indice_inject, ddi, dl1, ddshape, maxdd)

            self.handle_surface_depo(film_update_results, etch_yield, get_theta, pos_1, vel_1, weight_1, indice_inject, reactListAll, oscilationList, film_update_results['reactList'], oscilation_indice)

            return film_update_results['depo_count'], ddshape, maxdd, ddi, dl1
        else:
            return 0, 0, 0, 0, 0

    def get_indices(self):
        # 直接将切片操作和数据类型转换合并
        return self.parcel[:, 6].astype(int), self.parcel[:, 7].astype(int), self.parcel[:, 8].astype(int)

    def get_positions_velocities_weight(self, indice_inject):
        # 直接返回切片
        return self.parcel[indice_inject, :3], self.parcel[indice_inject, 3:6], self.parcel[indice_inject, 9]

    def calculate_injection(self, pos_1, vel_1):
        # self.planes = self.get_pointcloud(sumFilm)
        get_plane, etch_yield, get_theta, ddshape, maxdd, ddi, dl1, pos1e4, vel1e4, oscilation_indice = self.get_inject_normal(self.planes, pos_1, vel_1)
        return get_plane, etch_yield, get_theta, ddshape, maxdd, ddi, dl1, oscilation_indice

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
        inKDtree = np.argwhere(self.surface_etching_mirror == True) * self.celllength
        return KDTree(inKDtree)

    def handle_surface_depo(self, film_update_results, etch_yield, get_theta, pos_1, vel_1, weight_1, indice_inject, reactListAll, oscilationList, reactList, oscilation_indice):
        depo_parcel = film_update_results['depo_parcel']

        reactListAll[indice_inject] = reactList
        oscilationList[indice_inject] = oscilation_indice

        if np.any(reactListAll != -1):
            indice_inject[np.where(reactListAll == -1)] = False
            indice_inject[oscilationList == True] = False
            self.parcel = self.parcel[~indice_inject]

        for type in self.filmKDTree:
            # Check if the depo_parcel matches the current type
            react_classify = depo_parcel == type[0]
            # to_depo = np.where(depo_parcel == type[0])[0]
            if np.any(react_classify):
                
                # Process surface deposition
                self.process_surface_depo(type)

                # Build surface KDTree
                # surface_tree = cKDTree(np.argwhere(self.surface_etching_mirror == True) * self.celllength)
                surface_tree = self.toKDtree()
                
                # Query the KDTree for neighbors
                ii, dd, surface_indice = self.query_surface_tree(surface_tree, pos_1, react_classify)

                # Distribute deposition
                self.distribute_depo(surface_indice, ii, dd, type, etch_yield[react_classify], pos_1[react_classify], vel_1[react_classify], weight_1[react_classify], get_theta[react_classify])

                # Handle deposition or etching
                self.handle_deposition_or_etching(type)

        small_weight = self.parcel[:, 9] < 0.1
        self.parcel = self.parcel[~small_weight]
        # reactListAll[indice_inject] = reactList
        # oscilationList[indice_inject] = oscilation_indice

        # if np.any(reactListAll != -1):
        #     indice_inject[np.where(reactListAll == -1)] = False
        #     indice_inject[oscilationList == True] = False
        #     self.parcel = self.parcel[~indice_inject]

    def process_surface_depo(self, type):
        # Generate surface deposition mask
        if type[2] == -1:
            # surface_etching = np.array(self.film[:, :, :, type[1]] > 1) # etching
            surface_etching = np.array(self.film[:, :, :, type[1]] > 0) # etching
        elif type[2] == 1:
            surface_etching = np.logical_or(self.film[:, :, :, type[1]] == 0, self.film[:, :, :, type[1]] != self.density) #depo
        self.update_surface_mirror(surface_etching)
        # return surface_etching

    def query_surface_tree(self, surface_tree, pos_1, to_depo):
        # Adjust positions for mirror and query nearest neighbors
        # to_depo = np.where(depo_parcel == type[0])[0]
        pos_mirror = np.copy(pos_1)
        pos_mirror[:, 0] += self.mirrorGap * self.celllength
        pos_mirror[:, 1] += self.mirrorGap * self.celllength
        dd, ii = surface_tree.query(pos_mirror[to_depo], k=self.kdtreeN)
        surface_indice = np.argwhere(self.surface_etching_mirror == True)
        return ii, dd, surface_indice

    def handle_deposition_or_etching(self, type):
        surface_film_depo = np.logical_and(self.film[:, :, :, type[1]] > 1, self.film[:,:,:,type[1]] < 2)
        self.film[surface_film_depo, type[1]] = self.density

        surface_film_etching = np.logical_and(self.film[:,:,:,type[1]] < 9, self.film[:,:,:,type[1]] > 8)
        self.film[surface_film_etching, type[1]] = 0

        if np.any(surface_film_depo) or np.any(surface_film_etching):
            self.sumFilm = np.sum(self.film, axis=-1)
            self.update_surface_mirror_noetching(self.sumFilm)
            self.planes = self.get_pointcloud(self.surface_mirror)

        if np.any(surface_film_etching):
            self.depoFloat(type)

    def update_surface_mirror_noetching(self, surface_etching):
        self.surface_mirror[self.mirrorGap:self.mirrorGap+self.cellSizeX, self.mirrorGap:self.mirrorGap+self.cellSizeY, :] = surface_etching
        self.surface_mirror[:self.mirrorGap, self.mirrorGap:self.mirrorGap+self.cellSizeY, :] = surface_etching[-self.mirrorGap:, :, :]
        self.surface_mirror[-self.mirrorGap:, self.mirrorGap:self.mirrorGap+self.cellSizeY, :] = surface_etching[:self.mirrorGap, :, :]
        self.surface_mirror[self.mirrorGap:self.mirrorGap+self.cellSizeX, :self.mirrorGap, :] = surface_etching[:, -self.mirrorGap:, :]
        self.surface_mirror[self.mirrorGap:self.mirrorGap+self.cellSizeX:, -self.mirrorGap:, :] = surface_etching[:, :self.mirrorGap, :]
        self.surface_mirror[:self.mirrorGap, :self.mirrorGap, :] = surface_etching[-self.mirrorGap:, -self.mirrorGap:, :]
        self.surface_mirror[:self.mirrorGap, -self.mirrorGap:, :] = surface_etching[-self.mirrorGap:, :self.mirrorGap, :]
        self.surface_mirror[-self.mirrorGap:, :self.mirrorGap, :] = surface_etching[:self.mirrorGap, -self.mirrorGap:, :]
        self.surface_mirror[-self.mirrorGap:, -self.mirrorGap:, :] = surface_etching[:self.mirrorGap, :self.mirrorGap, :]

    def update_surface_mirror(self, surface_etching):
        self.surface_etching_mirror[self.mirrorGap:self.mirrorGap+self.cellSizeX, self.mirrorGap:self.mirrorGap+self.cellSizeY, :] = surface_etching
        self.surface_etching_mirror[:self.mirrorGap, self.mirrorGap:self.mirrorGap+self.cellSizeY, :] = surface_etching[-self.mirrorGap:, :, :]
        self.surface_etching_mirror[-self.mirrorGap:, self.mirrorGap:self.mirrorGap+self.cellSizeY, :] = surface_etching[:self.mirrorGap, :, :]
        self.surface_etching_mirror[self.mirrorGap:self.mirrorGap+self.cellSizeX, :self.mirrorGap, :] = surface_etching[:, -self.mirrorGap:, :]
        self.surface_etching_mirror[self.mirrorGap:self.mirrorGap+self.cellSizeX:, -self.mirrorGap:, :] = surface_etching[:, :self.mirrorGap, :]
        self.surface_etching_mirror[:self.mirrorGap, :self.mirrorGap, :] = surface_etching[-self.mirrorGap:, -self.mirrorGap:, :]
        self.surface_etching_mirror[:self.mirrorGap, -self.mirrorGap:, :] = surface_etching[-self.mirrorGap:, :self.mirrorGap, :]
        self.surface_etching_mirror[-self.mirrorGap:, :self.mirrorGap, :] = surface_etching[:self.mirrorGap, -self.mirrorGap:, :]
        self.surface_etching_mirror[-self.mirrorGap:, -self.mirrorGap:, :] = surface_etching[:self.mirrorGap, :self.mirrorGap, :]

    def distribute_depo(self, surface_indice, ii, dd, type, etch_yield, pos, vel, weight, normal):
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

            if type[2] == 1:
                self.film[i1, j1, k1, type[1]] += weight * dd[:, kdi] / ddsum # depo
            elif type[2] == -1:
                # self.film[i1, j1, k1, type[1]] -= 0 * etch_yield * dd[:, kdi] / ddsum  # etching
                self.film[i1, j1, k1, type[1]] -= weight * etch_yield * dd[:, kdi] / ddsum  # etching
                self.redepo_Generator(pos, vel, normal, weight * etch_yield)

    def redepo_Generator(self, pos, vel, normal, weight):
        # pos[:,0] -= 2*self.mirrorGap*self.celllength
        # pos[:,1] -= 2*self.mirrorGap*self.celllength
        vels = reemission_multi(vel, normal)
        typeID = np.zeros(vel.shape[0]) # 0 for Al depo
        pos += vels*self.timeStep*5
        self.Parcelgen(pos, vels, weight, typeID)

    # def redepo_Generator(self, i, j, k, vel, normal):
    #     # poses = np.array([i, j, k]).T
    #     # vels = reemission_multi(vel, normal)
    #     # typeID = np.zeros(vel.shape[0]) # 0 for Al depo
    #     poses, vels, typeID = redepo_Generator_numba(i, j, k, vel, normal)
    #     self.Parcelgen(poses, vels, typeID)

    def cleanMinusFilm(self):
        indice = self.film[:, :, :, 0] < 0
        self.film[indice, 0] = 0


    def correct_indices(self, i1, j1):
        i1[i1 >= self.cellSizeX] -= self.cellSizeX
        i1[i1 < 0] += self.cellSizeX
        j1[j1 >= self.cellSizeY] -= self.cellSizeY
        j1[j1 < 0] += self.cellSizeY
        return i1, j1

    def toboundary(self):
        # self.parcel = boundary(self.parcel, self.cellSizeX, self.cellSizeY, self.cellSizeZ, self.celllength)
        self.parcel = boundaryNumba(self.parcel, self.cellSizeX, self.cellSizeY, self.cellSizeZ, self.celllength)

    def toupdate_parcel(self, tStep):
        self.parcel = update_parcel(self.parcel, self.celllength, tStep)

    def getAcc_depo(self, tStep):

        self.toboundary()

        # self.removeFloat()
        self.removeFloatPolymer()

        depo_count, ddshape, maxdd, ddi, dl1 = self.etching_film()

        self.toupdate_parcel(tStep)

        return depo_count, ddshape, maxdd, ddi, dl1 #, film_max, surface_true

    # particle data struction np.array([posX, posY, posZ, velX, velY, velZ, i, j, k, weight, typeID])
    def Parcelgen(self, pos, vel, weight, typeID):

        # i = np.floor((pos[:, 0]/self.celllength) + 0.5).astype(int)
        # j = np.floor((pos[:, 1]/self.celllength) + 0.5).astype(int)
        # k = np.floor((pos[:, 2]/self.celllength) + 0.5).astype(int)
        i = np.floor((pos[:, 0]/self.celllength)).astype(int)
        j = np.floor((pos[:, 1]/self.celllength)).astype(int)
        k = np.floor((pos[:, 2]/self.celllength)).astype(int)
        # parcelIn = np.zeros((pos.shape[0], 10), order='F')
        parcelIn = np.zeros((pos.shape[0], 11))
        parcelIn[:, :3] = pos
        parcelIn[:, 3:6] = vel
        parcelIn[:, 6] = i
        parcelIn[:, 7] = j
        parcelIn[:, 8] = k
        parcelIn[:, 9] = weight
        parcelIn[:, 10] = typeID

        # print(self.parcel.flags.f_contiguous)
        self.parcel = np.concatenate((self.parcel, parcelIn))
        # print(self.parcel.flags)

    def posvelGenerator(self, velGeneratorType):
        # posGeneratorType
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

        # velGeneratorType
        if velGeneratorType == 'maxwell':
            self.log.info('using velGenerator_maxwell')
            velGenerator = self.velGenerator_maxwell_normal
        elif velGeneratorType == 'updown':
            self.log.info('using velGenerator_updown')
            velGenerator = self.velGenerator_updown_normal      
        elif velGeneratorType == 'benchmark':
            self.log.info('using velGenerator_benchmark')
            velGenerator = self.velGenerator_benchmark_normal   
        elif velGeneratorType == 'input':
            self.log.info('using velGenerator_benchmark')
            velGenerator = self.velGenerator_input_normal  
        return posGenerator, velGenerator  

    def runEtch(self, velGeneratorType, typeID, inputCount, runningCount, max_react_count, emptyZ, step):

        self.log.info('inputType:{}'.format(typeID))
        # if step == 0:
        #     self.parcel = np.zeros((1, 10))
        # tmax = time
        start_time = Time.time()
        tstep = self.timeStep
        t = 0
        # inputCount = int(v0.shape[0]/(tmax/tstep))
        self.sumFilm = np.sum(self.film, axis=-1)
        self.update_surface_mirror_noetching(self.sumFilm)
        self.planes = self.get_pointcloud(self.surface_mirror)
        count_reaction = 0
        inputAll = 0
        filmThickness = self.substrateTop

        posGenerator, velGenerator = self.posvelGenerator(velGeneratorType)

        p1 = posGenerator(inputCount, filmThickness, emptyZ)
        v1 = velGenerator(inputCount)
        typeIDIn = np.zeros(inputCount)
        typeIDIn[:] = typeID
        if self.depo_or_etching == 'depo':
            self.Parcelgen(p1, v1, self.weightDepo, typeIDIn)
        elif self.depo_or_etching == 'etching':
            self.Parcelgen(p1, v1, self.weightEtching, typeIDIn)
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
                
                if self.depo_or_etching == 'depo' and self.depoPoint[2] <= filmThickness and depo_count < 1 and self.parcel.shape[0] < 2000:
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
                weightMin = np.min(self.parcel[:,9])
                weightMax = np.max(self.parcel[:,9])
                # if self.inputMethod == 'bunch' and inputAll < max_react_count:
                if self.depo_or_etching == 'depo':
                    if self.parcel.shape[0] < runningCount and self.depoPoint[2] >= filmThickness and ti%3 == 0:
                        inputAll += inputCount
                        p1 = posGenerator(inputCount, filmThickness, emptyZ)
                        v1 = velGenerator(inputCount)
                        typeIDIn = np.zeros(inputCount)
                        typeIDIn[:] = typeID
                        self.Parcelgen(p1, v1, self.weightDepo, typeIDIn)
                elif self.parcel.shape[0] < runningCount and self.depo_or_etching == 'etching':
                    inputAll += inputCount
                    p1 = posGenerator(inputCount, filmThickness, emptyZ)
                    v1 = velGenerator(inputCount)
                    typeIDIn = np.zeros(inputCount)
                    typeIDIn[:] = typeID
                    self.Parcelgen(p1, v1, self.weightEtching, typeIDIn)

                # planes = self.get_pointcloud(np.sum(self.film, axis=-1))

                current_percentage = int(count_reaction / max_react_count * 100)  # 当前百分比
                if current_percentage > previous_percentage:
                    update_value = current_percentage - previous_percentage  # 计算进度差值
                    pbar.update(update_value)
                    previous_percentage = current_percentage  # 更新上一次的百分比

                gen_redepo = np.sum(self.parcel[:, -1] == 0)
                self.log.info('particleIn:{}, timeStep:{}, depo_count_step:{}, count_reaction_all:{},inputAll:{},vzMax:{:.3f},vzMin:{:.3f}, filmThickness:{}, input_count:{}, ddi:{}, dl1:{}, ddshape:{:.3f}, maxdd:{:.3f}, gen_redepo:{}, weightMax:{}, weightMin:{}'\
                            .format(previous_percentage, tstep, depo_count, count_reaction, inputAll,  vzMax, vzMin,  filmThickness, self.parcel.shape[0], ddi, dl1, ddshape, maxdd, gen_redepo, weightMax, weightMin))
            
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

                if ti%10 == 0:
                    self.removeFloat()
                    self.cleanMinusFilm()
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
     
    def posGenerator_benchmark(self, IN, thickness, emptyZ):
        position_matrix = np.array([np.random.rand(IN)*20 - 10 + self.cellSizeX/2 - 0.5, \
                                    np.random.rand(IN)*20 - 10 + self.cellSizeY/2 - 0.5, \
                                    np.ones(IN)*self.cellSizeZ/2]).T
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
        velosity_matrix[:, 0] = np.random.randn(IN)*0.001 - 0.0005 
        velosity_matrix[:, 1] = np.random.randn(IN)*0.001 - 0.0005
        velosity_matrix[:, 2] = -1 
        energy = np.linalg.norm(velosity_matrix, axis=1)
        velosity_matrix[:,0] = np.divide(velosity_matrix[:,0], energy)
        velosity_matrix[:,1] = np.divide(velosity_matrix[:,1], energy)
        velosity_matrix[:,2] = np.divide(velosity_matrix[:,2], energy)
        return velosity_matrix

    def velGenerator_benchmark_normal(self, IN):
        velosity_matrix = np.zeros((IN, 3))
        velosity_matrix[:, 0] = np.random.randn(IN)*0.01 - 0.005
        velosity_matrix[:, 1] = -np.sqrt(2)/2
        velosity_matrix[:, 2] = -np.sqrt(2)/2
        energy = np.linalg.norm(velosity_matrix, axis=1)
        velosity_matrix[:,0] = np.divide(velosity_matrix[:,0], energy)
        velosity_matrix[:,1] = np.divide(velosity_matrix[:,1], energy)
        velosity_matrix[:,2] = np.divide(velosity_matrix[:,2], energy)
        return velosity_matrix
    
    def velGenerator_input_normal(self, IN):

        velosity_matrix = np.random.default_rng().choice(self.vel_matrix, IN)

        return velosity_matrix
    # def posGenerator_benchmark(self, IN, thickness, emptyZ):
    #     position_matrix = np.array([np.random.rand(IN)*20 + self.cellSizeX/2, \
    #                                 np.random.rand(IN)*20 + self.cellSizeY/2, \
    #                                 np.ones(IN)*self.cellSizeZ - emptyZ]).T
    #     position_matrix *= self.celllength
    #     return position_matrix
    
    # def depo_position_increase(self, randomSeed, velosity_matrix, tmax, weight, Zgap):
    #     np.random.seed(randomSeed)
    #     weights = np.ones(velosity_matrix.shape[0])*weight
    #     result =  self.runEtch(velosity_matrix, tmax, self.film, weights, depoStep=1, emptyZ=Zgap)
    #     del self.log, self.fh
    #     return result
    
    #     # def runEtch(self, v0, typeID, time, emptyZ):
    # def depo_position_increase_cosVel_normal(self, randomSeed, N, tmax, Zgap):
    #     np.random.seed(randomSeed)
    #     for i in range(9):
    #         Random1 = np.random.rand(N)
    #         Random2 = np.random.rand(N)
    #         Random3 = np.random.rand(N)
    #         velosity_matrix = np.array([self.max_velocity_u(Random1, Random2), \
    #                                     self.max_velocity_w(Random1, Random2), \
    #                                         self.max_velocity_v(Random3)]).T

    #         energy = np.linalg.norm(velosity_matrix, axis=1)
    #         velosity_matrix[:,0] = np.divide(velosity_matrix[:,0], energy)
    #         velosity_matrix[:,1] = np.divide(velosity_matrix[:,1], energy)
    #         velosity_matrix[:,2] = np.divide(velosity_matrix[:,2], energy)

    #         typeID = np.zeros(N)
    #         # def runEtch(self, v0, typeID, time, emptyZ):
    #         result =  self.runEtch(velosity_matrix, typeID, tmax, emptyZ=Zgap)
    #         if np.any(result[0][self.depoPoint]) != 0:
    #             break             
    #     del self.log, self.fh
    #     return result
    

    def inputParticle(self, film, parcel, depo_or_etching, velGeneratorType, vel_matrix, typeID, inputCount, runningCount, max_react_count, depo_count_type, Zgap, step):
        self.depo_count_type = depo_count_type
        self.depo_or_etching = depo_or_etching
        self.film = film
        self.vel_matrix = vel_matrix
        self.parcel = parcel
        self.cellSizeX = self.film.shape[0]
        self.cellSizeY = self.film.shape[1]
        self.cellSizeZ = self.film.shape[2]
        self.surface_etching_mirror = np.zeros((self.cellSizeX+int(self.mirrorGap*2), self.cellSizeY+int(self.mirrorGap*2), self.cellSizeZ))
        self.surface_mirror = np.zeros((self.cellSizeX+int(self.mirrorGap*2), self.cellSizeY+int(self.mirrorGap*2), self.cellSizeZ))
        print(self.surface_etching_mirror.shape)
        self.log.info('circle step:{}'.format(step))
        result =  self.runEtch(velGeneratorType, typeID, inputCount,runningCount, max_react_count, Zgap, step)
        # if np.any(result[0][:, :, self.depoThick]) != 0:
        #     break             
        # del self.log, self.fh 
        return result  



if __name__ == "__main__":
    import pyvista as pv
    import cProfile


    def slide2D_fractionZ(film, start, end, direction, fraction, value):
        if fraction == '+':
            if direction == 'y':
                slit = np.linspace(0, np.abs(end[2] - start[2]), np.abs(end[1] - start[1]))
                fraction = np.abs(int(slit[0]-slit[1]))
                print('y', slit)
                print('fraction', fraction)
                for i in range(np.abs(end[1] - start[1])):
                    if end[1] > start[1]:
                        film[start[0]:end[0], start[1] + i, start[2]:start[2] + int(slit[i])] = value
                        for j in range(fraction):
                            film[start[0]:end[0], start[1] + i,start[2]+int(slit[i])+j] = 1/(fraction+1)*(fraction-j)
                    elif end[1] < start[1]:
                        film[start[0]:end[0], start[1] - i, start[2]:start[2] + int(slit[i])] = value
                        for j in range(fraction):
                            film[start[0]:end[0], start[1] - i,start[2]+int(slit[i])+j] = 1/(fraction+1)*(fraction-j)
            elif direction == 'x':
                slit = np.linspace(0, np.abs(end[2] - start[2]), np.abs(end[0] - start[0]))
                fraction = np.abs(int(slit[0]-slit[1]))
                print('x', slit)
                print('fraction', fraction)
                for i in range(np.abs(end[2] - start[2])):
                    if end[0] > start[0]:
                        film[start[0] + i, start[1]:end[1], start[2]:start[2] + int(slit[i])] = value
                        for j in range(fraction):
                            film[start[0] + i, start[1]:end[1], start[2]+int(slit[i])+j] = 1/(fraction+1)*(fraction-j)
                    elif end[0] < start[0]:
                        film[start[0] - i, start[1]:end[1], start[2]:start[2] + int(slit[i])] = value
                        for j in range(fraction):
                            film[start[0] - i, start[1]:end[1], start[2]+int(slit[i])+j] = 1/(fraction+1)*(fraction-j)
        elif fraction == '-':
            if direction == 'y':
                slit = np.linspace(0, np.abs(end[2] - start[2]), np.abs(end[1] - start[1]))
                fraction = np.abs(int(slit[0]-slit[1]))
                print('y', slit)
                print('fraction', fraction)
                for i in range(np.abs(end[1] - start[1])):
                    if end[1] > start[1]:
                        film[start[0]:end[0], start[1] + i, start[2] - int(slit[i]):start[2]+1] = value
                        for j in range(fraction):
                            film[start[0]:end[0], start[1] + i,start[2]-int(slit[i])-j] = 1/(fraction+1)*(fraction-j)
                    elif end[1] < start[1]:
                        film[start[0]:end[0], start[1] - i, start[2] - int(slit[i]):start[2]+1] = value
                        for j in range(fraction):
                            film[start[0]:end[0], start[1] - i,start[2]-int(slit[i])-j] = 1/(fraction+1)*(fraction-j)
            elif direction == 'x':
                slit = np.linspace(0, np.abs(end[2] - start[2]), np.abs(end[0] - start[0]))
                fraction = np.abs(int(slit[0]-slit[1]))
                print('x', slit)
                print('fraction', fraction)
                for i in range(np.abs(end[2] - start[2])):
                    if end[0] > start[0]:
                        film[start[0] + i, start[1]:end[1], start[2] - int(slit[i]):start[2]+1] = value
                        for j in range(fraction):
                            film[start[0] + i, start[1]:end[1], start[2] - int(slit[i]):start[2]] = 1/(fraction+1)*(fraction-j)
                    elif end[0] < start[0]:
                        film[start[0] - i, start[1]:end[1], start[2] - int(slit[i]):start[2]+1] = value
                        for j in range(fraction):
                            film[start[0] - i, start[1]:end[1], start[2] - int(slit[i]):start[2]] = 1/(fraction+1)*(fraction-j)
        return film

    film = np.zeros((70, 200, 150))

    bottom = 10
    # film[:, :, 0:bottom] = 10 # bottom

    height = 80
    left_side = 71
    right_side = 71

    slit = 8
    film[:, left_side+slit:200-right_side-slit, 0:height] = 10

    left_side_gap = 19
    right_side_gap = 181
    film[:, :left_side_gap, 0:height] = 10
    film[:, right_side_gap:, 0:height] = 10

    film = slide2D_fractionZ(film=film, start=[0, left_side, bottom], end=[70, left_side+slit, height], direction='y', fraction='+', value=10)
    film = slide2D_fractionZ(film=film, start=[0, 200-right_side-1, bottom], end=[70, 200-right_side-slit-1, height], direction='y', fraction='+', value=10)
    film = slide2D_fractionZ(film=film, start=[0, left_side_gap+slit-1, bottom], end=[70, left_side_gap-1, height], direction='y', fraction='+', value=10)
    film = slide2D_fractionZ(film=film, start=[0, right_side_gap-slit, bottom], end=[70, right_side_gap, height], direction='y', fraction='+', value=10)

    # film[:, 80:121, 0:31] = 10

    film[:, :, 0:bottom] = 10 # bottom
    film[:, :, height:] = 0 # bottom

    yield_hist = np.array([[1.0, 1.01, 1.05,  1.2,  1.4,  1.5, 1.07, 0.65, 0.28, 0.08,  0, \
                            0.08, 0.28,0.65,  1.07, 1.5, 1.4, 1.2, 1.05, 1.01, 1.0 ], \
                            [  0,  5,   10,   20,   30,   40,   50,   60,   70,   80, 90, \
                            100, 110, 120, 130, 140, 150, 160, 170, 175, 180]])
    yield_hist[1] *= np.pi/180

    etchfilm = np.zeros((70, 200, 150, 2))
    etchfilm[:, :, :, 0] = film
    # etchfilm[:, :, :, 1] = film

    center = 100



    # ----------------------------------------------------------------------------------------------

    logname = 'Multi_species_benchmark_1021_hole_ratio01'
    inputMethod='bunch'
    etchingPoint = np.array([center, center, 125])
    depoPoint = np.array([center, center, 125])
    density = 10
    center_with_direction=np.array([[int(etchfilm.shape[0]/2),int(etchfilm.shape[1]/2),150]])
    range3D=np.array([[0, etchfilm.shape[0], 0, etchfilm.shape[1], 0, etchfilm.shape[2]]])
    InOrOut=[1]
    # yield_hist=np.array([None])
    yield_hist = yield_hist
    maskTop=40, 
    maskBottom=98, 
    maskStep=10, 
    maskCenter=[int(etchfilm.shape[0]/2), int(etchfilm.shape[1]/2)]
    backup=False
    mirrorGap=5
    reaction_type=False
    param = [1.6, -0.7]
    n=1
    celllength=1e-5
    kdtreeN=5
    filmKDTree=np.array([[2, 0, 1], [3, 0, -1]]) # 1 for depo -1 for etching
    # filmKDTree=np.array([[2, 1], [3, 1]])
    weightDepo=0.2
    weightEtching = 0.2
    tstep=1e-5
    substrateTop=80
    posGeneratorType='top'
    testEtch = etching(
                        inputMethod,
                        etchingPoint,depoPoint,
                        density, center_with_direction, 
                        range3D, InOrOut, yield_hist,
                        maskTop, maskBottom, maskStep, maskCenter,backup, 
                        mirrorGap,
                        reaction_type, param,n,
                        celllength, kdtreeN, filmKDTree,weightDepo,weightEtching, tstep,
                        substrateTop,posGeneratorType, logname)
    

    cicle = 100
    celllength=1e-5
    parcel = np.array([[95*celllength, 95*celllength, 159*celllength, 0, 0, 1, 95, 95, 159, 0.2, 0]])
    step1 = testEtch.inputParticle(etchfilm, parcel, 'depo', 'maxwell', 0, 0, int(5e3), int(1e6), int(1e5),2, 4, 100)

    # np.save('./bosch_data_1117_timeit/bosch_sf_step_test_Ar', etchfilm)

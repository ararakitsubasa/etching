# import numpy as np
# cimport numpy as cnp
# from cython.parallel import prange

# def boundary(cnp.ndarray[double, ndim=2] parcel,
#              double cellSizeX, double cellSizeY, double cellSizeZ,
#              double celllength):
#     # 获取数组大小
#     cdef Py_ssize_t n = parcel.shape[0]
#     cdef double *p_parcel = &parcel[0, 0]
#     cdef int i, j

#     # 创建一个临时数组用于存储符合边界条件的粒子数据
#     cdef double[:, ::1] temp_parcel = np.empty((n, 10), dtype=np.float64)
#     cdef int valid_count = 0  # 记录有效粒子的数量

#     # 在 GIL 保护下进行并行处理
#     for i in range(n):
#         # 获取当前粒子的 x, y, z 位置
#         x = p_parcel[i * 10 + 6]
#         y = p_parcel[i * 10 + 7]
#         z = p_parcel[i * 10 + 8]

#         # X方向边界条件
#         if x >= cellSizeX:
#             p_parcel[i * 10 + 6] -= cellSizeX
#             p_parcel[i * 10] -= celllength * cellSizeX
#         elif x < 0:
#             p_parcel[i * 10 + 6] += cellSizeX
#             p_parcel[i * 10] += celllength * cellSizeX

#         # Y方向边界条件
#         if y >= cellSizeY:
#             p_parcel[i * 10 + 7] -= cellSizeY
#             p_parcel[i * 10 + 1] -= celllength * cellSizeY
#         elif y < 0:
#             p_parcel[i * 10 + 7] += cellSizeY
#             p_parcel[i * 10 + 1] += celllength * cellSizeY

#         # Z方向边界条件：如果粒子在Z方向上超出边界，则跳过
#         if z >= cellSizeZ or z < 0:
#             continue

#         # 在 GIL 保护下安全地更新有效粒子数组
#         with gil:
#             for j in range(10):
#                 temp_parcel[valid_count, j] = p_parcel[i * 10 + j]
#             valid_count += 1  # 增加有效粒子计数

#     # 将符合边界条件的粒子返回，并转为 Fortran 格式的数组
#     return np.asfortranarray(temp_parcel[:valid_count])

import numpy as np
cimport numpy as cnp

def boundary(cnp.ndarray[double, ndim=2] parcel,
             double cellSizeX, double cellSizeY, double cellSizeZ,
             double celllength):
    # 获取数组大小
    cdef Py_ssize_t n = parcel.shape[0]
    cdef Py_ssize_t valid_count = 0  # 计数器，用于记录有效粒子的数量

    # 获取数组指针
    cdef double *p_parcel = &parcel[0, 0]
    cdef int i
    cdef double x, y, z

    # 创建一个临时数组用于存储符合边界条件的粒子数据
    cdef double[:, ::1] temp_parcel = np.empty((n, 10), dtype=np.float64)

    # 遍历每个粒子
    for i in range(n):
        # 获取当前粒子的 x, y, z 位置
        x = p_parcel[i * 10 + 6]
        y = p_parcel[i * 10 + 7]
        z = p_parcel[i * 10 + 8]

        # X方向边界条件
        if x >= cellSizeX:
            p_parcel[i * 10 + 6] -= cellSizeX
            p_parcel[i * 10] -= celllength * cellSizeX
        elif x < 0:
            p_parcel[i * 10 + 6] += cellSizeX
            p_parcel[i * 10] += celllength * cellSizeX

        # Y方向边界条件
        if y >= cellSizeY:
            p_parcel[i * 10 + 7] -= cellSizeY
            p_parcel[i * 10 + 1] -= celllength * cellSizeY
        elif y < 0:
            p_parcel[i * 10 + 7] += cellSizeY
            p_parcel[i * 10 + 1] += celllength * cellSizeY

        # Z方向边界条件：如果粒子在Z方向上超出边界，则跳过
        if z >= cellSizeZ or z < 0:
            continue

        # 将符合边界条件的粒子复制到临时数组中
        for j in range(10):
            temp_parcel[valid_count, j] = p_parcel[i * 10 + j]

        valid_count += 1  # 增加有效粒子计数

    # 将符合边界条件的粒子复制回原数组，并返回Fortran布局的数组
    return np.asfortranarray(temp_parcel[:valid_count])
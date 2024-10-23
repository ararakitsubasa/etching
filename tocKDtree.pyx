import numpy as np
cimport numpy as cnp
from scipy.spatial import cKDTree
from cython.parallel import prange

cdef class YourClass:  # 替换为你的类名
    cdef double celllength
    cnp.ndarray[cnp.bool_t, ndim=2] surface_depo_mirror

    def toKDtree(self):
        cdef double celllength_local = self.celllength  # 存储为局部变量
        cdef int n = np.count_nonzero(self.surface_depo_mirror)  # 计算有效点的数量
        cdef np.ndarray[double, ndim=2] coords = np.empty((n, 2), dtype=np.double)  # 预分配存储坐标的数组
        cdef int idx = 0

        # 使用 NumPy 的有效索引获取坐标
        for i in range(self.surface_depo_mirror.shape[0]):
            for j in range(self.surface_depo_mirror.shape[1]):
                if self.surface_depo_mirror[i, j]:  # 检查是否为 True
                    coords[idx, 0] = i * celllength_local  # 第一个维度
                    coords[idx, 1] = j * celllength_local  # 第二个维度
                    idx += 1

        return cKDTree(coords)

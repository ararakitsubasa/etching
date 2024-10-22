import numpy as np
cimport numpy as cnp
from cython.parallel import prange

def boundary(cnp.ndarray[double, ndim=2] parcel,
             double cellSizeX, double cellSizeY, double cellSizeZ,
             double celllength):
    cdef Py_ssize_t n = parcel.shape[0]
    cdef double *p_parcel = &parcel[0, 0]
    cdef int i, valid_count = 0

    # 创建一个临时数组来存储有效粒子的索引
    cdef int[:] valid_indices = np.zeros(n, dtype=np.int32)

    # 使用并行循环
    with nogil:
        for i in range(n):
            x = p_parcel[i * 10 + 6]
            y = p_parcel[i * 10 + 7]
            z = p_parcel[i * 10 + 8]

            # 判断粒子是否在边界外
            if x < 0 or x >= cellSizeX or y < 0 or y >= cellSizeY or z < 0 or z >= cellSizeZ:
                # 如果粒子在边界外，处理它
                if x >= cellSizeX:
                    p_parcel[i * 10 + 6] -= cellSizeX
                    p_parcel[i * 10] -= celllength * cellSizeX
                elif x < 0:
                    p_parcel[i * 10 + 6] += cellSizeX
                    p_parcel[i * 10] += celllength * cellSizeX

                if y >= cellSizeY:
                    p_parcel[i * 10 + 7] -= cellSizeY
                    p_parcel[i * 10 + 1] -= celllength * cellSizeY
                elif y < 0:
                    p_parcel[i * 10 + 7] += cellSizeY
                    p_parcel[i * 10 + 1] += celllength * cellSizeY

                if z >= cellSizeZ:
                    p_parcel[i * 10 + 8] -= cellSizeZ
                elif z < 0:
                    p_parcel[i * 10 + 8] += cellSizeZ
            else:
                # 记录有效粒子的索引
                valid_indices[valid_count] = i
                valid_count += 1

    # 处理有效粒子并返回结果
    return np.asfortranarray(parcel[valid_indices[:valid_count]])


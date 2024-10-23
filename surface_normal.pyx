import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy import interpolate
from libc.math cimport sqrt, pi
import cython

cimport numpy as cnp
np.import_array()

cdef class SurfaceNormal:
    cdef double[:, :] center_with_direction
    cdef double[:, :] range3D
    cdef double celllength
    cdef double tstep
    cdef int[:, :] InOrOut
    cdef double[:, :] yield_hist
    cdef object yield_func
    cdef double maskTop
    cdef double maskBottom
    cdef double maskStep
    cdef double[:, :] maskCenter
    cdef bint backup
    cdef int zshape

    def __init__(self, double[:, :] center_with_direction, double[:, :] range3D, int[:, :] InOrOut,
                 double celllength, double tstep, double[:, :] yield_hist,
                 double maskTop, double maskBottom, double maskStep,
                 double[:, :] maskCenter, bint backup):
        self.center_with_direction = center_with_direction
        self.range3D = range3D 
        self.celllength = celllength
        self.tstep = tstep
        self.InOrOut = InOrOut 
        self.maskTop = maskTop
        self.maskBottom = maskBottom
        self.maskStep = maskStep
        self.maskCenter = maskCenter
        self.backup = backup

        if yield_hist.size == 0:
            self.yield_hist = np.array([[1.0, 1.05, 1.2, 1.4, 1.5, 1.07, 0.65, 0.28, 0.08, 0],
                                         [0, pi/18, pi/9, pi/6, 2*pi/9, 5*pi/18, pi/3, 7*pi/18, 4*pi/9, pi/2]])
        else:
            self.yield_hist = yield_hist
        
        self.yield_func = interpolate.interp1d(self.yield_hist[1], self.yield_hist[0], kind='quadratic')

    def scanZ(self, film):
        cdef int xshape, yshape, zshape
        xshape, yshape, zshape = film.shape
        self.zshape = zshape
        
        surface_sparse = torch.zeros((xshape, yshape, zshape))
        current_plane = film != 0

        neighbors = torch.zeros_like(film, dtype=torch.bool)
        neighbors[1:, :, :] |= film[:-1, :, :] == 0
        neighbors[:-1, :, :] |= film[1:, :, :] == 0
        neighbors[:, 1:, :] |= film[:, :-1, :] == 0
        neighbors[:, :-1, :] |= film[:, 1:, :] == 0
        neighbors[:, :, 1:] |= film[:, :, :-1] == 0
        neighbors[:, :, :-1] |= film[:, :, 1:] == 0
        
        condition = current_plane & neighbors
        
        surface_sparse[condition] = 1
        
        return surface_sparse.to_sparse()

    def normalconsistency_3D_real(self, double[:, :] planes):
        cdef int nbnormals = planes.shape[0]
        cdef list planes_consist = []
        cdef double[::1] sensorcentre = np.zeros(3, dtype=np.float64)

        for c in range(self.center_with_direction.shape[0]):
            sensorcentre[0] = self.center_with_direction[c, 0]
            sensorcentre[1] = self.center_with_direction[c, 1]
            sensorcentre[2] = self.center_with_direction[c, 2]
            sensorrange = self.range3D[c]
            sensorInOut = self.InOrOut[c]

            # Determine which planes are in range
            in_range_mask = (
                (planes[:, 3] < sensorrange[0]) & (planes[:, 3] >= sensorrange[1]) &
                (planes[:, 4] < sensorrange[2]) & (planes[:, 4] >= sensorrange[3]) &
                (planes[:, 5] < sensorrange[4]) & (planes[:, 5] >= sensorrange[5])
            )

            planes_in_range = planes[~in_range_mask]
            nbnormals_in_range = planes_in_range.shape[0]
            planes_in_range_consist = np.zeros((nbnormals_in_range, 6), dtype=np.float64)
            planes_in_range_consist[:, 3:6] = planes_in_range[:, 3:6]

            if nbnormals_in_range > 0:
                p1 = (sensorcentre - planes_in_range[:, 3:6]) / np.linalg.norm(sensorcentre - planes_in_range[:, 3:6], axis=1)[:, None]
                p2 = planes_in_range[:, 0:3]
                
                cross_prod = np.cross(p1, p2)
                dot_prod = np.einsum('ij,ij->i', p1, p2)
                angles = np.arctan2(np.linalg.norm(cross_prod, axis=1), dot_prod)

                flip_mask = (angles >= -pi/2) & (angles <= pi/2)

                planes_in_range_consist[flip_mask, 0:3] = sensorInOut * planes_in_range[flip_mask, 0:3]
                planes_in_range_consist[~flip_mask, 0:3] = -sensorInOut * planes_in_range[~flip_mask, 0:3]

            planes_consist.append(planes_in_range_consist)

        return np.concatenate(planes_consist, axis=0)

    def mask_normal(self, double[:, :] planes):
        cdef int[:, :] maskWall_indice = np.logical_and(
            planes[:, 5] > (self.zshape - self.maskBottom),
            planes[:, 5] < (self.zshape - self.maskTop)
        )
        
        test = planes[maskWall_indice, 3:]

        test[:, 0] -= self.maskCenter[0]
        test[:, 1] -= self.maskCenter[1]

        vector_z = sqrt(test[:, 0]**2 + test[:, 1]**2) / self.maskStep

        new_vector = np.zeros((test.shape[0], 3), dtype=np.float64)
        new_vector[:, 0] = -test[:, 0]
        new_vector[:, 1] = -test[:, 1]
        new_vector[:, 2] = -vector_z

        new_vector_norm = np.linalg.norm(new_vector, axis=-1)
        new_vector[:, 0] /= new_vector_norm
        new_vector[:, 1] /= new_vector_norm
        new_vector[:, 2] /= new_vector_norm

        planes[maskWall_indice, :3] = new_vector

        return planes

    def get_pointcloud(self, film):
        test = self.scanZ(film)
        points = test.indices().T
        surface_tree = cKDTree(points)
        dd, ii = surface_tree.query(points, k=18, workers=5)

        pointsNP = points.numpy()

        # Compute mean of all points
        knn_pts = pointsNP[ii]
        xmn = np.mean(knn_pts[:, :, 0], axis=1)
        ymn = np.mean(knn_pts[:, :, 1], axis=1)
        zmn = np.mean(knn_pts[:, :, 2], axis=1)

        c = knn_pts - np.stack([xmn, ymn, zmn], axis=1)[:, np.newaxis, :]

        # Compute covariance matrix
        cov = np.einsum('...ij,...ik->...jk', c, c)

        # Singular Value Decomposition (SVD)
        u, s, vh = np.linalg.svd(cov)

        # Select eigenvector corresponding to smallest eigenvalue
        minevindex = np.argmin(s, axis=1)
        normal_all = np.array([u[i, :, minevindex[i]] for i in range(u.shape[0])])

        # Generate planes matrix
        planes = np.hstack((normal_all, pointsNP))

        # Call normal consistency method
        planes_consist = self.normalconsistency_3D_real(planes)

        return planes_consist

    def get_inject_normal(self, double[:, :] plane, double[:, :] pos, double[:, :] vel):
        cdef double[:, :] plane_point = plane[:, 3:6]
        cdef double[:, :] normal = plane[:, :3]
        cdef cKDTree plane_tree = cKDTree(plane_point * self.celllength)

        cdef bint[:] indice_all = np.zeros(plane.shape[0], dtype=np.bool_)
        cdef bint[:] oscilation_indice = np.zeros(plane.shape[0], dtype=np.bool_)
        cdef double[:] dd_back = np.zeros(plane.shape[0], dtype=np.float64)
        cdef int[:] ii_back = np.zeros(plane.shape[0], dtype=np.int32)

        cdef int i = 0
        while i < len(plane):
            if dd[i] > self.celllength:
                break
            
            normal_current = normal[ii[i]]
            pos_current = plane_point[ii[i]]

            # Calculate if the point is inside
            plane_test = (pos - pos_current) @ normal_current

            if plane_test <= 0:  # If it's on the same side or behind the plane
                indice_all[i] = True
            
            if self.backup:
                # Backup calculations
                dd_back[i] = 9999
                ii_back[i] = ii[i]
                oscilation_indice[i] = True
            i += 1
        
        return indice_all, oscilation_indice, dd_back, ii_back


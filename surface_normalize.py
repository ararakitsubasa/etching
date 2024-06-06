import numpy as np
import torch
from scipy.spatial import KDTree
from scipy import interpolate
import math
from math import pi

class surface_normal:
    def __init__(self, center_with_direction, range3D, InOrOut, yield_hist = None):
        # center xyz inOrout
        self.center_with_direction = center_with_direction
        # boundary x1x2 y1y2 z1z2
        self.range3D = range3D 
        # In for +1 out for -1
        self.InOrOut = InOrOut 
        if yield_hist.all() == None:
            self.yield_hist = np.array([[1.0, 1.05,  1.2,  1.4,  1.5, 1.07, 0.65, 0.28, 0.08,  0], \
                                        [  0,   pi/18,   pi/9,   pi/6,   2*pi/9,   5*pi/18,   pi/3,   7*pi/18,   4*pi/9, pi/2]])
        else:
            self.yield_hist = yield_hist
        self.yield_func = interpolate.interp1d(self.yield_hist[1], self.yield_hist[0], kind='quadratic')

    # def scanZ(self, film):
    #     xshape = film.shape[0]
    #     yshape = film.shape[1]
    #     zshape = film.shape[2]
    #     surface_sparse = torch.zeros((xshape, yshape, zshape))
    #     # print(zshape)
    #     for i in range(zshape-1):
    #         for j in range(xshape-1):
    #             for k in range(yshape-1):
    #                 if (film[j, k, i] == 0 and film[j, k, i-1] != 0) or (film[j, k, i] == 0 and film[j, k, i+1] != 0) \
    #                 or (film[j, k, i] == 0 and film[j-1, k, i] != 0) or (film[j, k, i] == 0 and film[j+1, k, i] != 0) \
    #                 or (film[j, k, i] == 0 and film[j, k-1, i] != 0) or (film[j, k, i] == 0 and film[j, k+1, i] != 0):
    #                     surface_sparse[j,k,i] = 1
    #     return surface_sparse.to_sparse()
    
    def scanZ(self, film): # fast scanZ
        film = torch.Tensor(film)
        xshape, yshape, zshape = film.shape
        
        # 初始化一个全零的表面稀疏张量
        surface_sparse = torch.zeros((xshape, yshape, zshape))
        
        # 获取当前平面与前后平面的布尔索引
        current_plane = film == 0
        prev_plane = torch.zeros_like(film)
        next_plane = torch.zeros_like(film)
        
        prev_plane[:, :, 1:] = film[:, :, :-1]
        next_plane[:, :, :-1] = film[:, :, 1:]
        
        # 获取周围邻居的布尔索引
        neighbors = torch.zeros_like(film, dtype=torch.bool)
        
        neighbors[1:, :, :] |= film[:-1, :, :] != 0  # 上面
        neighbors[:-1, :, :] |= film[1:, :, :] != 0  # 下面
        neighbors[:, 1:, :] |= film[:, :-1, :] != 0  # 左边
        neighbors[:, :-1, :] |= film[:, 1:, :] != 0  # 右边
        neighbors[:, :, 1:] |= film[:, :, :-1] != 0  # 前面
        neighbors[:, :, :-1] |= film[:, :, 1:] != 0  # 后面
        
        # 获取满足条件的索引
        condition = (current_plane & (prev_plane != 0)) | (current_plane & (next_plane != 0)) | (current_plane & neighbors)
        
        # 更新表面稀疏张量
        surface_sparse[condition] = 1
        
        return surface_sparse.to_sparse()
    
    # def normalconsistency_3D_real(self, planes):
        
    #     """
        
    #     This function checks wherer the normals are oriented towards the outside of the surface, i.e., it 
    #     checks the consistency of the normals.
    #     The function changes the direction of the normals that do not point towards the outside of the shape
    #     The function checks whether the normals are oriented towards the centre of the ellipsoid, 
    #     and if YES, then, it turns their orientation
        
    #     INPUTS:
    #         planes: Vector N x 6, where M is the number of points whose normals and 
    #         centroid have been calculated. the columns are the coordinates of the normals and the centroids
            
    #     OUTPUTS:
    #         planesconsist: N x 6 array, where N is the number of points whose planes have been calculated. This array 
    #         has all the planes normals pointing outside the surface.
            
    #     """
        
    #     nbnormals = np.size(planes, 0)
    #     # planes_consist=np.zeros((nbnormals,6))
    #     planes_consist = []
    #     # planes_consist[:, 3:6] = planes[:, 3:6] # We just copy the columns corresponding to the coordinates of the centroids (from 3th to 5th)
        
    #     """ Try the atan2 function : https://uk.mathworks.com/help/vision/ref/pcnormals.html#buxdmoj"""
        
    #     # sensorcentre=np.array([0,0,0])
    #     sensorcentre=np.array([100,100,0]) # vertax shape

    #     for c in range(self.center_with_direction.shape[0]):
    #         sensorcentre = self.center_with_direction[c]
    #         sensorrange = self.range3D[c]
    #         sensorInOut = self.InOrOut[c]
    #         planes_in_range_indice  = np.logical_and(planes[:, 3] < sensorrange[0], planes[:, 3] >= sensorrange[1])
    #         planes_in_range_indice  |= np.logical_and(planes[:, 4] < sensorrange[2], planes[:, 4] >= sensorrange[3])
    #         planes_in_range_indice  |= np.logical_and(planes[:, 5] < sensorrange[4], planes[:, 5] >= sensorrange[5])
    #         planes_in_range = planes[~planes_in_range_indice]

    #         nbnormals_in_range = np.size(planes_in_range, 0)
    #         planes_in_range_consist=np.zeros((nbnormals_in_range,6))
    #         planes_in_range_consist[:, 3:6] = planes_in_range[:, 3:6]

    #         for i in range(nbnormals_in_range):
            
    #             p1 = (sensorcentre - planes_in_range[i,3:6]) / np.linalg.norm(sensorcentre - planes_in_range[i,3:6]) # Vector from the centroid to the centre of the ellipsoid (here the sensor is placed)
    #             p2 = planes_in_range[i,0:3]
                
    #             angle = math.atan2(np.linalg.norm(np.cross(p1,p2)), np.dot(p1,p2) ) # Angle between the centroid-sensor and plane normal
            
                
    #             if (angle >= -pi/2 and angle <= pi/2): # (angle >= -pi/2 and angle <= pi/2):
                    
    #                 planes_in_range_consist[i,0] = -sensorInOut * planes_in_range[i,0]
    #                 planes_in_range_consist[i,1] = -sensorInOut * planes_in_range[i,1]
    #                 planes_in_range_consist[i,2] = -sensorInOut * planes_in_range[i,2]  
                    
    #             else:
                    
    #                 planes_in_range_consist[i,0] = sensorInOut * planes_in_range[i,0]
    #                 planes_in_range_consist[i,1] = sensorInOut * planes_in_range[i,1]
    #                 planes_in_range_consist[i,2] = sensorInOut * planes_in_range[i,2]
                
    #         planes_consist.append(planes_in_range_consist)
            
    #         return np.array(planes_consist)
        
    def normalconsistency_3D_real(self, planes):
        """
        This function checks whether the normals are oriented towards the outside of the surface, i.e., it 
        checks the consistency of the normals. The function changes the direction of the normals that do not 
        point towards the outside of the shape. The function checks whether the normals are oriented towards 
        the centre of the ellipsoid, and if YES, then, it turns their orientation.
        
        INPUTS:
            planes: Vector N x 6, where N is the number of points whose normals and centroids have been calculated. 
            The columns are the coordinates of the normals and the centroids.
            
        OUTPUTS:
            planes_consist: N x 6 array, where N is the number of points whose planes have been calculated. This array 
            has all the planes normals pointing outside the surface.
        """
        
        nbnormals = np.size(planes, 0)
        planes_consist = []
        sensorcentre = np.array([100, 100, 0])  # default value, will be updated in loop

        for c in range(self.center_with_direction.shape[0]):
            sensorcentre = self.center_with_direction[c]
            sensorrange = self.range3D[c]
            sensorInOut = self.InOrOut[c]

            # Determine which planes are in range using broadcasting
            in_range_mask = (
                (planes[:, 3] < sensorrange[0]) & (planes[:, 3] >= sensorrange[1]) &
                (planes[:, 4] < sensorrange[2]) & (planes[:, 4] >= sensorrange[3]) &
                (planes[:, 5] < sensorrange[4]) & (planes[:, 5] >= sensorrange[5])
            )

            planes_in_range = planes[~in_range_mask]
            nbnormals_in_range = np.size(planes_in_range, 0)
            planes_in_range_consist = np.zeros((nbnormals_in_range, 6))
            planes_in_range_consist[:, 3:6] = planes_in_range[:, 3:6]

            if nbnormals_in_range > 0:
                p1 = (sensorcentre - planes_in_range[:, 3:6]) / np.linalg.norm(sensorcentre - planes_in_range[:, 3:6], axis=1)[:, None]
                p2 = planes_in_range[:, 0:3]
                
                cross_prod = np.cross(p1, p2)
                dot_prod = np.einsum('ij,ij->i', p1, p2)
                angles = np.arctan2(np.linalg.norm(cross_prod, axis=1), dot_prod)

                flip_mask = (angles >= -np.pi/2) & (angles <= np.pi/2)

                planes_in_range_consist[flip_mask, 0:3] = -sensorInOut * planes_in_range[flip_mask, 0:3]
                planes_in_range_consist[~flip_mask, 0:3] = sensorInOut * planes_in_range[~flip_mask, 0:3]

            planes_consist.append(planes_in_range_consist)

        return np.concatenate(planes_consist, axis=0) 

    def get_pointcloud(self, film):
        test = self.scanZ(film)

        points = test.indices().T
        # print(points.shape)

        surface_tree = KDTree(points)
        dd, ii = surface_tree.query(points, k=18, workers=5)

        pointsNP = points.numpy()


        normal_all = np.zeros((ii.shape[0], 3))
        for i in range(ii.shape[0]):
            knn_pt = pointsNP[ii[i]]
            # print(knn_pt.shape)
            # xmn = np.mean(knn_pt[:,0], axis=1)
            # ymn = np.mean(knn_pt[:,1], axis=1)
            # zmn = np.mean(knn_pt[:,2], axis=1)
            xmn = np.mean(knn_pt[:,0])
            ymn = np.mean(knn_pt[:,1])
            zmn = np.mean(knn_pt[:,2])

            c=np.zeros((np.size(knn_pt,0),3))

            c[:,0] = knn_pt[:,0]-xmn
            c[:,1] = knn_pt[:,1]-ymn
            c[:,2] = knn_pt[:,2]-zmn

            cov=np.zeros((3,3))    

            cov[0,0] = np.dot(c[:,0],c[:,0])
            cov[0,1] = np.dot(c[:,0],c[:,1])
            cov[0,2] = np.dot(c[:,0],c[:,2])

            cov[1,0] = cov[0,1]
            cov[1,1] = np.dot(c[:,1],c[:,1])
            cov[1,2] = np.dot(c[:,1],c[:,2])

            cov[2,0] = cov[0,2]
            cov[2,1] = cov[1,2]
            cov[2,2] = np.dot(c[:,2],c[:,2])

            "Single value decomposition (SVD)"

            u,s,vh = np.linalg.svd(cov) # U contains the orthonormal eigenvectors and S contains the eigenvectors

            "Selection of minimum eigenvalue"

            minevindex = np.argmin(s)

            "Selection of orthogonal vector corresponing to this eigenvalue --> vector normal to the plane defined by the kpoints"

            normal_vect = u[:,minevindex]
            normal_all[i, :] = normal_vect


        planes=np.zeros((points.shape[0],6))
        for i in range(points.shape[0]):

            planes[i,0:3] = normal_all[i] #Keep the coordinates of the normal vectors
            planes[i,3:6] = pointsNP[i] #Keep the coordinates of the centroid

        planes_consist = self.normalconsistency_3D_real(planes)

        return planes_consist[0]
    
    def get_pointcloud(self, film):
        test = self.scanZ(film)
        points = test.indices().T
        surface_tree = KDTree(points)
        dd, ii = surface_tree.query(points, k=18, workers=5)

        pointsNP = points.numpy()

        # 计算所有点的均值
        knn_pts = pointsNP[ii]
        xmn = np.mean(knn_pts[:, :, 0], axis=1)
        ymn = np.mean(knn_pts[:, :, 1], axis=1)
        zmn = np.mean(knn_pts[:, :, 2], axis=1)

        c = knn_pts - np.stack([xmn, ymn, zmn], axis=1)[:, np.newaxis, :]

        # 计算协方差矩阵
        cov = np.einsum('...ij,...ik->...jk', c, c)

        # 单值分解 (SVD)
        u, s, vh = np.linalg.svd(cov)

        # 选择最小特征值对应的特征向量
        minevindex = np.argmin(s, axis=1)
        normal_all = np.array([u[i, :, minevindex[i]] for i in range(u.shape[0])])

        # 生成平面矩阵
        planes = np.hstack((normal_all, pointsNP))

        # 调用 normalconsistency_3D_real 方法
        planes_consist = self.normalconsistency_3D_real(planes)

        return planes_consist

    def get_inject_theta(self, plane, pos, vel):
        # plane = self.get_pointcloud(film)
        plane_point = plane[:, 3:6]
        normal = plane[:, :3]
        velocity_normal = np.linalg.norm(vel, axis=1)
        velocity = np.divide(vel.T, velocity_normal).T
        plane_tree = KDTree(plane_point)

        dd, ii = plane_tree.query(pos, k=1, workers=1)

        dot_products = np.einsum('...i,...i->...', velocity, normal[ii])
        theta = np.arccos(dot_products)
        return theta
    
    def get_yield(self, theta):
        # yield_func = interpolate.interp1d(self.yield_hist[1], self.yield_hist[0], kind='quadratic')
        etch_yield = self.yield_func(theta)
        return etch_yield


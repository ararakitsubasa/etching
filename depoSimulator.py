import numpy as np
# import cupy as cp
from scipy.spatial import KDTree
import time as Time
from tqdm import tqdm, trange

class depo:
    def __init__(self, param, TS, N, sub_xy, film, n, cellSize, kdtreeN, tstep):
        self.param = param # n beta
        self.TS = TS
        self.kdtreeN = kdtreeN
        self.cellSizeX = cellSize[0]
        self.cellSizeY = cellSize[1]
        self.cellSizeZ = cellSize[2]
        self.timeStep = tstep
        self.sub_x = sub_xy[0]
        self.sub_y = sub_xy[1]
        self.substrate = film
        self.n = n
        self.N = N
        self.T = 300
        self.Cm = (2*1.380649e-23*self.T/(27*1.66e-27) )**0.5 # (2kT/m)**0.5 27 for the Al
        self.depo_pos = np.zeros((1,6))

    def rfunc(self,x): #Release factor function
        # print("-------rfunc------")
        # print(x)
        n = self.param[0]
        beta = self.param[1]
        y = np.cos(x) ** n * (1 + beta * np.cos(x) ** 2)# * (n ** 2 + 4 * n + 3) / (n * beta + n + beta + 3) /2 / pi
        return y

    def deposition(self,cpos,ppos):

        # pvec = np.array([0.0 ,0.0, 1.0])
        # cvec = np.array([0.0 ,0.0, 1.0])
        cvec = np.zeros((cpos.shape[0], 3))
        cvec[:,2] = 1

        l = np.linalg.norm(ppos-cpos, axis=1)

        dot_ew = np.sum(cvec * (ppos-cpos), axis=1)
        theta = np.arccos(dot_ew*(1/l))

        ppos2 = np.zeros_like(ppos)
        ppos2[:, 0] = ppos[:, 0]
        ppos2[:, 1] = ppos[:, 1]

        # direction of radius
        rvec = np.zeros_like(cpos)
        rvec[:, 0] = cpos[:, 0]
        rvec[:, 1] = cpos[:, 1]

        if rvec[0][0] == 0 and rvec[0][1] == 0:
            rvec[:,1] = 1

        dot_ew_phi = np.sum(rvec * (ppos2-cpos), axis=1)

        r = np.linalg.norm(rvec, axis=1)
        # print(r)
        r2 = np.linalg.norm(ppos2-cpos, axis=1)
        # print(r2)
        phi = np.arccos(dot_ew_phi*(1/(r*r2)))

        return self.rfunc(theta), theta*180/np.pi, phi*180/np.pi

    def target_substrate(self, Ero_dist_x, Ero_dist_y, sub_x, sub_y):
        ppos = np.array([Ero_dist_x, Ero_dist_y, np.ones_like(Ero_dist_x)*self.TS]).T
        # sub_x, sub_y = 0, 0 # center
        cpos = np.array([np.ones_like(Ero_dist_x)*sub_x ,np.ones_like(Ero_dist_x)*sub_y, np.zeros_like(Ero_dist_x)]).T

        filmMac = self.deposition(cpos,ppos)

        return filmMac

    def velocity_dist(self, Ero_dist_x, filmMac):
        N = Ero_dist_x.shape[0]
        Random1 = np.random.rand(N)
        Random2 = np.random.rand(N)
        Random3 = np.random.rand(N)
        velosity_matrix = np.array([self.max_velocity_u(Random1, Random2), self.max_velocity_w(Random1, Random2), self.max_velocity_v(Random3)]).T
        velosity_norm = np.linalg.norm(velosity_matrix, axis=1)
        vel_theta = filmMac[1]/180*np.pi
        vel_phi = filmMac[2]/180*np.pi

        vel_x = velosity_norm*np.sin(vel_theta)*np.cos(vel_phi)
        vel_y = velosity_norm*np.sin(vel_theta)*np.sin(vel_phi)*np.random.choice([-1,1], vel_phi.shape[0])
        vel_z = velosity_norm*np.cos(vel_theta)
        velosity_matrix_2 = np.array([vel_x, vel_y, -vel_z]).T

        return velosity_matrix_2


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

        # print(i_cp)
        indices = np.logical_or(i_cp >= cellSize_x_cp, i_cp <= 0)
        # print(indices)
        indices |= np.logical_or(j_cp >= cellSize_y_cp, j_cp <= 0)
        # print(indices)
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

        indice_inject = np.array(film[i, j, k] > 5)
        # print(indice_inject)

        pos_1 = pos[indice_inject]
        # print(pos_1)
        if np.any(indice_inject):
            self.depo_position(pos[indice_inject], vel[indice_inject])


        surface_depo = np.logical_and(film >= 0, film < 1) # depo
        # surface_depo = np.logical_and(film > 0, film < 2000) #etching
        surface_tree = KDTree(np.argwhere(surface_depo == True))

        dd, ii = surface_tree.query(pos_1, k=self.kdtreeN, workers=1)

        surface_indice = np.argwhere(surface_depo == True)

        ddsum = np.sum(dd, axis=1)

        # kdi order
        for kdi in range(self.kdtreeN):
            i1 = surface_indice[ii][:,kdi,0] #[particle, order, xyz]
            j1 = surface_indice[ii][:,kdi,1]
            k1 = surface_indice[ii][:,kdi,2]

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

        surface_film = np.logical_and(film >= 1, film < 2)
        film[surface_film] = 20

        return film, pos, vel, weights_arr

    def depo_position(self, pos, vel):
        PosVel = np.concatenate((pos, vel), axis=1)
        self.depo_pos = np.vstack((self.depo_pos, PosVel))

    def getAcc_depo(self, pos, vel, boxsize, tStep, film, weights_arr, depoStep):
        dx = boxsize

        pos_cp = pos
        vel_cp = vel

        tStep_cp = tStep

        i = np.floor((pos_cp[:, 0]) / dx).astype(int)
        j = np.floor((pos_cp[:, 1]) / dx).astype(int)
        k = np.floor((pos_cp[:, 2]) / dx).astype(int)

        # pos, vel, i, j, k, cellSize_x, cellSize_y, cellSize_z,
        pos_cp, Nvel_cp, i, j, k, weights_arr = self.boundary(pos_cp, vel_cp, i, j, k, weights_arr)
        # print(pos_cp)
        film_depo, pos_cp, Nvel_cp, weights_arr_depo = self.depo_film(film, pos_cp, Nvel_cp, i, j, k, weights_arr, depoStep)

        Npos2_cp = Nvel_cp * tStep_cp + pos_cp

        return np.array([pos_cp, Nvel_cp]), np.array([Npos2_cp, Nvel_cp]), film_depo, weights_arr_depo

    def runDepo(self, p0, v0, time, film, weights_arr, depoStep):

        tmax = time
        tstep = self.timeStep
        t = 0
        p1 = p0
        v1 = v0
        film_1 = self.substrate
        weights_arr_1 = weights_arr

        cell = 1

        with tqdm(total=100, desc='running', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
            i = 0
            while t < tmax:
                p2v2 = self.getAcc_depo(p1, v1, cell, tstep, film_1, weights_arr_1, depoStep)
                p2 = p2v2[1][0]
                if p2.shape[0] == 0:
                    break
                v2 = p2v2[1][1]
                p1 = p2v2[0][0]
                v1 = p2v2[0][1]
                film_1 = p2v2[2]
                weights_arr_1 = p2v2[3]
                t += tstep
                p1 = p2
                v1 = v2
                i += 1
                if i % (int((tmax/tstep)/100)) == 0:
                    Time.sleep(0.01)
                    # 更新发呆进度
                    pbar.update(1)
                # if i % (int((tmax/tstep)/20)) == 0:
                #     Time.sleep(0.01)
                #     # 更新发呆进度
                #     pbar.update(5)

        return film
    
    def stepRundepo(self, step, randomSeed, velosityDist, weights):

        for i in range(step):
            np.random.seed(randomSeed+i)
            position_matrix = np.array([np.random.rand(self.N)*self.cellSizeX, np.random.rand(self.N)*self.cellSizeY, np.random.uniform(0, 10, self.N)+ self.cellSizeZ - 10]).T
            result =  self.runDepo(position_matrix, velosityDist, 1, self.substrate, weights, depoStep=i+1)

        return result
    
    def run(self, step, seed, Ero_dist_x, Ero_dist_y):
        filmMac = self.target_substrate(Ero_dist_x, Ero_dist_y, self.sub_x, self.sub_y)
        velosity_matrix = self.velocity_dist(Ero_dist_x, filmMac)
        depoFilm = self.stepRundepo(step, seed, velosity_matrix, filmMac[0])

        return depoFilm
    
    def runDepoition(self, step, seed, N, weight):
        # filmMac = self.target_substrate(Ero_dist_x, Ero_dist_y, self.sub_x, self.sub_y)
        # velosity_matrix = self.velocity_dist(Ero_dist_x, filmMac)
        # N = int(631394)
        weights = np.ones(N)*weight
        Random1 = np.random.rand(N)
        Random2 = np.random.rand(N)
        Random3 = np.random.rand(N)
        velosity_matrix = np.array([self.max_velocity_u(Random1, Random2), self.max_velocity_w(Random1, Random2), self.max_velocity_v(Random3)]).T
        depoFilm = self.stepRundepo(step, seed, velosity_matrix, weights)

        return depoFilm, self.depo_pos[1:]
    
    def coverage(self, step, seed, Ero_dist_x, Ero_dist_y):
        depoFilm = self.run(step, seed, Ero_dist_x, Ero_dist_y)
        left = np.array(depoFilm[75, 75:125, 10:60] >= 20)
        right = np.array(depoFilm[125, 75:125, 10:60] >= 20)
        front = np.array(depoFilm[75:125, 75, 10:60] >= 20)
        behind = np.array(depoFilm[75:125, 125, 10:60] >= 20)
        down = np.array(depoFilm[75:125, 75:125, 10] >= 20)

        coverage = (left.sum() + right.sum() + front.sum() + behind.sum() + down.sum())/12500
        return coverage
    
    def depoProfile(self, step, seed, Ero_dist_x, Ero_dist_y):
        depoFilm = self.run(step, seed, Ero_dist_x, Ero_dist_y)
        profile3D = depoFilm[75:125, 75:125, 10:80]

        return profile3D
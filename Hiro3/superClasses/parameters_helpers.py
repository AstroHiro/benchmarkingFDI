import numpy as np

class initialization_helpers:
    def __init__(self,Nsc,Cj2d,Pj2d,n2,Lamdd):
        self.Nsc = Nsc  # number of spacecrafts
        self.Cj2d = Cj2d
        self.Pj2d = Pj2d
        self.n2 = n2
        self.Lamdd = Lamdd

    def gettransmats(self, xe, ye, ze, psie0, psiz0):
        p = self.Nsc
        n2 = self.n2
        lmin, lmax, PHI = self.getmaxes(xe, ye, ze, psie0, psiz0)
        R11 = -xe / lmin * np.sin(PHI - psie0)
        R12 = ye / lmin * np.cos(PHI - psie0)
        R13 = -ze / lmin * np.sin(PHI - psiz0)
        R21 = -xe / lmax * np.cos(PHI - psie0)
        R22 = -ye / lmax * np.sin(PHI - psie0)
        R23 = -ze / lmax * np.cos(PHI - psiz0)
        R31 = -ye * ze / lmin / lmax * np.cos(psie0 - psiz0)
        R32 = xe * ze / lmin / lmax * np.sin(psie0 - psiz0)
        R33 = xe * ye / lmin / lmax
        R1d = np.array([[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]])
        R2d = np.array([[1, 0, 0], [0, lmin / lmax, 0], [0, 0, 1]])
        Rfd = R2d @ R1d
        Tall = np.zeros((n2 * p, n2 * p))
        idx = 1
        nidx = 4
        phi0 = 0
        dphi = 0
        for j in range(p):
            if j > nidx / 2 * idx * (idx + 1):
                phi0 = phi0 + dphi
                idx = idx + 1
            sat_num = nidx * idx
            phi = 2 * np.pi / sat_num
            Tz = np.array([[np.cos((j - 1) * phi + phi0), -np.sin((j - 1) * phi + phi0)],
                           [np.sin((j - 1) * phi + phi0), np.cos((j - 1) * phi + phi0)]])
            Tz = idx * np.identity(2) @ Tz
            Tall[n2 * j:n2 * (j + 1), n2 * j:n2 * (j + 1)] = np.vstack(
                (np.hstack((Tz, np.zeros((2, 1)))), np.hstack((np.zeros(2), 1))))
        idx_targets = idx
        return Rfd, Tall, idx_targets

    def getmaxes(self, xe, ye, ze, psie0, psiz0):
        numP = (xe ** 2 - ye ** 2) * np.sin(2 * psie0) + ze ** 2 * np.sin(2 * psiz0)
        denP = (xe ** 2 - ye ** 2) * np.cos(2 * psie0) + ze ** 2 * np.cos(2 * psiz0)
        PHI = 1 / 2 * np.arctan2(numP, denP)
        xdfun = lambda psi: np.array([xe * np.sin(psi + psie0), ye * np.cos(psi + psie0), ze * np.sin(psi + psiz0)])
        lfun = lambda psi: np.linalg.norm(xdfun(psi))
        lmin = lfun(-PHI)
        lmax = lfun(3 / 2 * np.pi - PHI)
        return lmin, lmax, PHI

    def getparams(self):
        p = self.Nsc
        n2 = self.n2
        n_u = n2 * p
        xe = 2
        ye = 2
        ze = 0.
        pe0 = np.deg2rad(0.573)
        pz0 = np.deg2rad(11.46)
        nphase = 0.0011
        qd = lambda t: np.array(
            [xe * np.sin(nphase * t + pe0), ye * np.cos(nphase * t + pe0), ze * np.sin(nphase * t + pz0)])
        qd_dot = lambda t: nphase * np.array(
            [xe * np.cos(nphase * t + pe0), -ye * np.sin(nphase * t + pe0), ze * np.cos(nphase * t + pz0)])
        qd_ddot = lambda t: -nphase ** 2 * np.array(
            [xe * np.sin(nphase * t + pe0), ye * np.cos(nphase * t + pe0), ze * np.sin(nphase * t + pz0)])
        Rfd, Tall, idx_targets = self.gettransmats(xe, ye, ze, pe0, pz0)
        Qd = lambda t: np.array([qd(t), qd_dot(t), qd_ddot(t)])
        d0 = 0.2
        v0 = 0.4
        q0_all = np.zeros(n_u)
        q0_dot_all = np.zeros(n_u)
        for j in range(p):
            xj0 = -d0 + 2 * d0 * np.random.rand(1)[0]
            yj0 = -d0 + 2 * d0 * np.random.rand(1)[0]
            zj0 = -d0 + 2 * d0 * np.random.rand(1)[0]
            q0_all[n2 * j:n2 * (j + 1)] = np.array([xj0, yj0, zj0])
        self.Qdfun = Qd
        self.q0_all = q0_all
        self.q0_dot_all = q0_dot_all
        self.Rfd = Rfd
        self.R_d = np.kron(np.identity(p), Rfd)
        self.Tall = Tall
        self.idx_targets = idx_targets
        Rq2X = np.zeros((2 * p, p))
        Rv2X = np.zeros((2 * p, p))
        for j in range(p):
            Rq2X[2 * j, j] = 1
            Rv2X[2 * (j + 1) - 1, j] = 1
        Rq2X = np.kron(Rq2X, np.identity(n2))
        Rv2X = np.kron(Rv2X, np.identity(n2))
        Rqq2X = np.hstack((Rq2X, Rv2X))
        self.Rqq2X = Rqq2X
        self.Xjall0 = Rqq2X @ np.hstack((q0_all, q0_dot_all))
        self.Bj2d = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
        self.Pj2d = np.array([[0], [0], [0], [1]])
        self.Lj2d = np.array([[0], [0], [1], [0]])
        self.Cj2d = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        CP = self.Cj2d @ self.Pj2d
        self.Nj2d = self.Pj2d @ np.linalg.inv((CP).T @ CP) @ CP.T
        pass

    def getlaplacian(self, M1, M2):
        p = self.Nsc
        L = np.zeros((3 * p, 3 * p))
        for j in range(p):
            if j == 0:
                L[0:3, 0:3] = M1
                L[0:3, 3:6] = M2
                L[0:3, 3 * (p - 1):3 * p] = M2
            elif j == p - 1:
                L[3 * (p - 1):3 * p, 3 * (p - 1):3 * p] = M1
                L[3 * (p - 1):3 * p, 0:3] = M2
                L[3 * (p - 1):3 * p, 3 * (p - 2):3 * (p - 1)] = M2
            else:
                L[3 * j:3 * (j + 1), 3 * j:3 * (j + 1)] = M1
                L[3 * j:3 * (j + 1), 3 * (j + 1):3 * (j + 2)] = M2
                L[3 * j:3 * (j + 1), 3 * (j - 1):3 * j] = M2
        return L

    #Control function helper, moved here for simplicity
    def getsd(self, Xjall, Qd_t):
        p = self.Nsc
        Rfd = self.Rfd
        Tall = self.Tall
        Lamdd = self.Lamdd
        qq = self.Rqq2X.T @ Xjall
        q_all = qq[0:3 * p]
        q_dot_all = qq[3 * p:6 * p]
        qd = Qd_t[0, :]
        qd_dot = Qd_t[1, :]
        qd_ddot = Qd_t[2, :]
        R_d = self.R_d
        Lam_all = np.kron(np.identity(p), Lamdd)
        qd_all = np.tile(qd, p)
        qd_dot_all = np.tile(qd_dot, p)
        qd_ddot_all = np.tile(qd_ddot, p)
        q_d = R_d @ q_all
        q_dot_d = R_d @ q_dot_all
        qd_d = R_d @ qd_all
        qd_dot_d = R_d @ qd_dot_all
        qd_ddot_d = R_d @ qd_ddot_all
        qr_dot_d = Tall @ qd_dot_d - Lam_all @ (q_d - Tall @ qd_d)
        qr_ddot_d = Tall @ qd_ddot_d - Lam_all @ (q_dot_d - Tall @ qd_dot_d)
        s_d = q_dot_d - qr_dot_d
        return s_d, qr_dot_d, qr_ddot_d




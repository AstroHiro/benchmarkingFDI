import numpy as np
import control
from Hiro3.superClasses.residual_helpers import residual_helpers
from Hiro3.superClasses.dynamics_helpers import dynamics_helpers
from Hiro3.superClasses.parameters_helpers import initialization_helpers
from Hiro3.superClasses.lagrange_helper import lagrange_helper
from Hiro3.superClasses.get_initial_params import get_initial_params

class Benchmark(residual_helpers, dynamics_helpers, initialization_helpers, lagrange_helper, get_initial_params):
    def __init__(self, mu, Nsc=1):
        self.get_initial_params_function(Nsc, mu)
        residual_helpers.__init__(self,self.mu)
        dynamics_helpers.__init__(self,self.mu,self.rho,self.Area,self.kJ2,self.mass,self.ome,self.Cd)
        initialization_helpers.__init__(self,self.Nsc,self.Cj2d,self.Pj2d,self.n2,self.Lamdd)
        lagrange_helper.__init__(self, self.kJ2, self.Area, self.mu, self.mass, self.ome, self.rho, self.Cd)
        K1 = np.identity(3) * 5
        K2 = np.identity(3) * 2
        self.Lap = self.getlaplacian(K1, -K2)
        self.getparams()

    def dynamics_all(self, t, Xjcall, Ujcall, m1jcall, wjcall):
        p = self.Nsc
        Xjall = Xjcall[0:6 * p]
        Xc = Xjcall[6 * p:6 * (p + 1)]
        Ujall = Ujcall[0:3 * p]
        Uc = Ujcall[3 * p:3 * (p + 1)]
        m1c = m1jcall[p]
        wc = wjcall[p]
        dXdcalldt = np.zeros_like(Xjcall)
        for j in range(p):
            Xj = Xjall[6 * j:6 * (j + 1)]
            Uj = Ujall[3 * j:3 * (j + 1)]
            m1j = m1jcall[j]
            wj = wjcall[j]
            dXdcalldt[6 * j:6 * (j + 1)] = self.relativef(t, Xj, Xc, Uj, Uc, m1c, wc, m1j, wj)
        dXdcalldt[6 * p:6 * (p + 1)] = self.scdynamics_radial_3d(t, Xc, Uc, m1c, wc)
        return dXdcalldt

    def lagrangeMCGDj(self, t, Xj, Xc, Uc):
        dXcdt = self.scdynamics_radial_3d(t, Xc, Uc, 0, 0)
        Cmat, Dmat, Gmat, Mmat = self.lagrange_helper_function(Uc, Xc, Xj, dXcdt)
        return Mmat, Cmat, Gmat, Dmat, dXcdt

    def lagrangeMCGD(self, t, Xjall, Xc, Uc):
        p = self.Nsc
        M = np.zeros((3 * p, 3 * p))
        C = np.zeros((3 * p, 3 * p))
        G = np.zeros(3 * p)
        D = np.zeros(3 * p)
        for j in range(p):
            Xj = Xjall[6 * j:6 * (j + 1)]
            Mj, Cj, Gj, Dj, dXcdt = self.lagrangeMCGDj(t, Xj, Xc, Uc)
            M[3 * j:3 * (j + 1), 3 * j:3 * (j + 1)] = Mj
            C[3 * j:3 * (j + 1), 3 * j:3 * (j + 1)] = Cj
            G[3 * j:3 * (j + 1)] = Gj
            D[3 * j:3 * (j + 1)] = Dj
        return M, C, G, D, dXcdt

    def measurement_Xc(self, t, Xc):
        r = Xc[0]
        h = Xc[2]
        th = Xc[5]
        Yc = np.array([r, h, th])
        return Yc

    def measurement_Xj(self, t, Xj):
        xj = Xj[0]
        yj = Xj[1]
        dyjdt = Xj[4]
        Yj = np.array([xj, yj, dyjdt])
        return Yj

    def residual_filter_Xc(self, t, xi_c, Yc, Uc):
        h = Yc[1]
        r = Yc[0]
        u2 = Uc[1]
        k = 10
        dxi_c_dt = r * u2 + k * (h - xi_c)
        return dxi_c_dt

    def residual_filter_Xj(self, t, Zj2d, Yj, Uj, Xc, Uc, m1c, wc):
        B = self.Bj2d
        C = self.Cj2d
        N = self.Nj2d
        Xhatj2d = Zj2d + N @ Yj
        phij = self.residual_phij(t, Xhatj2d, Xc, Uc, m1c, wc)
        Aphij = self.residual_Aphij(t, Xhatj2d, Xc, Uc, m1c, wc)
        I = np.identity(4)
        M = I - N @ C
        G = M @ B
        Aekf = M @ Aphij
        Q = np.identity(4) * 10
        R = np.identity(3)
        KT, _, _ = control.lqr(Aekf.T, C.T, Q, R)
        K = -KT.T
        F = K @ C
        H = K @ C @ N - K
        dZj2d_dt = F @ Zj2d + G @ Uj[0:2] + M @ phij + H @ Yj
        return dZj2d_dt

    def residual_filter_w_Xc_EKF(self, t, Zc, Yc, Uc):
        mu = self.mu
        r_meas = Yc[0]
        B = np.array([[0, 0], [1, 0], [0, r_meas], [0, 0]])
        Pw = np.array([[0], [0], [1], [0]])
        Lw = np.array([[0], [1], [0], [0]])
        C = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        CP = C @ Pw
        N = Pw @ np.linalg.inv((CP).T @ CP) @ CP.T
        self.Nekf = N
        Xhat_c = Zc + N @ Yc
        phi = self.residual_phi_Xc(Xhat_c)
        Aphi = self.residual_Aphi_Xc(Xhat_c)
        I = np.identity(4)
        M = I - N @ C
        G = M @ B
        Aekf = M @ Aphi
        Q = np.identity(4) * 0.01
        R = np.identity(3)
        KT, _, _ = control.lqr(Aekf.T, C.T, Q, R)
        K = -KT.T
        F = K @ C
        H = K @ C @ N - K
        dZc_dt = F @ Zc + G @ Uc[0:2] + M @ phi + H @ Yc
        return dZc_dt

    def expcontrol(self, t, Xjall, Xc, Uc, Qd_t):
        p = self.Nsc
        Rfd = self.Rfd
        Lap = self.Lap
        R_d = self.R_d
        M, C, G, D, dXcdt = self.lagrangeMCGD(t, Xjall, Xc, Uc)
        M_d = R_d @ M @ R_d.T
        C_d = R_d @ C @ R_d.T
        G_d = R_d @ G
        D_d = R_d @ D
        s_d, qr_dot_d, qr_ddot_d = self.getsd(Xjall, Qd_t)
        Rfdtau = M_d @ qr_ddot_d + C_d @ qr_dot_d + G_d + D_d - Lap @ s_d
        U = R_d.T @ Rfdtau
        return U


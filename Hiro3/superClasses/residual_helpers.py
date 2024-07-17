import numpy as np

class residual_helpers:
    def __init__(self,mu):
        self.mu = mu  # gravitational constant

    def residual_phij(self, t, Xj2d, Xc, Uc, m1c, wc):
        mu = self.mu
        xj = Xj2d[0]
        yj = Xj2d[1]
        xjdot = Xj2d[2]
        yjdot = Xj2d[3]
        r = Xc[0]
        vx = Xc[1]
        h = Xc[2]
        uR = Uc[0]
        uT = Uc[1]
        omz = h / r ** 2
        rj = np.sqrt((r + xj) ** 2 + yj ** 2)
        alz = -2 * h * vx / r ** 3
        et = np.sqrt(mu / r ** 3)
        etj = np.sqrt(mu / rj ** 3)
        xjddot = 2 * yjdot * omz - xj * (etj ** 2 - omz ** 2) + yj * alz - r * (etj ** 2 - et ** 2) - (uR + wc)
        yjddot = -2 * xjdot * omz - xj * alz - yj * (etj ** 2 - omz ** 2) - (uT + m1c)
        dXj2d_dt = np.array([xjdot, yjdot, xjddot, yjddot])
        return dXj2d_dt

    def residual_Aphij(self, t, Xj2d, Xc, Uc, m1c, wc):
        dx = 0.001
        Aphi = np.zeros((4, 4))
        for i in range(4):
            ei = np.zeros(4)
            ei[i] = 1
            f1 = self.residual_phij(t, Xj2d + ei * dx, Xc, Uc, m1c, wc)
            f0 = self.residual_phij(t, Xj2d - ei * dx, Xc, Uc, m1c, wc)
            dfdx = (f1 - f0) / 2 / dx
            Aphi[:, i] = dfdx
        return Aphi

    def residual_phi_Xc(self, Xc2d):
        mu = self.mu
        r = Xc2d[0]
        vx = Xc2d[1]
        h = Xc2d[2]
        phi = np.array([vx, -mu / r ** 2 + h ** 2 / r ** 3, 0, h / r ** 2])
        return phi

    def residual_Aphi_Xc(self, Xc2d):
        dx = 0.001
        Aphi = np.zeros((4, 4))
        for i in range(4):
            ei = np.zeros(4)
            ei[i] = 1
            f1 = self.residual_phi_Xc(Xc2d + ei * dx)
            f0 = self.residual_phi_Xc(Xc2d - ei * dx)
            dfdx = (f1 - f0) / 2 / dx
            Aphi[:, i] = dfdx
        return Aphi





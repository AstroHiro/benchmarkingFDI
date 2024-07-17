import numpy as np

class dynamics_helpers:
    def __init__(self,mu,rho,Area,kJ2,mass,ome,Cd):
        self.mu=mu
        self.rho = rho
        self.rhoj = self.rho
        self.Area = Area
        self.Areaj = self.Area
        self.kJ2 = kJ2  # self.kJ2 = 2.6330e+10
        self.mass = mass
        self.massj = self.mass
        self.ome = ome
        self.Cd = Cd  # self.Cd = 1
        self.Cdj = self.Cd

    def scdynamics_radial(self, t, X, U, m1, w):
        r = X[0]
        v = X[2]
        om = X[3]
        u1 = U[0]
        u2 = U[1]
        mu = self.mu
        drhodv = v
        dphdt = om
        dvdt = r * om ** 2 - mu / r ** 2 + u1 + w
        domdt = -2 * v * om / r + u2 / r + m1 / r
        dXdt = np.array([drhodv, dphdt, dvdt, domdt])
        return dXdt

    def scdynamics_cartesian(self, t, X, U):
        mu = self.mu
        dxdt = -mu * X[0:2] / (np.linalg.norm(X)) ** 3 + U
        dXdt = np.hstack((X[2:4], dxdt))
        return dXdt

    def scdynamics_radial_3d(self, t, X, U, m1, w):
        mu = self.mu
        kJ2 = self.kJ2
        ome = self.ome
        Cd = self.Cd
        mass = self.mass
        rho = self.rho
        Area = self.Area
        uR = U[0]
        uT = U[1]
        uN = U[2]
        r = X[0]
        vx = X[1]
        h = X[2]
        Om = X[3]
        i = X[4]
        th = X[5]
        si = np.sin(i)
        ci = np.cos(i)
        s2i = np.sin(2 * i)
        sth = np.sin(th)
        cth = np.cos(th)
        s2th = np.sin(2 * th)
        C = Cd * (Area / mass) * rho / 2
        Va = np.array([vx, h / r - ome * r * ci, ome * r * cth * si])
        va = np.linalg.norm(Va)
        drdt = vx
        dvxdt = -mu / r ** 2 + h ** 2 / r ** 3 - kJ2 / r ** 4 * (1 - 3 * si ** 2 * sth ** 2) - C * va * vx + uR + w
        dhdt = -kJ2 * si ** 2 * s2th / r ** 3 - C * va * (h - ome * r ** 2 * ci) + r * uT + r * m1
        dOmdt = -2 * kJ2 * ci * sth ** 2 / h / r ** 3 - C * va * ome * r ** 2 * s2th / 2 / h + (r * sth / h / si) * uN
        didt = -kJ2 * s2i * s2th / 2 / h / r ** 3 - C * va * ome * r ** 2 * si * cth ** 2 / h + (r * cth / h) * uN
        dthdt = h / r ** 2 + 2 * kJ2 * ci ** 2 * sth ** 2 / h / r ** 3 + C * va * ome * r ** 2 * ci * s2th / 2 / h - (
                    r * sth * ci / h / si) * uN
        dXdt = np.array([drdt, dvxdt, dhdt, dOmdt, didt, dthdt])
        return dXdt

    def relativef(self, t, Xj, Xc, Uj, Uc, m1c, wc, m1j, wj):
        mu = self.mu
        kJ2 = self.kJ2
        ome = self.ome
        Cd = self.Cd
        mass = self.mass
        rho = self.rho
        Area = self.Area
        Cdj = self.Cdj
        massj = self.massj
        rhoj = self.rhoj
        Areaj = self.Areaj
        C = Cd * (Area / mass) * rho / 2
        Cj = Cdj * (Areaj / massj) * rhoj / 2
        xj = Xj[0]
        yj = Xj[1]
        zj = Xj[2]
        xjdot = Xj[3]
        yjdot = Xj[4]
        zjdot = Xj[5]
        r = Xc[0]
        vx = Xc[1]
        h = Xc[2]
        Om = Xc[3]
        i = Xc[4]
        th = Xc[5]
        uR = Uc[0]
        uT = Uc[1]
        uN = Uc[2]
        dXcdt = self.scdynamics_radial_3d(t, Xc, Uc, m1c, wc)
        Omdot = dXcdt[3]
        idot = dXcdt[4]
        thdot = dXcdt[5]
        si = np.sin(i)
        ci = np.cos(i)
        s2i = np.sin(2 * i)
        sth = np.sin(th)
        cth = np.cos(th)
        s2th = np.sin(2 * th)
        lj = Xj[0:3]
        ljdot = Xj[3:6]
        Va = np.array([vx, h / r - ome * r * ci, ome * r * cth * si])
        va = np.linalg.norm(Va)
        omx = idot * cth + Omdot * sth * si + r / h * uN
        omz = thdot + Omdot * ci
        om_vec = np.array([omx, 0, omz])
        Vaj = Va + ljdot + np.cross(om_vec, lj)
        vaj = np.linalg.norm(Vaj)
        rj = np.sqrt((r + xj) ** 2 + yj ** 2 + zj ** 2)
        rjZ = (r + xj) * si * sth + yj * si * cth + zj * ci
        zt = 2 * kJ2 * si * sth / r ** 4
        ztj = 2 * kJ2 * rjZ / rj ** 5
        alx = -kJ2 * s2i * cth / r ** 5 + 3 * vx * kJ2 * s2i * sth / r ** 4 / h - 8 * kJ2 ** 2 * si ** 3 * ci * sth ** 2 * cth / r ** 6 / h ** 2
        alz = -2 * h * vx / r ** 3 - kJ2 * si ** 2 * s2th / r ** 5
        et = np.sqrt(mu / r ** 3 + kJ2 / r ** 5 - 5 * kJ2 * si ** 2 * sth ** 2 / r ** 5)
        etj = np.sqrt(mu / rj ** 3 + kJ2 / rj ** 5 - 5 * kJ2 * rjZ ** 2 / rj ** 7)
        xjddot = 2 * yjdot * omz - xj * (etj ** 2 - omz ** 2) + yj * alz - zj * omx * omz - (
                    ztj - zt) * si * sth - r * (etj ** 2 - et ** 2) - Cj * vaj * (xjdot - yj * omz) - (
                             Cj * vaj - C * va) * vx - (uR + wc)
        yjddot = -2 * xjdot * omz + 2 * zjdot * omx - xj * alz - yj * (etj ** 2 - omz ** 2 - omx ** 2) + zj * alx - (
                    ztj - zt) * si * cth - Cj * vaj * (yjdot + xj * omz - zj * omx) - (Cj * vaj - C * va) * (
                             h / r - ome * r * ci) - (uT + m1c)
        zjddot = -2 * yjdot * omx - xj * omx * omz - yj * alx - zj * (etj ** 2 - omx ** 2) - (
                    ztj - zt) * ci - Cj * vaj * (zjdot + yj * omx) - (Cj * vaj - C * va) * ome * r * cth * si - uN
        eyeq = np.identity(3)
        Zj = np.zeros((3, 3))
        Bj = np.vstack((Zj, eyeq))
        Lj = np.array([0, 0, 0, 1, 0, 0])
        Pj = np.array([0, 0, 0, 0, 1, 0])
        dXjdt = np.array([xjdot, yjdot, zjdot, xjddot, yjddot, zjddot]) + Bj @ Uj + Lj * m1j + Pj * wj
        return dXjdt



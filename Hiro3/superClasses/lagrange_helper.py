import numpy as np
class lagrange_helper:
    def __init__(self,kJ2,Area,mu,mass,ome,rho,Cd):
        self.kJ2 = kJ2  # self.kJ2 = 2.6330e+10
        self.Area = Area
        self.Areaj = self.Area
        self.mu=mu
        self.mass = mass
        self.massj = self.mass
        self.ome = ome
        self.rho = rho
        self.rhoj = self.rho
        self.Cd = Cd  # self.Cd = 1
        self.Cdj = self.Cd

    def lagrange_helper_function(self, Uc, Xc, Xj, dXcdt):
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
        C, Cj, Omdot, h, i, idot, r, th, thdot, uN, uR, uT, vx, xj, xjdot, yj, yjdot, zj, zjdot = self.get_params_for_Lagrangian(
            Area, Areaj, Cd, Cdj, Uc, Xc, Xj, dXcdt, mass, massj, rho, rhoj)
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
        Mmat = np.identity(3)
        Cmat = -2 * np.array([[0, omz, 0], [-omz, 0, omx], [0, -omx, 0]])
        Gmat = np.array([[etj ** 2 - omz ** 2, -alz, omx * omz], [alz, etj ** 2 - omz ** 2 - omx ** 2, -alx],
                         [omx * omz, alx, etj ** 2 - omx ** 2]]) @ Xj[0:3] + (ztj - zt) * np.array(
            [si * sth, si * cth, ci]) + np.array([r * (etj ** 2 - et ** 2), 0, 0])
        Dmat = -np.array([-Cj * vaj * (xjdot - yj * omz) - (Cj * vaj - C * va) * vx - uR,
                          -Cj * vaj * (yjdot + xj * omz - zj * omx) - (Cj * vaj - C * va) * (h / r - ome * r * ci) - uT,
                          -Cj * vaj * (zjdot + yj * omx) - (Cj * vaj - C * va) * ome * r * cth * si - uN])
        return Cmat, Dmat, Gmat, Mmat

    def get_params_for_Lagrangian(self, Area, Areaj, Cd, Cdj, Uc, Xc, Xj, dXcdt, mass, massj, rho, rhoj):
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
        ############
        Omdot = dXcdt[3]
        idot = dXcdt[4]
        thdot = dXcdt[5]
        return C, Cj, Omdot, h, i, idot, r, th, thdot, uN, uR, uT, vx, xj, xjdot, yj, yjdot, zj, zjdot



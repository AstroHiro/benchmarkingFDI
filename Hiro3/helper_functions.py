import numpy as np
#Add for testing this comment
def rk4(h, t, X, U, dynamics):
    #h = self.h
    k1 = dynamics(t, X, U)
    k2 = dynamics(t + h / 2., X + k1 * h / 2., U)
    k3 = dynamics(t + h / 2., X + k2 * h / 2., U)
    k4 = dynamics(t + h, X + k3 * h, U)
    return t + h, X + h * (k1 + 2. * k2 + 2. * k3 + k4) / 6.


def rad2car(vrad, th):
    Rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    vcar = Rot @ vrad
    return vcar
import numpy as np


class get_initial_params:
    def __init__(self):
        pass

    def get_initial_params_function(self, Nsc, mu):
        # Used in dynamics_helpers
        self.Nsc = Nsc
        self.mu = mu
        self.h = 0.1

        # Used in parameters_helpers
        self.Cj2d = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        self.Pj2d = np.array([[0], [0], [0], [1]])
        self.n2 = 3
        self.Lamdd = 1 * np.identity(3)

        # Used in lagrange_helper
        self.kJ2 = 0  # self.kJ2 = 2.6330e+10
        self.Area = 1
        self.Areaj = self.Area
        self.mu = mu
        self.mass = 1
        self.massj = self.mass
        self.ome = 7.2921e-5
        self.rho = 1e-8
        self.rhoj = self.rho
        self.Cd = 0  # self.Cd = 1
        self.Cdj = self.Cd




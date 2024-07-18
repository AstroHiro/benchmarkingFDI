import numpy as np


def getXd_at_t(self, t):
    Qd_t = self.Qdfun(t)
    Rfd = self.Rfd
    Tall = self.Tall
    p = self.Nsc
    n2 = self.n2
    qd_val = Qd_t[0, :]
    qd_dot_val = Qd_t[1, :]
    qd_ddot_val = Qd_t[2, :]
    R_d = self.R_d
    qd_all = np.tile(qd_val, p)
    qd_dot_all = np.tile(qd_dot_val, p)
    qd_ddot_all = np.tile(qd_ddot_val, p)
    qd_all_trans = R_d.T @ Tall @ R_d @ qd_all
    qd_dot_all_trans = R_d.T @ Tall @ R_d @ qd_dot_all
    qd_ddot_all_trans = R_d @ Tall @ R_d @ qd_ddot_all
    # qd_all_trans = np.linalg.solve(R_d,Tall@R_d@qd_all)
    # qd_dot_all_trans = np.linalg.solve(R_d,Tall@R_d@qd_dot_all)
    # qd_ddot_all_trans = np.linalg.solve(R_d,Tall@R_d@qd_ddot_all)
    Xd = self.Rqq2X @ np.hstack((qd_all_trans, qd_dot_all_trans))
    return qd_all_trans, qd_dot_all_trans, qd_ddot_all_trans, Xd


def relativeA(self, Xj, Xc, Uc):
    dx = 0.001
    A = np.zeros((self.n_states, self.n_states))
    for i in range(self.n_states):
        ei = np.zeros(self.n_states)
        ei[i] = 1
        f1 = self.relativef(Xj + ei * dx, Xc, Uc)
        f0 = self.relativef(Xj - ei * dx, Xc, Uc)
        dfdx = (f1 - f0) / 2 / dx
        A[:, i] = dfdx
    return A
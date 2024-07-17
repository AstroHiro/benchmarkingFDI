import numpy as np
from Hiro3.helper_functions import rk4
import time
def initial_condition(bm):
    t0 = 0
    r0 = 6378 + 1000
    vx0 = 0
    vth0 = np.sqrt(bm.mu / r0)
    om0 = vth0 / r0
    h0 = om0 * r0 ** 2
    Om0 = np.pi / 3
    i0 = np.pi / 3
    th0 = 0
    Xc0 = np.array([r0, vx0, h0, Om0, i0, th0])
    Xjcall0 = np.hstack((bm.Xjall0, Xc0))
    xi_c0 = h0
    Yj0 = bm.measurement_Xj(t0, bm.Xjall0[0:6])
    Xhatj2d0 = np.array([bm.Xjall0[0], bm.Xjall0[1], bm.Xjall0[3], bm.Xjall0[4]])
    Zc2d0 = np.array([r0, vx0, h0, th0])
    Zj2d0 = Xhatj2d0 - bm.Nj2d @ Yj0
    return Xjcall0, Zc2d0, Zj2d0, om0, t0, xi_c0

def target_traj_for_relative_dynamics(bm, om0):
    global T
    T = 2 * np.pi / om0 / 10
    dt = bm.h
    Nint = int(T / dt)
    Qd_t_his = np.zeros((3, 3, Nint + 1))
    ti = 0
    for i in range(Nint + 1):
        Qd_t_his[:, :, i] = bm.Qdfun(ti)
        ti = ti + dt
    return Nint, Qd_t_his

def integration(Nint, Nsc, Xjcall0, Zc2d0, Zj2d0, t0, xi_c0):
    this = np.zeros(Nint + 1)
    Xjcallhis = np.zeros((Nint + 1, 6 * (Nsc + 1)))
    res_c_his = np.zeros(Nint)
    res_w_his = np.zeros(Nint)
    res_wj_his = np.zeros(Nint)
    m1c_his = np.zeros(Nint)
    wc_his = np.zeros(Nint)
    m1j_his = np.zeros(Nint)
    wj_his = np.zeros(Nint)
    this[0] = t0
    Xjcallhis[0, :] = Xjcall0
    t = t0
    Xjcall = Xjcall0
    xi_c = xi_c0
    Zc2d = Zc2d0
    Zj2d = Zj2d0
    return Xjcall, Xjcallhis, Zc2d, Zj2d, m1c_his, m1j_his, res_c_his, res_wj_his, t, this, wc_his, wj_his, xi_c

def nominal_traj_and_faults(Nint, Nsc, Qd_t_his, Xjcall, Xjcallhis, Zc2d, Zj2d, bm, m1c_his, m1j_his, res_c_his,
                            res_wj_his, t, t0, this, wc_his, wj_his, xi_c):
    global Uc, Xc, m1jcall, wjcall, Yc, Yj
    for k in range(Nint):
        Uc = np.zeros(3)
        Xjall = Xjcall[0:6 * Nsc]
        Xc = Xjcall[6 * Nsc:6 * (Nsc + 1)]
        Qd_t = Qd_t_his[:, :, k]
        Ujall = bm.expcontrol(t, Xjall, Xc, Uc, Qd_t)
        Ujcall = np.hstack((Ujall, Uc))
        m1jcall = np.zeros(Nsc + 1)
        wjcall = np.zeros(Nsc + 1)
        # fault signal
        if 1000 <= k <= 2000:
            wjcall[0] = 0.1
            m1jcall[Nsc] = 0.1
        if 3000 <= k <= 4000:
            wjcall[0] = -0.3
            m1jcall[Nsc] = -0.3
        if 5000 <= k <= 6000:
            wjcall[0] = 0.2
            m1jcall[Nsc] = 0.2
        # disturbance signal
        if 500 <= k <= 1500:
            m1jcall[0] = 0.2
            wjcall[Nsc] = 0.2
        if 2500 <= k <= 3500:
            m1jcall[0] = -0.1
            wjcall[Nsc] = -0.1
        if 4500 <= k <= 5500:
            m1jcall[0] = 0.3
            wjcall[Nsc] = 0.3
        Yc = bm.measurement_Xc(t, Xc)
        Yj = bm.measurement_Xj(t0, Xjall[0:6])
        _ = bm.residual_filter_w_Xc_EKF(t, Zc2d, Yc, Uc)
        Xhatc2d = Zc2d + bm.Nekf @ Yc
        Xhatj2d = Zj2d + bm.Nj2d @ Yj
        res_c_his[k] = Yc[1] - xi_c
        # res_w_his[k] = np.linalg.norm(Xhatc2d-np.array([Xc[0],Xc[1],Xc[2],Xc[5]]))
        res_wj_his[k] = np.linalg.norm(Xhatj2d - np.array([Xjall[0], Xjall[1], Xjall[3], Xjall[4]]))
        m1j_his[k] = m1jcall[0]
        wj_his[k] = wjcall[0]
        m1c_his[k] = m1jcall[Nsc]
        wc_his[k] = wjcall[Nsc]
        dynamics_jc = lambda t, X, U: bm.dynamics_all(t, X, U, m1jcall, wjcall)
        dynamics_rc = lambda t, X, U: bm.residual_filter_Xc(t, X, Yc, U)
        # dynamics_rw = lambda t,X,U: bm.residual_filter_w_Xc_EKF(t,X,Yc,U)
        dynamics_rj = lambda t, X, U: bm.residual_filter_Xj(t, X, Yj, U, Xc, Uc, m1jcall[Nsc], wjcall[Nsc])
        t, Xjcall = rk4(bm.h,t, Xjcall, Ujcall, dynamics_jc)
        _, xi_c = rk4(bm.h,t, xi_c, Uc, dynamics_rc)
        # _,Zc2d = rk4(bm.h,t,Zc2d,Uc,dynamics_rw)
        _, Zj2d = rk4(bm.h,t, Zj2d, Ujall[0:3], dynamics_rj)
        this[k + 1] = t
        Xjcallhis[k + 1, :] = Xjcall




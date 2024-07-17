from benchmark import Benchmark
from plotting import plot_positions,plot_residualfilter_j
from trajectories import initial_condition,integration,target_traj_for_relative_dynamics,nominal_traj_and_faults
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def main_Hiro():
    #global Nsc, bm
    Nsc = 2
    mu_mars = 42828.375816
    mu_earth = 3.9860e+05

    bm = Benchmark(mu_earth, Nsc)

    # initial condition
    Xjcall0, Zc2d0, Zj2d0, om0, t0, xi_c0 = initial_condition(bm)

    # target trajectory for relative dynamics
    Nint, Qd_t_his = target_traj_for_relative_dynamics(bm, om0)

    # integration
    Xjcall, Xjcallhis, Zc2d, Zj2d, m1c_his, m1j_his, res_c_his, res_wj_his, t, this, wc_his, wj_his, xi_c = integration(
        Nint, Nsc, Xjcall0, Zc2d0, Zj2d0, t0, xi_c0)

    # nominal trajectory
    nominal_traj_and_faults(Nint, Nsc, Qd_t_his, Xjcall, Xjcallhis, Zc2d, Zj2d, bm, m1c_his, m1j_his, res_c_his,
                            res_wj_his, t, t0, this, wc_his, wj_his, xi_c)

    # Positions Plot
    Xjallhis, x3dhis, y3dhis = plot_positions(Nsc, Xjcallhis)

    # residual filter plot
    plot_residualfilter_j(Nint, m1c_his, res_c_his, this, wc_his)

    # residual filter j plot
    plot_residualfilter_j(Nint, m1j_his, res_wj_his, this, wj_his)

if __name__ == "__main__":
    main_Hiro()
    plt.show()






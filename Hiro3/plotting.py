import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
def plot_residualfilter_j(Nint, m1j_his, res_wj_his, this, wj_his):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(this[0:Nint], res_wj_his, "C0")
    ax2.plot(this[0:Nint], m1j_his, "C1", linestyle="--")
    ax2.plot(this[0:Nint], wj_his, "C2", linestyle="--")
    ax2.legend(["fault signal", "disturbance signal"])
    ax1.set_xlabel("time")
    ax1.set_ylabel("residual")
    ax2.set_ylabel("fault and disturbance signal")

def get_positions(Nsc, Xjcallhis):
    Xchis = Xjcallhis[:, 6 * Nsc:6 * (Nsc + 1)]
    Xjallhis = Xjcallhis[:, 0:6 * Nsc]
    x3dhis = Xchis[:, 0] * np.cos(Xchis[:, 5])
    y3dhis = Xchis[:, 0] * np.sin(Xchis[:, 5])
    return Xjallhis, x3dhis, y3dhis

def plot_positions(Nsc, Xjcallhis):
    #global p
    Xjallhis, x3dhis, y3dhis = get_positions(Nsc, Xjcallhis)
    plt.figure()
    plt.plot(x3dhis, y3dhis, linestyle=':')
    plt.figure()
    for p in range(Nsc):
        Xjhis = Xjallhis[:, 6 * p:6 * (p + 1)]
        plt.plot(Xjhis[:, 0], Xjhis[:, 1])
    return Xjallhis, x3dhis, y3dhis





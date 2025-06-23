import numpy as np
import matplotlib.pyplot as plt
from tmm.tmm_core import inc_tmm
from n_index import get_n_eff, target_lbd

"""
# Param of the mirror - data from Elia
nh = 2.10   # refrac. index of layer at surface (z = 0)
dh = 77     # thickness [nm]
nl = 1.65   # other layer
dl = 98
"""

def tmm_compute(lbd_arr, t_arr, r_arr, solv="air"):
    nfrequ = len(lbd_arr)
    indices_arr = np.zeros(len(t_arr), dtype=complex)
    c_arr = ['c' for _ in indices_arr]  # 'i' for incoherent or 'c' for 'coherent'
    c_arr[0] = c_arr[-1] = 'i'
    R = np.zeros(nfrequ)
    T = np.zeros(nfrequ)
    for i_l, l in enumerate(lbd_arr):
        for i_r, r in enumerate(r_arr):
            indices_arr[i_r] = get_n_eff(r, l, solv)
        R[i_l], T[i_l] = tmm_get_R_T(l, t_arr, indices_arr, c_arr)
    return R, T

def tmm_get_R_T(wavelength, t_arr, n_arr, c_arr):
    theta = 0.  # incident ray angle [rad]
    s_data = inc_tmm('s', n_arr, t_arr, c_arr, theta, wavelength)
    p_data = inc_tmm('p', n_arr, t_arr, c_arr, theta, wavelength)
    # Keys in s/p_data dictionnary:
    # ['r', 't', 'R', 'T', 'power_entering', 'vw_list', 'kz_list', 'th_list',
    # 'pol', 'n_list', 'd_list', 'th_0', 'lam_vac']
    R = (s_data['R'] + p_data['R']) / 2.
    T = (s_data['T'] + p_data['T']) / 2.
    return R, T

def save_data(lbd_arr, R_arr, T_arr):
    np.savetxt("results/reflectance_transmittance.txt",
               np.transpose([lbd_arr, R_arr, T_arr]), header="lbd[m] R[-] T[-]")

def load_prev_data():
    # Get 'lbd, R, T' from previous execution
    return np.loadtxt("results/reflectance_transmittance.txt", skiprows=1, unpack=True)

def show_R_T_at_target(lbd, R, T, target_lbd):
    for i, l in enumerate(lbd):
        if i == len(lbd)-1 or lbd[i+1] > target_lbd:
            print(f"lbd={l/nm:.1f} nm -> R={R[i]:.4f} ; T={T[i]:.4f}")
            break

def plot_show_R_T(lbd_nm, R, T, fname=None):
    plt.plot(lbd_nm, R, label="R")
    plt.plot(lbd_nm, T, label="T")

    plt.xlabel("lbd [nm]")
    plt.ylabel("r,t")
    plt.legend()
    if fname is not None:
        plt.savefig("results/"+fname)
    plt.show()

if __name__ == '__main__':
    # Param of the wave
    nm = 1e-9
    lbd_nm = np.linspace(450, 850, 201)  # wavelength [m]
    lbd = lbd_nm*nm

    t_layers = [np.inf]
    rho_layers = [1.]
    stack = np.loadtxt("stack.txt", skiprows=1, dtype=complex)
    for layer in stack:
        t_layers.append(layer[0].real)
        rho_layers.append(layer[1])
    # ajouter couche de support: 50-100um ; p=50%
    t_layers.append(np.inf)
    rho_layers.append(1.)

    # Compuations
    R, T = tmm_compute(lbd, t_layers, rho_layers)
    show_R_T_at_target(lbd, R, T, target_lbd)
    save_data(lbd, R, T)
    plot_show_R_T(lbd_nm, R, T, "refl_40_layers_P=0.4,0.7_inc.png")

"""
N = 10:
Found thicknesses of 75.238nm (P=0.59, n=1.994) and 93.338nm (P=0.75, n=1.607)
As a result: 
R=94.775% in a windows of 82.023nm
Parameters for porosification process:
I1=14.850 mA - dt1=1.475256118257249 s
I1=43.031 mA - dt1=0.718327442679487 s

N = 40:
Found thicknesses of 75.238nm (P=0.59, n=1.994) and 93.338nm (P=0.75, n=1.607)
As a result: 
R=100.000% in a windows of 82.023nm
Parameters for porosification process:
I1=14.850 mA - dt1=1.475256118257249 s
I1=43.031 mA - dt1=0.718327442679487 s

N = 40:
Found thicknesses of 61.156nm (P=0.4, n=2.453) and 86.811nm (P=0.7, n=1.728)
As a result: 
R=100.000% in a windows of 132.460nm
Parameters for porosification process:
I1=2.700 mA - dt1=6.115550738204867 s
I1=34.200 mA - dt1=0.8138577628846807 s
"""


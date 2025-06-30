import numpy as np
import matplotlib.pyplot as plt
from tmm.tmm_core import inc_tmm, power_entering_from_r
from n_index import get_n_eff, target_lbd

"""
# Param of the mirror - data from Elia
nh = 2.10   # refrac. index of layer at surface (z = 0)
dh = 77     # thickness [nm]
nl = 1.65   # other layer
dl = 98
"""
stacks = None
inc_layers = None

M_interface = None
M_travel = None
M_layer = None

L_travel = None
L_interface = None
L_stack = None
L_equ = None
def allocate_matrices(c_arr):
    global M_interface
    M_interface = np.empty((len(c_arr), 2, 2))

def load(matrix, a, b, c, d):
    matrix[0,0] = a
    matrix[0,1] = b
    matrix[1,0] = c
    matrix[1,1] = d

def set_stacks()

def coh_R_T(pol, n_arr, t_arr, lbd_0):
    # (1  r) = M_ (t  0)
    # Ef1 = M_11 Ef2 + M_12 Eb2
    # Eb1 = M_21 Ef2 + M_22 Eb2
    pol_fact = 1.
    if pol == "p":
        pol_fact = -1.
    elif pol != "s":
        raise ValueError(f"Polarization must be 's' or 'p', got {pol}")
    N = len(t_arr)
    assert (N >= 2) and (N == len(n_arr))  # at least 2 layers to get an interface
    # between layer 0 and 1
    n1 = n_arr[0]
    n2 = n_arr[1]
    r = pol_fact*(n1-n2)/(n1+n2)
    t = 2*n1/(n1+n2)
    M_interface_0 = np.matrix([[1, r], [r, 1]], dtype=complex) / t
    M_layer = [M_interface_0]
    for i in range(1, N-1):
        # inside layer i
        kz = 2 * np.pi * n_arr[i] / lbd_0
        delta = t_arr[i]*kz  # phase shift inside layer i
        M_travel_i = np.matrix([[np.exp(-1j*delta), 0], [0, np.exp(1j*delta)]])
        # between layer i and i+1
        n1 = n_arr[i]
        n2 = n_arr[i+1]
        r = pol_fact*(n1-n2)/(n1+n2)
        t = 2*n1/(n1+n2)
        M_interface_i = np.matrix([[1, r], [r, 1]]) / t
        M_layer.append(np.dot(M_travel_i, M_interface_i))

    M_equ = M_layer[0]
    for M in M_layer[1:N-1]:
        M_equ = np.dot(M_equ, M)
    r = M_equ[1, 0]/M_equ[0, 0]
    t = 1./M_equ[0, 0]
    T_fact = n_arr[N-1].real/n_arr[0].real
    R = r.real*r.real+r.imag*r.imag
    T = (t.real*t.real+t.imag*t.imag)*T_fact
    return R, T

def nat_get_R_T_with_pol(pol, n_arr, t_arr, c_arr, theta=0, lbd_0=633e-9):
    # @PRE: infinite non-absorbing incoherent first and last layers, theta=0.+0.j
    if isinstance(n_arr, list):
        n_arr = np.array(n_arr, dtype=complex)
    if isinstance(t_arr, list):
        t_arr = np.array(t_arr)
    assert (len(n_arr) >= 2) and (len(n_arr) == len(t_arr) == len(c_arr))
    stacks = []
    stack_ind = [0]
    # power going through a layer (not absorbed): from layer 1 to N-1
    P_list = [1.]  # No loss before 2nd layer
    for i in range(1, len(c_arr)):
        stack_ind.append(i)
        if c_arr[i] == 'i':
            stacks.append(np.array(stack_ind))
            if i != len(c_arr)-1:  # we don't compute losses and reflection in/after last layer
                P = np.exp(-4*np.pi*n_arr[i].imag*t_arr[i]/lbd_0)
                if P < 1e-30:  # Avoid div by 0 errors (opaque layer)
                    P = 1e-30
                P_list.append(P)
            else:
                P_list.append(1.)
            stack_ind = [i]
    # Calculating power, R, T (used for incoherent layers)
    T_list = []  # power transmitted or reflected at each interface
    R_list = []
    T_inv_list = []
    R_inv_list = []
    for stack_ind in stacks:
        R, T = coh_R_T(pol, n_arr[stack_ind], t_arr[stack_ind], lbd_0)
        R_list.append(R)
        T_list.append(T)
    for stack_ind in stacks:
        inv_stack_ind = stack_ind[::-1]
        R, T = coh_R_T(pol, n_arr[inv_stack_ind], t_arr[inv_stack_ind], lbd_0)
        R_inv_list.append(R)
        T_inv_list.append(T)
    R_inv_list = R_inv_list
    T_inv_list = T_inv_list

    # Computing matrices for power transmission
    L_list = []
    R_diff = T_list[0]*T_inv_list[0]  - R_list[0]*R_inv_list[0]
    L_equ = np.matrix([[1, -R_inv_list[0]], [R_list[0], R_diff]]) / T_list[0]
    L_list.append(L_equ)
    for i in range(1, len(P_list)-1):
        # Losses in layer incoherent_layers_ind[i] and reflection between this one and next incoherent layer
        L_travel_i = np.matrix([[1./P_list[i], 0], [0, P_list[i]]])
        R_diff = T_list[i]*T_inv_list[i] - R_list[i]*R_inv_list[i]
        L_interface_i = np.matrix([[1, -R_inv_list[i]], [R_list[i], R_diff]]) / T_list[i]
        L_i = np.dot(L_travel_i, L_interface_i)
        L_equ = np.dot(L_equ, L_i)
        L_list.append(L_i)
    return {'R':L_equ[1, 0]/L_equ[0, 0], 'T':1./L_equ[0, 0]}

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
    s_data = nat_get_R_T_with_pol('s', n_arr, t_arr, c_arr, theta, wavelength)  # inc_tmm nat_get_R_T_with_pol
    p_data = nat_get_R_T_with_pol('p', n_arr, t_arr, c_arr, theta, wavelength)
    # Keys in s/p_data dictionnary:
    # ['r', 't', 'R', 'T', 'power_entering', 'vw_list', 'kz_list', 'th_list',
    # 'pol', 'n_list', 'd_list', 'th_0', 'lam_vac']
    R = (s_data['R'] + p_data['R']) / 2.
    T = (s_data['T'] + p_data['T']) / 2.
    return R, T

def save_data(lbd_arr, R_arr, T_arr, fname=None):
    if fname is None:
        fname = "reflectance_transmittance.txt"
    np.savetxt("temp/"+fname,
               np.transpose([lbd_arr, R_arr, T_arr]), header="lbd[m] R[-] T[-]")

def load_prev_data(fname=None):
    if fname is None:
        fname = "reflectance_transmittance.txt"
    # Get 'lbd, R, T' from previous execution
    return np.loadtxt("temp/"+fname, skiprows=1, unpack=True)

def show_R_T_at_target(lbd, R, T, target_lbd):
    for i, l in enumerate(lbd):
        if i == len(lbd)-1 or lbd[i+1] > target_lbd:
            print(f"lbd={l/nm:.1f} nm -> R={R[i]:.4f} ; T={T[i]:.4f}")
            break

def plot_show_R_T(lbd_nm, R, T, fname=None):
    plt.figure()
    plt.plot(lbd_nm, R, label="R")
    plt.plot(lbd_nm, T, label="T")

    plt.xlabel("$\\lambda$ [nm]")
    plt.ylabel("R, T [-]")
    plt.legend()
    if fname is not None:
        plt.savefig("results/"+fname)
    plt.show()

def plot_compare_R(lbd_nm, R_ref, R_test, fname=None):
    err = np.abs(R_ref - R_test)
    # default fig size: [6.4, 4.8]
    fig, ax = plt.subplots(1, 2, sharey=False, figsize=(10., 4.8))
    ax[0].plot(lbd_nm, R_ref, label="Model")
    ax[0].plot(lbd_nm, R_test, label="Measurement")
    ax[1].plot(lbd_nm, err)

    ax[0].set_xlabel("$\\lambda$ [nm]")
    ax[0].set_ylabel("R [-]")
    ax[1].set_xlabel("$\\lambda$ [nm]")
    ax[1].set_ylabel("Absolute error [-]")
    ax[0].legend()
    if fname is not None:
        fig.savefig("results/" + fname)
    fig.show()

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


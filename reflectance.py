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

class Optical_1D_tmm_simulator:
    def __init__(self):
        return

    def compute_single_R_T(self, wavelength, t_arr, n_arr, c_arr):
        theta = 0.  # incident ray angle [rad]
        s_data = inc_tmm('s', n_arr, t_arr, c_arr, theta, wavelength)  # nat_get_R_T_with_pol
        p_data = inc_tmm('p', n_arr, t_arr, c_arr, theta, wavelength)
        R = (s_data['R'] + p_data['R']) / 2.
        T = (s_data['T'] + p_data['T']) / 2.
        return R, T

    def compute_R_T_spectrum(self, lbd_arr, stack, c_arr):
        nfrequ = len(lbd_arr)
        R_arr = np.zeros(nfrequ)
        T_arr = np.zeros(nfrequ)
        for i_l, l in enumerate(lbd_arr):
            t_arr, n_arr = stack.get_t_n_arrays(l)
            R_arr[i_l], T_arr[i_l] = self.compute_single_R_T(l, t_arr, n_arr, c_arr)
        return R_arr, T_arr


def load(matrix, a, b, c, d):
    matrix[0,0] = a
    matrix[0,1] = b
    matrix[1,0] = c
    matrix[1,1] = d

class Optical_1D_local_simulator(Optical_1D_tmm_simulator):
    def __init__(self):
        self.stacks = self.inc_layers = self.P_array = None
        self.T_array = self.R_array = self.T_inv_array = self.R_inv_array = None
        self.M_interface = np.empty((2, 2), dtype=complex)
        self.M_travel = np.empty((2, 2), dtype=complex)
        self.M_temp = np.empty((2, 2), dtype=complex)
        self.M_equ = np.empty((2, 2), dtype=complex)

        self.L_travel = np.empty((2, 2))
        self.L_interface = np.empty((2, 2))
        self.L_temp = np.empty((2, 2))
        self.L_equ = np.empty((2, 2))

    def compute_single_R_T(self, wavelength, t_arr, n_arr, c_arr):
        theta = 0.  # incident ray angle [rad]
        s_data = self.inc_local_tmm('s', n_arr, t_arr, c_arr, theta, wavelength)
        p_data = self.inc_local_tmm('p', n_arr, t_arr, c_arr, theta, wavelength)
        R = (s_data['R'] + p_data['R']) / 2.
        T = (s_data['T'] + p_data['T']) / 2.
        return R, T

    def compute_R_T_spectrum(self, lbd_arr, stack, c_arr):
        self.set_stacks(c_arr)
        return super().compute_R_T_spectrum(lbd_arr, stack, c_arr)

    def set_stacks(self, c_arr):
        self.M_layer = np.empty((len(c_arr)-1, 2, 2), dtype=complex)
        self.stacks = []
        stack_ind = [0]
        self.inc_layers = [0]
        for i in range(1, len(c_arr)):
            stack_ind.append(i)
            if c_arr[i] == 'i':
                self.inc_layers.append(i)
                self.stacks.append(np.array(stack_ind))
                stack_ind = [i]
        self.P_array = np.empty(len(self.inc_layers))
        self.T_array = np.empty(len(self.inc_layers))
        self.R_array = np.empty(len(self.inc_layers))
        self.T_inv_array = np.empty(len(self.inc_layers))
        self.R_inv_array = np.empty(len(self.inc_layers))

    def coh_R_T(self, pol_fact, n_arr, t_arr, lbd_0):
        # (1  r) = M_ (t  0)
        # Ef1 = M_11 Ef2 + M_12 Eb2
        # Eb1 = M_21 Ef2 + M_22 Eb2
        N = len(t_arr)
        assert (N >= 2) and (N == len(n_arr))  # at least 2 layers to get an interface
        # between layer 0 and 1
        n1 = n_arr[0]
        n2 = n_arr[1]
        r = pol_fact*(n1-n2)/(n1+n2)
        t = 2*n1/(n1+n2)
        load(self.M_temp, 1, r, r, 1)
        self.M_equ = self.M_temp / t
        #M_interface_0 = np.matrix([[1, r], [r, 1]], dtype=complex) / t
        #M_layer = [M_interface_0]
        for i in range(1, N-1):
            # inside layer i
            kz = 2 * np.pi * n_arr[i] / lbd_0
            delta = t_arr[i]*kz  # phase shift inside layer i
            #M_travel_i = np.matrix([[np.exp(-1j*delta), 0], [0, np.exp(1j*delta)]])
            load(self.M_travel, np.exp(-1j*delta), 0, 0, np.exp(1j*delta))
            # between layer i and i+1
            n1 = n_arr[i]
            n2 = n_arr[i+1]
            r = pol_fact*(n1-n2)/(n1+n2)
            t = 2*n1/(n1+n2)
            #M_interface_i = np.matrix([[1, r], [r, 1]]) / t
            load(self.M_interface, 1, r, r, 1)
            #M_layer.append(np.dot(M_travel_i, M_interface_i))
            np.dot(self.M_equ, self.M_travel, out=self.M_temp)
            np.dot(self.M_temp, self.M_interface / t, out=self.M_equ)

        #M_equ = M_layer[0]
        #for M in M_layer[1:N-1]:
        #    M_equ = np.dot(M_equ, M)
        r = self.M_equ[1, 0]/self.M_equ[0, 0]
        t = 1./self.M_equ[0, 0]
        T_fact = n_arr[N-1].real/n_arr[0].real
        R = r.real*r.real+r.imag*r.imag
        T = (t.real*t.real+t.imag*t.imag)*T_fact
        return R, T

    def inc_local_tmm(self, pol, n_arr, t_arr, c_arr, theta=0, lbd_0=633e-9):
        # @PRE: infinite non-absorbing incoherent first and last layers, theta=0.+0.j
        if isinstance(n_arr, list):
            n_arr = np.array(n_arr, dtype=complex)
        if isinstance(t_arr, list):
            t_arr = np.array(t_arr)
        assert len(n_arr) == len(t_arr) == len(c_arr) >= 2
        assert ((t_arr[0] == t_arr[-1] == np.inf) and (c_arr[0] == c_arr[-1] == 'i') and
                (n_arr[0].imag == n_arr[-1].imag == 0.) and (theta == 0.))
        pol_fact = 1.
        if pol == "p":
            pol_fact = -1.
        elif pol != "s":
            raise ValueError(f"Polarization must be 's' or 'p', got {pol}")
        #stacks = []
        #stack_ind = [0]
        # power going through a layer (not absorbed): from layer 1 to N-1
        #P_list = [1.]  # No loss before 2nd layer
        for i_stack, i_layer in enumerate(self.inc_layers):
            if (i_layer == 0) or (i_layer == len(t_arr)-1):  # infinite layer with no loss
                self.P_array[i_stack] = 1.
            else:
                self.P_array[i_stack] = np.exp(-4*np.pi*n_arr[i_layer].imag*t_arr[i_layer]/lbd_0)
                if self.P_array[i_stack] < 1e-30:  # Avoid div by 0 errors (opaque layer)
                    self.P_array[i_stack] = 1e-30
        # Calculating power, R, T (used for incoherent layers)
        #T_list = []  # power transmitted or reflected at each interface
        #R_list = []
        #T_inv_list = []
        #R_inv_list = []
        for stack_ind, layers_ind in enumerate(self.stacks):
            R, T = self.coh_R_T(pol_fact, n_arr[layers_ind], t_arr[layers_ind], lbd_0)
            self.R_array[stack_ind] = R
            self.T_array[stack_ind] = T

            inv_layer_ind = layers_ind[::-1]
            R_inv, T_inv = self.coh_R_T(pol_fact, n_arr[inv_layer_ind], t_arr[inv_layer_ind], lbd_0)
            self.R_inv_array[stack_ind] = R_inv
            self.T_inv_array[stack_ind] = T_inv

        # Computing matrices for power transmission
        R_diff = self.T_array[0]*self.T_inv_array[0]  - self.R_array[0]*self.R_inv_array[0]
        #L_equ = np.matrix([[1, -R_inv_list[0]], [R_list[0], R_diff]]) / T_list[0]
        load(self.L_temp, 1, -self.R_inv_array[0], self.R_array[0], R_diff)
        self.L_equ = self.L_temp / self.T_array[0]
        #L_list.append(L_equ)
        for i in range(1, len(self.inc_layers)-1):
            # Losses in layer incoherent_layers_ind[i] and reflection between this one and next incoherent layer
            #L_travel_i = np.matrix([[1./self.P_array[i], 0], [0, self.P_array[i]]])
            load(self.L_travel, 1./self.P_array[i], 0, 0, self.P_array[i])
            R_diff = self.T_array[i]*self.T_inv_array[i] - self.R_array[i]*self.R_inv_array[i]
            #L_interface_i = np.matrix([[1, -R_inv_list[i]], [R_list[i], R_diff]]) / self.T_array[i]
            load(self.L_interface, 1, -self.R_inv_array[i], self.R_array[i], R_diff)
            #L_i = np.dot(L_travel_i, L_interface_i)
            #L_equ = np.dot(L_equ, L_i)
            #L_list.append(L_i)
            np.dot(self.L_equ, self.L_travel, out=self.L_temp)
            np.dot(self.L_temp, self.L_interface / self.T_array[i], out=self.L_equ)
        return {'R':self.L_equ[1, 0]/self.L_equ[0, 0], 'T':1./self.L_equ[0, 0]}


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

def show_R_spectra(lbd_nm, R_list, labels, fname):
    plt.figure()
    for i, R in enumerate(R_list):
        plt.plot(lbd_nm, R, label=labels[i])
    plt.xlabel("$\\lambda$ [nm]")
    plt.ylabel("R [-]")
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
    c_layers = ['c' for _ in t_layers]  # 'i' for incoherent or 'c' for 'coherent'
    c_layers[0] = c_layers[-1] = 'i'

    # Compuations
    nfrequ = len(lbd)
    indices_arr = np.zeros(len(t_layers), dtype=complex)
    R = np.zeros(nfrequ)
    T = np.zeros(nfrequ)
    for i_l, l in enumerate(lbd):
        for i_r, r in enumerate(rho_layers):
            indices_arr[i_r] = get_n_eff(r, l)
        R[i_l], T[i_l] = tmm_get_R_T(l, t_layers, indices_arr, c_layers)
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


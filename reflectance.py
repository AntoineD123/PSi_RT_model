import numpy as np
import matplotlib.pyplot as plt

from .n_index import get_n_eff, pkg_loc


# Fast assignations
def load(matrix, a, b, c, d):
    matrix[0,0] = a
    matrix[0,1] = b
    matrix[1,0] = c
    matrix[1,1] = d
def copy(matrix1, m_content):
    matrix1[0,0] = m_content[0,0]
    matrix1[0,1] = m_content[0,1]
    matrix1[1,0] = m_content[1,0]
    matrix1[1,1] = m_content[1,1]

# Class used to simulate reflectance with the transfer matrix method
# Only 'compute_R_T_spectrum' function should be used.
class Optical_1D_local_tmm:
    def __init__(self):
        # List of stacks and related properties
        self.stacks = self.inc_layers = self.P_array = None
        self.T_array = self.R_array = self.T_inv_array = self.R_inv_array = None
        # Matrices used during computations
        self.M_interface = np.empty((2, 2), dtype=complex)
        self.M_travel = np.empty((2, 2), dtype=complex)
        self.M_temp = np.empty((2, 2), dtype=complex)
        self.M_equ = np.empty((2, 2), dtype=complex)

        self.L_travel = np.empty((2, 2))
        self.L_interface = np.empty((2, 2))
        self.L_temp = np.empty((2, 2))
        self.L_equ = np.empty((2, 2))

    def compute_single_R_T(self, wavelength, t_arr, n_arr, c_arr):
        # Similar to 'compute_R_T_spectrum', for a single wavelength.
        # Function 'set_stacks' must be called first.
        theta = 0.  # incident ray angle [rad]
        s_data = self.inc_local_tmm('s', n_arr, t_arr, c_arr, theta, wavelength)
        p_data = self.inc_local_tmm('p', n_arr, t_arr, c_arr, theta, wavelength)
        R = (s_data['R'] + p_data['R']) / 2.
        T = (s_data['T'] + p_data['T']) / 2.
        return R, T

    def compute_R_T_spectrum(self, lbd_arr, stack, c_arr):
        # @PRE:
        #       lbd_arr: array of wavelengths of incident light (used to compute the output)
        #       stack: object defining the stack ; provides thickness and refractive index of each layer
        #              for any input wavelength 'l' with the method 'stack.get_t_n_arrays(l)'
        #       c_arr: array defining the coherence for each layer ('c' for cohrerent or 'i' for incoherent)
        #       first and last layers in the stack are infinite and incoherent
        # @POST:
        #      R, T: arrays with values of reflectance and transmittance for each wavelengths,
        #            for a beam perpendicular to the surface
        self.set_stacks(c_arr)
        nfrequ = len(lbd_arr)
        R_arr = np.zeros(nfrequ)
        T_arr = np.zeros(nfrequ)
        for i_l, l in enumerate(lbd_arr):
            t_arr, n_arr = stack.get_t_n_arrays(l)
            R_arr[i_l], T_arr[i_l] = self.compute_single_R_T(l, t_arr, n_arr, c_arr)
        return R_arr, T_arr

    def set_stacks(self, c_arr):
        # Identify coherent sub-stacks and array allocation
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
        self.T_array = np.empty(len(self.inc_layers)-1)   # power transmitted or reflected at each interface
        self.R_array = np.empty(len(self.inc_layers)-1)
        self.T_inv_array = np.empty(len(self.inc_layers)-1)  # same in opposite direction
        self.R_inv_array = np.empty(len(self.inc_layers)-1)

    def coh_R_T(self, pol_fact, n_arr, t_arr, lbd_0):
        # Retrieve reflectance and transmittance of a single stack
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
        copy(self.M_equ, self.M_temp / t)
        for i in range(1, N-1):
            # inside layer i
            kz = 2 * np.pi * n_arr[i] / lbd_0
            delta = t_arr[i]*kz  # phase shift inside layer i
            load(self.M_travel, np.exp(-1j*delta), 0, 0, np.exp(1j*delta))
            # between layer i and i+1
            n1 = n_arr[i]
            n2 = n_arr[i+1]
            r = pol_fact*(n1-n2)/(n1+n2)
            t = 2*n1/(n1+n2)
            load(self.M_interface, 1, r, r, 1)
            np.dot(self.M_equ, self.M_travel, out=self.M_temp)
            np.dot(self.M_temp, self.M_interface / t, out=self.M_equ)

        r = self.M_equ[1, 0]/self.M_equ[0, 0]
        t = 1./self.M_equ[0, 0]
        T_fact = n_arr[N-1].real/n_arr[0].real
        R = r.real*r.real+r.imag*r.imag
        T = (t.real*t.real+t.imag*t.imag)*T_fact
        return R, T

    def inc_local_tmm(self, pol, n_arr, t_arr, c_arr, theta=0., lbd_0=633e-9):
        # similar to 'inc_tmm' in tmmpy library (reduced version)
        # @PRE:
        #     infinite non-absorbing incoherent first and last layers, theta=0.+0.j
        # @POST:
        #     dict with the values of reflectance and transmittance of the incoherent stack.
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
        # power going through a layer (not absorbed): from layer 1 to N-1
        for i_stack, i_layer in enumerate(self.inc_layers):
            if (i_layer == 0) or (i_layer == len(t_arr)-1):  # infinite layer with no loss
                self.P_array[i_stack] = 1.
            else:
                self.P_array[i_stack] = np.exp(-4*np.pi*n_arr[i_layer].imag*t_arr[i_layer]/lbd_0)
                if self.P_array[i_stack] < 1e-30:  # Avoid div by 0 errors (opaque layer)
                    self.P_array[i_stack] = 1e-30
        # Calculating power, R, T (used for incoherent layers)
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
        load(self.L_temp, 1, -self.R_inv_array[0], self.R_array[0], R_diff)
        copy(self.L_equ, self.L_temp / self.T_array[0])
        for i in range(1, len(self.inc_layers)-1):
            # Losses in layer incoherent_layers_ind[i] and reflection between this one and next incoherent layer
            load(self.L_travel, 1./self.P_array[i], 0, 0, self.P_array[i])
            R_diff = self.T_array[i]*self.T_inv_array[i] - self.R_array[i]*self.R_inv_array[i]
            load(self.L_interface, 1, -self.R_inv_array[i], self.R_array[i], R_diff)
            np.dot(self.L_equ, self.L_travel, out=self.L_temp)
            np.dot(self.L_temp, self.L_interface / self.T_array[i], out=self.L_equ)
        return {'R':self.L_equ[1, 0]/self.L_equ[0, 0], 'T':1./self.L_equ[0, 0]}


# Functions used to manage inputs/outputs
def save_data(lbd_arr, R_arr, T_arr, fname=None):
    # Save the reflectance/transmittance spectrum with the given (or default) name.
    if fname is None:
        fname = "reflectance_transmittance.txt"
    np.savetxt(pkg_loc.joinpath("temp/"+fname),
               np.transpose([lbd_arr, R_arr, T_arr]), header="lbd[m] R[-] T[-]")

def load_prev_data(fname=None):
    # Load a reflectance spectrum that was saved with the given (or default) name.
    if fname is None:
        fname = "reflectance_transmittance.txt"
    # Get 'lbd, R, T' from previous execution
    return np.loadtxt(pkg_loc.joinpath("temp/"+fname), skiprows=1, unpack=True)

def show_R_T_at_target(lbd, R, T, target_lbd):
    # Print values of 'R' and 'T' at 'lbd'='target_lbd'.
    # 'R', 'T' and 'lbd' are lists of the same size and 'target_lbd' is float.
    for i, l in enumerate(lbd):
        if i == len(lbd)-1 or lbd[i+1] > target_lbd:
            print(f"lbd={l/nm:.1f} nm -> R={R[i]:.4f} ; T={T[i]:.4f}")
            break

def plot_show_R_T(lbd_nm, R, T, fname=None):
    # Plot reflectance and transmittance and show figure.
    # 'fname': name of the output image to save or None (not saved)
    plt.figure()
    plt.plot(lbd_nm, R, label="R")
    plt.plot(lbd_nm, T, label="T")

    plt.xlabel("$\\lambda$ [nm]")
    plt.ylabel("R, T [-]")
    plt.legend()
    if fname is not None:
        plt.savefig(pkg_loc.joinpath("results/"+fname))
    plt.show()

def plot_those_R(lbd, R_list, lab=None, fname=None, intensity=False):
    # R_list: list of arrays.
    # Plot the different curbed in R_list, with x-axis lbd.
    # lab is a list of labels, fname is the name of the image to save
    # or None is the output is not saved and intensity defined the units of the plot:
    # True -> number of counts in the spectrum and False -> reflectance with no units.
    lbd_nm = lbd * 1e9
    plt.figure()
    plt.xlabel("$\\lambda$ [nm]")
    if intensity:
        plt.ylabel("Intensity [cnt]")
    else:
        plt.ylabel("R [-]")
    for i, R in enumerate(R_list):
        if lab is None:
            plt.plot(lbd_nm, R, label=f"signal {i+1}")
        else:
            plt.plot(lbd_nm, R, label=lab[i])
    if len(R_list) > 1:
        plt.legend()
    if fname is not None:
        plt.savefig(pkg_loc.joinpath("results/" + fname))
    plt.show()


if __name__ == '__main__':
    # Testing the code
    # Param of the wave
    target_lbd = 633e-9
    nm = 1e-9
    lbd_nm = np.linspace(450, 850, 201)  # wavelength [m]
    lbd = lbd_nm*nm

    # Build stack
    t_layers = [np.inf]
    p_layers = [1.]
    pA, pB = 0.5, 0.7
    tA, tB = 61.4e-9, 79.1e-9
    for _ in range(10):
        t_layers.append(tA)
        p_layers.append(pA)
        t_layers.append(tB)
        p_layers.append(pB)
    t_layers.append(np.inf)
    p_layers.append(1.)
    c_layers = ['c' for _ in t_layers]  # 'i' for incoherent or 'c' for 'coherent'
    c_layers[0] = c_layers[-1] = 'i'

    # Computations
    model = Optical_1D_local_tmm()
    model.set_stacks(c_layers)
    nfrequ = len(lbd)
    indices_arr = np.zeros(len(t_layers), dtype=complex)
    R = np.zeros(nfrequ)
    T = np.zeros(nfrequ)
    for i_l, l in enumerate(lbd):
        for i_r, r in enumerate(p_layers):
            indices_arr[i_r] = get_n_eff(r, l, "air")
        R[i_l], T[i_l] = model.compute_single_R_T(l, t_layers, indices_arr, c_layers)
    show_R_T_at_target(lbd, R, T, target_lbd)
    save_data(lbd, R, T)
    plot_show_R_T(lbd_nm, R, T, "refl_40_layers_P=0.4,0.7_inc.png")



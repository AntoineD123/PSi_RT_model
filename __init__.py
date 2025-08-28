# Package used to compute the reflectance spectrum of a porous silicon (PSi)
# Bragg stack, with the transfer matrix method (TMM), in 1D.

import os.path
import numpy as np
from n_index import get_n_eff, pkg_loc
from reflectance import Optical_1D_local_tmm, save_data, load_prev_data, plot_show_R_T, plot_those_R
from set_layers import compute_t, get_opt_porosities
from etch_recipe import load_values, get_I_dt


# External function (used to show the results):
#     plot_those_R(lbd, R_list, lab=None, fname=None, intensity=False):


# Definition of the classes of this package:
# 1) for managing the optical simulation of a PSi multilayered stack (1D problem)
class PSiReflectanceSimulator:
    # default range and precision of the spectrum
    lbd_min_nm = 450
    lbd_max_nm = 800
    n_default_wavelength = 351
    # values computed in class function 'extract_peak_info'
    peak_info_str = "R_p [%], FWHM [nm], lbd_p [nm], lbd_c [nm]"

    def __init__(self, stack):
        # stack is an instance of 'PSiOpticalStack'
        self.stack = stack
        self.simulator = Optical_1D_local_tmm()
        # values of the spectrum: wavelengths, transmittance, reflectance
        self.lbd = self.R = self.T = None

    def compute_refl_spectrum(self, wavelengths=None):
        # Compute the reflectance and transmittance of the stack, at the
        # given or default wavelength (list).
        self.lbd = wavelengths
        if wavelengths is None:
            self.lbd = np.linspace(self.lbd_min_nm, self.lbd_max_nm, self.n_default_wavelength)*1e-9
        c_arr = self.stack.get_c_array()
        self.R, self.T = self.simulator.compute_R_T_spectrum(self.lbd, self.stack, c_arr)
        return self.R, self.T

    def save_spectrum(self, fname=None):
        # Save wavelength, reflectance and transmittance data.
        save_data(self.lbd, self.R, self.T, fname)

    def load_spectrum(self, fname=None):
        # Load wavelength, reflectance and transmittance data that were saved
        # with the same (or default) name.
        self.lbd, self.R, self.T = load_prev_data(fname)

    def load_from_ocean_view(self, fname, intensity=True, header=True):
        # Load data in text file produced with OceanView program.
        # The file location is provided. It can contain values of reflectance (in %)
        # or intensity (in counts) for different wavelengths. These are used to define
        # 'self.lbd' and 'self.R'.
        # By default, it is saved with 14 lines header,
        # that describes the content. Otherwise, header must be set to False.
        # All values outside of the default range ('lbd_min_nm' to 'lbd_max_nm') are removed.
        if header:
            lbd_nm, R_large = np.loadtxt(fname, skiprows=14, unpack=True)
        else:
            lbd_nm, R_large = np.loadtxt(fname, unpack=True)
        c1 = self.lbd_min_nm < lbd_nm
        c2 = lbd_nm < self.lbd_max_nm
        ind_to_keep = np.where(np.logical_and(c1, c2))
        self.lbd = lbd_nm[ind_to_keep]*1e-9
        self.R = R_large[ind_to_keep]
        self.T = np.zeros(len(self.R))
        if not intensity:
            self.R /= 100.
        return self.lbd, self.R

    def merge_ocean_view_data(self, flist, output_fname=None, index_range=None, **kwargs):
        # Same as 'load_from_ocean_view' with default parameters, but expects a list of files,
        # or a repository (containing the files). Computes the average spectrum between these.
        # The reflectance (or intensity) spectrum can be saved.
        # 'index_range' is a range of indices that can be defined to only use a subset
        # of the files given as an input.
        if type(flist) == str and os.path.isdir(flist):
            flist = [os.path.join(flist, f) for f in os.listdir(flist) if f.endswith(".txt")]
        if index_range is not None:
            flist = flist[index_range[0]:index_range[1]]
        lbd_nm, R_large = np.loadtxt(flist[0], skiprows=14, unpack=True)
        for fname in flist[1:]:
            lbd_temp, R_temp = np.loadtxt(fname, skiprows=14, unpack=True)
            assert (lbd_nm == lbd_temp).all()
            R_large += R_temp
        R_large /= len(flist)
        c1 = self.lbd_min_nm < lbd_nm
        c2 = lbd_nm < self.lbd_max_nm
        ind_to_keep = np.where(np.logical_and(c1, c2))
        self.lbd = lbd_nm[ind_to_keep]*1e-9
        self.R = R_large[ind_to_keep]
        self.T = np.zeros(len(self.R))
        if output_fname is not None:
            self.save_spectrum(output_fname)
        return flist

    def show_R_T(self, image_name=None):
        # Plot and show the reflectance/transmittance spectrum
        # that was computed.
        lbd_nm = self.lbd * 1e9
        plot_show_R_T(lbd_nm, self.R, self.T, image_name)

    def get_var(self, R_to_compare=None, lbd_lim=None):
        # Returns the variance of this signal, or the average squared error
        # compared to the model 'R_to_compare' (list to compare with 'self.R'),
        # in the interval (tuple) given as an argument (or on the whole spectrum).
        if lbd_lim is None:
            ind_to_keep = slice(0, len(self.R))
        else:
            c1 = lbd_lim[0] < self.lbd
            c2 = self.lbd < lbd_lim[1]
            ind_to_keep = np.where(np.logical_and(c1, c2))
        R1 = self.R[ind_to_keep]
        if R_to_compare is None:
            R2 = 0
        else:
            R2 = R_to_compare[ind_to_keep]
        return np.var(R1-R2, ddof=1)

    def extract_peak_info(self):
        # Can be called after the parameters 'self.lbd' and 'self.R' have been defined.
        # Returns peak (max) reflectance, its width, wavelength at maximum reflectance
        # and average position (wavelength) of the peak.
        Rp = self.R[0]
        lbd_p = self.lbd[0]
        iC = 0
        for i in range(1, len(self.R)):
            if self.R[i] > Rp:
                Rp = self.R[i]
                lbd_p = self.lbd[i]
                iC = i
        iL = iR = iC
        while self.R[iL] > Rp/2. and iL > 0:
            iL -= 1
        while self.R[iR] > Rp/2. and iR < len(self.R)-1:
            iR += 1
        fwhm = self.lbd[iR] - self.lbd[iL]
        lbd_c = self.lbd[(iL+iR)//2]
        return Rp*100., fwhm*1e9, lbd_p*1e9, lbd_c*1e9

    def make_relative(self, bkgnd, ref_spectrum=None):
        # Arg: 2 other 'PSiReflectanceSimulator' objects.
        # Given the reflectance intensity, background and reference (incident beam intensity),
        # compute the reflectance and store it in 'self.R'.
        if ref_spectrum is None:
            self.R = self.R - bkgnd
        else:
            self.R = (self.R - bkgnd.R) / (ref_spectrum.R - bkgnd.R)


# 2) for defining a PSi multilayered stack
class PSiOpticalStack:
    def __init__(self, n_method, p_J_r_data_location="data-Clementine"):
        self.n_method = n_method
        self.append_index = None
        self.cells_mult = self.cells_t = self.cells_p = None
        self.layers_t = self.layers_p = self.layers_c = None
        load_values(p_J_r_data_location)
        self.reset_stack()

    def save_stack(self, fname=None):
        # Save layers data (thickness, porosity, coherence) in text file.
        if fname is None:
            fname = "stack.txt"
        #d_list = [np.array(self.layers_t, dtype=object),
        #          np.array(self.layers_p, dtype=object),
        #          np.array(self.layers_c, dtype=object)]
        data = np.array([self.layers_t, self.layers_p, self.layers_c], dtype=object).T
        np.savetxt(pkg_loc.joinpath("temp/"+fname), data[1:-1],
                   header="thickness[m], porosity[-], coherence", fmt=["%e", "%e", "%s"])

    def load_stack(self, fname=None):
        # Copy the layers from data that were saved previously.
        # Does not take into account the repetitions (sub-stacks).
        if fname is None:
            fname = "stack.txt"
        data = np.loadtxt(pkg_loc.joinpath("temp/"+fname), skiprows=1,
                          dtype={'names': ('t[m]', 'p', 'c'), 'formats': ('f4', 'f4', 'S1')})
        for line in data:
            if line[2] == b'c':
                self.add_thin_layer(line[0], line[1])
            elif line[2] == b'i':
                self.add_thick_layer(line[0], line[1])
            else:
                raise ValueError(f"Unknown coherence value in file '{fname}': {line[2]}")

    def optimal_Bragg_stack(self, lbd_target, min_R=0.95):
        # Returns values of thickness and porosity of the 2 types of layer
        # in a PSi Bragg stack, and the number of cells (number of repetition of these layers),
        # in order to match the required target wavelength and minimum peak reflectance.
        # An approximation is used to compute these values (the results might be inaccurate).
        n_compute = lambda p: self.n_method.get_complex_n(lbd_target, p).real
        p1, p2, N = get_opt_porosities(lbd_target, n_compute, comment_N=True, Rmin=min_R)
        n1 = n_compute(p1)
        n2 = n_compute(p2)
        t1, t2 = compute_t(n1, n2, lbd_target)
        return t1, t2, p1, p2, N

    def reset_stack(self):
        # Remove all layers from the stack
        # Optical parameters
        self.layers_t = []
        self.layers_p = []
        self.layers_c = []
        self.append_index = 1
        # top semi-infinite layer
        self.layers_t.append(np.inf)
        self.layers_p.append(1)
        self.layers_c.append('i')
        # low semi-infinite layer
        self.layers_t.append(np.inf)
        self.layers_p.append(1)
        self.layers_c.append('i')
        # Repetition
        self.cells_mult = []
        self.cells_t = []
        self.cells_p = []

    def add_thin_layer(self, thickness, porosity):
        # Add coherent PSi layer
        self.layers_t.insert(self.append_index, thickness)
        self.layers_p.insert(self.append_index, porosity)
        self.layers_c.insert(self.append_index, 'c')
        self.append_index += 1
        self.cells_t.append([thickness])
        self.cells_p.append([porosity])
        self.cells_mult.append(1)

    def add_thick_layer(self, thickness, porosity):
        # Add incoherent PSi layer
        self.layers_t.insert(self.append_index, thickness)
        self.layers_p.insert(self.append_index, porosity)
        self.layers_c.insert(self.append_index, 'i')
        self.append_index += 1
        self.cells_t.append([thickness])
        self.cells_p.append([porosity])
        self.cells_mult.append(1)

    def add_Bragg_stack(self, t1, t2, p1, p2, N, transition_layer=None):
        # Add a coherent stack of layers (N repetition) with thickness and porosity t1 and p1,
        # then t2 and p2.
        # The parameters 'transition_layer' can be used to define the thickness of
        # a transition layer at each interface, where the porosity is the average of the
        # porosity of the previous and next layer. It can be used to smoothen the transitions
        # between layers in the model.
        dt = 0.
        if transition_layer is not None:
            dt = transition_layer
        p_transition = (p1+p2)/2.
        for _ in range(N):
            self.layers_t.insert(self.append_index, t1-dt)
            self.layers_p.insert(self.append_index, p1)
            self.layers_c.insert(self.append_index, 'c')
            self.append_index += 1
            if transition_layer is not None:
                self.layers_t.insert(self.append_index, dt)
                self.layers_p.insert(self.append_index, p_transition)
                self.layers_c.insert(self.append_index, 'c')
                self.append_index += 1
            self.layers_t.insert(self.append_index, t2-dt)
            self.layers_p.insert(self.append_index, p2)
            self.layers_c.insert(self.append_index, 'c')
            self.append_index += 1
            if transition_layer is not None:
                self.layers_t.insert(self.append_index, dt)
                self.layers_p.insert(self.append_index, p_transition)
                self.layers_c.insert(self.append_index, 'c')
                self.append_index += 1
        if transition_layer is not None:
            self.layers_t[self.append_index-4*N] += dt/2.
            self.layers_t[self.append_index-2] += dt/2.
            self.append_index -= 1
            self.layers_t.pop(self.append_index)
            self.layers_p.pop(self.append_index)
            self.layers_c.pop(self.append_index)
        self.cells_t.append([t1, t2])
        self.cells_p.append([p1, p2])
        self.cells_mult.append(N)

    def get_t_n_arrays(self, wavelength):
        # Internal method.
        # Get a list of thickness and refractive index for all layers that must be modelled.
        layers_n = []
        for p in self.layers_p:
            # TODO: improvement possible (repetitions)
            n = self.n_method.get_complex_n(wavelength, p)
            layers_n.append(n)
        return self.layers_t, layers_n

    def get_c_array(self):
        # Coherence ('c' for coherent and 'i' for incoherent) for the layers defined previously.
        return self.layers_c

    def produce_I_dt_data(self, sample_area=0.27):
        # Show the values of current and time used to produce the multilayered stack
        # by electroetching, using the values inside the folder 'p_J_r_data_location',
        # defined when and object is instantiated.
        print("\n --- \nCurrent [mA] - time [s] for all layers:")
        for i_m, m in enumerate(self.cells_mult):
            print(f"{m} time(s):")
            for i_l in range(len(self.cells_t[i_m])):
                I, dt = get_I_dt(self.cells_p[i_m][i_l], self.cells_t[i_m][i_l], sample_area)
                print(f"\t# {self.cells_t[i_m][i_l]*1e9:.3f}nm layer with p={self.cells_p[i_m][i_l]:.3f} (J={I/sample_area/10.:.3f}mA/cm^2)")
                print(f"\tI={I:.3f}mA - dt={dt:.3f}s")

        print(f"1 time:")
        print(f"\tlift-off (undefined current/time)")
        print(" --- ")


# 3) for defining the model, used to compute the refractive index
#    of porous silicon, for a given porosity. Each class must implement
#    the method 'get_complex_n', to be used with 'PSiOpticalStack'.
#  - 'PSiRefractiveIndexMethod': uses a mixing rule.
#  - 'CsttRefractiveIndexMethod': the refractive index of each layer is constant
#        (does not depend on the porosity or wavelength of the incident wave).
class PSiRefractiveIndexMethod:
    def __init__(self, method="Looyenga", rox=0., medium="air"):
        # material present in the pores: "air" (str) or refractive index (float or complex)
        self.medium = medium
        # relative volume of Si (between 0 and 1) that has reacted with oxygen to produce SiO2
        self.PSi_ox_rel_vol = rox
        # mixing rule: "Looyenga" (LLL method), "Bruggeman" or "CRIM" (linear combination)
        self.mixing_method = method

    def set_ox_state(self, rox=0.):
        self.PSi_ox_rel_vol = rox

    def get_complex_n(self, wavelength, porosity):
        return get_n_eff(porosity, wavelength, self.medium,
                         self.PSi_ox_rel_vol, self.mixing_method)

class CsttRefractiveIndexMethod(PSiRefractiveIndexMethod):
    def initialize(self, n):
        # Constant refractive index for each layer.
        self.n = n

    def get_complex_n(self, wavelength, porosity):
        return self.n



if __name__ == '__main__':
    # Testing the code
    m = PSiRefractiveIndexMethod(rox=0.)
    stack = PSiOpticalStack(m)

    # define a multilayered stack
    t1, t2, p1, p2, N = stack.optimal_Bragg_stack(633e-9, min_R=0.98)
    N = 10
    print(f"Impose N={N}")
    stack.add_Bragg_stack(t1, t2, p1, p2, N)
    t_supp = 80e-6  # 80 um support
    p_supp = 0.5
    stack.add_thick_layer(t_supp, p_supp)
    stack.save_stack()
    stack.produce_I_dt_data()
    # replace with this instruction after the stack is saved
    #stack.load_stack()

    # numerical simulation (no oxidation)
    simu = PSiReflectanceSimulator(stack)
    simu.compute_refl_spectrum()
    FOM = simu.extract_peak_info()
    # print info for the reflectance spectrum
    print("Results with no oxidation:")
    print(f"\t{simu.peak_info_str} = {FOM}\n")

    # simulation and plots with different level of oxidation
    lbd = None
    R_all = []
    lab_all = []
    print("Results with oxidation:")
    for ox in [0., 5e-2, 10e-2, 15e-2, 20e-2]:
        m.set_ox_state(ox)
        simu.compute_refl_spectrum()
        if lbd is None:
            lbd = simu.lbd
        R_all.append(simu.R)
        lab_all.append(f"r_ox={int(ox*100):d}%")
        print(f"\t{simu.peak_info_str} = {simu.extract_peak_info()}")
    # result saved in 'results/' folder
    plot_those_R(lbd, R_all, lab_all, "simu_ox_shift.png", intensity=False)





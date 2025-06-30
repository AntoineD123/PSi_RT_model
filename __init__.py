# Main class of the package
import numpy as np
from n_index import get_n_eff
# Using https://doi.org/10.5281/zenodo.1344878
from reflectance import allocate_matrices, tmm_get_R_T, save_data, load_prev_data, plot_show_R_T, plot_compare_R
from set_layers import compute_t, get_opt_porosities
from etch_recipe import load_values, get_I_dt

class PSiReflectanceSimulator:
    lbd_min_nm = 400
    lbd_max_nm = 850
    def __init__(self, stack):
        self.stack = stack
        self.lbd = self.R = self.T = None

    def compute_refl_spectum(self, wavelengths=None):
        self.lbd = wavelengths
        if wavelengths is None:
            self.lbd = np.linspace(self.lbd_min_nm, self.lbd_max_nm, 100)*1e-9
        n_wavelength = len(self.lbd)
        self.R = np.zeros(n_wavelength)
        self.T = np.zeros(n_wavelength)
        c_arr = stack.get_c_array()
        allocate_matrices(c_arr)
        for i_lbd, lbd in enumerate(self.lbd):
            t_arr, n_arr = self.stack.get_t_n_arrays(lbd)
            self.R[i_lbd], self.T[i_lbd] = tmm_get_R_T(lbd, t_arr, n_arr, c_arr)
        return self.R, self.T
        # TODO: adapted fct
        # TODO: don't use tmm ; improve perf
        # return tmm_compute(self.stack, lbd_arr)

    def save_spectrum(self):
        save_data(self.lbd, self.R, self.T)

    def load_spectrum(self):
        self.lbd, self.R, self.T = load_prev_data()

    def load_from_ocean_view(self, fname):
        lbd_nm, R_large = np.loadtxt(fname, skiprows=14, unpack=True)
        c1 = self.lbd_min_nm < lbd_nm
        c2 = lbd_nm < self.lbd_max_nm
        ind_to_keep = np.where(np.logical_and(c1, c2))
        self.lbd = lbd_nm[ind_to_keep]*1e-9
        self.R = R_large[ind_to_keep]/100.
        self.T = np.zeros(len(self.R))

    def show_R_T(self, image_name=None):
        lbd_nm = self.lbd * 1e9
        plot_show_R_T(lbd_nm, self.R, self.T, image_name)

    def compare_R(self, loaded_simulator, im_name=None):
        lbd_nm = self.lbd * 1e9
        plot_compare_R(lbd_nm, self.R, loaded_simulator.R, im_name)

    def extract_Rp_FWHM(self):
        Rp = self.R[0]
        lbd_p = self.lbd[0]
        iC = 0
        for i in range(1, len(self.R)):
            if self.R[i] > Rp:
                Rp = self.R[i]
                lbd_p = self.lbd[i]
                iC = i
        iL = iR = iC
        while self.R[iL] > Rp/2.:
            iL -= 1
        while self.R[iR] > Rp/2.:
            iR += 1
        fwhm = self.lbd[iR] - self.lbd[iL]
        return Rp, fwhm


class PSiOpticalStack:
    def __init__(self, n_method, p_J_r_data_location="data-Clementine"):
        # TODO: store/load stack to/from file in 'temp'
        self.n_method = n_method
        self.append_index = None
        self.cells_mult = self.cells_t = self.cells_p = None
        self.layers_t = self.layers_p = self.layers_c = None
        load_values(p_J_r_data_location)
        self.reset_stack()

    def optimal_Bragg_stack(self, lbd_target, min_R=0.95, p_bounds=[0.4, 0.7]):
        # TODO: use bounds and min_R (not default val)
        n_compute = lambda p: self.n_method.get_complex_n(lbd_target, p).real
        p1, p2, N = get_opt_porosities(lbd_target, n_compute, comment_N=True)
        n1 = n_compute(p1)
        n2 = n_compute(p2)
        t1, t2 = compute_t(n1, n2, lbd_target)
        return t1, t2, p1, p2, N


    def reset_stack(self):
        # Optical parameters
        self.layers_t = []
        self.layers_p = []
        self.layers_c = []
        self.append_index = 1
        # top semi-infinite layer
        self.layers_t.append(np.inf)
        self.layers_p.append(0)
        self.layers_c.append('i')
        # low semi-infinite layer
        self.layers_t.append(np.inf)
        self.layers_p.append(0)
        self.layers_c.append('i')
        # Repetitivity
        self.cells_mult = []
        self.cells_t = []
        self.cells_p = []

    def add_thin_layer(self, thickness, porosity):
        self.layers_t.insert(self.append_index, thickness)
        self.layers_p.insert(self.append_index, porosity)
        self.layers_c.insert(self.append_index, 'c')
        self.append_index += 1
        self.cells_t.append([thickness])
        self.cells_p.append([porosity])
        self.cells_mult.append(1)

    def add_thick_layer(self, thickness, porosity):
        self.layers_t.insert(self.append_index, thickness)
        self.layers_p.insert(self.append_index, porosity)
        self.layers_c.insert(self.append_index, 'i')
        self.append_index += 1
        self.cells_t.append([thickness])
        self.cells_p.append([porosity])
        self.cells_mult.append(1)

    def add_Bragg_stack(self, t1, t2, p1, p2, N, rugosity=0, transition_layer=None):
        # TODO: rugosity, intermediate layers
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
        layers_n = []
        for p in self.layers_p:
            # TODO: improvement possible (repetitions)
            n = self.n_method.get_complex_n(wavelength, p)
            layers_n.append(n)
        return self.layers_t, layers_n

    def get_c_array(self):
        return self.layers_c

    def produce_I_dt_data(self, sample_area=0.27):
        # TODO: write in file
        print("\n --- \nCurrent [mA] - time [s] for all layers:")
        for i_m, m in enumerate(self.cells_mult):
            print(f"{m} time(s):")
            for i_l in range(len(self.cells_t[i_m])):
                I, dt = get_I_dt(self.cells_p[i_m][i_l], self.cells_t[i_m][i_l], sample_area)
                print(f"\t# {self.cells_t[i_m][i_l]*1e9:.3f}nm layer with p={self.cells_p[i_m][i_l]:.3f} (J={I/sample_area/10.:.3f}mA/cm^2)")
                print(f"\tI={I:.3f}mA - dt={dt:.3f}s")

        print(f"1 time (lift-off):")
        print(f"\tI=340mA - dt=1.2s")
        print(" --- ")


class PSiRefractiveIndexMethod:
    def __init__(self, medium="air"):
        self.medium = medium
        self.ox_rel_vol = 0.
        self.mixing_method = "Looyenga"

    def get_complex_n(self, wavelength, porosity):
        return get_n_eff(porosity, wavelength, self.medium,
                         self.ox_rel_vol, self.mixing_method)



if __name__ == '__main__':
    m = PSiRefractiveIndexMethod()
    stack = PSiOpticalStack(m)
    stack.reset_stack()
    t1, t2, p1, p2, N = stack.optimal_Bragg_stack(633e-9)
    N = 10
    print(f"Impose N={N}")
    stack.add_Bragg_stack(t1, t2, p1, p2, N, transition_layer=10.e-9)
    # stack.add_Bragg_stack(0, 0, 0, 0, 40)
    t_supp = 80e-6  # 80 um support
    p_supp = 0.7   # TODO: this is 0.5
    stack.add_thick_layer(t_supp, p_supp)

    s_test = PSiReflectanceSimulator(None)
    s_test.load_spectrum()
    #s_test.load_from_ocean_view("temp/ad5_Relative__0__20.txt")
    simu = PSiReflectanceSimulator(stack)
    simu.compute_refl_spectum(s_test.lbd)
    #simu.save_spectrum()
    simu.show_R_T("test_method_loc.png")
    #Rp, d_lbd = simu.extract_Rp_FWHM()
    simu.compare_R(s_test, "test_R_50-70p_Oox.png")
    #stack.produce_I_dt_data()



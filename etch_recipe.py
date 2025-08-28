import numpy as np
from n_index import approx_1st_order, p_min_max, apply_on_arr, pkg_loc

# Units
percent = 0.01
mA_per_cm_2 = 1e-3*1e4
nm_per_s = 1e-9

# Arrays containing the empirical data :
# Prosity [-] and corresponding current density [A/m^2]
p1_arr, J1_arr = None, None
# Etch rate [m/s] and corresponding current density [A/m^2]
r2_arr, J2_arr = None, None

def load_values(files_loc="data-Clementine"):
    # Function to call before using any of the following functions.
    # The folder provided starts with 'data-' ('data-Romain', 'data-Clementine')
    # It contains 2 files:
    # - j-rho_data.txt: only contains values of current density and
    # corresponding porosity (and a single header line).
    # - j-etch_data.txt: only contains values of current density and
    # corresponding etch rate (and a single header line).
    # All values are presented in ascending order in the files,
    # always using SI units
    global p1_arr, J1_arr, r2_arr, J2_arr
    p1_arr, J1_arr = np.loadtxt(pkg_loc.joinpath(files_loc + "/j-rho_data.txt"), skiprows=1, unpack=True)
    r2_arr, J2_arr = np.loadtxt(pkg_loc.joinpath(files_loc + "/j-etch_data.txt"), skiprows=1, unpack=True)

def get_J_with_p(porosity):
    # ! p (and J) most be sorted in ascending order
    return approx_1st_order(p1_arr, porosity, J1_arr)

def get_etch_rate(j_in):
    # ! J(, etch_rate) most be sorted in ascending order
    return approx_1st_order(J2_arr, j_in, r2_arr)

def get_I_dt(p, t, sample_surface=0.27):
    # Arguments:
    #   p: estimated porosity (before oxidation) [-]
    #   t: estimated thickness of the porous layer [m]
    #   sample_surface: where the current passes [cm^2]
    # Output:
    #   I: required current [mA]
    #   dt: time required to finish the process [s]
    j = get_J_with_p(p)    # A/m^2
    r = get_etch_rate(j)    # m/s
    return j/mA_per_cm_2*sample_surface, t/r


if __name__ == '__main__':
    # Testing the code
    load_values("data-Clementine")
    import matplotlib.pyplot as plt
    p_arr = np.linspace(p_min_max[0], p_min_max[1], 20)
    j_arr = apply_on_arr(get_J_with_p, p_arr, dtype=float)
    e_arr = apply_on_arr(get_etch_rate, j_arr, dtype=float)

    load_values('data-Romain')
    j_arr2 = apply_on_arr(get_J_with_p, p_arr, dtype=float)
    e_arr2 = apply_on_arr(get_etch_rate, j_arr2, dtype=float)
    j_mult_R = 1.  # 3. to match the 2 curves
    e_mult_R = 1.  # 2. to match the 2 curves
    plt.figure()
    plt.title("Porosity wrt current density")
    plt.plot(j_arr, p_arr, label="C-data")
    plt.plot(j_arr2*j_mult_R, p_arr, label="R-data")
    plt.legend()
    plt.figure()
    plt.title("Etch rate wrt current density")
    plt.plot(j_arr, e_arr, label="C-data")
    plt.plot(j_arr2*j_mult_R, e_arr2*e_mult_R, label="R-data")
    plt.legend()
    plt.show()






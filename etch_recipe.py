import numpy as np
from n_index import approx_1st_order, rho_lim, apply_on_arr

# Units
percent = 0.01
mA_per_cm_2 = 1e-3*1e4
nm_per_s = 1e-9

# Arrays containing the empirical data :
# Prosity [-] and corresponding current density [A/m^2]
p1_arr, J1_arr = None, None
# Etch rate [m/s] and corresponding current density [A/m^2]
r2_arr, J2_arr = None, None

def load_values(files_loc):
    # The folder provided starts with 'data-' ('data-Romain', 'data-Clementine')
    # It contains 2 files:
    # - j-rho_data.txt: only contains values of current density and
    # corresponding porosity (and a single header line).
    # - j-etch_data.txt: only contains values of current density and
    # corresponding etch rate (and a single header line).
    # All values are presented in ascending order in the files,
    # always using SI units
    global p1_arr, J1_arr, r2_arr, J2_arr
    p1_arr, J1_arr = np.loadtxt(files_loc + "/j-rho_data.txt", skiprows=1, unpack=True)
    r2_arr, J2_arr = np.loadtxt(files_loc + "/j-etch_data.txt", skiprows=1, unpack=True)

def get_J_with_rho(porosity):
    # ! rho(, J) most be sorted in ascending order
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
    j = get_J_with_rho(p)    # A/m^2
    r = get_etch_rate(j)    # m/s
    return j/mA_per_cm_2*sample_surface, t/r


if __name__ == '__main__':
    load_values("data-Clementine")
    import matplotlib.pyplot as plt
    p_arr = np.linspace(rho_lim[0], rho_lim[1], 20)
    j_arr = apply_on_arr(get_J_with_rho, p_arr, dtype=float)
    e_arr = apply_on_arr(get_etch_rate, j_arr, dtype=float)
    """
    plt.figure()
    plt.title("Porosity wrt current density")
    plt.plot(j_arr, p_arr)
    plt.figure()
    plt.title("Etch rate wrt current density")
    plt.plot(j_arr, e_arr)
    plt.show()
    """
    load_values('data-Romain')
    j_arr2 = apply_on_arr(get_J_with_rho, p_arr, dtype=float)
    e_arr2 = apply_on_arr(get_etch_rate, j_arr2, dtype=float)
    j_mult_R = 3.  # 2.6, 2.84 at high porosity
    e_mult_R = 2.  # 1.64, 1.725 at high porosity
    plt.figure()
    plt.title("Porosity wrt current density")
    plt.plot(j_arr, p_arr, label="C-data")
    plt.plot(j_arr2*j_mult_R, p_arr, label="R-data(j*3)")
    plt.legend()
    plt.figure()
    plt.title("Etch rate wrt current density")
    plt.plot(j_arr, e_arr, label="C-data")
    plt.plot(j_arr2*j_mult_R, e_arr2*e_mult_R, label="R-data(j*3, r*2)")
    plt.legend()
    plt.show()
    # Recette Romain (lift-off):
    # 2 s à 120mA (32.4 mA/cm2)
    # Recette pour Clémentine (lift-off):
    # 1.15s à 340mA (91.8 mA/cm2) au lieu de 216 mA






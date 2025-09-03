import numpy as np
from scipy.optimize import fsolve
from pathlib import Path

# Units
um = 1e-6
nm = 1e-9
pkg_loc = Path(__file__).parent

# Import data once and use several times
l_Si_um,   n_Si,   k_Si   = np.loadtxt(pkg_loc.joinpath("data-literature/n_k_Si.txt"),   skiprows=1, unpack=True)
l_SiO2_um, n_SiO2, k_SiO2 = np.loadtxt(pkg_loc.joinpath("data-literature/n_k_SiO2.txt"), skiprows=1, unpack=True)
l_Al_um,   n_Al,   k_Al   = np.loadtxt(pkg_loc.joinpath("data-literature/n_k_Al.txt"),   skiprows=1, unpack=True)
l_Si   = l_Si_um*um
l_SiO2 = l_SiO2_um*um
l_Al   = l_Al_um*um

# Values used as boundaries for computations and optimizations
p_min_max = [0.5, 0.7]  # min/max porosities

def approx_1st_order(x_arr, x_test, y_arr, z_arr=None):
    # @PRE:
    #       x_arr, y_arr: arrays corresponding to monotonic
    #           function f1. 'x_arr' must be ordered.
    #       x_test: test value (associated to output).
    # @POST:
    #       Returns f1(x_test), calculated using a first
    #           order approximation on i_th interval of
    #           the discrete function f1.
    #       Also returns f2(x_test): z_arr[i] = f2(x_arr[i])
    #           if 'z_arr' is defined.
    assert x_arr[0] < x_test < x_arr[-1]
    i = 0
    while x_arr[i+1] < x_test:
        i += 1
    if z_arr is None:
        slope = (y_arr[i+1]-y_arr[i])/(x_arr[i+1]-x_arr[i])
        return y_arr[i] + slope * (x_test-x_arr[i])
    else:
        slope1 = (y_arr[i+1]-y_arr[i])/(x_arr[i+1]-x_arr[i])
        slope2 = (z_arr[i+1]-z_arr[i])/(x_arr[i+1]-x_arr[i])
        f1_x = y_arr[i] + slope1 * (x_test-x_arr[i])
        f2_x = z_arr[i] + slope2 * (x_test-x_arr[i])
        return f1_x, f2_x

def get_n_Si(lbd):
    # https://www.pveducation.org/pvcdrom/materials/optical-properties-of-silicon
    # return 4
    # https://refractiveindex.info/?shelf=main&book=Si&page=Schinke
    # using
    # 1) C. Schinke, P. C. Peest, J. Schmidt, R. Brendel, K. Bothe, M. R. Vogt, I. Kröger,
    #    S. Winter, A. Schirmacher, S. Lim, H. T. Nguyen, D. MacDonald. Uncertainty analysis for
    #    the coefficient of band-to-band absorption of crystalline silicon. AIP Advances 5, 67168 (2015)
    # 2) M. R. Vogt. Development of physical models for the simulation of optical properties
    #    of solar cell modules. PhD. Thesis (2015)
    if lbd < l_Si[0] or lbd > l_Si[-1]:
        raise ValueError(f"Can't find {lbd/nm}nm in range ({l_Si[0]/nm}, {l_Si[-1]/nm}) for the wavelength")
    n, k = approx_1st_order(l_Si, lbd, n_Si, k_Si)
    return n + k*1j

def get_n_SiO2(lbd):
    # ~ 1.5
    # https://refractiveindex.info/?shelf=main&book=SiO2&page=Gao
    # using
    # 1) L. Gao, F. Lemarchand, M. Lequime. Refractive index determination of SiO2 layer
    # in the UV/Vis/NIR range: spectrophotometric reverse engineering on single
    # and bi-layer designs. J. Europ. Opt. Soc. Rap. Public. 8, 13010 (2013)
    # (Numerical data kindly provided by F. Lemarchand)
    if lbd < l_SiO2[0] or lbd > l_SiO2[-1]:
        raise ValueError(f"Can't find {lbd/nm}nm in range ({l_SiO2[0]/nm}, {l_SiO2[-1]/nm}) for the wavelength")
    n, k = approx_1st_order(l_SiO2, lbd, n_SiO2, k_SiO2)
    return n + k*1j

def get_n_Al(lbd):
    # https://refractiveindex.info/?shelf=main&book=Al&page=Cheng
    # (epitaxially grown Al film at room temperature)
    # using
    # 1) F. Cheng, P.-H. Su, J. Choi, S. Gwo, X. Li, C.-K. Shih. Epitaxial growth of
    # atomically smooth aluminum on silicon and its intrinsic optical properties.
    # ACS Nano 10, 9852-9860 (2016)
    if lbd < l_Al[0] or lbd > l_Al[-1]:
        raise ValueError(f"Can't find {lbd/nm}nm in range ({l_Al[0]/nm}, {l_Al[-1]/nm}) for the wavelength")
    n, k = approx_1st_order(l_Al, lbd, n_Al, k_Al)
    return n + k*1j

def get_n_solv(lbd, solv_type):
    # Returns a constant refractive index 'solv_type' (real float/complex)
    # or the refractive index of the material 'solv_type' at
    # wavelength 'lbd'.
    # 'solv_type' is a float, complex, or one of the following str:
    # "air" or "Al" (aluminum)
    if type(solv_type) is not str:
        return solv_type
    elif solv_type == "air":
        # https://refractiveindex.info/?shelf=main&book=N2&page=Peck-0C
        return 1.003
    elif solv_type == "Al":
        return get_n_Al(lbd)
    else:
        raise ValueError(f"Refractive index not computer for {solv_type}")

def f(arg_in, n1, n2, n3, r_ox, vol_ox_rel, vol_inc, P):
    n = complex(arg_in[0], arg_in[1])
    err = (1-P-r_ox) * (n1-n)/(n1+2*n) + \
        vol_ox_rel*r_ox * (n2-n)/(n2+2*n) + \
        (P-vol_inc*r_ox) * (n3-n)/(n3+2*n)
    return err.real, err.imag
n_brugg_init = np.array([1.5, 0.])

# Low porosities - Bruggeman model
# High porosities - Looyenga model
# Function used to compute a refractive index of PSi, in a general case
def get_n_eff(P, lbd, solv_type, ox_state=0., mixing_method="Looyenga"):
    # Compute effective n
    # P: porosity in [0 (full); 1 (hollow)] (before oxidation)
    # lbd: wavelength [m]
    # solv: solvent in pores ("air")
    # ox_state: oxidation degree in [0 (pure silicon); 1 (fully oxidized)]
    # mixing_method: rule used to compute n, with the values of n
    # of the different constituents: "Looyenga" (LLL method), "Bruggeman", "CRIM"
    # (linear combination)
    if P == 0:
        return get_n_Si(lbd)
    if P == 1:
        return get_n_solv(lbd, solv_type)
    if isinstance(lbd, np.ndarray):
        sol = np.zeros(lbd.shape, dtype=complex)
        for i, l in enumerate(lbd):
            sol[i] = get_n_eff(P, l, solv_type)
        return sol
    r_ox = ox_state*(1-P)  # volume fraction of the flake that was oxidized -> [SiO2]/2.27
    r_ox = min(r_ox, P/1.27)
    vol_ox_rel = 2.27  # volume of SiO2 wrt volume of Si (same molar content)
    vol_inc = vol_ox_rel-1  # 1.27
    n1 = get_n_Si(lbd)
    n2 = get_n_SiO2(lbd)
    n3 = get_n_solv(lbd, solv_type)
    if mixing_method == "Bruggeman":
        res = fsolve(f, x0=n_brugg_init, diag=[1., 1e3],   # tolerance: xtol=1.49012e-8
                                    args=(n1, n2, n3, r_ox, vol_ox_rel, vol_inc, P))
        return complex(res[0], res[1])
    elif mixing_method == "CRIM":
        # Fastest
        # Complex refractive index method
        return (1-P-r_ox) * n1 + vol_ox_rel*r_ox * n2 + (P-vol_inc*r_ox) * n3
    elif mixing_method == "Looyenga":
        exp = 4./3.
        inv_exp = 3./4.
        n_exp_4_3 = (1-P-r_ox) * n1**exp + vol_ox_rel*r_ox * n2**exp + \
                    (P-vol_inc*r_ox) * n3**exp
        return n_exp_4_3**inv_exp
    else:
        raise ValueError(f"Unexpected value for variable 'mixing_method': {mixing_method}")


def apply_on_arr(f, x_arr, **kwargs):
    # Return an array, with the result of the function f
    # applied (1st arg of f) on each items in the list x_arr.
    dtype = complex
    if "dtype" in kwargs.keys():
        dtype = kwargs.pop("dtype")
    res = np.zeros(len(x_arr), dtype=dtype)
    for i, x_i in enumerate(x_arr):
        res[i] = f(x_i, **kwargs)
    return res



if __name__ == '__main__':
    # Testing the code
    import matplotlib.pyplot as plt
    lbd = np.linspace(450, 800, 100)*1e-9
    target_lbd = 633e-9
    target_lbd_nm = 633

    solv = "air"
    nair = apply_on_arr(get_n_solv, lbd, solv_type=solv)
    nSi = apply_on_arr(get_n_Si, lbd)
    nSiO2 = apply_on_arr(get_n_SiO2, lbd)

    p1_arr = np.linspace(p_min_max[0], p_min_max[1], 50)
    m = "CRIM"
    nPSiOx_CRIM = apply_on_arr(get_n_eff, p1_arr, lbd=target_lbd, solv_type=solv, mixing_method=m)
    m = "Looyenga"
    nPSiOx_Looyenga = apply_on_arr(get_n_eff, p1_arr, lbd=target_lbd, solv_type=solv, mixing_method=m)
    m = "Bruggeman"
    nPSiOx_Bruggeman = apply_on_arr(get_n_eff, p1_arr, lbd=target_lbd, solv_type=solv, mixing_method=m)

    # Evolution of complex n with the different models, for different materials (p=0.)
    plt.figure()
    plt.ylabel("n [-]")
    plt.xlabel("lbd [nm]")
    plt.plot(lbd*1e9, nair.real, label="air")
    plt.plot(lbd*1e9, nSi.real, label="$Si$")
    plt.plot(lbd*1e9, nSiO2.real, label="$Si0_2$")
    plt.legend()
    plt.savefig(f"results/n_pure_l={target_lbd_nm}nm.jpg")

    plt.figure()
    plt.ylabel("$\\kappa$ [-]")
    plt.xlabel("lbd [nm]")
    plt.plot(lbd*1e9, nair.imag, label="air")
    plt.plot(lbd*1e9, nSi.imag, label="$Si$")
    plt.plot(lbd*1e9, nSiO2.imag, label="$Si0_2$")
    plt.legend()
    plt.savefig(f"results/k_pure_l={target_lbd_nm}nm.jpg")

    # Values of n (complex) at 'target_lbd', for different porosities
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].set_ylabel("n [-]")
    ax[0].plot(p1_arr, nPSiOx_CRIM.real, label="CRI")
    ax[0].plot(p1_arr, nPSiOx_Looyenga.real, label="LLL")
    ax[0].plot(p1_arr, nPSiOx_Bruggeman.real, label="Bruggeman")
    ax[0].legend()

    ax[1].set_ylabel("$\\kappa$ [-]")
    ax[1].set_xlabel("porosity [-]")
    ax[1].plot(p1_arr, nPSiOx_CRIM.imag, label="CRI")
    ax[1].plot(p1_arr, nPSiOx_Looyenga.imag, label="LLL")
    ax[1].plot(p1_arr, nPSiOx_Bruggeman.imag, label="Bruggeman")
    ax[1].legend()
    plt.savefig(f"results/n_PSi_l={target_lbd_nm}nm.jpg")

    # If Bruggeman is more precise, compute the relative error of the 2 other methods
    fig, ax = plt.subplots(2, 1, sharex=True)
    errC1 = np.abs(nPSiOx_CRIM.real-nPSiOx_Bruggeman.real)/nPSiOx_Bruggeman.real*100
    errC2 = np.abs(nPSiOx_CRIM.imag-nPSiOx_Bruggeman.imag)/nPSiOx_Bruggeman.imag*100
    errL1 = np.abs(nPSiOx_Looyenga.real-nPSiOx_Bruggeman.real)/nPSiOx_Bruggeman.real*100
    errL2 = np.abs(nPSiOx_Looyenga.imag-nPSiOx_Bruggeman.imag)/nPSiOx_Bruggeman.imag*100
    ax[0].set_ylabel("Relative error on n [%]")
    ax[0].plot(p1_arr, errC1, label="CRIM")
    ax[0].plot(p1_arr, errL1, label="Looyenga")
    ax[0].legend()

    ax[1].set_ylabel("Relative error on $\\kappa$ [%]")
    ax[1].set_xlabel("porosity [-]")
    ax[1].plot(p1_arr, errC2, label="CRIM")
    ax[1].plot(p1_arr, errL2, label="Looyenga")
    ax[1].legend()

    data = [p1_arr, nPSiOx_Bruggeman, nPSiOx_Looyenga, nPSiOx_CRIM]
    plt.savefig(f"results/n_err_PSi_l={target_lbd_nm}nm.jpg")
    plt.show()



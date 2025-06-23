import numpy as np
import sympy
import sympy.abc

um = 1e-6
nm = 1e-9
l_Si_um,   n_Si,   k_Si   = np.loadtxt("n_k_Si.txt", skiprows=1, unpack=True)
l_SiO2_um, n_SiO2, k_SiO2 = np.loadtxt("n_k_SiO2.txt", skiprows=1, unpack=True)
l_Si = l_Si_um*um
l_SiO2 = l_SiO2_um*um

rho_lim = [0.5, 0.7]  # min/max porosities
target_lbd = 633e-9   # m

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

def get_n_solv(lbd, solv_type):
    if solv_type == "air":
        # https://refractiveindex.info/?shelf=main&book=N2&page=Peck-0C
        return 1.003
    else:
        raise ValueError(f"Refractive index not computer for {solv_type}")

# Low porosities - Bruggeman model:
# P*(get_n_solv-n_eff**2)/(gmixing_methodet_n_solv-2*n_eff**2) + (1-P)*(n_SiOx-n_eff**2)/(n_SiOx+2*n_eff**2) = 0
# High porosities - Looyenga model:
def get_n_eff(P, lbd, solv_type, ox_state=0.1, mixing_method="Looyenga"):
    # Compute effective n
    # P: porosity in [0 (full), 1 (hollow)] (before oxidation)
    # lbd: wavelength [m]
    # solv: solvent in pores ("air")
    if P == 0:
        return get_n_Si(lbd)
    if P == 1:
        return get_n_solv(lbd, solv_type)
    if isinstance(lbd, np.ndarray):
        sol = np.zeros(lbd.shape, dtype=complex)
        for i, l in enumerate(lbd):
            sol[i] = get_n_eff(P, l, solv_type)
        return sol
    r_ox = 0.1  # volume fraction of Si that was oxidized -> (1-P)*[SiO2]/([SiO2]+[Si])
    vol_ox_rel = 2.27  # volume of SiO2 wrt volume of Si (same molar content)
    vol_inc = vol_ox_rel-1  # 1.27
    n1 = get_n_Si(lbd)
    n2 = get_n_SiO2(lbd)
    n3 = get_n_solv(lbd, solv_type)
    if mixing_method == "Bruggeman":
        n = sympy.abc.x
        # using relative molar volume of SiO2 wrt Si ??
        f = (1-P-r_ox) * (n1-n)/(n1+2*n) + \
            vol_ox_rel*r_ox * (n2-n)/(n2+2*n) + \
            (P-vol_inc*r_ox) * (n3-n)/(n3+2*n)
        res = sympy.solve(f, n)
        # [(-1.39873889115937 - 0.0395169586948642*I,), (-0.61912151730361 - 0.00085134248036138*I,),
        # (2.40448222723785 + 0.0725646093174908*I,)]
        print(res)
        return complex(res[-1])
    elif mixing_method == "CRIM":
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
    dtype = complex
    if "dtype" in kwargs.keys():
        dtype = kwargs.pop("dtype")
    res = np.zeros(len(x_arr), dtype=dtype)
    for i, x_i in enumerate(x_arr):
        res[i] = f(x_i, **kwargs)
    return res

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    lbd = np.linspace(400, 850, 100)*1e-9

    solv = "air"
    nair = apply_on_arr(get_n_solv, lbd, solv_type=solv)
    nSi = apply_on_arr(get_n_Si, lbd)
    nSiO2 = apply_on_arr(get_n_SiO2, lbd)

    p1_arr = np.linspace(rho_lim[0], rho_lim[1], 50)
    m = "CRIM"
    nPSiOx_CRIM = apply_on_arr(get_n_eff, p1_arr, lbd=target_lbd, solv_type=solv, mixing_method=m)
    m = "Looyenga"
    nPSiOx_Looyenga = apply_on_arr(get_n_eff, p1_arr, lbd=target_lbd, solv_type=solv, mixing_method=m)
    m = "Bruggeman"
    nPSiOx_Bruggeman = apply_on_arr(get_n_eff, p1_arr, lbd=target_lbd, solv_type=solv, mixing_method=m)
    
    plt.figure()
    plt.ylabel("n [-]")
    plt.xlabel("lbd [nm]")
    plt.plot(lbd*1e9, nair.real, label="air")
    plt.plot(lbd*1e9, nSi.real, label="$Si$")
    plt.plot(lbd*1e9, nSiO2.real, label="$Si0_2$")
    plt.legend()

    plt.figure()
    plt.ylabel("$\\alpha$ [1/m]")
    plt.xlabel("lbd [nm]")
    plt.plot(lbd*1e9, nair.imag, label="air")
    plt.plot(lbd*1e9, nSi.imag, label="$Si$")
    plt.plot(lbd*1e9, nSiO2.imag, label="$Si0_2$")
    plt.legend()

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].set_ylabel("n [-]")
    ax[0].plot(p1_arr, nPSiOx_CRIM.real, label="CRIM")
    ax[0].plot(p1_arr, nPSiOx_Looyenga.real, label="Looyenga")
    ax[0].plot(p1_arr, nPSiOx_Bruggeman.real, label="Bruggeman")
    ax[0].legend()

    ax[1].set_ylabel("$\\alpha$ [1/m]")
    ax[1].set_xlabel("porosity [-]")
    ax[1].plot(p1_arr, nPSiOx_CRIM.imag, label="CRIM")
    ax[1].plot(p1_arr, nPSiOx_Looyenga.imag, label="Looyenga")
    ax[1].plot(p1_arr, nPSiOx_Bruggeman.imag, label="Bruggeman")
    ax[1].legend()
    plt.savefig(f"results/n_PSiOx_l={target_lbd}nm_rOx={5}%.jpg")

    fig, ax = plt.subplots(2, 1, sharex=True)
    errC1 = np.abs(nPSiOx_CRIM.real-nPSiOx_Bruggeman.real)/nPSiOx_Bruggeman.real*100
    errC2 = np.abs(nPSiOx_CRIM.imag-nPSiOx_Bruggeman.imag)/nPSiOx_Bruggeman.imag*100
    errL1 = np.abs(nPSiOx_Looyenga.real-nPSiOx_Bruggeman.real)/nPSiOx_Bruggeman.real*100
    errL2 = np.abs(nPSiOx_Looyenga.imag-nPSiOx_Bruggeman.imag)/nPSiOx_Bruggeman.imag*100
    ax[0].set_ylabel("Relative error on n [%]")
    ax[0].plot(p1_arr, errC1, label="CRIM")
    ax[0].plot(p1_arr, errL1, label="Looyenga")

    ax[1].set_ylabel("Relative error on $\\alpha$ [%]")
    ax[1].set_xlabel("porosity [-]")
    ax[1].plot(p1_arr, errC2, label="CRIM")
    ax[1].plot(p1_arr, errL2, label="Looyenga")

    data = [p1_arr, nPSiOx_Bruggeman, nPSiOx_Looyenga, nPSiOx_CRIM]
    np.savetxt("results/n_2_methods.txt", np.transpose(data), header="p[-] n_Bruggeman[-] n_Looyenga[-] n_CRIM[-]")
    plt.savefig(f"results/n_err_PSiOx_rOx={5}%.jpg")
    plt.show()


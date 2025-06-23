import numpy as np
from scipy.optimize import minimize
from etch_recipe import load_values, get_I_dt, percent

# For a Bragg mirror, a stack of 2x'N' layers
# is created, with 2 different refractive indices.
# The optical response for the reflectivity is an
# oscillating signal with a reflection window
# around a 'target_lbd', that spans on a range 'target_win'.
from n_index import get_n_eff, target_lbd, rho_lim
target_win = 100e-9  # m -> not used here
N = 40  # number of cells (of 2 layers of different porosities)

def compute_t(n1, n2, lbd_center):
    # Computing the thickness for best
    # reflectivity around lbd.
    # n1, n2 are real refractive indies.
    t1 = lbd_center/(4*n1)
    t2 = lbd_center/(4*n2)
    return t1, t2

def compute_R_W(n1, n2, lbd_center, n_bilayers=1):
    # Computing performance (span, max reflectivity)
    dn = n1-n2
    na = (n1+n2)/2.
    Wwin = 2*dn/(np.pi*na) * lbd_center
    cstt_r = (n1/n2)**(2*n_bilayers)
    Rcenter = ((cstt_r-1)/(cstt_r+1))**2
    return Rcenter, Wwin

def rho_to_R_inv(arg_list, n_compute_method, lbd):
    # Assuming a single cell of 2 layers
    rho1, rho2 = arg_list
    n1 = n_compute_method(rho1)
    n2 = n_compute_method(rho2)
    # Computations for a single layer
    R, W = compute_R_W(n1, n2, lbd)
    return 1./R

def show_min_N(rho1, rho2, n_compute_method, lbd_center, show_comment):
    n1 = n_compute_method(rho1)
    n2 = n_compute_method(rho2)
    R = W = n_cells = 0
    while R < 0.95:
        n_cells += 1
        R, W = compute_R_W(n1, n2, lbd_center, n_bilayers=n_cells)
    if show_comment:
        print(f"\nMin reflectivity reached for N={n_cells} -> R={R*100:.3f} %, W={W:.3f} %")
    return n_cells

def get_opt_porosities(lbd, n_compute_method=None, show_fct=False, comment_N=False):
    # Return porosity of 2 layers, in bilayered
    # Bragg mirror, that maximizes the reflectivity
    # at wavelenght lbd
    if n_compute_method is None:
        n_compute_method = lambda p: get_n_eff(p, lbd, "air").real

    tol = 1e-6
    if show_fct:
        import matplotlib.pyplot as plt
        x = y = np.linspace(rho_lim[0], rho_lim[1], 10)
        z = np.array([1/rho_to_R_inv([i,j], n_compute_method, lbd) for j in y for i in x])
        X, Y = np.meshgrid(x, y)
        Z = z.reshape(10, 10)
        plt.contourf(X, Y, Z, 100)
        plt.colorbar()
        # Solutions in the top left corner -> max. R, W
        plt.show()
    start = np.array([0.5, 0.6])
    # Extracted from matlab code:
    # nA_on_nB=((1+((delta_lamda*pi)/(4*lamda)))/(1-((delta_lamda*pi)/(4*lamda))));
    res = minimize(rho_to_R_inv, start, method='L-BFGS-B', bounds=[rho_lim, rho_lim],
                   tol=tol, args=(n_compute_method, lbd),
                   options={'disp': True})
    p1, p2 = res.x  # [0.59, 0.75]
    N_min = show_min_N(p1, p2, n_compute_method, lbd, comment_N)
    return p1, p2, N_min

def make_stack(t1, p1, t2, p2, n_cells, l_base=None):
    if l_base is None:
        l_base = []
    layer1 = [t1, p1]
    layer2 = [t2, p2]
    for i in range(n_cells):
        l_base.append(layer1)
        l_base.append(layer2)
    return l_base

def make_stack_with_transition(t1, p1, t2, p2, n_cells, l_base=None, l_int=0.1e-9):
    if l_base is None:
        l_base = []
    layer1 = [t1-l_int/2., p1]
    layer2 = [l_int, (p1+p2)/2.]
    layer3 = [t2-l_int/2., p2]
    for i in range(n_cells):
        l_base.append(layer1)
        l_base.append(layer2)
        l_base.append(layer3)
    return l_base

def write_stack(stack, f_out="stack.txt"):
    np.savetxt(f_out, stack, header="thickness[m] porosity[-]",
               fmt=["%e", "%e"])  # complex val: "%e%+ej"


if __name__ == '__main__':
    # Each layer is made of oxidized PSi, with a porosity
    # P_A or P_B (P_A < P_B). We also impose the solvent.
    load_values("data-Clementine")
    P_A, P_B = get_opt_porosities(target_lbd, show_fct=True)
    solv = "air"

    nA_ = get_n_eff(P_A, target_lbd, solv)
    nB_ = get_n_eff(P_B, target_lbd, solv)
    nA = nA_.real
    nB = nB_.real
    tA, tB = compute_t(nA, nB, target_lbd)
    R_M, lbd_span = compute_R_W(nA, nB, target_lbd, N)
    print(f"\nFound thicknesses of {tA*1e9:.3f}nm (P={P_A}, n={nA:.3f}) and {tB*1e9:.3f}nm (P={P_B}, n={nB:.3f})")
    print(f"As a result: \nR={R_M/percent:.3f}% in a windows of {lbd_span*1e9:.3f}nm")

    surf = 0.27  # cm^2
    I_A, dt_A = get_I_dt(P_A, tA, surf)    # mA and sec
    I_B, dt_B = get_I_dt(P_B, tB, surf)
    print(f"Parameters for porosification process:")
    print(f"I1={I_A:.3f} mA - dt1={dt_A:.3f} s")
    print(f"I1={I_B:.3f} mA - dt1={dt_B:.3f} s")
    s = make_stack(tA, P_A, tB, P_B, N)
    write_stack(s)

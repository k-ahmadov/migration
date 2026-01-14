import numpy as np
import matplotlib.pyplot as plt
import sys
work_dir = '/home/kahmadov/phd/migration'
sys.path.append(f'{work_dir}/modules')
import aperture_solver

# parameter set
# flow rate per unit width (m^2/s)
q_0_values = [1e-3, 1e-4, 1e-5, 1e-6]
# fracture length (m)
L = 100
# fracture normal stiffness (Pa/m)
k_n = 50e9
# fluid viscosity (Pa.s)
mu = 1e-3
# for simplicity
a = k_n / (12*mu)
# initial aperture
w_i = 1e-4
# final time of simulation
t_fin_values = [25, 80, 100, 120]

if __name__=="__main__":
    # plt.plot(aperture_solver.n3_flux(Nx=Nx, Nt=Nt, ui_hat=w_i_hat, T_hat=t_fin_hat)[-1, :]*w_char)
    # plt.show()
    work_dir = "/home/kahmadov/phd/migration"
    # inputs for the numerical solver
    # number of cells
    Nx = int(1e3)
    # number of time steps
    Nt = int(5e2)
    # loop through every applied flow rate
    for t_fin, q_0 in zip(t_fin_values, q_0_values):
        # characteristic aperture
        w_char = (L*q_0/a)**(1/4)
        # characteristic time
        t_char = L**2/(a*w_char**3)
        # initial dimensionless (not-similarity) aperture
        w_i_hat = w_i/w_char
        # final dimensionless time
        t_fin_hat = t_fin/t_char
        # solution in space and time
        w_hat_xt = aperture_solver.n3_flux(Nx=Nx, Nt=Nt, ui_hat=w_i_hat, T_hat=t_fin_hat)
        # dimensionalize aperture
        w_xt = w_hat_xt*w_char
        # save aperture results
        np.save(f"{work_dir}/results/fvm/w-q-{q_0:.0e}.npy", w_xt)
        # convert to pressure
        p_xt = k_n*(w_xt-w_i)
        # save pressure results
        np.save(f"{work_dir}/results/fvm/p-q-{q_0:.0e}.npy", p_xt)
# endif

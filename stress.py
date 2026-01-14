import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
work_dir = '/home/kahmadov/phd/migration'
sys.path.append(f'{work_dir}/modules')
import aperture_solver
import elastic_solution

# %%

with open(f'{work_dir}/results/results_fvm.pkl', 'rb') as f:
    results_fvm = pickle.load(f)

with open(f'{work_dir}/results/parameters.pkl', 'rb') as f:
    parameters = pickle.load(f)

# %%

q_0 = 1e-3
t = 10
t_fin = 25
P_xt_fvm = results_fvm[q_0]['p']
x_fvm = np.linspace(0, 100, P_xt_fvm.shape[1])
Nt_fvm = P_xt_fvm.shape[0]
dt_fvm = t_fin / Nt_fvm
# Compute index at time t for FVM data
idx_t_fvm = int(t / dt_fvm) - 1

# %%

def symmetrize(x_data, y_data):
    x_sym = np.concatenate((-x_data[::-1][:-1], x_data))
    y_sym = np.concatenate((y_data[::-1][:-1], y_data))
    return x_sym, y_sym

def make_RHS(P_xt_data, t, t_fin):
    x_arr = np.linspace(0, 100, P_xt_data.shape[1])
    Nt = P_xt_data.shape[0]
    dt = t_fin / Nt
    # Compute index at time t for FVM data
    idx_t = int(t / dt) - 1
    P_x = P_xt_data[idx_t]

    def RHS(s):
        # symmetrize
        x_sym, P_sym = symmetrize(x_arr, P_x)
        # interpolate normalize by an order of magnitude
        return np.interp(s*parameters["L"], x_sym, P_sym)/1e6
    return RHS

# %%


RHS = make_RHS(P_xt_fvm, t, t_fin)
s = np.linspace(-1, 1, 100)
plt.plot(s, RHS(s), '-o')
plt.savefig(f'{work_dir}/tmp.png')
plt.clf()


# %%
# plt.plot(x_fvm, P_xt_fvm[idx_t_fvm])
# plt.show()
# plt.savefig(f'{work_dir}/tmp.png')

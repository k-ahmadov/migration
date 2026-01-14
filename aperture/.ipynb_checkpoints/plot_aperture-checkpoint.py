import numpy as np
import matplotlib.pyplot as plt

work_dir="/home/kahmadov/phd/migration"
w_3dec_data = np.loadtxt(f"{work_dir}/results/w-q-1e-3-processed.csv", delimiter=',')
w_fvm_data = np.load(f"{work_dir}/results/w_data.npy")

x_3dec = w_3dec_data[0, 1:]
x_fvm = np.linspace(0, 100, w_fvm_data.shape[1])

plt.plot(x_3dec, w_3dec_data[-1, 1:], label=f't={w_3dec_data[-1, 0]}', marker='^')
plt.plot(x_fvm, w_fvm_data[-1, :])
plt.legend()
plt.show()

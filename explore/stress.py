from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mypackages import file_io


def stress_peaks(sn):
    return np.max(-sn, axis=1)


def plot_stress_peaks(t, sn_max):
    plt.figure()
    plt.plot(t, sn_max, ".")
    plt.xlabel("Time [s]")
    plt.ylabel("Tensile stress peak [Pa]")
    plt.title("")
    plt.tight_layout()
    plt.show()


def plot_stress_profiles(x_sc, sn, *, n=10, cut=0, title=""):
    x_sn_max = x_sc[np.argmin(sn, axis=1)]
    if cut == 0:
        cut = sn.shape[0]
    plt.figure()
    for i in range(2, cut, (cut) // n):
        plt.plot(x_sc, -sn[i], ".-")
        plt.plot(x_sn_max[i], -np.min(sn[i]), "k.")
    plt.xlabel("Distance [m]")
    plt.ylabel("Tensile stress [Pa]")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# %%
# def main():
result_dir = Path.cwd() / "results" / "3dec" / "runs"
run = file_io.read_run(result_dir / "run-q-1e-02.hdf5")

x_sc, sn = file_io.sort_fields(run.x_sc, run.sn)
sn_max = stress_peaks(sn)

# plot_stress_peaks(run.t[:-250], sn_max[:-250])
title = f"$q_0={run.params.q_0:.0e}\\mathrm{{m^2/s}}$"
plot_stress_profiles(x_sc, sn, n=10, cut=0, title=title)


# if __name__ == "__main__":
#     main()

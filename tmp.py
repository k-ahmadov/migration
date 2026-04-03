import h5py
from pathlib import Path
import matplotlib.pyplot as plt

filepath = Path.cwd() / "results" / "3dec" / "runs" / "run_01.hdf5"
with h5py.File(filepath, "r") as f:
    x_vertices = f["coordinates/x_vertices"][:]
    x_subcontacts = f["coordinates/x_subcontacts"][:]
    p_tx = f["fields/fluid_pressure"][:]
    sn_tx = f["fields/stress_normal"][:]
    tau_tx = f["fields/stress_shear"][:]
    t_points = f["coordinates/t"][:]

from pathlib import Path

import h5py
import matplotlib.pyplot as plt

filepath = Path.cwd() / "tmp.hdf5"
with h5py.File(filepath, "r") as f:
    x_vert = f["coordinates/x_vertices"][()]
    # id_vert = f["coordinates/id_vertices"][()]
    # x_sc = f["coordinates/x_subcontacts"][()]
    w_tx = f["fields/aperture"][()]

# print(len(x_vert), len(w))
print(w_tx.shape, '\n')
#
fig, ax = plt.subplots()
ax.plot(x_vert, w_tx[0],'.')
ax.set_xlabel("x label")
ax.set_ylabel("y label")
ax.set_title("title")
fig.tight_layout()
plt.show()

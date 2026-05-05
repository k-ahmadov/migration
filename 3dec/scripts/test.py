from pathlib import Path

import h5py
import itasca as it
import numpy as np


class MeshIDs:
    def __init__(self, filepath):
        with h5py.File(filepath, "r") as f:
            self.vertices = f["coordinates/id_vertices"][()]
            self.subcontacts = f["coordinates/id_subcontacts"][()]
            self.flowzones = f["coordinates/id_flowzones"][()]


def _restore_variables(mesh_ids, element_type, find_func, field_variable):
    ids = getattr(mesh_ids, element_type)
    values = [field_variable(find_func(elem_id)) for elem_id in ids]
    return np.asarray(values, dtype=np.float64)


fp = Path.cwd() / "tmp.hdf5"
mesh_ids = MeshIDs(fp)

vals = _restore_variables(
    mesh_ids,
    'vertices',
    it.flowplane.vertex.find,
    it.flowplane.vertex.Vertex.aperture_hydraulic,
)
print(vals)

print("finished, \n")

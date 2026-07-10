import itasca as it  # pyright: ignore[reportMissingImports]
import numpy as np
from scripts.store_coordinates import _filter_elements

v = it.flowplane.vertex.Vertex
sc = it.block.subcontact.Subcontact
z = it.flowplane.zone.Zone

elem_type = sc
iterable = it.block.subcontact.list()
#iterable = it.flowplane.vertex.list()
# iterable = it.flowplane.zone.list()

x, ids = _filter_elements(
    iterable,
    elem_type.pos_x,
    elem_type.pos_y,
    elem_type.id,
    y_threshold=float(it.fish.get("mesh_size_min")),
)

print(x, "\n")
print(x.shape, "\n")
print(ids.shape, "\n")
print("finished, \n")

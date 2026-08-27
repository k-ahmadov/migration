from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import h5py
import itasca as it  # pyright: ignore[reportMissingImports]
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]


# --- Filtering ---


def _filter_elements(
    iterable: Iterable[Any],
    pos_x: Callable[[Any], float],
    pos_y: Callable[[Any], float],
    get_id: Callable[[Any], int],
    y_threshold: float = 2e-1,
) -> tuple[FloatArray, IntArray]:
    x_vals: list[float] = []
    id_vals: list[int] = []
    for elem in iterable:
        if -y_threshold <= pos_y(elem) <= y_threshold:
            x_vals.append(pos_x(elem))
            id_vals.append(get_id(elem))
    x_arr = np.asarray(x_vals, dtype=np.float64)
    id_arr = np.asarray(id_vals, dtype=np.int32)
    x_uniq, ind = np.unique(x_arr, return_index=True)
    ids_uniq = id_arr[ind]
    return (x_uniq, ids_uniq)


# --- Coordinates container ---


@dataclass
class ElementCoords:
    """X-coordinates and IDs of filtered model elements."""

    x: FloatArray
    ids: IntArray


class Coordinates:
    """Filtered element coordinates extracted from the current ITASCA model."""

    vertices: ElementCoords
    subcontacts: ElementCoords
    flowzones: ElementCoords

    def __init__(self, y_threshold: float = float(it.fish.get('mesh_size_min'))) -> None:
        v = it.flowplane.vertex.Vertex
        sc = it.block.subcontact.Subcontact
        z = it.flowplane.zone.Zone

        def extract(iterable: Iterable[Any], elem_type: Any) -> ElementCoords:
            x, ids = _filter_elements(
                iterable,
                elem_type.pos_x,
                elem_type.pos_y,
                elem_type.id,
                y_threshold=y_threshold,
            )
            return ElementCoords(x=x, ids=ids)

        self.vertices = extract(it.flowplane.vertex.list(), v)
        self.subcontacts = extract(it.block.subcontact.list(), sc)
        self.flowzones = extract(it.flowplane.zone.list(), z)


# --- Parameters ---


@dataclass
class Param:
    value: float
    unit: str
    description: str


# --- HDF5 writing ---


def save_coordinates_hdf5(
    filepath: str | Path,
    coords: Coordinates,
    parameters: dict[str, Param],
) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    coord_datasets: dict[str, FloatArray | IntArray] = {
        "x_vertices": coords.vertices.x,
        "id_vertices": coords.vertices.ids,
        "x_subcontacts": coords.subcontacts.x,
        "id_subcontacts": coords.subcontacts.ids,
        "x_flowzones": coords.flowzones.x,
        "id_flowzones": coords.flowzones.ids,
    }

    with h5py.File(filepath, "w") as f:
        g_coords = f.create_group("coordinates")
        for name, data in coord_datasets.items():
            g_coords.create_dataset(
                name, data=data, compression="gzip", compression_opts=4
            )

        g_params = f.create_group("parameters")
        for name, p in parameters.items():
            ds = g_params.create_dataset(name, data=p.value)
            ds.attrs["unit"] = p.unit
            ds.attrs["description"] = p.description


# --- Main ---


def main() -> None:
    out_file = Path(it.fish.get("filepath"))

    coords = Coordinates()

    def fish(key: str) -> float:
        return float(it.fish.get(key))

    parameters: dict[str, Param] = {
        "k_n": Param(fish("k_n"), "Pa/m", "Normal stiffness"),
        "k_s": Param(fish("k_s"), "Pa/m", "Shear stiffness"),
        "L": Param(fish("L"), "m", "Model size (cube)"),
        "mu": Param(fish("mu"), "Pa.s", "Fluid viscosity"),
        "w_i": Param(fish("w_i"), "m", "Initial aperture"),
        "w_min": Param(fish("w_min"), "m", "Minimum aperture"),
        "w_max": Param(fish("w_max"), "m", "Maximum aperture"),
        "q_0": Param(fish("q_0"), "m^2/s", "Applied injection rate"),
        "E": Param(fish("E"), "Pa/m", "Young's modulus"),
        "nu": Param(fish("nu"), "-", "Poisson's ratio"),
    }

    save_coordinates_hdf5(out_file, coords, parameters)


if __name__ == "__main__":
    main()

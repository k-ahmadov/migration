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
    x_decimals: int = 6,
) -> tuple[FloatArray, IntArray]:
    x_vals: list[float] = []
    id_vals: list[int] = []
    for elem in iterable:
        if -y_threshold <= pos_y(elem) <= y_threshold:
            x_vals.append(pos_x(elem))
            id_vals.append(get_id(elem))
    x_arr = np.asarray(x_vals, dtype=np.float64)
    id_arr = np.asarray(id_vals, dtype=np.int32)
    # The same x on different y-rows differs in the last digits, so exact
    # np.unique keeps near-duplicate points (~1e-15 m apart). Round before
    # de-duplicating (1e-6 m << mesh size), but keep the original x values.
    _, ind = np.unique(np.round(x_arr, x_decimals), return_index=True)
    return (x_arr[ind], id_arr[ind])


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

    def __init__(self, y_threshold: float | None = None) -> None:
        # default: one zone size (`dx` from geometry.dat); read here, not as a
        # default argument, so the FISH global is looked up only when called
        if y_threshold is None:
            y_threshold = float(it.fish.get("dx"))
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


# inj_key -> (unit, description), as in fracinj.Parameters
INJECTION_DRIVERS: dict[str, tuple[str, str]] = {
    "q_0": ("m^2/s", "Constant injection rate"),
    "m_q": ("m^2/s^2", "Injection-rate ramp slope"),
    "DP": ("Pa", "Constant overpressure"),
}


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
        "L": Param(fish("L"), "m", "Fracture length"),
        "mu": Param(fish("mu"), "Pa.s", "Fluid viscosity"),
        "w_i": Param(fish("w_i"), "m", "Initial aperture"),
        "w_min": Param(fish("w_min"), "m", "Minimum aperture"),
        "w_max": Param(fish("w_max"), "m", "Maximum aperture"),
        "E": Param(fish("E"), "Pa", "Young's modulus"),
        "nu": Param(fish("nu"), "-", "Poisson's ratio"),
        "T": Param(fish("T_fin"), "s", "Simulated duration"),
        "sigma0": Param(
            fish("sigma0"), "Pa", "In-situ normal stress sigma_zz (compression < 0)"
        ),
    }

    # Injection driver: `inj_key` (set by injection.dat) is q_0, m_q or DP,
    # matching the mutually exclusive drivers of fracinj.Parameters.
    key = str(it.fish.get("inj_key"))
    unit, description = INJECTION_DRIVERS[key]
    parameters[key] = Param(fish("inj_value"), unit, description)

    save_coordinates_hdf5(out_file, coords, parameters)


if __name__ == "__main__":
    main()

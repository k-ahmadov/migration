import itasca as it
import h5py
from pathlib import Path
from scripts.helpers import apply_function_to_iterable

class XCoordinates:
    def __init__(self):
        self.vertices = apply_function_to_iterable(
            func=it.flowplane.vertex.Vertex.pos_x,
            iterable=it.flowplane.vertex.list(),
            count=it.flowplane.vertex.count()
        )


        self.subcontacts = apply_function_to_iterable(
            func=it.block.subcontact.Subcontact.pos_x,
            iterable=it.block.subcontact.list(),
            count=it.block.subcontact.count()
        )


        self.flowzones = apply_function_to_iterable(
            func=it.flowplane.zone.Zone.pos_x,
            iterable=it.flowplane.zone.list(),
            count=it.flowplane.zone.count()
        )

class Param:
    def __init__(self, value: float, unit: str, description: str) -> None:
        self.value = value
        self.unit = unit
        self.description = description

def save_coordinates_hdf5(filepath, x_coords, parameters):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, "a") as f:  # <-- append mode
        g_coords = f.require_group("coordinates")

        # Replace datasets if they already exist
        for name, data in {
            "x_vertices": x_coords.vertices,
            "x_flowzones": x_coords.flowzones,
            "x_subcontacts": x_coords.subcontacts,
        }.items():
            if name in g_coords:
                del g_coords[name]
            g_coords.create_dataset(name, data=data, compression="gzip", compression_opts=4)

        g_params = f.require_group("parameters")
        for name, P in parameters.items():
            if name in g_params:
                del g_params[name]
            dset = g_params.create_dataset(name, data=P.value)
            dset.attrs["unit"] = P.unit
            dset.attrs["description"] = P.description

def main():
    out_dir = Path.cwd().parent / "results" / "3dec" / "runs"
    out_file = out_dir / "run-q-1e-03.hdf5"

    x_coords = XCoordinates()

    parameters: dict[str, Param] = {
        "k_n": Param(float(it.fish.get("k_n")), "Pa/m", "Normal stiffness"),
        "k_s": Param(float(it.fish.get("k_s")), "Pa/m", "Shear stiffness"),
        "L": Param(float(it.fish.get("L")), "m", "Model size (cube)"),
        "mu": Param(float(it.fish.get("mu")), "Pa.s", "Fluid viscosity"),
        "w_i": Param(float(it.fish.get("w_i")), "m", "Initial aperture"),
        "w_min": Param(float(it.fish.get("w_min")), "m", "Minimum aperture"),
        "w_max": Param(float(it.fish.get("w_max")), "m", "Maximum aperture"),
        "T": Param(float(it.fish.get("T")), "s", "Duration"),
        "q_0": Param(float(it.fish.get("q_0")), "m^2/s", "Applied injection rate"),
        "E": Param(float(it.fish.get("E")), "Pa/m", "Young's modulus"),
        "nu": Param(float(it.fish.get("nu")), "-", "Poisson's ratio"),
    }

    save_coordinates_hdf5(out_file, x_coords, parameters)

if __name__=="__main__":
    main()

"""Export flow-plane zone areas and flow knot volumes to CSV.

Run inside 3DEC once the model is set up and after "model step 0":
    [global flow_mesh_tag = "area-min-0.1"]   ; optional, default "base"
    program call 'scripts/export_flow_mesh.py'

Files go to results/flow_mesh/<tag>/ so runs can be compared.

zones.csv : x, y, z, area              (flow-plane zones)
knots.csv : x, y, z, area, vol, timestep (flow knots)
"""

from pathlib import Path

import itasca as it  # pyright: ignore[reportMissingImports]
import numpy as np

try:
    tag = str(it.fish.get("flow_mesh_tag"))
except Exception:
    tag = "base"
OUT_DIR = Path("/home/kahmadov/phd/migration/3dec/data/flow_mesh") / tag
OUT_DIR.mkdir(parents=True, exist_ok=True)

zones = np.array(
    [(z.pos_x(), z.pos_y(), z.pos_z(), z.area()) for z in it.flowplane.zone.list()]
)
knots = np.array(
    [
        (k.pos_x(), k.pos_y(), k.pos_z(), k.area(), k.vol(), k.timestep())
        for k in it.flowknot.list()
    ]
)

np.savetxt(
    OUT_DIR / "zones.csv", zones, delimiter=",", header="x,y,z,area", comments=""
)
np.savetxt(
    OUT_DIR / "knots.csv",
    knots,
    delimiter=",",
    header="x,y,z,area,vol,timestep",
    comments="",
)
it.command(
    f"[io.out('flow mesh exported: {len(zones)} zones, {len(knots)} knots -> {OUT_DIR}')]"
)

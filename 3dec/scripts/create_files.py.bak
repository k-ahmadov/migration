import itasca as it
import csv

# Function to write CSV headers
def write_header(filename, x_values):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([
            ["time"] + ["id_{}".format(i) for i in range(len(x_values))],
            ["x"] + x_values
        ])

kn = int(float(it.fish.get('kn')) / 1e9)
base_filename = "kn-{}e9-sn-5e6.csv".format(kn)

# x positions
x_fpv = list(map(float, (it.flowplane.vertex.Vertex.pos_x(v) for v in it.flowplane.vertex.list())))
x_bsc = list(map(float, (it.block.subcontact.Subcontact.pos_x(s) for s in it.block.subcontact.list())))
x_fpz = list(map(float, (it.flowplane.zone.Zone.pos_x(z) for z in it.flowplane.zone.list())))

# Writing headers to CSV files
for var in ["P", "w"]:
    write_header("{}-{}".format(var, base_filename), x_fpv)
for var in ["sn"]:
    write_header("{}-{}".format(var, base_filename), x_bsc)
for var in ["q", "v"]:
    write_header("{}-{}".format(var, base_filename), x_fpz)

print("Headers written to all CSV files successfully.")

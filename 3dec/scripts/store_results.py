import itasca as it
import csv

sn_arr = []
for bsc in it.block.subcontact.list():
    sigma_eff_n = float(it.block.subcontact.Subcontact.stress_norm(bsc))
    pressure = float(it.block.subcontact.Subcontact.pp(bsc))
    sn_arr.append(sigma_eff_n + pressure)

# Extract hydraulic aperture, pressure
w_arr, P_arr = [], []
for fpv in it.flowplane.vertex.list():
    aperture = float(it.flowplane.vertex.Vertex.aperture_hydraulic(fpv))
    knot = it.flowplane.vertex.Vertex.knot(fpv)
    pressure = float(it.flowknot.Flowknot.pp(knot))
    w_arr.append(aperture)
    P_arr.append(pressure)
        
# Extract normal stress values
sn_arr = []
for bsc in it.block.subcontact.list():
    sigma_eff_n = float(it.block.subcontact.Subcontact.stress_norm(bsc))
    pressure = float(it.block.subcontact.Subcontact.pp(bsc))
    sn_arr.append(sigma_eff_n + pressure)

q_arr = []
v_arr = []
for fpzp in it.flowplane.zone.list():
    q = it.flowplane.zone.Zone.discharge_x(fpzp)
    v = it.flowplane.zone.Zone.velocity_x(fpzp)
    q_arr.append(q)
    v_arr.append(v)
    
# Retrieve fluid time and parameters
t = float(it.fish.get('fluidTime'))
kn = int(float(it.fish.get('kn')) / 1e9)

# Function to write data to CSV files
def write_to_csv(filename, time, values):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([time] + values)

# File writing
base_filename = "kn-{}e9-sn-5e6.csv".format(kn)

write_to_csv("P-" + base_filename, t, P_arr)
write_to_csv("w-" + base_filename, t, w_arr)
write_to_csv("sn-" + base_filename, t, sn_arr)
write_to_csv("q-" + base_filename, t, q_arr)
write_to_csv("v-" + base_filename, t, v_arr)

import opendssdirect as dss
import numpy as np
import matplotlib.pyplot as plt

# Initialize OpenDSS
dss.Text.Command('clear')
dss.Text.Command('set defaultBaseFrequency=60')

# Define the IEEE 33-bus system
circuit_def = """
New Circuit.IEEE33 bus1=1 basekv=12.66 pu=1.0 phases=3 MVAsc3=80 MVAsc1=80
"""
dss.Text.Command(circuit_def)

# Define Buses and Lines (Modify this as per IEEE 33 bus data)
line_data = [
        (1, 2, 0.5, 0.0922, 0.0470),
        (2, 3, 0.5, 0.493, 0.2511),
        (3, 4, 0.5, 0.366, 0.1864),
        (4, 5, 0.5, 0.3811, 0.1941),
        (5, 6, 0.5, 0.819, 0.707),
        (6, 7, 0.5, 0.1872, 0.6188),
        (7, 8, 0.5, 1.7114, 1.2351),
        (8, 9, 0.5, 1.03, 0.74),
        (9, 10, 0.5, 1.04, 0.74),
        (10, 11, 0.5, 0.1966, 0.065),
        (11, 12, 0.5, 0.3744, 0.1238),
        (12, 13, 0.5, 1.468, 1.155),
        (13, 14, 0.5, 0.5416, 0.7129),
        (14, 15, 0.5, 0.591, 0.526),
        (15, 16, 0.5, 0.7463, 0.545),
        (16, 17, 0.5, 1.289, 1.721),
        (17, 18, 0.5, 0.732, 0.574),
        (2, 19, 0.5, 0.164, 0.1565),
        (19, 20, 0.5, 1.5042, 1.3554),
        (20, 21, 0.5, 0.4095, 0.4784),
        (21, 22, 0.5, 0.7089, 0.9373),
        (3, 23, 0.5, 0.4512, 0.3083),
        (23, 24, 0.5, 0.898, 0.7091),
        (24, 25, 0.5, 0.896, 0.7011),
        (6, 26, 0.5, 0.203, 0.1034),
        (26, 27, 0.5, 0.2842, 0.1447),
        (27, 28, 0.5, 1.059, 0.9337),
        (28, 29, 0.5, 0.8042, 0.7006),
        (29, 30, 0.5, 0.5075, 0.2585),
        (30, 31, 0.5, 0.9744, 0.963),
        (31, 32, 0.5, 0.3105, 0.3619),
        (32, 33, 0.5, 0.341, 0.5302)
    ]
for from_bus, to_bus, r, x, length in line_data:
    dss.Text.Command(f"""
        New Line.Line_{from_bus}_{to_bus} bus1={from_bus} bus2={to_bus} length={length} units=km
        ~ R1={r} X1={x} C1=0""")

# Add a PV Generator at Bus 18
dss.Text.Command('New Generator.PV1 bus1=18 phases=3 kv=12.66 kw=300 kvar=100 model=3')

# Solve Power Flow
dss.Text.Command('set mode=snap')
dss.Text.Command('solve')

# Check for errors
dss.Text.Command('?')
print(dss.Text.Result())

# Extract Bus Voltages
voltages_pu = []
distances = []
buses = dss.Circuit.AllBusNames()

for i, bus_name in enumerate(buses):
    dss.Circuit.SetActiveBus(bus_name)
    v_pu = dss.Bus.puVmagAngle()
    
    if v_pu:
        voltages_pu.append(np.mean(v_pu[::2]))  # Extract per-unit voltage magnitude
    
    try:
        distances.append(int(bus_name))
    except ValueError:
        distances.append(i + 1)

# Plot Voltage Profile
plt.figure(figsize=(8, 5))
plt.plot(distances, voltages_pu, marker='o', linestyle='-', color='b')
plt.xlabel('Bus Number')
plt.ylabel('Voltage (p.u.)')
plt.title('Voltage Profile of IEEE 33-Bus System')
plt.grid(True)
plt.show()

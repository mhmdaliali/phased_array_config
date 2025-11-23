import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import os

folder_path = r'rawdata'

# Change the filenames to remove the _ 
for filename in os.listdir(folder_path):
    if "_" in filename:
        new_name = filename.replace("_", ".")
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)

# Get the s2p files in the directory, sort them according to voltage value
files = [f for f in os.listdir(folder_path) if f.endswith('.s2p')]
files.sort(key=lambda f: float(f[1:-4]))  # Sort files based on the number in the filename (vx.s2p)

# Get the voltages from the files names
voltages = []
for fnames in files:
    voltages.append(float(fnames[1:-4])) 

# Frequency at which we want to extract the phase (in Hz)
f0 = 5.8e9  # Example: 1 GHz

# List to store the unwrapped phases and frequency vector
wrapped_phases = []
frequencies = None


# Loop over all sorted .s2p files
for filename in files:
    # Read the .s2p file
    filepath = os.path.join(folder_path, filename)
    network = rf.Network(filepath)

    # Extract frequency vector (in Hz)
    if frequencies is None:
        frequencies = network.f  # Assuming the frequencies are the same for all files

    # Extract the S21 parameter and calculate the phase
    phase = np.angle(network.s[:, 1, 0], deg=False)  # S21 phase in radian
    
    wrapped_phases.append(phase)

# Convert the list of unwrapped phases to a numpy array (matrix)

wrapped_phases_matrix =  np.array(wrapped_phases).T

# Extract phase at frequency f0
# Find the closest frequency to f0
f_index = np.argmin(np.abs(frequencies - f0))


wphase_at_f0 = wrapped_phases_matrix[f_index, :]

wphase_at_f0 = np.unwrap(wphase_at_f0)*180/np.pi
# wphase_at_f0 = wphase_at_f0 + 360
plotdata = np.column_stack((voltages, wphase_at_f0))
np.save("PSH_table.npy", plotdata)
np.savetxt("PSH_table.csv", plotdata, delimiter=",", fmt="%.6f")

# print(plotdata)

# Plot the phase at f0 for all files
plt.figure(figsize=(10, 6))
plt.plot(voltages,wphase_at_f0, marker='o', linestyle='-', color='b')
plt.title(f'Phase at {f0/1e9} GHz')
plt.xlabel('Applied Voltage (volts)')
plt.ylabel('Phase (degrees)')
plt.grid(True)
plt.show()

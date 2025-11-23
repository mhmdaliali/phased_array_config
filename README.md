# phased_array_config

This tool selects the optimal phase-shifter settings for a linear phased-array,
given:

- measured S21 data files (.s2p) for the phase shifters at each control voltage. 
- desired beam steering angle
- number of columns (default: 6)

The algorithm generates a phase-voltage lookup table and then chooses the best vector of phase values that achieves the closest possible
inter-column phase progression. It includes:

- robust phase unwrapping
- selection of the best phase vector from the lookup table
- proper fallback for case of large phase margins

### Features
- Works on raw measurement data files for the phase shifters.
- Option to select any input frequency
- Fast selection of phase states for each element\Column
- Handles phase discontinuities
- Easy to integrate into a real time beamsteering algorithm

### Usage
### 1. Process the S-parameters and extract phase–voltage data
Run the phase-extraction script to process a folder of .s2p files.
Each file represents the S21 response of the phase shifter at a specific control voltage.

This step:
- Reads all .s2p files in the folder
- Extracts the phase at the specified input frequency
- Builds a phase-vs-voltage table
- Saves the result as a .npy file

Command:

    python process_s2p_files.py --path ./S2P_files --freq 5.8e9 --out PSH_table.npy

Output:

    PSH_table.npy    # Phase (deg) vs Voltage (V)


### 2. Generate the beam-steering lookup table (LUT)

This step uses the previously generated PSH_table.npy.

The LUT generator:
- Computes the required phase for each desired range of beam angles
- Selects the best voltage for each of the 6 phase shifters
- Produces a table where each row corresponds to one steering angle

Command:

    python generate_LookUpTable.py --phase-data PSH_table.npy --angles angles.txt --out LookupTable.csv

Output format:

    LookupTable.csv
    # angle_deg, V1, V2, V3, V4, V5, V6

Example entry:

    10, 7.3, 5.8, 2.1, 1.9, 0.4, 6.2


### 3. Integration

Use the generated LookupTable.csv as the command table for the MCU/FPGA driving the 6-element phase-shifter array.


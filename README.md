# phased_array_config

This tool selects the optimal phase-shifter settings for a linear phased-array,
given:

- measured S21 data files (.s2p) for the phase shifters at each control voltage. 
- desired beam steering angle
- working frequency
- elemnents separation in the phased array
- number of columns (default: 6)

The algorithm generates a phase-voltage data table for the phase shifters at the given frequency and then chooses the best vector of voltage values that achieves the closest possible inter-column phase progression. It includes:

- robust phase unwrapping
- selection of the best phase vector from the phase shifter data
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

    python process_s2p_files.py --path rawdata --freq 5.8e9 --out PSH_table

Output:

    PSH_table.npy    # Phase (deg) vs Voltage (V)


### 2. Generate the beam-steering lookup table (LUT)

This step uses the previously generated PSH_table.npy, along with the output range of beam angles, and elements separation in terms of wavelength.

The LUT generator:
- Computes the required phase for each desired range of beam angles
- Selects the best voltage for each of the 6 phase shifters
- Produces a table where each row corresponds to one steering angle

Command:

    python generate_LUT.py --psh PSH_table.npy --out LookupTable.csv --theta 0 45 5 --dlambda 0.638  # Default incidence angle 0, the feed is by a generator
    python generate_LUT.py --psh PSH_table.npy --out LookupTable.csv --theta_in 10 --theta 0 45 5 --dlambda 0.638  # Incidence angle 10, the feed is a receiving array


Output format:

    LookupTable.csv
    # angle_deg, V1, V2, V3, V4, V5, V6

Example entry:

    10, 7.3, 5.8, 2.1, 1.9, 0.4, 6.2


### 3. Integration

Use the generated LookupTable.csv as the command table for the MCU/FPGA driving the 6-element phase-shifter array.

### 4. Optional

If the input of the phased array is a feeding from another receiving array, the incidence angle to that array must be used in the inputs (set to 0 by default). 


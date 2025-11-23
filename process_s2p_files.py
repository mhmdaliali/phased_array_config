import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import os
import argparse

def process_s2p_files(folder_path, f0, out_file):
    # --- Step 1: Clean filenames (remove underscores) ---
    for filename in os.listdir(folder_path):
        if "_" in filename:
            new_name = filename.replace("_", ".")
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)

    # --- Step 2: Collect .s2p files and sort by voltage ---
    files = [f for f in os.listdir(folder_path) if f.endswith('.s2p')]
    files.sort(key=lambda f: float(f[1:-4]))  # 'v1.2.s2p' → 1.2

    # --- Extract voltages from filenames ---
    voltages = [float(f[1:-4]) for f in files]

    wrapped_phases = []
    frequencies = None

    # --- Step 3: Read each S2P file ---
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        network = rf.Network(filepath)

        if frequencies is None:
            frequencies = network.f

        # S21 phase (radians)
        phase = np.angle(network.s[:, 1, 0], deg=False)
        wrapped_phases.append(phase)

    # Convert to matrix
    wrapped_matrix = np.array(wrapped_phases).T

    # --- Step 4: Extract phase at desired frequency f0 ---
    f_index = np.argmin(np.abs(frequencies - f0))
    wphase_at_f0 = wrapped_matrix[f_index, :]
    wphase_at_f0 = np.unwrap(wphase_at_f0) * 180 / np.pi  # degrees

    # --- Step 5: Save results ---
    plotdata = np.column_stack((voltages, wphase_at_f0))
    np.save(out_file + ".npy", plotdata)
    np.savetxt(out_file + ".csv", plotdata, delimiter=",", fmt="%.6f")

    # --- Step 6: Optional plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(voltages, wphase_at_f0, marker='o')
    plt.title(f'Phase at {f0 / 1e9:.3f} GHz')
    plt.xlabel('Applied Voltage (V)')
    plt.ylabel('Phase (degrees)')
    plt.grid(True)
    plt.show()

    print(f"\nSaved: {out_file}.npy and {out_file}.csv")
    return plotdata


# ======================
# MAIN with argparse
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract phase-vs-voltage from S2P files."
    )

    parser.add_argument("--path", "-p", type=str, required=True,
                        help="Folder containing .s2p files")

    parser.add_argument("--freq", "-f", type=float, required=True,
                        help="Frequency (Hz) at which to extract the phase")

    parser.add_argument("--out", "-o", type=str, default="PSH_table",
                        help="Output filename (without extension)")

    args = parser.parse_args()

    process_s2p_files(args.path, args.freq, args.out)

import math
import numpy as np
import pandas as pd
import argparse

def select_phases(phases, beta, N, debug=False):
    """
    Select N phases from a sorted vector 'phases' starting from the first phase,
    building a tree of candidates with differences close to beta within tolerance,
    and choosing the vector that minimizes the total relative error.
    All angles are in degrees.
    """
    if N > len(phases) or N < 1 or abs(beta) >= 360:
        if debug:
            print("Invalid input: N > len(phases) or N < 1 or beta >= 360")
        return []
    if abs(beta) <= 1:
        return np.repeat(phases[0], N)

    M = len(phases)
    if M >= 2:
        ph_resol = min(np.diff(phases))
        ph_resol = min(5.0, ph_resol)
    ph_tol = max(ph_resol, 0.1 * abs(beta))

    for nn in range(M):
        selection_tree = phases[nn].reshape(1, 1)
        last_indices = [nn]
        for kk in range(1, N):
            new_row_indices = []
            rep_sizes = []
            for idx in last_indices:
                branch_indices = []
                for jj in range(idx + 1, M):
                    delta_ph = phases[jj] - phases[idx] - abs(beta)
                    if -ph_tol <= delta_ph <= ph_tol:
                        branch_indices.append(jj)
                for jj in range(idx):
                    delta_ph = phases[jj] - phases[idx] + 360 - abs(beta)
                    if -ph_tol <= delta_ph <= ph_tol:
                        branch_indices.append(jj)
                if not branch_indices:
                    if idx + 2 < M:
                        rel_idx = np.argmin(np.abs((phases[idx + 1:M] - phases[idx]) - abs(beta)))
                        branch_indices.append(idx + 1 + rel_idx)
                    if idx > 1:
                        rel_idx = np.argmin(np.abs((phases[0:idx] - phases[idx]) + 360 - abs(beta)))
                        branch_indices.append(rel_idx)
                new_row_indices.extend(branch_indices)
                rep_sizes.append(len(branch_indices))

            new_columns = [np.tile(col.reshape(-1, 1), (1, reps)) for col, reps in zip(selection_tree.T, rep_sizes)]
            selection_tree = np.hstack(new_columns)
            selection_tree = np.vstack((selection_tree, phases[new_row_indices]))
            last_indices = new_row_indices

        tree_diff = np.diff(selection_tree, axis=0)
        result = np.sum((np.abs(tree_diff) - abs(beta)) ** 2, axis=0)
        indx_min = np.argmin(result)
        best_candidate = selection_tree[:, indx_min]
        diff_best_candidate = np.diff(best_candidate)
        positive_diff = [x for x in diff_best_candidate if x > 0]
        negative_diff = [x for x in diff_best_candidate if x < 0]
        if all(abs(x - abs(beta)) < 0.25 * abs(beta) for x in positive_diff) and all(abs(x + 360 - abs(beta)) < 0.25 * abs(beta) for x in negative_diff):
            break
    if beta < 0:
        best_candidate = best_candidate[::-1]

    return best_candidate


def generate_lookup_table(psh_file, out_file, theta_out_range=(0, 45, 5),
                          d_lambda=0.638, theta_in_value=0, debug=False):
    # Load phase-shifter data
    PSH_data = np.array(np.load(psh_file))
    voltages = PSH_data[:, 0]
    phases = PSH_data[:, 1]

    # Sort phases
    sort_indices = np.argsort(phases)
    phases = phases[sort_indices]
    voltages = voltages[sort_indices]

    # Beam angles
    theta_in_ticks = [theta_in_value]  # user-defined or default input angle
    theta_out_ticks = np.arange(*theta_out_range)

    theta_in, theta_out = np.meshgrid(theta_in_ticks, theta_out_ticks)
    theta_in_vec = theta_in.flatten(order="F")
    theta_out_vec = theta_out.flatten(order="F")

    # Beta calculation
    beta = 360 * d_lambda * (np.sin(np.deg2rad(theta_in_vec)) - np.sin(np.deg2rad(theta_out_vec)))

    N = len(beta)
    out_voltages = np.zeros((N, 6))
    for kk in range(N):
        selected_phases = select_phases(phases, beta[kk], 6, debug=debug)
        selected_phases = np.array(selected_phases)
        indices = [np.where(phases == val)[0][0] for val in selected_phases]
        out_voltages[kk, :] = voltages[indices]

    full_data = np.column_stack((theta_in_vec, theta_out_vec, out_voltages))
    columns_labels = ['Source', 'Output Beam', 'Col_1', 'Col_2', 'Col_3', 'Col_4', 'Col_5', 'Col_6']
    df = pd.DataFrame(full_data, columns=columns_labels)
    df.to_csv(out_file, index=False)

    if debug:
        print(f"Lookup table saved to {out_file}")
    return df


# ========================
# MAIN with argparse
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate lookup table for 6-element phased array")
    parser.add_argument("--psh", "-p", type=str, required=True, help="Input .npy file from S2P processing")
    parser.add_argument("--out", "-o", type=str, default="LookupTable.csv", help="Output CSV file")
    parser.add_argument("--theta", "-t", type=float, nargs=3, default=[0, 45, 5],
                        help="Theta_out range: start stop step (deg), e.g., 0 45 5")
    parser.add_argument("--theta_in", "-ti", type=float, default=0,
                        help="Input (incidence) angle in degrees for feeding array, default=0")
    parser.add_argument("--dlambda", "-d", type=float, default=0.638, help="d/lambda ratio")
    parser.add_argument("--debug", action="store_true", help="Enable debug printing")
    args = parser.parse_args()

    generate_lookup_table(args.psh, args.out, theta_out_range=args.theta,
                          d_lambda=args.dlambda, theta_in_value=args.theta_in,
                          debug=args.debug)


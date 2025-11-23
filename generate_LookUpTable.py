import math
import numpy as np
import pandas as pd


def select_phases(phases, beta, N, debug=False):
    """
    Select N phases from a sorted vector 'phases' starting from the first phase,
    building a tree of candidates with differences close to beta within tolerance,
    and choosing the vector that minimizes the total relative error.
    All angles are in degrees.
    
    Args:
        phases: List of float, sorted phase values in [0, 360].
        beta: Float, target phase difference (< 360 degrees).
        N: Integer, number of phases to select.
        debug: Boolean, if True, print debug information.
    
    Returns:
        List of selected phase values in degrees.
    """
    print(f"beta = {beta}")
    M = len(phases)
    if N > M or N < 1 or beta >= 360:
        if debug:
            print("Invalid input: N > M or N < 1 or beta >= 360")
        return []
    if np.abs(beta) <= 1:
        return np.repeat(phases[0],N)
    # Stage 1: Input Analysis
    # Calculate phase resolution (minimum difference)
    if M >= 2:
        ph_resol = min(np.diff(phases))
        ph_resol = min(5.0,ph_resol)

    ph_tol = max(ph_resol,0.1*np.abs(beta))
    print(f"The applied Phase Tolrence in selection process is {ph_tol}")
    # Starting the phase selection process
    for nn in range(M): # Loop to try different starting phases
        selection_tree = phases[nn].reshape(1, 1)
        best_candidate = []
        last_indices = [nn]
        for kk in range(1, N): # for the chosen start phase, formulate the selection tree of vectors complying with phase_tol
            new_row_indices = []
            rep_sizes = []
            for idx in last_indices: # Tree formulation by branching upon each last index
                branch_indices = []
                for jj in range(idx + 1, M): # The two loops (forward and wrap backward) for phase_tol based selection
                    delta_ph = phases[jj] - phases[idx] - np.abs(beta)
                    if (delta_ph >= - ph_tol) and (delta_ph <= ph_tol): 
                        branch_indices.append(jj)
                for jj in range(idx):
                    delta_ph = phases[jj] - phases[idx] + 360 - np.abs(beta)
                    if (delta_ph >= - ph_tol) and (delta_ph <= ph_tol): 
                        branch_indices.append(jj)
                if not branch_indices: # Fallback, if phase_tol selection returns empty, just take minimum difference whatever it is
                    if idx+2 < M: # We need to take minimum of two remaining differences
                        rel_idx = np.argmin(np.abs((phases[idx+1:M] - phases[idx]) - np.abs(beta)))
                        branch_indices.append(idx + 1 + rel_idx)
                    if idx > 1:
                        rel_idx = np.argmin(np.abs((phases[0:idx] - phases[idx]) + 360 - np.abs(beta)))
                        branch_indices.append(rel_idx)

                new_row_indices.extend(branch_indices)
                rep_sizes.append(len(branch_indices))

            # Repeat each column of the tree
            new_columns = []
            for col, reps in zip(selection_tree.T, rep_sizes):
                new_columns.append(np.tile(col.reshape(-1, 1), (1, reps)))
            selection_tree = np.hstack(new_columns)
            selection_tree = np.vstack((selection_tree, phases[new_row_indices]))
            last_indices = new_row_indices
        tree_diff = np.diff(selection_tree, axis=0)
        result = np.sum((np.abs(tree_diff) - np.abs(beta)) ** 2, axis=0)
        indx_min = np.argmin(result)
        best_candidate = selection_tree[:,indx_min] 
        # Testing the chosen candidate
        diff_best_candidate = np.diff(best_candidate)
        positive_diff = [x for x in diff_best_candidate if x > 0]
        negative_diff = [x for x in diff_best_candidate if x < 0]
        if all(np.abs(x - np.abs(beta)) < 0.25*np.abs(beta) for x in positive_diff) and all(np.abs(x+360-np.abs(beta)) < 0.25*np.abs(beta) for x in negative_diff) :
            break
    if beta < 0:
        best_candidate = best_candidate[::-1]

    return best_candidate

# Reading the phase shifter data and sorting it
PSH_data    = np.array(np.load("PSH_table.npy"))
voltages    = PSH_data[:,0]
phases      = PSH_data[:,1]

sort_indices = np.argsort(phases)
phases = phases[sort_indices]
voltages = voltages[sort_indices]
# The range of source and output angles
theta_in_ticks = 0    # If the array is fed by another receiving array, the angle should be input here
theta_out_ticks = np.arange(0,45,5)
# Adjusting for the look up table grid
theta_in, theta_out = np.meshgrid(theta_in_ticks,theta_out_ticks)
theta_in_vec = theta_in.flatten(order="F")
theta_out_vec = theta_out.flatten(order="F")
# Beta calculation, considering d / lambda
d_lambda = 0.638
beta = 360*d_lambda*(np.sin(np.deg2rad(theta_in_vec)) - np.sin(np.deg2rad(theta_out_vec)))
# Now, getting the selected phases and voltages for each table row
N = len(beta)
out_voltages = np.zeros((N,6))
for kk in range(0,N):
    selected_phases = select_phases(phases, beta[kk], 6)
    selected_phases = np.array(selected_phases)
    print(f"selected_phases = {selected_phases}")
    print(f"differences = {np.diff(selected_phases)}")
    indices = [np.where(phases == val)[0][0] for val in selected_phases]
    out_voltages[kk,:] = voltages[indices]
    
# Adjusting before writing in the csv file
full_data = np.column_stack((theta_in_vec, theta_out_vec, out_voltages))
columns_labels = ['Source' , 'Output Beam' , 'Col_1' , 'Col_2' , 'Col_3' , 'Col_4' , 'Col_5' , 'Col_6'  ]
df = pd.DataFrame(full_data, columns=columns_labels)
df.to_csv('LookupTable.csv', index=False)
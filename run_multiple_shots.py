'''
Pass in a list of shots. Performs a pedestal fit to every time slice in the shot using the master_fit_ne_Te_1D function.
Saves the data as a dictionary in the 'saved_files' directory.
'''


import pickle
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import MDSplus
import eqtools
from eqtools import CModEFIT

from functions.functions_fit_1D import master_fit_ne_Te_1D


##################################
### EDIT LIST OF SHOTS TO FIT HERE
##################################
list_of_shots = [1030523030, 1050413031]

dictionary_of_all_data = {}

for shot in list_of_shots:
    generated_psi_grid, list_of_successful_te_fit_times_ms, list_fitted_te_profiles, list_of_te_reduced_chi_squared, list_of_te_fit_type, list_of_successful_ne_fit_times_ms, list_fitted_ne_profiles, list_of_ne_reduced_chi_squared, list_of_ne_fit_type = master_fit_ne_Te_1D(shot, t_min=0, t_max=3000, plot_the_fits=True, set_Te_floor=None, set_ne_floor=None, )
    dictionary_of_all_data[shot] = {
        'generated_psi_grid': generated_psi_grid,

        'te_fit_times_ms': list_of_successful_te_fit_times_ms,
        'te_reduced_chi_squared': list_of_te_reduced_chi_squared,
        'te_fit_type': list_of_te_fit_type,
        'te_fitted_profile': list_fitted_te_profiles, # 2D array. First index gives the profile at the first time slice
        
        'ne_fit_times_ms': list_of_successful_ne_fit_times_ms,
        'ne_reduced_chi_squared': list_of_ne_reduced_chi_squared,
        'ne_fit_type': list_of_ne_fit_type,
        'ne_fitted_profile': list_fitted_ne_profiles # 2D array. First index gives the profile at the first time slice
    }

# Save the dictionary as a pickle file
directory = 'saved_files'
file_path = os.path.join(directory, 'db_of_Thomson_fits.pkl')
os.makedirs(directory, exist_ok=True)

with open(file_path, 'wb') as f:
    pickle.dump(dictionary_of_all_data, f)

# Example plot
shot_data = dictionary_of_all_data[1030523030] # choose a shot
t_idx = 10 #plot the Te data at the 10th time point
plt.plot(
    shot_data['generated_psi_grid'], 
    shot_data['te_fitted_profile'][t_idx], 
    label=rf"{shot_data['te_fit_type'][t_idx]}: $\chi^2$ = {shot_data['te_reduced_chi_squared'][t_idx]:.2f}"
)
plt.xlabel('psi')
plt.ylabel('Te (eV)')
plt.title('Te profile at t = ' + str(shot_data['te_fit_times_ms'][t_idx]) + ' ms')
plt.legend()
plt.show()

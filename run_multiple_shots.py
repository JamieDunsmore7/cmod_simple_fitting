import pickle
import numpy as np
import os
import sys
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import MDSplus
import eqtools
from eqtools import CModEFIT

from functions.functions_fit_1D import master_fit_ne_Te_1D


##################################
### EDIT LIST OF SHOTS TO FIT HERE
##################################
list_of_shots = [1030523030, 1050413031]
num_shots = len(list_of_shots)
dictionary_of_all_data = {}

print(f"\nProcessing {num_shots} shots")
for idx,shot in enumerate(list_of_shots):
    percentage = (idx+1)/num_shots*100
    print(f"Processing shot {shot} ({percentage}% {idx+1}/{num_shots})")
    generated_psi_grid, list_of_successful_te_fit_times_ms, list_fitted_te_profiles, list_of_te_reduced_chi_squared, list_of_te_fit_type, list_of_successful_ne_fit_times_ms, list_fitted_ne_profiles, list_of_ne_reduced_chi_squared, list_of_ne_fit_type = master_fit_ne_Te_1D(shot, t_min=1000, t_max=2000, plot_the_fits=True,verbose=0)
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
print("Finished processing all shots")

# Save the dictionary as a pickle file
directory = 'saved_files'
file_path = os.path.join(directory, 'db_of_Thomson_fits.pkl')
os.makedirs(directory, exist_ok=True)










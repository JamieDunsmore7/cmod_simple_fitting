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
list_of_shots = [1030523030, 1050413029]
num_shots = len(list_of_shots)
dictionary_of_all_data = {}

print(f"\nProcessing {num_shots} shots")
for idx,shot in enumerate(list_of_shots):
    percentage = (idx+1)/num_shots*100
    print(f"Processing shot {shot} ({percentage}% {idx+1}/{num_shots})")
    dictionary_of_all_data[shot] = master_fit_ne_Te_1D(shot, t_min=1000, t_max=2000, 
                                                       plot_the_fits=False,verbose=0,
                                                       return_processed_raw_data=True)
print("Finished processing all shots")

# Save the dictionary as a pickle file
directory = 'saved_files'
file_path = os.path.join(directory, 'db_of_Thomson_fits.pkl')
os.makedirs(directory, exist_ok=True)
with open(file_path, 'wb') as f:
    pickle.dump(dictionary_of_all_data, f)










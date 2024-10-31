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
import xarray as xr

from functions.functions_fit_1D import master_fit_ne_Te_1D


##################################
### EDIT SETTINGS HERE
##################################
list_of_shots = list(np.loadtxt('Cmod_unstable_shotlist.txt',dtype=int))#[1030523030, 1050413029]
t_min=100
t_max=2400
save_type= 'dict' #xarray/netcdf or dict or both
save_path = 'saved_files'

##################################
num_shots = len(list_of_shots)
dictionary_of_all_data = {}
os.makedirs(save_path, exist_ok=True)

print(f"\nProcessing {num_shots} shots")
for idx,shot in enumerate(list_of_shots):
    percentage = (idx+1)/num_shots*100
    print(f"Processing shot {shot} ({percentage}% {idx+1}/{num_shots})")
    shot_data = master_fit_ne_Te_1D(shot, t_min=t_min, t_max=t_max, 
                                                       plot_the_fits=False,verbose=0,
                                                       return_processed_raw_data=True)
    
    
    #Save
    file_path = os.path.join(save_path, f'shot_{shot}.pkl')

    if save_type == 'xarray' or save_type == 'netcdf' or save_type == 'both':
        ds=convert_to_xarray(data=shot_data)
        ds.to_netcdf(save_path+'/shot_'+str(shot)+'.nc')

    if save_type == 'dict' or save_type == 'both':
        
        with open(file_path, 'wb') as f:
            pickle.dump(shot_data, f)

print("Finished processing all shots")
    

def convert_to_xarray(data: dict):
    """ 
    Saves data dictionary from master_fit_ne_Te_1D as xarray dataset in netcdf format

    Arguments:
        data (dict): data dictionary from master_fit_ne_Te_1D
        path (str): path to location to save files to
    """
    # Convert the dictionary to an xarray Dataset
    dataset = xr.Dataset(
        {
            "te_fitted_profile": (["time_te", "psi"], data["te_fitted_profile"]),
            "te_reduced_chi_squared": (["time_te"], data["te_reduced_chi_squared"]),
            "te_fit_type": (["time_te"], data["te_fit_type"]),
            "ne_fitted_profile": (["time_ne", "psi"], data["ne_fitted_profile"]),
            "ne_reduced_chi_squared": (["time_ne"], data["ne_reduced_chi_squared"]),
            "ne_fit_type": (["time_ne"], data["ne_fit_type"]),
        },
        coords={
            "psi": data["generated_psi_grid"],
            "time_te": data["te_fit_times_ms"],
            "time_ne": data["ne_fit_times_ms"],
        }
    )
    #Save the xarray dataset to a netcdf file

    return dataset
    









import numpy as np
import pickle
import xarray as xr

file = 'saved_files/db_of_Thomson_fits.pkl'

#Load the data from pickle file
with open(file, 'rb') as f:
    data = pickle.load(f)

shotlist = data.keys()
num_shots = len(shotlist)

for idx,shot in enumerate(shotlist):
    percentage = (idx+1)/num_shots*100
    print(f"Processing shot {shot} ({percentage}% {idx+1}/{num_shots})")
    
    shot_dict = data[shot]

    # Convert the dictionary to an xarray Dataset
    shot_dataset = xr.Dataset(
        {
            "te_fitted_profile": (["time_te", "psi"], shot_dict["te_fitted_profile"]),
            "te_reduced_chi_squared": (["time_te"], shot_dict["te_reduced_chi_squared"]),
            "te_fit_type": (["time_te"], shot_dict["te_fit_type"]),
            "ne_fitted_profile": (["time_ne", "psi"], shot_dict["ne_fitted_profile"]),
            "ne_reduced_chi_squared": (["time_ne"], shot_dict["ne_reduced_chi_squared"]),
            "ne_fit_type": (["time_ne"], shot_dict["ne_fit_type"]),
        },
        coords={
            "psi": shot_dict["generated_psi_grid"],
            "time_te": shot_dict["te_fit_times_ms"],
            "time_ne": shot_dict["ne_fit_times_ms"],

        }
    )
    #Save the xarray dataset to a netcdf file
    shot_dataset.to_netcdf('saved_files/shot_'+str(shot)+'.nc')






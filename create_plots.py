import pickle
import matplotlib.pyplot as plt
import os

directory = 'saved_files'
file_path = os.path.join(directory, 'db_of_Thomson_fits.pkl')

with open(file_path, 'rb') as f:
    dictionary_of_all_data = pickle.load(f)

shotlist = list(dictionary_of_all_data.keys())

for idx,shot in enumerate(shotlist):
    print(f"Creating plot for shot {shot} ({(idx+1)/len(shotlist)*100:.2f}% {idx+1}/{len(shotlist)})")
    shot_data_directory = os.path.join('plots', str(shot))
    os.makedirs(shot_data_directory, exist_ok=True)
    
    shot_data = dictionary_of_all_data[shot] # choose a shot
    num_times = len(shot_data['te_fit_times_ms'])

    for t_idx in range(num_times):
        time = shot_data['te_fit_times_ms'][t_idx]
        plt.plot(
            shot_data['generated_psi_grid'], 
            shot_data['te_fitted_profile'][t_idx], 
            label=rf"{shot_data['te_fit_type'][t_idx]}: $\chi^2$ = {shot_data['te_reduced_chi_squared'][t_idx]:.2f}"
        )
        plt.xlabel('psi')
        plt.ylabel('Te (eV)')
        plt.title('Te profile at t = ' + str(time) + ' ms')
        plt.legend()

        plt.savefig(f'{shot_data_directory}/Te_profile_t_{time}ms.png',dpi=100)
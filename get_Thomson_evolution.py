import MDSplus
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import sys
import pickle
import shutil, os, scipy, copy
from scipy.interpolate import interp1d, interp2d

sys.path.append('/home/jduns/Documents/phd_research/year_1/ly_alpha_workflow/')

from functions.functions_fit_2D import master_fit_ne_Te_2D_window_smoothing, master_fit_ne_Te_2D_quadratic
import eqtools
from eqtools import CModEFIT

sys.path.append('/home/jduns/Documents/phd_research/year_1/ly_a_inversions/')
#from andres_tomographic_inversion import tomoCMOD



experiment = 'I_1000kA_ICRF_2_5MW'
transition='1'
list_of_good_shots = ['1091210025', '1091210026', '1091210027', '1091210028']
shot_of_interest = '1091210027'
list_of_good_shots = [shot_of_interest] #for single shot case


### FOR SAVING THE DATA ###


directory = f'pedestal_evolution_window_smoothing/single_shots/{shot_of_interest}/transition_{transition}' #for single shot case
#directory = f'pedestal_evolution_params_psi/{experiment}/transition_{transition}' #for multiple shots
os.makedirs(directory, exist_ok=True)

print('directory:', directory)

# Choose the pickle file (four sets of shots to choose from)
absolute_path = '/home/jduns/Documents/phd_research/year_1/transition_info_files/' + str(experiment) + '.pkl'


# Load the dictionary from the pickle file
with open(absolute_path, 'rb') as file:
    experiment_file = pickle.load(file)


for shot in experiment_file:
    if shot in list_of_good_shots:
        t_LH = experiment_file[shot][transition][0]
        t_HL = experiment_file[shot][transition][1]

        t_LH_ms = t_LH*1000
        t_HL_ms = t_HL*1000


        times, psi_grid, Rmid_values, ne_values, ne_error, Te_values, Te_error = master_fit_ne_Te_2D_quadratic(int(shot), t_min = t_LH_ms, t_max = t_HL_ms)

        print(stop)

        # save the data
        np.savetxt(f'{directory}/times.txt', times)
        np.savetxt(f'{directory}/psi_grid.txt', psi_grid)
        np.savetxt(f'{directory}/Rmid_values.txt', Rmid_values)
        np.savetxt(f'{directory}/ne_values.txt', ne_values)
        np.savetxt(f'{directory}/ne_error.txt', ne_error)
        np.savetxt(f'{directory}/Te_values.txt', Te_values)
        np.savetxt(f'{directory}/Te_error.txt', Te_error)



        for t in range(len(times)):
            if t%10 == 0:
                plt.plot(psi_grid, ne_values[t], label = f'time = {times[t]}')
        plt.show()

        for t in range(len(times)):
            if t%10 == 0:
                plt.plot(psi_grid, Te_values[t], label = f'time = {times[t]}')
        plt.show()

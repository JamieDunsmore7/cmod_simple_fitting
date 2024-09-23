            #directory = f'pedestal_evolution_params_psi/{experiment}/transition_{transition}' #for multiple shots

import MDSplus
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import sys
import pickle
import shutil, os, scipy, copy
from functions.functions_fit_2D import master_fit_ne_Te_2D_window_smoothing, master_fit_ne_Te_2D_quadratic
import eqtools
from eqtools import CModEFIT

sys.path.append('/home/jduns/Documents/phd_research/year_1/ly_alpha_workflow/')



# INPUTS
experiment_selection = ['I_800kA_ICRF_2_5MW', 
                       'I_1000kA_ICRF_2_5MW', 
                       'I_800kA_ICRF_2MW', 
                       'I_650kA_ICRF_2_5MW'] # can choose one or multiple experiments

shot_selection = 'all' # 'all' or list of specific shots
transition_selection = 'all' # 'all' or list of specific transitions
evolution_selection = 'quadratic' # 'quadratic' or 'window_smoothing'


# cycle through experiments
for experiment in experiment_selection:

    absolute_path = '/home/jduns/Documents/phd_research/year_1/transition_info_files/' + str(experiment) + '.pkl'

    with open(absolute_path, 'rb') as file:
        experiment_file = pickle.load(file)

    if list_of_shots == 'all':
        list_of_shots = experiment_file.keys()
    else:
        list_of_shots = shot_selection

    # cycle through shots
    for shot in list_of_shots:

        if transition_selection == 'all':
            transitions = experiment_file[shot].keys()
        else:
            list_of_transitions = transition_selection

        # cycle through transitions
        for transition in list_of_transitions:

            directory = f'saved_pedestal_evolutions/single_shots/{experiment}/{shot}/transition_{transition}/{evolution_selection}' #for single shot case
            os.makedirs(directory, exist_ok=True)

            t_LH = experiment_file[shot][transition][0]
            t_HL = experiment_file[shot][transition][1]
            t_LH_ms = t_LH*1000
            t_HL_ms = t_HL*1000

            if evolution_selection == 'quadratic':
                times, psi_grid, Rmid_values, ne_values, ne_error, Te_values, Te_error = master_fit_ne_Te_2D_quadratic(int(shot), t_min = t_LH_ms, t_max = t_HL_ms)

            else:
                times, psi_grid, Rmid_values, ne_values, ne_error, Te_values, Te_error = master_fit_ne_Te_2D_window_smoothing(int(shot), t_min = t_LH_ms, t_max = t_HL_ms)


            directory = f'saved_pedestal_evolutions/single_shots/{experiment}/{shot}/transition_{transition}/{evolution_selection}' #for single shot case

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
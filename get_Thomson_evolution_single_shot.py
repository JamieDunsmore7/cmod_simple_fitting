'''
Perform 2D Thomson fits for single shots in Jamie's database of LH transition shots from 2010.
TODO: make this more general so it just takes in a list of shots and a list of time ranges.
'''

import MDSplus
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import sys
import pickle
import shutil, os, scipy, copy
from functions.functions_fit_2D import master_fit_ne_Te_2D_window_smoothing, master_fit_ne_Te_2D_quadratic, master_fit_2D_alt_combined_shots, plot_outputs_of_2D_fitting
import eqtools
from eqtools import CModEFIT

sys.path.append('/home/jduns/Documents/phd_research/year_1/ly_alpha_workflow/')



# INPUTS
experiment_selection = ['I_800kA_ICRF_2_5MW', 
                       'I_1000kA_ICRF_2_5MW', 
                       'I_800kA_ICRF_2MW', 
                       'I_650kA_ICRF_2_5MW'] # can choose one or multiple experiments

shot_selection = ['1091210027'] # 'all' or LIST of specific shots
transition_selection = '1' # 'all' or LIST of specific transitions
evolution_selection = 'window_smoothing' # 'quadratic' or 'window_smoothing'


# cycle through experiments
for experiment in experiment_selection:

    absolute_path = '/home/jduns/Documents/phd_research/year_1/transition_info_files/' + str(experiment) + '.pkl'

    with open(absolute_path, 'rb') as file:
        experiment_file = pickle.load(file)

    if shot_selection == 'all':
        list_of_shots = experiment_file.keys()
    else:
        list_of_shots = shot_selection
    

    # cycle through shots
    for shot in list_of_shots:

        if transition_selection == 'all':
            list_of_transitions = experiment_file[shot].keys()
        else:
            list_of_transitions = transition_selection

        # cycle through transitions
        for transition in list_of_transitions:

            try:
                t_LH = experiment_file[shot][transition][0]
                t_HL = experiment_file[shot][transition][1]
                t_LH_ms = t_LH*1000
                t_HL_ms = t_HL*1000

                output_dict = {}

                if evolution_selection == 'quadratic':
                    output_dict['times'], output_dict['psi_grid'], output_dict['Rmid_values'], output_dict['ne_values'], output_dict['ne_error'], output_dict['Te_values'], output_dict['Te_error'], \
                    output_dict['Thomson_times_ne'], output_dict['Thomson_times_ne_fit'], output_dict['Thomson_times_ne_fit_err'], \
                    output_dict['Thomson_times_Te'], output_dict['Thomson_times_Te_fit'], output_dict['Thomson_times_Te_fit_err'] = \
                        master_fit_ne_Te_2D_quadratic(int(shot), list_of_t_min = t_LH_ms, list_of_t_max = t_HL_ms, time_window_for_evolution=[20,100]) # NOTE THE TIME WINDOW HERE!!
                    

                else:
                    output_dict['times'], output_dict['psi_grid'], output_dict['Rmid_values'], output_dict['ne_values'], output_dict['ne_error'], output_dict['Te_values'], output_dict['Te_error'], \
                    output_dict['Thomson_times_ne'], output_dict['Thomson_times_ne_fit'], output_dict['Thomson_times_ne_fit_err'], \
                    output_dict['Thomson_times_Te'], output_dict['Thomson_times_Te_fit'], output_dict['Thomson_times_Te_fit_err'] = \
                        master_fit_2D_alt_combined_shots(int(shot), t_LH_ms, t_HL_ms)
                    

                    
                directory = f'saved_pedestal_evolutions/single_shots/{experiment}/{evolution_selection}' #for single shot case
                os.makedirs(directory, exist_ok=True)
                file_path_to_save = os.path.join(directory, f'shot_{shot}_transition_{transition}')
                
                
                # Plot and save how well the fit did
                plot_outputs_of_2D_fitting( output_dict['times'], output_dict['psi_grid'], output_dict['ne_values'], output_dict['ne_error'], output_dict['Te_values'], output_dict['Te_error'], \
                output_dict['Thomson_times_ne'], output_dict['Thomson_times_ne_fit'], \
                output_dict['Thomson_times_Te'], output_dict['Thomson_times_Te_fit'], path_to_save_plots=file_path_to_save)


                file_path_to_save_pickle = file_path_to_save + '.pkl'

                # Save the data as a pickle file
                with open(file_path_to_save_pickle, 'wb') as file:
                    pickle.dump(output_dict, file)

            except:
                print('Could not fit the data for shot ' + str(shot) + ' transition ' + str(transition))

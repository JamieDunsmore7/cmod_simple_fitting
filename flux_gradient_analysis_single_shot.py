'''
Analysis into the core. Single shot case.
'''

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
from Aurora import aurora



from functions_profile_fitting import Osborne_Tanh_linear, Osborne_linear_initial_guesses, remove_zeros, Osborne_Tanh_linear_gradient, add_SOL_zeros
from pedestal_evolution_functions import fit_top, fit_centre, fit_pedestal_evolution_full, pedestal_top_mtanh, linear_function, quadratic_function, fit_ne_pedestal_evolution_Hmode, fit_te_pedestal_evolution_Hmode
from functions_mapping import get_limiter_R_z, get_separatrix_midplane, apply_2pt_shift, Ly2Thom_Time, map_at_arbitrary_t
from functions_neutral_calculations import Lya_to_ion_rate, Lya_to_pflux, get_Deff, Ion_to_pflux, Ion_to_pflux_dndt, Ion_to_pflux_ion, Ion_to_pflux_best, Ion_to_pflux_full_profile
from scipy import stats
from mit_edge_profiles.lyman_data import get_geqdsk_cmod
from mit_edge_profiles.fit_2D import Teu_2pt_model
from functions_inversions import tomoCMOD
import eqtools
from eqtools import CModEFIT

sys.path.append('/home/jduns/Documents/phd_research/year_1/ly_a_inversions/')
#from andres_tomographic_inversion import tomoCMOD

#INITIAL UNITS ARE
#EMISS: Wm-1
#ne: m^-3
#Te: keV
print('ok')

fsize=18



# INPUTS
experiment_selection = ['I_800kA_ICRF_2_5MW', 
                       'I_1000kA_ICRF_2_5MW', 
                       'I_800kA_ICRF_2MW', 
                       'I_650kA_ICRF_2_5MW'] # can choose one or multiple experiments

experiment_selection = ['I_800kA_ICRF_2MW', 
                       'I_650kA_ICRF_2_5MW']


shot_selection = 'all' # 'all' or LIST of specific shots
transition_selection = 'all' # 'all' or LIST of specific transitions
evolution_selection = 'quadratic' # 'quadratic' or 'window_smoothing'



# cycle through experiments
for experiment in experiment_selection:
    print('EXPERIMENT: ', experiment)

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
            #try:
                print('TRYING SHOT: ', shot, 'TRANSITION: ', transition)
                data_path = f'/home/jduns/Documents/phd_research/cmod_simple_fitting/saved_pedestal_evolutions/single_shots/{experiment}/{evolution_selection}/shot_{shot}_transition_{transition}.pkl'

                # Open the pickle file
                try:
                    with open(data_path, 'rb') as f:
                        Thomson_data = pickle.load(f)
                except:
                    print('Could not load Thomson fits for shot ' + str(shot) + ' transition ' + str(transition))
                    continue

                # Thomson data has:
                # ne_values, ne_error, Te_values, Te_error, times, psi_grid, Rmid_values
                    
                for key in Thomson_data.keys():
                    Thomson_data[key] = np.array(Thomson_data[key])


                print(Thomson_data['times'])

                # Calculate the differences between consecutive elements
                time_gaps = np.diff(Thomson_data['times'])
                if np.any(time_gaps > 2):
                    raise ValueError('There is a gap in the timebase of the Thomson data greater than 2ms. This shot should be used with caution.')


                # IMPORTANT: SET A FLOOR ON TE AND NE TO STOP THEM GOING NEGATIVE
                Thomson_data['ne_values'] = np.maximum(Thomson_data['ne_values'], 1e19)
                Thomson_data['Te_values'] = np.maximum(Thomson_data['Te_values'], 10)







                for i in range(len(Thomson_data['times'])):
                    if i % 10 == 0 and i != 0:
                        plt.plot(Thomson_data['psi_grid'], Thomson_data['ne_values'][i], label = Thomson_data['times'][i])
                        #plt.fill_between(Thomson_data['psi_grid'], Thomson_data['ne_values'][i] - Thomson_data['ne_error'][i], Thomson_data['ne_values'][i] + Thomson_data['ne_error'][i], alpha=0.1)
                #plt.legend()
                #plt.show()



                lya_data = {
                    'times': [],
                    'psi_grid': [],
                    'emiss': [],
                    'emiss_error': [],
                    'shots': [],
                    'seps': [],
                    'R0s': [],
                }
                list_of_times = []
                list_of_psi = []
                list_of_emiss = []
                list_of_emiss_error = []
                list_of_shots = [] #this is important to avoid repeating the slow eqtools call
                list_of_seps = []
                list_of_R0 = []




                e = eqtools.CModEFIT.CModEFITTree(int(shot), tree='EFIT20')

                z0 = -0.243



                t_LH = experiment_file[shot][transition][0]
                t_HL = experiment_file[shot][transition][1]
                tr_no = transition

                try:
                    llama_time, llama_r_grid, llama_emiss, llama_emiss_error, llama_back_projection = tomoCMOD(int(shot), 'WB4LY', sys_err=20, r_end=0.84, apply_offsets=True, t_window=[t_LH, t_HL], force_cubic=True)
                except:
                    # skip to the next transition
                    print('Could not load Ly-alpha data for shot ' + str(shot) + ' transition ' + str(transition))
                    continue

                z_array = np.full(len(llama_r_grid), z0)


                #just want the Hmode points
                Hmode_indices = np.where((llama_time >= t_LH) & (llama_time < t_HL))[0]

                for idx in Hmode_indices:
                    #map to psi here
                    t = llama_time[idx]
                    llama_psi_grid = map_at_arbitrary_t('RZ', 'psinorm', [llama_r_grid, z_array], int(shot), t, tree = 'EFIT20', eqtools_file = e) # currently just interpolates
                    separatrix_position = map_at_arbitrary_t('psinorm', 'Rmid', [1], int(shot), t, tree = 'EFIT20', eqtools_file = e)

                    R0_value_at_mag_axis = map_at_arbitrary_t('psinorm', 'Rmid', [0], int(shot), t, tree = 'EFIT20', eqtools_file = e)

                    t_norm = t - t_LH

                    lya_data['times'].append(t_norm)
                    lya_data['psi_grid'].append(llama_psi_grid)
                    lya_data['emiss'].append(llama_emiss[idx])
                    lya_data['emiss_error'].append(llama_emiss_error[idx])
                    lya_data['shots'].append(int(shot))
                    lya_data['seps'].append(separatrix_position)
                    lya_data['R0s'].append(R0_value_at_mag_axis)
                    
                    list_of_times.append(t_norm)
                    list_of_psi.append(llama_psi_grid)
                    list_of_emiss.append(llama_emiss[idx])
                    list_of_emiss_error.append(llama_emiss_error[idx])
                    list_of_shots.append(int(shot))
                    list_of_seps.append(separatrix_position)
                    list_of_R0.append(R0_value_at_mag_axis)

                    '''
                    plt.plot(llama_psi_grid, llama_emiss[idx]/max(llama_emiss[idx]), marker='x', label=str(int(t_norm*1000))+' ms')
                    plt.plot(Thomson_data['psi_grid'], Thomson_data['ne_values'][10]/max(Thomson_data['ne_values'][10]), marker='x')
                    plt.grid(linestyle='--', alpha=0.3)
                    plt.show()
                    '''





                for key in lya_data.keys():
                    lya_data[key] = np.array(lya_data[key])


                # ALL LYMAN ALPHA STUFF
                list_of_times = np.array(list_of_times)
                list_of_psi = np.array(list_of_psi)
                list_of_emiss = np.array(list_of_emiss)
                list_of_emiss_error = np.array(list_of_emiss_error)
                list_of_shots = np.array(list_of_shots)
                list_of_seps = np.array(list_of_seps)
                list_of_R0 = np.array(list_of_R0)

                #curtail the times here. IMPORTANT!!!
                min_t = max(0.02, min(Thomson_data['times'])/1000) # make sure that the Ly-alpha data sits within the range of Thomson data
                max_t = min(0.099, max(Thomson_data['times'])/1000)



                mask = (list_of_times > min_t) & (list_of_times < max_t)

                list_of_times = list_of_times[mask]
                list_of_psi = list_of_psi[mask]
                list_of_emiss = list_of_emiss[mask]
                list_of_emiss_error = list_of_emiss_error[mask]
                list_of_shots = list_of_shots[mask]
                list_of_seps = list_of_seps[mask]
                list_of_R0 = list_of_R0[mask]


                list_of_times_ms = list_of_times * 1000



                list_of_complete_emissivity = []
                list_of_complete_emissivity_error = []

                # MAPS THE LYMAN ALPHA EMISSIVITY ONTO THE THOMSON GRID. JUST PUTS ZERO FOR ALL THE CORE POINTS
                for t_idx in range(len(list_of_times)):
                    Lya_psi_grid = list_of_psi[t_idx]
                    Thomson_mask = Thomson_data['psi_grid'] < min(Lya_psi_grid)
                    points_where_no_emiss = Thomson_data['psi_grid'][Thomson_mask]
                    new_Thomson_mask = (Thomson_data['psi_grid'] > min(Lya_psi_grid)) & (Thomson_data['psi_grid'] < max(Lya_psi_grid))
                    points_where_emiss = Thomson_data['psi_grid'][new_Thomson_mask]

                    f = interp1d(Lya_psi_grid, list_of_emiss[t_idx])
                    new_emiss = f(points_where_emiss)

                    f_error = interp1d(Lya_psi_grid, list_of_emiss_error[t_idx])
                    new_emiss_error = f_error(points_where_emiss)

                    core_emissivity = np.zeros(len(points_where_no_emiss))
                    core_emissivity_error = np.zeros(len(points_where_no_emiss))

                    complete_psi_grid = np.concatenate((points_where_no_emiss, points_where_emiss))
                    complete_emissivity = np.concatenate((core_emissivity, new_emiss))
                    complete_emissivity_error = np.concatenate((core_emissivity_error, new_emiss_error))
                    list_of_complete_emissivity.append(complete_emissivity)
                    list_of_complete_emissivity_error.append(complete_emissivity_error)


                # clip the Ly-alpha data time range so that 


                    


                # TODO: need to curtail the ne and Te values to the emissivity grid I think, in the case that the Thomson grid extends beyond the Ly-alpha grid.


                list_of_complete_emissivity = np.array(list_of_complete_emissivity)
                list_of_complete_emissivity_error = np.array(list_of_complete_emissivity_error)


                '''
                plt.plot(Thomson_data['times'], Thomson_data['ne_values'][:,0], label = complete_psi_grid[0])
                plt.plot(Thomson_data['times'], Thomson_data['ne_values'][:,5], label=complete_psi_grid[5])
                plt.plot(Thomson_data['times'], Thomson_data['ne_values'][:,50], label=complete_psi_grid[50])
                plt.plot(Thomson_data['times'], Thomson_data['ne_values'][:,70], label=complete_psi_grid[70])
                plt.ylabel('ne (m^-3)')
                plt.xlabel('t - t_LH (ms)')
                plt.grid(linestyle='--', alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.show()

                print(stop)
                '''

                #get gradient of ne in time
                dn_dt = np.gradient(Thomson_data['ne_values'], Thomson_data['times']/1000, axis=0) #NOTE: need to get error bars on this


                '''
                # now interpolate onto the Lya timebase
                list_of_times_ms = list_of_times_ms[list_of_times_ms < 100] #Ly-alpha times
                list_of_times_ms = list_of_times_ms[list_of_times_ms > 20] #Ly-alpha times
                list_of_times_ms = list_of_times_ms[list_of_times_ms < max(Thomson_data['times'])] # just in case the Thomson grid is less than 100ms long

                list_of_times = list_of_times[list_of_times < 0.1]
                list_of_times = list_of_times[list_of_times > 0.02] 
                '''

                list_of_ne_on_Lya_timebase = []
                list_of_ne_err_on_Lya_timebase = []
                list_of_Te_on_Lya_timebase = []
                list_of_Te_err_on_Lya_timebase = []
                list_of_dn_dt_on_Lya_timebase = []
                list_of_Rmid_on_Lya_timebase = []
                for r_idx in range(len(complete_psi_grid)):
                    f_Te = interp1d(Thomson_data['times'], Thomson_data['Te_values'][:,r_idx])

                    print(f"f_Te time range: {f_Te.x.min()} to {f_Te.x.max()}")
                    print(f"list_of_times_ms range: {list_of_times_ms.min()} to {list_of_times_ms.max()}")


                    list_of_Te_on_Lya_timebase.append(f_Te(list_of_times_ms))

                    f_Te_err = interp1d(Thomson_data['times'], Thomson_data['Te_error'][:,r_idx])
                    list_of_Te_err_on_Lya_timebase.append(f_Te_err(list_of_times_ms))

                    f_ne = interp1d(Thomson_data['times'], Thomson_data['ne_values'][:,r_idx])
                    list_of_ne_on_Lya_timebase.append(f_ne(list_of_times_ms))

                    f_ne_err = interp1d(Thomson_data['times'], Thomson_data['ne_error'][:,r_idx])
                    list_of_ne_err_on_Lya_timebase.append(f_ne_err(list_of_times_ms))

                    f_dn_dt = interp1d(Thomson_data['times'], dn_dt[:,r_idx])
                    list_of_dn_dt_on_Lya_timebase.append(f_dn_dt(list_of_times_ms))

                    f_Rmid = interp1d(Thomson_data['times'], Thomson_data['Rmid_values'][:,r_idx])
                    list_of_Rmid_on_Lya_timebase.append(f_Rmid(list_of_times_ms))




                list_of_ne_on_Lya_timebase = np.array(list_of_ne_on_Lya_timebase)
                list_of_ne_on_Lya_timebase = list_of_ne_on_Lya_timebase.T

                list_of_Te_on_Lya_timebase = np.array(list_of_Te_on_Lya_timebase)
                list_of_Te_on_Lya_timebase = list_of_Te_on_Lya_timebase.T

                list_of_ne_err_on_Lya_timebase = np.array(list_of_ne_err_on_Lya_timebase)
                list_of_ne_err_on_Lya_timebase = list_of_ne_err_on_Lya_timebase.T

                list_of_Te_err_on_Lya_timebase = np.array(list_of_Te_err_on_Lya_timebase)
                list_of_Te_err_on_Lya_timebase = list_of_Te_err_on_Lya_timebase.T

                list_of_dn_dt_on_Lya_timebase = np.array(list_of_dn_dt_on_Lya_timebase)
                list_of_dn_dt_on_Lya_timebase = list_of_dn_dt_on_Lya_timebase.T
                list_of_Rmid_on_Lya_timebase = np.array(list_of_Rmid_on_Lya_timebase)
                list_of_Rmid_on_Lya_timebase = list_of_Rmid_on_Lya_timebase.T




                for i in range(len(list_of_ne_on_Lya_timebase)):
                    if i % 10 == 0:
                        plt.plot(complete_psi_grid, list_of_ne_on_Lya_timebase[i], label=list_of_times[i], marker='x')
                        plt.fill_between(complete_psi_grid, list_of_ne_on_Lya_timebase[i] - list_of_ne_err_on_Lya_timebase[i], list_of_ne_on_Lya_timebase[i] + list_of_ne_err_on_Lya_timebase[i], alpha=0.1)
                plt.legend()
                #plt.show()
                plt.close()









                for idx in range(len(list_of_complete_emissivity)):
                    if idx%5==0:
                        plt.plot(complete_psi_grid, list_of_complete_emissivity[idx], label=str(int(list_of_times[idx]*1000))+' ms')
                        plt.fill_between(complete_psi_grid, list_of_complete_emissivity[idx] - list_of_complete_emissivity_error[idx], list_of_complete_emissivity[idx] + list_of_complete_emissivity_error[idx], alpha=0.1)
                        break
                plt.ylabel('Emissivity (Wm^-3)')
                plt.xlabel('Psi')
                plt.legend()
                plt.grid(linestyle='--', alpha=0.3)
                plt.tight_layout()
                #plt.show()
                plt.close()

                
                list_of_psis = [0.8, 0.9, 0.99, 1]

                for psi_of_interest in list_of_psis:

                    print('PSI OF INTEREST: ', psi_of_interest)





                    list_of_nes = []
                    list_of_tes = []

                    #now calculate ionisation rate, neutral density and flux at every time point



                    #NEED TO CREATE A BIG PLOT OF ne, Te, emiss, ionisation rate, flux, Deff
                    #CHECK ORDERS OF MAGNTIUTE
                    #Look at neutrals
                    #Do D and V analysis


                    list_of_times_for_plotting = [] # add this in here because sometimes the source can't be calculated. I don't want to add these times.

                    list_of_n = []
                    list_of_n_low = []
                    list_of_n_high = []

                    list_of_grad_n = []
                    list_of_grad_n_low = []
                    list_of_grad_n_high = []

                    list_of_flux = []
                    list_of_flux_low = []
                    list_of_flux_high = []

                    list_of_flux_dn_dt = []
                    list_of_flux_ion = []
                    list_of_flux_ion_low = []
                    list_of_flux_ion_high = []

                    list_of_x = []
                    list_of_x_low = []
                    list_of_x_high = []

                    list_of_y = []
                    list_of_y_low = []
                    list_of_y_high = []

                    list_of_source = []

                    list_of_x_r = []


                    #set up the density and temperature pedestal evolution plots
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
                    #ax1.axvline(x=1, linestyle='-', color='black')
                    #ax1.axvline(x=0.99, linestyle='--', color='black')
                    #ax1.axvline(x=0.98, linestyle='--', color='black')
                    ax1.grid(linestyle='--', alpha=0.3)

                    #ax2.axvline(x=1, linestyle='-', color='black')
                    #ax2.axvline(x=0.99, linestyle='--', color='black')
                    #ax2.axvline(x=0.98, linestyle='--', color='black')
                    ax2.grid(linestyle='--', alpha=0.3)

                    ax3.grid(linestyle='--', alpha=0.3)


                    #ax1.set_xlim([0.9,1.1])
                    #ax2.set_xlim([0.9,1.1])
                    #ax2.set_ylim([0, 1000])

                    ax1.set_xlabel('psi')
                    ax1.set_ylabel('ne (m^-3)')

                    ax2.set_xlabel('psi')
                    ax2.set_ylabel('T (eV)')

                    plt.tight_layout()

                    for idx in range(1, len(list_of_times)):
                        if list_of_times[idx] < 0.09: # can change back to 0.07 is I want
                                time = list_of_times[idx]
                                #print(list_of_times[idx])

                                r_of_interest = 0.898

                                psi_LCFS = 1


                                #plt.plot(list_of_psi[idx], rmid_approx)

                                #ne_profile = 1e20*Osborne_Tanh_linear(list_of_psi[idx], ne_centres[idx], ne_widths[idx], ne_tops[idx], ne_bases[idx], ne_slopes[idx])

                                ne_profile = list_of_ne_on_Lya_timebase[idx]
                                ne_profile_error = list_of_ne_err_on_Lya_timebase[idx]
                                te_profile = list_of_Te_on_Lya_timebase[idx]
                                te_profile_error = list_of_Te_err_on_Lya_timebase[idx]
                                dn_dt = list_of_dn_dt_on_Lya_timebase[idx]
                                emiss = list_of_complete_emissivity[idx]
                                emiss_error = list_of_complete_emissivity_error[idx]

                                Rmid = list_of_Rmid_on_Lya_timebase[idx] #gets the Rmid values that correspond to the psi grid at this time point

                                if complete_psi_grid[0] > 0:
                                    raise ValueError ('The psi grid should have a zero for the conversion from Rmid to Rminor to work')

                                r_minor_new = Rmid - Rmid[0]
                                r_minor_new[0] += 1e-6 #to avoid divide by zero error

                                '''
                                if idx % 10 == 0:
                                    plt.close()
                                    plt.plot(complete_psi_grid, rmid_approx, label='old', marker='x')
                                    plt.plot(complete_psi_grid, Rmid, label='new', marker='x')
                                    plt.xlabel('psi')
                                    plt.ylabel('Rmid')
                                    plt.grid(linestyle='--', alpha=0.3)
                                    plt.legend()
                                    plt.show()
                                '''

                                






                                list_of_nes.append(ne_profile)
                                list_of_tes.append(te_profile)    

                                try:
                                    llama_source = Lya_to_ion_rate(emiss / 1e6, ne_profile/1e6, te_profile, rates_source='adas')
                                except:
                                    plt.plot(complete_psi_grid, ne_profile/1e6, label=str(int(list_of_times[idx]*1000)) + 'ms')
                                    plt.show()

                                    plt.plot(complete_psi_grid, te_profile/1e6, label=str(int(list_of_times[idx]*1000)) + 'ms')
                                    plt.show()
                                llama_source_high = Lya_to_ion_rate((emiss+emiss_error) / 1e6, ne_profile/1e6, te_profile, rates_source='adas')
                                llama_source_low = Lya_to_ion_rate((emiss-emiss_error) / 1e6, ne_profile/1e6, te_profile, rates_source='adas')

                                '''
                                if idx%5==0:
                                    plt.plot(list_of_psi[idx], llama_source, label=str(int(list_of_times[idx]*1000)), marker='x')
                                '''

                                

                                #llama_source_high = Lya_to_ion_rate((llama_emiss_final+llama_error_final), llama_ne/1e6, llama_te, rates_source='adas')
                                #llama_source_low = Lya_to_ion_rate((llama_emiss_final-llama_error_final), llama_ne/1e6, llama_te, rates_source='adas')

                                

                                llama_source = llama_source * 1e6 #convert to m^-3 s^-1
                                llama_source_high = llama_source_high * 1e6
                                llama_source_low = llama_source_low * 1e6
                                list_of_source.append(llama_source)



                                llama_pflux = Ion_to_pflux_full_profile(llama_source, r_minor_new, rates_source='adas', coord_system='spherical',  dn_dt=dn_dt)
                                llama_pflux_high = Ion_to_pflux_full_profile(llama_source_high, r_minor_new, rates_source='adas', coord_system='spherical',  dn_dt=dn_dt)
                                llama_pflux_low = Ion_to_pflux_full_profile(llama_source_low, r_minor_new, rates_source='adas', coord_system='spherical',  dn_dt=dn_dt)

                                llama_pflux_ion = Ion_to_pflux_ion(llama_source, r_minor_new, rates_source='adas', coord_system='spherical')
                                llama_pflux_ion_high = Ion_to_pflux_ion(llama_source_high, r_minor_new, rates_source='adas', coord_system='spherical')
                                llama_pflux_ion_low = Ion_to_pflux_ion(llama_source_low, r_minor_new, rates_source='adas', coord_system='spherical')

                                llama_pflux_dndt = Ion_to_pflux_dndt(llama_source, r_minor_new, rates_source='adas', coord_system='slab', dn_dt=dn_dt)


                                flux = np.interp(psi_of_interest, complete_psi_grid, llama_pflux)
                                flux_high = np.interp(psi_of_interest, complete_psi_grid, llama_pflux_high)
                                flux_low = np.interp(psi_of_interest, complete_psi_grid, llama_pflux_low)

                                flux_ion = np.interp(psi_of_interest, complete_psi_grid, llama_pflux_ion)
                                flux_ion_high = np.interp(psi_of_interest, complete_psi_grid, llama_pflux_ion_high)
                                flux_ion_low = np.interp(psi_of_interest, complete_psi_grid, llama_pflux_ion_low)

                                flux_dn_dt = np.interp(psi_of_interest, complete_psi_grid, llama_pflux_dndt)


                                if idx%5==0:
                                    ax1.plot(r_minor_new, ne_profile, label=str(int(list_of_times[idx]*1000)) + 'ms')
                                    ax2.plot(r_minor_new, dn_dt, label=str(int(list_of_times[idx]*1000)) + 'ms')
                                    #ax1.fill_between(complete_psi_grid, ne_profile - ne_profile_error, ne_profile + ne_profile_error, alpha=0.1)
                                    #ax2.plot(complete_psi_grid, te_profile, label=str(int(list_of_times[idx]*1000)) + 'ms')
                                    ax3.plot(r_minor_new, llama_pflux_dndt, label=str(int(list_of_times[idx]*1000)) + 'ms')
                                    #ax2.fill_between(complete_psi_grid, np.gradient(ne_profile - ne_profile_error), np.gradient(ne_profile + ne_profile_error), alpha=0.1) 




                                list_of_flux.append(flux)
                                list_of_flux_low.append(flux_low)
                                list_of_flux_high.append(flux_high)
                                list_of_flux_ion.append(flux_ion)
                                list_of_flux_ion_low.append(flux_ion_low)
                                list_of_flux_ion_high.append(flux_ion_high)
                                list_of_flux_dn_dt.append(-flux_dn_dt) # just make it negative for plotting purposes later


                                ###
                                ### NOTE THAT I SHOULD HAVE A CORRECTION FACTOR OF 0.9 TO ACCOUNT FOR THE FACT THAT I'M DEALING WITH IONS NOT ELECTRONS HERE
                                ###

                                ne_profile *=0.9
                                ne_profile_error *=0.9



                                n = np.interp(psi_of_interest, complete_psi_grid, ne_profile)
                                list_of_n.append(n)

                                n_high = np.interp(psi_of_interest, complete_psi_grid, ne_profile + ne_profile_error)
                                n_low = np.interp(psi_of_interest, complete_psi_grid, ne_profile - ne_profile_error)

                                list_of_n_high.append(n_high)
                                list_of_n_low.append(n_low)




                                grad_n = np.interp(psi_of_interest, complete_psi_grid, np.gradient(ne_profile, r_minor_new))
                                list_of_grad_n.append(grad_n)

                                grad_n_low = np.interp(psi_of_interest, complete_psi_grid, np.gradient(ne_profile - ne_profile_error, r_minor_new))
                                list_of_grad_n_low.append(grad_n_low)

                                grad_n_high = np.interp(psi_of_interest, complete_psi_grid, np.gradient(ne_profile + ne_profile_error, r_minor_new))
                                list_of_grad_n_high.append(grad_n_high)

                                '''

                                if idx%5==0:
                                    ax1.plot(complete_psi_grid, ne_profile, label=str(int(list_of_times[idx]*1000)) + 'ms')
                                    #ax1.fill_between(complete_psi_grid, ne_profile - ne_profile_error, ne_profile + ne_profile_error, alpha=0.1)
                                    ax2.plot(complete_psi_grid, dn_dt, label=str(int(list_of_times[idx]*1000)) + 'ms')
                                    #ax2.fill_between(complete_psi_grid, te_profile - te_profile_error, te_profile + te_profile_error, alpha=0.1)

                                '''


                                '''
                                if idx%5==0:
                                    ax1.plot(complete_psi_grid, ne_profile, label=str(int(list_of_times[idx]*1000)) + 'ms')
                                    ax1.fill_between(complete_psi_grid, ne_profile - ne_profile_error, ne_profile + ne_profile_error, alpha=0.1)
                                    #ax2.plot(complete_psi_grid, te_profile, label=str(int(list_of_times[idx]*1000)) + 'ms')
                                    ax2.plot(complete_psi_grid, np.gradient(ne_profile, r_minor_new), label=str(int(list_of_times[idx]*1000)) + 'ms')
                                    ax2.fill_between(complete_psi_grid, np.gradient(ne_profile - ne_profile_error), np.gradient(ne_profile + ne_profile_error), alpha=0.1)
                                '''


                                list_of_y.append(flux / n)

                                list_of_y_low.append(flux_low / n_high)
                                list_of_y_high.append(flux_high / n_low)


                                #print(llama_pflux[key_r] / ne_profile[key_r])
                                #print(np.gradient(ne_profile, llama_r_grid)[key_r] / ne_profile[key_r])
                                list_of_x.append(grad_n / n)
                                list_of_x_low.append(grad_n_low / n_high)
                                list_of_x_high.append(grad_n_high / n_low)

                                list_of_times_for_plotting.append(time)
                            #except:
                            #    print('Could not calculate source at time: ', time)




                    '''
                    plt.ylabel('ne (m^-3)')
                    plt.xlabel('psi')
                    plt.axvline(x=1, linestyle='-', color='black')
                    plt.grid(True, linestyle='--', alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    plt.show()
                    '''

                    #Plot the ne and Te profiles
                    ax1.legend()
                    ax2.legend()
                    #plt.show()
                    plt.close()
                            


                    list_of_times = list_of_times_for_plotting # just redfine the times now with only the times that actually could calculate a source.        
                    list_of_times = np.array(list_of_times)

                    list_of_x = np.array(list_of_x)
                    list_of_y = np.array(list_of_y)

                    list_of_flux = np.array(list_of_flux)
                    list_of_flux_dn_dt = np.array(list_of_flux_dn_dt)
                    list_of_flux_ion = np.array(list_of_flux_ion)

                    list_of_n = np.array(list_of_n)
                    list_of_grad_n = np.array(list_of_grad_n)
                    list_of_x_low = np.array(list_of_x_low)
                    list_of_x_high = np.array(list_of_x_high)
                    list_of_y_low = np.array(list_of_y_low)
                    list_of_y_high = np.array(list_of_y_high)
                    list_of_n_low = np.array(list_of_n_low)
                    list_of_n_high = np.array(list_of_n_high)
                    list_of_grad_n_low = np.array(list_of_grad_n_low)
                    list_of_grad_n_high = np.array(list_of_grad_n_high)
                    list_of_flux_low = np.array(list_of_flux_low)
                    list_of_flux_high = np.array(list_of_flux_high)
                    list_of_flux_ion_low = np.array(list_of_flux_ion_low)
                    list_of_flux_ion_high = np.array(list_of_flux_ion_high)








                    times_of_interest = list_of_times < 0.09
                    list_of_times = list_of_times[times_of_interest]

                    mask1 = (list_of_times > 0.005) & (list_of_times < 0.01)
                    mask2 = (list_of_times > 0.01) & (list_of_times < 0.02)
                    mask3 = (list_of_times > 0.02) & (list_of_times < 0.03)
                    mask4 = (list_of_times > 0.03) & (list_of_times < 0.04)



                    # PLOTS
                    directory_to_save_plots = f'flux_gradient_results/single_shots/{experiment}/{evolution_selection}/psi_{int(psi_of_interest*100)}'
                    os.makedirs(directory_to_save_plots, exist_ok=True)



                    #useful plot 1

                    plt.plot(list_of_times, list_of_flux_ion, color='red', label = 'ion')
                    plt.plot(list_of_times, list_of_flux_dn_dt, color='green', label='dn/dt')
                    plt.plot(list_of_times, list_of_flux, color='black', label='total')
                    plt.legend()
                    plt.ylabel('flux')
                    plt.xlabel('t - t_LH')
                    plt.title('At Psi = ' + str(psi_of_interest))
                    plt.grid(linestyle='--', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f'{directory_to_save_plots}/ion_vs_dn_dt_shot{shot}_transition_{transition}.png')
                    #plt.show()
                    plt.close()


                    #useful plot 1
                    plt.plot(list_of_times, list_of_flux_dn_dt/list_of_flux_ion, color='blue')
                    plt.ylabel('Fuelling Efficiency')
                    plt.xlabel('t - t_LH')
                    plt.title('At Psi = ' + str(psi_of_interest))
                    plt.grid(linestyle='--', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f'{directory_to_save_plots}/fuelling_efficiency_shot_{shot}_transition_{transition}.png')
                    #plt.show()
                    plt.close()



                    #useful plot 1
                    plt.scatter(list_of_times, list_of_flux, s = 50, color='red')
                    plt.errorbar(list_of_times, list_of_flux, yerr=[list_of_flux - list_of_flux_low, list_of_flux_high - list_of_flux], fmt='o', color='red')
                    plt.ylabel('flux')
                    plt.xlabel('t - t_LH')
                    plt.title('At Psi = ' + str(psi_of_interest))
                    plt.grid(linestyle='--', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f'{directory_to_save_plots}/flux_shot_{shot}_transition_{transition}.png')
                    #plt.show()
                    plt.close()

                    #useful plot 1
                    plt.scatter(list_of_times, list_of_n, s = 50, color='red')
                    plt.errorbar(list_of_times, list_of_n, yerr=[list_of_n - list_of_n_low, list_of_n_high - list_of_n], fmt='o', color='red')
                    plt.ylabel('n')
                    plt.xlabel('t - t_LH')
                    plt.title('At Psi = ' + str(psi_of_interest))
                    plt.grid(linestyle='--', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f'{directory_to_save_plots}/n_shot_{shot}_transition_{transition}.png')
                    #plt.show()
                    plt.close()

                    #useful plot 2
                    plt.scatter(list_of_times, list_of_grad_n, s = 50, color='navy')
                    plt.errorbar(list_of_times, list_of_grad_n, yerr=[list_of_grad_n - list_of_grad_n_low, list_of_grad_n_high - list_of_grad_n], fmt='o', color='navy')
                    plt.ylabel('grad n')
                    plt.xlabel('t - t_LH')
                    plt.title('At Psi = ' + str(psi_of_interest))
                    plt.grid(linestyle='--', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f'{directory_to_save_plots}/grad_n_shot_{shot}_transition_{transition}.png')
                    #plt.show()
                    plt.close()

                    #useful plot 3
                    plt.errorbar(np.abs(list_of_grad_n), list_of_flux, 
                                xerr=[np.abs(list_of_grad_n - list_of_grad_n_low), np.abs(list_of_grad_n_high - list_of_grad_n)], 
                                yerr=[list_of_flux - list_of_flux_low, list_of_flux_high - list_of_flux], 
                                fmt='none', ecolor='lightgray', elinewidth=1, capsize=3, zorder=1)

                    scatter = plt.scatter(np.abs(list_of_grad_n), list_of_flux, s = 50, c=list_of_times*1000, cmap = 'plasma')
                    cbar = plt.colorbar(scatter)  # Optional: add a colorbar to show the color scale
                    cbar.set_label('t - t_LH (ms)')
                    plt.ylabel('flux')
                    plt.xlabel('|grad n|')
                    plt.title('At Psi = ' + str(psi_of_interest))
                    plt.grid(linestyle='--', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f'{directory_to_save_plots}/flux_vs_grad_n_shot_{shot}_transition_{transition}.png')
                    #plt.show()
                    plt.close()

                    #useful plot 4
                    # Plot the error bars as light black lines behind the points
                    plt.errorbar(
                        list_of_x, list_of_y,
                        xerr=[list_of_x - list_of_x_low, list_of_x_high - list_of_x],
                        yerr=[list_of_y - list_of_y_low, list_of_y_high - list_of_y],
                        fmt='none', ecolor='lightgray', elinewidth=1, capsize=3, zorder=1
                    )
                    scatter = plt.scatter(list_of_x, list_of_y, s = 50, c=list_of_times*1000, cmap = 'plasma')
                    cbar = plt.colorbar(scatter)  # Optional: add a colorbar to show the color scale
                    cbar.set_label('t - t_LH (ms)')
                    plt.ylabel('flux / n')
                    plt.xlabel('grad n / n')
                    plt.title('At Psi = ' + str(psi_of_interest))
                    plt.grid(linestyle='--', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f'{directory_to_save_plots}/D_and_v_{shot}_transition_{transition}.png')
                    #plt.show()
                    plt.close()

                    dict_of_flux_gradient_outputs = {
                        'times' : list_of_times,
                        'flux ion' : list_of_flux_ion,
                        'flux dn/dt' : list_of_flux_dn_dt,
                        'flux' : list_of_flux,
                        'flux low' : list_of_flux_low,
                        'flux high' : list_of_flux_high,

                        'grad n' : list_of_grad_n,
                        'grad n low' : list_of_grad_n_low,
                        'grad n high' : list_of_grad_n_high,

                        'n' : list_of_n,
                        'n low' : list_of_n_low,
                        'n high' : list_of_n_high,

                        'fuelling efficiency' : list_of_flux_dn_dt/list_of_flux_ion,

                        'flux / n' : list_of_y,
                        'flux / n low' : list_of_y_low,
                        'flux / n high' : list_of_y_high,

                        'grad n / n' : list_of_x,
                        'grad n / n low' : list_of_x_low,
                        'grad n / n high' : list_of_x_high
                    }


                    file_path_for_dict = os.path.join(directory_to_save_plots, f'output_data_shot_{shot}_transition_{transition}.pkl')

                    with open(file_path_for_dict, 'wb') as pickle_file:
                        pickle.dump(dict_of_flux_gradient_outputs, pickle_file)
            #except:
            #    print('Could not calculate flux gradient for shot: ', shot, 'transition: ', transition)

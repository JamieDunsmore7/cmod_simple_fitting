### Functions for fitting individual profile slices ###

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import optimize
import MDSplus
from scipy.optimize import curve_fit
from scipy.constants import Boltzmann as kB, e as q_electron, m_p
import eqtools
from eqtools import CModEFIT

from functions.functions_fetching_raw_data import *
from functions.functions_coordinate_mapping import *
from functions.functions_profile_fitting import *
from functions.functions_utility import *
from functions.functions_TS_to_TCI_scaling import *
from functions.functions_two_point_model import *




def master_fit_ne_Te_1D(shot, t_min=0, t_max=5000, scale_core_TS_to_TCI = False, set_Te_floor = 20, set_ne_floor = None, set_minimum_errorbar = True, 
                        remove_zeros_before_fitting = True, add_zero_in_SOL = True, shift_to_2pt_model=False, plot_the_fits = False,
                        return_processed_raw_data = False, return_error_bars_on_fits = False, use_edge_chi_squared = False, enforce_mtanh = False,
                        verbose = 0):
    '''
    INPUTS
    --------
    shot: integer, C-Mod shot number
    t_min: float, minimum time in ms
    t_max: float, maximum time in ms

    scale_core_TS_to_TCI: boolean, if True, the core Thomson data is scaled to match the interferometry data over the course of the shot.
    set_Te_floor: integer, if not None, this sets a minimum value for the Te profile at all time points. [eV]
    set_ne_floor: integer, if not None, this sets a minimum value for the ne profile at all time points. [m^-3]
    set_minimum_errorbar: boolean, if True, a minimum errorbar is set for all points (this is because sometimes the error bars have been set to zero in the core, messing up the fits)
    remove_zeros_before_fitting: boolean, if True, zeros are removed from the data before fitting. Deafult is to only remove zeros < psi = 1, but this can be modified.
    add_zero_in_SOL: boolean, if True, a zero is added at the SOL edge to help the mtanh fit (hardcoded at psi=1.05).
    shift_to_2pt_model: boolean, if True, the Thomson data and fits are shifted post-fit to align the separatrix Te with the 2-point model prediction. NOTE: cubic fits can give unreliable 2-pt model shifts.
    plot_the_fits: boolean, option to plot the fits at each Thomson time point.
    return_processed_raw_data: boolean, if True, the processed raw data is returned as well as the fits. Processing involves adding/removing zeros, as well as shifting according to 2-pt model.
    return_error_bars_on_fits: boolean, if True, a Monte-Carlo approach to errorbar calculation is performed and the errorbars on the fits are returned. This makes the routine much slower.
    use_edge_chi_squared: boolean, if True, the reduced chi squared is calculated between 0.6 < psi < 1.0 only. This is useful if good edge data exists and if a good edge fit is the priority.
    enforce_mtanh: boolean, if True, only an mtanh fit will be used. If this fails then no fit will be returned.
    verbose: integer, if 0, no print statements. If 1, some print statements. If 2, all print statements.

    
    RETURNS
    --------
    generated_psi_grid: 1D array of psi values. This is the x-grid for the ne and Te fits

    list_of_successful_te_fit_times_ms: 1D array of times where the Te fit was successful (in ms)
    list_fitted_te_profiles: 2D array of fitted Te profiles. Each row is a profile in space (i.e list_fitted_te_profiles[0] is the Te profile at the first time-point)
    list_of_te_reduced_chi_squared: 1D array of reduced chi squared values for the Te fits
    list_of_te_fit_type: 1D array of strings, either 'cubic' or 'mtanh' depending on the fit type that yielded the lowest reduced chi squared
    
    list_of_successful_ne_fit_times_ms: 1D array of times where the ne fit was successful (in ms)
    list_fitted_ne_profiles: 2D array of fitted ne profiles. Each row is a profile in space (i.e list_fitted_ne_profiles[0] is the ne profile at the first time-point)
    list_of_ne_reduced_chi_squared: 1D array of reduced chi squared values for the ne fits
    list_of_ne_fit_type: 1D array of strings, either 'cubic' or 'mtanh' depending on the fit type that yielded the lowest reduced chi squared

    DESCRIPTION
    --------
    Function to fit all Thomson data within a certain shot.
    Chooses either a cubic or mtanh fit depending on which gives the lowest reduced chi squared.
    '''


    Thomson_times, ne_array_edge, ne_err_array_edge, te_array_edge, te_err_array_edge, rmid_array_edge, r_array_edge, z_array_edge = get_raw_edge_Thomson_data(shot, t_min=t_min, t_max=t_max)
    Thomson_times_core, ne_array_core, ne_err_array_core, te_array_core, te_err_array_core, rmid_array_core, r_array_core, z_array_core = get_raw_core_Thomson_data(shot, t_min = t_min, t_max = t_max)


    if np.any(Thomson_times != Thomson_times_core):
        print('Thomson times are not the same for core and edge data. This is a problem.')
        raise ValueError('Thomson times are not the same for core and edge data. This is a problem.')


    # if the EFIT20 equilibrium doesn't exist (which is the one on the Thomson timebase), just use the normal ANALYSIS one instead.
    try:
        e = eqtools.CModEFIT.CModEFITTree(int(shot), tree='EFIT20')
    except:
        e = eqtools.CModEFIT.CModEFITTree(int(shot), tree='ANALYSIS')


    # Scale the core Thomson data by the interferometry data.
    # NOTE: this is hardcoded to scale between 500ms and 1500ms for every shot.
    # TODO: set up a catch so that this is only done if the current is > 400kA at 500ms and 1500ms.
    if scale_core_TS_to_TCI==True:
        ne_array_core = scale_core_Thomson(shot, Thomson_times_core, ne_array_core) # SCALE THE CORE THOMSON DATA BY THE INTERFEROMETRY DATA



    list_of_successful_te_fit_times_ms = []
    list_fitted_te_profiles = []
    list_of_te_reduced_chi_squared = []
    list_of_te_fit_type = [] #either cubic or mtanh

    list_of_successful_ne_fit_times_ms = []
    list_fitted_ne_profiles = []
    list_of_ne_reduced_chi_squared = []
    list_of_ne_fit_type = [] #either cubic or mtanh

    # the raw data, which can also be returned if required
    list_of_total_psi_te = []
    list_of_total_te = []
    list_of_total_te_err = []

    list_of_total_psi_ne = []
    list_of_total_ne = []
    list_of_total_ne_err = []

    # errors on the fits, which take some time to run but can be calculated if requested
    list_of_te_fit_errors = []
    list_of_ne_fit_errors = []


    generated_psi_grid_core = np.arange(0, 0.8, 0.01)
    generated_psi_grid_edge = np.arange(0.8, 1.2001, 0.002) #higher resolution at the edge
    generated_psi_grid = np.append(generated_psi_grid_core, generated_psi_grid_edge) #this is the grid that the fits will be placed on

    




    # Often a very good initial guess is actually the fitted mtanh parameters from the last time point.
    # I'm initialising the locations for these variables here.
    ne_params_from_last_successful_fit = None
    te_params_from_last_successful_fit = None



    number_of_te_cubic_fits = 0
    number_of_te_mtanh_fits = 0
    number_of_te_failed_fits = 0

    number_of_ne_cubic_fits = 0
    number_of_ne_mtanh_fits = 0
    number_of_ne_failed_fits = 0



    for t_idx in range(len(Thomson_times)):
            time = Thomson_times[t_idx] #in ms
            time_in_s = time / 1000 #in s

            if verbose > 0:
                print('TIME: ', time)

            # Get the data at this time point.
            raw_ne_edge = ne_array_edge[:,t_idx]
            raw_ne_err_edge = ne_err_array_edge[:,t_idx]
            raw_te_edge = te_array_edge[:, t_idx]
            raw_te_err_edge = te_err_array_edge[:, t_idx]
            raw_rmid_edge = rmid_array_edge[:, t_idx]

            raw_ne_core = ne_array_core[:,t_idx]
            raw_ne_err_core = ne_err_array_core[:,t_idx]
            raw_te_core = te_array_core[:, t_idx]
            raw_te_err_core = te_err_array_core[:, t_idx]
            raw_rmid_core = rmid_array_core[:, t_idx]

            # Catch time points with poor/no data.
            # Often the rmid values are set to -1 in the tree if there is no data at that time point.
            if np.all(np.isnan(raw_rmid_edge)) or np.all(raw_rmid_core < 0):
                if verbose > 0:
                    print('No Edge Thomson data at this time point. Skipping.')
                continue

            #Switch from Rmid to psi coordinates here using eqtools
            raw_psi_edge = e.rho2rho('Rmid', 'psinorm', raw_rmid_edge, time_in_s)
            raw_psi_core = e.rho2rho('Rmid', 'psinorm', raw_rmid_core, time_in_s)

            # Option to remove zeros from the data for fitting.
            if remove_zeros_before_fitting == True:

                # NOTE: Remove the point from BOTH arrays if EITHER array has a zero to keep the arrays the same length.

                raw_ne_psi_edge, raw_te_psi_edge, raw_ne_edge, raw_te_edge, raw_ne_err_edge, raw_te_err_edge = \
                    remove_zeros_from_ne_and_Te_simultaneously(raw_psi_edge, raw_ne_edge, raw_te_edge, raw_ne_err_edge, raw_te_err_edge, core_only=True) #edge
                

                raw_ne_psi_core, raw_te_psi_core, raw_ne_core, raw_te_core, raw_ne_err_core, raw_te_err_core = \
                    remove_zeros_from_ne_and_Te_simultaneously(raw_psi_core, raw_ne_core, raw_te_core, raw_ne_err_core, raw_te_err_core, core_only=True) #core
                

                #raw_te_psi_edge, raw_te_edge, raw_te_err_edge = remove_zeros(raw_psi_edge, raw_te_edge, raw_te_err_edge, core_only=True)
                #raw_ne_psi_edge, raw_ne_edge, raw_ne_err_edge = remove_zeros(raw_psi_edge, raw_ne_edge, raw_ne_err_edge, core_only=True)

                # core
                #raw_te_psi_core, raw_te_core, raw_te_err_core = remove_zeros(raw_psi_core, raw_te_core, raw_te_err_core, core_only=True)
                #raw_ne_psi_core, raw_ne_core, raw_ne_err_core = remove_zeros(raw_psi_core, raw_ne_core, raw_ne_err_core, core_only=True)

            else:
                raw_te_psi_edge = raw_psi_edge
                raw_ne_psi_edge = raw_psi_edge
                raw_te_psi_core = raw_psi_core
                raw_ne_psi_core = raw_psi_core                


            # Add in a zero in the SOL at a pre-specified psi value (hardcoded in the function as 1.05).
            if add_zero_in_SOL == True:
                raw_te_psi_edge, raw_te_edge, raw_te_err_edge = add_SOL_zeros_in_psi_coords(raw_te_psi_edge, raw_te_edge, raw_te_err_edge)
                raw_ne_psi_edge, raw_ne_edge, raw_ne_err_edge = add_SOL_zeros_in_psi_coords(raw_ne_psi_edge, raw_ne_edge, raw_ne_err_edge)


            # combine the core and edge data
            total_psi_te = np.append(raw_te_psi_core, raw_te_psi_edge)
            total_psi_ne = np.append(raw_ne_psi_core, raw_ne_psi_edge)
            total_ne = np.append(raw_ne_core, raw_ne_edge)
            total_ne_err = np.append(raw_ne_err_core, raw_ne_err_edge)
            total_te = np.append(raw_te_core, raw_te_edge)
            total_te_err = np.append(raw_te_err_core, raw_te_err_edge)

            # Sometimes, the error bars in the core are set to zero in the tree (or are very small).
            # This flag artificially places a minimum reasonable error bar on all points to avoid problems like this.
            
            if set_minimum_errorbar == True:
                minimum_ne_mask = (total_ne_err < 2e19) | (total_ne_err < (total_ne * 0.05))
                minimum_te_mask = (total_te_err < 20) | (total_te_err < (total_te * 0.05))
                total_ne_err[minimum_ne_mask] = np.maximum(2e19, total_ne[minimum_ne_mask] * 0.05)
                total_te_err[minimum_te_mask] = np.maximum(20, total_te[minimum_te_mask] * 0.05)








            # remove any points with errorbars of zero as this causes the fit to break and they are probably not physical points anyway
            # applying a combined mask keeps the length of ne and Te the same length, which makes the next steps much easier
            ne_no_errors_mask = total_ne_err != 0
            te_no_errors_mask = total_te_err != 0

            combined_no_errors_mask = np.logical_and(ne_no_errors_mask, te_no_errors_mask)

            total_psi_ne = total_psi_ne[combined_no_errors_mask]
            total_ne = total_ne[combined_no_errors_mask]
            total_ne_err = total_ne_err[combined_no_errors_mask]
            total_psi_te = total_psi_te[combined_no_errors_mask]
            total_te = total_te[combined_no_errors_mask]
            total_te_err = total_te_err[combined_no_errors_mask]


            # remove any points whose value is > 1.5x the central (closest to psi = 0) value.
            # gets rid of some of the very obvious outliers, which can distort the fits.
            ne_physical_points_mask = total_ne < 1.5 * total_ne[0]
            te_physical_points_mask = total_te < 1.5 * total_te[0]

            combined_physical_points_mask = np.logical_and(ne_physical_points_mask, te_physical_points_mask)

            total_psi_ne = total_psi_ne[combined_physical_points_mask]
            total_ne = total_ne[combined_physical_points_mask]
            total_ne_err = total_ne_err[combined_physical_points_mask]
            total_psi_te = total_psi_te[combined_physical_points_mask]
            total_te = total_te[combined_physical_points_mask]
            total_te_err = total_te_err[combined_physical_points_mask]


            # Some more catches on poor-quality raw data, which will lead to bad fits.
            if len(total_psi_te) == 0:
                if verbose > 0:
                    print('No good data for this time point. Skipping.')
                continue

            if len(total_psi_te[total_psi_te<0.8]) < 3:
                if verbose > 0:
                    print('Fewer than 3 TS points below psi = 0.8, so do not try to fit the profile. Skipping')
                continue

            if max(total_te) < 100 or max(total_ne) < 1e19:
                if verbose > 0:
                    print('No good data for this time point. Skipping.')
                continue


            ### FIT TE PROFILE ###

            # Try to get some reasonable initial guesses to help the mtanh fits converge.
            try:
                te_guesses = Osborne_linear_initial_guesses(raw_ne_psi_edge, raw_te_edge)
                te_guesses.extend([0,0]) #just for the quadratic and cubic terms
            except:
                te_guesses = None


            # Add in some more options for initial guesses. The hardcoded has proved to work well in the past.
            list_of_te_guesses = []
            list_of_te_guesses.append(te_guesses)
            list_of_te_guesses.append([ 9.92614859e-01,  4.01791101e-02,  2.55550908e+02,  1.28542623e+01,  2.17777084e-01, -3.45196862e-03,  1.42947373e-04])
            # Want the first guess to be the fit parameters from the last successful mtanh fit (as this will likely be a good guess for the current time point).
            if te_params_from_last_successful_fit is not None:
                list_of_te_guesses.insert(0, te_params_from_last_successful_fit)


            # cycle through initial guesses and try to fit
            for te_guess_idx in range(len(list_of_te_guesses)):
                te_guess = list_of_te_guesses[te_guess_idx]
                try:
                    te_params, te_covariance = curve_fit(Osborne_Tanh_cubic, total_psi_te, total_te, p0=te_guess, sigma=total_te_err, absolute_sigma=True, maxfev=2000, bounds=([0.85, 0.01, 50, -0.001, -np.inf, -np.inf, -np.inf], [1.1, 0.2, 500, np.inf, np.inf, np.inf, np.inf])) #should now be in psi
                    te_fitted = Osborne_Tanh_cubic(generated_psi_grid, te_params[0], te_params[1], te_params[2], te_params[3], te_params[4], te_params[5], te_params[6])
                    
                    #print('te params')
                    #print('centre', te_params[0])
                    #print('width', te_params[1])
                    #print('top', te_params[2])
                    #print('bottom', te_params[3])
                    #print('linear', te_params[4])
                    #print('quadratic', te_params[5])
                    #print('cubic', te_params[6])

                    te_fitted_for_chi_squared = Osborne_Tanh_cubic(total_psi_te, te_params[0], te_params[1], te_params[2], te_params[3], te_params[4], te_params[5], te_params[6])
                    te_chi_squared = reduced_chi_squared_inside_separatrix(total_psi_te, total_te, te_fitted_for_chi_squared, total_te_err, len(te_params), only_edge = use_edge_chi_squared)
                    te_params_from_last_successful_fit = te_params
                    break #guess worked so exit the for loop

                except:
                    if te_guess_idx == len(list_of_te_guesses) - 1:
                        # If all the guesses failed, set the fit parameters to none
                        te_params = None
                        te_covariance = None
                        te_fitted = None

                    else:
                        # If this wasn't the last guess, simply move onto the next one
                        continue

            if te_params is None:
                if verbose > 0:
                    print('Te mtanh fit failed.')

            # Do the cubic fits
            te_params_cubic, te_covariance_cubic = curve_fit(Cubic, total_psi_te, total_te, sigma=total_te_err, absolute_sigma=True, maxfev=2000)
            te_fitted_cubic = Cubic(generated_psi_grid, te_params_cubic[0], te_params_cubic[1], te_params_cubic[2], te_params_cubic[3])

            te_fitted_cubic_for_chi_squared = Cubic(total_psi_te, te_params_cubic[0], te_params_cubic[1], te_params_cubic[2], te_params_cubic[3])
            te_chi_squared_cubic = reduced_chi_squared_inside_separatrix(total_psi_te, total_te, te_fitted_cubic_for_chi_squared, total_te_err, len(te_params_cubic), only_edge = use_edge_chi_squared)

            # print the reduced chi-squared values of the respective fits
            if verbose > 0:
                print(f'te reduced chi squared cubic: {te_chi_squared_cubic:.2f}')
            if te_params is not None:
                if verbose > 0:
                    print(f'te reduced chi squared mtanh: {te_chi_squared:.2f}')

            if enforce_mtanh == True:
                if te_fitted is not None:
                    te_fitted_best = te_fitted
                    te_chi_squared_best = te_chi_squared
                    te_best_fit_type = 'mtanh'
                else:
                    te_fitted_best = None
                    te_chi_squared_best = None
                    te_best_fit_type = None
            
            else:
                # Do not use the mtanh fit if there are fewer than 3 points in the pedestal region.
                if te_params is not None:
                    pedestal_start = te_params[0] - te_params[1]
                    pedestal_end = te_params[0] + te_params[1]
                    pedestal_mask = (total_psi_te > pedestal_start) & (total_psi_te < pedestal_end)
                    if len(total_psi_te[pedestal_mask]) < 3:
                        if verbose > 0:
                            print(f'Reject Te mtanh fit because there are only {len(total_psi_te[pedestal_mask])} points in the pedestal (require at least 3)')
                        te_params = None
                        te_covariance = None
                        te_fitted = None
                
                

                # Do not use the mtanh fit if the reduced chi-squared is below 0 or above 20.
                if te_params is not None:
                    if te_chi_squared <= 0 or te_chi_squared > 20:
                        if verbose > 0:
                            print('Reject Te mtanh fit because the reduced chi squared is below 0 or greater than 20.')
                        te_params = None
                        te_covariance = None
                        te_fitted = None

                # Do not use the cubic fit if the reduced chi-squared is below 0 or above 20.
                if te_params_cubic is not None:
                    if te_chi_squared_cubic <= 0 or te_chi_squared_cubic > 20:
                        if verbose > 0:
                            print('Reject Te cubic fit because the reduced chi squared is below 0 or greater than 20.')
                        te_params_cubic = None
                        te_covariance_cubic = None
                        te_fitted_cubic = None

                # Choose the best fit based on the reduced chi-squared values.
                if te_params is not None and te_params_cubic is not None:
                    if te_chi_squared_cubic < te_chi_squared:
                        if verbose > 0:
                            print('CUBIC IS BEST FIT')
                        te_fitted_best = te_fitted_cubic
                        te_chi_squared_best = te_chi_squared_cubic
                        te_best_fit_type = 'cubic'
                        number_of_te_cubic_fits += 1
                    else:
                        if verbose > 0:
                            print('MTANH IS BEST FIT')
                        te_fitted_best = te_fitted
                        te_chi_squared_best = te_chi_squared
                        te_best_fit_type = 'mtanh'
                        number_of_te_mtanh_fits += 1
                elif te_params is not None:
                    if verbose > 0:
                        print('MTANH IS BEST FIT')
                    te_fitted_best = te_fitted
                    te_chi_squared_best = te_chi_squared
                    te_best_fit_type = 'mtanh'
                    number_of_te_mtanh_fits += 1
                elif te_params_cubic is not None:
                    if verbose > 0:
                        print('CUBIC IS BEST FIT')
                    te_fitted_best = te_fitted_cubic
                    te_chi_squared_best = te_chi_squared_cubic
                    te_best_fit_type = 'cubic'
                    number_of_te_cubic_fits += 1
                else:
                    if verbose > 0:
                        print('NO FITS WORKED')
                    te_fitted_best = None
                    te_chi_squared_best = None
                    te_best_fit_type = None
                    number_of_te_failed_fits += 1


            # Measurement floor on the Thomson system is 20eV on C-Mod.
            # Sometimes a good idea to enforce this floor in case the fit tries to go below zero at any points (particularly for very low temperature shots).
            if set_Te_floor is not None:
                if te_fitted_best is not None:
                    te_fitted_best[te_fitted_best < set_Te_floor] = set_Te_floor

            if set_ne_floor is not None:
                if ne_fitted_best is not None:
                    ne_fitted_best[ne_fitted_best < set_ne_floor] = set_ne_floor
               
            ### FIT NE PROFILE ###

            # Try to get some reasonable initial guesses to help the mtanh fits converge.
            try:
                ne_guesses = Osborne_linear_initial_guesses(raw_ne_psi_edge, raw_ne_edge)
                ne_guesses[2] = ne_guesses[2] / 1e20 #just divide height and base by 1e20 to make the minimisation easier
                ne_guesses[3] = ne_guesses[3] / 1e20
                ne_guesses.extend([0,0]) #just for the quadratic and cubic terms
            except:
                ne_guesses = None


            # Add in some hardcoded initial guesses that have worked well in the past.
            list_of_ne_guesses = []
            list_of_ne_guesses.append(ne_guesses) #just from my own rough method
            list_of_ne_guesses.append([1.00604712, 0.037400836, 2.10662412, 0.0168897974, -0.0632778417, 0.00229233952, -2.0627212e-05]) #good initial guess for 650kA shots
            list_of_ne_guesses.append([1.02123755e+00,  5.02744526e-02,  2.54219267e+00, -9.99999694e-04, 2.58724602e-02, -2.32961078e-03,  4.20279037e-05]) #good guess for 1MA shots
            
            # Want the first guess to be the fit parameters from the last successful mtanh fit (as this will likely be a good guess for the current time point).
            if ne_params_from_last_successful_fit is not None:
                list_of_ne_guesses.insert(0, ne_params_from_last_successful_fit)

            # cycle through initial guesses and try to fit
            for ne_guess_idx in range(len(list_of_ne_guesses)):
                ne_guess = list_of_ne_guesses[ne_guess_idx]
                try:
                    ne_params, ne_covariance = curve_fit(Osborne_Tanh_cubic, total_psi_ne, total_ne/1e20, p0=ne_guess, sigma=total_ne_err/1e20, absolute_sigma=True, maxfev=2000, bounds=([0.85, 0.01, 0, -0.001, -np.inf, -np.inf, -np.inf], [1.1, 0.15, max(total_ne/1e20), np.inf, np.inf, np.inf, np.inf])) #should now be in psi
                    ne_fitted = 1e20*Osborne_Tanh_cubic(generated_psi_grid, ne_params[0], ne_params[1], ne_params[2], ne_params[3], ne_params[4], ne_params[5], ne_params[6])
                    #print('ne params')
                    #print('centre', ne_params[0])
                    #print('width', ne_params[1])
                    #print('top', ne_params[2])
                    #print('bottom', ne_params[3])
                    #print('linear', ne_params[4])
                    #print('quadratic', ne_params[5])
                    #print('cubic', ne_params[6])

                    ne_fitted_for_chi_squared = 1e20*Osborne_Tanh_cubic(total_psi_ne, ne_params[0], ne_params[1], ne_params[2], ne_params[3], ne_params[4], ne_params[5], ne_params[6])
                    ne_chi_squared = reduced_chi_squared_inside_separatrix(total_psi_ne, total_ne, ne_fitted_for_chi_squared, total_ne_err, len(ne_params), only_edge = use_edge_chi_squared)
                    ne_params_from_last_successful_fit = ne_params
                    break #guess worked so exit the for loop
                except:
                    if ne_guess_idx == len(list_of_ne_guesses) - 1:
                        # If all the guesses failed, set the fit parameters to none
                        ne_params = None
                        ne_covariance = None
                        ne_fitted = None
                    else:
                        # If this wasn't the last guess, simply move onto the next one
                        continue

            
            if ne_params is None and verbose > 0:
                print('Ne mtanh fit failed.')


            # Do the cubic fits
            ne_params_cubic, ne_covariance_cubic = curve_fit(Cubic, total_psi_ne, total_ne/1e20, sigma=total_ne_err/1e20, absolute_sigma=True, maxfev=2000)
            ne_fitted_cubic = 1e20*Cubic(generated_psi_grid, ne_params_cubic[0], ne_params_cubic[1], ne_params_cubic[2], ne_params_cubic[3])
            
            ne_fitted_cubic_for_chi_squared = 1e20*Cubic(total_psi_ne, ne_params_cubic[0], ne_params_cubic[1], ne_params_cubic[2], ne_params_cubic[3])
            ne_chi_squared_cubic = reduced_chi_squared_inside_separatrix(total_psi_ne, total_ne, ne_fitted_cubic_for_chi_squared, total_ne_err, len(ne_params_cubic), only_edge = use_edge_chi_squared)

            # print the reduced chi-squared values of the respective fits
            if verbose > 0:
                print(f'ne reduced chi squared cubic: {ne_chi_squared_cubic:.2f}')
                if ne_params is not None:
                    print(f'ne reduced chi squared mtanh: {ne_chi_squared:.2f}')

            if enforce_mtanh == True:
                if ne_fitted is not None:
                    ne_fitted_best = ne_fitted
                    ne_chi_squared_best = ne_chi_squared
                    ne_best_fit_type = 'mtanh'
                else:
                    ne_fitted_best = None
                    ne_chi_squared_best = None
                    ne_best_fit_type = None

            else:          # Choose between the mtanh and the cubic fit.  
                # Do not use the mtanh fit if there are fewer than 3 points in the pedestal region.
                if ne_params is not None:
                    pedestal_start = ne_params[0] - ne_params[1]
                    pedestal_end = ne_params[0] + ne_params[1]
                    pedestal_mask = (total_psi_ne > pedestal_start) & (total_psi_ne < pedestal_end)
                    if len(total_psi_ne[pedestal_mask]) < 3:
                        if verbose > 0:
                            print(f'Reject Ne mtanh fit because there are only {len(total_psi_ne[pedestal_mask])} points in the pedestal (require at least 3)')
                        plt.errorbar(total_psi_ne, total_ne, yerr=total_ne_err, fmt='o')
                        plt.plot(generated_psi_grid, ne_fitted)
                        plt.show()

                        ne_params = None
                        ne_covariance = None
                        ne_fitted = None
                
                
                # Do not use the mtanh fit if the reduced chi-squared is below 0 or above 20.
                if ne_params is not None:
                    if ne_chi_squared <= 0 or ne_chi_squared > 20:
                        if verbose > 0:
                            print('Reject Ne mtanh fit because the reduced chi squared is below 0 or greater than 20.')
                        ne_params = None
                        ne_covariance = None
                        ne_fitted = None
                
                # Do not use the cubic fit if the reduced chi-squared is below 0 or above 20.
                if ne_params_cubic is not None:
                    if ne_chi_squared_cubic <= 0 or ne_chi_squared_cubic > 20:
                        if verbose > 0:
                            print('Reject Ne cubic fit because the reduced chi squared is below 0 or greater than 20.')
                        ne_params_cubic = None
                        ne_covariance_cubic = None
                        ne_fitted_cubic = None


                # Choose the best fit based on the reduced chi-squared values.
                if ne_params is not None and ne_params_cubic is not None:
                    if ne_chi_squared_cubic < ne_chi_squared:
                        if verbose > 0:
                            print('CUBIC IS BEST FIT')
                        ne_fitted_best = ne_fitted_cubic
                        ne_chi_squared_best = ne_chi_squared_cubic
                        ne_best_fit_type = 'cubic'
                        number_of_ne_cubic_fits += 1
                    else:
                        if verbose > 0:
                            print('MTANH IS BEST FIT')
                        ne_fitted_best = ne_fitted
                        ne_chi_squared_best = ne_chi_squared
                        ne_best_fit_type = 'mtanh'
                        number_of_ne_mtanh_fits += 1

                elif ne_params is not None:
                    if verbose > 0:
                        print('MTANH IS BEST FIT')
                    ne_fitted_best = ne_fitted
                    ne_chi_squared_best = ne_chi_squared
                    ne_best_fit_type = 'mtanh'
                    number_of_ne_mtanh_fits += 1

                elif ne_params_cubic is not None:
                    if verbose > 0:
                        print('CUBIC IS BEST FIT')
                    ne_fitted_best = ne_fitted_cubic
                    ne_chi_squared_best = ne_chi_squared_cubic
                    ne_best_fit_type = 'cubic'
                    number_of_ne_cubic_fits += 1

                else:
                    if verbose > 0:
                        print('NO FITS WORKED')
                    ne_fitted_best = None
                    ne_chi_squared_best = None
                    ne_best_fit_type = None
                    number_of_ne_failed_fits += 1

            # Option to estimate errors on the fits using a Monte-Carlo approach.
            if return_error_bars_on_fits == True:
                print('STARTING MONTE CARLO METHOD FOR TE...')
                list_of_perturbed_te_fits = []
                for idx in range(100):
                    perturbed_te_values = np.random.normal(loc = total_te, scale = total_te_err) #perturb the data to see how the fit changes
                    if te_best_fit_type == 'cubic':
                        perturbed_te_params, te_covariance = curve_fit(Cubic, total_psi_te, perturbed_te_values, sigma=total_te_err, absolute_sigma=True, maxfev=2000)
                        perturbed_te_fitted = Cubic(generated_psi_grid, perturbed_te_params[0], perturbed_te_params[1], perturbed_te_params[2], perturbed_te_params[3])
                        list_of_perturbed_te_fits.append(perturbed_te_fitted)

                    else:
                        try:
                            perturbed_te_params, te_covariance = curve_fit(Osborne_Tanh_cubic, total_psi_te, perturbed_te_values, p0=te_params, sigma=total_te_err, absolute_sigma=True, maxfev=2000, bounds=([0.85, 0.01, 50, -0.001, -np.inf, -np.inf, -np.inf], [1.1, 0.2, 500, np.inf, np.inf, np.inf, np.inf]))
                            perturbed_te_fitted = Osborne_Tanh_cubic(generated_psi_grid, perturbed_te_params[0], perturbed_te_params[1], perturbed_te_params[2], perturbed_te_params[3], perturbed_te_params[4], perturbed_te_params[5], perturbed_te_params[6])
                            
                            # Sometimes the perturbed fits can converge but give a value of infinity at psi = 0 when the pedestal width is very small. 
                            # The tiny width causes z to blow up in Osborne Tanh Cubic on the line above, giving the infinities.
                            # Since these are only needed as averages for the error bars, I can safely ignore the infinities by setting them to nan.
                            perturbed_te_fitted[np.isinf(perturbed_te_fitted)] = np.nan
                            list_of_perturbed_te_fits.append(perturbed_te_fitted)

                        except:
                            print('TE IDX: ', idx, ' could not fit')
                            pass


                list_of_perturbed_te_fits = np.array(list_of_perturbed_te_fits)
                list_of_te_fitted_std = np.nanstd(list_of_perturbed_te_fits, axis=0) #np.nanstd is just like np.std but ignores nans


                print('STARTING MONTE CARLO METHOD FOR NE...')
                list_of_perturbed_ne_fits = []
                for idx in range(100):
                    perturbed_ne_values = np.random.normal(loc = total_ne, scale = total_ne_err)
                    if ne_best_fit_type == 'cubic':
                        perturbed_ne_params, ne_covariance = curve_fit(Cubic, total_psi_ne, perturbed_ne_values/1e20, sigma=total_ne_err/1e20, absolute_sigma=True, maxfev=2000)
                        perturbed_ne_fitted = 1e20*Cubic(generated_psi_grid, perturbed_ne_params[0], perturbed_ne_params[1], perturbed_ne_params[2], perturbed_ne_params[3])
                        list_of_perturbed_ne_fits.append(perturbed_ne_fitted)

                    else:
                        try:
                            perturbed_ne_params, ne_covariance = curve_fit(Osborne_Tanh_cubic, total_psi_ne, perturbed_ne_values/1e20, p0=ne_params, sigma=total_ne_err/1e20, absolute_sigma=True, maxfev=2000, bounds=([0.85, 0.001, 0, -0.001, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])) #should now be in psi
                            perturbed_ne_fitted = 1e20*Osborne_Tanh_cubic(generated_psi_grid, perturbed_ne_params[0], perturbed_ne_params[1], perturbed_ne_params[2], perturbed_ne_params[3], perturbed_ne_params[4], perturbed_ne_params[5], perturbed_ne_params[6])
                            perturbed_ne_fitted[np.isinf(perturbed_ne_fitted)] = np.nan
                            list_of_perturbed_ne_fits.append(perturbed_ne_fitted)
                        except:
                            print('NE IDX: ', idx, ' could not fit')
                            pass
  
                list_of_perturbed_ne_fits = np.array(list_of_perturbed_ne_fits)
                list_of_ne_fitted_std = np.nanstd(list_of_perturbed_ne_fits, axis=0) # np.nanstd is just like np.std but ignores nans


            # OPTION TO SHIFT THE DATA AND FIT ACCORDING TO THE 2-PT MODEL
            if shift_to_2pt_model == True and te_fitted_best is not None:
                Te_sep_eV = Teu_2pt_model(shot, time_in_s, lam_T_mm=1, geqdsk=4, pressure_opt = 3, lambdaq_opt=1) #lam_T_mm and geqdsk are just placeholders here. They don't do anything.
                new_x, shift = apply_2pt_shift(generated_psi_grid, te_fitted_best, Te_sep_eV, 1, only_shift_edge=True) #separatrix in psi coords is just 1
                print('T SEP: ', Te_sep_eV)
                print('SHIFT: ', shift)

                te_interp_function = interp1d(new_x, te_fitted_best, fill_value='extrapolate')
                te_fitted_best = te_interp_function(generated_psi_grid)

                if ne_fitted_best is not None:
                    ne_interp_function = interp1d(new_x, ne_fitted_best, fill_value='extrapolate')
                    ne_fitted_best = ne_interp_function(generated_psi_grid)
            else:
                shift = 0



            # plotting option for debugging/checking fit quality
            if plot_the_fits == True:

                plt.errorbar(total_psi_te+shift, total_te, yerr=total_te_err, fmt = 'o',mfc='white', color='red', alpha=0.7) # raw data
                
                # option to plot the mtanh and cubic fits separately here
                #if te_fitted is not None:
                    #plt.plot(generated_psi_grid, te_fitted, label = rf'mtanh: $\chi^2$ = {te_chi_squared:.2f}', linewidth=2)
                #plt.plot(generated_psi_grid, te_fitted_cubic, label=rf'cubic: $\chi^2$ = {te_chi_squared_cubic:.2f}', linewidth=2)

                # just plot the best fit
                if te_fitted_best is not None:
                    plt.plot(generated_psi_grid, te_fitted_best, label = rf'best fit: $\chi^2$ = {te_chi_squared_best:.2f}')
                if return_error_bars_on_fits == True:
                    plt.fill_between(generated_psi_grid, te_fitted_best - list_of_te_fitted_std, te_fitted_best + list_of_te_fitted_std, color='red', alpha=0.3)
                plt.grid(linestyle='--', alpha=0.3)
                plt.xlabel(r'$\psi$')
                plt.ylabel('Te (eV)')
                if shift_to_2pt_model == True:
                    plt.axvline(1, color='black', linestyle='--', alpha=1, label = 'New Sep: ' + str(int(Te_sep_eV)) + 'eV')
                    plt.axvline(1 - shift, color='purple', linestyle='--', alpha=0.5, label = 'Old Sep')
                plt.legend()
                plt.title('Shot ' + str(shot) + ', Time ' + str(time) + ' ms, te fits')
                plt.show()

            # plotting option for debugging/checking fit quality
            if plot_the_fits == True:

                plt.errorbar(total_psi_ne+shift, total_ne, yerr=total_ne_err, fmt = 'o',mfc='white', color='green', alpha=0.7) # raw data
                
                # can plot the mtanh and cubic fits separately here
                #if ne_fitted is not None:
                     #plt.plot(generated_psi_grid, ne_fitted, label = rf'mtanh: $\chi^2$ = {ne_chi_squared:.2f}', linewidth=2)
                #plt.plot(generated_psi_grid, ne_fitted_cubic, label= rf'cubic: $\chi^2$ = {ne_chi_squared_cubic:.2f}', linewidth=2)

                # or just plot the best fit
                if ne_fitted_best is not None:
                    plt.plot(generated_psi_grid, ne_fitted_best, label = rf'best fit: $\chi^2$ = {ne_chi_squared_best:.2f}')
                if return_error_bars_on_fits == True:
                    plt.fill_between(generated_psi_grid, ne_fitted_best - list_of_ne_fitted_std, ne_fitted_best + list_of_ne_fitted_std, color='green', alpha=0.3)
                plt.grid(linestyle='--', alpha=0.3)
                plt.xlabel(r'$\psi$')
                plt.ylabel(r'$n_e$ ($m^{-3}$)')
                if shift_to_2pt_model == True:
                    plt.axvline(1, color='black', linestyle='--', alpha=1, label = 'New Sep: ' + str(int(Te_sep_eV)) + 'eV')
                    plt.axvline(1 - shift, color='purple', linestyle='--', alpha=0.5, label = 'Old Sep')
                plt.legend()
                plt.title('Shot ' + str(shot) + ', Time ' + str(time) + ' ms, ne fits')
                plt.show()


            if te_fitted_best is not None:
                list_of_successful_te_fit_times_ms.append(time)
                list_fitted_te_profiles.append(te_fitted_best)
                list_of_te_reduced_chi_squared.append(te_chi_squared_best)
                list_of_te_fit_type.append(te_best_fit_type)

                # also save the raw data used to perform these fits
                list_of_total_psi_te.append(total_psi_te + shift)
                list_of_total_te.append(total_te)
                list_of_total_te_err.append(total_te_err)


            if ne_fitted_best is not None:
                list_of_successful_ne_fit_times_ms.append(time)
                list_fitted_ne_profiles.append(ne_fitted_best)
                list_of_ne_reduced_chi_squared.append(ne_chi_squared_best)
                list_of_ne_fit_type.append(ne_best_fit_type)

                # also save the raw data used to perform these fits
                list_of_total_psi_ne.append(total_psi_ne + shift)
                list_of_total_ne.append(total_ne)
                list_of_total_ne_err.append(total_ne_err)

            if return_error_bars_on_fits == True:
                list_of_te_fit_errors.append(list_of_te_fitted_std)
                list_of_ne_fit_errors.append(list_of_ne_fitted_std)
    


    if verbose>0:
        print('Number of Time points')
        print(len(Thomson_times))

        print('Number of Te cubic fits')
        print(number_of_te_cubic_fits)
        print('Number of Te mtanh fits')
        print(number_of_te_mtanh_fits)
        print('Number of Te failed fits')
        print(number_of_te_failed_fits)

        print('Number of Ne cubic fits')
        print(number_of_ne_cubic_fits)
        print('Number of Ne mtanh fits')
        print(number_of_ne_mtanh_fits)
        print('Number of Ne failed fits')
        print(number_of_ne_failed_fits)

    # quantities_to_return = [
    #     generated_psi_grid,
    #     list_of_successful_te_fit_times_ms,
    #     list_fitted_te_profiles,
    #     list_of_te_reduced_chi_squared,
    #     list_of_te_fit_type,
    #     list_of_successful_ne_fit_times_ms,
    #     list_fitted_ne_profiles,
    #     list_of_ne_reduced_chi_squared,
    #     list_of_ne_fit_type
    # ]

    # if return_error_bars_on_fits == True:
    #     quantities_to_return.insert(3, list_of_te_fit_errors)
    #     quantities_to_return.insert(8, list_of_ne_fit_errors)

    # if return_processed_raw_data == True:
    # quantities_to_return.extend([
    #     list_of_total_psi_te,
    #     list_of_total_te,
    #     list_of_total_te_err,
    #     list_of_total_psi_ne,
    #     list_of_total_ne,
    #     list_of_total_ne_err
    # ])



    quantities_to_return = {
        'generated_psi_grid': generated_psi_grid,
        'te_fit_times_ms': list_of_successful_te_fit_times_ms,
        'te_fitted_profile': list_fitted_te_profiles,
        'te_reduced_chi_squared': list_of_te_reduced_chi_squared,
        'te_fit_type': list_of_te_fit_type,
        'ne_fit_times_ms': list_of_successful_ne_fit_times_ms,
        'ne_fitted_profiles': list_fitted_ne_profiles,
        'ne_reduced_chi_squared': list_of_ne_reduced_chi_squared,
        'ne_fit_type': list_of_ne_fit_type,
    }

    if return_error_bars_on_fits == True:
        quantities_to_return['te_fit_errors'] = list_of_te_fit_errors
        quantities_to_return['ne_fit_errors'] = list_of_ne_fit_errors

    if return_processed_raw_data == True:
        quantities_to_return['total_psi_te'] = list_of_total_psi_te
        quantities_to_return['total_te'] = list_of_total_te
        quantities_to_return['total_te_err'] = list_of_total_te_err
        quantities_to_return['total_psi_ne'] = list_of_total_psi_ne
        quantities_to_return['total_ne'] = list_of_total_ne
        quantities_to_return['total_ne_err'] = list_of_total_ne_err



    return quantities_to_return

### Time evolving fits for the Thomson data ###

import numpy as np
import sys, os
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
from functions.functions_fit_1D import *


def evolve_fits_by_radius_example_for_panel_plots(times, psi_grid, yvalues, output_time_grid = None):
    '''
    INPUTS
    --------
    times: 1D array of time points in ms
    psi_grid: 1D array of psi values
    yvalues: 2D array of Thomson data (either ne or Te).

    DESCRIPTION
    --------
    This function just takes in some fitted Thomson profiles, and fits a qudratic function through the time
    evolution of the profile at each radial location (e.g plot of Te at psi = 0.6 vs time).
    Doesn't return anything, just shows whether a quadratic fit is reasonable.
    '''

    #new_psi_grid = np.linspace(0.95, 1.045, 20)
    new_psi_grid_core = np.linspace(0.1, 0.9, 9)
    new_psi_grid_edge = np.linspace(0.95, 1.05, 11)
    new_psi_grid = np.append(new_psi_grid_core, new_psi_grid_edge)

    if output_time_grid is not None:
        high_res_time_grid = np.arange(output_time_grid[0], output_time_grid[-1], 1)
    else:
        high_res_time_grid = np.arange(times[0], times[-1], 1)

    interpolated_yvalues = []



    for idx in range(len(times)):
        yvalues_on_new_grid = interp1d(psi_grid, yvalues[idx])(new_psi_grid)
        #yvalues_on_new_grid = np.interp(new_psi_grid, psi_grid[idx], yvalues[idx])
        interpolated_yvalues.append(yvalues_on_new_grid)

    interpolated_yvalues = np.array(interpolated_yvalues)
    
    # Create figure to show the evolution of each radial location
    fig, axs = plt.subplots(5, 4, figsize=(20, 15))

    for idx, ax in enumerate(axs.flatten()):
        if idx < len(new_psi_grid):
            ax.scatter(times, interpolated_yvalues[:, idx], label=f'psi = {new_psi_grid[idx]:.2f}')
            ax.tick_params(axis='both', which='major', labelsize=6)
            ax.grid(True)
            ax.legend()

            #fit a quadratic to the data
            quadratic_fit = np.polyfit(times, interpolated_yvalues[:, idx], 2)
            ax.plot(high_res_time_grid, np.polyval(quadratic_fit, high_res_time_grid), color='r')

        else:
            ax.axis('off')  # Turn off axes if there are more subplots than new_psi_grid points

    
    
    plt.tight_layout()
    plt.show()





def master_fit_ne_Te_2D_window_smoothing(shot, t_min, t_max, smoothing_window = 15):
    '''
    INPUTS
    --------
    shot: integer, C-Mod shot number
    t_min: minimum time in ms
    t_max: maximum time in ms
    smoothing_window: integer, the standard deviation of the gaussian smoothing window in ms.

    OUTPUTS
    --------
    new_times_for_results: list of times in ms of the output fits (should be every ms if the fits were successful)
    generated_psi_grid: the output psi grid that the fits are placed on
    Rmid_grid: An alternative option to psi
    list_of_ne_fitted: list of ne fitted profiles on the output grid. First index is a profile at first time point.
    list_of_ne_fitted_error: corresponding error bars
    list_of_te_fitted: list of Te fitted profiles on the output grid. First index is a profile at first time point.
    list_of_te_fitted_error: corresponding error bars

    DESCRIPTION
    --------
    Takes in raw Thomson data straight from the Tree
    Fits ne and Te with a cubic inboard mtanh (+ adds in a SOL zero), includes a 2-pt model shift of the profiles
    Returns the fits on a high resolution grid in psi space.

    A Gaussian filter is used to smooth the raw data in time, so that smooth evolution of the profiles is achieved.
    Smoothing is obtained by increasing the error bars of raw data points away from the time of interest.
    The width of the smoothing window is the standard deviation of this gaussian.
    Gaussian function is given by f(x) = exp(-x^2 / (2 * sigma^2)):
    so Thomson data at the current time are weighted by x1, data 1 smoothing window away from current time point are weighted by
    exp(-1/2) = 0.6, etc.
    '''

    # Grab the raw data
    Thomson_times, ne_array, ne_err_array, te_array, te_err_array, rmid_array, r_array, z_array = get_raw_edge_Thomson_data(shot, t_min=t_min, t_max=t_max)
    Thomson_times_core, ne_array_core, ne_err_array_core, te_array_core, te_err_array_core, rmid_array_core, r_array_core, z_array_core = get_raw_core_Thomson_data(shot, t_min = t_min, t_max = t_max)

    if np.any(Thomson_times != Thomson_times_core):
        print('Thomson times are not the same for core and edge data. This is a problem.')
        raise ValueError('Thomson times are not the same for core and edge data. This is a problem.')


    e = eqtools.CModEFIT.CModEFITTree(int(shot), tree='EFIT20')

    # Scale the core Thomson data by the interferometry data.
    # NOTE: this is hardcoded to scale between 500ms and 1500ms for every shot.
    #ne_array_core = scale_core_Thomson(shot, Thomson_times_core, ne_array_core) # SCALE THE CORE THOMSON DATA BY THE INTERFEROMETRY DATA


    #cycle through all the time points and fit the Thomson data

    list_of_times = []
    list_of_raw_Te_xvalues_shifted = []
    list_of_raw_ne_xvalues_shifted = []
    list_of_raw_ne = []
    list_of_raw_ne_err = []
    list_of_raw_Te = []
    list_of_raw_Te_err = []


    list_of_initial_fit_times_for_checking_smoothing = []
    list_of_te_fitted_at_Thomson_times = [] #on the generated psi grid. Just for plotting purposes
    list_of_ne_fitted_at_Thomson_times = [] #on the generated psi grid. Just for plotting purposes

    list_of_te_fitted_std_at_Thomson_times = [] # for error bar calculations
    list_of_ne_fitted_std_at_Thomson_times = [] # for error bar calculations

    # Often a very good initial guess is actually the fitted mtanh parameters from the last time point.
    # I'm initialising the locations for these variables here.
    ne_params_from_last_successful_fit = None
    te_params_from_last_successful_fit = None



    for t_idx in range(len(Thomson_times)):
            time = Thomson_times[t_idx] #in ms
            time_in_s = time / 1000 #in s
            raw_ne = ne_array[:,t_idx]
            raw_ne_err = ne_err_array[:,t_idx]
            raw_te = te_array[:, t_idx]
            raw_te_err = te_err_array[:, t_idx]
            raw_rmid = rmid_array[:, t_idx]

            raw_ne_core = ne_array_core[:,t_idx]
            raw_ne_err_core = ne_err_array_core[:,t_idx]
            raw_te_core = te_array_core[:, t_idx]
            raw_te_err_core = te_err_array_core[:, t_idx]
            raw_rmid_core = rmid_array_core[:, t_idx]


            total_rmid = np.append(raw_rmid, raw_rmid_core)
            total_te = np.append(raw_te, raw_te_core)
            total_te_err = np.append(raw_te_err, raw_te_err_core)
            total_ne = np.append(raw_ne, raw_ne_core)
            total_ne_err = np.append(raw_ne_err, raw_ne_err_core)


            print('TIME: ', time)




            # NOTE: THIS CODE HERE MAY BE DEPRECATED COMPARED TO MY MORE RECENT PRE-PROCESSING IN THE MASTER 1D FITS.
            te_radii, te, te_err = remove_zeros(raw_rmid, raw_te, raw_te_err, core_only=True)
            ne_radii, ne, ne_err = remove_zeros(raw_rmid, raw_ne, raw_ne_err, core_only=True)

            #Switch from Rmid to psi coordinates here using eqtools
            te_radii = e.rho2rho('Rmid', 'psinorm', te_radii, time_in_s)
            ne_radii = e.rho2rho('Rmid', 'psinorm', ne_radii, time_in_s)

            # Add in a zero in the SOL at a pre-specified psi value. The location of these zeros will get shifted about from psi=1.05 once I apply the 2-pt model.
            te_radii, te, te_err = add_SOL_zeros_in_psi_coords(te_radii, te, te_err)  #zeros added at psi = 1.05 (hardcoded)
            ne_radii, ne, ne_err = add_SOL_zeros_in_psi_coords(ne_radii, ne, ne_err)  #zeros added at psi = 1.05 (hardcoded)

            te_radii_core = e.rho2rho('Rmid', 'psinorm', raw_rmid_core, time_in_s)
            ne_radii_core = e.rho2rho('Rmid', 'psinorm', raw_rmid_core, time_in_s)


            Te_sep, twopt_shift_psi = get_twopt_shift_from_edge_Te_fit(shot, time_in_s, te_radii, te, te_err)

            #twopt_shift_psi = 0

            te_radii = te_radii + twopt_shift_psi
            ne_radii = ne_radii + twopt_shift_psi

            print('TWO PT SHIFT: ', twopt_shift_psi)

            # save the original arrays before adding the zer
            total_psi_te = np.append(te_radii_core, te_radii)
            total_psi_ne = np.append(ne_radii_core, ne_radii)
            total_ne = np.append(raw_ne_core, ne)
            total_ne_err = np.append(raw_ne_err_core, ne_err)
            total_te = np.append(raw_te_core, te)
            total_te_err = np.append(raw_te_err_core, te_err)


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



            #fitting
            te_guesses = Osborne_linear_initial_guesses(te_radii, te)
            ne_guesses = Osborne_linear_initial_guesses(ne_radii, ne)
            ne_guesses[2] = ne_guesses[2] / 1e20 #just divide height and base by 1e20 to make the minimisation easier
            ne_guesses[3] = ne_guesses[3] / 1e20

            te_guesses.extend([0,0]) #just for the quadratic and cubic terms
            ne_guesses.extend([0,0]) #just for the quadratic and cubic terms

            generated_psi_grid_core = np.arange(0, 0.8, 0.01) #I've boosted this for better combination with the Ly-alpha data. Less sensitive to the end-points.
            generated_psi_grid_edge = np.arange(0.8, 1.2001, 0.002) #higher resolution at the edge
            generated_psi_grid = np.append(generated_psi_grid_core, generated_psi_grid_edge)

            #print(te_guesses)
            #print(len(te_guesses))

            # some initial te guesses
            list_of_te_guesses = []
            list_of_te_guesses.append(te_guesses)
            list_of_te_guesses.append([ 9.92614859e-01,  4.01791101e-02,  2.55550908e+02,  1.28542623e+01,  2.17777084e-01, -3.45196862e-03,  1.42947373e-04])
            if te_params_from_last_successful_fit is not None:
                list_of_te_guesses.insert(0, te_params_from_last_successful_fit)


            for te_guess_idx in range(len(list_of_te_guesses)):
                te_guess = list_of_te_guesses[te_guess_idx]
                
                try:
                    print('IN THE TE LOOP')
                    print('te guess')
                    print(te_guess)
                    #print('try te')
                    #print(te_guess)
                    te_params, te_covariance = curve_fit(Osborne_Tanh_cubic, total_psi_te, total_te, p0=te_guess, sigma=total_te_err, absolute_sigma=True, maxfev=2000, bounds=([0.85, 0, 0, -0.001, -np.inf, -np.inf, -np.inf], np.inf)) #should now be in psi
                    te_fitted = Osborne_Tanh_cubic(generated_psi_grid, te_params[0], te_params[1], te_params[2], te_params[3], te_params[4], te_params[5], te_params[6])
                    #print('te WORKED')
                    #print(te_params)
                    #plt.show()
                    print('OUT THE TE LOOP')
                    te_params_from_last_successful_fit = te_params
                    break #guess worked so exit the for loop
                except:
                    if te_guess_idx == len(list_of_te_guesses) - 1:
                        # If all the guesses failed, set the fit parameters to none
                        te_params = None
                        te_covariance = None
                        te_fitted = None
                    else:
                        # move onto the next guess
                        continue

            print('te params')
            print(te_params)


            # some initial ne guesses
            list_of_ne_guesses = []
            list_of_ne_guesses.append(ne_guesses) #just from my own rough method
            list_of_ne_guesses.append([1.00604712, 0.037400836, 2.10662412, 0.0168897974, -0.0632778417, 0.00229233952, -2.0627212e-05]) #good initial guess for 650kA shots
            list_of_ne_guesses.append([1.02123755e+00,  5.02744526e-02,  2.54219267e+00, -9.99999694e-04, 2.58724602e-02, -2.32961078e-03,  4.20279037e-05]) #good guess for 1MA shots
            if ne_params_from_last_successful_fit is not None:
                list_of_ne_guesses.insert(0, ne_params_from_last_successful_fit)


            for ne_guess_idx in range(len(list_of_ne_guesses)):
                ne_guess = list_of_ne_guesses[ne_guess_idx]
                try:
                    print('try ne')
                    print(ne_guess)
                    ne_params, ne_covariance = curve_fit(Osborne_Tanh_cubic, total_psi_ne, total_ne/1e20, p0=ne_guess, sigma=total_ne_err/1e20, absolute_sigma=True, maxfev=2000, bounds=([0.85, 0.001, 0, -0.001, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])) #should now be in psi
                    ne_fitted = 1e20*Osborne_Tanh_cubic(generated_psi_grid, ne_params[0], ne_params[1], ne_params[2], ne_params[3], ne_params[4], ne_params[5], ne_params[6])
                    #print(ne_params)
                    #print('ne WORKED')
                    print('out the ne loop')
                    ne_params_from_last_successful_fit = ne_params
                    break #guess worked so exit the for loop
                except:
                    if ne_guess_idx == len(list_of_ne_guesses) - 1:
                        # If all the guesses failed, set the fit parameters to none
                        ne_params = None
                        ne_covariance = None
                        ne_fitted = None
                    else:
                        # move onto the next guess
                        continue
            print('ne params')
            print(ne_params)
            
                    
            '''
            try:
                print('try 1')
                plt.plot(generated_psi_grid, 1e20*Osborne_Tanh_cubic(generated_psi_grid, ne_guesses[0], ne_guesses[1], ne_guesses[2], ne_guesses[3], ne_guesses[4], ne_guesses[5], ne_guesses[6]))
                plt.errorbar(total_psi_ne, total_ne, yerr=total_ne_err, fmt = 'o', color='green')
                plt.show()
                ne_params, ne_covariance = curve_fit(Osborne_Tanh_cubic, total_psi_ne, total_ne/1e20, p0=ne_guesses, sigma=total_ne_err/1e20, absolute_sigma=True, maxfev=2000, bounds=([0.85, 0.001, 0, -0.001, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])) #should now be in psi
                ne_fitted = 1e20*Osborne_Tanh_cubic(generated_psi_grid, ne_params[0], ne_params[1], ne_params[2], ne_params[3], ne_params[4], ne_params[5], ne_params[6])
            except:
                print('try 2')
                ne_params, ne_covariance = curve_fit(Osborne_Tanh_cubic, total_psi_ne, total_ne/1e20, p0=ne_params, sigma=total_ne_err/1e20, absolute_sigma=True, maxfev=2000, bounds=([0.85, 0.001, 0, -0.001, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]))
                ne_fitted = 1e20*Osborne_Tanh_cubic(generated_psi_grid, ne_params[0], ne_params[1], ne_params[2], ne_params[3], ne_params[4], ne_params[5], ne_params[6])
            '''



            

            # Te outlier rejection
            te_fitted_for_outlier_rejection = Osborne_Tanh_cubic(total_psi_te, te_params[0], te_params[1], te_params[2], te_params[3], te_params[4], te_params[5], te_params[6])
            total_te_residuals = np.abs(total_te - te_fitted_for_outlier_rejection)
            te_outliers_mask = total_te_residuals < 3*total_te_err #reject any points that are more than 3 sigma away from the fit

            # ne outlier rejection
            ne_fitted_for_outlier_rejection = 1e20*Osborne_Tanh_cubic(total_psi_ne, ne_params[0], ne_params[1], ne_params[2], ne_params[3], ne_params[4], ne_params[5], ne_params[6])
            total_ne_residuals = np.abs(total_ne - ne_fitted_for_outlier_rejection)
            ne_outliers_mask = total_ne_residuals < 3*total_ne_err #reject any points that are more than 3 sigma away from the fit

            # this just makes sure that the ne and Te arrays are the same length, which is easier for data processing
            all_outliers_mask = np.logical_and(te_outliers_mask, ne_outliers_mask)


            total_psi_te = total_psi_te[all_outliers_mask]
            total_te = total_te[all_outliers_mask]
            total_te_err = total_te_err[all_outliers_mask]

            total_psi_ne = total_psi_ne[all_outliers_mask]
            total_ne = total_ne[all_outliers_mask]
            total_ne_err = total_ne_err[all_outliers_mask]





            for idx in range(len(total_psi_te)):
                list_of_times.append(time - t_min) # save the time relative to the LH transition
                list_of_raw_ne_xvalues_shifted.append(total_psi_ne[idx])
                list_of_raw_ne.append(total_ne[idx])
                list_of_raw_ne_err.append(total_ne_err[idx])
                list_of_raw_Te_xvalues_shifted.append(total_psi_te[idx])
                list_of_raw_Te.append(total_te[idx])
                list_of_raw_Te_err.append(total_te_err[idx])

            
            print('te params')
            print(te_params)

            # just refit quickly with the outlier removal. This is a good check on the smoothing function
            te_params, te_covariance = curve_fit(Osborne_Tanh_cubic, total_psi_te, total_te, p0=te_params, sigma=total_te_err, absolute_sigma=True, maxfev=2000, bounds=([0.85, 0, 0, -0.001, -np.inf, -np.inf, -np.inf], np.inf)) #should now be in psi
            te_fitted_on_generated_psi_grid = Osborne_Tanh_cubic(generated_psi_grid, te_params[0], te_params[1], te_params[2], te_params[3], te_params[4], te_params[5], te_params[6])

            ne_params, ne_covariance = curve_fit(Osborne_Tanh_cubic, total_psi_ne, total_ne/1e20, p0=ne_params, sigma=total_ne_err/1e20, absolute_sigma=True, maxfev=2000, bounds=([0.85, 0.001, 0, -0.001, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])) #should now be in psi
            ne_fitted_on_generated_psi_grid = 1e20*Osborne_Tanh_cubic(generated_psi_grid, ne_params[0], ne_params[1], ne_params[2], ne_params[3], ne_params[4], ne_params[5], ne_params[6])

            list_of_initial_fit_times_for_checking_smoothing.append(time-t_min)
            list_of_te_fitted_at_Thomson_times.append(te_fitted_on_generated_psi_grid)
            list_of_ne_fitted_at_Thomson_times.append(ne_fitted_on_generated_psi_grid)

            print('STARTING MONTE CARLO METHOD')



            # use a monte carlo method to get the error bars at every time point, and save these
            # for the moment, I will simply take the error at each radial location as the AVERAGE error
            # over all the time points. This is a bit of a simplification but it's a start.

            list_of_perturbed_te_fits = []
            for idx in range(100): # just go with 50 not 100 for the moment because it's a bit quicker
                perturbed_te_values = np.random.normal(loc = total_te, scale = total_te_err) #perturb the data to see how the fit changes
                try:
                    perturbed_te_params, te_covariance = curve_fit(Osborne_Tanh_cubic, total_psi_te, perturbed_te_values, p0=te_params, sigma=total_te_err, absolute_sigma=True, maxfev=2000, bounds=([0.85, 0, 0, -0.001, -np.inf, -np.inf, -np.inf], np.inf)) #should now be in psi
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



            # get the standard deviation at every radial location for the fits
            list_of_te_fitted_std = np.nanstd(list_of_perturbed_te_fits, axis=0) #np.nanstd is just like np.std but ignores nans




            list_of_perturbed_ne_fits = []
            for idx in range(100):
                perturbed_ne_values = np.random.normal(loc = total_ne, scale = total_ne_err)
                try:
                    perturbed_ne_params, ne_covariance = curve_fit(Osborne_Tanh_cubic, total_psi_ne, perturbed_ne_values/1e20, p0=ne_params, sigma=total_ne_err/1e20, absolute_sigma=True, maxfev=2000, bounds=([0.85, 0.001, 0, -0.001, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])) #should now be in psi
                    perturbed_ne_fitted = 1e20*Osborne_Tanh_cubic(generated_psi_grid, perturbed_ne_params[0], perturbed_ne_params[1], perturbed_ne_params[2], perturbed_ne_params[3], perturbed_ne_params[4], perturbed_ne_params[5], perturbed_ne_params[6])
                    
                    perturbed_ne_fitted[np.isinf(perturbed_ne_fitted)] = np.nan
                    list_of_perturbed_ne_fits.append(perturbed_ne_fitted)
                except:
                    print('NE IDX: ', idx, ' could not fit')
                    pass


            # get the standard deviation at every radial location for the fits
            list_of_ne_fitted_std = np.nanstd(list_of_perturbed_ne_fits, axis=0) # np.nanstd is just like np.std but ignores nans

            list_of_te_fitted_std_at_Thomson_times.append(list_of_te_fitted_std)
            list_of_ne_fitted_std_at_Thomson_times.append(list_of_ne_fitted_std)

            #plt.errorbar(list_of_raw_ne_xvalues_shifted, list_of_raw_ne, yerr=list_of_raw_ne_err, fmt = 'o', color='green')
            #plt.plot(generated_psi_grid, ne_fitted_on_generated_psi_grid)
            #plt.axvline(x=1, color='black', linestyle='--')
            #plt.title('Time = ' + str(time) + 'ms')
            #plt.show()



        
    print('OUT OF LOOP')
    plt.scatter(list_of_raw_ne_xvalues_shifted, list_of_raw_ne, marker='x')
    plt.errorbar(list_of_raw_ne_xvalues_shifted, list_of_raw_ne, yerr=list_of_raw_ne_err, fmt = 'o', color='green')
    plt.show()

    plt.scatter(list_of_raw_Te_xvalues_shifted, list_of_raw_Te, marker='x')
    plt.errorbar(list_of_raw_Te_xvalues_shifted, list_of_raw_Te, yerr=list_of_raw_Te_err, fmt = 'o', color='red')
    plt.show()






    # get an average errorbar
    average_te_error_band = np.mean(list_of_te_fitted_std_at_Thomson_times, axis=0) #just take an average for the standard deviation
    average_ne_error_band = np.mean(list_of_ne_fitted_std_at_Thomson_times, axis=0) #just take an average for the standard deviation





    output_time_grid = np.arange(0, t_max-t_min, 1)

    list_of_times = np.array(list_of_times)
    list_of_raw_ne_xvalues_shifted = np.array(list_of_raw_ne_xvalues_shifted)
    list_of_raw_ne = np.array(list_of_raw_ne)
    list_of_raw_ne_err = np.array(list_of_raw_ne_err)
    list_of_raw_Te_xvalues_shifted = np.array(list_of_raw_Te_xvalues_shifted)
    list_of_raw_Te = np.array(list_of_raw_Te)
    list_of_raw_Te_err = np.array(list_of_raw_Te_err)


    list_of_ne_fitted = []
    list_of_te_fitted = []

    list_of_ne_fitted_error = []
    list_of_te_fitted_error = []

    successful_te_fit_mask = []
    successful_ne_fit_mask = []


    # now do the window smoothing
    for t_idx in range(len(output_time_grid)):
        time = output_time_grid[t_idx]

        print('TIME: ', time)



        #apply the Gaussian filter by making the error bars larger
        weights = 1 / np.sqrt(np.exp((-1/2) * (list_of_times - time)**2 / (smoothing_window**2)))


        list_of_raw_ne_err_weights_applied = weights*list_of_raw_ne_err
        list_of_raw_Te_err_weights_applied = weights*list_of_raw_Te_err


        #fitting
        te_guesses = Osborne_linear_initial_guesses(te_radii, te)
        ne_guesses = Osborne_linear_initial_guesses(ne_radii, ne)
        ne_guesses[2] = ne_guesses[2] / 1e20 #just divide height and base by 1e20 to make the minimisation easier
        ne_guesses[3] = ne_guesses[3] / 1e20

        te_guesses.extend([0,0]) #just for the quadratic and cubic terms
        ne_guesses.extend([0,0]) #just for the quadratic and cubic terms

        generated_psi_grid_core = np.arange(0, 0.8, 0.01) #I've boosted this for better combination with the Ly-alpha data. Less sensitive to the end-points.
        generated_psi_grid_edge = np.arange(0.8, 1.2001, 0.002) #higher resolution at the edge
        generated_psi_grid = np.append(generated_psi_grid_core, generated_psi_grid_edge)



        # some initial te guesses
        list_of_te_guesses = []
        list_of_te_guesses.append(te_guesses)
        list_of_te_guesses.append([ 9.92614859e-01,  4.01791101e-02,  2.55550908e+02,  1.28542623e+01,  2.17777084e-01, -3.45196862e-03,  1.42947373e-04])
        if te_params_from_last_successful_fit is not None:
            list_of_te_guesses.insert(0, te_params_from_last_successful_fit) #use the parameters from the last successful fit as a first guess

        for te_guess_idx in range(len(list_of_te_guesses)):
            te_guess = list_of_te_guesses[te_guess_idx]
            try:
                te_params, te_covariance = curve_fit(Osborne_Tanh_cubic, list_of_raw_Te_xvalues_shifted, list_of_raw_Te, p0=te_guess, sigma=list_of_raw_Te_err_weights_applied, absolute_sigma=False, maxfev=2000, bounds=([0.85, 0, 0, -0.001, -np.inf, -np.inf, -np.inf], np.inf)) #should now be in psi
                te_fitted = Osborne_Tanh_cubic(generated_psi_grid, te_params[0], te_params[1], te_params[2], te_params[3], te_params[4], te_params[5], te_params[6])

                #plt.plot(generated_psi_grid, te_fitted)
                #plt.scatter(list_of_raw_ne_xvalues_shifted, list_of_raw_Te, marker='x')
                #plt.show()
                list_of_te_fitted.append(te_fitted)
                list_of_te_fitted_error.append(average_te_error_band)
                successful_te_fit_mask.append(True)
                te_params_from_last_successful_fit = te_params # to be used as a first guess for the next time point
                break #guess worked so exit the for loop
            except:
                if te_guess_idx == len(list_of_te_guesses) - 1:
                    # If all the guesses failed, set the fit parameters to none
                    list_of_te_fitted.append(np.full(len(generated_psi_grid), np.nan))
                    list_of_te_fitted_error.append(np.full(len(generated_psi_grid), np.nan))

                    te_params = None
                    te_covariance = None
                    te_fitted = None
                    successful_te_fit_mask.append(False)
                    print('TE FIT FAILED at time: ', time)
                else:
                    # move onto the next guess
                    continue



        # some initial ne guesses
        list_of_ne_guesses = []
        list_of_ne_guesses.append(ne_guesses) #just from my own rough method
        list_of_ne_guesses.append([1.00604712, 0.037400836, 2.10662412, 0.0168897974, -0.0632778417, 0.00229233952, -2.0627212e-05]) #good initial guess for 650kA shots
        list_of_ne_guesses.append([1.02123755e+00,  5.02744526e-02,  2.54219267e+00, -9.99999694e-04, 2.58724602e-02, -2.32961078e-03,  4.20279037e-05]) #good guess for 1MA shots
        if ne_params_from_last_successful_fit is not None:
            list_of_ne_guesses.insert(0, ne_params_from_last_successful_fit) #use the parameters from the last successful fit as a first guess


        for ne_guess_idx in range(len(list_of_ne_guesses)):
            ne_guess = list_of_ne_guesses[ne_guess_idx]
            try:
                ne_params, ne_covariance = curve_fit(Osborne_Tanh_cubic, list_of_raw_ne_xvalues_shifted, list_of_raw_ne/1e20, p0=ne_guess, sigma=list_of_raw_ne_err_weights_applied/1e20, absolute_sigma=False, maxfev=2000, bounds=([0.85, 0.001, 0, -0.001, -np.inf, -np.inf, -np.inf], [1.05, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])) #should now be in psi
                ne_fitted = 1e20*Osborne_Tanh_cubic(generated_psi_grid, ne_params[0], ne_params[1], ne_params[2], ne_params[3], ne_params[4], ne_params[5], ne_params[6])
                #plt.plot(generated_psi_grid, ne_fitted)
                #plt.scatter(list_of_raw_ne_xvalues_shifted, list_of_raw_ne, marker='x')


                list_of_ne_fitted.append(ne_fitted)
                list_of_ne_fitted_error.append(average_ne_error_band)
                successful_ne_fit_mask.append(True)
                ne_params_from_last_successful_fit = ne_params # to be used as a first guess for the next time point

                break #guess worked so exit the for loop
            except:
                if ne_guess_idx == len(list_of_ne_guesses) - 1:
                    # If all the guesses failed, set the fit parameters to none
                    list_of_ne_fitted.append(np.full(len(generated_psi_grid), np.nan))
                    list_of_ne_fitted_error.append(np.full(len(generated_psi_grid), np.nan))

                    ne_params = None
                    ne_covariance = None
                    ne_fitted = None
                    successful_ne_fit_mask.append(False)
                    print('NE FIT FAILED at time: ', time)

                    '''

                    xvalues_increasing_order = np.argsort(list_of_raw_ne_xvalues_shifted)

                    list_of_raw_ne_xvalues_shifted = list_of_raw_ne_xvalues_shifted[xvalues_increasing_order]
                    list_of_raw_ne = list_of_raw_ne[xvalues_increasing_order]
                    list_of_raw_ne_err_weights_applied = list_of_raw_ne_err_weights_applied[xvalues_increasing_order]


                    print('list_of_raw_ne_xvalues_shifted')
                    print(list_of_raw_ne_xvalues_shifted)

                    print('list_of_raw_ne')
                    print(list_of_raw_ne)

                    print('list_of_raw_ne_err_weights_applied')
                    print(list_of_raw_ne_err_weights_applied)

                    plt.errorbar(list_of_raw_ne_xvalues_shifted, list_of_raw_ne, yerr=list_of_raw_ne_err_weights_applied, fmt = 'o', color='green')
                    plt.plot(generated_psi_grid, 1e20*Osborne_Tanh_cubic(generated_psi_grid, ne_guess[0], ne_guess[1], ne_guess[2], ne_guess[3], ne_guess[4], ne_guess[5], ne_guess[6]))
                    plt.show()
                    '''




                else:
                    # move onto the next guess
                    continue


    list_of_ne_fitted = np.array(list_of_ne_fitted)
    list_of_te_fitted = np.array(list_of_te_fitted)
    list_of_ne_fitted_error = np.array(list_of_ne_fitted_error)
    list_of_te_fitted_error = np.array(list_of_te_fitted_error)
    successful_te_fit_mask = np.array(successful_te_fit_mask)
    successful_ne_fit_mask = np.array(successful_ne_fit_mask)



    combined_successful_fit_mask = np.logical_and(successful_te_fit_mask, successful_ne_fit_mask) # make sure the fit succeeded for both ne and Te
    output_time_grid = output_time_grid[combined_successful_fit_mask]

    list_of_ne_fitted = list_of_ne_fitted[combined_successful_fit_mask]
    list_of_te_fitted = list_of_te_fitted[combined_successful_fit_mask]
    list_of_ne_fitted_error = list_of_ne_fitted_error[combined_successful_fit_mask]
    list_of_te_fitted_error = list_of_te_fitted_error[combined_successful_fit_mask]

    

    list_of_ne_fitted_at_Thomson_times = np.array(list_of_ne_fitted_at_Thomson_times)
    list_of_te_fitted_at_Thomson_times = np.array(list_of_te_fitted_at_Thomson_times)
    list_of_initial_fit_times_for_checking_smoothing = np.array(list_of_initial_fit_times_for_checking_smoothing)

    # if either fit failed at any time-point, just remove this time-point from the list

    

    for idx in range(len(output_time_grid)):
        if idx % 20 == 0:
            plt.plot(generated_psi_grid, list_of_ne_fitted[idx], label = output_time_grid[idx])
            plt.fill_between(generated_psi_grid, list_of_ne_fitted[idx] - list_of_ne_fitted_error[idx], list_of_ne_fitted[idx] + list_of_ne_fitted_error[idx], alpha=0.5)
            plt.legend()
    plt.show()
    for idx in range(len(output_time_grid)):
        if idx % 20 == 0:
            plt.plot(generated_psi_grid, list_of_te_fitted[idx], label = output_time_grid[idx])
            plt.fill_between(generated_psi_grid, list_of_te_fitted[idx] - list_of_te_fitted_error[idx], list_of_te_fitted[idx] + list_of_te_fitted_error[idx], alpha=0.5)
            plt.legend()
    plt.show()





    radii_to_plot = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05]
    list_of_generated_psi_grid_indices = [10, 20, 30, 40, 50, 60, 70, 80, 130, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205]
    # Create figure to show the evolution of each radial location
    fig, axs = plt.subplots(5, 4, figsize=(20, 15))

    for idx, ax in enumerate(axs.flatten()):
        print('idx')
        print(idx)
        #cycle through every psi value and plot its evolution in time.
        psi_value_to_evolve = radii_to_plot[idx]
        psi_idx = list_of_generated_psi_grid_indices[idx]
        print('psi_idx')
        print(psi_idx)
        ax.scatter(output_time_grid, list_of_ne_fitted[:, psi_idx], label=f'psi = {psi_value_to_evolve:.2f}', marker='o')
        ax.scatter(list_of_initial_fit_times_for_checking_smoothing, list_of_ne_fitted_at_Thomson_times[:, psi_idx], marker='x', color='red')
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    

    fig, axs = plt.subplots(5, 4, figsize=(20, 15))

    for idx, ax in enumerate(axs.flatten()):
        #cycle through every psi value and plot its evolution in time.
        psi_value_to_evolve = radii_to_plot[idx]
        psi_idx = list_of_generated_psi_grid_indices[idx]

        ax.plot(output_time_grid, list_of_te_fitted[:, psi_idx], label=f'psi = {psi_value_to_evolve:.2f}')
        ax.scatter(list_of_initial_fit_times_for_checking_smoothing, list_of_te_fitted_at_Thomson_times[:, psi_idx], marker='x', color='red')
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.show()

    Rmid_grid = psi_to_Rmid_map(shot, t_min, t_max, generated_psi_grid, output_time_grid) #this is a 2D array of Rmid values at every psi value at every time point




    return output_time_grid, generated_psi_grid, Rmid_grid, list_of_ne_fitted, list_of_ne_fitted_error, list_of_te_fitted, list_of_te_fitted_error



def master_fit_2D_alt(shot, t_min, t_max, smoothing_window=15):
    '''
    Exactly the same as the window_smoothing 2D fitting function, except that the 1D fitting
    function (master_fit_ne_Te_1D) is used to do the fits at every time point.
    This gives a bit more flexibility (since it also tries a cubic fit), but currently
    does not have a post-fitting outlier rejection method.

    TODO:
    Implement some option for post-fit outlier rejection
    Let this function also use a cubic to fit if it wants.
    '''

    # get the ne and Te fits at each time point from the 1D fitting function
    generated_psi_grid, list_of_Thomson_times_te_ms, list_of_te_fitted_at_Thomson_times, list_of_te_fitted_std_at_Thomson_times, list_of_te_reduced_chi_squared, \
    list_of_te_fit_type, list_of_Thomson_times_ne_ms, list_of_ne_fitted_at_Thomson_times, list_of_ne_fitted_std_at_Thomson_times, list_of_ne_reduced_chi_squared, \
    list_of_ne_fit_type, list_of_total_psi_te, list_of_total_te, list_of_total_te_err, \
    list_of_total_psi_ne, list_of_total_ne, list_of_total_ne_err = master_fit_ne_Te_1D(shot, t_min, t_max, plot_the_fits=False, remove_zeros_before_fitting=True, shift_to_2pt_model=True, return_processed_raw_data=True, return_error_bars_on_fits=True, scale_core_TS_to_TCI=True, enforce_mtanh=True)

    list_of_te_successful_fit_times_flattened = []
    list_of_total_psi_te_flattened = []
    list_of_total_te_flattened = []
    list_of_total_te_err_flattened = []

    list_of_ne_successful_fit_times_flattened = []
    list_of_total_psi_ne_flattened = []
    list_of_total_ne_flattened = []
    list_of_total_ne_err_flattened = []



    # CONVERT THE 2D ARRAYS INTO 1D ARRAYS SO THAT WEIGHTS CAN BE APPLIED AND THE SMOOTHED FITS CAN BE APPLIED
    for idx in range(len(list_of_Thomson_times_te_ms)):
        no_of_points = len(list_of_total_psi_te[idx])

        list_of_te_successful_fit_times_flattened.extend(list_of_Thomson_times_te_ms[idx]*np.ones(no_of_points))
        list_of_total_psi_te_flattened.extend(list_of_total_psi_te[idx])
        list_of_total_te_flattened.extend(list_of_total_te[idx])
        list_of_total_te_err_flattened.extend(list_of_total_te_err[idx])


    for idx in range(len(list_of_Thomson_times_ne_ms)):
        no_of_points = len(list_of_total_psi_ne[idx])
        list_of_ne_successful_fit_times_flattened.extend(list_of_Thomson_times_ne_ms[idx]*np.ones(no_of_points))
        list_of_total_psi_ne_flattened.extend(list_of_total_psi_ne[idx])
        list_of_total_ne_flattened.extend(list_of_total_ne[idx])
        list_of_total_ne_err_flattened.extend(list_of_total_ne_err[idx])



    # Rename and normalise to the minimum time chosen
    list_of_raw_ne_times = np.array(list_of_ne_successful_fit_times_flattened) - t_min # use the time relative to the start of the window
    list_of_raw_ne_xvalues_shifted = np.array(list_of_total_psi_ne_flattened)
    list_of_raw_ne = np.array(list_of_total_ne_flattened)
    list_of_raw_ne_err = np.array(list_of_total_ne_err_flattened)

    list_of_raw_Te_times = np.array(list_of_te_successful_fit_times_flattened) - t_min # use the time relative to the start of the window
    list_of_raw_Te_xvalues_shifted = np.array(list_of_total_psi_te_flattened)
    list_of_raw_Te = np.array(list_of_total_te_flattened)
    list_of_raw_Te_err = np.array(list_of_total_te_err_flattened)



    # get an average errorbar
    average_te_error_band = np.mean(list_of_te_fitted_std_at_Thomson_times, axis=0) #just take an average for the standard deviation
    average_ne_error_band = np.mean(list_of_ne_fitted_std_at_Thomson_times, axis=0) #just take an average for the standard deviation


    new_times_for_results = np.arange(0, t_max-t_min, 1) # return fits on 1ms timebase


    te_params_from_last_successful_fit = None
    ne_params_from_last_successful_fit = None


    # Lists to store the smoothed fits
    list_of_ne_fitted = []
    list_of_te_fitted = []
    list_of_ne_fitted_error = []
    list_of_te_fitted_error = []
    successful_te_fit_mask = []
    successful_ne_fit_mask = []
    list_of_ne_params_that_worked = []
    list_of_indices_that_worked = []

    # now do the window smoothing
    for t_idx in range(len(new_times_for_results)):
        time = new_times_for_results[t_idx]

        print('TIME: ', time)

        #apply the Gaussian filter by making the error bars larger for further away points
        ne_weights = 1 / np.sqrt(np.exp((-1/2) * (list_of_raw_ne_times - time)**2 / (smoothing_window**2)))
        list_of_raw_ne_err_weights_applied = ne_weights*list_of_raw_ne_err

        Te_weights = 1 / np.sqrt(np.exp((-1/2) * (list_of_raw_Te_times - time)**2 / (smoothing_window**2)))
        list_of_raw_Te_err_weights_applied = Te_weights*list_of_raw_Te_err


        # FITTING

        # some initial te guesses
        list_of_te_guesses = []
        list_of_te_guesses.append([ 9.92614859e-01,  4.01791101e-02,  2.55550908e+02,  1.28542623e+01,  2.17777084e-01, -3.45196862e-03,  1.42947373e-04])
        if te_params_from_last_successful_fit is not None:
            list_of_te_guesses.insert(0, te_params_from_last_successful_fit) #use the parameters from the last successful fit as a first guess

        for te_guess_idx in range(len(list_of_te_guesses)):
            te_guess = list_of_te_guesses[te_guess_idx]
            try:
                te_params, te_covariance = curve_fit(Osborne_Tanh_cubic, list_of_raw_Te_xvalues_shifted, list_of_raw_Te, p0=te_guess, sigma=list_of_raw_Te_err_weights_applied, absolute_sigma=False, maxfev=2000, bounds=([0.85, 0, 0, -0.001, -np.inf, -np.inf, -np.inf], np.inf)) #should now be in psi
                te_fitted = Osborne_Tanh_cubic(generated_psi_grid, te_params[0], te_params[1], te_params[2], te_params[3], te_params[4], te_params[5], te_params[6])

                #plt.plot(generated_psi_grid, te_fitted)
                #plt.scatter(list_of_raw_ne_xvalues_shifted, list_of_raw_Te, marker='x')
                #plt.show()
                list_of_te_fitted.append(te_fitted)
                list_of_te_fitted_error.append(average_te_error_band)
                successful_te_fit_mask.append(True)
                te_params_from_last_successful_fit = te_params # to be used as a first guess for the next time point
                break #guess worked so exit the for loop
            except:
                if te_guess_idx == len(list_of_te_guesses) - 1:
                    # If all the guesses failed, set the fit parameters to none
                    list_of_te_fitted.append(np.full(len(generated_psi_grid), np.nan))
                    list_of_te_fitted_error.append(np.full(len(generated_psi_grid), np.nan))

                    te_params = None
                    te_covariance = None
                    te_fitted = None
                    successful_te_fit_mask.append(False)
                    print('TE FIT FAILED')
                else:
                    # move onto the next guess
                    continue



        # some initial ne guesses
        list_of_ne_guesses = []
        list_of_ne_guesses.append([1.00604712, 0.037400836, 2.10662412, 0.0168897974, -0.0632778417, 0.00229233952, -2.0627212e-05]) #good initial guess for 650kA shots
        list_of_ne_guesses.append([1.02123755e+00,  5.02744526e-02,  2.54219267e+00, -9.99999694e-04, 2.58724602e-02, -2.32961078e-03,  4.20279037e-05]) #good guess for 1MA shots
        if ne_params_from_last_successful_fit is not None:
            list_of_ne_guesses.insert(0, ne_params_from_last_successful_fit) #use the parameters from the last successful fit as a first guess


        for ne_guess_idx in range(len(list_of_ne_guesses)):
            ne_guess = list_of_ne_guesses[ne_guess_idx]
            try:
                print('ne guess')
                print(ne_guess)
                ne_params, ne_covariance = curve_fit(Osborne_Tanh_cubic, list_of_raw_ne_xvalues_shifted, list_of_raw_ne/1e20, p0=ne_guess, sigma=list_of_raw_ne_err_weights_applied/1e20, absolute_sigma=False, maxfev=2000, bounds=([0.85, 0.001, 0, -0.001, -np.inf, -np.inf, -np.inf], [1.05, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])) #should now be in psi
                ne_fitted = 1e20*Osborne_Tanh_cubic(generated_psi_grid, ne_params[0], ne_params[1], ne_params[2], ne_params[3], ne_params[4], ne_params[5], ne_params[6])
                #plt.plot(generated_psi_grid, ne_fitted)
                #plt.scatter(list_of_raw_ne_xvalues_shifted, list_of_raw_ne, marker='x')


                list_of_ne_fitted.append(ne_fitted)
                list_of_ne_fitted_error.append(average_ne_error_band)
                successful_ne_fit_mask.append(True)
                ne_params_from_last_successful_fit = ne_params # to be used as a first guess for the next time point
                list_of_ne_params_that_worked.append(ne_params)
                list_of_indices_that_worked.append(t_idx)

                break #guess worked so exit the for loop
            except:
                if ne_guess_idx == len(list_of_ne_guesses) - 1:
                    # If all the guesses failed, set the fit parameters to none
                    list_of_ne_fitted.append(np.full(len(generated_psi_grid), np.nan))
                    list_of_ne_fitted_error.append(np.full(len(generated_psi_grid), np.nan))

                    ne_params = None
                    ne_covariance = None
                    ne_fitted = None
                    successful_ne_fit_mask.append(False)
                    print('NE FIT FAILED')
                else:
                    # move onto the next guess
                    continue


    list_of_ne_fitted = np.array(list_of_ne_fitted)
    list_of_te_fitted = np.array(list_of_te_fitted)
    list_of_ne_fitted_error = np.array(list_of_ne_fitted_error)
    list_of_te_fitted_error = np.array(list_of_te_fitted_error)
    successful_te_fit_mask = np.array(successful_te_fit_mask)
    successful_ne_fit_mask = np.array(successful_ne_fit_mask)


    combined_successful_fit_mask = np.logical_and(successful_te_fit_mask, successful_ne_fit_mask) # make sure the fit succeeded for both ne and Te
    new_times_for_results = new_times_for_results[combined_successful_fit_mask]

    list_of_ne_fitted = list_of_ne_fitted[combined_successful_fit_mask]
    list_of_te_fitted = list_of_te_fitted[combined_successful_fit_mask]
    list_of_ne_fitted_error = list_of_ne_fitted_error[combined_successful_fit_mask]
    list_of_te_fitted_error = list_of_te_fitted_error[combined_successful_fit_mask]


    list_of_ne_fitted_at_Thomson_times = np.array(list_of_ne_fitted_at_Thomson_times)
    list_of_te_fitted_at_Thomson_times = np.array(list_of_te_fitted_at_Thomson_times)
    list_of_initial_fit_times_for_checking_smoothing_ne = np.array(list_of_Thomson_times_ne_ms) - t_min
    list_of_initial_fit_times_for_checking_smoothing_te = np.array(list_of_Thomson_times_te_ms) - t_min


    for idx in range(len(new_times_for_results)):
        if idx % 20 == 0:
            plt.plot(generated_psi_grid, list_of_ne_fitted[idx], label = new_times_for_results[idx])
            plt.fill_between(generated_psi_grid, list_of_ne_fitted[idx] - list_of_ne_fitted_error[idx], list_of_ne_fitted[idx] + list_of_ne_fitted_error[idx], alpha=0.5)
            plt.legend()
    plt.show()
    for idx in range(len(new_times_for_results)):
        if idx % 20 == 0:
            plt.plot(generated_psi_grid, list_of_te_fitted[idx], label = new_times_for_results[idx])
            plt.fill_between(generated_psi_grid, list_of_te_fitted[idx] - list_of_te_fitted_error[idx], list_of_te_fitted[idx] + list_of_te_fitted_error[idx], alpha=0.5)
            plt.legend()
    plt.show()





    radii_to_plot = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05]
    list_of_generated_psi_grid_indices = [10, 20, 30, 40, 50, 60, 70, 80, 130, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205]
    # Create figure to show the evolution of each radial location
    fig, axs = plt.subplots(5, 4, figsize=(20, 15))

    for idx, ax in enumerate(axs.flatten()):
        print('idx')
        print(idx)
        #cycle through every psi value and plot its evolution in time.
        psi_value_to_evolve = radii_to_plot[idx]
        psi_idx = list_of_generated_psi_grid_indices[idx]
        print('psi_idx')
        print(psi_idx)
        ax.scatter(new_times_for_results, list_of_ne_fitted[:, psi_idx], label=f'psi = {psi_value_to_evolve:.2f}', marker='o')
        ax.scatter(list_of_initial_fit_times_for_checking_smoothing_ne, list_of_ne_fitted_at_Thomson_times[:, psi_idx], marker='x', color='red')
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    

    fig, axs = plt.subplots(5, 4, figsize=(20, 15))

    for idx, ax in enumerate(axs.flatten()):
        #cycle through every psi value and plot its evolution in time.
        psi_value_to_evolve = radii_to_plot[idx]
        psi_idx = list_of_generated_psi_grid_indices[idx]

        ax.plot(new_times_for_results, list_of_te_fitted[:, psi_idx], label=f'psi = {psi_value_to_evolve:.2f}')
        ax.scatter(list_of_initial_fit_times_for_checking_smoothing_te, list_of_te_fitted_at_Thomson_times[:, psi_idx], marker='x', color='red')
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.show()

    Rmid_grid = psi_to_Rmid_map(shot, t_min, t_max, generated_psi_grid, new_times_for_results) #this is a 2D array of Rmid values at every psi value at every time point




    return new_times_for_results, generated_psi_grid, Rmid_grid, list_of_ne_fitted, list_of_ne_fitted_error, list_of_te_fitted, list_of_te_fitted_error





def evolve_fits_by_radius(new_times, old_times, psi_grid, yvalues, y_error):
    '''
    INPUTS
    ------
    new_times: 1D array of times for the output fits (usually i set this to be every ms)
    old_times: 1D array of times for the input fits (this is just the times of the raw Thomson data)
    psi_grid: 1D array of psi values that the profiles are defined on
    yvalues: 2D array of Thomson profiles at every time in old_times
    y_error: 2D array with the corresponding error bars

    OUTPUTS
    -------
    list_of_fits_at_new_times: 2D array of evolved fits on psi_grid at every time in new_times
    list_of_errors_at_new_times: 2D array of the corresponding error bars
    
    DESCRIPTION
    -----------
    For every psi value, the input Thomson fits is evolved in time with a quadratic function.
    These fits are then evaluated at the new times and returned in a 2D array.
    NOTE that a quadratic temporal evolution isn't always reasonable, and some manual checking should be done.
    '''


    yvalues = np.array(yvalues)


    list_of_quadratic_fits = []
    list_of_new_values_at_each_radius = []
    list_of_std_at_each_radius = []


    for idx in range(len(psi_grid)):
        quadratic_fit = np.polyfit(old_times, yvalues[:, idx], 2) #quadratic fit in time at every radial location
        list_of_quadratic_fits.append(quadratic_fit) #save the quadratic coefficients
        new_yvalues = np.polyval(quadratic_fit, new_times) #evaluate the quadratic on the new timebase

        error = np.mean(y_error[:, idx]) #just take an average for the standard deviation
        average_std = np.full(len(new_times), error)
        list_of_std_at_each_radius.append(average_std)
        list_of_new_values_at_each_radius.append(new_yvalues) # got rid of fit_mean

    list_of_new_values_at_each_radius = np.array(list_of_new_values_at_each_radius)
    list_of_std_at_each_radius = np.array(list_of_std_at_each_radius)


    list_of_fits_at_new_times = []
    list_of_errors_at_new_times = []
    for idx in range(len(list_of_new_values_at_each_radius[0])):
        list_of_fits_at_new_times.append(list_of_new_values_at_each_radius[:, idx])
        list_of_errors_at_new_times.append(list_of_std_at_each_radius[:, idx])

    return list_of_fits_at_new_times, list_of_errors_at_new_times




def master_fit_ne_Te_2D_quadratic(list_of_shots, list_of_t_min, list_of_t_max, time_window_for_evolution = None, plot = False):
    '''
    INPUTS
    ------
    shot: int, C-Mod shot number
    t_min: int, start time for the fits (ms)
    t_max: int, end time for the fits (ms)
    time_window_for_evolution: 2 element list, start and end time for the quadratic evolution (ms). 
                                I have found that evolving between [20,100] is often a good choice for H-mode formation.

    OUTPUTS
    -------
    output_time_grid: output time grid normalised to t_min (this is set to be every ms by default)
    generated_psi_grid: 1D array of psi values that the profiles are defined on
    Rmid_grid: 2D array of Rmid values at every psi value at every time point
    new_ne_values: 2D array containing ne profiles at every time point on the output time grid
    new_ne_err: 2D array containing the corresponding error bars
    new_Te_values: 2D array containing Te profiles at every time point on the output time grid
    new_Te_err: 2D array containing the corresponding error bars

    DESCRIPTION
    -----------
    Function fits every Thomson time-point separately.
    Then takes these fits and evolves every radial location in time using a quadratic function.
    The result is then an array of profiles on the output time grid (which I've set as every ms).
    TODO: fix errors implementation once they've been added in the 1D fitting function.
    '''

    # is a single shot is passed in, need to convert this to a 1 element list so I can cycle through it
    if isinstance(list_of_shots, int):
        list_of_shots = [list_of_shots]
        list_of_t_min = [list_of_t_min]
        list_of_t_max = [list_of_t_max]



    # Now psi is just the generated psi grid. Te and Te_err are 2D arrays with each row being a fit at a different time.'times' just contains these fit times
    fitted_Te_data_at_Thomson_times = {
        'times': [],
        'Te': [],
        'Te_err': []
    }

    fitted_ne_data_at_Thomson_times = {
        'times': [],
        'ne': [],
        'ne_err': []
    }


    for individual_shot, individual_t_min, individual_t_max in zip(list_of_shots, list_of_t_min, list_of_t_max):

        # get the ne and Te fits at each time point from the 1D fitting function
        generated_psi_grid, list_of_Thomson_times_te_ms, list_of_te_fitted_at_Thomson_times, list_of_te_fitted_err_at_Thomson_times, list_of_te_reduced_chi_squared, \
        list_of_te_fit_type, list_of_Thomson_times_ne_ms, list_of_ne_fitted_at_Thomson_times, list_of_ne_fitted_err_at_Thomson_times, list_of_ne_reduced_chi_squared, \
        list_of_ne_fit_type = master_fit_ne_Te_1D(individual_shot, individual_t_min, individual_t_max, scale_core_TS_to_TCI = True, plot_the_fits=False, remove_zeros_before_fitting=True, shift_to_2pt_model=True, return_processed_raw_data=False, return_error_bars_on_fits=True, enforce_mtanh=True)

        # CONVERT THE 2D ARRAYS OF RAW DATA INTO 1D ARRAYS SO THAT WEIGHTS CAN BE APPLIED AND THE SMOOTHED FITS CAN BE APPLIED
        for idx in range(len(list_of_Thomson_times_te_ms)):

            fitted_Te_data_at_Thomson_times['times'].append(list_of_Thomson_times_te_ms[idx] - individual_t_min)
            fitted_Te_data_at_Thomson_times['Te'].append(list_of_te_fitted_at_Thomson_times[idx])
            fitted_Te_data_at_Thomson_times['Te_err'].append(list_of_te_fitted_err_at_Thomson_times[idx])


        for idx in range(len(list_of_Thomson_times_ne_ms)):
            fitted_ne_data_at_Thomson_times['times'].append(list_of_Thomson_times_ne_ms[idx] - individual_t_min)
            fitted_ne_data_at_Thomson_times['ne'].append(list_of_ne_fitted_at_Thomson_times[idx])
            fitted_ne_data_at_Thomson_times['ne_err'].append(list_of_ne_fitted_err_at_Thomson_times[idx])
    
    for key in fitted_ne_data_at_Thomson_times.keys():
        fitted_ne_data_at_Thomson_times[key] = np.array(fitted_ne_data_at_Thomson_times[key])
    
    for key in fitted_Te_data_at_Thomson_times.keys():
        fitted_Te_data_at_Thomson_times[key] = np.array(fitted_Te_data_at_Thomson_times[key])


    # Find the biggest time window and use this as the time grid
    largest_t_delta = 0
    for t_min, t_max in zip(list_of_t_min, list_of_t_max):
        t_delta = t_max - t_min
        if t_delta > largest_t_delta:
            largest_t_delta = t_delta

    # Time window for evolution
    if time_window_for_evolution is not None:

        ne_times_mask = (fitted_ne_data_at_Thomson_times['times'] > time_window_for_evolution[0]) & (fitted_ne_data_at_Thomson_times['times'] < time_window_for_evolution[1])
        for key in fitted_ne_data_at_Thomson_times.keys():
            fitted_ne_data_at_Thomson_times[key] = np.array(fitted_ne_data_at_Thomson_times[key])[ne_times_mask]

        te_times_mask = (fitted_Te_data_at_Thomson_times['times'] > time_window_for_evolution[0]) & (fitted_Te_data_at_Thomson_times['times'] < time_window_for_evolution[1])
        for key in fitted_Te_data_at_Thomson_times.keys():
            fitted_Te_data_at_Thomson_times[key] = np.array(fitted_Te_data_at_Thomson_times[key])[te_times_mask]

        output_time_grid = np.arange(time_window_for_evolution[0], time_window_for_evolution[1], 1)
    else:
        output_time_grid = np.arange(0, largest_t_delta, 1)



    if plot == True:
        # Plot to show how the quadratic fit actually looks
        evolve_fits_by_radius_example_for_panel_plots(fitted_ne_data_at_Thomson_times['times'], generated_psi_grid, fitted_ne_data_at_Thomson_times['ne'], output_time_grid=output_time_grid)
        evolve_fits_by_radius_example_for_panel_plots(fitted_Te_data_at_Thomson_times['times'], generated_psi_grid, fitted_Te_data_at_Thomson_times['Te'], output_time_grid=output_time_grid)
    
    # Evolve the fits in time with a qaudatic
    new_ne_values, new_ne_err = evolve_fits_by_radius(output_time_grid, fitted_ne_data_at_Thomson_times['times'], generated_psi_grid, fitted_ne_data_at_Thomson_times['ne'], fitted_ne_data_at_Thomson_times['ne_err'])
    new_Te_values, new_Te_err = evolve_fits_by_radius(output_time_grid, fitted_Te_data_at_Thomson_times['times'], generated_psi_grid, fitted_Te_data_at_Thomson_times['Te'], fitted_Te_data_at_Thomson_times['Te_err'])
    
    Rmid_grid = psi_to_Rmid_map_multiple_shots(list_of_shots, list_of_t_min, list_of_t_max, generated_psi_grid, output_time_grid)

    # return the fits + errors with time eovlution (every ms).
    # Also return the values generated by the fits to INDIVIDUAL time slices (at Thomson times) so the evolution can be compared against these.
    return output_time_grid, generated_psi_grid, Rmid_grid, new_ne_values, new_ne_err, new_Te_values, new_Te_err, \
        fitted_ne_data_at_Thomson_times['times'], fitted_ne_data_at_Thomson_times['ne'], fitted_ne_data_at_Thomson_times['ne_err'], \
        fitted_Te_data_at_Thomson_times['times'], fitted_Te_data_at_Thomson_times['Te'], fitted_Te_data_at_Thomson_times['Te_err']







def master_fit_2D_alt_combined_shots(list_of_shots, list_of_t_min, list_of_t_max, smoothing_window=15):
    '''
    Exactly the same as the window_smoothing 2D fitting function, except that the 1D fitting
    function (master_fit_ne_Te_1D) is used to do the fits at every time point.
    This gives a bit more flexibility (since it also tries a cubic fit), but currently
    does not have a post-fitting outlier rejection method.

    Return Rmid_grid should only be an option when a single shot is given.

    TODO:
    Add in functionality for the psi_to_Rmid_map to take in multiple shots. This shouldn't be too difficult but just haven't implemented it yet.
    Implement some option for post-fit outlier rejection
    Let this function also use a cubic to fit if it wants.
    '''

    # is a single shot is passed in, need to convert this to a 1 element list so I can cycle through it
    if isinstance(list_of_shots, int):
        list_of_shots = [list_of_shots]
        list_of_t_min = [list_of_t_min]
        list_of_t_max = [list_of_t_max]


    # STORE THE RAW DATA FOR ALL SHOTS IN A 1D ARRAY (WITH CORRESPONDING TIMES) SO WEIGHTS CAN BE APPLIED
    # All the elements in the dictionary are 1D arrays of the same length (i.e every Te value has a corresponding time, psi value and error bar)
    raw_te_data_flattened = {
        'times': [],
        'psi': [],
        'Te': [],
        'Te_err': []
    }

    raw_ne_data_flattened = {
        'times': [],
        'psi': [],
        'ne': [],
        'ne_err': []
    }

    # Now psi is just the generated psi grid. Te and Te_err are 2D arrays with each row being a fit at a different time.'times' just contains these fit times
    fitted_Te_data_at_Thomson_times = {
        'times': [],
        'Te': [],
        'Te_err': []
    }

    fitted_ne_data_at_Thomson_times = {
        'times': [],
        'ne': [],
        'ne_err': []
    }

    for individual_shot, individual_t_min, individual_t_max in zip(list_of_shots, list_of_t_min, list_of_t_max):

        # get the ne and Te fits at each time point from the 1D fitting function
        generated_psi_grid, list_of_Thomson_times_te_ms, list_of_te_fitted_at_Thomson_times, list_of_te_fitted_std_at_Thomson_times,\
        list_of_te_reduced_chi_squared, list_of_te_fit_type,\
        list_of_Thomson_times_ne_ms, list_of_ne_fitted_at_Thomson_times, list_of_ne_fitted_std_at_Thomson_times,\
        list_of_ne_reduced_chi_squared, list_of_ne_fit_type,\
        list_of_total_psi_te, list_of_total_te, list_of_total_te_err,\
        list_of_total_psi_ne, list_of_total_ne, list_of_total_ne_err = master_fit_ne_Te_1D(individual_shot, individual_t_min, individual_t_max, plot_the_fits=False, remove_zeros_before_fitting=True, shift_to_2pt_model=True, return_processed_raw_data=True, return_error_bars_on_fits=True, scale_core_TS_to_TCI=True, enforce_mtanh=True)


        # CONVERT THE 2D ARRAYS OF RAW DATA INTO 1D ARRAYS SO THAT WEIGHTS CAN BE APPLIED AND THE SMOOTHED FITS CAN BE APPLIED
        for idx in range(len(list_of_Thomson_times_te_ms)):
            no_of_points = len(list_of_total_psi_te[idx])

            raw_te_data_flattened['times'].extend((list_of_Thomson_times_te_ms[idx]*np.ones(no_of_points)) - individual_t_min) # use the time relative to the start of the window
            raw_te_data_flattened['psi'].extend(list_of_total_psi_te[idx])
            raw_te_data_flattened['Te'].extend(list_of_total_te[idx])
            raw_te_data_flattened['Te_err'].extend(list_of_total_te_err[idx])

            fitted_Te_data_at_Thomson_times['times'].append(list_of_Thomson_times_te_ms[idx] - individual_t_min)
            fitted_Te_data_at_Thomson_times['Te'].append(list_of_te_fitted_at_Thomson_times[idx])
            fitted_Te_data_at_Thomson_times['Te_err'].append(list_of_te_fitted_std_at_Thomson_times[idx])



        for idx in range(len(list_of_Thomson_times_ne_ms)):
            no_of_points = len(list_of_total_psi_ne[idx])

            raw_ne_data_flattened['times'].extend((list_of_Thomson_times_ne_ms[idx]*np.ones(no_of_points)) - individual_t_min) # use the time relative to the start of the window
            raw_ne_data_flattened['psi'].extend(list_of_total_psi_ne[idx])
            raw_ne_data_flattened['ne'].extend(list_of_total_ne[idx])
            raw_ne_data_flattened['ne_err'].extend(list_of_total_ne_err[idx])

            fitted_ne_data_at_Thomson_times['times'].append(list_of_Thomson_times_ne_ms[idx] - individual_t_min)
            fitted_ne_data_at_Thomson_times['ne'].append(list_of_ne_fitted_at_Thomson_times[idx])
            fitted_ne_data_at_Thomson_times['ne_err'].append(list_of_ne_fitted_std_at_Thomson_times[idx])


    for key in raw_te_data_flattened.keys():
        raw_te_data_flattened[key] = np.array(raw_te_data_flattened[key])
    
    for key in raw_ne_data_flattened.keys():
        raw_ne_data_flattened[key] = np.array(raw_ne_data_flattened[key])
    
    for key in fitted_Te_data_at_Thomson_times.keys():
        fitted_Te_data_at_Thomson_times[key] = np.array(fitted_Te_data_at_Thomson_times[key])

    for key in fitted_ne_data_at_Thomson_times.keys():
        fitted_ne_data_at_Thomson_times[key] = np.array(fitted_ne_data_at_Thomson_times[key])


    sorted_ne_indices = np.argsort(raw_ne_data_flattened['times'])
    for key in raw_ne_data_flattened.keys():
        raw_ne_data_flattened[key] = raw_ne_data_flattened[key][sorted_ne_indices]

    sorted_te_indices = np.argsort(raw_te_data_flattened['times'])
    for key in raw_te_data_flattened.keys():
        raw_te_data_flattened[key] = raw_te_data_flattened[key][sorted_te_indices]



    # get an average errorbar
    average_te_error_band = np.mean(fitted_Te_data_at_Thomson_times['Te_err'], axis=0) #just take an average for the standard deviation
    average_ne_error_band = np.mean(fitted_ne_data_at_Thomson_times['ne_err'], axis=0) #just take an average for the standard deviation



    # Find the biggest time window and use this as the time grid
    largest_t_delta = 0
    for t_min, t_max in zip(list_of_t_min, list_of_t_max):
        t_delta = t_max - t_min
        if t_delta > largest_t_delta:
            largest_t_delta = t_delta


    output_time_grid = np.arange(0, largest_t_delta, 1) # return fits on 1ms timebase


    te_params_from_last_successful_fit = None
    ne_params_from_last_successful_fit = None


    # Lists to store the smoothed fits
    te_fitted_at_output_times = {
        'Te': [],
        'Te_err': [],
        'Te_successful_fit_mask': []
    }

    ne_fitted_at_output_times = {
        'ne': [],
        'ne_err': [],
        'ne_successful_fit_mask': []
    }


    # now do the window smoothing
    for t_idx in range(len(output_time_grid)):
        time = output_time_grid[t_idx]

        print('TIME: ', time)

        #apply the Gaussian filter by making the error bars larger for further away points
        ne_weights = 1 / np.sqrt(np.exp((-1/2) * (raw_ne_data_flattened['times'] - time)**2 / (smoothing_window**2)))
        raw_ne_data_flattened['ne_err_weights_applied'] = ne_weights*raw_ne_data_flattened['ne_err']

        Te_weights = 1 / np.sqrt(np.exp((-1/2) * (raw_te_data_flattened['times'] - time)**2 / (smoothing_window**2)))
        raw_te_data_flattened['Te_err_weights_applied'] = Te_weights*raw_te_data_flattened['Te_err']


        # FITTING

        # some initial te guesses
        list_of_te_guesses = []
        list_of_te_guesses.append([ 9.92614859e-01,  4.01791101e-02,  2.55550908e+02,  1.28542623e+01,  2.17777084e-01, -3.45196862e-03,  1.42947373e-04])
        if te_params_from_last_successful_fit is not None:
            list_of_te_guesses.insert(0, te_params_from_last_successful_fit) #use the parameters from the last successful fit as a first guess

        for te_guess_idx in range(len(list_of_te_guesses)):
            te_guess = list_of_te_guesses[te_guess_idx]
            try:
                te_params, te_covariance = curve_fit(Osborne_Tanh_cubic, raw_te_data_flattened['psi'], raw_te_data_flattened['Te'], p0=te_guess, sigma=raw_te_data_flattened['Te_err_weights_applied'], absolute_sigma=False, maxfev=2000, bounds=([0.85, 0, 0, -0.001, -np.inf, -np.inf, -np.inf], np.inf)) #should now be in psi
                te_fitted = Osborne_Tanh_cubic(generated_psi_grid, te_params[0], te_params[1], te_params[2], te_params[3], te_params[4], te_params[5], te_params[6])

                #plt.plot(generated_psi_grid, te_fitted)
                #plt.scatter(list_of_raw_ne_xvalues_shifted, list_of_raw_Te, marker='x')
                #plt.show()
                te_fitted_at_output_times['Te'].append(te_fitted)
                te_fitted_at_output_times['Te_err'].append(average_te_error_band)
                te_fitted_at_output_times['Te_successful_fit_mask'].append(True)
                te_params_from_last_successful_fit = te_params # to be used as a first guess for the next time point
                break #guess worked so exit the for loop
            except:
                if te_guess_idx == len(list_of_te_guesses) - 1:
                    # If all the guesses failed, set the fit parameters to none
                    te_fitted_at_output_times['Te'].append(np.full(len(generated_psi_grid), np.nan))
                    te_fitted_at_output_times['Te_err'].append(np.full(len(generated_psi_grid), np.nan))

                    te_params = None
                    te_covariance = None
                    te_fitted = None
                    te_fitted_at_output_times['Te_successful_fit_mask'].append(False)
                    print('TE FIT FAILED')
                else:
                    # move onto the next guess
                    continue



        # some initial ne guesses
        list_of_ne_guesses = []
        list_of_ne_guesses.append([1.00604712, 0.037400836, 2.10662412, 0.0168897974, -0.0632778417, 0.00229233952, -2.0627212e-05]) #good initial guess for 650kA shots
        list_of_ne_guesses.append([1.02123755e+00,  5.02744526e-02,  2.54219267e+00, -9.99999694e-04, 2.58724602e-02, -2.32961078e-03,  4.20279037e-05]) #good guess for 1MA shots
        if ne_params_from_last_successful_fit is not None:
            list_of_ne_guesses.insert(0, ne_params_from_last_successful_fit) #use the parameters from the last successful fit as a first guess


        for ne_guess_idx in range(len(list_of_ne_guesses)):
            ne_guess = list_of_ne_guesses[ne_guess_idx]
            try:
                print('ne guess')
                print(ne_guess)
                ne_params, ne_covariance = curve_fit(Osborne_Tanh_cubic, raw_ne_data_flattened['psi'], raw_ne_data_flattened['ne']/1e20, p0=ne_guess, sigma=raw_ne_data_flattened['ne_err_weights_applied']/1e20, absolute_sigma=False, maxfev=2000, bounds=([0.85, 0.001, 0, -0.001, -np.inf, -np.inf, -np.inf], [1.05, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])) #should now be in psi
                ne_fitted = 1e20*Osborne_Tanh_cubic(generated_psi_grid, ne_params[0], ne_params[1], ne_params[2], ne_params[3], ne_params[4], ne_params[5], ne_params[6])
                #plt.plot(generated_psi_grid, ne_fitted)
                #plt.scatter(list_of_raw_ne_xvalues_shifted, list_of_raw_ne, marker='x')


                ne_fitted_at_output_times['ne'].append(ne_fitted)
                ne_fitted_at_output_times['ne_err'].append(average_ne_error_band)
                ne_fitted_at_output_times['ne_successful_fit_mask'].append(True)
                ne_params_from_last_successful_fit = ne_params # to be used as a first guess for the next time point

                break #guess worked so exit the for loop
            except:
                if ne_guess_idx == len(list_of_ne_guesses) - 1:
                    # If all the guesses failed, set the fit parameters to none
                    ne_fitted_at_output_times['ne'].append(np.full(len(generated_psi_grid), np.nan))
                    ne_fitted_at_output_times['ne_err'].append(np.full(len(generated_psi_grid), np.nan))

                    ne_params = None
                    ne_covariance = None
                    ne_fitted = None
                    ne_fitted_at_output_times['ne_successful_fit_mask'].append(False)
                    print('NE FIT FAILED')
                else:
                    # move onto the next guess
                    continue
        '''
        if t_idx == 90:
            for idx in range(len(fitted_Te_data_at_Thomson_times['Te'])):
                if fitted_Te_data_at_Thomson_times['times'][idx] > 80 and fitted_Te_data_at_Thomson_times['times'][idx] < 100:
                    plt.plot(generated_psi_grid, fitted_Te_data_at_Thomson_times['Te'][idx], label = fitted_Te_data_at_Thomson_times['times'][idx])
            
            plt.plot(generated_psi_grid, te_fitted, label = 'smoothed fit', color='black')
            plt.axvline(x=1, color='black', linestyle='--')
            plt.grid(linestyle='--', alpha=0.3)
            plt.legend()
            plt.show()
        '''


    for key in ne_fitted_at_output_times.keys():
        ne_fitted_at_output_times[key] = np.array(ne_fitted_at_output_times[key])

    for key in te_fitted_at_output_times.keys():
        te_fitted_at_output_times[key] = np.array(te_fitted_at_output_times[key])


    combined_successful_fit_mask = np.logical_and(te_fitted_at_output_times['Te_successful_fit_mask'], ne_fitted_at_output_times['ne_successful_fit_mask']) # make sure the fit succeeded for both ne and Te
    
    output_time_grid = output_time_grid[combined_successful_fit_mask]
    ne_fitted_at_output_times['ne'] = ne_fitted_at_output_times['ne'][combined_successful_fit_mask]
    te_fitted_at_output_times['Te'] = te_fitted_at_output_times['Te'][combined_successful_fit_mask]
    ne_fitted_at_output_times['ne_err'] = ne_fitted_at_output_times['ne_err'][combined_successful_fit_mask]
    te_fitted_at_output_times['Te_err'] = te_fitted_at_output_times['Te_err'][combined_successful_fit_mask]



    '''
    for idx in range(len(output_time_grid)):
        if idx % 20 == 0:
            plt.plot(generated_psi_grid, ne_fitted_at_output_times['ne'][idx], label = output_time_grid[idx])
            plt.fill_between(generated_psi_grid, ne_fitted_at_output_times['ne'][idx] - ne_fitted_at_output_times['ne_err'][idx], ne_fitted_at_output_times['ne'][idx] + ne_fitted_at_output_times['ne_err'][idx], alpha=0.5)
            plt.legend()
    plt.show()
    for idx in range(len(output_time_grid)):
        if idx % 20 == 0:
            plt.plot(generated_psi_grid, te_fitted_at_output_times['Te'][idx], label = output_time_grid[idx])
            plt.fill_between(generated_psi_grid, te_fitted_at_output_times['Te'][idx] - te_fitted_at_output_times['Te_err'][idx], te_fitted_at_output_times['Te'][idx] + te_fitted_at_output_times['Te_err'][idx], alpha=0.5)
            plt.legend()
    plt.show()
    '''




    '''
    radii_to_plot = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05]
    list_of_generated_psi_grid_indices = [10, 20, 30, 40, 50, 60, 70, 80, 130, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205]
    # Create figure to show the evolution of each radial location
    fig, axs = plt.subplots(5, 4, figsize=(20, 15))

    for idx, ax in enumerate(axs.flatten()):
        print('idx')
        print(idx)
        #cycle through every psi value and plot its evolution in time.
        psi_value_to_evolve = radii_to_plot[idx]
        psi_idx = list_of_generated_psi_grid_indices[idx]
        print('psi_idx')
        print(psi_idx)
        ax.scatter(output_time_grid, ne_fitted_at_output_times['ne'][:, psi_idx], label=f'psi = {psi_value_to_evolve:.2f}', marker='o')
        ax.scatter(fitted_ne_data_at_Thomson_times['times'], fitted_ne_data_at_Thomson_times['ne'][:, psi_idx], marker='x', color='red')
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    

    fig, axs = plt.subplots(5, 4, figsize=(20, 15))

    for idx, ax in enumerate(axs.flatten()):
        #cycle through every psi value and plot its evolution in time.
        psi_value_to_evolve = radii_to_plot[idx]
        psi_idx = list_of_generated_psi_grid_indices[idx]

        ax.plot(output_time_grid, te_fitted_at_output_times['Te'][:, psi_idx], label=f'psi = {psi_value_to_evolve:.2f}')
        ax.scatter(fitted_Te_data_at_Thomson_times['times'], fitted_Te_data_at_Thomson_times['Te'][:, psi_idx], marker='x', color='red')
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    '''


    Rmid_grid = psi_to_Rmid_map_multiple_shots(list_of_shots, list_of_t_min, list_of_t_max, generated_psi_grid, output_time_grid)



    # return the fits + errors with time eovlution (every ms).
    # Also return the values generated by the fits to INDIVIDUAL time slices (at Thomson times) so the evolution can be compared against these.
    return output_time_grid, generated_psi_grid, Rmid_grid, ne_fitted_at_output_times['ne'], ne_fitted_at_output_times['ne_err'], te_fitted_at_output_times['Te'], te_fitted_at_output_times['Te_err'], \
        fitted_ne_data_at_Thomson_times['times'], fitted_ne_data_at_Thomson_times['ne'], fitted_ne_data_at_Thomson_times['ne_err'], \
        fitted_Te_data_at_Thomson_times['times'], fitted_Te_data_at_Thomson_times['Te'], fitted_Te_data_at_Thomson_times['Te_err']





def plot_outputs_of_2D_fitting(output_time_grid, generated_psi_grid, ne_fitted, ne_err, Te_fitted, Te_err, ne_Thomson_times, ne_Thomson_times_fitted, te_Thomson_times, te_Thomson_times_fitted, path_to_save_plots=None, show_plots=False):
    '''
    INPUTS
    ------
    output_time_grid: 1D array of times (ms) that the fits are evaluated at
    generated_psi_grid: 1D array of psi values that the profiles are defined on
    Rmid_grid: 2D array of Rmid values at every psi value at every time point
    ne_fitted: 2D array of ne profiles at every time point on the output time grid
    ne_err: 2D array of the corresponding error bars
    Te_fitted: 2D array of Te profiles at every time point on the output time grid
    Te_err: 2D array of the corresponding error bars

    DESCRIPTION
    -----------
    Function to plot the outputs of the 2D fitting function.
    '''

    ne_fitted = np.array(ne_fitted)
    ne_Thomson_times_fitted = np.array(ne_Thomson_times_fitted)
    Te_fitted = np.array(Te_fitted)
    te_Thomson_times_fitted = np.array(te_Thomson_times_fitted)


    radii_to_plot = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05]
    list_of_generated_psi_grid_indices = [10, 20, 30, 40, 50, 60, 70, 80, 130, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205]


    # Plot the ne and Te fits
    fig, axs = plt.subplots(5, 4, figsize=(20, 15))

    for idx, ax in enumerate(axs.flatten()):
        #cycle through every psi value and plot its evolution in time.

        psi_value_to_evolve = radii_to_plot[idx]
        psi_idx = list_of_generated_psi_grid_indices[idx]
        ax.plot(output_time_grid, ne_fitted[:, psi_idx], label=f'psi = {psi_value_to_evolve:.2f}')
        ax.scatter(ne_Thomson_times, ne_Thomson_times_fitted[:, psi_idx], marker='x', color='red')
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    if path_to_save_plots is not None:
        plt.savefig(f'{path_to_save_plots}_ne.pdf')
    if show_plots == True:
        plt.show()
    plt.clf()
    

    fig, axs = plt.subplots(5, 4, figsize=(20, 15))

    for idx, ax in enumerate(axs.flatten()):
        #cycle through every psi value and plot its evolution in time.
        psi_value_to_evolve = radii_to_plot[idx]
        psi_idx = list_of_generated_psi_grid_indices[idx]
        ax.plot(output_time_grid, Te_fitted[:, psi_idx], label=f'psi = {psi_value_to_evolve:.2f}')
        ax.scatter(te_Thomson_times, te_Thomson_times_fitted[:, psi_idx], marker='x', color='red')
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    if path_to_save_plots is not None:
        plt.savefig(f'{path_to_save_plots}_Te.pdf')
    if show_plots == True:
        plt.show()
    plt.clf()



def plot_outputs_of_2D_fitting_NEAT(output_time_grid, generated_psi_grid, ne_fitted, ne_err, Te_fitted, Te_err, ne_Thomson_times, ne_Thomson_times_fitted, te_Thomson_times, te_Thomson_times_fitted, path_to_save_plots=None, show_plots=False):
    '''
    INPUTS
    ------
    output_time_grid: 1D array of times (ms) that the fits are evaluated at
    generated_psi_grid: 1D array of psi values that the profiles are defined on
    Rmid_grid: 2D array of Rmid values at every psi value at every time point
    ne_fitted: 2D array of ne profiles at every time point on the output time grid
    ne_err: 2D array of the corresponding error bars
    Te_fitted: 2D array of Te profiles at every time point on the output time grid
    Te_err: 2D array of the corresponding error bars

    DESCRIPTION
    -----------
    Same as above but make tidy for APS poster.
    '''

    ne_fitted = np.array(ne_fitted)
    ne_Thomson_times_fitted = np.array(ne_Thomson_times_fitted)
    Te_fitted = np.array(Te_fitted)
    te_Thomson_times_fitted = np.array(te_Thomson_times_fitted)


    radii_to_plot = [0.5, 0.9, 0.95, 0.99]
    list_of_generated_psi_grid_indices = [50, 130, 155, 175]

    font_size = 20
    line_width = 5
    marker_size = 150
    dpi_value = 300

    # Plot the ne fits
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5))  # Smaller plot size

    for idx, ax in enumerate(axs.flatten()):
        # Cycle through every psi value and plot its evolution in time.
        psi_value_to_evolve = radii_to_plot[idx]
        psi_idx = list_of_generated_psi_grid_indices[idx]
        
        # Plot line and scatter
        ax.plot(output_time_grid, ne_fitted[:, psi_idx] / 1e20, label=f'psi = {psi_value_to_evolve:.2f}', linewidth=line_width)
        ax.scatter(ne_Thomson_times, ne_Thomson_times_fitted[:, psi_idx] / 1e20, marker='x', color='red', s=marker_size)
        
        # Set axis labels with smaller font
        ax.set_xlabel(r't - $t_{LH}$ (ms)', fontsize=font_size)
        ax.set_ylabel(r'$n_e$ [x$10^{20}$ m$^{-3}$]', fontsize=font_size)

        # Set fixed y-limits for all panels
        ax.set_ylim(1.5, 2.7)
        
        # Tick size adjustments for clarity
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        
        # Grid and legend
        ax.grid(True)
        ax.legend(fontsize=font_size)

    plt.tight_layout()

    if path_to_save_plots is not None:
        plt.savefig(f'{path_to_save_plots}_ne.pdf')
    if show_plots == True:
        plt.show()
    plt.clf()

    '''
    

    fig, axs = plt.subplots(5, 4, figsize=(2, 2))

    for idx, ax in enumerate(axs.flatten()):
        #cycle through every psi value and plot its evolution in time.
        psi_value_to_evolve = radii_to_plot[idx]
        psi_idx = list_of_generated_psi_grid_indices[idx]
        ax.plot(output_time_grid, Te_fitted[:, psi_idx], label=f'psi = {psi_value_to_evolve:.2f}')
        ax.scatter(te_Thomson_times, te_Thomson_times_fitted[:, psi_idx], marker='x', color='red')
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    if path_to_save_plots is not None:
        plt.savefig(f'{path_to_save_plots}_Te.pdf')
    if show_plots == True:
        plt.show()
    plt.clf()
    '''




def plot_pedestal_evolution_NEAT(output_time_grid, generated_psi_grid, ne_fitted, ne_err, Te_fitted, Te_err, ne_Thomson_times, ne_Thomson_times_fitted, te_Thomson_times, te_Thomson_times_fitted, path_to_save_plots=None, show_plots=False):
    '''
    INPUTS
    ------
    output_time_grid: 1D array of times (ms) that the fits are evaluated at
    generated_psi_grid: 1D array of psi values that the profiles are defined on
    Rmid_grid: 2D array of Rmid values at every psi value at every time point
    ne_fitted: 2D array of ne profiles at every time point on the output time grid
    ne_err: 2D array of the corresponding error bars
    Te_fitted: 2D array of Te profiles at every time point on the output time grid
    Te_err: 2D array of the corresponding error bars

    DESCRIPTION
    -----------
    Same as above but make tidy for APS poster.
    '''


    ne_fitted = np.array(ne_fitted)
    ne_Thomson_times_fitted = np.array(ne_Thomson_times_fitted)
    Te_fitted = np.array(Te_fitted)
    te_Thomson_times_fitted = np.array(te_Thomson_times_fitted)

    plt.close('all')


    # Plot the profile every 20 ms
    font_size = 20


    fig, ax = plt.subplots(figsize=(10, 7.5))  # Smaller plot size

    for idx in range(len(output_time_grid)):
        if idx % 40 == 0:
            ax.plot(generated_psi_grid, ne_fitted[idx]/1e20, label = rf't - $t_{{LH}}$ = {output_time_grid[idx]}ms')
            ax.fill_between(generated_psi_grid, (ne_fitted[idx] - ne_err[idx])/1e20, (ne_fitted[idx] + ne_err[idx])/1e20, alpha=0.5)

    ax.set_xlabel(r'$\psi$', fontsize=font_size)
    ax.set_ylabel(r'$n_e$ [x$10^{20}$ m$^{-3}$]', fontsize=font_size)
    ax.legend(fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.grid(True)

    plt.tight_layout()
    if path_to_save_plots is not None:
        plt.savefig(f'{path_to_save_plots}_ne_evolution.pdf')
    if show_plots == True:
        plt.show()
    plt.clf()

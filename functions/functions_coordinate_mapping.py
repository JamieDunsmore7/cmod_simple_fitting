####### FUNCTIONS FOR MAPPING BETWEEN COORDINATE SYSTEMS #######

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



def psi_to_Rmid_map(shot, t_min, t_max, psi_grid, output_time_grid):
    '''
    INPUTS
    ---------
    shot: C-Mod shot number (integer)
    t_min: minimum time in ms
    t_max: maximum time in ms
    psi_grid: 1D array of psi values
    output_time_grid: 1D array of times in ms


    RETURNS
    ---------
    list_of_Rmids: 2D array of Rmid values at each psi value at each time point in the output_time_grid.
                   i.e list_of_Rmids[0] contains the Rmid values corresponding to psi_grid at the first output_time_grid time point.

    DESCRIPTION
    ---------
    The function uses eqtools to convert the input psi grid to an Rmid grid on the EFIT20 timebase at every EFIT20 time between t_min and t_max.
    This gives Rmid as a function of time at every psi value.
    The function then fits a QUADRATIC function through the Rmid evolution at every psi value, and returns the values of this function on the output_time_grid.
    '''

    t_min_s = t_min/1000
    t_max_s = t_max/1000

    try:
        e = eqtools.CModEFIT.CModEFITTree(shot, tree='EFIT20')
    except:
        e = eqtools.CModEFIT.CModEFITTree(shot, tree='ANALYSIS')
    eqtools_times = e.getTimeBase()
    eqtools_times_mask = (eqtools_times > t_min_s) & (eqtools_times < t_max_s)
    eqtools_times = eqtools_times[eqtools_times_mask] #just the times in the desired time window now

    norm_eqtools_times_ms = eqtools_times*1000 - t_min #convert to ms



    list_of_Rmids = []
    
    for psi in psi_grid:
        Rmid = e.rho2rho('psinorm', 'Rmid', psi, eqtools_times)

        quadratic_coefficients = np.polyfit(norm_eqtools_times_ms, Rmid, 2)
        quadratic_function = np.poly1d(quadratic_coefficients) 
        Rmid_at_output_time_grid = quadratic_function(output_time_grid)
        list_of_Rmids.append(Rmid_at_output_time_grid)
    
    list_of_Rmids = np.array(list_of_Rmids)
    list_of_Rmids = list_of_Rmids.T

    return list_of_Rmids


def psi_to_Rmid_map_multiple_shots(list_of_shots, list_of_t_min, list_of_t_max, psi_grid, output_time_grid):
    '''
    INPUTS
    ---------
    list_of_shots: list of C-Mod shot numbers (integers)
    list_of_t_min: list of minimum times in ms
    list_of_t_max: list of maximum times in ms
    psi_grid: 1D array of psi values
    output_time_grid: 1D array of times in ms


    RETURNS
    ---------
    list_of_Rmids: 2D array of Rmid values at each psi value at each time point in the output_time_grid.
                   i.e list_of_Rmids[0] contains the Rmid values corresponding to psi_grid at the first output_time_grid time point.

    DESCRIPTION
    ---------
    This function is exactly the same as psi_to_Rmid_map, but it combines data from multiple shots when doing the mapping.
    Instead of fitting a quadratic through the Rmid vs time plot at each psi value for a single shot, it does this for multiple shots.
    '''

    # Connect to the equilibria here because it is slow so don't want to have to do it for every shot for every psi value.
    list_of_eqtools_files = []
    for shot in list_of_shots:
        try:
            e = eqtools.CModEFIT.CModEFITTree(shot, tree='EFIT20')
        except:
            e = eqtools.CModEFIT.CModEFITTree(shot, tree='ANALYSIS')
        list_of_eqtools_files.append(e)


    list_of_Rmids = [] # to return


    for psi in psi_grid:
        Rmid_all_shots = []
        norm_eqtools_times_ms_all_shots = []
        for shot, t_min, t_max, eq in zip(list_of_shots, list_of_t_min, list_of_t_max, list_of_eqtools_files): # get the Rmid vs time data for each shot

            t_min_s = t_min/1000
            t_max_s = t_max/1000


            eqtools_times = eq.getTimeBase()
            eqtools_times_mask = (eqtools_times > t_min_s) & (eqtools_times < t_max_s)
            eqtools_times = eqtools_times[eqtools_times_mask] #just the times in the desired time window now

            norm_eqtools_times_ms = eqtools_times*1000 - t_min #convert to ms

            Rmid = eq.rho2rho('psinorm', 'Rmid', psi, eqtools_times)

            for Rmid_item in Rmid:
                Rmid_all_shots.append(Rmid_item)     
            for norm_eqtools_times_ms_item in norm_eqtools_times_ms:
                norm_eqtools_times_ms_all_shots.append(norm_eqtools_times_ms_item)

        # Now this should have data points from every shot for the quadratic fit.
        quadratic_coefficients = np.polyfit(norm_eqtools_times_ms_all_shots, Rmid_all_shots, 2)
        quadratic_function = np.poly1d(quadratic_coefficients)
        Rmid_at_output_time_grid = quadratic_function(output_time_grid)
        list_of_Rmids.append(Rmid_at_output_time_grid)
          

    list_of_Rmids = np.array(list_of_Rmids)
    list_of_Rmids = list_of_Rmids.T

    return list_of_Rmids
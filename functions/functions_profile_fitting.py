### Functions for cleaning up Thomson data and functional forms of the fits ###

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



def remove_zeros(xdata, ydata, y_error, core_only = True):
    '''
    INPUTS
    --------
    xdata: 1D array of x values
    ydata: 1D array of y values (ne or Te)
    y_error: 1D array of y errors (ne error or Te error)
    core_only: boolean, if True, only remove zeros from the core data. If False, remove zeros from the SOL data as well.
    RETURNS
    --------
    new_x: 1D array of x values with zeros removed
    new_y: 1D array of y values with zeros removed
    new_yerr: 1D array of y errors with zeros removed

    DESCRIPTION
    --------
    Function removes any data points where the Thomson data is zero.
    Can sometimes help to clean up the raw data before fitting.
    '''

    if core_only == True:
        mask  = (ydata != 0) | (xdata > 1) #only remove zeros from inside the LCFS
    else: #let the last two values in the array be zero because these zeros may be physical
        mask = ydata != 0

    new_x = xdata[mask]
    new_y = ydata[mask]
    new_yerr = y_error[mask]
    return new_x, new_y, new_yerr


def add_SOL_zeros(xdata, ydata, y_error):
    '''
    INPUTS
    --------
    xdata: 1D array of x values in Rmid coordinates (since the zero is added at 0.91m in Rmid coordinates)
    ydata: 1D array of y values (ne or Te)
    y_error: 1D array of y errors (ne error or Te error)

    DESCRIPTION
    --------
    Returns exactly the same arrays as the input, but with a zero added at R=0.91m.
    This can help to force the mtanh fit to zero if there aren't many data points in the SOL.
    '''
    xdata = np.append(xdata, 0.91)
    ydata = np.append(ydata, 0)
    y_error = np.append(y_error, 2*np.mean(y_error))
    return xdata, ydata, y_error


def add_SOL_zeros_in_psi_coords(xdata, ydata, y_error):
    '''
    INPUTS
    --------
    xdata: 1D array of x values in psi coordinates (since the zero is added at psi=1.05)
    ydata: 1D array of y values (ne or Te)
    y_error: 1D array of y errors (ne error or Te error)

    DESCRIPTION
    --------
    Returns exactly the same arrays as the input, but with a zero added at psi=1.05.
    This can help to force the mtanh fit to zero if there aren't many data points in the SOL.
    '''
    xdata = np.append(xdata, 1.05)
    ydata = np.append(ydata, 0)
    y_error = np.append(y_error, 2*np.mean(y_error))
    return xdata, ydata, y_error



### FITTING FUNCTIONS ###



def Osborne_linear_initial_guesses(radius, thomson_value):
    """
    INPUTS
    --------
    radius: 1D array of x-values (can be in any coordinate system). NOTE: works best if only the edge data is used
    thomson_value: 1D array of y-values (ne or Te)

    RETURNS
    --------
    initial_guesses: 1D array of reasonable initial guesses for the 5 Osborne linear fit parameters
    """
    average = sum(thomson_value) / len(thomson_value)
    bottom = average*0.3
    top = average*1.1
    #work inwards and find first radius above the bottom
    for idx in range(len(radius)-1, -1, -1):
        if thomson_value[idx] > bottom:
            max_r = radius[idx]
            break
    #keep working inwards to find first radius above the top
    for idx in range(len(radius)-1, -1, -1):
        if thomson_value[idx] > top:
            min_r = radius[idx]
            break
    width = max_r - min_r
    centre = (max_r + min_r) / 2
    if width <= 0 :
        width = radius[-3] - radius[-5] #just uses distance between 2 points that should be roughly in the pedestal region if the previous method failed
    initial_guesses = [centre, width, top, bottom, 0]
    return initial_guesses


def Osborne_Tanh_linear(x, c0, c1, c2, c3, c4):
    """
    INPUTS
    --------
    x: 1D array of x-values
    c0: float, pedestal center position
    c1: float, pedestal full width
    c2: float, Pedestal top
    c3: float, Pedestal bottom
    c4: float, inboard linear slope

    RETURNS
    --------
    F: 1D array of y-values corresponding to an Osborne tanh function with a linear inboard slope

    DESCRIPTION
    --------
    Function returns Osborne tanh function with a linear inboard slope and no outer slope (i.e flat in the SOL).
    Adapted from Osborne via Hughes idl script.
    """


    z = 2. * ( c0 - x ) / c1
    P1 = 1. + c4 * z
    P2 = 1.
    E1 = np.exp(z)
    E2 = np.exp(-1.*z)
    F = 0.5 * ( c2 + c3 + ( c2 - c3 ) * ( P1 * E1 - P2 * E2 ) / ( E1 + E2 ) )

    return F


def Osborne_Tanh_cubic(x, c0, c1, c2, c3, c4, c5, c6):
    """
    INPUTS
    --------
    x: 1D array of x-values
    c0: float, pedestal center position
    c1: float, pedestal full width
    c2: float, Pedestal top
    c3: float, Pedestal bottom
    c4: float, inboard linear slope
    c5: float, inboard quadratic term
    c6: float, inboard cubic term

    RETURNS
    --------
    F: 1D array of y-values corresponding to an Osborne tanh function with linear, quadratic and cubic inboard terms

    DESCRIPTION
    --------
    Function returns Osborne tanh function with linear, quadratic and cubic inboard terms and no outer slope (i.e flat in the SOL).
    Adapted from Osborne via Hughes idl script.
    """

    z = 2. * ( c0 - x ) / c1

    P1 = 1. + c4 * z + c5 * z**2 + c6 * z**3
    P2 = 1.
    E1 = np.exp(z)
    E2 = np.exp(-1.*z)
    
    F = 0.5 * ( c2 + c3 + ( c2 - c3 ) * ( P1 * E1 - P2 * E2 ) / ( E1 + E2 ) )

    return F


def Cubic(x, c0, c1, c2, c3):
    return c0 + c1*x + c2*x**2 + c3*x**3


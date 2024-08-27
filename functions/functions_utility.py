### Miscellaneous functions ###


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



def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def chi_squared(ydata, yfit, yerr):
    return np.sum((ydata - yfit)**2 / yerr**2)


def reduced_chi_squared(ydata, yfit, yerr, num_params):
    return chi_squared(ydata, yfit, yerr) / (len(ydata) - num_params)


def reduced_chi_squared_inside_separatrix(psi_values, ydata, yfit, yerr, num_params, only_edge = False):
    '''
    If only edge mask is true, will just calculate the reduced chi squared between psi = 0.6 and psi = 1
    '''
    if only_edge == True:
        inside_separatrix_mask = (psi_values > 0.6) & (psi_values < 1.0)
    else:
        inside_separatrix_mask = psi_values < 1.0
    ydata = ydata[inside_separatrix_mask]
    yfit = yfit[inside_separatrix_mask]
    yerr = yerr[inside_separatrix_mask]
    return chi_squared(ydata, yfit, yerr) / (len(ydata) - num_params)
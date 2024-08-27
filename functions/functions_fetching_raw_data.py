### FUNCTIONS FOR FETCHING RAW DATA FROM THE TREE ###


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

from functions.functions_utility import *


def get_raw_edge_Thomson_data(shot, t_min = None, t_max = None):
    '''
    INPUTS
    --------
    shot: C-Mod shot number (integer)
    t_min: minimum time in ms
    t_max: maximum time in ms

    RETURNS
    --------
    thomson_time: 1D array of time points at which Thomson data is available in ms
    ne: 2D array of electron density values at each radial value at each time point (m^-3)
    ne_err: 2D array of electron density errors at each radial value at each time point (m^-3)
    te: 2D array of electron temperature values at each radial value at each time point (eV)
    te_err: 2D array of electron temperature errors at each radial value at each time point (eV)
    rmid_array: 2D array of Rmid values at each radial value at each time point (m)
    r_array: single value (the radius in m of the edge Thomson system)
    z_array: 1D array of of the real-space z coordinates of the edge Thomson system in m


    DESCRIPTION
    --------
    Just grabs the raw edge Thomson data from the tree for a given time window.
    NOTE that on C-Mod the Thomson system was a vertical system.
    NOTE: edge Thomson data are only available for SHOT > 1000000000

    '''
    tree = MDSplus.Tree('CMOD', shot)
    te = tree.getNode('\\TOP.ELECTRONS.YAG_EDGETS.RESULTS:TE').data()
    te_err = tree.getNode('\\TOP.ELECTRONS.YAG_EDGETS.RESULTS:TE:ERROR').data()
    ne = tree.getNode('\\TOP.ELECTRONS.YAG_EDGETS.RESULTS:NE').data()
    ne_err = tree.getNode('\\TOP.ELECTRONS.YAG_EDGETS.RESULTS:NE:ERROR').data()
    rmid_array = tree.getNode('\\TOP.ELECTRONS.YAG_EDGETS.RESULTS:RMID').data()
    z_array = tree.getNode('\\TOP.ELECTRONS.YAG_EDGETS.DATA:FIBER_Z').data()
    r_array = tree.getNode('\\TOP.ELECTRONS.YAG.RESULTS.PARAM:R').data()
    thomson_time = tree.getNode('\\TOP.ELECTRONS.YAG_EDGETS.RESULTS:NE').dim_of().data()

    thomson_time *= 1000 #conversion to ms
    thomson_time = np.round(thomson_time).astype(int) #convert to integer number of ms
    mask = (thomson_time > t_min) & (thomson_time < t_max) #only want to return data within my chosen time range


    thomson_time = thomson_time[mask]
    te = te[:,mask]
    te_err = te_err[:,mask]
    ne = ne[:,mask]
    ne_err = ne_err[:,mask]
    rmid_array = rmid_array[:,mask]

    return thomson_time, ne, ne_err, te, te_err, rmid_array, r_array, z_array




def get_raw_core_Thomson_data(shot, t_min = None, t_max = None):
    '''
    INPUTS
    --------
    shot: C-Mod shot number (integer)
    t_min: minimum time in ms
    t_max: maximum time in ms


    RETURNS
    --------
    thomson_time: 1D array of time points at which Thomson data is available in ms
    ne: 2D array of electron density values at each radial value at each time point (m^-3)
    ne_err: 2D array of electron density errors at each radial value at each time point (m^-3)
    te: 2D array of electron temperature values at each radial value at each time point (eV)
    te_err: 2D array of electron temperature errors at each radial value at each time point (eV)
    rmid_array: 2D array of Rmid values at each radial value at each time point (m)
    r_array: single value of the real-space R value of the core Thomson system in m.
    z_array: 1D array of of the real-space z coordinates of the core Thomson system in m

    
    DESCRIPTION
    --------
    Just grabs the raw core Thomson data from the tree for a given time window.
    NOTE: this only grabs data from the NEW core system ( SHOT>1020000000). 
          Details on how to read in data from the old system can be found on the C-Mod wiki.

    '''
    tree = MDSplus.Tree('CMOD', shot)
    if shot > 1020000000: #get data from the new core system
        te = tree.getNode('\\TOP.ELECTRONS.YAG_NEW.RESULTS.PROFILES:TE_RZ').data() * 1000 # convert from keV to eV straight away
        te_err = tree.getNode('\\TOP.ELECTRONS.YAG_NEW.RESULTS.PROFILES:TE_ERR').data() * 1000 # convert from keV to eV straight away
        ne = tree.getNode('\\TOP.ELECTRONS.YAG_NEW.RESULTS.PROFILES:NE_RZ').data()
        ne_err = tree.getNode('\\TOP.ELECTRONS.YAG_NEW.RESULTS.PROFILES:NE_ERR').data()
        rmid_array = tree.getNode('\\TOP.ELECTRONS.YAG_NEW.RESULTS.PROFILES:R_MID_T').data()
        r_array = tree.getNode('\\TOP.ELECTRONS.YAG.RESULTS.PARAM:R').data()
        z_array = tree.getNode('\\TOP.ELECTRONS.YAG_NEW.RESULTS.PROFILES:Z_SORTED').data()
        thomson_time = tree.getNode('\\TOP.ELECTRONS.YAG_NEW.RESULTS.PROFILES:NE_RZ').dim_of().data()

    else: #get data from the old core system
        te = tree.getNode('\\TOP.ELECTRONS.YAG.RESULTS.GLOBAL.PROFILE:TE_RZ_T').data() * 1000 # convert from keV to eV straight away
        te_err = tree.getNode('\\TOP.ELECTRONS.YAG.RESULTS.GLOBAL.PROFILE:TE_ERR_ZT').data() * 1000 # convert from keV to eV straight away
        ne = tree.getNode('\\TOP.ELECTRONS.YAG.RESULTS.GLOBAL.PROFILE:NE_RZ_T').data()
        ne_err = tree.getNode('\\TOP.ELECTRONS.YAG.RESULTS.GLOBAL.PROFILE:NE_ERR_ZT').data()
        rmid_array = tree.getNode('\\TOP.ELECTRONS.YAG.RESULTS.GLOBAL.PROFILE:R_MID_T').data()
        r_array = tree.getNode('\\TOP.ELECTRONS.YAG.RESULTS.PARAM:R').data()
        z_array = tree.getNode('\\TOP.ELECTRONS.YAG.RESULTS.GLOBAL.PROFILE:Z_SORTED').data()
        thomson_time = tree.getNode('\\TOP.ELECTRONS.YAG.RESULTS.GLOBAL.PROFILE:NE_RZ_T').dim_of().data()

    thomson_time *= 1000 #conversion to ms
    thomson_time = np.round(thomson_time).astype(int) #convert to integer number of ms
    mask = (thomson_time > t_min) & (thomson_time < t_max) #only want to return data within my chosen time range

    thomson_time = thomson_time[mask]
    te = te[:,mask]
    te_err = te_err[:,mask]
    ne = ne[:,mask]
    ne_err = ne_err[:,mask]
    rmid_array = rmid_array[:,mask]

    return thomson_time, ne, ne_err, te, te_err, rmid_array, r_array, z_array



def get_P_ohmic(shot):
    ''' 
    INPUTS
    --------
    shot: C-Mod shot number (integer)


    RETURNS
    --------
    time: 1D array of time points at which the Ohmic power is available in seconds
    P_oh: 1D array of Ohmic power values at each time point in MW


    DESCRIPTION
    --------
    Calculates the Ohmic power in the plasma over time for a given shot.
    This is important for power balance in the 2-point model calculations.
    NOTE: times are in SECONDS NOT MILLISECONDS
    '''

    # psi at the edge:
    main_node = MDSplus.Tree('cmod', shot)

    ssibry_node = main_node.getNode('\\top.MHD.ANALYSIS.EFIT.RESULTS.G_EQDSK.SSIBRY')
    time = ssibry_node.dim_of(0) #in seconds
    ssibry = ssibry_node.data()
    
    # total voltage associated with magnetic flux inside LCFS
    vsurf = np.gradient(smooth(ssibry,5),time) * 2 * np.pi

    # calculated plasma current
    ip_node= main_node.getNode('\\top.MHD.ANALYSIS.EFIT.RESULTS.A_EQDSK:CPASMA')
    ip = np.abs(ip_node.data())

    # internal inductance
    li = main_node.getNode('\\top.MHD.ANALYSIS.EFIT.RESULTS.A_EQDSK:ali').data()

    R_cm = 67.0 # value chosen/fixed in scopes
    L = li*2.*np.pi*R_cm*1e-9  # total inductance (nH)
    
    vi = L * np.gradient(smooth(ip,2),time)   # induced voltage
    
    P_oh = ip * (vsurf - vi)/1e6 # P=IV   #MW
    return time, P_oh


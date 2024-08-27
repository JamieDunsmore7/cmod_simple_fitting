### Functions for implementing the 2-pt model ###


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

from functions.functions_profile_fitting import *
from functions.functions_fetching_raw_data import *


def apply_2pt_shift(xvalues, Te_values, sep_Te, old_sep, only_shift_edge = True):
        '''
        INPUTS
        --------
        xvalues: 1D array of x values (I think this can be in any coordinate system)
        Te_values: 1D array of Te values
        sep_Te: float, the separatrix Te in eV according to the 2-point model
        old_sep: float, the separatrix radius before the shift
        only_shift_edge: boolean, if True, only shift data points with psi > 0.8. 
                        NOTE: xvalues must be in psi coords if this is turned on.

        RETURNS
        --------
        shifted_x: 1D array of x values shifted such that the sepertarix Te agrees with the 2-point model
        shift: float, the amount by which the separatrix has been shifted

        DESCRIPTION
        --------
        Function applies a uniform shift to all the Thomson data points such that the separatrix Te agrees with the 2-point model prediction.
        '''

        for rad in range(len(xvalues)):
            if Te_values[rad] < sep_Te:
                len_along = (Te_values[rad-1] - sep_Te) / (Te_values[rad-1] - Te_values[rad])
                delta_R = xvalues[rad] - xvalues[rad-1]
                new_sep = xvalues[rad-1] + delta_R * len_along
                shift = old_sep - new_sep #this does make sense, because we want to shift the Thomson data rather than the sep position
                break
        
        if only_shift_edge == False:
            shifted_x = xvalues + shift
        else:
            xvalues_to_shift = xvalues[xvalues > 0.8]
            xvalues_to_keep = xvalues[xvalues <= 0.8]
            shifted_x = xvalues_to_shift + shift
            shifted_x = np.append(xvalues_to_keep, shifted_x)


        shifted_x = xvalues + shift
        return shifted_x, shift



def get_twopt_shift_from_edge_Te_fit(shot, time_in_s, edge_psi, edge_Te, edge_Te_err):
    '''
    INPUTS
    --------
    shot: C-Mod shot number (integer)
    time_in_s: time in seconds
    edge_psi: 1D array of psi values at the edge
    edge_Te: 1D array of Te values at the edge
    edge_Te_err: 1D array of Te errors at the edge

    RETURNS
    --------
    Te_sep_eV: float, the separatrix Te in eV according to the 2-point model
    shift: float, the amount by which the separatrix must be shifted to agree with the 2-point model

    DESCRIPTION
    --------
    1) Takes in raw Te data at a given time. 
    2) Tries to fit an mtanh with a linear inboard term and flat outboard term to the data.
    3) Uses the 2-point model to predict the separatrix Te.
    4) Shifts the Thomson data such that separatrix Te from the mtanh fit agrees with the 2-point model.

    NOTE: Remember that the shift and the separatrix Te are returned in psi coordinates.
    '''

    te_guesses = Osborne_linear_initial_guesses(edge_psi, edge_Te)

    # some initial te guesses. Hardcoded ones have been successful in the past.
    list_of_te_guesses = []
    list_of_te_guesses.append(te_guesses)
    list_of_te_guesses.append([ 9.92614859e-01,  4.01791101e-02,  2.55550908e+02,  1.28542623e+01,  2.17777084e-01])
    list_of_te_guesses.append([ 1.006,  2.01791101e-02,  150,  10,  0.35])
    list_of_te_guesses.append([ 1.006,  2.88458215e-03,  1.31618059e+02,  3.10797195e+01,  6.93728274e-02])


    num_gridpoints = 100

    #initial grid to place the fit on. I will shift this in line with 2-pt model later on
    generated_xvalues = np.linspace(min(edge_psi), max(edge_psi), num_gridpoints)

    for te_guess in list_of_te_guesses:
        try:
            te_params, te_covariance = curve_fit(Osborne_Tanh_linear, edge_psi, edge_Te, p0=te_guess, sigma=edge_Te_err, absolute_sigma=True, maxfev=2000, bounds=([0.85, 0, 0, -0.001, -np.inf], np.inf)) #should now be in psi
            te_fitted = Osborne_Tanh_linear(generated_xvalues, te_params[0], te_params[1], te_params[2], te_params[3], te_params[4])
            break #guess worked so exit the for loop
        except:
            if te_guess == list_of_te_guesses[-1]:
                # If all the guesses failed, set the fit parameters to none
                print('Could not fit Te profile to get Te_sep')
                # set all these values to none to throw an error later on in the code. This happens if all the attemps at fitting Te have failed.
                te_params = None
                te_covariance = None
                te_fitted = None
            else:
                # move onto the next guess
                continue


    geqdsk = 4 #unimportant placeholder
    lam_T_nl = 1 # unimportant placeholder

    Te_sep_eV = Teu_2pt_model(shot, time_in_s, lam_T_nl, geqdsk, pressure_opt = 3, lambdaq_opt=1)
    new_x, shift = apply_2pt_shift(generated_xvalues, te_fitted, Te_sep_eV, 1, only_shift_edge=False) #separatrix in psi coords is just 1

    print('Te_sep_eV', Te_sep_eV)
    print('shift', shift)
  

    return Te_sep_eV, shift



def Teu_2pt_model(shot,time,lam_T_mm, geqdsk, pressure_opt = 3, lambdaq_opt=1, rhop_vec=None, ne=None, Te=None):
    '''
    Get 2-point model prediction for Te at the LCFS.

    Parameters
    ----------
    shot : C-Mod shot number
    tmin, tmax : time window in seconds
    geqdsk : dictionary containing the processed geqdsk file
    pressure_opt: int, choice of method to get the volume-averaged plasma pressure used
    rhop_vec : 1D arr, sqrt(norm.pol.flux) vector for ne and Te profiles, only used if pressure_opt=2    
    ne : 1D arr, electron density profile in units of 1e20m^-3, only used if pressure_opt=2
    Te : 1D arr, electron temperature profile in units of keV, only used if pressure_opt=2

    Returns
    -------
    Tu_eV : 2-point model prediction for the electron temperature at the LCFS
    '''


    two_pt_tree = MDSplus.Tree('cmod', shot)

    P_rad_main = None

    
    try:
        P_rad_main_array = two_pt_tree.getNode('\\top.spectroscopy.bolometer:results:foil:main_power').data()
        P_rad_main_time_array = two_pt_tree.getNode('\\top.spectroscopy.bolometer:results:foil:main_power').dim_of().data()
        P_rad_main = np.interp(time, P_rad_main_time_array, P_rad_main_array) #in W
        P_rad_main /= 1e6 #in MW

    except:
        P_rad_main = None

    
    P_rad_diode_array = two_pt_tree.getNode('\\top.spectroscopy.bolometer:twopi_diode').data()
    P_rad_diode_time_array = two_pt_tree.getNode('\\top.spectroscopy.bolometer:twopi_diode').dim_of().data()
    P_rad_diode = np.interp(time, P_rad_diode_time_array, P_rad_diode_array)
    P_rad_diode  = P_rad_diode * 3 * 1000 #in W using same scaling as the scope. Don't know why there is a factor of 3.
    P_rad_diode /= 1e6 #in MW

    P_RF_array = two_pt_tree.getNode('\\top.RF.antenna.results:pwr_net_tot').data()
    P_RF_time_array = two_pt_tree.getNode('\\top.RF.antenna.results:pwr_net_tot').dim_of().data()
    P_RF = np.interp(time, P_RF_time_array, P_RF_array) #in MW


    P_oh_time_array,  P_oh_array = get_P_ohmic(shot)
    P_oh = np.interp(time, P_oh_time_array, P_oh_array) #in MW

    Wplasma_array = two_pt_tree.getNode('\\top.MHD.ANALYSIS.EFIT.RESULTS.A_EQDSK:WPLASM').data() / 1e6 #in MJ now
    Wplasma_time_array = two_pt_tree.getNode('\\top.MHD.ANALYSIS.EFIT.RESULTS.A_EQDSK:WPLASM').dim_of().data()
    dW_dt_array = np.gradient(Wplasma_array, Wplasma_time_array)
    dW_dt = np.interp(time, Wplasma_time_array, Wplasma_array)



    q95_array = two_pt_tree.getNode('\\top.MHD.ANALYSIS.EFIT.RESULTS.A_EQDSK:qpsib').data()
    q95_time_array = two_pt_tree.getNode('\\top.MHD.ANALYSIS.EFIT.RESULTS.A_EQDSK:qpsib').dim_of().data()
    q95 = np.interp(time, q95_time_array, q95_array)


    eff = 0.8 # value suggested by paper by Bonoli. This is the efficiency of the RF heating. NOTE: it may not always be 0.8
    P_rad = P_rad_main if P_rad_main is not None else P_rad_diode # sometimes P_rad main will be missing

    Psol = eff *P_RF + P_oh - dW_dt - P_rad

    if Psol<0:
        print('Inaccuracies in Psol determination do not allow a precise 2-point model prediction. Set Te_sep=60 eV')
        return 60.
        
    # Several options to find vol-average pressure for Brunner scaling:
    if pressure_opt==1:
        # Use W_mhd rather than plasma pressure:
        vol =  geqdsk['fluxSurfaces']['geo']['vol']
        W_mhd_array = two_pt_tree.getNode('\\top.MHD.EFIT.RESULTS.A_EQDSK:wplasm').data()
        W_mhd_time_array = two_pt_tree.getNode('\\top.MHD.EFIT.RESULTS.A_EQDSK:wplasm').dim_of().data()
        W_mhd = np.interp(time, W_mhd_time_array, W_mhd_array)
        p_Pa_vol_avg = 2./3. * W_mhd / vol[-1]


    elif pressure_opt==2:
        raise ValueError('Pressure Option 2 is deprecated. Original code relied on Aurora. Choose another option.')
        # find volume average within LCFS from ne and Te profiles
        # OLD CODE IS HERE...
        # p_Pa = 2 * (ne*1e20) * (Te*1e3*q_electron)  # taking pe=pi
        # indLCFS = np.argmin(np.abs(rhop_vec-1.0))
        # p_Pa_vol_avg = aurora.vol_average(p_Pa[:indLCFS], rhop_vec[:indLCFS], geqdsk=geqdsk)[-1]

    elif pressure_opt==3:  # default for C-Mod, since Brunner himself used this formula
        # use tor beta to extract vol-average pressure
        BTaxis_array = two_pt_tree.getNode('\\top.MHD.magnetics.diamag_coils:btor').data()
        BTaxis_time_array = two_pt_tree.getNode('\\top.MHD.magnetics.diamag_coils:btor').dim_of().data()
        BTaxis = np.interp(time, BTaxis_time_array, BTaxis_array)

        betat_array = two_pt_tree.getNode('\\top.MHD.ANALYSIS.EFIT.RESULTS.A_EQDSK:betat').data()
        betat_time_array = two_pt_tree.getNode('\\top.MHD.ANALYSIS.EFIT.RESULTS.A_EQDSK:betat').dim_of().data()
        betat = np.interp(time, betat_time_array, betat_array)
        p_Pa_vol_avg = (betat/100)*BTaxis**2.0/(2.0*4.0*np.pi*1e-7)   # formula used by D.Brunner

    else:
        raise ValueError('Undefined option for volume-averaged pressure')



    e = eqtools.CModEFIT.CModEFITTree(shot, tree='EFIT20')
    Rlcfs = e.rho2rho('psinorm', 'Rmid', 1, time)

    Bt = np.abs(e.rz2BT(Rlcfs, 0, time))
    Bp = np.abs(e.rz2BZ(Rlcfs, 0, time))  #we can use this expression because at the midplane separatrix Bz IS Bp.


    if lambdaq_opt==1:
        # now get 2-point model prediction
        lam_q_mm, Tu_eV = two_point_model(
            0.69, 0.22,
            Psol, Bp, Bt, q95, p_Pa_vol_avg,
            1.0,  # dummy value of ne at the sepatrix, not used for Tu_eV calculation
            lam_T_mm, lam_q_model='Brunner'
        )

    elif lambdaq_opt==2:
        # use lambdaT calculated from TS points to get lambda_q using lambda_q = 2/7*lambda_T
        lam_q_mm, Tu_eV = two_point_model(
            0.69, 0.22,
            Psol, Bp, Bt, q95, p_Pa_vol_avg,
            1.0, 
            lam_T_mm, lam_q_model='lam_T'
        )

    return Tu_eV



def two_point_model(R0_m, a0_m, P_sol_MW, B_p, B_t, q95, p_Pa_vol_avg, nu_m3, lam_T_mm, lam_q_model='Brunner'):
    '''
    2-point model results, all using SI units in outputs (inputs have other units as indicated)
    Refs: 
    - H J Sun et al 2017 Plasma Phys. Control. Fusion 59 105010 
    - Eich NF 2013
    - A. Kuang PhD thesis

    Parameters
    ----------
    R0_m : float, major radius on axis
    a0_m : float, minor radius
    P_sol_MW : float, power going into the SOL, in MW.
    B_p : float, poloidal field in T
    B_t : float, toroidal field in T
    q95 : float
    p_Pa_vol_avg : float, pressure in Pa units to use for the Brunner scaling.
    nu_m3 : float, upstream density in [m^-3], i.e. ne_sep.
    '''

    R_lcfs = R0_m+a0_m
    eps = a0_m/R0_m
    L_par = np.pi *R_lcfs * q95

    # coefficients for heat conduction by electrons or H ions
    k0_e = 2000.  # W m^{-1} eV^{7/2}
    k0_i = 60.    # W m^{-1} eV^{7/2}
    gamma = 7  # sheat heat flux transmission coeff (Stangeby tutorial)

    # lam_q in mm units from Eich NF 2013
    lam_q_mm_eich = 1.35 * P_sol_MW**(-0.02)* R_lcfs**(0.04)* B_p**(-0.92) * eps**0.42

    # lam_q in mm units from Brunner NF 2018
    Cf = 0.08
    lam_q_mm_brunner = (Cf/p_Pa_vol_avg)**0.5 *1e3

    if lam_q_model == 'Brunner':
        lam_q_mm = lam_q_mm_brunner

    elif lam_q_model == 'Eich':
        lam_q_mm = lam_q_mm_eich

    elif lam_q_model == 'lam_T':
        lam_q_mm = 2/7*lam_T_mm # from T. Eich 2021 NF
    else:
        raise ValueError('Undefined option for lambda_q model')
    
    # Parallel heat flux in MW/m^2.
    # Assumes all Psol via the outboard midplane, half transported by electrons (hence the factor of 0.5 in the front).
    # See Eq.(B.2) in Adam Kuang's thesis (Appendix B).
    q_par_MW_m2 = 0.5*P_sol_MW / (2.*np.pi*R_lcfs* (lam_q_mm*1e-3))*\
                        np.hypot(B_t,B_p)/B_p

    # Upstream temperature (LCFS) in eV. See Eq.(B.2) in Adam Kuang's thesis (Appendix B).
    # Note: k0_i gives much larger Tu than k0_e (the latter is right because electrons conduct)
    Tu_eV = ((7./2.) * (q_par_MW_m2*1e6) * L_par/(2.*k0_e))**(2./7.)

    # Upsteam temperature (LCFS) in K
    Tu_K = Tu_eV * q_electron/kB

    # downstream density (rough...) - source?
    nt_m3= (nu_m3**3/((q_par_MW_m2*1e6)**2)) *\
                (7.*q_par_MW_m2*L_par/(2.*k0_e))**(6./7.)*\
                (gamma**2 * q_electron**2)/(4.*(2.*m_p))

    #print('lambda_{q} (mm)'+' = {:.2f}'.format(lam_q_mm))
    #print('T_{e,sep} (eV)'+' = {:.1f}'.format(Tu_eV))

    return lam_q_mm, Tu_eV

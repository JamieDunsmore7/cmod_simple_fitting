# A script for doing 2D fits of the Thomson evolution
# Plan is:
# 1. Fit Thomson profile at each time point in the window of interest (in psi space)
# 2. Shift the raw data in the edge accordingly
# 3. Use these as the raw data-points for the 2D fit. Don't try and do a 2-pt shift at every time point
#    (since power calculations for the 2-pt model rely on EFIT interpolation anyway so these may be uncertain)
# 4. Apply some smoothing weights to the raw data


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



### FUNCTIONS FOR ADJUSTING THOMSON SCATTERING DATA BASED ON TCI DATA ###
### Originally written by M.A. Miller, modified by J. Dunsmore        ###

def compare_ts_tci(shot, tmin, tmax, nl_num=4):
    '''
    Returns the line-integrated density from the synthetic TCI diagnostic (created from the TS data) and the actual TCI diagnostic.
    Returns separate arrays for each laser.
    Values returned on the Thomson timebase.
    '''

    nl_ts1 = 1e32
    nl_ts2 = 1e32
    nl_tci1 = 1e32
    nl_tci2 = 1e32
    ts_time1 = 1e32
    ts_time2 = 1e32


    shot_str = str(shot)
    ch_str = '0' + str(nl_num) if nl_num < 10 else str(nl_num)
    tci_node = '.tci.results:nl_{}'.format(ch_str)

    #ts_time = electrons.getNode('\\electrons::top.yag_new.results.profiles:ne_rz').dim_of(0).data()

    electrons = MDSplus.Tree('electrons', shot)

    try:
        ts_time = electrons.getNode('\\electrons::top.yag_new.results.profiles:ne_rz').dim_of(0).data()
        # ts_time = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag_new.results.profiles:ne_rz').dim_of(0)
        _ = len(ts_time) # this is just to test if it's empty 
    except:
        ts_time = electrons.getNode('\\electrons::top.yag.results.global.profile:ne_rz_t').dim_of(0).data()
        # ts_time = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag.results.global.profile:ne_rz_t').dim_of(0)


    #ts_time = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag_new.results.profiles:ne_rz').dim_of(0)
        

    tci = electrons.getNode('\\electrons::top{}'.format(tci_node)).data()
    tci_t = electrons.getNode('\\electrons::top{}'.format(tci_node)).dim_of(0).data()
    # tci = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top{}'.format(tci_node)).data()
    # tci_t = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top{}'.format(tci_node)).dim_of(0)
    

    nl_ts, nl_ts_t = integrate_ts2tci(shot, tmin, tmax, nl_num=nl_num)

    t0, t1 = np.min(nl_ts_t), np.max(nl_ts_t)

    n_yag1, n_yag2, indices1, indices2 = parse_yags(shot)

    # laser 1
    if n_yag1 > 0:

        ts_time1 = ts_time[indices1]
        ind = (ts_time1 >=  t0) & (ts_time1 <= t1)
        cnt = np.sum(ind)

        if cnt > 0:

            nl_tci1 = interp1d(tci_t, tci)(ts_time1[ind])
            nl_ts1 = interp1d(nl_ts_t, nl_ts)(ts_time1[ind])
            time1 = ts_time1[ind]

    else:

        time1 = -1

    # laser 2
    if n_yag2 > 0:

        ts_time2 = ts_time[indices2]
        ind = (ts_time2 >=  t0) & (ts_time2 <= t1)
        cnt = np.sum(ind)
        if cnt > 0:
            nl_tci2 = interp1d(tci_t, tci)(ts_time2[ind])
            nl_ts2 = interp1d(nl_ts_t, nl_ts)(ts_time2[ind])
            time2 = ts_time2[ind]

    else:

        time2 = -1


    return nl_ts1, nl_ts2, nl_tci1, nl_tci2, time1, time2


def integrate_ts2tci(shot, tmin, tmax, nl_num=4):
    '''
    Integrate the synthetic TCI data (from the TS data) along the chord to get a line-integrated density.
    '''

    shot_str = str(shot)

    t, z, n_e, n_e_sig = map_ts2tci(shot, tmin, tmax, nl_num=nl_num)

    n_ts = len(t)
    nl_ts_t = t
    nl_ts = np.zeros(n_ts)

    ### perform integral

    from scipy.integrate import cumtrapz

    # define this silly idl function
    def idl_uniq(x):

        y_inds = []
        for x_ind in range(len(x)):
            y_inds.append(True)
            if x[x_ind] == x[x_ind-1]:
                y_inds[-2] = False

        return np.array(y_inds)


    for i in range(n_ts):

        n_e_sig[i,:] = np.select([n_e_sig[i,:] == 0], [np.nan], n_e_sig[i,:]) # replaces 0s with nans so no error is thrown

        ind = (np.abs(z[i,:]) < 0.5) & (n_e[i,:] > 0) & (n_e[i,:] < 1e21) & (n_e[i,:]/n_e_sig[i,:] > 2)
        cnt = np.sum(ind)

        if cnt < 3:
            nl_ts[i] = 0

        else:
            x = z[i,ind]
            y = n_e[i,ind]
            ind_uniq = idl_uniq(x) # this function returns the indices that have unique adjacent indices, but will only choose the 2nd adjacent index to return to you for some reason
    
            x = x[ind_uniq]
            y = y[ind_uniq]

            nl_ts[i] = cumtrapz(y,x)[-1]

    return nl_ts, nl_ts_t


def map_ts2tci(shot, tmin, tmax, nl_num=4):
    '''
    Maps the Thomson data to one of the TCI interferometer chords (specified in the nl_num argument)
    nl04 is used because it is the most reliable (it's used for the feedback control).
    If nl04 isn't working, there won't be a proper shot.
    '''

    shot_str = str(shot)


    analysis = MDSplus.Tree('analysis', shot)
    psi_a_t = analysis.getNode('\\efit_aeqdsk:sibdry').dim_of(0).data()

    #psi_a = OMFITmdsValue(server='CMOD', shot=shot, treename='analysis', TDI='\\EFIT_AEQDSK:sibdry').data()
    #psi_a_t = OMFITmdsValue(server='CMOD', shot=shot, treename='analysis', TDI='\\EFIT_AEQDSK:sibdry').dim_of(0)
    #psi_0 = OMFITmdsValue(server='CMOD', shot=shot, treename='analysis', TDI='\\EFIT_AEQDSK:sigmax').data()

    t1, t2 = np.min(psi_a_t), np.max(psi_a_t) # in JWH's script, it uses efit_times from efit_check, 
                                            # but I'm not sure how to get that so hopefully this is good enough

    ### get thomson data

    # core

    electrons = MDSplus.Tree('electrons', shot)
    try: # needs to be shot > ~1020900000 for this one to work

        t_ts = electrons.getNode('\\electrons::top.yag_new.results.profiles:ne_rz').dim_of(0).data()
        ne_ts_core = electrons.getNode('\\electrons::top.yag_new.results.profiles:ne_rz').data()
        ne_ts_core_err = electrons.getNode('\\electrons::top.yag_new.results.profiles:ne_err').data()
        z_ts_core = electrons.getNode('\\electrons::top.yag_new.results.profiles:z_sorted').data()
        # t_ts = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag_new.results.profiles:ne_rz').dim_of(0)
        # ne_ts_core = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag_new.results.profiles:ne_rz').data()
        # ne_ts_core_err = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag_new.results.profiles:ne_err').data()
        # z_ts_core = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag_new.results.profiles:z_sorted').data()
        m_ts_core = len(z_ts_core)

    except: # if not, old core TS system

        t_ts = electrons.getNode('\\electrons::top.yag.results.global.profile:ne_rz_t').dim_of(0).data()
        ne_ts_core = electrons.getNode('\\electrons::top.yag.results.global.profile:ne_rz_t').data()
        ne_ts_core_err = electrons.getNode('\\electrons::top.yag.results.global.profile:ne_err_zt').data()
        z_ts_core = electrons.getNode('\\electrons::top.yag.results.global.profile:z_sorted').data()
        # t_ts = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag.results.global.profile:ne_rz_t').dim_of(0)
        # ne_ts_core = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag.results.global.profile:ne_rz_t').data()
        # ne_ts_core_err = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag.results.global.profile:ne_err_zt').data()
        # z_ts_core = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag.results.global.profile:z_sorted').data()
        m_ts_core = len(z_ts_core)

    # edge

    ne_ts_edge = electrons.getNode('\\electrons::top.yag_edgets.results:ne').data()
    ne_ts_edge_err = electrons.getNode('\\electrons::top.yag_edgets.results:ne:error').data()
    # ne_ts_edge = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag_edgets.results:ne').data()
    # ne_ts_edge_err = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag_edgets.results:ne:error').data()

    z_ts_edge = electrons.getNode('\\electrons::top.yag_edgets.data:fiber_z').data()
    # z_ts_edge = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag_edgets.data:fiber_z').data()
    m_ts_edge = len(z_ts_edge)


    m_ts = m_ts_core + m_ts_edge

    r_ts = electrons.getNode('\\electrons::top.yag.results.param:r').data()
    r_tci = electrons.getNode('\\electrons::top.tci.results:rad').data()
    # r_ts = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag.results.param:r').data()
    # r_tci = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.tci.results:rad').data()



    ### compute mapping
 
    n_ts = len(t_ts) #number of time points
    z_ts = np.zeros(m_ts) #number of spatial points

    z_ts[:m_ts_core] = z_ts_core
    z_ts[m_ts_core:] = z_ts_edge

    ne_ts = np.zeros((n_ts, m_ts))
    ne_ts_err = np.zeros((n_ts, m_ts))

    ne_ts[:,:m_ts_core] = np.transpose(ne_ts_core)
    ne_ts_err[:,:m_ts_core] = np.transpose(ne_ts_core_err)
    ne_ts[:,m_ts_core:] = np.transpose(ne_ts_edge)
    ne_ts_err[:,m_ts_core:] = np.transpose(ne_ts_edge_err)

    ind = (t_ts > t1) & (t_ts < t2)
    n_ts = np.sum(ind)
    t_ts = t_ts[ind]
    ne_ts = ne_ts[ind,:]   # now we have 2D array of ne and ne_err as a function of time and z.
    ne_ts_err = ne_ts_err[ind,:]


    #gets the magnetic equilibrium at every time point - note EFIT20 is the TS timebase
    e = eqtools.CModEFIT.CModEFITTree(int(shot), tree='EFIT20')

    r_ts = np.full(len(z_ts), r_ts) #just to make r_ts the same length as z_ts

    # map the TS data from RZ space to psi space
    psin_ts = e.rho2rho('RZ', 'psinorm', r_ts, z_ts, t=t_ts, each_t = True)


    #define some points along the interferometer chord of choice (to map the Thomson onto)
    m_tci = 501
    z_tci = -0.4 + 0.8*np.linspace(1,m_tci-1,m_tci)/float(m_tci - 1)
    r_tci = r_tci[nl_num-1] + np.zeros(m_tci)


    # map the TCI data from RZ space to psi space
    psin_tci = e.rho2rho('RZ', 'psinorm', r_tci, z_tci, t=t_ts, each_t = True)

    # perform mapping

    z_mapped = np.zeros((n_ts, int(2*m_ts)))
    ne_mapped = np.zeros_like(z_mapped)
    ne_mapped_err = np.zeros_like(z_mapped)


    # This loop cycles through every time point and maps the TS data to the pre-defined TCI chord points (z_tci)
    # Note that we have the TS points in terms of psi, so there are two z values which correspond to the each TS psi value (upper and lower).
    for i in range(n_ts):

        psin_min = np.min(psin_tci[i,:]) # think we want to find where psi crosses zero - i.e where we start going outwards again rather than inwards
        i_min = np.argmin(psin_tci[i,:])


        for j in range(m_ts):

            try:
                a1 = interp1d(psin_tci[i,:i_min+1], z_tci[:i_min+1])(psin_ts[i,j])
                a2 = interp1d(psin_tci[i,i_min+1:], z_tci[i_min+1:])(psin_ts[i,j])
            except:
                a1 = np.nan #I don't trust the extrapolation so just set to nan if the interpolation fails
                a2 = np.nan #I don't trust the extrapolation so just set to nan if the interpolation fails
                


            z_mapped[i,[j,j+m_ts]] = [a1,a2]
            ne_mapped[i,[j,j+m_ts]] = ne_ts[i,j]
            ne_mapped_err[i,[j,j+m_ts]] = ne_ts_err[i,j]

        # sort before returning
        ind = np.argsort(z_mapped[i,:])
        z_mapped[i,:] = z_mapped[i,ind]
        ne_mapped[i,:] = ne_mapped[i,ind]
        ne_mapped_err[i,:] = ne_mapped_err[i,ind]

    z = z_mapped
    n_e = ne_mapped
    n_e_sig = ne_mapped_err
    t = t_ts

    # return the times, the z values of the synthetic TCI and the associated ne values at each t and z.
    return t, z, n_e, n_e_sig


def parse_yags(shot):
    '''
    Function to return the time indices of when the two lasers fire.
    '''


    #n_yag1 = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\knobs:pulses_q').data()[0]
    #n_yag2 = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\knobs:pulses_q_2').data()[0]

    #dark = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\n_dark_prior').data()[0]
    #n_total = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\n_total').data()[0]

    electrons = MDSplus.Tree('electrons', shot)
    try:
        n_yag1 = electrons.getNode('\\knobs:pulses_q').data() #this option can also return n_yag1 = 0 if the laser isn't working
    except:
        n_yag1 = 0
        print('Laser 1 may not be working')
    try:
        n_yag2 = electrons.getNode('\\knobs:pulses_q_2').data()
    except:
        n_yag2 = 0
        print('Laser 2 may not be working')

    dark = electrons.getNode('\\n_dark_prior').data()
    n_total = electrons.getNode('\\n_total').data()
    n_t = n_total - dark



    if n_yag1 == 0:
        if n_yag2 == 0:
            indices1 = -1
            indices2 = -1
        else:
            indices1 = -1
            indices2 = np.linspace(0,n_yag2-1,n_yag2)
    else:
        if n_yag2 == 0:
            indices1 = np.linspace(0,n_yag1-1,n_yag1)
            indices2 = -1
        else:
            if n_yag1 == n_yag2:
                indices1 = 2*np.linspace(0,n_yag1-1,n_yag1)
                indices2 = indices1 + 1
            else:
                if n_yag1 < n_yag2:
                    indices1 = 2*np.linspace(0,n_yag1-1,n_yag1)
                    indices2 = np.hstack((2*np.linspace(0,n_yag1-1,n_yag1)+1,2*n_yag1 + np.linspace(0,n_yag2-n_yag1-1,n_yag2-n_yag1)))
                else:
                    indices2 = 2*np.linspace(0,n_yag2-1,n_yag2)
                    indices1 = np.hstack((2*np.linspace(0,n_yag2-1,n_yag2),2*n_yag2 + np.linspace(0,n_yag1-n_yag2-1,n_yag1-n_yag2)))

    ind = indices1 < n_t
    cnt = np.sum(ind)
    indices1 = indices1[ind] if (n_yag1 > 0) & (cnt > 0) else -1
    
    ind = indices2 < n_t
    cnt = np.sum(ind)
    indices2 = indices2[ind] if (n_yag2 > 0) & (cnt > 0) else -1

    indices1, indices2 = np.array(indices1).astype(int), np.array(indices2).astype(int) # These are indices in TIME not space. The lasers fire alternately.



    return n_yag1, n_yag2, indices1, indices2


def get_ts_tci_ratio(shot, tmin, tmax, nl_num = 4):
    '''
    Compares the actual TCI data over the whole shot to the synthetic TCI data (created from the TS data over the same time window)
    Calculates multiplicative factors to be applied to the TS data such that it matches the TCI data.
    '''

    print('Scaling raw Thomson to TCI data...')

    nl_ts1, nl_ts2, nl_tci1, nl_tci2, time1, time2 = compare_ts_tci(shot, tmin, tmax, nl_num=nl_num)

    # Occasionally, one of the elements in nl_ts1 or nl_ts2 can be zero. To avoid division by zero errors, set them to nan. 
    # They will also be ignored in the mean calculations by default.


    nl_ts1 = np.where(nl_ts1 == 0, np.nan, nl_ts1)
    nl_ts2 = np.where(nl_ts2 == 0, np.nan, nl_ts2)

    ratio_laser1 = nl_tci1/nl_ts1
    ratio_laser2 = nl_tci2/nl_ts2
    
    in_window1 = (time1 > tmin) & (time1 < tmax)
    in_window2 = (time2 > tmin) & (time2 < tmax)

    if np.sum(in_window1) > 0:
        mult_factor1 = np.mean(ratio_laser1[in_window1])
    else:
        mult_factor1 = 0
        print('No ratio computed for laser1')
    if np.sum(in_window2) > 0:
        mult_factor2 = np.mean(ratio_laser2[in_window2])
    else:
        mult_factor2 = 0
        print('No ratio computed for laser2')

    mult_factor = (mult_factor1 + mult_factor2)/2

    # return a multiplicative factor for each laser, and for the average of the two.
    print('Multiplicative factor for laser 1: {}'.format(mult_factor1))
    print('Multiplicative factor for laser 2: {}'.format(mult_factor2))
    return mult_factor1, mult_factor2, mult_factor


def scale_core_Thomson(shot, core_time_ms, core_ne):
    '''
    Scales the core Thomson to the TCI data for a given shot.
    core_time_ms and core_ne are arrays. Time should be in ms.
    NOTE: the scaling is currently hardcoded to occur between 500ms and 1500ms. This may need to be changed for shots of different lengths.
    TODO: Adjust the scaling to be based on a more flexible metric.
    '''
    
    # get the scaling factor for the two lasers based on TCI measurements
    mult_factor1, mult_factor2, mult_factor = get_ts_tci_ratio(shot, 0.5, 1.5) # may want to change this. TODO: be able to vary the time window

    # get the times in ms for the two lasers
    n_yag1, n_yag2, indices1, indices2 = parse_yags(shot)

    tree = MDSplus.Tree('CMOD', shot)

    ts_time = tree.getNode('\\TOP.ELECTRONS.YAG_NEW.RESULTS.PROFILES:NE_RZ').dim_of().data()

    print('ts time', ts_time)

    print('indices1')
    print(indices1)
    print('indices2')
    print(indices2)

    laser_1 = ts_time[indices1]
    print('laser 1', laser_1)
    laser_1 *= 1000 #conversion to ms
    laser_1 = np.round(laser_1).astype(int) #convert to integer number of ms

    laser_2 = ts_time[indices2]
    laser_2 *= 1000 #conversion to ms
    laser_2 = np.round(laser_2).astype(int) #convert to integer number of ms


    # cycle through input times, check which laser this input time corresponds to, and scale the ne accordingly
    for idx, t in enumerate(core_time_ms):
        print('core time', core_time_ms)
        if t in laser_1:
            core_ne[:,idx] *= mult_factor1
        elif t in laser_2:
            core_ne[:,idx] *= mult_factor2
        else:
            raise ValueError('Could not find a matching time to scale the Thomson Data!' + # just to catch any bugs/weird cases I haven't thought about
                             'Time is {}. '.format(t) + 
                             'Laser 1 time is {}. '.format(laser_1) + 
                             'laser 2 time is {}. '.format(laser_2))
        
    return core_ne




####### FUNCTIONS FOR MAPPING BETWEEN COORDINATE SYSTEMS #######

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


    e = eqtools.CModEFIT.CModEFITTree(shot, tree='EFIT20')
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

### FUNCTIONS FOR FETCHING RAW DATA FROM THE TREE ###



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




def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


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




### FUNCTIONS FOR CLEANING RAW THOMSON DATA ###

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




### 2-PT MODEL FUNCTIONS ###


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


def chi_squared(ydata, yfit, yerr):
    return np.sum((ydata - yfit)**2 / yerr**2)


def reduced_chi_squared(ydata, yfit, yerr, num_params):
    return chi_squared(ydata, yfit, yerr) / (len(ydata) - num_params)


def reduced_chi_squared_inside_separatrix(psi_values, ydata, yfit, yerr, num_params):
    inside_separatrix_mask = psi_values < 1.0
    ydata = ydata[inside_separatrix_mask]
    yfit = yfit[inside_separatrix_mask]
    yerr = yerr[inside_separatrix_mask]
    return chi_squared(ydata, yfit, yerr) / (len(ydata) - num_params)




##### TIME EVOLUTION OF FITS #####
##################################


def evolve_fits_by_radius_example_for_panel_plots(times, psi_grid, yvalues):
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
            ax.plot(times, np.polyval(quadratic_fit, times), color='r')

        else:
            ax.axis('off')  # Turn off axes if there are more subplots than new_psi_grid points
    
    plt.tight_layout()
    plt.show()




def master_fit_ne_Te_1D(shot, t_min=0, t_max=5000, scale_core_TS_to_TCI = False, set_Te_floor_to_20eV = True, add_articifial_core_TS_error_where_missing = True, 
                        remove_zeros_before_fitting = True, add_zero_in_SOL = True, shift_to_2pt_model=False, plot_the_fits = False,
                        return_processed_raw_data = False):
    '''
    INPUTS
    --------
    shot: integer, C-Mod shot number
    t_min: float, minimum time in ms
    t_max: float, maximum time in ms

    scale_core_TS_to_TCI: boolean, if True, the core Thomson data is scaled to match the interferometry data over the course of the shot.
    set_Te_floor_to_20eV: boolean, if True, the Te floor is set to 20eV if the fit goes below this value.
    add_articifial_core_TS_error_where_missing: boolean, if True, the error bars in the core are set to 10% if they don't exist yet (this is because quite often the error bars have been set to zero in the core, messing up the fits)
    remove_zeros_before_fitting: boolean, if True, zeros are removed from the data before fitting. Deafult is to only remove zeros < psi = 1, but this can be modified.
    add_zero_in_SOL: boolean, if True, a zero is added at the SOL edge to help the mtanh fit (hardcoded at psi=1.05).
    shift_to_2pt_model: boolean, if True, the Thomson data and fits are shifted post-fit to align the separatrix Te with the 2-point model prediction.
    plot_the_fits: boolean, option to plot the fits at each Thomson time point.
    return_processed_raw_data: boolean, if True, the processed raw data is returned as well as the fits. Processing involves adding/removing zeros, as well as shifting according to 2-pt model.

    
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
                print('No Edge Thomson data at this time point. Skipping.')
                continue

            if add_articifial_core_TS_error_where_missing == True:
                raw_ne_err_core[raw_ne_err_core == 0] = raw_ne_core[raw_ne_err_core==0]*0.1 # set error bars in the core to 10% if they don't exist yet
                raw_te_err_core[raw_te_err_core == 0] = raw_te_core[raw_te_err_core==0]*0.1 # set error bars in the core to 10% if they don't exist yet


            #Switch from Rmid to psi coordinates here using eqtools
            raw_psi_edge = e.rho2rho('Rmid', 'psinorm', raw_rmid_edge, time_in_s)
            raw_psi_core = e.rho2rho('Rmid', 'psinorm', raw_rmid_core, time_in_s)

            # Option to remove zeros from the data for fitting.
            if remove_zeros_before_fitting == True:
                # edge
                raw_te_psi_edge, raw_te_edge, raw_te_err_edge = remove_zeros(raw_psi_edge, raw_te_edge, raw_te_err_edge, core_only=True)
                raw_ne_psi_edge, raw_ne_edge, raw_ne_err_edge = remove_zeros(raw_psi_edge, raw_ne_edge, raw_ne_err_edge, core_only=True)

                # core
                raw_te_psi_core, raw_te_core, raw_te_err_core = remove_zeros(raw_psi_core, raw_te_core, raw_te_err_core, core_only=True)
                raw_ne_psi_core, raw_ne_core, raw_ne_err_core = remove_zeros(raw_psi_core, raw_ne_core, raw_ne_err_core, core_only=True)

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
                print('No good data for this time point. Skipping.')
                continue

            if len(total_psi_te[total_psi_te<0.8]) < 3:
                print('Fewer than 3 TS points below psi = 0.8, so do not try to fit the profile. Skipping')
                continue

            if max(total_te) < 100 or max(total_ne) < 1e19:
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
                    te_chi_squared = reduced_chi_squared_inside_separatrix(total_psi_te, total_te, te_fitted_for_chi_squared, total_te_err, len(te_params))
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
                print('Te mtanh fit failed.')

            # Do the cubic fits
            te_params_cubic, te_covariance_cubic = curve_fit(Cubic, total_psi_te, total_te, sigma=total_te_err, absolute_sigma=True, maxfev=2000)
            te_fitted_cubic = Cubic(generated_psi_grid, te_params_cubic[0], te_params_cubic[1], te_params_cubic[2], te_params_cubic[3])

            te_fitted_cubic_for_chi_squared = Cubic(total_psi_te, te_params_cubic[0], te_params_cubic[1], te_params_cubic[2], te_params_cubic[3])
            te_chi_squared_cubic = reduced_chi_squared_inside_separatrix(total_psi_te, total_te, te_fitted_cubic_for_chi_squared, total_te_err, len(te_params_cubic))

            # print the reduced chi-squared values of the respective fits
            print(f'te reduced chi squared cubic: {te_chi_squared_cubic:.2f}')
            if te_params is not None:
                print(f'te reduced chi squared mtanh: {te_chi_squared:.2f}')

            '''
            # Do not use the mtanh fit if there are fewer than 3 points in the pedestal region.
            if te_params is not None:
                pedestal_start = te_params[0] - te_params[1]
                pedestal_end = te_params[0] + te_params[1]
                pedestal_mask = (total_psi_te > pedestal_start) & (total_psi_te < pedestal_end)
                if len(total_psi_te[pedestal_mask]) < 3:
                    print(f'Reject Te mtanh fit because there are only {len(total_psi_te[pedestal_mask])} points in the pedestal (require at least 3)')
                    te_params = None
                    te_covariance = None
                    te_fitted = None
            '''
            

            # Do not use the mtanh fit if the reduced chi-squared is below 0 or above 20.
            if te_params is not None:
                if te_chi_squared <= 0 or te_chi_squared > 20:
                    print('Reject Te mtanh fit because the reduced chi squared is below 0 or greater than 20.')
                    te_params = None
                    te_covariance = None
                    te_fitted = None

            # Do not use the cubic fit if the reduced chi-squared is below 0 or above 20.
            if te_params_cubic is not None:
                if te_chi_squared_cubic <= 0 or te_chi_squared_cubic > 20:
                    print('Reject Te cubic fit because the reduced chi squared is below 0 or greater than 20.')
                    te_params_cubic = None
                    te_covariance_cubic = None
                    te_fitted_cubic = None

            # Choose the best fit based on the reduced chi-squared values.
            if te_params is not None and te_params_cubic is not None:
                if te_chi_squared_cubic < te_chi_squared:
                    print('CUBIC IS BEST FIT')
                    te_fitted_best = te_fitted_cubic
                    te_chi_squared_best = te_chi_squared_cubic
                    te_best_fit_type = 'cubic'
                    number_of_te_cubic_fits += 1
                else:
                    print('MTANH IS BEST FIT')
                    te_fitted_best = te_fitted
                    te_chi_squared_best = te_chi_squared
                    te_best_fit_type = 'mtanh'
                    number_of_te_mtanh_fits += 1
            elif te_params is not None:
                print('MTANH IS BEST FIT')
                te_fitted_best = te_fitted
                te_chi_squared_best = te_chi_squared
                te_best_fit_type = 'mtanh'
                number_of_te_mtanh_fits += 1
            elif te_params_cubic is not None:
                print('CUBIC IS BEST FIT')
                te_fitted_best = te_fitted_cubic
                te_chi_squared_best = te_chi_squared_cubic
                te_best_fit_type = 'cubic'
                number_of_te_cubic_fits += 1
            else:
                print('NO FITS WORKED')
                te_fitted_best = None
                te_chi_squared_best = None
                te_best_fit_type = None
                number_of_te_failed_fits += 1


            # Te measurement floor is 20eV so set this as a minimum for the fitted profiles
            # Highly recommended to keep this flag as True since fits can go negative otherwise.
            if set_Te_floor_to_20eV == True:
                if te_fitted_best is not None:
                    te_fitted_best[te_fitted_best < 20] = 20
               
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
                    ne_chi_squared = reduced_chi_squared_inside_separatrix(total_psi_ne, total_ne, ne_fitted_for_chi_squared, total_ne_err, len(ne_params))
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

            
            if ne_params is None:
                print('Ne mtanh fit failed.')


            # Do the cubic fits
            ne_params_cubic, ne_covariance_cubic = curve_fit(Cubic, total_psi_ne, total_ne/1e20, sigma=total_ne_err/1e20, absolute_sigma=True, maxfev=2000)
            ne_fitted_cubic = 1e20*Cubic(generated_psi_grid, ne_params_cubic[0], ne_params_cubic[1], ne_params_cubic[2], ne_params_cubic[3])
            
            ne_fitted_cubic_for_chi_squared = 1e20*Cubic(total_psi_ne, ne_params_cubic[0], ne_params_cubic[1], ne_params_cubic[2], ne_params_cubic[3])
            ne_chi_squared_cubic = reduced_chi_squared_inside_separatrix(total_psi_ne, total_ne, ne_fitted_cubic_for_chi_squared, total_ne_err, len(ne_params_cubic))

            # print the reduced chi-squared values of the respective fits
            print(f'ne reduced chi squared cubic: {ne_chi_squared_cubic:.2f}')
            if ne_params is not None:
                print(f'ne reduced chi squared mtanh: {ne_chi_squared:.2f}')            

            '''
            # Do not use the mtanh fit if there are fewer than 3 points in the pedestal region.
            if ne_params is not None:
                pedestal_start = ne_params[0] - ne_params[1]
                pedestal_end = ne_params[0] + ne_params[1]
                pedestal_mask = (total_psi_ne > pedestal_start) & (total_psi_ne < pedestal_end)
                if len(total_psi_ne[pedestal_mask]) < 3:
                    print(f'Reject Ne mtanh fit because there are only {len(total_psi_ne[pedestal_mask])} points in the pedestal (require at least 3)')
                    ne_params = None
                    ne_covariance = None
                    ne_fitted = None
            '''
            
            # Do not use the mtanh fit if the reduced chi-squared is below 0 or above 20.
            if ne_params is not None:
                if ne_chi_squared <= 0 or ne_chi_squared > 20:
                    print('Reject Ne mtanh fit because the reduced chi squared is below 0 or greater than 20.')
                    ne_params = None
                    ne_covariance = None
                    ne_fitted = None
            
            # Do not use the cubic fit if the reduced chi-squared is below 0 or above 20.
            if ne_params_cubic is not None:
                if ne_chi_squared_cubic <= 0 or ne_chi_squared_cubic > 20:
                    print('Reject Ne cubic fit because the reduced chi squared is below 0 or greater than 20.')
                    ne_params_cubic = None
                    ne_covariance_cubic = None
                    ne_fitted_cubic = None


            # Choose the best fit based on the reduced chi-squared values.
            if ne_params is not None and ne_params_cubic is not None:
                if ne_chi_squared_cubic < ne_chi_squared:
                    print('CUBIC IS BEST FIT')
                    ne_fitted_best = ne_fitted_cubic
                    ne_best_chi_squared = ne_chi_squared_cubic
                    ne_best_fit_type = 'cubic'
                    number_of_ne_cubic_fits += 1
                else:
                    print('MTANH IS BEST FIT')
                    ne_fitted_best = ne_fitted
                    ne_best_chi_squared = ne_chi_squared
                    ne_best_fit_type = 'mtanh'
                    number_of_ne_mtanh_fits += 1

            elif ne_params is not None:
                print('MTANH IS BEST FIT')
                ne_fitted_best = ne_fitted
                ne_best_chi_squared = ne_chi_squared
                ne_best_fit_type = 'mtanh'
                number_of_ne_mtanh_fits += 1

            elif ne_params_cubic is not None:
                print('CUBIC IS BEST FIT')
                ne_fitted_best = ne_fitted_cubic
                ne_best_chi_squared = ne_chi_squared_cubic
                ne_best_fit_type = 'cubic'
                number_of_ne_cubic_fits += 1

            else:
                print('NO FITS WORKED')
                ne_fitted_best = None
                ne_best_chi_squared = None
                ne_best_fit_type = None
                number_of_ne_failed_fits += 1


            # OPTION TO SHIFT THE DATA AND FIT ACCORDING TO THE 2-PT MODEL
            if shift_to_2pt_model == True:
                Te_sep_eV = Teu_2pt_model(shot, time_in_s, lam_T_mm=1, geqdsk=4, pressure_opt = 3, lambdaq_opt=1) #lam_T_mm and geqdsk are just placeholders here. They don't do anything.
                new_x, shift = apply_2pt_shift(generated_psi_grid, te_fitted_best, Te_sep_eV, 1, only_shift_edge=True) #separatrix in psi coords is just 1
                print('T SEP: ', Te_sep_eV)
                print('SHIFT: ', shift)

                # now interpolate back onto the generated psi grid
                te_interp_function = interp1d(new_x, te_fitted, fill_value='extrapolate')
                te_fitted_best = te_interp_function(generated_psi_grid)

                ne_interp_function = interp1d(new_x, ne_fitted_best, fill_value='extrapolate')
                ne_fitted_best = ne_interp_function(generated_psi_grid)
            else:
                shift = 0



            # plotting option for debugging/checking fit quality
            if plot_the_fits == True:

                plt.errorbar(total_psi_te, total_te, yerr=total_te_err, fmt = 'o',mfc='white', color='red', alpha=0.7) # raw data
                
                # option to plot the mtanh and cubic fits separately here
                #if te_fitted is not None:
                    #plt.plot(generated_psi_grid, te_fitted, label = rf'mtanh: $\chi^2$ = {te_chi_squared:.2f}', linewidth=2)
                #plt.plot(generated_psi_grid, te_fitted_cubic, label=rf'cubic: $\chi^2$ = {te_chi_squared_cubic:.2f}', linewidth=2)

                # just plot the best fit
                if te_fitted_best is not None:
                    plt.plot(generated_psi_grid, te_fitted_best, label = rf'best fit: $\chi^2$ = {te_chi_squared_best:.2f}')
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

                plt.errorbar(total_psi_ne, total_ne, yerr=total_ne_err, fmt = 'o',mfc='white', color='green', alpha=0.7) # raw data
                
                # can plot the mtanh and cubic fits separately here
                #if ne_fitted is not None:
                     #plt.plot(generated_psi_grid, ne_fitted, label = rf'mtanh: $\chi^2$ = {ne_chi_squared:.2f}', linewidth=2)
                #plt.plot(generated_psi_grid, ne_fitted_cubic, label= rf'cubic: $\chi^2$ = {ne_chi_squared_cubic:.2f}', linewidth=2)

                # or just plot the best fit
                if ne_fitted_best is not None:
                    plt.plot(generated_psi_grid, ne_fitted_best, label = rf'best fit: $\chi^2$ = {ne_best_chi_squared:.2f}')
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

            if ne_fitted_best is not None:
                list_of_successful_ne_fit_times_ms.append(time)
                list_fitted_ne_profiles.append(ne_fitted_best)
                list_of_ne_reduced_chi_squared.append(ne_best_chi_squared)
                list_of_ne_fit_type.append(ne_best_fit_type)

            # also save the raw data points used to perform these fits
            list_of_total_psi_te.append(total_psi_te + shift)
            list_of_total_te.append(total_te)
            list_of_total_te_err.append(total_te_err)

            list_of_total_psi_ne.append(total_psi_ne + shift)
            list_of_total_ne.append(total_ne)
            list_of_total_ne_err.append(total_ne_err)
    


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

    # can choose to return the processed raw data used to create the fits if desired
    if return_processed_raw_data == True:
        return generated_psi_grid, list_of_successful_te_fit_times_ms, list_fitted_te_profiles, list_of_te_reduced_chi_squared, list_of_te_fit_type, list_of_successful_ne_fit_times_ms, list_fitted_ne_profiles, list_of_ne_reduced_chi_squared, list_of_ne_fit_type, \
            list_of_total_psi_te, list_of_total_te, list_of_total_te_err, list_of_total_psi_ne, list_of_total_ne, list_of_total_ne_err
    else:
        return generated_psi_grid, list_of_successful_te_fit_times_ms, list_fitted_te_profiles, list_of_te_reduced_chi_squared, list_of_te_fit_type, list_of_successful_ne_fit_times_ms, list_fitted_ne_profiles, list_of_ne_reduced_chi_squared, list_of_ne_fit_type



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
            te_fitted = Osborne_Tanh_cubic(total_psi_te, te_params[0], te_params[1], te_params[2], te_params[3], te_params[4], te_params[5], te_params[6])
            total_te_residuals = np.abs(total_te - te_fitted)
            te_outliers_mask = total_te_residuals < 3*total_te_err #reject any points that are more than 3 sigma away from the fit

            # ne outlier rejection
            ne_fitted = 1e20*Osborne_Tanh_cubic(total_psi_ne, ne_params[0], ne_params[1], ne_params[2], ne_params[3], ne_params[4], ne_params[5], ne_params[6])
            total_ne_residuals = np.abs(total_ne - ne_fitted)
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





    new_times_for_results = np.arange(0, t_max-t_min, 1)

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


    list_of_ne_params_that_worked = []
    list_of_indices_that_worked = []

    # now do the window smoothing
    for t_idx in range(len(new_times_for_results)):
        time = new_times_for_results[t_idx]

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
                    print('TE FIT FAILED')
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
    new_times_for_results = new_times_for_results[combined_successful_fit_mask]

    list_of_ne_fitted = list_of_ne_fitted[combined_successful_fit_mask]
    list_of_te_fitted = list_of_te_fitted[combined_successful_fit_mask]
    list_of_ne_fitted_error = list_of_ne_fitted_error[combined_successful_fit_mask]
    list_of_te_fitted_error = list_of_te_fitted_error[combined_successful_fit_mask]

    

    list_of_ne_fitted_at_Thomson_times = np.array(list_of_ne_fitted_at_Thomson_times)
    list_of_te_fitted_at_Thomson_times = np.array(list_of_te_fitted_at_Thomson_times)
    list_of_initial_fit_times_for_checking_smoothing = np.array(list_of_initial_fit_times_for_checking_smoothing)

    # if either fit failed at any time-point, just remove this time-point from the list

    

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

        ax.plot(new_times_for_results, list_of_te_fitted[:, psi_idx], label=f'psi = {psi_value_to_evolve:.2f}')
        ax.scatter(list_of_initial_fit_times_for_checking_smoothing, list_of_te_fitted_at_Thomson_times[:, psi_idx], marker='x', color='red')
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.show()

    Rmid_grid = psi_to_Rmid_map(shot, t_min, t_max, generated_psi_grid, new_times_for_results) #this is a 2D array of Rmid values at every psi value at every time point




    return new_times_for_results, generated_psi_grid, Rmid_grid, list_of_ne_fitted, list_of_ne_fitted_error, list_of_te_fitted, list_of_te_fitted_error



def master_fit_2D_alt(shot, t_min, t_max, smoothing_window=15):
    '''
    Exactly the same as the other 2D fitting function, except that the 1D fitting
    function (master_fit_ne_Te_1D) is used to do the fits at every time point.
    This gives a bit more flexibility (since it also tries a cubic fit), but currently
    does not have a post-fitting outlier rejection method and also does not scale the
    core Thomson data to the TCI data currently. The 1D method also doesn't do error bars yet

    TODO:
    Implement some option for post-fit outlier rejection
    Implement some method for scaling of the core Thomson data in the 1D fit
    Implement some method for getting error bars in the 1D fits.
    Let this function also use a cubic to fit if it wants.
    '''

    # get the ne and Te fits at each time point from the 1D fitting function
    generated_psi_grid, list_of_Thomson_times_te_ms, list_of_te_fitted_at_Thomson_times, list_of_te_reduced_chi_squared, \
    list_of_te_fit_type, list_of_Thomson_times_ne_ms, list_of_ne_fitted_at_Thomson_times, list_of_ne_reduced_chi_squared, \
    list_of_ne_fit_type, list_of_total_psi_te, list_of_total_te, list_of_total_te_err, \
    list_of_total_psi_ne, list_of_total_ne, list_of_total_ne_err = master_fit_ne_Te_1D(shot, t_min, t_max, plot_the_fits=False, remove_zeros_before_fitting=True, shift_to_2pt_model=True, return_processed_raw_data=True)



    # CONVERT THE 2D ARRAYS INTO 1D ARRAYS SO THAT WEIGHTS CAN BE APPLIED AND THE SMOOTHED FITS CAN BE APPLIED
    list_of_te_successful_fit_times_flattened = []
    list_of_total_psi_te_flattened = []
    list_of_total_te_flattened = []
    list_of_total_te_err_flattened = []

    list_of_ne_successful_fit_times_flattened = []
    list_of_total_psi_ne_flattened = []
    list_of_total_ne_flattened = []
    list_of_total_ne_err_flattened = []

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

    plt.scatter(list_of_raw_ne_xvalues_shifted, list_of_raw_ne, marker='x')
    plt.errorbar(list_of_raw_ne_xvalues_shifted, list_of_raw_ne, yerr=list_of_raw_ne_err, fmt = 'o', color='green')
    plt.show()

    plt.scatter(list_of_raw_Te_xvalues_shifted, list_of_raw_Te, marker='x')
    plt.errorbar(list_of_raw_Te_xvalues_shifted, list_of_raw_Te, yerr=list_of_raw_Te_err, fmt = 'o', color='red')
    plt.show()




    new_times_for_results = np.arange(0, t_max-t_min, 1) # return fits on 1ms timebase


    te_params_from_last_successful_fit = None
    ne_params_from_last_successful_fit = None

    # TODO: implement a proper error function for the 1D fits
    average_te_error_band = 0
    average_ne_error_band = 0

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



def master_fit_ne_Te_2D_quadratic(shot, t_min, t_max, time_window_for_evolution = None):
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

    # get the ne and Te fits at each time point from the 1D fitting function
    generated_psi_grid, list_of_Thomson_times_te_ms, list_of_te_fitted_at_Thomson_times, list_of_te_reduced_chi_squared, \
    list_of_te_fit_type, list_of_Thomson_times_ne_ms, list_of_ne_fitted_at_Thomson_times, list_of_ne_reduced_chi_squared, \
    list_of_ne_fit_type = master_fit_ne_Te_1D(shot, t_min, t_max, plot_the_fits=False, remove_zeros_before_fitting=True, shift_to_2pt_model=True, return_processed_raw_data=False)


    list_of_Thomson_times_te_ms_norm = np.array(list_of_Thomson_times_te_ms) - t_min
    list_of_Thomson_times_ne_ms_norm = np.array(list_of_Thomson_times_ne_ms) - t_min

    list_of_te_fitted_at_Thomson_times = np.array(list_of_te_fitted_at_Thomson_times)
    list_of_ne_fitted_at_Thomson_times = np.array(list_of_ne_fitted_at_Thomson_times)



    # PLACEHOLDER WITH 10pc error for the moment. TODO: proper errors
    list_of_te_fitted_err_at_Thomson_times = 0.1*list_of_te_fitted_at_Thomson_times
    list_of_ne_fitted_err_at_Thomson_times = 0.1*list_of_ne_fitted_at_Thomson_times

    

    # Time window for evolution
    if time_window_for_evolution is not None:

        ne_times_mask = (list_of_Thomson_times_te_ms_norm > time_window_for_evolution[0]) & (list_of_Thomson_times_te_ms_norm < time_window_for_evolution[1])
        list_of_Thomson_times_te_ms_norm = list_of_Thomson_times_ne_ms_norm[ne_times_mask]
        list_of_te_fitted_at_Thomson_times = list_of_te_fitted_at_Thomson_times[ne_times_mask]
        list_of_te_fitted_err_at_Thomson_times = list_of_te_fitted_err_at_Thomson_times[ne_times_mask]

        te_times_mask = (list_of_Thomson_times_te_ms_norm > time_window_for_evolution[0]) & (list_of_Thomson_times_te_ms_norm < time_window_for_evolution[1])
        list_of_Thomson_times_te_ms_norm = list_of_Thomson_times_te_ms_norm[te_times_mask]
        list_of_te_fitted_at_Thomson_times = list_of_te_fitted_at_Thomson_times[te_times_mask]
        list_of_te_fitted_err_at_Thomson_times = list_of_te_fitted_err_at_Thomson_times[te_times_mask]

        output_time_grid = np.arange(time_window_for_evolution[0], time_window_for_evolution[1], 1)
    else:
        output_time_grid = np.arange(t_min, t_max, 1)





    evolve_fits_by_radius_example_for_panel_plots(list_of_Thomson_times_ne_ms_norm, generated_psi_grid, list_of_ne_fitted_at_Thomson_times)
    evolve_fits_by_radius_example_for_panel_plots(list_of_Thomson_times_te_ms_norm, generated_psi_grid, list_of_te_fitted_at_Thomson_times)
    


    new_ne_values, new_ne_err = evolve_fits_by_radius(output_time_grid, list_of_Thomson_times_ne_ms_norm, generated_psi_grid, list_of_ne_fitted_at_Thomson_times, list_of_ne_fitted_err_at_Thomson_times)
    new_Te_values, new_Te_err = evolve_fits_by_radius(output_time_grid, list_of_Thomson_times_te_ms_norm, generated_psi_grid, list_of_te_fitted_at_Thomson_times, list_of_te_fitted_err_at_Thomson_times)

    Rmid_grid = psi_to_Rmid_map(shot, t_min, t_max, generated_psi_grid, output_time_grid) #this is a 2D array of Rmid values at every psi value at every time point

    return output_time_grid, generated_psi_grid, Rmid_grid, new_ne_values, new_ne_err, new_Te_values, new_Te_err #returns ne and Te profiles at every ms.


    
    




###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################




#master_fit_ne_Te_1D(1091210028, 587, 700, plot_the_fits=True, remove_zeros_before_fitting=True, shift_to_2pt_model=True)

#master_fit_ne_Te_2D_window_smoothing(1091210027, 587, 700, smoothing_window=15)

master_fit_2D_alt(1091210027, 587, 700, smoothing_window=15)
### FUNCTIONS FOR ADJUSTING THOMSON SCATTERING DATA BASED ON TCI DATA ###
### Originally written by M.A. Miller, modified by J. Dunsmore        ###


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



def compare_ts_tci(shot, tmin, tmax, nl_num=4):
    '''
    INPUTS
    --------
    shot: C-Mod shot number (integer)
    tmin: minimum time in seconds
    tmax: maximum time in seconds
    nl_num: interferometer chord to use (4 is default as it is most reliably on during operation)

    RETURNS
    --------
    nl_ts1: 1D array of line-integrated density values from the synthetic TCI diagnostic for Thomson laser 1
    nl_ts2: 1D array of line-integrated density values from the synthetic TCI diagnostic for Thomson laser 2
    nl_tci1: 1D array of line-integrated density values from the actual TCI diagnostic (on the timebase of Thomson laser 1)
    nl_tci2: 1D array of line-integrated density values from the actual TCI diagnostic (on the timebase of Thomson laser 2)
    time1: 1D array of times in seconds for the Thomson laser 1
    time2: 1D array of times in seconds for the Thomson laser 2

    DESCRIPTION
    --------
    Takes the Thomson data from each laser pulse and returns the line-integrated density along an interferometer chord 
    in order to compare this to the actual TCI data. Returns separate arrays for each Thomson laser.
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

            print('HERE')
            print(tci_t)
            print(tci)

            print(len(tci_t))
            print(len(tci))

            print(tci_t.shape)
            print(tci.shape)

            tci_t = np.array(tci_t)
            tci = np.array(tci)

            tci = tci.flatten()

            if not np.all(np.diff(tci_t) > 0):
                print("tci_t is not monotonically increasing")

            if not np.all(np.isfinite(tci_t)) or not np.all(np.isfinite(tci)):
                print("Non-finite values found in tci_t or tci")

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
    INPUTS
    --------
    shot: C-Mod shot number (integer)
    tmin: minimum time in seconds
    tmax: maximum time in seconds
    nl_num: interferometer chord to use (4 is default as it is most reliably on during operation)

    RETURNS
    --------
    nl_ts: 1D array of line-integrated density values from the synthetic TCI diagnostic (created from the TS data)
    nl_ts_t: corresponding timebase
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
    INPUTS
    --------
    shot: C-Mod shot number (integer)
    tmin: minimum time in seconds
    tmax: maximum time in seconds
    nl_num: interferometer chord to use (4 is default as it is most reliably on during operation)

    RETURNS
    --------
    t: 1D array of times of the output synthetic TCI system
    z: 2D array of z values of the output synthetic TCI system
    n_e: 2D array of ne values at each z value at each time point of the synthetic TCI system
    n_e_sig: 2D array of ne errors at each z value at each time point of the synthetic TCI system

    DESCRIPTION
    --------
    This is the main part of the algorithm. The Thomson scattering data is mapped in space to lie along the chosen TCI chord.
    This is done by mapping the TS data from RZ space to psi space, and then mapping from psi space to the TCI radius.
    Once this has been done, the synthetic TCI data can be integrated in integrated_ts2tci to give a line-integrated density.
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


    #gets the magnetic equilibrium at every time point - note EFIT20 is the TS timebase. If this doesn't work, try ANALYSIS (the main one)
    try:
        e = eqtools.CModEFIT.CModEFITTree(int(shot), tree='EFIT20')
    except:
        e = eqtools.CModEFIT.CModEFITTree(int(shot), tree='ANALYSIS')


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
    INPUTS
    --------
    shot: C-Mod shot number (integer)

    RETURNS
    --------
    n_yag1: number of times laser 1 fires
    n_yag2: number of times laser 2 fires
    indices1: time indices of when laser 1 fires
    indices2: time indices of when laser 2 fires

    DESCRIPTION
    --------
    Function to return the time indices of when the two Thomson lasers fire.
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
    INPUTS
    --------
    shot: C-Mod shot number (integer)
    tmin: minimum time in seconds
    tmax: maximum time in seconds
    nl_num: interferometer chord to use (4 is default as it is most reliably on during operation)

    RETURNS
    --------
    mult_factor1: multiplication factor that should be applied to Thomson laser 1 to match the TCI data over the shot
    mult_factor2: multiplication factor that should be applied to Thomson laser 2 to match the TCI data over the shot
    mult_factor: average of the two multiplicative factors

    DESCRIPTION
    --------
    Compares the actual TCI data over the shot to the synthetic TCI data (created from the TS data over the same time window)
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
    INPUTS
    --------
    shot: C-Mod shot number (integer)
    core_time_ms: 1D array of time points for the core Thomson data in ms (not sure this is actually vital as an input)
    core_ne: 2D array of electron density values for the core Thomson data (only needed to be multiplied by the scaling factor for output)

    RETURNS
    --------
    core_ne: 2D array of electron density values for the core Thomson data, scaled to match the TCI data

    DESCRIPTION
    --------
    Takes in the raw core Thomson data for a shot, and scales it to match the TCI data between 0.5s and 1.5s.
    The time window chosen is hopefully long enough to get a meaningful scaling factor, and usually reflects the time when the plasma is most stable for C-Mod shots.
    Requires that the current be >0.4MA at 0.5s and 1.5s before scaling the data (this just ensures that we have a reasonable plasma).
    '''

    ip_node = MDSplus.Tree('cmod', shot).getNode('\ip')

    ip_data = ip_node.data()
    ip_times = ip_node.dim_of().data()

    indices = np.where((ip_times > 0.5) & (ip_times < 1.5))[0]
    idx_start, idx_end = indices[0], indices[-1]

    if np.abs(ip_data[idx_start]) < 0.4e6 or np.abs(ip_data[idx_end]) < 0.4e6:
        print('The plasma current is not higher than 0.4MA for the full range 0.5s-1.5s.\
              Therefore, a Thomson scaling to the TCI data may not be reliable for this shot \
              and has not beed performed.')
        return core_ne
    
    # get the scaling factor for the two lasers based on TCI measurements
    mult_factor1, mult_factor2, mult_factor = get_ts_tci_ratio(shot, 0.5, 1.5) # may want to change this. TODO: be able to vary the time window

    # get the times in ms for the two lasers
    n_yag1, n_yag2, indices1, indices2 = parse_yags(shot)

    tree = MDSplus.Tree('CMOD', shot)

    ts_time = tree.getNode('\\TOP.ELECTRONS.YAG_NEW.RESULTS.PROFILES:NE_RZ').dim_of().data()

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

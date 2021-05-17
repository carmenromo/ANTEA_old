import numpy  as np
import pandas as pd

from typing import Sequence, Tuple


def apply_spe_dist(time: np.array, tau_sipm: Tuple[float, float]) -> Tuple[np.array, float]:
    """
    Returns a normalized array following the double exponential
    distribution of the sipm response.
    """
    spe_response = spe_dist(time, tau_sipm)
    if np.sum(spe_response) == 0:
        return np.zeros(len(time)), 0.
    norm_dist    = np.sum(spe_response)
    spe_response = spe_response/norm_dist #Normalization
    return spe_response, norm_dist


def spe_dist(time: np.array, tau_sipm: Tuple[float, float]) -> np.array:
    """
    Analitic function that calculates the double exponential decay for
    the sipm response.
    """
    alfa      = 1.0/tau_sipm[1]
    beta      = 1.0/tau_sipm[0]
    t_p       = np.log(beta/alfa)/(beta-alfa)
    K         = (beta)*np.exp(alfa*t_p)/(beta-alfa)
    time_dist = K*(np.exp(-alfa*time)-np.exp(-beta*time))
    return time_dist


def convolve_tof(spe_response: Sequence[float],
                 signal: Sequence[float]) -> Sequence[float]:
    """
    Computes the spe_response distribution for the given signal.
    """
    if not np.count_nonzero(spe_response):
        print('spe_response values are zero')
        return np.zeros(len(spe_response)+len(signal)-1)
    conv_first = np.hstack([spe_response, np.zeros(len(signal)-1)])
    conv_res   = np.zeros(len(signal)+len(spe_response)-1)
    pe_pos     = np.argwhere(signal > 0)
    pe_recov   = signal[pe_pos]
    for i in range(len(pe_recov)): #Loop over the charges
        conv_first_ch = conv_first*pe_recov[i]
        desp          = np.roll(conv_first_ch, pe_pos[i])
        conv_res     += desp
    return conv_res


def tdc_convolution(tof_response: pd.DataFrame,
                    spe_response: Sequence[float],
                    s_id: int,
                    time_window: float) -> Sequence[float]:
    """
    Calculates the tof convolution along the time window for the given sensor_id.
    """
    pe_vect = np.zeros(time_window)
    sel_tof = tof_response[(tof_response.sensor_id == s_id) &
                           (tof_response.time < time_window)]
    pe_vect[sel_tof.time.values] = sel_tof.charge.values
    tdc_conv = convolve_tof(spe_response, pe_vect)
    return tdc_conv


def translate_charge_conv_to_wf_df(event_id: int,
                                   s_id: int,
                                   conv_vect: Sequence[float]) -> pd.DataFrame:
    """
    Translates a given numpy array into a tof type dataframe.
    """
    keys        = np.array(['event_id', 'sensor_id', 'time', 'charge'])
    t_bin       = np.where(conv_vect>0)[0]
    charge      = conv_vect[conv_vect>0]
    evt         = np.full(len(t_bin), event_id)
    sns_id_full = np.full(len(t_bin), s_id)
    a_wf        = np.array([evt, sns_id_full, t_bin, charge])
    wf_df       = pd.DataFrame(a_wf.T, columns=keys).astype({'event_id': 'int32',
                                                            'sensor_id': 'int32',
                                                            'time' : 'int32'})
    return wf_df

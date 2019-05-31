import tables as tb
import numpy  as np

from antea.io.mc_io                   import read_mcsns_response
from invisible_cities.io  .mcinfo_io  import read_mcinfo
from invisible_cities.core.exceptions import NoHits


def get_true_pos(part):
    hit_positions = [h.pos for h in part.hits]
    energies      = [h.E   for h in part.hits]
    energy        = sum(energies)
    if energy: return np.average(hit_positions, axis=0, weights=energies)
    else: raise NoHits


def find_closest_sipm(x, y, z, sens_pos, sns_over_thr, charges_over_thr):
    """Returns the sensor ID of the closest sipm to the given true event
    """
    min_dist = 1.e9
    min_sns  = 0
    for sns_id, charge in zip(sns_over_thr, charges_over_thr):
        pos  = sens_pos[sns_id]
        dist = np.linalg.norm(np.subtract((x, y, z), pos))
        if dist < min_dist:
            min_dist = dist
            min_sns  = sns_id
    return min_sns


def sensor_position(h5in):
    """A dictionary that stores the position of all the sensors
    in cartesian coordinates is created.
    """
    sipms    = h5in.root.MC.sensor_positions[:]
    sens_pos = {}
    for sipm in sipms:
        sens_pos[sipm[0]] = (sipm[1], sipm[2], sipm[3])
    return sens_pos


def sensor_position_cyl(h5in):
    """A dictionary that stores the position of all the sensors
    in cylindrical coordinates is created.
    """
    sipms        = h5in.root.MC.sensor_positions[:]
    sens_pos_cyl = {}
    for sipm in sipms:
        sens_pos_cyl[sipm[0]] = (np.sqrt(sipm[1]*sipm[1] + sipm[2]*sipm[2]),
                                 np.arctan2(sipm[2], sipm[1]),
                                 sipm[3])
    return sens_pos_cyl


def sensors_info(ave_true, sens_pos, sens_pos_cyl, sns_over_thr, charges_over_thr):

    """For a given true position of an event, returns all the information of the sensors
    for the corresponding half of the ring.

    Parameters
    ----------
    ave_true         : np.array
    Position of the true hits (cart coordinates).
    sens_pos         : dict
    Contains the position of each sensor (cart coordinates).
    sens_pos_cyl     : dict
    Contains the position of each sensor (cyl coordinates).
    sns_over_thr     : np.array
    IDs of the sensors that detected charge above a certain threshold.
    charges_over_thr : np.array
    Charges of the sensors above a certain threshold

    Returns
    -------
    ampl1    : int
    Total charge detected for this single event.
    count1   : int
    Number of sensors that detected charge.
    pos1     : np.array
    Position of every sensor that detected some charge (cart coordinates).
    pos1_cyl : np.array
    Position of every sensor that detected some charge (cyl coordinates).
    q1       : np.array
    Charge detected by every sensor.
    """

    closest = find_closest_sipm(ave_true[0], ave_true[1], ave_true[2],
                                sens_pos, sns_over_thr, charges_over_thr)

    ampl1    = 0
    count1   = 0
    pos1     = []
    pos1_cyl = []
    q1       = []

    for sns_id, charge in zip(sns_over_thr, charges_over_thr):
        pos         = sens_pos    [sns_id]
        pos_cyl     = sens_pos_cyl[sns_id]
        pos_closest = sens_pos    [closest]
        scalar_prod = sum(a*b for a, b in zip(pos, pos_closest))
        if scalar_prod > 0.:
            pos1    .append(pos)
            pos1_cyl.append(pos_cyl)
            q1      .append(charge)
            ampl1   += charge
            count1  += 1

    return ampl1, count1, pos1, pos1_cyl, q1


def get_reco_r_and_var_phi(ave_true, cyl_pos, q):
    r        = np.sqrt(ave_true[0]**2 + ave_true[1]**2)
    phi_pos  = np.array([el[1] for el in cyl_pos])
    mean_phi = np.average(phi_pos, weights=q)
    var_phi  = np.average((phi_pos - mean_phi)**2, weights=q)
    return r, var_phi

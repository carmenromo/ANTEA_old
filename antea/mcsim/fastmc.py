#
#  The fast MC generates pairs of interaction points based on
#  pre-determined matrices of true r, phi, and z coordinates vs. their
#  reconstructed error. It uses the true information coming from GEANT4 simulations
#

import numpy  as np
import pandas as pd

from invisible_cities.core import system_of_units as units

from antea.mcsim.errmat import errmat
import antea.reco.reco_functions as rf

def simulate_reco_event(evt_id: int, hits: pd.DataFrame, particles: pd.DataFrame,
                        errmat_p_r: errmat, errmat_p_phi: errmat, errmat_p_z: errmat,
                        errmat_p_t: errmat, errmat_c_r: errmat, errmat_c_phi: errmat,
                        errmat_c_z: errmat, errmat_c_t: errmat,
                        true_e_threshold: float = 0., photo_range: float = 1.) -> pd.DataFrame:
    """
    Simulate the reconstructed coordinates for 1 coincidence from true GEANT4 dataframes.
    Notice that the time binning must be provided in ps.
    """

    evt_parts = particles[particles.event_id == evt_id]
    evt_hits  = hits     [hits.event_id      == evt_id]
    energy    = evt_hits.energy.sum()
    if energy < true_e_threshold:
        events = pd.DataFrame({'event_id':  [float(evt_id)],
                               'true_energy': [energy],
                               'true_r1':   [0.],
                               'true_phi1': [0.],
                               'true_z1':   [0.],
                               'true_t1':   [0.],
                               'true_r2':   [0.],
                               'true_phi2': [0.],
                               'true_z2':   [0.],
                               'true_t2':   [0.],
                               'phot_like1':[0.],
                               'phot_like2':[0.],
                               'reco_r1':   [0.],
                               'reco_phi1': [0.],
                               'reco_z1':   [0.],
                               'reco_t1':   [0.],
                               'reco_r2':   [0.],
                               'reco_phi2': [0.],
                               'reco_z2':   [0.],
                               'reco_t2':   [0.]
                               })
        return events

    pos1, pos2, t1, t2, phot1, phot2 = rf.find_first_interactions_in_active(evt_parts, evt_hits, photo_range)

    if len(pos1) == 0 or len(pos2) == 0:
        events = pd.DataFrame({'event_id':  [float(evt_id)],
                               'true_energy': [energy],
                               'true_r1':   [0.],
                               'true_phi1': [0.],
                               'true_z1':   [0.],
                               'true_t1':   [0.],
                               'true_r2':   [0.],
                               'true_phi2': [0.],
                               'true_z2':   [0.],
                               'true_t2':   [0.],
                               'phot_like1':[0.],
                               'phot_like2':[0.],
                               'reco_r1':   [0.],
                               'reco_phi1': [0.],
                               'reco_z1':   [0.],
                               'reco_t1':   [0.],
                               'reco_r2':   [0.],
                               'reco_phi2': [0.],
                               'reco_z2':   [0.],
                               'reco_t2':   [0.]
                               })
        return events

    t1 = t1 / units.ps
    t2 = t2 / units.ps

    # Transform in cylindrical coordinates
    cyl_pos = rf.from_cartesian_to_cyl(np.array([pos1, pos2]))

    r1   = cyl_pos[0, 0]
    phi1 = cyl_pos[0, 1]
    z1   = cyl_pos[0, 2]
    r2   = cyl_pos[1, 0]
    phi2 = cyl_pos[1, 1]
    z2   = cyl_pos[1, 2]

    # Get all errors.
    if phot1:
        er1   = errmat_p_r.get_random_error(r1)
        ephi1 = errmat_p_phi.get_random_error(phi1)
        ez1   = errmat_p_z.get_random_error(z1)
        et1   = errmat_p_t.get_random_error(t1)
    else:
        er1   = errmat_c_r.get_random_error(r1)
        ephi1 = errmat_c_phi.get_random_error(phi1)
        ez1   = errmat_c_z.get_random_error(z1)
        et1   = errmat_c_t.get_random_error(t1)

    if phot2:
        er2   = errmat_p_r.get_random_error(r2)
        ephi2 = errmat_p_phi.get_random_error(phi2)
        ez2   = errmat_p_z.get_random_error(z2)
        et2   = errmat_p_t.get_random_error(t2)
    else:
        er2   = errmat_c_r.get_random_error(r2)
        ephi2 = errmat_c_phi.get_random_error(phi2)
        ez2   = errmat_c_z.get_random_error(z2)
        et2   = errmat_c_t.get_random_error(t2)

    # Compute reconstructed quantities.
    r1_reco = r1 - er1
    r2_reco = r2 - er2
    phi1_reco = phi1 - ephi1
    phi2_reco = phi2 - ephi2
    z1_reco = z1 - ez1
    z2_reco = z2 - ez2
    t1_reco = t1 - et1
    t2_reco = t2 - et2

    event_ids = [float(evt_id)]
    energies  = [energy]

    true_r1   = [r1]
    true_phi1 = [phi1]
    true_z1   = [z1]
    true_t1   = [t1]
    true_r2   = [r2]
    true_phi2 = [phi2]
    true_z2   = [z2]
    true_t2   = [t2]

    phot_like1 = [float(phot1)]
    phot_like2 = [float(phot2)]

    reco_r1   = [r1_reco]
    reco_phi1 = [phi1_reco]
    reco_z1   = [z1_reco]
    reco_t1   = [t1_reco]
    reco_r2   = [r2_reco]
    reco_phi2 = [phi2_reco]
    reco_z2   = [z2_reco]
    reco_t2   = [t2_reco]

    events = pd.DataFrame({'event_id':  event_ids,
                           'true_energy': energies,
                           'true_r1':   true_r1,
                           'true_phi1': true_phi1,
                           'true_z1':   true_z1,
                           'true_t1':   true_t1,
                           'true_r2':   true_r2,
                           'true_phi2': true_phi2,
                           'true_z2':   true_z2,
                           'true_t2':   true_t2,
                           'phot_like1':phot_like1,
                           'phot_like2':phot_like2,
                           'reco_r1':   reco_r1,
                           'reco_phi1': reco_phi1,
                           'reco_z1':   reco_z1,
                           'reco_t1':   reco_t1,
                           'reco_r2':   reco_r2,
                           'reco_phi2': reco_phi2,
                           'reco_z2':   reco_z2,
                           'reco_t2':   reco_t2})
    return events

import numpy as np
import tables as tb

from enum        import Enum
from .. reco import reco_functions as rf

from   antea.io.mc_io                   import read_mcsns_response
from   invisible_cities.io  .mcinfo_io  import read_mcinfo


class CoordSys(Enum):
    cart = 0
    cyl  = 1

# def sensor_position(paths, coord_sys):
#     for path in paths:
#         sipms    = path.root.MC.sensor_positions[:]
#         sens_pos = {}
#         if   coord_sys is CoordSys.cart:
#             for sipm in sipms:
#                 sens_pos[sipm[0]] = (sipm[1], sipm[2], sipm[3])
#         elif coord_sys is CoordSys.cyl :
#             for sipm in sipms:
#                 sens_pos[sipm[0]] = (np.linalg.norm(sipm[1:3]),
#                                      np.arctan2(sipm[2], sipm[1]),
#                                      sipm[3])
#         else: raise  TypeError(f"Invalid CoordSys: {type(coord_sys)}")
#         return sens_pos

def true_photoelect(compton=False):
    """Returns the position of the true photoelectric energy deposition
    calculated with barycenter algorithm.
    It allows the possibility of including compton events.
    """
    def true_photoelect(particles):
        ave_true1 = []
        ave_true2 = []
        for indx, part in particles.items():
            if part.name == 'e-' :
                mother = particles[part.mother_indx]
                if part.initial_volume == 'ACTIVE' and part.final_volume == 'ACTIVE':
                    if mother.primary and np.isclose(mother.E*1000., 510.999, atol=1.e-3):
                        if compton==True: pass
                        else:
                            if np.isclose(sum(h.E for h in part.hits), 0.476443, atol=1.e-6): pass
                            else: continue

                        if mother.p[1] > 0.: ave_true1 = rf.get_true_pos(part)
                        else:                ave_true2 = rf.get_true_pos(part)
        if not len(ave_true1) and not len(ave_true2):
            return [0,0,0],[0,0,0]
        return ave_true1, ave_true2
    return true_photoelect


def part_from_files(paths):
    def part_from_files(paths, evt):
        for path in paths:
            with tb.open_file(path, "r") as h5in:
                this_event_dict = read_mcinfo(h5in, (evt, evt+1))
                part_dict       = list(this_event_dict.values())[0]
                return part_dict
    return part_from_files


def wf_from_files(paths):
    def wf_from_files(paths, evt):
        for path in paths:
            with tb.open_file(path, "r") as h5in:
                this_event_wvf = read_mcsns_response(path, (evt, evt+1))
                sns_dict       = list(this_event_wvf.values())[0]
                return sns_dict
    return wf_from_files

def part_wfs_from_files(paths):
        for path in paths:
            with tb.open_file(path, "r") as h5in:

                h5extents      = h5in.root.MC.extents
                events_in_file = len(h5extents)

                sens_pos       = rf.sensor_position    (h5in)
                sens_pos_cyl   = rf.sensor_position_cyl(h5in)

                for evt in range(events_in_file):

                    this_event_part = read_mcinfo        (h5in, (evt, evt+1))
                    this_event_wvf  = read_mcsns_response(path, (evt, evt+1))
                    part_dict       = list(this_event_part.values())[0]
                    sns_dict        = list(this_event_wvf .values())[0]

                    yield dict(particles=part_dict, wfs=sns_dict, sens_pos=sens_pos,
                                sens_pos_cyl=sens_pos_cyl, event_number=evt,
                                run_number=0, timestamp=0)

def wvf_passing_filter(nsteps, th_start):
    def wvf_passing_filter(wfs, ave_true1, ave_true2, sens_pos, sens_pos_cyl):
        tot_charges = np.array(list(map(lambda x: sum(x.charges), list(wfs.values()))))
        sns_ids     = np.array(list(wfs.keys()))

        true_r1        = np.zeros(nsteps)
        true_r2        = np.zeros(nsteps)
        var_phi1       = np.zeros(nsteps)
        var_phi2       = np.zeros(nsteps)

        for ith,threshold in enumerate(range(th_start, nsteps + th_start)):
            indices_over_thr = (tot_charges > threshold)
            sns_over_thr     = sns_ids    [indices_over_thr]
            charges_over_thr = tot_charges[indices_over_thr]

            if len(charges_over_thr) == 0:
                raise ValueError('no charges above threshold')
                continue

            ampl1, count1, pos1, pos1_cyl, q1 = rf.sensors_info(ave_true1,
                                           sens_pos,
                                           sens_pos_cyl,
                                           sns_over_thr,
                                           charges_over_thr)

            ampl2, count2, pos2, pos2_cyl, q2 = rf.sensors_info(ave_true2,
                                           sens_pos,
                                           sens_pos_cyl,
                                           sns_over_thr,
                                           charges_over_thr)

            if ampl1 and sum(q1) != 0:
                r1, var_phi = rf.get_reco_r_and_var_phi(ave_true1, pos1_cyl, q1)
                var_phi1[ith] = var_phi
                true_r1 [ith] = r1
            else:
                var_phi1[ith] = 1.e9
                true_r1 [ith] = 1.e9

            if ampl2 and sum(q2) != 0:
                r2, var_phi = rf.get_reco_r_and_var_phi(ave_true2, pos2_cyl, q2)
                var_phi2[ith] = var_phi
                true_r2 [ith] = r2
            else:
                var_phi2[ith] = 1.e9
                true_r2 [ith] = r2

        return true_r1, true_r2, var_phi1, var_phi2

    return wvf_passing_filter

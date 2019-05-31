import os
import tables as tb
import numpy  as np

from invisible_cities.io.mcinfo_io import read_mcinfo
from antea           .io.mc_io     import read_mcsns_response
from . reco_functions              import true_photoelect
from . reco_functions              import sensors_info
from . reco_functions              import sensor_position
from . reco_functions              import sensor_position_cyl
from . reco_functions              import get_reco_r_and_var_phi


def test_true_photoelect():
    test_file      = os.environ['ANTEADIR'] + '/testdata/full_ring_test.pet.h5'
    h5in           = tb.open_file(test_file, mode='r')
    h5extents      = h5in.root.MC.extents
    events_in_file = len(h5extents)

    n_phot      = 0
    n_phot_comp = 0
    for evt in range(events_in_file):
        this_event_dict = read_mcinfo(h5in, (evt, evt+1))
        this_event_wvf  = read_mcsns_response(test_file, (evt, evt+1))
        part_dict       = list(this_event_dict.values())[0]
        ave_true1, ave_true2 = true_photoelect(h5in, test_file, evt, compton=False)
        if np.any(ave_true1) or np.any(ave_true2):
            n_phot += 1
        ave_true_comp1, ave_true_comp2 = true_photoelect(h5in, test_file, evt, compton=True)
        if np.any(ave_true_comp1) or np.any(ave_true_comp2):
            n_phot_comp += 1

    assert n_phot_comp >= n_phot


def test_sensors_info():
    test_file      = os.environ['ANTEADIR'] + '/testdata/full_ring_test.pet.h5'
    h5in           = tb.open_file(test_file, mode='r')
    h5extents      = h5in.root.MC.extents
    events_in_file = len(h5extents)

    sens_pos       = sensor_position    (h5in)
    sens_pos_cyl   = sensor_position_cyl(h5in)

    for evt in range(events_in_file):
        this_event_dict = read_mcinfo(h5in, (evt, evt+1))
        this_event_wvf  = read_mcsns_response(test_file, (evt, evt+1))
        part_dict       = list(this_event_dict.values())[0]
        ave_true1, ave_true2 = true_photoelect(h5in, test_file, evt, compton=False)

        sns_dict    = list(this_event_wvf.values())[0]
        tot_charges = np.array(list(map(lambda x: sum(x.charges), list(sns_dict.values()))))
        sns_ids     = np.array(list(sns_dict.keys()))

        threshold        = 2
        indices_over_thr = (tot_charges > threshold)
        sns_over_thr     = sns_ids    [indices_over_thr]
        charges_over_thr = tot_charges[indices_over_thr]

        if np.any(ave_true1) and np.any(ave_true2):
            ampl1, count1, pos1, pos1_cyl, q1 = sensors_info(ave_true1,
                                                             sens_pos,
                                                             sens_pos_cyl,
                                                             sns_over_thr,
                                                             charges_over_thr)

            ampl2, count2, pos2, pos2_cyl, q2 = sensors_info(ave_true2,
                                                             sens_pos,
                                                             sens_pos_cyl,
                                                             sns_over_thr,
                                                             charges_over_thr)

            assert type(ampl1)  == type(ampl2)   == int
            assert type(count1) == type(count2)  == int
            assert len(pos1)    == len(pos1_cyl) == len(q1)
            assert len(pos2)    == len(pos2_cyl) == len(q2)

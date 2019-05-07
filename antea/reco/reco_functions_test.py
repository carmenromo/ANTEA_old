import tables as tb
import numpy  as np

from antea.io.mc_io                   import read_mcsns_response
from invisible_cities.io  .mcinfo_io  import read_mcinfo
from antea.reco.reco_functions        import true_photoelect

def true_photoelect_test():
    test_file      = os.environ['ANTEADIR'] + '/testdata/full_ring_test.pet.h5'
    #true_file      = 'full_ring_iradius165mm_z140mm_depth3cm_pitch7mm.000.pet.h5'
    h5in           = tb.open_file(test_file, mode='r')
    h5extents      = h5in.root.MC.extents
    events_in_file = len(h5extents)
    n_phot      = 0
    n_phot_comp = 0
    for evt in range(events_in_file):
        this_event_dict = read_mcinfo(h5in, (evt, evt+1))
        this_event_wvf  = read_mcsns_response(true_file, (evt, evt+1))
        part_dict       = list(this_event_dict.values())[0]
        ave_true1, ave_true2 = true_photoelect(h5in, true_file, evt, compton=False)
        if ave_true1 or ave_true2:
            n_phot += 1
        ave_true_comp1, ave_true_comp2 = true_photoelect(h5in, true_file, evt, compton=True)
        if ave_true_comp1 or ave_true_comp2:
            n_phot_comp += 1

    assert n_phot_comp >= n_phot

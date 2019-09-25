import os
import math
import numpy                 as np
import pandas                as pd
import hypothesis.strategies as st

from hypothesis  import given
from collections import OrderedDict
from .           import reco_functions   as rf
from .           import mctrue_functions as mcf
from .. database import load_db          as db


part_id = st.integers(min_value=1, max_value=1000)

@given(part_id)
def test_find_hits_of_given_particle(ANTEADATADIR, part_id):
    """
    This test checks that for one given particle, the function find_hits_of_given_particle
    returns a dataframe with the corresponding hits of that particle.
    """
    PATH_IN  = os.path.join(ANTEADATADIR, 'ring_test_1000ev.h5')
    hits     = pd.read_hdf(PATH_IN, 'MC/hits')

    sel_hits = mcf.find_hits_of_given_particles([part_id], hits)
    if len(sel_hits):
        assert len(sel_hits.particle_id.unique()) == 1


l = st.lists(part_id, min_size=2, max_size=1000)

@given(l)
def test_find_hits_of_given_particles(ANTEADATADIR, l):
    """
    This test checks that for some given particles, the function find_hits_of_given_particle
    returns a dataframe with the corresponding hits of those particles.
    """
    PATH_IN  = os.path.join(ANTEADATADIR, 'ring_test_1000ev.h5')
    hits     = pd.read_hdf(PATH_IN, 'MC/hits')

    sel_hits             = mcf.find_hits_of_given_particles(l, hits)
    non_repeated_part_id = list(OrderedDict.fromkeys(l))
    parts_without_hits   = 0
    for part in non_repeated_part_id:
        if not len(hits[hits.particle_id==part]):
            parts_without_hits += 1
    if len(sel_hits):
                assert len(sel_hits.particle_id.unique()) == len(non_repeated_part_id) - parts_without_hits


def test_select_photoelectric(ANTEADATADIR):
    """
    This test checks that the function select_photoelectric takes the events
    in which at least one of the initial gammas interacts via photoelectric
    effect depositing its 511 keV and stores the weighted average position
    of its/their hits.
    """
    PATH_IN      = os.path.join(ANTEADATADIR, 'ring_test_1000ev.h5')
    DataSiPM     = db.DataSiPM('petalo', 0)
    DataSiPM_idx = DataSiPM.set_index('SensorID')
    sns_response = pd.read_hdf(PATH_IN, 'MC/waveforms')
    threshold    = 2
    sel_df       = rf.find_SiPMs_over_threshold(sns_response, threshold)

    particles = pd.read_hdf(PATH_IN, 'MC/particles')
    hits      = pd.read_hdf(PATH_IN, 'MC/hits')
    events    = particles.event_id.unique()

    for evt in events[:]:
        evt_parts = particles[particles.event_id == evt]
        evt_hits  = hits     [hits     .event_id == evt]

        select, true_pos = mcf.select_photoelectric(evt_parts, evt_hits)

        if select:
            assert rf.greater_or_equal(evt_hits.energy.sum(), 0.476443, 1.e-3)
        else:
            assert len(true_pos) == 0

        if len(true_pos) == 1:
            assert rf.lower_or_equal(evt_hits.energy.sum(), 0.511, 1.e-3)
        elif len(true_pos) == 2:
            assert evt_hits.energy.sum() > 0.511


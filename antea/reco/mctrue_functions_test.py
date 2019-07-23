import os
import math
import numpy  as np
import pandas as pd

from .           import reco_functions   as rf
from .           import mctrue_functions as mcf
from .. database import load_db          as db

def test_select_photoelectric(ANTEADATADIR):
    PATH_IN      = os.path.join(ANTEADATADIR, 'ring_test_new_tbs.h5')
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

        sel    = False
        true_p = []

        sel_volume   = (evt_parts.initial_volume == 'ACTIVE') & (evt_parts.final_volume == 'ACTIVE')
        sel_name     =  evt_parts.name == 'e-'
        sel_vol_name = evt_parts[sel_volume & sel_name]
        ids          = sel_vol_name.particle_id.values

        sel_hits   = mcf.find_hits_of_given_particles(ids, evt_hits)
        energies   = sel_hits.groupby(['particle_id'])[['energy']].sum()
        energies   = energies.reset_index()
        energy_sel = energies[rf.greater_or_equal(energies.energy, 0.476443, allowed_error=1.e-6)]

        sel_vol_name_e  = sel_vol_name[sel_vol_name.particle_id.isin(energy_sel.particle_id)]

        primaries = evt_parts[evt_parts.primary == True]
        sel_all   = sel_vol_name_e[sel_vol_name_e.mother_id.isin(primaries.particle_id.values)]
        if len(sel_all) == 0:
            sel    = False
            true_p = []

        ids      = sel_all.particle_id.values
        sel_hits = mcf.find_hits_of_given_particles(ids, evt_hits)
        sel_hits = sel_hits.groupby(['particle_id'])
        for _, df in sel_hits:
            hit_positions = np.array([df.x.values, df.y.values, df.z.values]).transpose()
            true_p.append(np.average(hit_positions, axis=0, weights=df.energy))

        if not select:
            assert len(true_pos) == 0

        if (len(true_p) == 1) & (evt_hits.energy.sum() > 0.511):
            assert select        == False
            assert len(true_pos) == 0

        if len(true_pos) == 2:
            assert evt_hits.energy.sum() > 0.511


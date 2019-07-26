import os
import math
import numpy  as np
import pandas as pd

from hypothesis.strategies import floats
from hypothesis            import given

from .           import reco_functions   as rf
from .           import mctrue_functions as mcf
from .. database import load_db          as db


f             = floats(min_value=1,     max_value=2)
f_lower       = floats(min_value=0,     max_value=1)
allowed_error = floats(min_value=1.e-8, max_value=1.e-6)


@given(f, f_lower)
def test_lower_or_equal(f1, f2):
   assert rf.lower_or_equal(f2, f1)

   
@given(f, f_lower)
def test_greater_or_equal(f1, f2):
   assert rf.greater_or_equal(f1, f2)


@given(f, allowed_error)
def test_allowed_error_in_inequality(f1, err):

    f2 = f1 + 2*err
    assert not rf.greater_or_equal(f1, f2, err)


def test_from_cartesian_to_cyl():
    """
    This test checks that the function from_cartesian_to_cyl transforms
    an array of cartesian coordinates into cylindrical coordinates.
    For example:
    cart_array = (       1,        1, 1)
    cyl_array  = (1.414213, 0.785398, 1)
    """
    cart_pos = np.array([np.array([1, 1, 1])])
    cyl_pos  = rf.from_cartesian_to_cyl(cart_pos)
    assert np.isclose(cyl_pos[0][0], math.sqrt(2))
    assert np.isclose(cyl_pos[0][1], math.pi/4)
    assert np.isclose(cyl_pos[0][2], cart_pos[0][2])


x = floats(min_value=-1000, max_value=1000)
y = floats(min_value=-1000, max_value=1000)
z = floats(min_value=-1000, max_value=1000)

@given(x, y, z)
def test_from_cartesian_to_cyl2(x, y, z):
    """
    This tests checks properties of the cylindrical coordinates:
    that r is always positive and phi is [-pi, pi]
    """
    cart_pos = np.array([np.array([x, y, z])])
    cyl_pos  = rf.from_cartesian_to_cyl(cart_pos)
    assert  cyl_pos[0][0] >= 0
    assert (cyl_pos[0][1] >= -math.pi) & (cyl_pos[0][1] <= math.pi)


def test_find_SiPMs_over_threshold(ANTEADATADIR):
    PATH_IN      = os.path.join(ANTEADATADIR, 'ring_test_new_tbs.h5')
    sns_response = pd.read_hdf(PATH_IN, 'MC/waveforms')
    threshold    = 2
    df_over_thr  = rf.find_SiPMs_over_threshold(sns_response, threshold)
    df_below_thr = sns_response.groupby(['event_id','sensor_id'])[['charge']].sum()
    df_below_thr = df_below_thr[df_below_thr.charge <= threshold].reset_index()
    assert len(df_over_thr) == len(sns_response) - len(df_below_thr)


@given(x, y, z)
def test_find_closest_sipm(x, y, z):
    DataSiPM     = db.DataSiPM('petalo', 0)
    DataSiPM_idx = DataSiPM.set_index('SensorID')
    point        = np.array([x, y, z])
    closest_sipm = rf.find_closest_sipm(point, DataSiPM_idx)

    sns_positions = np.array([DataSiPM_idx.X.values, DataSiPM_idx.Y.values, DataSiPM_idx.Z.values]).transpose()
    subtr         = np.subtract(point, sns_positions)
    distances     = [np.linalg.norm(d) for d in subtr]
    min_dist      = np.min(distances)
    min_sipm      = np.isclose(distances, min_dist)
    closest_sipm2 = DataSiPM_idx[min_sipm]

    assert np.all(closest_sipm) == np.all(closest_sipm2)
    assert min_dist > 0


def test_divide_sipms_in_two_hemispheres(ANTEADATADIR):
    DataSiPM      = db.DataSiPM('petalo', 0)
    DataSiPM_idx  = DataSiPM.set_index('SensorID')
    PATH_IN       = os.path.join(ANTEADATADIR, 'ring_test_new_tbs.h5')
    sns_response  = pd.read_hdf(PATH_IN, 'MC/waveforms')
    max_sns       = sns_response[sns_response.charge == sns_response.charge.max()]
    max_sipm      = DataSiPM_idx.loc[max_sns.sensor_id]
    max_pos       = np.array([max_sipm.X.values, max_sipm.Y.values, max_sipm.Z.values]).transpose()[0]
    sipms         = DataSiPM_idx.loc[sns_response.sensor_id]
    sns_positions = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).transpose()
    sns_charges   = sns_response.charge

    pos1, pos2, q1, q2 = rf.divide_sipms_in_two_hemispheres(sns_positions, sns_charges, max_pos)

    scalar_prod1 = np.array([np.dot(max_pos, p1) for p1 in pos1])
    scalar_prod2 = np.array([np.dot(max_pos, p2) for p2 in pos2])

    assert len(pos1) == len(q1)
    assert len(pos2) == len(q2)
    assert (scalar_prod1 > 0).all()
    assert (scalar_prod2 < 0).all()


def test_assign_sipms_to_gammas(ANTEADATADIR):
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

        if not select: continue
        if (len(true_pos) == 1) & (evt_hits.energy.sum() > 0.511): continue

        waveforms = sel_df[sel_df.event_id == evt]
        if len(waveforms) == 0: continue

        pos1, pos2, q1, q2 = rf.assign_sipms_to_gammas(waveforms, true_pos, DataSiPM_idx)

        sipms           = DataSiPM_idx.loc[sns_response.sensor_id]
        sns_closest_pos = [np.array([rf.find_closest_sipm(pos, sipms).X.values,
                                     rf.find_closest_sipm(pos, sipms).Y.values,
                                     rf.find_closest_sipm(pos, sipms).Z.values]).transpose()[0] for pos in true_pos]
        scalar_prod1 = np.array([np.dot(sns_closest_pos[0], p1) for p1 in pos1])

        assert len(q1) == len(pos1)
        assert len(sns_closest_pos) <= 2
        assert (scalar_prod1 > 0).all()

        if len(true_pos) < 2:
            assert len(q2)   == 0
            assert len(pos2) == 0
        else:
            scalar_prod2 = np.array([np.dot(sns_closest_pos[0], p2) for p2 in pos2])
            assert len(q2) == len(pos2)
            assert (scalar_prod2 < 0).all()

import os
import math
import numpy                 as np
import pandas                as pd
import hypothesis.strategies as st

from hypothesis  import given
from .           import reco_functions   as rf
from .           import mctrue_functions as mcf
from .. database import load_db          as db


f             = st.floats(min_value=1,     max_value=2)
f_lower       = st.floats(min_value=0,     max_value=1)
allowed_error = st.floats(min_value=1.e-8, max_value=1.e-6)


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


x = st.floats(min_value=-1000, max_value=1000)
y = st.floats(min_value=-1000, max_value=1000)
z = st.floats(min_value=-1000, max_value=1000)

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
    """
    Checks that the number of SiPMs that detected charge above
    the threshold plus the number of SiPMs that detected charge
    below the threshold are equal to the total number of sensors.
    """
    PATH_IN      = os.path.join(ANTEADATADIR, 'ring_test_new_tbs.h5')
    sns_response = pd.read_hdf(PATH_IN, 'MC/waveforms')
    threshold    = 2
    df_over_thr  = rf.find_SiPMs_over_threshold(sns_response, threshold)
    df_below_thr = sns_response.groupby(['event_id','sensor_id'])[['charge']].sum()
    df_below_thr = df_below_thr[df_below_thr.charge <= threshold].reset_index()
    assert len(df_over_thr) == len(sns_response) - len(df_below_thr)


@given(x, y, z)
def test_find_closest_sipm(x, y, z):
    """
    Checks that the function find_closest_sipm returns the position of the
    closest SiPM to a given point, and the distance between them is a positive
    quantity.
    If the point is in the center of the coordinate system or far from the
    sensors, the function takes more than one sipm because many of them are
    at the same distance, so a range for the radius is imposed.
    """
    r = np.sqrt(x**2+y**2)
    if r < 1 or r > 200: return

    DataSiPM     = db.DataSiPM('petalo', 0)
    DataSiPM_idx = DataSiPM.set_index('SensorID')
    point        = np.array([x, y, z])
    closest_sipm = rf.find_closest_sipm(point, DataSiPM_idx)
    sns_pos1     = np.array([closest_sipm.X.values, closest_sipm.Y.values, closest_sipm.Z.values]).transpose()
    dist1        = [np.linalg.norm(np.subtract(point, pos)) for pos in sns_pos1][0]

    random_sns = DataSiPM_idx.iloc[[np.random.randint(len(DataSiPM))]]
    sns_pos2   = np.array([random_sns.X.values, random_sns.Y.values, random_sns.Z.values]).transpose()
    dist2      = np.linalg.norm(np.subtract(point, sns_pos2))

    assert dist1 < dist2
    assert dist1 > 0


a = st.floats(min_value=-1000, max_value=1000)
b = st.floats(min_value=-1000, max_value=1000)
c = st.floats(min_value=-1000, max_value=1000)

elements =st.tuples(st.floats  (min_value=-1000, max_value=1000),
                    st.floats  (min_value=-1000, max_value=1000),
                    st.floats  (min_value=-1000, max_value=1000),
                    st.integers(min_value=1,     max_value=10000))
l = st.lists(elements, min_size=2, max_size=1000)

@given(x, y, z, a, b, c, l)
def test_divide_sipms_in_two_hemispheres(x, y, z, a, b, c, l):
    """
    This test checks that given a point, all the positions of a list of sensor
    positions and charges are divided in two hemispheres.
    The way of testing it is by checking that the scalar product of the position
    of the sensors in the same hemisphere as the point, is positive.
    Every point in the center of coordinates is neglected in order to avoid
    null scalar prod.
    """
    point         = np.array([x, y, z])
    max_pos       = np.array([a, b, c])

    if np.isclose(  point.all(), 0.): return
    if np.isclose(max_pos.all(), 0.): return
    if (np.all(el)==0. for el in l) : return

    sns_positions = np.array([el[:3] for el in l])
    sns_charges   = np.array([el [3] for el in l])

    pos1, pos2, q1, q2 = rf.divide_sipms_in_two_hemispheres(sns_positions, sns_charges, point)

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


def test_select_coincidences(ANTEADATADIR):
    """
    This test checks that the function select_coincidences returns the position
    of the true events and the position and charge of the sensors that detected
    charge only when coincidences are produced.
    """
    PATH_IN      = os.path.join(ANTEADATADIR, 'ring_test_new_tbs.h5')
    DataSiPM     = db.DataSiPM('petalo', 0)
    DataSiPM_idx = DataSiPM.set_index('SensorID')
    sns_response = pd.read_hdf(PATH_IN, 'MC/waveforms')
    threshold    = 2
    charge_range = (1000, 1400)
    radius_range = ( 165,  195)
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

        pos1, pos2, q1, q2, true_pos1, true_pos2 = rf.select_coincidences(waveforms, charge_range, DataSiPM_idx, evt_parts, evt_hits)

        if len(true_pos) == 2:
            scalar_prod1 = np.array([np.dot(true_pos1, p1) for p1 in pos1])
            scalar_prod2 = np.array([np.dot(true_pos2, p2) for p2 in pos2])
            assert (scalar_prod1 > 0).all()
            assert (scalar_prod2 > 0).all()

            dist11 = np.array([np.linalg.norm(true_pos1 - p1) for p1 in pos1])
            dist21 = np.array([np.linalg.norm(true_pos2 - p1) for p1 in pos1])
            dist12 = np.array([np.linalg.norm(true_pos1 - p2) for p2 in pos2])
            dist22 = np.array([np.linalg.norm(true_pos2 - p2) for p2 in pos2])
            assert ((dist11 - dist21) < 0).all()
            assert ((dist22 - dist12) < 0).all()

            assert len(pos1) > 0 and len(pos2) > 0
            assert len(true_pos1) == len(true_pos2)
            assert len(pos1)      == len(q1)
            assert len(pos2)      == len(q2)

            true_pos_cyl = rf.from_cartesian_to_cyl(np.array([true_pos1, true_pos2]))
            true_r       = np.array([i[0] for i in true_pos_cyl])
            assert (true_r > radius_range[0]).all() and (true_r < radius_range[1]).all()

        else:
            assert not true_pos1 and not true_pos2
            assert not len(pos1) and not len(pos2)
            assert not len(q1)   and not len(q2)

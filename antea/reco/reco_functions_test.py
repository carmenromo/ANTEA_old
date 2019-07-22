import os
import math
import numpy  as np
import pandas as pd

from hypothesis.strategies import floats
from hypothesis            import given

from .           import reco_functions as rf
from .. database import load_db        as db

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
    cart_pos = np.array([np.array([1, 1, 1])])
    cyl_pos  = rf.from_cartesian_to_cyl(cart_pos)
    assert cyl_pos[0][0] == math.sqrt(2)
    assert cyl_pos[0][1] == math.pi/4
    assert cyl_pos[0][2] == cart_pos[0][2]


def test_find_SiPMs_over_threshold(ANTEADATADIR):
    PATH_IN      = os.path.join(ANTEADATADIR, 'ring_test_new_tbs.h5')
    sns_response = pd.read_hdf(PATH_IN, 'MC/waveforms')
    threshold    = 2
    df_over_thr  = rf.find_SiPMs_over_threshold(sns_response, threshold)
    df_below_thr = sns_response.groupby(['event_id','sensor_id'])[['charge']].sum()
    df_below_thr = df_below_thr[df_below_thr.charge <= threshold].reset_index()
    assert len(df_over_thr) == len(sns_response) - len(df_below_thr)


def test_find_closest_sipm():
    DataSiPM     = db.DataSiPM('petalo', 0)
    DataSiPM_idx = DataSiPM.set_index('SensorID')
    point        = np.array([26.70681, -183.4894, -20.824465])
    closest_sipm = rf.find_closest_sipm(point, DataSiPM_idx)

    sns_positions = np.array([DataSiPM_idx.X.values, DataSiPM_idx.Y.values, DataSiPM_idx.Z.values]).transpose()
    subtr         = np.subtract(point, sns_positions)
    distances     = [np.linalg.norm(d) for d in subtr]
    min_dist      = np.min(distances)
    min_sipm      = np.isclose(distances, min_dist)
    closest_sipm2 = DataSiPM_idx[min_sipm]
    assert np.all(closest_sipm) == np.all(closest_sipm2)


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

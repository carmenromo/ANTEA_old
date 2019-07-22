import os
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

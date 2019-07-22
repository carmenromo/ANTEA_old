import os
import numpy  as np
import pandas as pd

from hypothesis.strategies import floats
from hypothesis            import given

from . import reco_functions as rf

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
    wvf_df       = pd.read_hdf(PATH_IN, 'MC/waveforms')
    threshold    = 2
    df_over_thr  = rf.find_SiPMs_over_threshold(wvf_df, threshold)
    df_below_thr = wvf_df.groupby(['event_id','sensor_id'])[['charge']].sum()
    df_below_thr = df_below_thr[df_below_thr.charge <= threshold].reset_index()
    assert len(df_over_thr) == len(wvf_df) - len(df_below_thr)

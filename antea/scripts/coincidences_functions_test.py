import os

from .. database import load_db as db
from . coincidences_functions import characterize_coincidences


def test_characterize_coincidences(ANTEADATADIR, config_tmpdir):
    """
    Checks that the script to reconstruct the coincidences runs.
    """
    input_file  = os.path.join(ANTEADATADIR, 'full_body_coinc.h5')
    output_file = os.path.join(config_tmpdir, 'test_run_script')
    rpos_file   = os.path.join(ANTEADATADIR, 'r_table_full_body.h5')

    try:
        characterize_coincidences(input_file, output_file, rpos_file)
    except:
        raise AssertionError('Function reconstruct_coincidences_script has failed running.')



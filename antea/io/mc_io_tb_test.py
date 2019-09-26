import os
import numpy  as np
import tables as tb

from . mc_io_tb import read_mcsns_response
from . mc_io_tb import read_mcTOFsns_response


def test_read_sensor_response(ANTEADATADIR):
    test_file = os.path.join(ANTEADATADIR,'ring_test_tb.h5')

    mc_sensor_dict = read_mcsns_response(test_file)
    waveforms = mc_sensor_dict[0]

    n_of_sensors = 796
    sensor_id    = 4162

    assert len(waveforms) == n_of_sensors
    assert waveforms[sensor_id].times == np.array([0.])
    assert waveforms[sensor_id].charges == np.array([8.])

def test_read_sensor_tof_response(ANTEADATADIR):
    test_file = os.path.join(ANTEADATADIR,'ring_test_tb.h5')

    mc_sensor_dict = read_mcTOFsns_response(test_file)
    waveforms = mc_sensor_dict[0]

    sensor_id = 4371
    bin_width = waveforms[-sensor_id].bin_width
    times = np.array([358, 1562, 5045, 5229, 5960, 6311, 14192]) * bin_width
    charges = np.array([1, 1, 1, 1, 1, 1, 1])

    assert np.allclose(waveforms[-sensor_id].times, times)
    assert np.allclose(waveforms[-sensor_id].charges, charges)

def test_read_last_sensor_response(ANTEADATADIR):
    test_file = os.path.join(ANTEADATADIR,'ring_test_tb.h5')

    mc_sensor_dict = read_mcsns_response(test_file)
    waveforms = mc_sensor_dict[0]

    with tb.open_file(test_file, mode='r') as h5in:
        last_written_id = h5in.root.MC.sensor_positions[-1][0]
        last_read_id = list(waveforms.keys())[-1]

        assert last_read_id == last_written_id

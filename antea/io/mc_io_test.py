import os
import numpy  as np
import tables as tb
import pandas as pd

from invisible_cities.core import system_of_units as units

from . mc_io import load_mchits, load_mcparticles, load_configuration
from . mc_io import load_mcsns_response
from . mc_io import load_mcTOFsns_response
from . mc_io import mc_writer, mc_sns_response_writer


def test_read_sensor_response(ANTEADATADIR):
    test_file = os.path.join(ANTEADATADIR,'ring_test.h5')

    waveforms = load_mcsns_response(test_file)

    evt_id       = 16
    n_of_sensors = 90
    sensor_id, charge = 2945, 3

    evt_waveforms = waveforms[waveforms.event_id == evt_id]
    sns_charge    = evt_waveforms[evt_waveforms.sensor_id == sensor_id].charge.sum()

    assert len(evt_waveforms.sensor_id.unique()) == n_of_sensors
    assert sns_charge == charge


def test_read_sensor_tof_response(ANTEADATADIR):
    test_file = os.path.join(ANTEADATADIR,'ring_test.h5')

    waveforms = load_mcTOFsns_response(test_file)

    evt_id    = 16
    sensor_id = -2945
    times     = np.array([200, 1025, 3271])
    charges   = np.array([1, 1, 1])

    evt_waveforms = waveforms[waveforms.event_id == evt_id]
    evt_sns_waveforms = evt_waveforms[evt_waveforms.sensor_id == sensor_id]

    assert np.allclose(evt_sns_waveforms.time_bin, times)
    assert np.allclose(evt_sns_waveforms.charge, charges)


def test_write_mc_info(ANTEADATADIR, output_tmpdir):
    test_file_in  = os.path.join(ANTEADATADIR,'ring_test.h5')
    test_file_out = os.path.join(output_tmpdir, 'test_output.h5')

    writer = mc_writer(test_file_in, test_file_out)
    hits_in = load_mchits(test_file_in)
    events = hits_in.event_id.unique()
    events_to_write = events[:5]
    for evt in events_to_write:
        writer(evt)
    writer.close_file()

    hits_out        = load_mchits(test_file_out)
    hits_in_written = hits_in[np.isin(hits_in.event_id, events_to_write)]

    particles_out = load_mcparticles(test_file_out)
    particles_in  = load_mcparticles(test_file_in)
    particles_in_written = particles_in[np.isin(particles_in.event_id, events_to_write)]

    conf_out = load_configuration(test_file_out)
    conf_in  = load_configuration(test_file_in)

    assert hits_out.equals(hits_in_written)
    assert particles_out.equals(particles_in_written)
    assert conf_out.equals(conf_in)


def test_write_sns_info(ANTEADATADIR, output_tmpdir):
    test_file_in  = os.path.join(ANTEADATADIR,'ring_test.h5')

    event_id = 0
    sns_response = {event_id : {1000: 1, 1001: 3}}

    writer = mc_sns_response_writer(test_file_in, 'test_sns_response')
    writer(sns_response, 0)
    writer.close_file()

    sns_response_written = pd.read_hdf(test_file_in, 'MC/test_sns_response')

    events = sns_response_written.event_id.unique()
    sns    = sns_response_written.sensor_id
    charge = sns_response_written.charge

    assert events        == np.array([event_id])
    assert np.all(sns    == list(sns_response[event_id].keys()))
    assert np.all(charge == list(sns_response[event_id].values()))

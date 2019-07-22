import os
import pytest
import numpy  as np
import tables as tb

from pandas      import DataFrame
from collections import namedtuple

from invisible_cities.core.system_of_units_c import units

db_data  = namedtuple('db_data', 'detector nsipms')


@pytest.fixture(scope = 'session')
def ANTEADIR():
    return os.environ['ANTEADIR']


@pytest.fixture(scope = 'session')
def ANTEADATADIR(ANTEADIR):
    return os.path.join(ANTEADIR, "testdata/")


@pytest.fixture(scope = 'session')
def config_tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp('configure_tests')


@pytest.fixture(scope = 'session')
def output_tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp('output_files')


@pytest.fixture(scope='session',
                params=[db_data('petalo' ,  3500)],
                ids=["petit"])
def db(request):
    return request.param


@pytest.fixture(scope='session',
                params=[db_data('petalo' ,  102304)],
                ids=["sim-only"])
def db_sim_only(request):
    return request.param

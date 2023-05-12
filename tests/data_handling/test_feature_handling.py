# -*- coding: utf-8 -*-
"""pytests for feature handling / parsing"""

from sup3r.preprocessing.feature_handling import (UWind, BVFreqMon,
                                                  BVFreqSquaredH5,
                                                  BVFreqSquaredNC,
                                                  ClearSkyRatioH5)
from sup3r.preprocessing.data_handling import (DataHandlerH5,
                                               DataHandlerNC,
                                               DataHandlerH5SolarCC,
                                               DataHandlerNCforCC)


WTK_FEAT = ['windspeed_100m', 'winddirection_100m',
            'windspeed_200m', 'winddirection_200m',
            'temperature_100m', 'temperature_200m',
            'pressure_100m', 'pressure_200m',
            'inversemoninobukhovlength_2m']

WRF_FEAT = ['U', 'V', 'T', 'UST', 'HFX', 'HGT']

ERA_FEAT = ['u', 'v']

NSRDB_FEAT = ['ghi', 'clearsky_ghi', 'wind_speed', 'wind_direction']

CC_FEAT = ['ua', 'uv', 'tas', 'hurs', 'zg', 'orog', 'ta']


def test_feature_inputs_h5():
    """Test basic H5 feature name / inputs parsing"""
    out = DataHandlerH5.get_inputs_recursive('topography', WTK_FEAT)
    assert out == ['topography']

    out = DataHandlerH5.get_inputs_recursive('U_100m', WTK_FEAT)
    assert out == ['windspeed_100m', 'winddirection_100m', 'lat_lon']

    out = DataHandlerH5.get_inputs_recursive('V_100m', WTK_FEAT)
    assert out == ['windspeed_100m', 'winddirection_100m', 'lat_lon']

    out = DataHandlerH5.get_inputs_recursive('P_100m', WTK_FEAT)
    assert out == ['pressure_100m']

    out = DataHandlerH5.get_inputs_recursive('BVF_MO_200m', WTK_FEAT)
    assert out == ['temperature_200m', 'temperature_100m', 'pressure_200m',
                   'pressure_100m', 'inversemoninobukhovlength_2m']

    out = DataHandlerH5.get_inputs_recursive('BVF2_200m', WTK_FEAT)
    assert out == ['temperature_200m', 'temperature_100m',
                   'pressure_200m', 'pressure_100m']


def test_feature_inputs_nc():
    """Test basic WRF NC feature name / inputs parsing"""
    out = DataHandlerNC.get_inputs_recursive('U_100m', WRF_FEAT)
    assert out == ['U_100m']

    out = DataHandlerNC.get_inputs_recursive('topography', WRF_FEAT)
    assert out == ['HGT']

    out = DataHandlerNC.get_inputs_recursive('BVF_MO_200m', WRF_FEAT)
    assert out == ['T_200m', 'T_100m', 'UST', 'HFX']

    out = DataHandlerNC.get_inputs_recursive('BVF2_200m', WRF_FEAT)
    assert out == ['T_200m', 'T_100m']

    out = DataHandlerNC.get_inputs_recursive('windspeed_200m', WRF_FEAT)
    assert out == ['U_200m', 'V_200m', 'lat_lon']


def test_feature_inputs_lowercase():
    """Test basic NC feature name / inputs parsing with lowercase raw
    features."""
    out = DataHandlerNC.get_inputs_recursive('windspeed_200m', ERA_FEAT)
    assert out == ['U_200m', 'V_200m', 'lat_lon']


def test_feature_inputs_cc():
    """Test basic CC feature name / inputs parsing"""
    out = DataHandlerNCforCC.get_inputs_recursive('U_100m', CC_FEAT)
    assert out == ['ua_100m']

    out = DataHandlerNCforCC.get_inputs_recursive('topography', CC_FEAT)
    assert out == ['orog']

    out = DataHandlerNCforCC.get_inputs_recursive('temperature_2m', CC_FEAT)
    assert out == ['tas']

    out = DataHandlerNCforCC.get_inputs_recursive('temperature_100m', CC_FEAT)
    assert out == ['ta_100m']

    out = DataHandlerNCforCC.get_inputs_recursive('pressure_100m', CC_FEAT)
    assert out == ['plev_100m']

    out = DataHandlerNCforCC.get_inputs_recursive('relativehumidity_2m',
                                                  CC_FEAT)
    assert out == ['hurs']


def test_feature_inputs_solar():
    """Test solar H5 (nsrdb) feature name / inputs parsing"""
    out = DataHandlerH5SolarCC.get_inputs_recursive('clearsky_ratio',
                                                    NSRDB_FEAT)
    assert out == ['clearsky_ghi', 'ghi']
    out = DataHandlerH5SolarCC.get_inputs_recursive('U',
                                                    NSRDB_FEAT)
    assert out == ['wind_speed', 'wind_direction', 'lat_lon']


def test_lookup_h5():
    """Test methods lookup for base h5 files (wtk)"""
    out = DataHandlerH5.lookup('U_100m', 'inputs', WTK_FEAT)
    assert out == UWind.inputs

    out = DataHandlerH5.lookup('BVF_MO_200m', 'inputs', WTK_FEAT)
    assert out == BVFreqMon.inputs

    out = DataHandlerH5.lookup('BVF2_200m', 'inputs', WTK_FEAT)
    assert out == BVFreqSquaredH5.inputs


def test_lookup_nc():
    """Test methods lookup for base NC files (wrf)"""
    out = DataHandlerNC.lookup('BVF2_200m', 'inputs', WTK_FEAT)
    assert out == BVFreqSquaredNC.inputs


def test_lookup_cc():
    """Test methods lookup for CC NC files (cmip6)"""
    out = DataHandlerNCforCC.lookup('temperature_2m', 'inputs', CC_FEAT)
    assert out('temperature_2m') == ['tas']


def test_lookup_solar():
    """Test solar H5 (nsrdb) feature method lookup"""
    out = DataHandlerH5SolarCC.lookup('clearsky_ratio', 'inputs', NSRDB_FEAT)
    assert out == ClearSkyRatioH5.inputs

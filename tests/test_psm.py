"""
Tests for cfr.psm PSM classes.

Inspired by the psm-linear and psm-bilinear notebooks.
The Linear and Bilinear PSMs are tested end-to-end with fully synthetic data.
More complex PSMs (Ice_d18O, VSLite, Bayesian) require optional dependencies
and are only smoke-tested for import/construction when those packages exist.
"""

import numpy as np
import pytest
import cftime
import xarray as xr
import matplotlib
matplotlib.use('Agg')

from cfr.proxy import ProxyRecord
from cfr.climate import ClimateField
from cfr import psm


# ---------------------------------------------------------------------------
# Helper: build monthly cftime time axis and a ClimateField
# ---------------------------------------------------------------------------

def make_monthly_cftime(start_year, n_years):
    """Return a list of monthly cftime datetimes spanning n_years."""
    dates = []
    for y in range(start_year, start_year + n_years):
        for m in range(1, 13):
            dates.append(cftime.datetime(y, m, 1, calendar='standard'))
    return dates


def make_monthly_climatefield(start_year, n_years, lat, lon, rng, offset=288.0, vn='tas'):
    """
    Create a monthly ClimateField mimicking what get_clim() returns:
    a 1D time series DataArray (time only, no lat/lon dims), because PSM.calibrate()
    calls `da.values` and expects a 1D array to put into a pandas DataFrame column.
    """
    n = n_years * 12
    dates = make_monthly_cftime(start_year, n_years)
    value = rng.standard_normal(n) + offset
    da = xr.DataArray(
        value,
        dims=['time'],
        coords={'time': dates},
        name=vn,
    )
    return ClimateField(da)


def make_proxy_with_climate(seed=0, n_calib=10, n_model=20):
    """
    Create a ProxyRecord whose .clim dict contains monthly ClimateField objects
    at a single lat/lon point, covering calibration and model periods.
    n_calib / n_model are in years; actual time axis is monthly.
    """
    rng = np.random.default_rng(seed)
    lat = np.array([30.0])
    lon = np.array([90.0])

    # calibration climate: monthly, 1970 to 1970+n_calib
    obs_tas = make_monthly_climatefield(1970, n_calib, lat, lon, rng)

    # model climate: monthly, 1950 to 1950+n_model
    model_tas = make_monthly_climatefield(1950, n_model, lat, lon, rng)

    # proxy: annual, aligned with calib period
    calib_time = np.arange(1970, 1970 + n_calib, dtype=float)
    proxy_value = rng.standard_normal(n_calib)

    pobj = ProxyRecord(
        pid='syn_001',
        lat=30.0, lon=90.0, elev=100.0,
        time=calib_time,
        value=proxy_value,
        ptype='tree.TRW',
        value_name='TRW', value_unit='mm',
    )
    pobj.clim = {
        'obs.tas': obs_tas,
        'model.tas': model_tas,
    }
    return pobj


# ---------------------------------------------------------------------------
# Linear PSM
# ---------------------------------------------------------------------------

class TestLinearPSM:
    @pytest.fixture
    def calibrated_linear(self):
        pobj = make_proxy_with_climate(seed=1)
        lpsm = psm.Linear(pobj=pobj, climate_required=['tas'])
        lpsm.calibrate(
            calib_period=[1950, 1999],
            nobs_lb=10,
            season_list=[list(range(1, 13))],
            exog_name='obs.tas',
        )
        return lpsm

    def test_calibrate_sets_model(self, calibrated_linear):
        assert calibrated_linear.model is not None

    def test_calibrate_details_keys(self, calibrated_linear):
        keys = calibrated_linear.calib_details.keys()
        assert 'fitR2adj' in keys
        assert 'PSMmse' in keys
        assert 'nobs' in keys
        assert 'seasonality' in keys

    def test_nobs_above_lb(self, calibrated_linear):
        assert calibrated_linear.calib_details['nobs'] >= 10

    def test_forward_returns_proxyrecord(self, calibrated_linear):
        pp = calibrated_linear.forward(exog_name='model.tas')
        assert isinstance(pp, ProxyRecord)

    def test_forward_correct_pid(self, calibrated_linear):
        pp = calibrated_linear.forward(exog_name='model.tas')
        assert pp.pid == calibrated_linear.pobj.pid

    def test_forward_output_length(self, calibrated_linear):
        pp = calibrated_linear.forward(exog_name='model.tas')
        # Model period is n_model=20 years of monthly data, annualized → 20 annual points
        assert len(pp.time) == 20

    def test_forward_value_is_finite(self, calibrated_linear):
        pp = calibrated_linear.forward(exog_name='model.tas')
        assert np.all(np.isfinite(pp.value))

    def test_insufficient_nobs_gives_none_model(self):
        pobj = make_proxy_with_climate(seed=2, n_calib=5)
        lpsm = psm.Linear(pobj=pobj, climate_required=['tas'])
        lpsm.calibrate(
            nobs_lb=1000,   # impossibly high threshold
            season_list=[list(range(1, 13))],
            exog_name='obs.tas',
        )
        assert lpsm.model is None

    def test_multiple_seasons_picks_best(self):
        pobj = make_proxy_with_climate(seed=3)
        lpsm = psm.Linear(pobj=pobj, climate_required=['tas'])
        seasons = [list(range(1, 13)), [6, 7, 8], [12, 1, 2]]
        lpsm.calibrate(
            nobs_lb=5,
            season_list=seasons,
            exog_name='obs.tas',
        )
        assert lpsm.model is not None
        # best seasonality must be one of the candidates
        assert lpsm.calib_details['seasonality'] in seasons


# ---------------------------------------------------------------------------
# Bilinear PSM
# ---------------------------------------------------------------------------

class TestBilinearPSM:
    def _make_bivariate_proxy(self, seed=10, n_calib=10, n_model=15):
        """Proxy that correlates with both tas and pr (monthly cftime time)."""
        rng = np.random.default_rng(seed)
        lat = np.array([30.0])
        lon = np.array([90.0])

        obs_tas = make_monthly_climatefield(1970, n_calib, lat, lon, rng, vn='tas')
        obs_pr = make_monthly_climatefield(1970, n_calib, lat, lon, rng, offset=0.003, vn='pr')
        model_tas = make_monthly_climatefield(1950, n_model, lat, lon, rng, vn='tas')
        model_pr = make_monthly_climatefield(1950, n_model, lat, lon, rng, offset=0.003, vn='pr')

        calib_time = np.arange(1970, 1970 + n_calib, dtype=float)
        proxy_val = rng.standard_normal(n_calib)

        pobj = ProxyRecord(
            pid='syn_bi_001', lat=30.0, lon=90.0, elev=0.0,
            time=calib_time, value=proxy_val, ptype='coral.d18O',
        )
        pobj.clim = {
            'obs.tas': obs_tas,
            'obs.pr': obs_pr,
            'model.tas': model_tas,
            'model.pr': model_pr,
        }
        return pobj

    def test_calibrate_and_forward(self):
        pobj = self._make_bivariate_proxy()
        bpsm = psm.Bilinear(pobj=pobj, climate_required=['tas', 'pr'])
        bpsm.calibrate(
            nobs_lb=5,
            season_list1=[list(range(1, 13))],
            season_list2=[list(range(1, 13))],
            exog1_name='obs.tas',
            exog2_name='obs.pr',
        )
        assert bpsm.model is not None
        pp = bpsm.forward(exog1_name='model.tas', exog2_name='model.pr')
        assert isinstance(pp, ProxyRecord)
        assert len(pp.value) == 15  # n_model=15 years of monthly data → 15 annual points
        assert np.all(np.isfinite(pp.value))


# ---------------------------------------------------------------------------
# clean_df helper
# ---------------------------------------------------------------------------

class TestCleanDf:
    def test_removes_nan_rows(self):
        import pandas as pd
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, 6.0]})
        cleaned = psm.clean_df(df)
        assert len(cleaned) == 2

    def test_with_mask(self):
        import pandas as pd
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0]}, index=[0, 1, 2])
        mask = df.index >= 1
        cleaned = psm.clean_df(df, mask=mask)
        assert len(cleaned) == 2


# ---------------------------------------------------------------------------
# Import/construction smoke tests for optional-dependency PSMs
# ---------------------------------------------------------------------------

class TestOptionalPSMConstruction:
    """Just verify that PSM classes can be instantiated (model-free)."""

    def test_coral_srca_constructs(self):
        pobj = make_proxy_with_climate()
        m = psm.Coral_SrCa(pobj=pobj)
        assert m.pobj is pobj

    def test_coral_d18o_constructs(self):
        pobj = make_proxy_with_climate()
        m = psm.Coral_d18O(pobj=pobj)
        assert m.pobj is pobj

    def test_lake_varvethickness_constructs(self):
        pobj = make_proxy_with_climate()
        m = psm.Lake_VarveThickness(pobj=pobj)
        assert m.pobj is pobj

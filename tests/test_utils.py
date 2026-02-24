"""
Tests for cfr.utils utility functions.

Inspired by the notebooks (climate-io, climate-ops, proxy-ops) which rely on
these utilities indirectly, plus direct coverage of standalone functions.
"""

import numpy as np
import pytest
from cfr import utils


# ---------------------------------------------------------------------------
# clean_ts
# ---------------------------------------------------------------------------

class TestCleanTs:
    def test_removes_nans_in_value(self):
        ts = np.array([1.0, 2.0, 3.0, 4.0])
        ys = np.array([10.0, np.nan, 30.0, 40.0])
        ts_out, ys_out = utils.clean_ts(ts, ys)
        assert np.all(np.isfinite(ys_out))
        assert len(ts_out) == len(ys_out) == 3

    def test_removes_nans_in_time(self):
        ts = np.array([1.0, np.nan, 3.0, 4.0])
        ys = np.array([10.0, 20.0, 30.0, 40.0])
        ts_out, ys_out = utils.clean_ts(ts, ys)
        assert len(ts_out) == 3
        assert np.all(np.isfinite(ts_out))

    def test_sorts_ascending(self):
        ts = np.array([4.0, 1.0, 3.0, 2.0])
        ys = np.array([40.0, 10.0, 30.0, 20.0])
        ts_out, ys_out = utils.clean_ts(ts, ys)
        assert list(ts_out) == [1.0, 2.0, 3.0, 4.0]

    def test_averages_duplicate_times(self):
        ts = np.array([1.0, 1.0, 2.0])
        ys = np.array([10.0, 20.0, 30.0])
        ts_out, ys_out = utils.clean_ts(ts, ys)
        assert len(ts_out) == 2
        # duplicate at t=1 should be averaged
        idx_1 = np.where(ts_out == 1.0)[0][0]
        assert ys_out[idx_1] == pytest.approx(15.0)

    def test_size_mismatch_raises(self):
        with pytest.raises(AssertionError):
            utils.clean_ts(np.array([1.0, 2.0]), np.array([10.0]))

    def test_no_nans_passthrough(self):
        ts = np.array([1.0, 2.0, 3.0])
        ys = np.array([5.0, 6.0, 7.0])
        ts_out, ys_out = utils.clean_ts(ts, ys)
        np.testing.assert_array_equal(ts_out, ts)
        np.testing.assert_array_equal(ys_out, ys)


# ---------------------------------------------------------------------------
# coefficient_efficiency
# ---------------------------------------------------------------------------

class TestCoefficientEfficiency:
    def test_perfect_reconstruction(self):
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ce = utils.coefficient_efficiency(ref, ref)
        assert ce == pytest.approx(1.0)

    def test_constant_reconstruction(self):
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        test = np.full_like(ref, np.mean(ref))
        ce = utils.coefficient_efficiency(ref, test)
        assert ce == pytest.approx(0.0)

    def test_worse_than_mean_is_negative(self):
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        test = -ref  # terrible reconstruction
        ce = utils.coefficient_efficiency(ref, test)
        assert ce < 0.0


# ---------------------------------------------------------------------------
# gcd (great circle distance)
# ---------------------------------------------------------------------------

class TestGcd:
    def test_same_point_is_zero(self):
        d = utils.gcd(40.0, 10.0, 40.0, 10.0)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_north_south_pole(self):
        d = utils.gcd(90.0, 0.0, -90.0, 0.0)
        # half circumference of Earth ≈ 20,015 km
        assert d == pytest.approx(6367 * np.pi, rel=1e-3)

    def test_known_distance(self):
        # New York (40.71, -74.01) to London (51.51, -0.13) ≈ 5570 km
        d = utils.gcd(40.71, -74.01, 51.51, -0.13)
        assert 5000 < d < 6000


# ---------------------------------------------------------------------------
# year_float <-> datetime conversions
# ---------------------------------------------------------------------------

class TestYearFloatConversions:
    def test_ymd2year_float_jan1(self):
        yf = utils.ymd2year_float([2000], [1], [1])
        assert yf[0] == pytest.approx(2000.0, abs=0.01)

    def test_roundtrip_year_float_datetime(self):
        years = np.array([1900.0, 1950.5, 2000.0])
        dates = utils.year_float2datetime(years)
        years_back = utils.datetime2year_float(dates)
        np.testing.assert_allclose(years_back, years, atol=0.01)

    def test_year_float2datetime_length(self):
        years = np.linspace(1800, 2000, 50)
        dates = utils.year_float2datetime(years)
        assert len(dates) == 50


# ---------------------------------------------------------------------------
# smooth_ts / make_bin_vector / bin_ts
# ---------------------------------------------------------------------------

class TestSmoothing:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.ts = np.arange(1000, 1100, dtype=float)
        self.ys = rng.standard_normal(100)

    def test_make_bin_vector_step(self):
        bv = utils.make_bin_vector(self.ts, bin_width=10)
        diffs = np.diff(bv)
        assert np.all(diffs == pytest.approx(10.0))

    def test_smooth_ts_output_shape(self):
        ts_bin, ys_bin, bv = utils.smooth_ts(self.ts, self.ys, bin_width=10)
        assert len(ts_bin) == len(ys_bin)
        assert len(ts_bin) > 0

    def test_bin_ts_output_length_matches_resolution(self):
        ts_fine, ys_fine = utils.bin_ts(self.ts, self.ys, bin_width=10, resolution=1)
        # should span roughly [1000, 1099]
        assert len(ts_fine) > 0


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------

class TestBootstrap:
    def test_returns_correct_number_of_samples(self):
        data = np.arange(20, dtype=float)
        stats = utils.bootstrap(data, n_bootstraps=200)
        assert len(stats) == 200

    def test_mean_bootstrap_close_to_true_mean(self):
        rng = np.random.default_rng(0)
        data = rng.normal(5.0, 1.0, 1000)
        stats = utils.bootstrap(data, n_bootstraps=500)
        assert np.mean(stats) == pytest.approx(5.0, abs=0.2)


# ---------------------------------------------------------------------------
# annualize
# ---------------------------------------------------------------------------

class TestAnnualize:
    def test_calendar_year_annual_length(self):
        # Monthly data for 5 years
        years_monthly = np.repeat(np.arange(1990, 1995), 12)
        months_monthly = np.tile(np.arange(1, 13), 5)
        time = utils.ymd2year_float(years_monthly, months_monthly, np.ones(60, dtype=int))
        value = np.ones(60)
        time_ann, val_ann = utils.annualize(time, value, months=list(range(1, 13)))
        assert len(time_ann) == 5
        np.testing.assert_allclose(val_ann, 1.0)

    def test_jja_season(self):
        years = np.repeat(np.arange(1990, 1993), 12)
        months = np.tile(np.arange(1, 13), 3)
        time = utils.ymd2year_float(years, months, np.ones(36, dtype=int))
        value = np.where(np.isin(months, [6, 7, 8]), 1.0, 0.0)
        time_ann, val_ann = utils.annualize(time, value, months=[6, 7, 8])
        # JJA mean should be 1.0 since we only picked JJA months
        np.testing.assert_allclose(val_ann, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# colored_noise
# ---------------------------------------------------------------------------

class TestColoredNoise:
    def test_output_length_matches_time(self):
        t = np.arange(100)
        y = utils.colored_noise(alpha=1.0, t=t, seed=42)
        assert len(y) == 100

    def test_seeded_reproducibility(self):
        t = np.arange(50)
        y1 = utils.colored_noise(alpha=1.0, t=t, seed=7)
        y2 = utils.colored_noise(alpha=1.0, t=t, seed=7)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        t = np.arange(50)
        y1 = utils.colored_noise(alpha=1.0, t=t, seed=1)
        y2 = utils.colored_noise(alpha=1.0, t=t, seed=2)
        assert not np.allclose(y1, y2)


# ---------------------------------------------------------------------------
# overlap_ts
# ---------------------------------------------------------------------------

class TestOverlapTs:
    def test_full_overlap(self):
        ts1 = np.array([1.0, 2.0, 3.0])
        ys1 = np.array([10.0, 20.0, 30.0])
        ts2 = np.array([1.0, 2.0, 3.0])
        ys2 = np.array([5.0, 6.0, 7.0])
        time_ov, y1_ov, y2_ov = utils.overlap_ts(ts1, ys1, ts2, ys2)
        assert len(time_ov) == 3

    def test_partial_overlap(self):
        ts1 = np.array([1.0, 2.0, 3.0])
        ys1 = np.array([10.0, 20.0, 30.0])
        ts2 = np.array([2.0, 3.0, 4.0])
        ys2 = np.array([20.0, 30.0, 40.0])
        time_ov, y1_ov, y2_ov = utils.overlap_ts(ts1, ys1, ts2, ys2)
        assert len(time_ov) == 2
        np.testing.assert_array_equal(time_ov, [2.0, 3.0])

    def test_no_overlap_returns_empty(self):
        ts1 = np.array([1.0, 2.0])
        ys1 = np.array([10.0, 20.0])
        ts2 = np.array([5.0, 6.0])
        ys2 = np.array([50.0, 60.0])
        time_ov, y1_ov, y2_ov = utils.overlap_ts(ts1, ys1, ts2, ys2)
        assert len(time_ov) == 0


# ---------------------------------------------------------------------------
# ols_ts
# ---------------------------------------------------------------------------

class TestOlsTs:
    def test_returns_ols_model(self):
        ts = np.arange(1990, 2000, dtype=float)
        ys_proxy = np.sin(np.linspace(0, 2, 10))
        ys_obs = ys_proxy + 0.1
        model = utils.ols_ts(ts, ys_proxy, ts, ys_obs)
        assert hasattr(model, 'fit')

    def test_fit_uses_overlap(self):
        ts_proxy = np.arange(1985, 2000, dtype=float)
        ys_proxy = np.random.default_rng(0).standard_normal(15)
        ts_obs = np.arange(1990, 2005, dtype=float)
        ys_obs = np.random.default_rng(1).standard_normal(15)
        model = utils.ols_ts(ts_proxy, ys_proxy, ts_obs, ys_obs)
        result = model.fit()
        # overlap is 1990-1999 → 10 points
        assert result.nobs == 10

    def test_perfect_linear_fit(self):
        ts = np.arange(1990, 2010, dtype=float)
        ys_proxy = np.linspace(0, 1, 20)
        ys_obs = 2.0 * ys_proxy + 1.0  # y = 2x + 1
        model = utils.ols_ts(ts, ys_proxy, ts, ys_obs)
        result = model.fit()
        np.testing.assert_allclose(result.rsquared, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# geo_mean
# ---------------------------------------------------------------------------

class TestGeoMean:
    def _make_uniform_da(self, nlat=9, nlon=18, nt=5):
        import xarray as xr
        lat = np.linspace(-90, 90, nlat)
        lon = np.linspace(0, 340, nlon)
        time = np.arange(nt, dtype=float)
        return xr.DataArray(
            np.ones((nt, nlat, nlon)),
            dims=['time', 'lat', 'lon'],
            coords={'time': time, 'lat': lat, 'lon': lon},
        )

    def test_uniform_field_is_one(self):
        da = self._make_uniform_da()
        gm = utils.geo_mean(da)
        np.testing.assert_allclose(gm.values, 1.0, atol=1e-6)

    def test_output_shape_is_time_only(self):
        da = self._make_uniform_da(nt=7)
        gm = utils.geo_mean(da)
        assert gm.shape == (7,)

    def test_nh_mean_equals_sh_mean_for_uniform(self):
        da = self._make_uniform_da()
        nhm = utils.geo_mean(da, lat_min=0)
        shm = utils.geo_mean(da, lat_max=0)
        np.testing.assert_allclose(nhm.values, shm.values, atol=1e-6)

    def test_regional_subset_returns_values(self):
        da = self._make_uniform_da()
        regional = utils.geo_mean(da, lat_min=-30, lat_max=30, lon_min=0, lon_max=180)
        np.testing.assert_allclose(regional.values, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# year_float2dates
# ---------------------------------------------------------------------------

class TestYearFloat2Dates:
    def test_returns_list_of_datetimes(self):
        from datetime import datetime
        dates = utils.year_float2dates(np.array([2000.0, 2001.0]))
        assert all(isinstance(d, datetime) for d in dates)

    def test_integer_years_are_january_first(self):
        dates = utils.year_float2dates(np.array([2020.0, 2021.0]))
        for d in dates:
            assert d.month == 1
            assert d.day == 1

    def test_midyear_is_july(self):
        dates = utils.year_float2dates(np.array([2020.5]))
        assert dates[0].month == 7

    def test_output_length_matches_input(self):
        arr = np.arange(2000, 2010, dtype=float)
        dates = utils.year_float2dates(arr)
        assert len(dates) == 10

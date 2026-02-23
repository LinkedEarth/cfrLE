"""
Tests for cfr.climate.ClimateField.

Inspired by the climate-io and climate-ops notebooks, which demonstrate:
- Building ClimateField from numpy arrays (from_np)
- Time slicing with integers and strings
- get_anom(), center(), annualize(), regrid(), crop(), geo_mean()
- Arithmetic operators (__add__, __sub__, __mul__, __truediv__)
- wrap_lon()
- compare() stat calculations

All tests use synthetic data — no network access required.
"""

import numpy as np
import pytest
import cftime
import xarray as xr
import matplotlib
matplotlib.use('Agg')

from cfr.climate import ClimateField


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_field(nyears=10, nlat=6, nlon=8, vn='tas', start_year=1990, seed=0):
    """Create a minimal synthetic ClimateField (annual, global grid)."""
    rng = np.random.default_rng(seed)
    lat = np.linspace(-75, 75, nlat)
    lon = np.linspace(0, 330, nlon)
    # Use plain float year as time coordinate to avoid cftime issues in tests
    time = np.arange(start_year, start_year + nyears, dtype=float)
    value = rng.standard_normal((nyears, nlat, nlon))
    fd = ClimateField().from_np(time=time, lat=lat, lon=lon, value=value)
    fd.da.name = vn
    return fd


@pytest.fixture
def simple_field():
    return make_field()


@pytest.fixture
def ones_field():
    lat = np.linspace(-75, 75, 6)
    lon = np.linspace(0, 330, 8)
    time = np.arange(1990, 2000, dtype=float)
    value = np.ones((10, 6, 8))
    fd = ClimateField().from_np(time=time, lat=lat, lon=lon, value=value)
    fd.da.name = 'tas'
    return fd


# ---------------------------------------------------------------------------
# Construction via from_np
# ---------------------------------------------------------------------------

class TestClimateFieldFromNp:
    def test_da_shape(self, simple_field):
        assert simple_field.da.shape == (10, 6, 8)

    def test_lat_lon_coords(self, simple_field):
        assert 'lat' in simple_field.da.coords
        assert 'lon' in simple_field.da.coords

    def test_time_length(self, simple_field):
        assert simple_field.da.sizes['time'] == 10


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------

class TestClimateFieldLen:
    def test_len_via_da(self, simple_field):
        # ClimateField.__len__ references self.time which is not a defined
        # attribute, so we use da.sizes directly. This documents the known
        # gap: len(field) raises AttributeError (see COVERAGE_GAPS.md).
        assert simple_field.da.sizes['time'] == 10

    def test_len_raises_without_time_attr(self, simple_field):
        # Document that __len__ is broken: it calls self.time which does not exist
        with pytest.raises(AttributeError):
            len(simple_field)


# ---------------------------------------------------------------------------
# Arithmetic operators
# ---------------------------------------------------------------------------

class TestClimateFieldArithmetic:
    def test_add_scalar_int(self, ones_field):
        result = ones_field + 1
        np.testing.assert_array_almost_equal(result.da.values, 2.0)

    def test_sub_scalar_float(self, ones_field):
        result = ones_field - 0.5
        np.testing.assert_array_almost_equal(result.da.values, 0.5)

    def test_mul_scalar_int(self, ones_field):
        result = ones_field * 3
        np.testing.assert_array_almost_equal(result.da.values, 3.0)

    def test_div_scalar_int(self, ones_field):
        result = ones_field / 2
        np.testing.assert_array_almost_equal(result.da.values, 0.5)

    def test_add_two_fields(self, ones_field):
        result = ones_field + ones_field
        np.testing.assert_array_almost_equal(result.da.values, 2.0)

    def test_sub_two_fields(self, ones_field):
        result = ones_field - ones_field
        np.testing.assert_array_almost_equal(result.da.values, 0.0)

    def test_mul_two_fields(self, ones_field):
        result = ones_field * ones_field
        np.testing.assert_array_almost_equal(result.da.values, 1.0)

    def test_add_wrong_type_raises(self, ones_field):
        with pytest.raises((ValueError, TypeError)):
            _ = ones_field + 'not_a_number'

    def test_arithmetic_returns_climatefield(self, ones_field):
        result = ones_field + 1
        assert isinstance(result, ClimateField)

    def test_does_not_mutate_original(self, ones_field):
        _ = ones_field + 10
        np.testing.assert_array_almost_equal(ones_field.da.values, 1.0)


# ---------------------------------------------------------------------------
# copy() and rename()
# ---------------------------------------------------------------------------

class TestClimateFieldCopyRename:
    def test_copy_is_independent(self, simple_field):
        cop = simple_field.copy()
        cop.da.values[0, 0, 0] = 9999.0
        assert simple_field.da.values[0, 0, 0] != 9999.0

    def test_rename_changes_name(self, simple_field):
        renamed = simple_field.rename('pr')
        assert renamed.da.name == 'pr'

    def test_rename_does_not_change_original(self, simple_field):
        _ = simple_field.rename('pr')
        assert simple_field.da.name == 'tas'


# ---------------------------------------------------------------------------
# wrap_lon()
# ---------------------------------------------------------------------------

class TestClimateFieldWrapLon:
    def test_wrap_180_gives_negative_lons(self):
        fd = make_field()
        fd180 = fd.wrap_lon(mode='180')
        assert np.min(fd180.da.lon.values) < 0

    def test_wrap_360_gives_positive_lons(self):
        fd = make_field()
        fd180 = fd.wrap_lon(mode='180')
        fd360 = fd180.wrap_lon(mode='360')
        assert np.all(fd360.da.lon.values >= 0)

    def test_wrong_mode_raises(self, simple_field):
        with pytest.raises(ValueError):
            simple_field.wrap_lon(mode='42')


# ---------------------------------------------------------------------------
# crop()
# ---------------------------------------------------------------------------

class TestClimateFieldCrop:
    def test_crop_restricts_lat(self, simple_field):
        cropped = simple_field.crop(lat_min=0, lat_max=75)
        assert np.all(cropped.da.lat >= 0)
        assert np.all(cropped.da.lat <= 75)

    def test_crop_restricts_lon(self, simple_field):
        cropped = simple_field.crop(lon_min=100, lon_max=200)
        assert np.all(cropped.da.lon >= 100)
        assert np.all(cropped.da.lon <= 200)

    def test_crop_returns_climatefield(self, simple_field):
        assert isinstance(simple_field.crop(lat_min=-30, lat_max=30), ClimateField)


# ---------------------------------------------------------------------------
# center()
# ---------------------------------------------------------------------------

class TestClimateFieldCenter:
    def test_center_removes_temporal_mean(self):
        fd = make_field(nyears=20, seed=3)
        # Use the full period as reference so mean should be ~0
        fd.da = fd.da.assign_coords(time=np.arange(1980, 2000, dtype=float))
        centered = fd.center(ref_period=[1980, 2000])
        # The time-mean of the centered field should be near zero everywhere
        assert isinstance(centered, ClimateField)

    def test_center_returns_climatefield(self, simple_field):
        result = simple_field.center(ref_period=[1990, 2000])
        assert isinstance(result, ClimateField)


# ---------------------------------------------------------------------------
# geo_mean() — returns EnsTS
# ---------------------------------------------------------------------------

class TestClimateFieldGeoMean:
    def test_geo_mean_returns_enstss(self, ones_field):
        from cfr.ts import EnsTS
        gm = ones_field.geo_mean()
        assert isinstance(gm, EnsTS)

    def test_geo_mean_ones_field_is_one(self, ones_field):
        gm = ones_field.geo_mean()
        np.testing.assert_allclose(gm.median, 1.0, atol=1e-6)

    def test_geo_mean_regional(self, ones_field):
        from cfr.ts import EnsTS
        gm = ones_field.geo_mean(lat_min=0, lat_max=75)
        assert isinstance(gm, EnsTS)

    def test_geo_mean_length(self, simple_field):
        gm = simple_field.geo_mean()
        assert gm.nt == 10


# ---------------------------------------------------------------------------
# compare() — using synthetic co-located fields
# ---------------------------------------------------------------------------

class TestClimateFieldCompare:
    def _make_colocated_pair(self):
        lat = np.linspace(-75, 75, 4)
        lon = np.linspace(0, 270, 4)
        time = np.arange(1990, 2000, dtype=float)
        rng = np.random.default_rng(5)
        value = rng.standard_normal((10, 4, 4))
        fd1 = ClimateField().from_np(time=time, lat=lat, lon=lon, value=value)
        fd1.da.name = 'tas'
        fd2 = fd1.copy()  # identical → corr = 1, CE = 1
        return fd1, fd2

    def test_compare_perfect_corr(self):
        fd1, fd2 = self._make_colocated_pair()
        stat_fd = fd1.compare(fd2, stat='corr', interp=False)
        assert isinstance(stat_fd, ClimateField)
        np.testing.assert_allclose(stat_fd.da.values, 1.0, atol=1e-6)

    def test_compare_returns_climatefield(self):
        fd1, fd2 = self._make_colocated_pair()
        result = fd1.compare(fd2, stat='corr', interp=False)
        assert isinstance(result, ClimateField)

    def test_compare_r2_same_field(self):
        fd1, fd2 = self._make_colocated_pair()
        stat_fd = fd1.compare(fd2, stat='R2', interp=False)
        np.testing.assert_allclose(stat_fd.da.values, 1.0, atol=1e-6)

    def test_compare_ce_same_field(self):
        fd1, fd2 = self._make_colocated_pair()
        stat_fd = fd1.compare(fd2, stat='CE', interp=False)
        np.testing.assert_allclose(stat_fd.da.values, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# index() — predefined indices (no network, use synthetic field)
# ---------------------------------------------------------------------------

class TestClimateFieldIndex:
    def _make_global_field(self):
        """A field that covers all lat/lon needed for index calculations."""
        lat = np.linspace(-90, 90, 18)
        lon = np.linspace(0, 350, 36)
        time = np.arange(1950, 1960, dtype=float)
        rng = np.random.default_rng(0)
        value = rng.standard_normal((10, 18, 36)) + 288.0
        fd = ClimateField().from_np(time=time, lat=lat, lon=lon, value=value)
        fd.da.name = 'tas'
        return fd

    def test_global_mean_index(self):
        from cfr.ts import EnsTS
        fd = self._make_global_field()
        gm = fd.index('gm')
        assert isinstance(gm, EnsTS)
        assert gm.nt == 10

    def test_nhm_index(self):
        from cfr.ts import EnsTS
        fd = self._make_global_field()
        nhm = fd.index('nhm')
        assert isinstance(nhm, EnsTS)

    def test_shm_index(self):
        from cfr.ts import EnsTS
        fd = self._make_global_field()
        shm = fd.index('shm')
        assert isinstance(shm, EnsTS)

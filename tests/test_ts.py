"""
Tests for cfr.ts.EnsTS (ensemble timeseries class).

Inspired by the climate-ops and lmr-* notebooks, which use EnsTS objects
(returned by ClimateField.geo_mean()) for plotting, arithmetic, and validation.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for tests

from cfr.ts import EnsTS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_ens():
    """10 time steps, 5 ensemble members, values = 1.0 everywhere."""
    time = np.arange(1900, 1910, dtype=float)
    value = np.ones((10, 5))
    return EnsTS(time=time, value=value)


@pytest.fixture
def rng_ens():
    """100-step, 20-member ensemble with reproducible random values."""
    rng = np.random.default_rng(42)
    time = np.arange(1900, 2000, dtype=float)
    value = rng.standard_normal((100, 20))
    return EnsTS(time=time, value=value)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestEnstsConstruction:
    def test_1d_value_becomes_2d(self):
        time = np.arange(10, dtype=float)
        value = np.ones(10)
        ens = EnsTS(time=time, value=value)
        assert ens.value.ndim == 2
        assert ens.value.shape == (10, 1)

    def test_attributes_populated(self, simple_ens):
        assert simple_ens.nt == 10
        assert simple_ens.nEns == 5

    def test_median_shape(self, simple_ens):
        assert simple_ens.median.shape == (10,)

    def test_mean_equals_median_for_uniform(self, simple_ens):
        np.testing.assert_array_equal(simple_ens.mean, simple_ens.median)

    def test_std_zero_for_uniform(self, simple_ens):
        np.testing.assert_array_almost_equal(simple_ens.std, 0.0)


# ---------------------------------------------------------------------------
# Aggregate accessors
# ---------------------------------------------------------------------------

class TestEnstsAggregates:
    def test_get_mean_shape(self, rng_ens):
        m = rng_ens.get_mean()
        assert m.value.shape == (100, 1)

    def test_get_median_shape(self, rng_ens):
        md = rng_ens.get_median()
        assert md.value.shape == (100, 1)

    def test_get_std_shape(self, rng_ens):
        s = rng_ens.get_std()
        assert s.value.shape == (100, 1)

    def test_get_mean_value(self, simple_ens):
        m = simple_ens.get_mean()
        np.testing.assert_array_almost_equal(m.value[:, 0], 1.0)

    def test_get_std_value(self, simple_ens):
        s = simple_ens.get_std()
        np.testing.assert_array_almost_equal(s.value[:, 0], 0.0)


# ---------------------------------------------------------------------------
# Arithmetic operators
# ---------------------------------------------------------------------------

class TestEnstsArithmetic:
    def test_add_scalar(self, simple_ens):
        result = simple_ens + 2.0
        np.testing.assert_array_almost_equal(result.value, 3.0)

    def test_sub_scalar(self, simple_ens):
        result = simple_ens - 1.0
        np.testing.assert_array_almost_equal(result.value, 0.0)

    def test_mul_scalar(self, simple_ens):
        result = simple_ens * 3.0
        np.testing.assert_array_almost_equal(result.value, 3.0)

    def test_div_scalar(self, simple_ens):
        result = simple_ens / 2.0
        np.testing.assert_array_almost_equal(result.value, 0.5)

    def test_add_ensts(self, simple_ens):
        result = simple_ens + simple_ens
        # adds the median of simple_ens (all 1s) to each member
        np.testing.assert_array_almost_equal(result.value, 2.0)

    def test_sub_ensts(self, simple_ens):
        result = simple_ens - simple_ens
        np.testing.assert_array_almost_equal(result.value, 0.0)

    def test_arithmetic_does_not_mutate_original(self, simple_ens):
        _ = simple_ens + 10.0
        np.testing.assert_array_almost_equal(simple_ens.value, 1.0)


# ---------------------------------------------------------------------------
# Subscript operator
# ---------------------------------------------------------------------------

class TestEnstsSubscript:
    def test_integer_slice(self, rng_ens):
        sub = rng_ens[0:10]
        assert sub.nt == 10

    def test_time_preserved(self, rng_ens):
        sub = rng_ens[5:15]
        np.testing.assert_array_equal(sub.time, rng_ens.time[5:15])

    def test_tuple_key(self, rng_ens):
        # select first 10 time steps, first 5 members
        sub = rng_ens[0:10, 0:5]
        assert sub.nt == 10
        assert sub.nEns == 5


# ---------------------------------------------------------------------------
# DataFrame round-trip
# ---------------------------------------------------------------------------

class TestEnstsDf:
    def test_to_df_columns(self, simple_ens):
        df = simple_ens.to_df()
        assert 'time' in df.columns
        # 5 members → 5 value columns
        assert len(df.columns) == 1 + 5

    def test_roundtrip(self, rng_ens):
        df = rng_ens.to_df()
        # Provide explicit column order so from_df produces the same column order
        value_columns = [c for c in df.columns if c != 'time']
        restored = EnsTS().from_df(df, value_columns=value_columns)
        assert restored.nt == rng_ens.nt
        assert restored.nEns == rng_ens.nEns
        np.testing.assert_array_almost_equal(
            restored.value, rng_ens.value, decimal=10
        )


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------

class TestEnstsCompare:
    def test_compare_perfect_reconstruction(self):
        time = np.arange(1900, 1950, dtype=float)
        value = np.sin(np.linspace(0, 2 * np.pi, 50))[:, np.newaxis]
        ens = EnsTS(time=time, value=value)
        compared = ens.compare(
            ref_time=time, ref_value=value[:, 0],
            ref_name='truth', stats=['corr', 'CE']
        )
        assert compared.valid_stats['corr'] == pytest.approx(1.0, abs=1e-6)
        assert compared.valid_stats['CE'] == pytest.approx(1.0, abs=1e-6)

    def test_compare_stores_ref_name(self):
        time = np.arange(1900, 1950, dtype=float)
        value = np.ones((50, 1))
        ens = EnsTS(time=time, value=value)
        compared = ens.compare(
            ref_time=time, ref_value=np.ones(50),
            ref_name='myref', stats=['corr']
        )
        assert compared.ref_name == 'myref'

    def test_compare_with_timespan(self):
        time = np.arange(1900, 2000, dtype=float)
        rng = np.random.default_rng(1)
        value = rng.standard_normal((100, 5))
        ens = EnsTS(time=time, value=value)
        compared = ens.compare(
            ref_time=time, ref_value=ens.median,
            ref_name='ref', stats=['CE'],
            timespan=[1920, 1960]
        )
        assert 'CE' in compared.valid_stats

    def test_compare_bad_stat_raises(self):
        time = np.arange(10, dtype=float)
        value = np.ones((10, 1))
        ens = EnsTS(time=time, value=value)
        with pytest.raises(ValueError):
            ens.compare(ref_time=time, ref_value=np.ones(10), stats=['bad_stat'])

    def test_mismatched_lengths_raise(self):
        # Annual ens (10 years), biennial ref (5 points) → different slice lengths
        ens_time = np.arange(1900, 1910, dtype=float)
        value = np.ones((10, 1))
        ens = EnsTS(time=ens_time, value=value)
        # ref has only every-other-year; within 1900-1909 gives 5 points vs 10
        ref_time = np.arange(1900, 1910, 2, dtype=float)
        ref_value = np.ones(5)
        with pytest.raises(ValueError):
            ens.compare(
                ref_time=ref_time, ref_value=ref_value,
                stats=['corr'],
            )


# ---------------------------------------------------------------------------
# annualize()
# ---------------------------------------------------------------------------

class TestEnstsAnnualize:
    def test_annual_from_monthly(self):
        from cfr import utils
        years = np.repeat(np.arange(1990, 1995), 12)
        months = np.tile(np.arange(1, 13), 5)
        time = utils.ymd2year_float(years, months, np.ones(60, dtype=int))
        value = np.ones((60, 3))  # 3 ensemble members, all ones
        ens = EnsTS(time=time, value=value)
        ann = ens.annualize()
        assert ann.nt == 5

    def test_annualize_returns_new_object(self):
        from cfr import utils
        years = np.repeat(np.arange(1990, 1993), 12)
        months = np.tile(np.arange(1, 13), 3)
        time = utils.ymd2year_float(years, months, np.ones(36, dtype=int))
        value = np.ones((36, 2))
        ens = EnsTS(time=time, value=value)
        ann = ens.annualize()
        assert ann is not ens


# ---------------------------------------------------------------------------
# copy()
# ---------------------------------------------------------------------------

class TestEnstsCopy:
    def test_copy_is_independent(self, rng_ens):
        copy = rng_ens.copy()
        copy.value[0, 0] = 9999.0
        assert rng_ens.value[0, 0] != 9999.0


# ---------------------------------------------------------------------------
# plot methods (smoke tests — just confirm no exceptions raised)
# ---------------------------------------------------------------------------

class TestEnstsPlots:
    def test_plot_qs_runs(self, rng_ens):
        import matplotlib.pyplot as plt
        fig, ax = rng_ens.plot_qs()
        plt.close(fig)

    def test_plot_runs(self, rng_ens):
        import matplotlib.pyplot as plt
        fig, ax = rng_ens.plot()
        plt.close(fig)

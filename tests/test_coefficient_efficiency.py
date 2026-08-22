''' Tests for cfr.utils.coefficient_efficiency, focused on NaN robustness.
'''
import numpy as np
import pytest

from cfr.utils import coefficient_efficiency


def old_coefficient_efficiency(ref, test, valid=None):
    ''' The pre-patch implementation, recreated here for a numerical-identity check
    on complete (NaN-free) inputs.
    '''
    dims_ref = ref.shape

    if len(dims_ref) == 3:
        dims = dims_ref[1:3]
    elif len(dims_ref) == 2:
        dims = dims_ref[1:2]
    elif len(dims_ref) == 1:
        dims = 1

    error = test - ref
    numer = np.sum(np.power(error, 2), axis=0)
    denom = np.sum(np.power(ref - np.nanmean(ref, axis=0), 2), axis=0)
    CE = 1. - np.divide(numer, denom)

    if valid:
        nbok = np.sum(np.isfinite(ref), axis=0)
        nball = float(dims_ref[0])
        ratio = np.divide(nbok, nball)
        indbad = np.where(ratio < valid)
        dim_indbad = len(indbad)
        testlist = [indbad[k].size for k in range(dim_indbad)]
        if not all(v == 0 for v in testlist):
            if isinstance(dims, (tuple, list)):
                CE[indbad] = np.nan
            else:
                CE = np.nan

    return CE


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_complete_data_1d_matches_old(rng):
    ref = rng.normal(size=50)
    test = ref + rng.normal(scale=0.5, size=50)
    assert coefficient_efficiency(ref, test) == pytest.approx(old_coefficient_efficiency(ref, test))


def test_complete_data_2d_matches_old(rng):
    ref = rng.normal(size=(50, 10))
    test = ref + rng.normal(scale=0.5, size=(50, 10))
    np.testing.assert_allclose(coefficient_efficiency(ref, test), old_coefficient_efficiency(ref, test))


def test_complete_data_3d_matches_old(rng):
    ref = rng.normal(size=(50, 4, 5))
    test = ref + rng.normal(scale=0.5, size=(50, 4, 5))
    np.testing.assert_allclose(coefficient_efficiency(ref, test), old_coefficient_efficiency(ref, test))


def test_one_nan_year_at_one_cell_is_finite_and_others_unchanged(rng):
    ref = rng.normal(size=(50, 3, 3))
    test = ref + rng.normal(scale=0.5, size=(50, 3, 3))

    ref_with_nan = ref.copy()
    ref_with_nan[10, 1, 1] = np.nan

    ce_full = coefficient_efficiency(ref, test)
    ce_nan = coefficient_efficiency(ref_with_nan, test)

    # the affected cell should still be finite, computed from the remaining years
    assert np.isfinite(ce_nan[1, 1])

    mask = np.delete(np.arange(50), 10)
    expected_cell = old_coefficient_efficiency(ref[mask, 1, 1], test[mask, 1, 1])
    assert ce_nan[1, 1] == pytest.approx(expected_cell)

    # all other cells should be unaffected
    other = np.ones((3, 3), dtype=bool)
    other[1, 1] = False
    np.testing.assert_allclose(ce_nan[other], ce_full[other])


def test_nan_in_test_is_also_tolerated_pairwise(rng):
    ref = rng.normal(size=50)
    test = ref + rng.normal(scale=0.5, size=50)
    test_with_nan = test.copy()
    test_with_nan[5] = np.nan

    ce = coefficient_efficiency(ref, test_with_nan)
    assert np.isfinite(ce)

    mask = np.delete(np.arange(50), 5)
    expected = old_coefficient_efficiency(ref[mask], test[mask])
    assert ce == pytest.approx(expected)


def test_valid_fraction_below_threshold_returns_nan(rng):
    ref = rng.normal(size=(20, 2))
    test = ref + rng.normal(scale=0.5, size=(20, 2))

    ref_masked = ref.copy()
    # cell 0: 15/20 valid years (0.75, passes valid=0.5)
    ref_masked[:5, 0] = np.nan
    # cell 1: 8/20 valid years (0.4, fails valid=0.5)
    ref_masked[:12, 1] = np.nan

    ce = coefficient_efficiency(ref_masked, test, valid=0.5)
    assert np.isfinite(ce[0])
    assert np.isnan(ce[1])


def test_zero_variance_reference_gives_nan_without_warnings(recwarn):
    ref = np.ones(30)  # zero variance -> denominator is 0
    test = np.random.default_rng(0).normal(size=30)

    ce = coefficient_efficiency(ref, test)

    assert np.isnan(ce)
    assert len(recwarn) == 0


def test_no_valid_pairs_gives_nan_without_warnings(recwarn):
    ref = np.full((30, 2), np.nan)
    ref[:, 0] = np.arange(30, dtype=float)  # cell 0 has data, cell 1 is all-NaN
    test = np.arange(30, dtype=float)[:, None] * np.ones((1, 2))

    ce = coefficient_efficiency(ref, test)

    assert np.isfinite(ce[0])
    assert np.isnan(ce[1])
    assert len(recwarn) == 0


def test_1d_input_returns_scalar(rng):
    ref = rng.normal(size=10)
    test = ref + rng.normal(scale=0.5, size=10)
    ce = coefficient_efficiency(ref, test)
    assert np.isscalar(ce) or (isinstance(ce, np.ndarray) and ce.ndim == 0)


def test_2d_input_works(rng):
    ref = rng.normal(size=(10, 4))
    test = ref + rng.normal(scale=0.5, size=(10, 4))
    ce = coefficient_efficiency(ref, test)
    assert ce.shape == (4,)
    assert np.all(np.isfinite(ce))

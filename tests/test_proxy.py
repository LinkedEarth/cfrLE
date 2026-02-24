"""
Tests for cfr.proxy.ProxyRecord and cfr.proxy.ProxyDatabase.

Inspired by the proxy-io, proxy-ops, and proxy-analysis notebooks.
All tests use purely synthetic data — no network access required.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')

from cfr.proxy import ProxyRecord, ProxyDatabase, get_ptype


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_record(pid='test_001', lat=30.0, lon=90.0, elev=100.0,
                ptype='tree.TRW', n=50, seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(1950, 1950 + n, dtype=float)
    value = rng.standard_normal(n)
    return ProxyRecord(
        pid=pid, lat=lat, lon=lon, elev=elev,
        time=time, value=value, ptype=ptype,
        value_name='TRW', value_unit='mm',
    )


def make_database(n_records=3):
    records = {
        f'rec_{i:03d}': make_record(
            pid=f'rec_{i:03d}',
            lat=float(i * 10 - 10),
            lon=float(i * 20),
            ptype='tree.TRW' if i % 2 == 0 else 'coral.d18O',
            seed=i,
        )
        for i in range(n_records)
    }
    return ProxyDatabase(records=records)


# ---------------------------------------------------------------------------
# get_ptype
# ---------------------------------------------------------------------------

class TestGetPtype:
    def test_known_pair(self):
        assert get_ptype('tree', 'TRW') == 'tree.TRW'

    def test_known_pair_coral(self):
        assert get_ptype('coral', 'd18O') == 'coral.d18O'

    def test_unknown_pair_fallback(self):
        # Unknown pairs become "archive.proxy" with spaces stripped
        result = get_ptype('mystery archive', 'weird proxy')
        assert result == 'mysteryarchive.weirdproxy'

    def test_case_insensitive(self):
        # Fuzzy matching normalises case
        assert get_ptype('Tree', 'TRW') == 'tree.TRW'

    def test_spaces_stripped(self):
        assert get_ptype('glacier ice', 'd18O') == 'ice.d18O'


# ---------------------------------------------------------------------------
# ProxyRecord construction
# ---------------------------------------------------------------------------

class TestProxyRecordConstruction:
    def test_attributes_set(self):
        rec = make_record()
        assert rec.pid == 'test_001'
        assert rec.lat == 30.0
        assert rec.ptype == 'tree.TRW'
        assert len(rec.time) == 50
        assert len(rec.value) == 50

    def test_lon_modulo_360(self):
        rec = make_record(lon=-90.0)
        assert rec.lon == pytest.approx(270.0)

    def test_dt_computed(self):
        rec = make_record()
        assert rec.dt == pytest.approx(1.0)

    def test_tags_default_empty(self):
        rec = make_record()
        assert rec.tags == set()


# ---------------------------------------------------------------------------
# ProxyRecord.copy()
# ---------------------------------------------------------------------------

class TestProxyRecordCopy:
    def test_copy_is_independent(self):
        rec = make_record()
        cop = rec.copy()
        cop.value[0] = 9999.0
        assert rec.value[0] != 9999.0


# ---------------------------------------------------------------------------
# ProxyRecord.slice()
# ---------------------------------------------------------------------------

class TestProxyRecordSlice:
    def test_slice_returns_subset(self):
        rec = make_record()
        sliced = rec.slice([1960, 1979])
        assert np.all(sliced.time >= 1960)
        assert np.all(sliced.time <= 1979)
        assert len(sliced.time) < len(rec.time)

    def test_slice_preserves_ptype(self):
        rec = make_record()
        sliced = rec.slice([1960, 1979])
        assert sliced.ptype == rec.ptype

    def test_slice_odd_elements_raises(self):
        rec = make_record()
        with pytest.raises(ValueError):
            rec.slice([1960, 1970, 1980])

    def test_slice_two_segments(self):
        rec = make_record()
        sliced = rec.slice([1952, 1960, 1975, 1985])
        assert np.all(
            ((sliced.time >= 1952) & (sliced.time <= 1960)) |
            ((sliced.time >= 1975) & (sliced.time <= 1985))
        )


# ---------------------------------------------------------------------------
# ProxyRecord.concat()
# ---------------------------------------------------------------------------

class TestProxyRecordConcat:
    def test_concat_merges_time(self):
        rec1 = make_record(n=20, seed=1)
        rec1_part = rec1.slice([1950, 1960])
        rec2 = make_record(n=20, seed=2)
        rec2_adjusted = ProxyRecord(
            pid='rec_adj', lat=30.0, lon=90.0,
            time=np.arange(1961, 1981, dtype=float),
            value=np.ones(20),
            ptype='tree.TRW',
        )
        merged = rec1_part.concat([rec2_adjusted])
        assert len(merged.time) == len(rec1_part.time) + 20
        assert merged.time[-1] == pytest.approx(1980.0)


# ---------------------------------------------------------------------------
# ProxyRecord.center() / standardize()
# ---------------------------------------------------------------------------

class TestProxyRecordCenter:
    def test_center_subtracts_mean(self):
        rng = np.random.default_rng(0)
        time = np.arange(1900, 1960, dtype=float)
        value = rng.standard_normal(60) + 5.0
        rec = ProxyRecord(pid='c', time=time, value=value, lat=0, lon=0, ptype='tree.TRW')
        centered = rec.center(ref_period=[1900, 1960], thresh=1)
        assert 'centered' in centered.tags
        # mean over the reference period should be near zero
        ref_mask = (centered.time >= 1900) & (centered.time <= 1960)
        assert np.mean(centered.value[ref_mask]) == pytest.approx(0.0, abs=1e-10)

    def test_center_insufficient_data_not_centered(self):
        rec = ProxyRecord(pid='c', time=np.array([1900.0]), value=np.array([5.0]),
                          lat=0, lon=0, ptype='tree.TRW')
        centered = rec.center(ref_period=[1900, 1960], thresh=5, force=False)
        # should not have been centered (thresh not met)
        assert 'centered' not in centered.tags


class TestProxyRecordStandardize:
    def test_standardize_unit_std(self):
        rng = np.random.default_rng(7)
        time = np.arange(1900, 1960, dtype=float)
        value = rng.standard_normal(60) * 3.0 + 2.0
        rec = ProxyRecord(pid='s', time=time, value=value, lat=0, lon=0, ptype='tree.TRW')
        std_rec = rec.standardize(ref_period=[1900, 1960], thresh=1)
        assert 'standardized' in std_rec.tags
        ref_mask = (std_rec.time >= 1900) & (std_rec.time <= 1960)
        assert np.std(std_rec.value[ref_mask]) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# ProxyRecord.__getitem__ (string slicing)
# ---------------------------------------------------------------------------

class TestProxyRecordSubscript:
    def test_string_year_slice(self):
        rec = make_record()
        sub = rec['1960':'1970']
        assert np.all(sub.time >= 1960)
        assert np.all(sub.time <= 1970)

    def test_string_year_slice_with_step(self):
        rec = make_record()
        sub = rec['1950':'1980':5]
        diffs = np.diff(sub.time)
        assert np.all(diffs >= 4.9)  # step ≥ 5 years


# ---------------------------------------------------------------------------
# ProxyRecord.to_da() / from_da()
# ---------------------------------------------------------------------------

class TestProxyRecordDataArray:
    def test_to_da_name_is_pid(self):
        rec = make_record()
        da = rec.to_da()
        assert da.name == rec.pid

    def test_roundtrip_via_da(self):
        rec = make_record()
        da = rec.to_da()
        restored = ProxyRecord().from_da(da)
        np.testing.assert_allclose(restored.value, rec.value, atol=1e-10)
        assert restored.lat == rec.lat
        assert restored.ptype == rec.ptype


# ---------------------------------------------------------------------------
# ProxyRecord.annualize()
# ---------------------------------------------------------------------------

class TestProxyRecordAnnualize:
    def test_annual_data_unchanged_count(self):
        rec = make_record(n=50)  # already annual
        ann = rec.annualize()
        # Already annual → roughly same number of points
        assert abs(len(ann.time) - len(rec.time)) <= 2

    def test_annualize_tags(self):
        rec = make_record(n=50)
        ann = rec.annualize()
        assert 'annualized' in ann.tags


# ---------------------------------------------------------------------------
# ProxyDatabase construction
# ---------------------------------------------------------------------------

class TestProxyDatabaseConstruction:
    def test_nrec(self):
        pdb = make_database(3)
        assert pdb.nrec == 3

    def test_pids_list(self):
        pdb = make_database(3)
        assert len(pdb.pids) == 3

    def test_type_dict(self):
        pdb = make_database(4)
        # 0, 2 → tree.TRW; 1, 3 → coral.d18O
        assert 'tree.TRW' in pdb.type_dict
        assert 'coral.d18O' in pdb.type_dict

    def test_empty_database(self):
        pdb = ProxyDatabase()
        # refresh() is only called when records is provided, so access .records directly
        assert len(pdb.records) == 0


# ---------------------------------------------------------------------------
# ProxyDatabase.__getitem__
# ---------------------------------------------------------------------------

class TestProxyDatabaseGetItem:
    def test_get_by_pid(self):
        pdb = make_database(3)
        pid = pdb.pids[0]
        rec = pdb[pid]
        assert isinstance(rec, ProxyRecord)
        assert rec.pid == pid

    def test_get_by_index(self):
        pdb = make_database(3)
        rec = pdb[0]
        assert isinstance(rec, ProxyRecord)

    def test_get_slice(self):
        pdb = make_database(5)
        sub = pdb[1:3]
        assert isinstance(sub, ProxyDatabase)
        assert sub.nrec == 2


# ---------------------------------------------------------------------------
# ProxyDatabase.__add__ / __sub__
# ---------------------------------------------------------------------------

class TestProxyDatabaseAddSub:
    def test_add_record(self):
        pdb = make_database(2)
        new_rec = make_record(pid='extra', seed=99)
        pdb2 = pdb + new_rec
        assert pdb2.nrec == 3

    def test_add_database(self):
        pdb1 = make_database(2)
        # pdb2 with distinct PIDs (update both dict key and record.pid)
        records2 = {}
        for pid, rec in make_database(2).records.items():
            new_rec = rec.copy()
            new_rec.pid = f'z_{pid}'
            records2[f'z_{pid}'] = new_rec
        pdb2b = ProxyDatabase(records=records2)
        combined = pdb1 + pdb2b
        assert combined.nrec == 4

    def test_sub_removes_records(self):
        pdb = make_database(3)
        sub_pdb = make_database(1)  # has rec_000
        result = pdb - sub_pdb
        assert result.nrec == 2
        assert 'rec_000' not in result.pids


# ---------------------------------------------------------------------------
# ProxyDatabase.filter()
# ---------------------------------------------------------------------------

class TestProxyDatabaseFilter:
    def setup_method(self):
        self.pdb = make_database(6)

    def test_filter_by_ptype(self):
        filtered = self.pdb.filter(by='ptype', keys=['tree.TRW'])
        for pid, rec in filtered.records.items():
            assert rec.ptype == 'tree.TRW'

    def test_filter_by_pid(self):
        target = self.pdb.pids[0]
        filtered = self.pdb.filter(by='pid', keys=[target])
        assert filtered.nrec == 1
        assert target in filtered.pids

    def test_filter_by_lat(self):
        filtered = self.pdb.filter(by='lat', keys=[0, 30])
        for pid, rec in filtered.records.items():
            assert 0 <= rec.lat <= 30

    def test_filter_by_tag(self):
        pdb = self.pdb.copy()
        # manually tag first record
        first_pid = pdb.pids[0]
        pdb.records[first_pid].tags.add('selected')
        filtered = pdb.filter(by='tag', keys=['selected'])
        assert filtered.nrec == 1


# ---------------------------------------------------------------------------
# ProxyDatabase.slice()
# ---------------------------------------------------------------------------

class TestProxyDatabaseSlice:
    def test_slice_restricts_time(self):
        pdb = make_database(3)
        sliced = pdb.slice([1960, 1980])
        for pid, rec in sliced.records.items():
            assert np.all(rec.time >= 1960)
            assert np.all(rec.time <= 1980)


# ---------------------------------------------------------------------------
# ProxyDatabase.center()
# ---------------------------------------------------------------------------

class TestProxyDatabaseCenter:
    def test_center_all_records(self):
        pdb = make_database(3)
        centered = pdb.center(ref_period=[1950, 1999], thresh=1)
        for pid, rec in centered.records.items():
            assert 'centered' in rec.tags


# ---------------------------------------------------------------------------
# ProxyDatabase.to_df()
# ---------------------------------------------------------------------------

class TestProxyDatabaseDf:
    def test_to_df_has_correct_rows(self):
        pdb = make_database(3)
        df = pdb.to_df()
        assert len(df) == 3

    def test_to_df_has_lat_lon(self):
        pdb = make_database(3)
        df = pdb.to_df()
        assert 'lat' in df.columns or 'geo_meanLat' in df.columns


# ---------------------------------------------------------------------------
# ProxyDatabase.annualize()
# ---------------------------------------------------------------------------

class TestProxyDatabaseAnnualize:
    def test_annualize_returns_database(self):
        pdb = make_database(3)
        ann = pdb.annualize()
        assert isinstance(ann, ProxyDatabase)
        assert ann.nrec <= pdb.nrec  # might drop records if can't annualize


# ---------------------------------------------------------------------------
# ProxyRecord.get_clim() / del_clim() / del_pseudo()
# ---------------------------------------------------------------------------

class TestProxyRecordGetClim:
    def _make_climate_field(self):
        from cfr.climate import ClimateField
        lat = np.linspace(-75, 75, 6)
        lon = np.linspace(0, 330, 8)
        time = np.arange(1950, 2000, dtype=float)
        value = np.random.default_rng(0).standard_normal((50, 6, 8))
        fd = ClimateField().from_np(time=time, lat=lat, lon=lon, value=value)
        fd.da.name = 'tas'
        return fd

    def test_clim_key_created(self):
        rec = make_record()
        fd = self._make_climate_field()
        rec.get_clim(fd, tag='obs')
        assert 'obs.tas' in rec.clim

    def test_clim_is_climatefield(self):
        from cfr.climate import ClimateField
        rec = make_record()
        fd = self._make_climate_field()
        rec.get_clim(fd, tag='obs')
        assert isinstance(rec.clim['obs.tas'], ClimateField)

    def test_clim_is_1d(self):
        rec = make_record()
        fd = self._make_climate_field()
        rec.get_clim(fd, tag='obs')
        assert rec.clim['obs.tas'].da.ndim == 1

    def test_del_clim_removes_attribute(self):
        rec = make_record()
        fd = self._make_climate_field()
        rec.get_clim(fd, tag='obs')
        assert hasattr(rec, 'clim')
        rec.del_clim()
        assert not hasattr(rec, 'clim')

    def test_del_clim_idempotent(self):
        rec = make_record()
        rec.del_clim()  # no clim to begin with — should not raise
        rec.del_clim()  # second call also safe


# ---------------------------------------------------------------------------
# ProxyRecord.to_nc() / load_nc()
# ---------------------------------------------------------------------------

class TestProxyRecordIO:
    def test_roundtrip_nc(self, tmp_path):
        rec = make_record()
        path = str(tmp_path / 'rec.nc')
        rec.to_nc(path, verbose=False)
        rec2 = ProxyRecord().load_nc(path)
        np.testing.assert_allclose(rec2.value, rec.value, atol=1e-6)
        assert rec2.lat == pytest.approx(rec.lat)
        assert rec2.ptype == rec.ptype


# ---------------------------------------------------------------------------
# ProxyDatabase.from_df()
# ---------------------------------------------------------------------------

class TestProxyDatabaseFromDf:
    def _make_df(self, n=2, seed=99):
        import pandas as pd
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            'paleoData_pages2kID': [f'tree_{i:03d}' for i in range(n)],
            'geo_meanLat': [float(i * 20 - 20) for i in range(n)],
            'geo_meanLon': [float(i * 30) for i in range(n)],
            'geo_meanElev': [0.0] * n,
            'year': [np.arange(1900, 1950, dtype=float) for _ in range(n)],
            'paleoData_values': [rng.standard_normal(50) for _ in range(n)],
            'ptype': ['tree.TRW'] * n,
            'paleoData_variableName': ['TRW'] * n,
            'paleoData_units': ['mm'] * n,
        })

    def test_nrec_matches_df_rows(self):
        pdb = ProxyDatabase().from_df(self._make_df(n=3))
        assert pdb.nrec == 3

    def test_pids_from_pid_column(self):
        pdb = ProxyDatabase().from_df(self._make_df(n=2))
        assert 'tree_000' in pdb.pids
        assert 'tree_001' in pdb.pids

    def test_ptype_preserved(self):
        pdb = ProxyDatabase().from_df(self._make_df(n=1))
        assert pdb.records['tree_000'].ptype == 'tree.TRW'

    def test_wrong_input_type_raises(self):
        with pytest.raises(TypeError):
            ProxyDatabase().from_df([1, 2, 3])


# ---------------------------------------------------------------------------
# ProxyDatabase.find_duplicates()
# ---------------------------------------------------------------------------

class TestProxyDatabaseFindDuplicates:
    def test_identical_records_detected(self):
        rng = np.random.default_rng(0)
        time = np.arange(1950, 2000, dtype=float)
        value = rng.standard_normal(50)
        rec1 = ProxyRecord(pid='rec_A', lat=30.0, lon=90.0,
                           time=time.copy(), value=value.copy(), ptype='tree.TRW')
        rec2 = ProxyRecord(pid='rec_B', lat=30.0, lon=90.0,
                           time=time.copy(), value=value.copy(), ptype='tree.TRW')
        pdb = ProxyDatabase(records={'rec_A': rec1, 'rec_B': rec2})
        pdb_dups = pdb.find_duplicates(r_thresh=0.99, time_period=[1950, 1999])
        assert pdb_dups.nrec == 2

    def test_uncorrelated_records_not_flagged(self):
        rng = np.random.default_rng(0)
        time = np.arange(1950, 2000, dtype=float)
        rec1 = ProxyRecord(pid='rec_A', lat=30.0, lon=90.0,
                           time=time.copy(), value=rng.standard_normal(50), ptype='tree.TRW')
        rec2 = ProxyRecord(pid='rec_B', lat=30.0, lon=90.0,
                           time=time.copy(), value=rng.standard_normal(50), ptype='tree.TRW')
        pdb = ProxyDatabase(records={'rec_A': rec1, 'rec_B': rec2})
        pdb_dups = pdb.find_duplicates(r_thresh=0.99, time_period=[1950, 1999])
        assert pdb_dups.nrec == 0

    def test_duplicate_groups_attribute(self):
        rng = np.random.default_rng(5)
        time = np.arange(1950, 2000, dtype=float)
        value = rng.standard_normal(50)
        rec1 = ProxyRecord(pid='rec_A', lat=0.0, lon=0.0,
                           time=time.copy(), value=value.copy(), ptype='tree.TRW')
        rec2 = ProxyRecord(pid='rec_B', lat=0.0, lon=0.0,
                           time=time.copy(), value=value.copy(), ptype='tree.TRW')
        pdb = ProxyDatabase(records={'rec_A': rec1, 'rec_B': rec2})
        pdb_dups = pdb.find_duplicates(r_thresh=0.9, time_period=[1950, 1999])
        assert hasattr(pdb_dups, 'groups')
        assert len(pdb_dups.groups) == 1


# ---------------------------------------------------------------------------
# ProxyDatabase.filter() — extended modes
# ---------------------------------------------------------------------------

class TestProxyDatabaseExtendedFilter:
    def test_filter_by_dt_includes_annual(self):
        pdb = make_database(4)  # all records have dt=1 (annual)
        filtered = pdb.filter(by='dt', keys=[0.9, 1.1])
        assert filtered.nrec == 4

    def test_filter_by_dt_excludes_annual(self):
        pdb = make_database(4)  # dt=1; filter for sub-annual only
        filtered = pdb.filter(by='dt', keys=[0.01, 0.5])
        assert filtered.nrec == 0

    def test_filter_by_loc_circle(self):
        # make_database(3): lats=[-10,0,10], lons=[0,20,40]
        pdb = make_database(3)
        # centre (0, 20), radius 5000 km covers all three records
        filtered = pdb.filter(by='loc-circle', keys=[0, 20, 5000])
        assert filtered.nrec == 3

    def test_filter_by_loc_square(self):
        # make_database(6): lats=[-10,0,10,20,30,40], lons=[0,20,40,60,80,100]
        pdb = make_database(6)
        # lat in [-15, 15], lon in [0, 45] → records (lat=-10,lon=0), (lat=0,lon=20), (lat=10,lon=40)
        filtered = pdb.filter(by='loc-square', keys=[-15, 15, -1, 45])
        assert filtered.nrec == 3
        for _, r in filtered.records.items():
            assert -15 <= r.lat <= 15


# ---------------------------------------------------------------------------
# ProxyDatabase.standardize()
# ---------------------------------------------------------------------------

class TestProxyDatabaseStandardize:
    def test_returns_database(self):
        pdb = make_database(3)
        result = pdb.standardize(ref_period=[1950, 1999], thresh=1)
        assert isinstance(result, ProxyDatabase)

    def test_standardized_tag_present(self):
        pdb = make_database(3)
        result = pdb.standardize(ref_period=[1950, 1999], thresh=1)
        for _, rec in result.records.items():
            assert 'standardized' in rec.tags


# ---------------------------------------------------------------------------
# ProxyDatabase.nrec_tags()
# ---------------------------------------------------------------------------

class TestProxyDatabaseNrecTags:
    def test_counts_matching_records(self):
        pdb = make_database(4)
        pdb.records['rec_000'].tags.add('selected')
        pdb.records['rec_001'].tags.add('selected')
        assert pdb.nrec_tags('selected') == 2

    def test_returns_zero_for_absent_tag(self):
        pdb = make_database(3)
        assert pdb.nrec_tags('nonexistent') == 0

    def test_all_tagged(self):
        pdb = make_database(3)
        for pid in pdb.pids:
            pdb.records[pid].tags.add('mytag')
        assert pdb.nrec_tags('mytag') == 3

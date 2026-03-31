"""
Minimal end-to-end tests for cfr.reconjob.ReconJob using fully synthetic data.

Inspired by docsrc/notebooks/pp2k-ppe-pda.ipynb, simplified to:
- A synthetic annual prior (4×4 grid, 100 years: 1200–1299)
- 3 synthetic proxy records co-located with prior grid points (TempPlusNoise PSM)
- recon_period=[1257, 1258], recon_seeds=[0], nens=5
- No network access, no file I/O for inputs

The fixture runs the pipeline once (module scope); individual tests inspect
the intermediate state and the saved netCDF output.
"""

import numpy as np
import pytest
import xarray as xr
import matplotlib
matplotlib.use('Agg')

from cfr.climate import ClimateField
from cfr.proxy import ProxyRecord, ProxyDatabase
from cfr.reconjob import ReconJob


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_prior(nlat=4, nlon=4, nyears=100, start_year=1200, seed=0):
    """Annual ClimateField for use as prior — no file I/O."""
    rng = np.random.default_rng(seed)
    lat = np.linspace(-75, 75, nlat)
    lon = np.linspace(0, 270, nlon)
    time = np.arange(start_year, start_year + nyears, dtype=float)
    value = rng.standard_normal((nyears, nlat, nlon)) + 288.0
    fd = ClimateField().from_np(time=time, lat=lat, lon=lon, value=value)
    fd.da.name = 'tas'
    return fd


def make_proxy_db(n=3, start_year=1200, nyears=100, seed=1):
    """ProxyDatabase with records co-located at prior grid points."""
    rng = np.random.default_rng(seed)
    # lat/lon match grid points produced by make_prior(nlat=4, nlon=4)
    lats = [-75.0, -25.0, 25.0, 75.0][:n]
    lons = [0.0, 90.0, 180.0, 270.0][:n]
    time = np.arange(start_year, start_year + nyears, dtype=float)
    records = {}
    for i in range(n):
        pid = f'syn_{i:03d}'
        value = rng.standard_normal(nyears)
        rec = ProxyRecord(
            pid=pid, lat=lats[i], lon=lons[i], elev=0.0,
            time=time.copy(), value=value, ptype='tree.TRW',
        )
        records[pid] = rec
    return ProxyDatabase(records=records)


# ---------------------------------------------------------------------------
# Shared fixture: run the pipeline once per module
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def calibrated_job():
    """Return a ReconJob after calib_psms + forward_psms (no DA)."""
    prior = make_prior()
    pdb = make_proxy_db()

    job = ReconJob()
    job.prior = {'tas': prior}
    job.proxydb = pdb

    job.calib_psms(
        ptype_psm_dict={'tree.TRW': 'TempPlusNoise'},
        ptype_clim_dict={'tree.TRW': ['tas']},
    )
    job.forward_psms()
    return job


@pytest.fixture(scope='module')
def recon_output(tmp_path_factory):
    """Run the full DA pipeline once; return output directory (Path)."""
    outdir = tmp_path_factory.mktemp('recon')

    prior = make_prior()
    pdb = make_proxy_db()

    job = ReconJob()
    job.prior = {'tas': prior}
    job.proxydb = pdb

    job.calib_psms(
        ptype_psm_dict={'tree.TRW': 'TempPlusNoise'},
        ptype_clim_dict={'tree.TRW': ['tas']},
    )
    job.forward_psms()
    job.run_da_mc(
        recon_period=[1257, 1258],
        recon_seeds=[0],
        nens=5,
        trim_prior=False,       # sample from full prior, avoids edge-case with only 2 yrs
        save_dirpath=str(outdir),
        output_indices=['gm'],  # skip nino3.4 etc. that need a global grid
    )
    return outdir


# ---------------------------------------------------------------------------
# PSM calibration
# ---------------------------------------------------------------------------

class TestReconJobCalibPSMs:
    def test_all_records_tagged_calibrated(self, calibrated_job):
        for pid, pobj in calibrated_job.proxydb.records.items():
            assert 'calibrated' in pobj.tags, f'{pid} not tagged calibrated'

    def test_calibrated_records_have_R(self, calibrated_job):
        for pid, pobj in calibrated_job.proxydb.records.items():
            if 'calibrated' in pobj.tags:
                assert hasattr(pobj, 'R'), f'{pid} missing R'
                assert np.isfinite(pobj.R), f'{pid}.R is not finite'

    def test_calibrated_records_have_psm(self, calibrated_job):
        for pid, pobj in calibrated_job.proxydb.records.items():
            if 'calibrated' in pobj.tags:
                assert hasattr(pobj, 'psm')
                assert pobj.psm.calib_details is not None


# ---------------------------------------------------------------------------
# PSM forward
# ---------------------------------------------------------------------------

class TestReconJobForwardPSMs:
    def test_calibrated_records_have_pseudo(self, calibrated_job):
        for pid, pobj in calibrated_job.proxydb.records.items():
            if 'calibrated' in pobj.tags:
                assert hasattr(pobj, 'pseudo'), f'{pid} missing pseudo'
                assert pobj.pseudo is not None

    def test_pseudo_is_proxyrecord(self, calibrated_job):
        from cfr.proxy import ProxyRecord
        for pid, pobj in calibrated_job.proxydb.records.items():
            if 'calibrated' in pobj.tags:
                assert isinstance(pobj.pseudo, ProxyRecord)

    def test_pseudo_values_finite(self, calibrated_job):
        for pid, pobj in calibrated_job.proxydb.records.items():
            if 'calibrated' in pobj.tags:
                assert np.all(np.isfinite(pobj.pseudo.value)), f'{pid}.pseudo.value has non-finite values'


# ---------------------------------------------------------------------------
# Full DA run — output file checks
# ---------------------------------------------------------------------------

class TestReconJobDA:
    def test_output_file_created(self, recon_output):
        assert (recon_output / 'job_r00_recon.nc').exists()

    def test_output_has_gm_variable(self, recon_output):
        ds = xr.open_dataset(recon_output / 'job_r00_recon.nc')
        assert 'tas_gm' in ds
        ds.close()

    def test_output_time_length(self, recon_output):
        ds = xr.open_dataset(recon_output / 'job_r00_recon.nc')
        # recon_period=[1257, 1258], recon_timescale=1 → 2 reconstructed years
        assert ds['tas_gm'].sizes['time'] == 2
        ds.close()

    def test_output_values_finite(self, recon_output):
        ds = xr.open_dataset(recon_output / 'job_r00_recon.nc')
        assert np.all(np.isfinite(ds['tas_gm'].values))
        ds.close()

    def test_output_time_values(self, recon_output):
        ds = xr.open_dataset(recon_output / 'job_r00_recon.nc')
        times = ds['tas_gm']['time'].values
        assert 1257 in times or 1257.0 in times
        ds.close()


# ---------------------------------------------------------------------------
# _auto_recon_period() / _validated_recon_period()
# ---------------------------------------------------------------------------

class TestReconJobPeriod:
    def test_auto_period_no_proxydb_returns_default(self):
        job = ReconJob()
        assert job._auto_recon_period() == [0, 2000]

    def test_auto_period_with_proxydb(self):
        job = ReconJob()
        job.proxydb = make_proxy_db(start_year=1200, nyears=100)
        lo, hi = job._auto_recon_period()
        assert lo == 1200
        assert hi == 1299

    def test_validated_period_within_coverage_unchanged(self):
        job = ReconJob()
        job.proxydb = make_proxy_db(start_year=1200, nyears=100)
        result = job._validated_recon_period([1250, 1280])
        assert result == [1250, 1280]

    def test_validated_period_clipped_to_coverage(self):
        job = ReconJob()
        job.proxydb = make_proxy_db(start_year=1200, nyears=100)
        result = job._validated_recon_period([1100, 1400])
        assert result[0] >= 1200
        assert result[1] <= 1299

    def test_validated_period_none_returns_none(self):
        job = ReconJob()
        job.proxydb = make_proxy_db()
        assert job._validated_recon_period(None) is None


# ---------------------------------------------------------------------------
# split_proxydb()
# ---------------------------------------------------------------------------

class TestReconJobSplitProxydb:
    def _make_calibrated_job(self, n=4):
        pdb = make_proxy_db(n=n)
        for pid in pdb.pids:
            pdb.records[pid].tags.add('calibrated')
        job = ReconJob()
        job.proxydb = pdb
        return job

    def test_creates_assim_and_eval_tags(self):
        job = self._make_calibrated_job(n=4)
        job.split_proxydb(assim_frac=0.75)
        n_assim = sum(1 for _, r in job.proxydb.records.items() if 'assim' in r.tags)
        n_eval  = sum(1 for _, r in job.proxydb.records.items() if 'eval' in r.tags)
        assert n_assim + n_eval == 4
        assert n_assim == 3   # int(4 * 0.75) = 3
        assert n_eval  == 1

    def test_same_seed_gives_same_split(self):
        job = self._make_calibrated_job(n=4)
        job.split_proxydb(seed=42, assim_frac=0.5)
        tags1 = {pid: frozenset(r.tags) for pid, r in job.proxydb.records.items()}

        # reset tags and re-split with same seed
        for _, r in job.proxydb.records.items():
            r.tags = {'calibrated'}
        job.split_proxydb(seed=42, assim_frac=0.5)
        tags2 = {pid: frozenset(r.tags) for pid, r in job.proxydb.records.items()}
        assert tags1 == tags2

    def test_all_tagged_records_assigned(self):
        # Every calibrated record gets either assim or eval — none left unassigned
        job = self._make_calibrated_job(n=4)
        job.split_proxydb(assim_frac=0.5)
        for pid, r in job.proxydb.records.items():
            has_assim = 'assim' in r.tags
            has_eval  = 'eval' in r.tags
            assert has_assim or has_eval, f'{pid} got neither tag'

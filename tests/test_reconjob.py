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
    lats = [-75.0, -25.0, 25.0][:n]
    lons = [0.0, 90.0, 180.0][:n]
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

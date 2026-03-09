# Coverage Gaps — Features Not Covered by Notebooks

This file catalogues `cfr` API features that are **not demonstrated in any
existing notebook** under `docsrc/notebooks/` and therefore lack notebook-level
"documentation-as-test" coverage. Unit tests for these features still need to
be written.

Items marked ✅ have been added to the test suite.

---

## cfr.ts.EnsTS

| Method | Gap |
|--------|-----|
| `fetch(name)` | Remote network load; needs mocking or integration-test fixture |
| `line_density(...)` | 2-D histogram plot; only smoke-tested in notebooks implicitly |

---

## cfr.climate.ClimateField

| Method | Gap |
|--------|-----|
| `fetch(name)` | Downloads from cloud; needs mock or integration test |
| ✅ `get_anom(ref_period)` | Added in `test_climate.py::TestClimateFieldGetAnom` |
| `get_eof(n, ...)` | EOF decomposition; no notebook exercises this |
| `plot_eof(...)` | EOF plot; ditto |
| `plotly_grid(...)` | Interactive plotly map; no notebook covers it |
| ✅ `regrid(lats, lons)` | Added in `test_climate.py::TestClimateFieldRegrid` |
| ✅ `annualize(months)` | Added in `test_climate.py::TestClimateFieldAnnualize` |
| `index('nino3.4')` / `'nino1+2'` / `'tpi'` / `'dmi'` / `'iobw'` / `'wpi'` | Only `'gm'`/`'nhm'`/`'shm'` tested; others untested |
| `__floordiv__` | Not demonstrated anywhere |
| `wrap_lon()` | Covered in `test_climate.py::TestClimateFieldWrapLon` |
| ✅ `load_nc` / `to_nc` | Round-trip added in `test_climate.py::TestClimateFieldIO` |

---

## cfr.proxy.ProxyRecord

| Method | Gap |
|--------|-----|
| ✅ `get_clim(fields, ...)` | Added in `test_proxy.py::TestProxyRecordGetClim` |
| `correct_elev_tas(t_rate)` | Elevation-lapse-rate correction; never demonstrated |
| `get_pseudo(psm, ...)` | Indirectly covered via `test_reconjob.py` (forward_psms) |
| ✅ `del_clim()` | Added in `test_proxy.py::TestProxyRecordGetClim` |
| `del_pseudo()` | Trivial; `del_clim` pattern is covered |
| `dashboard()` | Requires `pyleoclim`; no unit test |
| `dashboard_clim()` | Requires `pyleoclim`; no unit test |
| `plot_dups()` | Duplicate-plot method; no test |
| `plot_compare(ref, ...)` | Compare-plot method; no test |
| `plotly(...)` | Interactive plot; no test |
| ✅ `to_nc` / `load_nc` | Round-trip added in `test_proxy.py::TestProxyRecordIO` |

---

## cfr.proxy.ProxyDatabase

| Method | Gap |
|--------|-----|
| `fetch(name)` | Cloud load; needs mock |
| ✅ `from_df(df, ...)` | Added in `test_proxy.py::TestProxyDatabaseFromDf` |
| `to_nc` / `load_nc` | Database-level I/O round-trip; still missing |
| ✅ `find_duplicates(r_thresh, time_period)` | Added in `test_proxy.py::TestProxyDatabaseFindDuplicates` |
| `squeeze_dups(pids_to_keep)` | Untested |
| ✅ `nrec_tags(keys)` | Added in `test_proxy.py::TestProxyDatabaseNrecTags` |
| `plotly(...)` / `plotly_concise(...)` / `plotly_count(...)` | Interactive visualisations; no tests |
| ✅ `filter(by='dt', ...)` | Added in `test_proxy.py::TestProxyDatabaseExtendedFilter` |
| ✅ `filter(by='loc-circle', ...)` / `filter(by='loc-square', ...)` | Added in same class |
| `clear_proxydb_tags()` | Convenience method; no test |
| ✅ `standardize(ref_period)` | Added in `test_proxy.py::TestProxyDatabaseStandardize` |

---

## cfr.psm — Advanced PSMs

All of the following PSM classes are absent from unit tests and rely on
optional/external packages (`pathos`, `fbm`, `PyVSL`, PRYSM, BayMag, etc.):

| Class | Missing coverage |
|-------|-----------------|
| ✅ `TempPlusNoise` | `calibrate()` + `forward()` covered in `test_reconjob.py` |
| `Ice_d18O` | `forward()` — complex multi-variable ice model |
| `Lake_VarveThickness` | `forward()` — construction smoke-tested only |
| `Coral_SrCa` | `forward()` — construction smoke-tested only |
| `Coral_d18O` | `forward()` — construction smoke-tested only |
| `VSLite` | `calibrate()` + `forward()` |
| `BayTEX86` | `forward()` |
| `BayUK37` | `forward()` |
| `BayD18O` | `forward()` |
| `BayMgCa` | `forward()` |

Recommended approach: use `pytest.importorskip` to skip each test when the
required optional package is absent, then test with a minimal synthetic climate.

---

## cfr.reconjob.ReconJob

| Method | Gap |
|--------|-----|
| ✅ `run_da(...)` | Covered via `run_da_mc` in `test_reconjob.py` |
| ✅ `run_da_mc(...)` | Added in `test_reconjob.py::TestReconJobDA` |
| ✅ `calib_psms(...)` | Added in `test_reconjob.py::TestReconJobCalibPSMs` |
| ✅ `forward_psms(...)` | Added in `test_reconjob.py::TestReconJobForwardPSMs` |
| `run_graphem(...)` | GraphEM solver; requires optional `cfr-graphem` package |
| `graphem_kcv(...)` | Cross-validation; same |
| `prep_graphem(...)` | Setup; same |
| ✅ `split_proxydb(...)` | Added in `test_reconjob.py::TestReconJobSplitProxydb` |
| `load_clim(...)` / `regrid_clim(...)` / `crop_clim(...)` / `annualize_clim(...)` | Climate preprocessing steps; testable with synthetic netCDF |
| `save(...)` / `load(...)` | Pickle I/O round-trip |
| `save_cfg(...)` / `io_cfg(...)` | Config I/O |
| ✅ `_auto_recon_period()` | Added in `test_reconjob.py::TestReconJobPeriod` |
| ✅ `_validated_recon_period(...)` | Added in same class |
| `mark_pids(...)` / `clear_proxydb_tags(...)` | Tag management |
| `prep_da_cfg(...)` / `run_da_cfg(...)` / `run_graphem_cfg(...)` | Config-file-driven workflows |

---

## cfr.reconres.ReconRes

| Method | Gap |
|--------|-----|
| `load(vn_list)` | Requires saved reconstruction output on disk |
| `valid(target_dict, stat, timespan)` | Validation pipeline; needs saved recon |
| `plot_valid(...)` | Validation plot |
| `load_proxylabels()` | Proxy label loading |
| `indpdt_verif(...)` | Independent verification; needs saved job |
| `plot_indpdt_verif()` | Independent verification plot |

Recommended approach: create a small synthetic reconstruction output in a
pytest fixture (via `tmp_path`) and load it with `ReconRes`.

---

## cfr.gcm.GCMCase / GCMCases

No notebook demonstrates `GCMCase` or `GCMCases` in an accessible way. Gaps:

| Method | Gap |
|--------|-----|
| `GCMCase.load(vars, ...)` | Requires CESM/CAM NetCDF output |
| `GCMCase.calc_atm_gm(vars)` | Atmosphere global mean calculation |
| `GCMCase.calc_som_forcings(...)` | Slab ocean model; highly specialised |
| `GCMCase.calc_cam_forcings(...)` | CAM forcings; highly specialised |
| `GCMCase.to_ds()` / `to_nc()` / `load_nc()` | I/O |
| `GCMCases.calc_atm_gm(vars)` | Multi-case wrapper |
| `GCMCases.plot_ts(...)` | Multi-case plot |

---

## cfr.utils — Standalone Functions

| Function | Gap |
|----------|-----|
| ✅ `ols_ts(...)` | Added in `test_utils.py::TestOlsTs` |
| `regrid_field(...)` | Spectral regridding (requires `spharm`); no test |
| `regrid_field_curv_rect(...)` | Curvilinear→rectilinear regridding; no test |
| ✅ `geo_mean(da, ...)` | Added in `test_utils.py::TestGeoMean` |
| `annualize_var(year_float, var, ...)` | Variable array annualisation; no test |
| `colored_noise_2regimes(...)` | Two-regime colored noise; no test |
| `arr_str2np(arr)` | String-to-numpy array parser; no test |
| `is_numeric(obj)` | Duck-type numeric check; no test |
| `replace_str(fpath, d)` | File string replacement; no test |
| `download(url, fname, ...)` | Network download; needs mock |
| ✅ `year_float2dates(year_float)` | Added in `test_utils.py::TestYearFloat2Dates` |

---

## Summary Statistics

| Module | Total public API items | Items with unit tests | Coverage estimate |
|--------|----------------------|----------------------|-------------------|
| `utils.py` | ~25 | ~15 | ~60% |
| `ts.py` (EnsTS) | ~20 | ~18 | ~90% |
| `proxy.py` (ProxyRecord) | ~25 | ~19 | ~76% |
| `proxy.py` (ProxyDatabase) | ~20 | ~17 | ~85% |
| `climate.py` (ClimateField) | ~25 | ~22 | ~88% |
| `psm.py` | ~30 | ~8 | ~27% |
| `reconjob.py` | ~35 | ~10 | ~29% |
| `reconres.py` | ~7 | 0 | 0% |
| `gcm.py` | ~12 | 0 | 0% |

**Remaining priority targets:**
1. `ReconRes.load()` + `valid()` — create synthetic recon output in a fixture, test loading and validation
2. `ReconJob.save()` / `load()` — pickle round-trip (straightforward once synthetic pipeline works)
3. `ReconJob.annualize_clim()` / `regrid_clim()` / `crop_clim()` — testable with synthetic cftime prior
4. PSM `forward()` for `Coral_SrCa`, `Coral_d18O`, `Lake_VarveThickness` — skip when deps missing
5. `utils.geo_mean` extended tests (already done); `annualize_var`; `arr_str2np`
6. `ProxyDatabase.to_nc` / `load_nc` — database-level I/O round-trip

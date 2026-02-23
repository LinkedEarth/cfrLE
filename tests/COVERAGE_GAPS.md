# Coverage Gaps — Features Not Covered by Notebooks

This file catalogues `cfr` API features that are **not demonstrated in any
existing notebook** under `docsrc/notebooks/` and therefore lack notebook-level
"documentation-as-test" coverage. Unit tests for these features still need to
be written.

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
| `get_anom(ref_period)` | Anomaly w.r.t. reference period; not directly asserted in any notebook code |
| `get_eof(n, ...)` | EOF decomposition; no notebook exercises this |
| `plot_eof(...)` | EOF plot; ditto |
| `plotly_grid(...)` | Interactive plotly map; no notebook covers it |
| `regrid(lats, lons)` | Regridding; notebooks show it but no assertions are made; needs unit test verifying output grid |
| `annualize(months)` | Notebook shows call but no output check |
| `index('nino3.4')` / `'nino1+2'` / `'tpi'` / `'dmi'` / `'iobw'` / `'wpi'` | Only `'gm'`/`'nhm'`/`'shm'`/`'nino3.4'` are notebook-exercised; others untested |
| `__floordiv__` | Not demonstrated anywhere |
| `wrap_lon()` | Called internally; no notebook assertion |
| `load_nc` / `to_nc` | I/O; notebooks call but assertions are implicit; deserves a round-trip test |

---

## cfr.proxy.ProxyRecord

| Method | Gap |
|--------|-----|
| `get_clim(fields, ...)` | Nearest-climate extraction; requires attached ClimateField objects — no isolated test |
| `correct_elev_tas(t_rate)` | Elevation-lapse-rate correction; never demonstrated |
| `get_pseudo(psm, ...)` | Pseudoproxy generation pipeline; tested indirectly in pp2k notebooks but not unit-tested |
| `del_clim()` / `del_pseudo()` | Cleanup methods; never tested |
| `dashboard()` | Requires `pyleoclim`; no unit test |
| `dashboard_clim()` | Requires `pyleoclim`; no unit test |
| `plot_dups()` | Duplicate-plot method; no test |
| `plot_compare(ref, ...)` | Compare-plot method; no test |
| `plotly(...)` | Interactive plot; no test |
| `to_nc` / `load_nc` | Proxy round-trip I/O; deserves a tmp-file round-trip test |
| `from_da` / `to_da` | Covered in `test_proxy.py` but only for simple fields |

---

## cfr.proxy.ProxyDatabase

| Method | Gap |
|--------|-----|
| `fetch(name)` | Cloud load; needs mock |
| `from_df(df, ...)` | Covered in proxy-io notebook but not in unit tests |
| `to_nc` / `load_nc` | Database-level I/O round-trip |
| `find_duplicates(r_thresh, time_period)` | Never notebook-tested with assertions |
| `squeeze_dups(pids_to_keep)` | Ditto |
| `nrec_tags(keys)` | Tag count helper; untested |
| `plotly(...)` / `plotly_concise(...)` / `plotly_count(...)` | Interactive visualisations; no tests |
| `filter(by='dt', ...)` | Resolution-based filtering; not covered by current unit tests |
| `filter(by='loc-circle', ...)` / `filter(by='loc-square', ...)` | Geographic filtering modes; no unit tests |
| `clear_proxydb_tags()` | Convenience method; no test |
| `standardize(ref_period)` | Database-level standardisation; no unit test (only `center` is tested) |

---

## cfr.psm — Advanced PSMs

All of the following PSM classes are absent from unit tests and rely on
optional/external packages (`pathos`, `fbm`, `PyVSL`, PRYSM, BayMag, etc.):

| Class | Missing coverage |
|-------|-----------------|
| `TempPlusNoise` | `calibrate()` + `forward()` |
| `Ice_d18O` | `forward()` — complex multi-variable ice model |
| `Lake_VarveThickness` | `forward()` |
| `Coral_SrCa` | `forward()` |
| `Coral_d18O` | `forward()` |
| `VSLite` | `calibrate()` + `forward()` |
| `BayTEX86` | `forward()` |
| `BayUK37` | `forward()` |
| `BayD18O` | `forward()` |
| `BayMgCa` | `forward()` |

Recommended approach: use `pytest.importorskip` to skip each test when the
required optional package is absent, then test with a minimal synthetic climate.

---

## cfr.reconjob.ReconJob

Almost the entire class is untested at the unit level. Key gaps:

| Method | Gap |
|--------|-----|
| `run_da(...)` | Core DA reconstruction; needs synthetic prior + proxies; computationally expensive |
| `run_da_mc(...)` | Monte-Carlo DA; same challenge |
| `calib_psms(...)` | PSM calibration orchestration; needs full pipeline setup |
| `forward_psms(...)` | PSM forward orchestration |
| `run_graphem(...)` | GraphEM solver; requires optional `cfr-graphem` package |
| `graphem_kcv(...)` | Cross-validation; same |
| `prep_graphem(...)` | Setup; same |
| `split_proxydb(...)` | Random assimilation/verification split; testable with synthetic data |
| `load_clim(...)` / `regrid_clim(...)` / `crop_clim(...)` / `annualize_clim(...)` | Climate preprocessing steps; testable with synthetic netCDF |
| `save(...)` / `load(...)` | Pickle I/O round-trip |
| `save_cfg(...)` / `io_cfg(...)` | Config I/O |
| `_auto_recon_period()` / `_validated_recon_period(...)` | Period inference logic; unit-testable but not tested |
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
| `ols_ts(...)` | OLS regression between two time series; no unit test |
| `regrid_field(...)` | Spectral regridding (requires `spharm`); no test |
| `regrid_field_curv_rect(...)` | Curvilinear→rectilinear regridding; no test |
| `geo_mean(da, ...)` | Weighted spatial mean on DataArray; no isolated unit test |
| `annualize_var(year_float, var, ...)` | Variable array annualisation; no test |
| `colored_noise_2regimes(...)` | Two-regime colored noise; no test |
| `arr_str2np(arr)` | String-to-numpy array parser; no test |
| `is_numeric(obj)` | Duck-type numeric check; no test |
| `replace_str(fpath, d)` | File string replacement; no test |
| `download(url, fname, ...)` | Network download; needs mock |
| `year_float2dates(year_float)` | Float-to-`datetime.datetime` conversion; no test |

---

## Summary Statistics

| Module | Total public API items | Items with unit tests | Coverage estimate |
|--------|----------------------|----------------------|-------------------|
| `utils.py` | ~25 | ~12 | ~48% |
| `ts.py` (EnsTS) | ~20 | ~18 | ~90% |
| `proxy.py` (ProxyRecord) | ~25 | ~15 | ~60% |
| `proxy.py` (ProxyDatabase) | ~20 | ~12 | ~60% |
| `climate.py` (ClimateField) | ~25 | ~18 | ~72% |
| `psm.py` | ~30 | ~5 (Linear + Bilinear only) | ~17% |
| `reconjob.py` | ~35 | 0 | 0% |
| `reconres.py` | ~7 | 0 | 0% |
| `gcm.py` | ~12 | 0 | 0% |

**Priority targets for the next round of tests:**
1. `ReconJob._auto_recon_period()` and `_validated_recon_period()` — pure logic, easy to test
2. `ReconJob.split_proxydb()` — random split, testable with synthetic data
3. `utils.ols_ts` and `utils.geo_mean` — standalone numerical functions
4. `ProxyDatabase.find_duplicates()` — important for data-quality workflows
5. `ProxyRecord.get_pseudo()` with `TempPlusNoise` — core pseudoproxy pipeline
6. Round-trip I/O tests for `ProxyRecord.to_nc`/`load_nc` and `ClimateField.to_nc`/`load_nc`

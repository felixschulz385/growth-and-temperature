# esacci — ESA CCI Land Cover annual composites

- **Registry id:** `esacci`
- **Class:** `EsacciSource` (`src/data/sources/esacci.py`)
- **Aliases:** `esa_cci`, `esacci_lc`, `landcover`
- **Steps implemented (`STEPS`):** `FETCH`, `PREPARE`, `GRID`
- **`REQUIRES`:** none (default `()`, not overridden)
- **Config key in `data.yaml`:** `sources.esacci`
  ```yaml
  esacci:
    type: "esacci"
    data_path: "esacci/landcover"
    year_range: [1992, 2022]
    verification:
      expected_vars: ["lccs_class"]
      value_range: [0, 220]
  ```
  `namespace` is not set for this source (defaults to `None`), so no `/<namespace>` path segment applies to any step below.

Source data: ESA CCI Land Cover, the `satellite-land-cover` dataset on the Copernicus Climate Data Store (CDS) — a categorical LCCS land-cover class map (`lccs_class`), one annual composite per year.

## FETCH

Issues one CDS API request per year via `cdsapi.Client.retrieve(...)` (synchronous; `download_async` runs it in a thread-pool executor since the CDS client has no native async mode). The remote "URL" is a virtual `cdsapi://satellite-land-cover?year=<YYYY>&version=...` URI parsed back into a `(dataset, request)` pair by `_parse_cdsapi_url`. Years default to `cfg.year_range` if set, else `range(1992, 2023)` (`list_remote_files`/`get_all_entrypoints`). Versions requested default to `["v2_0_7cds", "v2_1_1"]` (`DEFAULT_VERSIONS`, overridable via `cfg.raw["versions"]`); CDS auth reads `~/.cdsapirc` unless `cfg.raw["cdsapi_rc"]` points elsewhere. Requires `ctx.ssh_target` to be configured, same as the other two sources.

- **Output path**
  - `<data_root>/raw/esacci/landcover`
- **Format:** raw files as downloaded from CDS — one file per year, named `<year>/ESACCI-LC-L4-LCCS-Map-300m-P1Y-<year>-v2.0.7.nc` per the `rel_path` template in `list_remote_files`; the actual file the CDS API returns is a zip archive wrapping a NetCDF (see PREPARE, which unzips it), despite the `.nc`-looking name in the template.
- **Caveats (from code):** on a failed `retrieve()`, `download` removes any partially-written `output_path` before re-raising, so a failed fetch doesn't leave a corrupt file behind. `Completion.NEVER`: the FETCH target always re-runs.

## PREPARE

Builds one annual zarr per year (`.nc4` preferred over `.nc` if both exist for a year, per `_plan_prepare`'s candidate selection). Handles the zip-wrapped NetCDF transparently: tries `zipfile.ZipFile` first and extracts the first `.nc` member to a temp path, falling back to opening the file directly if it isn't actually a zip (`BadZipFile`). Opens with `xr.open_dataset(..., engine="h5netcdf", mask_and_scale=False, ...)`, keeps only the `lccs_class` variable (logs an error and returns `None` if missing), renames dims to `latitude`/`longitude`, ensures `time` (`<year>-12-31`) and `band` dims, writes `EPSG:4326`.

- **Output path**
  - `<data_root>/prepared/esacci/landcover/<year>.zarr`
- **Format:** one zarr store per year, dims `(time=1, band=1, latitude, longitude)`, CRS `EPSG:4326`, chunks `(1, 1, 512, 512)`, Blosc-zstd (level 3, bitshuffle) compression, `zarr_format=3`, `consolidated=False`. Encoding preserves the source dtype explicitly (`"dtype": str(ds[var].dtype)`), rather than forcing a cast.
- **Schema**

  | variable | dtype | notes |
  |---|---|---|
  | `lccs_class` | source dtype preserved (module docstring: "uint8") | categorical LCCS land-cover class code |

- **Caveats:** marker-based completion; same ledger dependency as `acag` (`_plan_prepare` needs the `esacci/landcover` ledger's `completed_fetch_files()` and warns/returns `[]` if it's missing).

## GRID

Reprojects every annual PREPARE zarr onto the pipeline's canonical target geobox into one multi-year timeseries zarr. Since `lccs_class` is categorical, this source explicitly overrides `SpatialProcessor.process_spatial_standard`'s defaults with `dst_nodata=0` and `packaging_attrs={}` (disabling the scale/offset packing the other two sources get by default) — resampling stays at the function's default `"nearest"` (not passed explicitly, but the module docstring calls this out as deliberate: "categorical -> always nearest").

- **Output path**
  - `<data_root>/prepared/<data_path>/crs/<grid_id>/land_cover.zarr` (`<grid_id>` is `ease6933` under the checked-in config)
- **Format:** single multi-year zarr, dims `(time, band=1, <y>, <x>)` (axis names follow the target geobox's CRS — `latitude`/`longitude` for geographic, `y`/`x` for projected e.g. EASE-Grid 2.0 EPSG:6933), CRS via `.rio.write_crs()`/`grid_mapping="spatial_ref"`, Blosc-zstd compression, chunks `(1, 1, 512, 512)`. Storage dtype is `uint16` (`SpatialProcessor.create_empty_target_zarr`'s default `dtype` parameter — not overridden by this source) holding the raw class codes, unscaled (`packaging_attrs={}` means no `scale_factor`/`add_offset` are applied).

- **Variables**

  | name | on-disk dtype | physical meaning | nodata/fill | `value_range` (verification) |
  |---|---|---|---|---|
  | `lccs_class` | `uint16` (unpacked class code, no scale/offset) | LCCS categorical land-cover class | `0` (`dst_nodata=0`) | `[0, 220]` |

  `expected_vars`/`value_range` come from `verify.verification_meta(self.cfg.raw, expected_vars=("lccs_class",), value_range=(0, 220))` in `_plan_grid`, and match `data.yaml`'s `sources.esacci.verification` block (`expected_vars: ["lccs_class"]`, `value_range: [0, 220]`) — the config block reiterates rather than overrides the code defaults.
- **Caveats:** marker-based completion; `_plan_grid` only includes years whose annual PREPARE zarr already exists on disk (`_list_annual_zarrs` scans the PREPARE output directory directly, not the ledger).

**TODO (needs live data):** actual year coverage achieved in a real run (vs. the 1992-2022 `year_range`/1992-2022 fetch default), on-disk store sizes, and the actual set of LCCS class codes present have not been verified against real output and are not claimed here.

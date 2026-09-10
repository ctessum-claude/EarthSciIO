# EarthSciIO.jl

The Julia track of [EarthSciIO](../README.md) — the cross-language data-provider
library. It fulfils the ESS data-loader contract for `.esm` nodes (URL / vars /
native-grid / temporal-cadence) on a content-addressed cache **shared** with the
Python and Rust tracks, so a file fetched by one language is reused byte-for-byte
by the others.

It is the first data-loader machinery in the Julia track, implemented against
the language-neutral spec in [`../spec`](../spec) and the conformance corpus in
[`../conformance`](../conformance). Two components ship here:

- **component (a)** (`esio-9nb.4`): URL download + the content-addressed cache.
- **component (b)** (`esio-9nb.5`): the format readers (native-array decode) and
  the cadence `Provider`.

## What's here

| Piece | Spec | Notes |
|---|---|---|
| Content-addressed cache | [cache-format.md](../spec/cache-format.md) | key = `sha256(resolved_url)`; `$EARTHSCIDATADIR` layout; atomic-rename writes + `mkpidlock` advisory lock |
| Offline mode | [offline-mode.md](../spec/offline-mode.md) | cache-only; a miss raises `CacheMiss`; no socket opened |
| `transport` / `format` / `store` registries | [registries.md](../spec/registries.md) | `http`/`file` transports + `local` store + `netcdf`/`csv`/`geotiff`/`ff10`/`shapefile`/`parquet`/`zarr` readers active; `s3` store registered as a stub |
| Format readers (`read_native`) | [conformance.md](../spec/conformance.md) | `NetCDFReader` (NCDatasets) + `CSVReader` → RAW native-grid arrays keyed by the on-disk `file_variable`; CF scale/offset + `_FillValue`→NaN; time axis kept raw (calendar decode is ESS's job) |
| Cadence `Provider` | [conformance.md](../spec/conformance.md) | `materialize`/`refresh`/`refresh_times`/`prefetch` over `CONST`/`DISCRETE`; provides DATA, not a solver |
| Manifest | [schemas/manifest.schema.json](../spec/schemas/manifest.schema.json) | per-blob validation/provenance; never stores credentials |

Variable remap and unit conversion stay in ESS; regrid stays in ESD/C4. A new
format is one more `register!(FORMAT_REGISTRY, …)` line — never a `Provider`
change.

## Usage

```julia
using EarthSciIO

# --- component (a): fetch a resolved URL into the shared cache ---
cache = Cache(; root = "/scratch.local/$(ENV["USER"])/earthsci-cache")
entry = fetch_blob(cache, "https://data.earthsci.dev/era5/2018/11/20181108.nc")
entry.path        # path to the cached blob (native source bytes)
entry.status      # :downloaded | :hit | :not_modified

# offline (hermetic CI / conformance): cache-only, a miss raises CacheMiss
offline = Cache(; root = corpus_cache_dir, offline = true)

# --- component (b): a Provider returns RAW native arrays ---
# CONST: time-invariant; refresh_times is empty, materialized once.
p = const_provider(offline, url; format = "netcdf", source_loader = "era5")
nds = materialize(p)              # NativeDataset
nds["t2m"].data                   # Float64 array, NaN at _FillValue cells
nds["time"].data                  # raw Int32 axis; nds["time"].attrs["units"]/["calendar"]

# DISCRETE: time-varying; the library EXPOSES the cadence, the solver drives it.
pd = discrete_provider(offline, url, times; format = "netcdf", time_dim = "time")
refresh_times(pd)                 # the cadence grid (matches `times`)
# e.g. PresetTimeCallback(refresh_times(pd), integ -> use(refresh(pd, integ.t)))
prefetch(pd)                      # warm the cache for every tick's URL, no decode
```

`offline` defaults to the `EARTHSCI_OFFLINE` environment variable; an explicit
argument wins. `EARTHSCIDATADIR` selects the cache root (default on
`/scratch.local`, never `/u`).

### HTTP transport timeouts

The `http`/`https`/`s3` transport bounds a fetch on three independent axes: a
bytes/s floor (a stalled socket), a per-attempt wall-clock cap (a lost-wakeup
deadlock, which no socket-level floor can see), and a cap on the whole call. A
per-attempt cap alone would also cap the *size* of a fetchable blob, so an
attempt that timed out **having written bytes** buys a 4× bigger cap and is not
charged a retry — up to `EARTHSCIIO_HTTP_TIMEOUT_MAX`, which bounds the entire
`fetch!` call, retries and backoff included.

| Variable | Default | Meaning |
| --- | --- | --- |
| `EARTHSCIIO_HTTP_TIMEOUT` | `90` | per-attempt wall-clock cap (s) |
| `EARTHSCIIO_HTTP_TIMEOUT_MAX` | `7200` | cap on any one extended attempt, and — together with a `RETRIES × TIMEOUT` floor — on the **whole** fetch of one URL (s) |
| `EARTHSCIIO_HTTP_RETRIES` | `5` | attempts charged (a progress-earning attempt is refunded) |
| `EARTHSCIIO_HTTP_LOW_SPEED_LIMIT` | `1024` | bytes/s floor |
| `EARTHSCIIO_HTTP_LOW_SPEED_TIME` | `30` | abort after this long below the floor (s) |
| `EARTHSCIIO_HTTP_CONNECT_TIMEOUT` | `30` | connect timeout (s) |

A retry restarts the transfer at byte zero (no `Range` resume), so a blob that
needs more than `EARTHSCIIO_HTTP_TIMEOUT` seconds is downloaded more than once;
raise `EARTHSCIIO_HTTP_TIMEOUT` for a workload that is known to be large.

## Tests

```bash
julia --project=julia -e 'using Pkg; Pkg.test()'   # offline + a hermetic localhost server
EARTHSCI_LIVE=1 julia --project=julia -e 'using Pkg; Pkg.test()'  # + opt-in live network smoke
```

The suite covers the fetch/cache/offline cycle, the cross-language reuse of the
committed corpus (shared `sha256` layout), registry dispatch, the multi-process
locking contract, and — for component (b) — the format-decode conformance
(corpus checks 3–4: native-array equality with the Python oracle) plus the
`CONST`/`DISCRETE` cadence Provider over the shared fixture.

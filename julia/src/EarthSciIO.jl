"""
    EarthSciIO

Cross-language data-provider library — Julia track. It fulfils the ESS
data-loader CONTRACT for `.esm` nodes (URL / vars / native-grid /
temporal-cadence) on a content-addressed cache that is **shared** across the
Python / Julia / Rust tracks (key = `sha256(resolved_url)`), so a file fetched
by one language is reused byte-for-byte by the others.

This is the first data-loader machinery in the Julia track. It implements the
EarthSciIO spec (`spec/cache-format.md`, `spec/registries.md`,
`spec/offline-mode.md`, `spec/conformance.md`):

  * **component (a)** — `esio-9nb.4`: the cache. Content-addressed on
    `\$EARTHSCIDATADIR` (atomic-rename writes + per-blob advisory `mkpidlock`),
    ETag / Last-Modified conditional GET, content-hash integrity, TTL, OFFLINE
    mode (cache-only; a miss raises [`CacheMiss`]); the `transport`
    (http/file/+s3-stub) and `store` (local/+s3-stub) registries.
  * **component (b)** — `esio-9nb.5`: the format readers and the cadence
    [`Provider`]. [`NetCDFReader`] (NCDatasets) and [`CSVReader`] register into
    [`FORMAT_REGISTRY`] and return RAW native-grid arrays ([`read_native`]);
    [`Provider`] resolves+fetches+decodes per [`Cadence`] ([`CONST`]/[`DISCRETE`])
    and exposes [`materialize`]/[`refresh`]/[`refresh_times`]/[`prefetch`]. The
    library provides DATA, not a solver: it exposes `refresh_times`; the
    user/solver drives the discrete-cadence update (no solver embedded).

Variable remap and unit conversion stay in ESS; regrid stays in ESD/C4 — the
readers return arrays keyed by the on-disk `file_variable` name, unremapped.
"""
module EarthSciIO

using SHA: sha256
using Dates
using UUIDs: uuid4
using CRC32c: crc32c
import Downloads
import JSON
using FileWatching.Pidfile: mkpidlock
import NCDatasets
import ZipFile

# interfaces + the three extensibility registries
export Registry, register!, registered_names, status_of
export TRANSPORT_REGISTRY, FORMAT_REGISTRY, STORE_REGISTRY, WRITER_REGISTRY
export Transport, Store, Reader, Writer

# cache + store + transport
export Cache, CacheEntry, fetch_blob, cache_key, datadir, is_offline
export Store, LocalStore, S3Store, make_store
export Manifest, OutputManifest, TimeShardRecord
export write_output_manifest, read_output_manifest

# write boundary (Zarr v3 sharded output) — the write mirror of the readers
export ZarrWriter, OutputSchema, OutputVar, CodecProfile, BloscProfile, ZstdProfile
export BLOSC_DIAGNOSTIC, BLOSC_CHECKPOINT, ZSTD_WASM
export write_open!, write_record!, write_flush!, write_close!
export Transport, HttpTransport, FileTransport, S3Transport
export AuthResolver, NoAuth, BearerAuth

# CDS (Copernicus CDS) transport + ERA5 request mapping (esio-9nb.11)
export CdsTransport, CdsAuth, cds_url, cds_api_key, cds_api_endpoint, cds_retrieve
export ERA5_PL_DATASET, ERA5_PRESSURE_LEVELS_HPA, ERA5_VARIABLES
export era5_area, era5_pressure_request, era5_pressure_url

# format readers + native arrays (component b)
export NetCDFReader, CSVReader, GeoTIFFReader, FF10Reader, ShapefileReader,
       ParquetReader, ZarrReader, read_native
export read_store, store_backed, supports_selection, array_shape, dim_length, reader_option_keys
export NativeField, NativeDataset, variable_names, coord_names

# cadence provider (component b)
export Provider, const_provider, discrete_provider, Cadence, CONST, DISCRETE
export materialize, refresh, refresh_times, prefetch, is_const

# errors
export CacheMiss, IntegrityError

include("registries.jl")
include("writer.jl")
include("manifest.jl")
include("store.jl")
include("transport.jl")
include("cds.jl")
include("era5.jl")
include("cache.jl")
include("readers.jl")
include("zarr.jl")
include("zarr_write.jl")
include("provider.jl")

"""Register the built-in transports + stores into the shared registries.

`active` backends ship now; `stub` backends are registered now (so the
registry dispatch is complete and `esio-9nb.8` can exercise them) and gain
real implementations later — with zero change to caller code.
"""
function _register_defaults()
    # transport registry — keyed by URL scheme
    register!(TRANSPORT_REGISTRY, ("http", "https"), HttpTransport(); status = :active)
    register!(TRANSPORT_REGISTRY, "file", FileTransport(); status = :active)
    register!(TRANSPORT_REGISTRY, "cds", CdsTransport(); status = :active)
    # s3 transport (active): anonymous s3:// -> regional-HTTPS rewrite over http.
    register!(TRANSPORT_REGISTRY, "s3", S3Transport(); status = :active)

    # store registry — keyed by store name; value is a factory `(; root, …) -> Store`
    register!(STORE_REGISTRY, "local",
              (; root = datadir(), _kw...) -> LocalStore(root); status = :active)
    register!(STORE_REGISTRY, "s3", (; _kw...) -> S3Store(); status = :stub)

    # format registry — readers returning native arrays (component b). NetCDF
    # (NCDatasets) + CSV are active; the future NetCDF→Zarr path stays a stub so
    # the seam is provable. A new format is one more register! line — never a
    # Provider change.
    register!(FORMAT_REGISTRY, "netcdf", NetCDFReader(); status = :active)
    register!(FORMAT_REGISTRY, "csv", CSVReader(); status = :active)
    # GeoTIFF (LANDFIRE / USGS 3DEP rasters): the decode is supplied by the
    # `EarthSciIOTiffImagesExt` weakdep extension (`using TiffImages`); the format
    # is active once that backend is loaded — without it, `read_native` errors with
    # an install hint (the Python sibling lazy-imports `tifffile` the same way).
    register!(FORMAT_REGISTRY, "geotiff", GeoTIFFReader(); status = :active)
    # FF10 point (SMOKE / Emissions.jl FF10_POINT): '#' header skipped, fixed
    # 77-col schema, numeric->Float64 (blank->NaN), ids/codes/text->String; a zip
    # member is extracted reader-side via the `member` kwarg (ZipFile.jl).
    register!(FORMAT_REGISTRY, "ff10", FF10Reader(); status = :active)
    # shapefile (ESRI feature table): one row per PART, `geometry`
    # [index, vertex, xy] padded by repeating the final vertex, the record's
    # stored bbox and the `.dbf` columns. The decode is supplied by the
    # `EarthSciIOShapefileExt` weakdep extension (`using Shapefile`); without it,
    # `read_native` errors with an install hint (the Python sibling lazy-imports
    # `pyshp` the same way).
    register!(FORMAT_REGISTRY, "shapefile", ShapefileReader(); status = :active)
    # parquet (columnar table): every column a rank-1 field over `index` keyed by
    # its on-disk name, no coords; the dtype is a total function of the parquet
    # logical type (the netcdf reader's narrow/wide integer split, verbatim),
    # temporal columns ride as their RAW integer, a null float is NaN and a null
    # int/string/bool is an error unless `null_int`/`null_string` is declared, and
    # `variables` is a real projection PUSHDOWN. The decode is supplied by the
    # `EarthSciIOParquet2Ext` weakdep extension (`using Parquet2`); without it,
    # `read_native` errors with an install hint (the Python sibling lazy-imports
    # `pyarrow` the same way).
    register!(FORMAT_REGISTRY, "parquet", ParquetReader(); status = :active)
    # zarr (active, store-backed): lazy orthogonal chunk selection; blosc decode
    # is supplied by the `EarthSciIOBloscExt` weakdep extension (`using Blosc`).
    register!(FORMAT_REGISTRY, "zarr", ZarrReader(); status = :active)

    # writer registry — the write mirror of FORMAT_REGISTRY. The `zarr` writer
    # emits sharded Zarr v3; blosc ENcode is supplied by the same
    # `EarthSciIOBloscExt` weakdep. A new output format is one more register! line.
    register!(WRITER_REGISTRY, "zarr", ZarrWriter(); status = :active)
    # netcdf output is a later wave: registered as a STUB so the write seam is
    # provable and dispatch is complete (mirrors the read-side stub discipline).
    register!(WRITER_REGISTRY, "netcdf",
              StubWriter("netcdf", "NetCDF streaming output is a later wave");
              status = :stub)
    return nothing
end

function __init__()
    _register_defaults()
    return nothing
end

end # module

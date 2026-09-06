# EarthSciIO extensibility registries (v1)

**Deliverable (b) of `esio-9nb.1`.** Status: normative.
Machine-readable companion: [`registries.json`](registries.json).

EarthSciIO is **extensible by construction** through three registries. Each
registry is a **name → implementation lookup**. The single load-bearing rule:

> **A new backend registers under a new name WITHOUT touching the Provider API.**
> The Provider depends only on the three *interfaces* below and resolves the
> concrete implementation by name at runtime. Adding S3 transport, a Zarr
> reader, or an object-store backend is a *registration*, never a Provider edit.

This is what lets the future S3-proxy + NetCDF→Zarr cloud path slot in later.
S3 and Zarr are registered **now as stubs** (`status:"stub"` in
`registries.json`) and exercised through the registries by `esio-9nb.8`; their
real implementations land later — with **zero** change to Provider code. What
those real implementations must deliver is charted in
[`cloud-future.md`](cloud-future.md) (the `esio-cloud` epic).

The three registries are orthogonal: a single fetch composes one entry from
each — `transport` gets the bytes, `store` holds them, `format` decodes them.

```
  resolved_url ──▶ [transport:scheme] ──▶ bytes ──▶ [store:name] ──▶ cached blob
                                                          │
                                          cache key = sha256(resolved_url)
                                                          ▼
                                              [format:name] ──▶ native arrays
```

Interfaces are given as **language-neutral pseudo-signatures**. Each language
track binds them to its idiom (Python `Protocol`/ABC, Julia abstract type +
methods, Rust trait); the per-language `Provider` signatures live in the plan
(§4.6) and the per-track beads.

---

## 1. `transport` registry

Keyed by **URL scheme**. Fetches a resolved URL's bytes into the cache.
**Bypassed entirely in offline mode** (the transport is never constructed when
`offline=true`).

```
interface Transport:
    schemes() -> [string]                         # e.g. ["http","https"]
    fetch(resolved_url: string,
          dest: WritablePath,                     # a tmp/<uuid>.part staging path
          conditional: {etag?, last_modified?},   # for revalidation; may be empty
          auth: AuthResolver?) -> FetchResult

FetchResult = {
    status: "downloaded" | "not_modified",        # not_modified ⇒ 304, reuse cache
    etag?: string, last_modified?: string,        # to persist into the manifest
    bytes_written: int
}
```

| name | schemes | status | notes |
|---|---|---|---|
| `http` | `http`, `https` | **active** | GET + conditional GET; mirror failover at the call site |
| `file` | `file` | **active** | local copy; expands `${EARTHSCIDATADIR}` in `file://` templates |
| `cds` | `cds` | **active** | Copernicus CDS API v1: `cds://<dataset>?<request-json>` → submit → poll job → download asset href; auth via the `cds` realm (`PRIVATE-TOKEN`) |
| `s3` | `s3` | **active** | anonymous `s3://<bucket>/<key>` → regional virtual-hosted HTTPS (region default `us-east-2` via `$EARTHSCI_S3_REGION`/`$AWS_REGION`); delegates to the `http` transport (no AWS SDK/SigV4). The `s3://` URL stays canonical in the cache key + manifest |

Registration key = **URL scheme**. The fetch layer reads the resolved URL's
scheme and looks up the transport; an unknown scheme is a registration gap, not
a Provider change. Auth resolvers (CDS/FIRMS/OpenAQ/RDA/bearer) are a separate
pluggable map injected as `auth`, never baked into a transport.

---

## 2. `format` registry

Keyed by **format name** (the "reader" registry). Opens a cached blob and
returns **CF-decoded native-grid arrays keyed by the on-disk `file_variable`
name**, plus native coordinates.

```
interface Reader:
    formats() -> [string]                         # e.g. ["netcdf"]
    extensions() -> [string]                      # sniff hints: ["nc","nc4"]
    open(blob_path: Path) -> Handle
    read_native(handle: Handle,
                variables: [string],              # file_variable names to read
                select: Selection) -> { string: NativeField }   # + coords
    configured(options: {string: any}) -> Reader?  # loader-declared decode options

NativeField = { dtype, dims: [string], shape: [int], data, fill_value? }
```

### 2.1 Reader options (`reader_kwargs`)

A loader carries **format-specific decode options** alongside its format name:
Python/Julia spell them `reader_kwargs` on the `DataSource` and splat them into
`read_native`; Rust spells them `DataSource.reader_options` and resolves them
once, at Provider construction, through `Reader::configured` (its trait method
signature takes no kwargs, so a *configured reader instance* is the Rust idiom).
Either way the meaning is one thing:

> **How a format is decoded is a property of the DECLARATION, not of the
> caller.** An `.esm` data loader that says `{"member_glob": "*egu*",
> "skip_header_row": true}` decodes correctly through any binding, without a
> caller hand-building a registry of pre-configured readers.

An option a reader does not recognise **MUST** be an error at construction, not
a silently-ignored key — an ignored `member_filter` typo reads back as an empty
selection or a mis-parsed table, arbitrarily far from its cause. Reader options
never enter the cache key (the blob is the whole file; member selection and
header handling are decode-side).

| name | ext | status | notes |
|---|---|---|---|
| `netcdf` | `nc`,`nc4`,`cdf` | **active** | CF decode (§decode in [conformance.md](conformance.md#decode)). **`variables` is a projection pushed into the decode** in all three tracks: an unrequested data variable is never decoded (a GEOS-FP A1 file carries 47 of them and a loader wants one), coordinates are always returned, an empty list reads every variable, and a requested name absent from the blob is an error listing what is present |
| `geotiff` | `tif`,`tiff` | **active** | raster bands via GDAL; Py first, Jl/Rs may lag (R5) |
| `csv` | `csv` | **active** | points: numeric cols → float64, others → string |
| `json` | `json` | **active** | points (e.g. station-discovery payloads) |
| `shapefile` | `shp`,`zip` | **active** | ESRI shapefile as a feature table, decoded by a third-party library per track (pyshp / Shapefile.jl / the `shapefile` crate). **One row per PART** (outer ring + holes, mainland + islands), `.dbf` attributes replicated, plus `shape_index`/`part_index`/`n_parts`; `geometry` is `float64[index, vertex, xy]` right-padded by **repeating the final vertex** (esm-spec §8.6.1) with vertices stored verbatim; `nvert_max` pads to a DOCUMENT-declared vertex-axis length instead (a longer part is an error, never a truncation); `xmin`/`ymin`/`xmax`/`ymax` are the record's **stored** bbox; `shape_type` + `crs_wkt` are one-element `meta` string fields. dbf dtypes `N`/`F`→float64 (blank→NaN), `L`→bool, `D`→`YYYYMMDD`, `C`/`M`→string, with `numeric_columns` forcing named `C` columns to float64. A row is deleted **iff** its flag byte is `*` — any other byte, `NUL` included, is live — and a deleted row drops its shape too. The fetchable form is a `.zip`: `member` names the `.shp` (sidecars share its stem), absent `member` requires exactly one, and a bare `.shp` decodes geometry only; none of `member`/`numeric_columns`/`nvert_max` enters the cache key. Reader-only (no reproject/convert/orient/classify/remap) |
| `ff10` | `ff10`,`csv` | **active** | FF10 point long-format (SMOKE/Emissions.jl `FF10_POINT`): `#` header skipped, fixed 77-col schema, numeric→float64 (blank→NaN), ids/codes/text→string; zip member via reader `member` kwarg; multi-member via `members` (explicit list) and/or `member_glob` (fnmatch-style `*`/`?`/`[...]`, case-sensitive, matched against the full member path) — union, deduplicated, rows concatenated in ascending lexicographic (byte) member-name order; absent explicit member / zero-match glob ⇒ error; directory placeholder entries (names ending `/`) never selected; `member` mutually exclusive with `members`/`member_glob`; `skip_header_row` drops exactly one asserted `country_cd` header line per selected input (first non-comment line's first field must be `country_cd` case-insensitively, else error — never silently drops a data row). None of member/members/member_glob/skip_header_row enter the cache key (the blob is the whole zip). Reader-only (no pivot/convert/normalize/filter) |
| `parquet` | `parquet`,`parq`,`pq` | **active** (all three tracks) | Apache Parquet as a **flat table**: every column becomes a rank-1 field over `index`, keyed by its on-disk column name, no coordinates. Decoded by a third-party library per track (arrow-rs's `parquet` crate in Rust, pyarrow in Python, Parquet2.jl in Julia). A Parquet column's logical type is explicit, so the dtype is a total function of the Arrow type: `Boolean`→bool; `Int8`/`Int16`/`Int32`/`UInt8`/`UInt16`→int32 and `Int64`/`UInt32`/`UInt64`→int64 (**the same narrow/wide split as `netcdf`**); floats and `Decimal128`/`Decimal256`→float64 (unscaled ÷ 10^scale); `Utf8`/`LargeUtf8`/`Utf8View`→string; `Dictionary(_,V)`→`V` expanded; `Null`→float64 all-NaN. A `uint64` past `int64::MAX` is an error naming the row, never a wraparound. **Temporal columns ride as their raw integer, undecoded** (`Date32`/`Time32`→int32, `Date64`/`Time64`/`Timestamp`/`Duration`→int64): the Arrow unit and timezone are **not** applied, the rule a CF time axis gets, because an epoch offset → instant is ESS's job. Nested/binary columns have no rank-1 reading — naming one in `variables` is an error, unrequested it is simply not a field. **Null policy:** a null float is `NaN` (the CF `_FillValue` fold) with `fill_value` null; a null **integer, string or boolean is an error** naming the column and row, since those types have no NaN and a default would be a real value silently standing in for a missing one. `null_int` (substituted **and** reported in `fill_value`, like a surviving CF integer fill) and `null_string` open that gate only when the document declares them; a boolean has no such option. `float_columns` forces named columns to float64 whatever their on-disk type — the Parquet twin of shapefile's `numeric_columns` — and does double duty: an integer column whose missing cells must be NaN, **and** a column of fixed-decimal **text** (corpora needing byte-reproducible floats store them as strings, not IEEE doubles; the MOVES snapshots write `meanBaseRate` as `"261.000000000000"`), trimmed and parsed, blank→NaN, anything else unparseable an error naming column/row/text. **Column projection pushes down**: `variables` become a `ProjectionMask`, so only those column chunks are read off disk; empty reads all; an absent name is an error listing what is present. A zero-row file is **typed, not absent** (the schema is in the footer). None of `float_columns`/`null_int`/`null_string` enters the cache key. Reader-only (no row selection/filter/code-map/remap/convert) |
| `zarr` | `zarr` | **active** | **store-backed** Zarr v2: per-array `.zarray`/`.zattrs`, lazy orthogonal chunk selection (fetch only intersecting chunk objects), blosc/lz4+shuffle decode, `<f4`/`<f8`→float64, dims from `_ARRAY_DIMENSIONS`, `fill_value` not→NaN, no coords |

**Hard boundary (Risk R3):** the reader applies **read/decode** semantics only —
CF `scale_factor`/`add_offset`, `_FillValue` → NaN, endianness, chunking. It
does **not** remap `file_variable` → schema name and does **not** apply the
loader's `unit_conversion`. Those are ESS contract semantics and stay in ESS.
The native array is keyed by the **on-disk** variable name.

Format is selected by the loader's declared format (or a content-type /
extension sniff), **never** by trusting the cache-blob suffix alone.

---

## 3. `store` registry

Keyed by **store name** (the "backend" registry). Where the content-addressed
cache physically lives. Realizes the layout in
[cache-format.md](cache-format.md#2-on-disk-layout); the cache **key** is
store-independent.

```
interface Store:
    name() -> string                              # "local"
    exists(key: string) -> bool
    get_blob(key: string) -> Path | bytes | None  # None ⇒ cache miss
    put_blob(key: string, staged: Path) -> void   # atomic commit from tmp staging
    get_meta(key: string) -> Manifest | None
    put_meta(key: string, manifest: Manifest) -> void
    lock(key: string) -> Lock                      # advisory; scope = one blob fetch
```

| name | status | notes |
|---|---|---|
| `local` | **active** | `$EARTHSCIDATADIR` filesystem; `flock` + atomic rename |
| `s3` | **stub** | object store; conditional PUT / `If-None-Match` as the lock analog |

Registration key = **store name** (config-selected). Swapping `local`→`s3`
changes where blobs live; the Provider, the key scheme, and every reader are
untouched.

---

## 4. How the Provider stays unchanged (the invariant, restated)

```
Provider(loader, *, window, offline, auth)        # depends on the 3 INTERFACES only
   transport = transport_registry[scheme_of(url)] # resolved by name
   store     = store_registry[config.store]        # resolved by name
   reader    = format_registry[loader.format]      # resolved by name
```

Adding a backend = add one row to the relevant table in `registries.json` and
register its implementation. No row in this document, and no line in the
Provider, changes shape when S3/Zarr/object-store arrive. `esio-9nb.8` proves
this by registering and exercising the S3 + Zarr **stubs** through exactly these
three lookups.

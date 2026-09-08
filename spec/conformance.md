# EarthSciIO conformance corpus + runner (v1)

**Deliverable (d) of `esio-9nb.1`.** Status: normative.
Corpus: [`../conformance/corpus/`](../conformance/corpus). Reference runner:
[`../conformance/verify.py`](../conformance/verify.py). Generator:
[`../conformance/generate.py`](../conformance/generate.py).

The corpus is the **cross-language correctness guarantee**: the same loader +
the same cached blob must yield the **same native arrays** in Python, Julia, and
Rust — proven **offline**, so FFI is unnecessary (the architecture decision).
The full cross-language harness is `esio-9nb.9`; this spec defines the format it
consumes and ships the Python oracle so the corpus is self-validating today.

---

## 1. The corpus *is* a populated cache

```
conformance/corpus/
  cache/v1/blobs/<key[:2]>/<key>.<ext>    # golden cached blobs (a real $EARTHSCIDATADIR)
  cache/v1/meta/<key>.json                # their manifests
  cases/<id>.json                         # one conformance case per blob (below)
  cases.json                              # case index
```

Point any provider at `conformance/corpus/cache` as `$EARTHSCIDATADIR` with
`offline=true` and every case resolves from disk — no network, no per-language
data tooling beyond the format reader.

### Committed cases (worked examples)

| id | loader | kind | format | transport | store | what it pins |
|---|---|---|---|---|---|---|
| `era5-grid-sub-tile` | era5 | grid | netcdf | file | local | CF scale/offset + `_FillValue`→NaN + a masked cell; packed int16 → float64 |
| `era5-window-slice` | era5 | grid | netcdf | file | local | the **same blob and cache key** as `era5-grid-sub-tile`, read through a **decode-time `select`** (lat `{slice:[1,3]}`, lon `{indices:[0,2]}`, time `"all"`). Pins that a whole-file reader materialises only the hyperslab and returns exactly the full read sliced afterwards, that the lat/lon **coordinates are sliced with the data**, that the masked cell still decodes to NaN inside the window, that the time axis is untouched and raw — and that a selection changes the decode, never the fetch |
| `station-labels-text` | openaq | points | netcdf | file | local | **NetCDF text variables**, all three readings in one blob: `char site_id(site, strlen)` on a private `strlen` is 4 **strings** on `dims ["site"]` (length consumed, trailing NULs stripped, the trailing space of `"cd  "` kept as data, an all-NUL row `""`); `char flag(site)` shares `site` with `float64 value(site)` so `site` is a real axis and every cell is its own **one-character** string (its NUL cell `""`); `char title(titlelen)` alone on a private dimension is a **scalar** string (`dims []`). The rule is about the file, not the variable |
| `station-labels-window` | openaq | points | netcdf | file | local | the **same blob and cache key** as `station-labels-text` under a one-axis `select` (`{indices:[0,2]}`). Pins that a text field is windowed like any other — `site_id` and `flag` lose the cells `value` loses — and that the positional axis match is over the **decoded** field's dims: `site_id` is rank 1 on two on-disk dimensions and the scalar `title` is rank 0 and untouched. Counting on-disk dims would bind the selector to `titlelen` too and return `"h"` for `title` |
| `openaq-points-slice` | openaq | points | csv | file | local | a 2nd reader behind the `format` registry; numeric→float64, text→string |
| `ff10-point-slice` | nei2016 | points | ff10 | file | local | FF10 point long-format: `#` header skipped, fixed 77-col schema, RFC-4180 quoted `FACILITY_NAME`, numeric→float64 (blank→NaN), ids/codes→string; 3 rows share one stack (no pivot). member=null decodes the extracted CSV member |
| `ff10-zip-egu-glob` | nei2016 | points | ff10 | file | local | EPA-2016fd-shaped **zip** of FF10 members (two `*egu*` + one excluded + a glob-matching **directory placeholder** entry, ignored), each member with a non-comment `country_cd,…` header line. Pins `member_glob` selection (exclusion, **sorted member-name concatenation**) + `skip_header_row` (one asserted header line dropped per member). The blob is the whole zip; member selection is reader config, never part of the cache key |
| `shapefile-polygon-zip` | emis_polygons | points | shapefile | file | local | ESRI **shapefile** zipped with its `.shx`/`.dbf`/`.prj` sidecars. Pins the whole reader contract in one layer: **one row per PART** (a mainland + an island decode to two rows with the `.dbf` attributes replicated), the esm-spec §8.6.1 **repeat-final-vertex** padding, the `*`-only deletion rule (a NUL flag byte is NOT a deletion), the record's **stored** bbox replicated to its parts, and the dtype rules (`C`→string, `N`→float64 with blank→NaN, `L`→bool, a `C` code column forced float64 by `numeric_columns`) |
| `isrm-zarr-tile` | isrm | grid | zarr | s3 | local | **store-backed** Zarr v2: lazy orthogonal chunk selection (fetch only the intersecting chunk objects), blosc/lz4+shuffle decode, partial edge chunk, `fill_value` 0.0 NOT→NaN, no coords. `objects[]` per-object key/integrity. |
| `moves-rate-table-parquet` | moves | points | parquet | file | local | MOVES-shaped **Parquet** rate table, 6 rows in **2 row groups** (so the batch seam and row order are pinned), snappy. One column per supported Arrow family: the narrow/wide integer split (a `uint8` 255 read unsigned, a `uint32` 4294967295 that does not fit an int32), a **categorical expanded**, float32 widened bit-exactly, `Decimal128` as unscaled/10^scale, temporal columns as their **raw** stored integer at their stored width, an all-`Null` column as float64/all-NaN. Null policy + both gates: `null_int=-1` fills every integer null **and** is reported in `fill_value`; `null_string` fills a text null *and* a dictionary null while a real `""` survives. `float_columns` does **both** its jobs — decimal TEXT (blank→NaN) and an integer measurement whose nulls become NaN, so no `fill_value` survives it. `variables` is a real **projection**: a readable column and a `Binary` one are both left out |

The `parquet` case carries no nested column, no sub-millisecond timestamp and no
decimal outside `Dec64`'s exact range — each because a Parquet2.jl limit
(recorded in §3 below) makes that shape unenforceable across three tracks, not
because the contract is unclear. Those rules stay normative and are exercised
per-track by `rust/tests/parquet_reader.rs`, `julia/test/test_parquet_reader.jl`
and `tests/test_parquet_reader.py`.
| `permuted-order-tile` | isrm | grid | zarr | s3 | local | **ordered** lazy orthogonal selection: a non-contiguous, PERMUTED index list is returned in the requested order exactly (a reader that sorted the indices fails), fetching 2 of 15 chunks |
| `landfire-raster-tile` | landfire | grid | geotiff | file | local | **GeoTIFF** georef contract: `Band1` naming, `lat`/`lon` axis names from `GTModelTypeGeoKey`, **cell-centre** axes offset half a cell from the tiepoint's corner, **non-square** cells (`sx`≠`sy`), lat decreasing with row (y-up model vs y-down rows), `GDAL_NODATA`→NaN on an interior cell, int16→float64 |

The GeoTIFF entry is **no longer reserved**: `landfire-raster-tile` commits a
424-byte single-band raster authored by `generate.py` with `tifffile`, so no
GDAL and no git-lfs are needed to carry it. Adding it immediately earned its
keep — all three tracks decode the same bytes, and the Rust reader turned out
never to have implemented `GDAL_NODATA`→NaN, which two tracks and two docstrings
had asserted for as long as the reader existed.

The **S3-store** entry remains reserved: the case + manifest shape is defined
here, but the store-backed variant is not committed (binary hosting for a
realistic object store is still an open decision, plan §8).

---

## 2. Case format

Each `cases/<id>.json` validates against
[`schemas/cache-case.schema.json`](schemas/cache-case.schema.json) and carries:

- the registry triple (`transport`/`format`/`store`) that reads it;
- `resolved_url` + `cache_key` (with the invariant `sha256(resolved_url) ==
  cache_key`), `blob_path`, `manifest_path`, `content_sha256`, `bytes`;
- optional `select` (which record/rows to slice) and `decode` (conventions hit);
- `expected.variables` — `file_variable` → **native field** (CF-decoded), and
  `expected.coords` — coordinate → native field. Native fields validate against
  [`schemas/native-field.schema.json`](schemas/native-field.schema.json).

---

## 3. <a name="decode"></a>Decode conventions (the parity contract, Risk R4)

Every reader MUST decode identically, or cross-language equality fails. Pinned:

- **CF packing** — apply `scale_factor` / `add_offset`: `value = raw*scale+off`.
  Packed numeric variables are returned as **float64** (the scale/offset math is
  done in double regardless of the on-disk integer width).
- **Fill / missing** — `_FillValue` (and `missing_value` if present) compares
  **before** unpacking; masked elements become **NaN** (encoded as `null` in the
  corpus `data`).
- **Numeric dtype** — unpacked numeric file variables keep an integer logical
  type (`int64`/`int32`); all other numeric reads are **float64**. This removes
  float32-vs-float64 ambiguity between xarray / NCDatasets / netcdf-rs.
- **Time** — the time coordinate is returned **raw** (the stored integer/float
  values) with its `units` + `calendar` carried as metadata. Calendar decoding
  to wall-clock instants is **ESS's** job, not the reader's.
- **Variable identity** — arrays are keyed by the **on-disk `file_variable`**
  name. No remap, no `unit_conversion` (Risk R3 — those stay in ESS).
- **Projection (`variables`)** — a loader's `variables` are pushed INTO the
  reader wherever the format can honour them (`parquet` column chunks;
  `netcdf`, where an unrequested variable is simply never decoded — a GEOS-FP A1
  file carries 47 and a loader wants one), rather than being applied to an
  already-decoded dataset. **Coordinates are always returned**; an **empty** list
  means every data variable, never none; a requested name absent from the blob is
  an error listing what is present, never a silently missing array. A reader with
  no such option keeps the read-everything-then-select path, which must produce
  the identical result. **"Never a silently missing array" covers a name that is
  present but has no native array reading too** — a Parquet nested/binary column:
  unrequested it is simply not a field, but *requested* it is an error naming the
  type, because the document named an array it would not get. **A variable one
  track cannot decode is not in that category** — it is a divergence, and it is
  fixed by teaching the reader, not by documenting the gap.
- **Strings** — text columns (CSV/JSON) are returned as `string` arrays verbatim.
- **NetCDF text variables are `string` fields in every track.** A `char` variable
  (an ERA5 `expver`, a `char label(n)`) or a NetCDF-4 `NC_STRING` is decoded and
  returned like any other field — read-everything returns it, and the projection
  may name it. No track may skip it: the same bytes must yield the same set of
  fields everywhere. The classic character-array convention decides its shape,
  and the convention is about the file, not the variable:
  - A `char` variable's **last dimension is the string length** exactly when
    nothing else claims it as an axis — the dimension has no coordinate variable
    of its own, **and** every variable using it is a `char` variable using it
    last. That dimension is then consumed: `char label(n, strlen)` is `n`
    strings of up to `strlen` bytes, **not** `n × strlen` single characters, and
    a 1-D `char label(strlen)` on a private dimension is a **scalar** string
    (`dims == []`).
  - Otherwise the last dimension is a real axis and every cell is its own
    one-character string, on the variable's own `dims`/`shape`: `char label(n)`
    beside a `float value(n)` is `n` one-character strings.
  - **Trailing NUL padding is stripped; trailing spaces are data.** A cell that
    is all NULs is the **empty string**, never `"\0"`.
  - An `NC_STRING` is already one string per element; `dims`/`shape` are the
    variable's own.

  This is what xarray produces (its `CharacterArrayCoder`, gated on
  `conventions.stackable`), which makes the Python track the reference for these
  bytes. All three tracks now produce it: neither NCDatasets nor `netcdf-reader`
  applies the convention itself (both hand back the raw characters of the full
  on-disk shape), so the Julia and Rust readers apply it above their backend, and
  `station-labels-text` pins the result three-way. Its second face is under a
  `select`: a `char label(n, strlen)` is a **rank-1 field on two on-disk
  dimensions**, so the positional axis match the selection notes below require
  must count the DECODED field's dims. A track that counted on-disk dims would
  let that variable answer for rank 2, bind an axis to a string LENGTH, and then
  apply that selector by name to every other array in the blob —
  `station-labels-window` pins that too.
- **FF10 zip member selection** — an `ff10` blob may be a `.zip`; the reader's
  `member` (singular), `members` (explicit list), and `member_glob`
  (fnmatch-style `*`/`?`/`[...]`, case-sensitive, matched against the full
  member path) select which member CSV(s) to decode. `members`/`member_glob`
  combine as a **union, deduplicated**, and the selected members are read and
  their rows **concatenated in ascending lexicographic (byte) order of member
  name** — the deterministic multi-member ordering rule all three tracks share.
  An explicit member absent from the archive, and a glob matching zero members,
  are errors. Selection considers only **file** members — directory placeholder
  entries (names ending in `/`, e.g. the real 2016fd zip's `…/ptegu/`) are
  ignored. `member` is mutually exclusive with `members`/`member_glob`. All
  of these are reader config and **never part of the cache key** — the blob is
  the whole zip.
- **FF10 header skip** — with `skip_header_row = true`, after comment (`#`) and
  blank lines are dropped, the first remaining line of **each selected input**
  (each selected zip member, or the bare file) must be a header row — its first
  delimiter-separated field, compared case-insensitively, must equal
  `country_cd` — and **exactly that one line is skipped per member**. If the
  first field is anything else (or no line remains) the reader errors: the
  option asserts a header row and never silently drops a data row. (The EPA
  2016fd members carry this `country_cd,region_cd,…` line as a non-comment row
  of 77 fields, which would otherwise die at the numeric parse of `ann_value`.)
- **Shapefile file set** — a shapefile is a **file set** (`.shp` + `.shx` +
  `.dbf` + `.prj`) but the content-addressed cache holds ONE blob, so the
  fetchable form is a `.zip`. The reader's `member` names the `.shp` inside it
  (sidecars are the same stem with `.dbf`/`.shx`/`.prj`, matched
  case-insensitively on the extension); with `member` absent the archive must
  contain **exactly one** `.shp` — zero or several is an error naming the
  candidates. A **bare `.shp`** blob decodes too, geometry only. `member` is
  reader config and **never part of the cache key**.
- **Shapefile row identity: one row per PART** — a record may carry several
  parts (a polygon's outer ring plus its holes, a county's mainland plus its
  islands). The geometry ops that consume a ring
  (`polygon_intersection_area`, `intersect_polygon`) take ONE ring, so **each
  part becomes one row** of the `index` axis, with the record's `.dbf`
  attributes **replicated** across its parts and `shape_index` / `part_index` /
  `n_parts` naming where the row came from. A Point or MultiPoint record has no
  parts and is one row (a MultiPoint's points stay together); a Null shape is
  one row of `NaN`s. A single-part layer decodes 1:1.
- **Shapefile vertex padding** — `geometry` is `float64[index, vertex, xy]`,
  right-padded to the longest part by **repeating the final vertex** — the
  rectangular-storage convention `esm-spec` §8.6.1 pins, which a geometry kernel
  evaluates as the deduplicated ring, so a padded ring has the ring's own area.
  Padding is never `NaN` (only a Null shape's row is). Vertices are the STORED
  ones: an explicitly closed ring keeps its closing vertex (dropping it is the
  kernel's job, §8.6.1) and winding is untouched — no orientation fix, no
  outer/hole classification. `nvert_max` pads to exactly that many slots
  instead, so a DOCUMENT declares the vertex-axis length rather than inheriting
  whatever the file happens to hold; a part longer than it is an error naming
  the row, never a silent truncation.
- **Shapefile bounding box** — `xmin`/`ymin`/`xmax`/`ymax` are the parent
  RECORD's **stored** `Box`, replicated to its parts (a Point record stores no
  box, so its own coordinate is used). On disk, not recomputed: it is the
  shapefile's own broad-phase envelope.
- **dBASE deletion flag** — a `.dbf` row is deleted **iff its flag byte is `*`
  (0x2A)**; every other byte — including the `NUL` that some writers emit —
  means live. A deleted row drops the `.dbf` row AND its `.shp` shape together,
  so the two stay aligned. (`dbase` (Rust) and DBFTables.jl already use this
  rule; pyshp treats any non-space flag as deleted, so the Python reader
  normalizes the flag bytes before decoding.)
- **dBASE column dtypes** — `N`/`F` → **float64** (a blank cell → `NaN`; an
  integer-typed `N` column is NOT kept integral, so the three dbf libraries
  cannot disagree), `L` → **bool**, `D` → the string `YYYYMMDD`, `C`/`M` →
  string. `numeric_columns` parses named `C` columns as float64 instead — the
  `CSVReader`/`FF10Reader` spelling, for a code column (a FIPS `GEOID`) a model
  wants as a number. A `.dbf` column whose name collides with one of the
  reader's own fields is an error, never a silent shadowing.
- **Shapefile CRS and shape type** — the `.prj` WKT (verbatim, when present) and
  the layer's shape-type name are one-element `string` fields on a `meta`
  dimension: `crs_wkt` and `shape_type`. They are FIELDS rather than field
  attributes because the Rust `NativeField` carries no attrs and native-array
  equality compares fields. The CRS is DECLARED, never acted on — reprojection
  is ESD's job (Risk R3).

### Parquet decode notes (columnar table reader)

The `parquet` reader takes an Apache Parquet file as a **flat table**: every
column becomes a rank-1 field over `index`, keyed by its **on-disk column name**,
and a table produces **no coordinates** (like the CSV and Zarr readers). `index`
has length `num_rows`.

Parquet carries an explicit logical type per column, so unlike the CF attribute
sniffing the NetCDF reader must do, the dtype is a **total function of the Arrow
type** — which is what makes cross-language parity cheap here, and why any track
that deviates is wrong rather than merely different:

| Arrow type | dtype |
|---|---|
| `Boolean` | `bool` |
| `Int8`/`Int16`/`Int32`/`UInt8`/`UInt16` | `int32` |
| `Int64`/`UInt32`/`UInt64` | `int64` |
| `Float16`/`Float32`/`Float64` | `float64` |
| `Decimal128`/`Decimal256` | `float64` — unscaled value ÷ 10^scale |
| `Utf8`/`LargeUtf8`/`Utf8View` | `string` |
| `Date32`/`Time32` | `int32` — **raw, undecoded** |
| `Date64`/`Time64`/`Timestamp`/`Duration` | `int64` — **raw, undecoded** |
| `Dictionary(_, V)` | as `V`, expanded to one value per row |
| `Null` | `float64`, every cell `NaN` |

- **The narrow/wide integer split is the NetCDF reader's, verbatim.** It is
  restated rather than re-derived so a MOVES `int32` ID column and a CF `int32`
  time axis cannot drift apart.
- **A `uint64` above `int64::MAX` is an error** naming the column and row, never
  a wraparound into a negative ID — *when the column is read as an integer*.
  Under `float_columns` the document has declared it a float64 measurement,
  there is no integer to wrap into, and it decodes. (The range check belongs to
  the integer coercion, not to the decode; a track that puts it in the decode
  refuses a read the other two perform.)
- **Temporal columns are carried verbatim as their raw stored integer.** The
  Arrow unit (`s`/`ms`/`us`/`ns`) and any timezone are **not** applied and **not**
  reported — the same rule a CF time axis gets, because turning an epoch offset
  into a wall-clock instant is ESS's job (Risk R3). A document needing the unit
  must state it itself.
- **A categorical is expanded.** `Dictionary(K, V)` decodes to one `V` per row;
  the key encoding is a storage detail and never reaches the native array.
- **Nested and binary columns have no rank-1 reading.** `List`/`Struct`/`Map`/
  `Union`/`Binary` naming: requesting one in `variables` is an **error** (the
  document named an array it would not get); unrequested, it is simply not a
  native field. The NetCDF reader applies the same rule to what is genuinely
  unreadable there — a compound/opaque/enum/vlen variable. Its `char` and
  `NC_STRING` variables are **not** in that set: they decode to `string` fields
  (§3, "NetCDF text variables are `string` fields in every track").
- **A zero-row file is TYPED, not absent.** The schema lives in the footer, so
  every column comes back empty with its declared dtype. This matters: most of a
  MOVES fixture's ~770 tables are empty, and a document binding one must still
  see the array it named.

#### Null policy

Nearly every Parquet column is nullable in its schema whether or not it holds a
null — a table exported from a relational database usually marks every column
nullable — so **nullability cannot pick the dtype**. The policy is about values:

- a null in a **float** column (a `Decimal`, a `Null` column, or any column
  forced float by `float_columns` included) becomes **`NaN`**, the same fold CF
  `_FillValue` gets, and `fill_value` stays null;
- a null in an **integer**, **string** or **boolean** column is an **error**
  naming the column and the row. Those types have no NaN, so any default would
  be a real value silently standing in for a missing one — the failure mode that
  surfaces much later as wrong numbers.

Two reader options open that gate, and only when a document declares them:
`null_int` substitutes an integer sentinel **and reports it back in
`fill_value`** (an integer sentinel cannot be NaN, so it survives into the array
exactly as a CF integer fill does); `null_string` substitutes text, which is
then indistinguishable from a real cell holding the same text — which is why the
document has to choose it. A boolean has no such option; declare the column in
`float_columns` if a third state is genuinely meant.

**Where `fill_value` is reported is per-track, and normative.** The three
`NativeField` types are not the same shape: Rust's carries a dedicated
`fill_value: Option<f64>` field, while the Python and Julia ones carry only
`data` / `dims` / `attrs`. A reader therefore reports the surviving `null_int`
sentinel:

| track | location |
|---|---|
| Rust | the `fill_value` field |
| Python | `attrs["fill_value"]` |
| Julia | `attrs["fill_value"]` |

These are the *same* datum in three spellings, not three decisions. A track
whose `NativeField` grows a real `fill_value` slot later should move to it and
this table should change with it; what must never happen is one track reporting
the sentinel where another silently drops it, because that difference is
invisible until a document reads a fill-bearing integer column and gets a real
value where a missing one was meant. A cross-language corpus case comparing
these dumps has to normalise the three spellings to one before diffing.

That normalisation is **done, in two steps**: each provider dumper emits its own
spelling under the single `fill_value` key of `earthsciio/native-dump/v1`, and
[`crosscheck.py`](../conformance/crosscheck.py) compares that key numerically
(Rust's slot is an `f64`, so it dumps `-1.0` where the other two dump `-1`, and
those are one datum) against the oracle and pairwise. The corpus case
`moves-rate-table-parquet` declares `fill_value` on every integer column, so a
track that silently dropped the sentinel now fails the gate.

#### Backend limits that a corpus case must route around

Every track hits the contract above through a third-party decoder, and one of
those decoders cannot reach all of it. These are **known, permitted
divergences** — recorded so a cross-language corpus case does not encode a
difference it cannot fix, and so nobody "fixes" a track into disagreeing with
its own library. A backend defect that produces a *wrong number* rather than a
gap is **not** on this list: that gets compensated for in the track (see the
FLBA decimal below), because a silently wrong value is never a permitted
divergence.

- **Julia cannot open a file that carries a nested column at all.** Parquet2.jl
  builds a `Column` for every schema node when the file is opened, so a file
  with a list/struct/map column errors on open rather than decoding its flat
  columns beside it. Rust and Python decode the flat columns and treat the
  nested one as absent (§3's rule). A corpus case must therefore **not** put a
  nested column in a shared fixture; the rule stays normative and is exercised
  per-track instead.
- **Julia recovers a timestamp only to millisecond resolution.** Parquet2.jl
  decodes every timestamp to a `DateTime` before the reader sees it, so a
  MICROS or NANOS column's raw integer is not recoverable. A shared fixture
  should keep timestamps millisecond-aligned.
- **Julia's decimals arrive as `Dec64`.** Rust and Python both compute
  `f64(unscaled) / 10.0^scale`; Julia cannot see the unscaled integer. For
  `|unscaled| < 2^53` and `scale <= 22` the two agree bit-for-bit, because both
  operands of that division are exact and the quotient is correctly rounded.
  Outside that range they may differ, bounded by `Dec64`'s 16 digits.
- **Julia does not decode a `Float16` column at all.** Parquet2.jl does not
  surface a FLOAT16 column, so it is not even in the file's column list there:
  it is simply not a native field, and naming it in `variables` is an
  "absent from the file" error. Rust and Python both return it as float64. A
  shared fixture must therefore not carry a `Float16` column.
- **A negative `Decimal` in a 7-byte `FIXED_LEN_BYTE_ARRAY`** (pyarrow precision
  15–16) cannot be read in the Julia track *at all*: Parquet2.jl folds the
  two's-complement bytes into an `Int64` **unsigned**, and the resulting 2^56
  overflows a `Dec64`'s 16 digits inside page decode. The reader reports that as
  an error naming the limit. **Narrower** widths (precision ≤ 14) are NOT a
  permitted divergence — they were silently returning the unsigned
  reinterpretation (`-2.50` in a `decimal128(9, 2)` as `42949670.46`), and
  `EarthSciIOParquet2Ext` now repairs the sign exactly, so all three tracks
  agree there. Widths ≥ 8 bytes were always correct.

The first is a genuine gap in coverage, not merely a formatting difference: the
"nested column is simply not a field" rule is normative but is unenforceable
cross-language until Parquet2.jl can open such a file. The `Float16` gap is the
same shape, one column type down.

#### `float_columns`, and floats stored as text

`float_columns` forces the named columns to `float64` whatever their on-disk
type — the Parquet twin of the shapefile reader's `numeric_columns` — and does
two jobs on purpose, because they are the same statement about the source
("this column is a float64 measurement"):

1. an **integer** column that is really a measurement, whose missing cells must
   become `NaN` rather than a sentinel;
2. a column of **fixed-decimal text**. A corpus that needs byte-reproducible
   floats stores them as decimal strings rather than IEEE doubles — the MOVES
   snapshots write `meanBaseRate` as `"261.000000000000"` — and this is how a
   document says so. The text is **trimmed and parsed**; an empty or
   all-whitespace cell is `NaN` (the FF10/shapefile blank→NaN rule); anything
   else unparseable is an **error** naming the column, the row and the text.
   Without the option such a column stays `string`: the reader never guesses
   that text is really a number.

#### Projection pushdown

The loader's `variables` are pushed into the Parquet reader as a projection, so
**only those column chunks are read off disk** — not read-then-discarded. These
tables are wide (a MOVES table runs to dozens of columns) and a document
typically wants three. Empty `variables` reads every column. A requested name
absent from the file is an error listing what is present, never a silently
missing array.

An **empty** `variables` list is the same as none at all — every column — and a
track that reads it as "no columns" disagrees with the other two. (`select` is
separate, and the `parquet` reader honours none: it declares
`supports_selection = false`, so the Provider hands it `Selection::All`
unconditionally and a `select` aimed at it is an error at the call site.)

Row selection is **not** a reader concern: esm-spec §8.9 puts `codes`,
`record_filter`, `select` and `extent` downstream of the decode. Only
`reader_options` reaches this reader.

### NetCDF decode notes (whole-file reader with a decode-time `select`)

The `netcdf` reader fetches ONE blob — a `select` never changes the URL, the
`sha256(resolved_url)` cache key, or the bytes downloaded (that is what
distinguishes it from the store-backed zarr reader, and why `store_backed` stays
`false`). What a selection changes is what gets **materialised**: NCDatasets /
xarray `.isel` / the `netcdf-reader` slice API each read only the intersecting
chunks out of the already-fetched blob. The reader therefore declares
`supports_selection = true` (`registries.md` §2.2).

- **Vocabulary** — the zarr reader's, unchanged and **0-based**:
  `select = {axes: [<axis>, …]}` with each `<axis>` `"all"`, `{indices: [...]}`
  (explicit, possibly non-contiguous, returned **in the order given**) or
  `{slice: [start, stop, step?]}` (half-open, `step` default 1). One spelling
  across every reader and every track; a caller that translates a 1-based
  model-side selection for zarr does not translate differently here.
- **Axis order and base** — the axes are **positional over FILE-order dims** —
  the order `dims` reports, `[time, lev, lat, lon]` for a GEOS-FP A3dyn variable,
  which the Julia track reaches by permuting NCDatasets' reversed arrays back —
  of every array whose **rank equals the axis count** (the zarr rule). "Array"
  means the **decoded field**, so the rank counted is the one the field reports,
  never a variable's on-disk dimension count where the two differ.
- **A text variable is not exempt, and a string LENGTH is not an axis** — the
  induced map is applied by name to `char`/`NC_STRING` fields like any other, so
  a `char label(n)` beside a `float value(n)` loses exactly the cells `value`
  loses, in the same order. But a **consumed** string-length dimension (the one
  the character-array convention above folds away) is not an axis of the decoded
  field: `char label(n, strlen)` is a **rank-1** field, so it answers a 1-axis
  `select` on `n` and never a 2-axis one, and no selector can bind to `strlen`.
  A selection that names a consumed string length is an **error**, never
  silently ignored and never applied — applying it would slice *characters*,
  handing back `"ef"` for `"efgh"`: a wrong string, which is no more permitted
  than a wrong number. (This falls out for free in Python, where xarray's
  decoded `ds.variables` already have the dimension consumed; Rust matches ranks
  over its decoded fields for the same reason.)
- **Applied by dimension NAME** — those axes induce a dimension → selector map,
  which is applied to every other array **and to the coordinate fields**. A
  windowed variable beside a full-length `lon`/`lat` would be a silent trap, and
  the zarr reader never had to answer this because it returns no coords. Two
  same-rank arrays disagreeing about a dimension is an error, not a coin toss.
- **Time is the Provider's axis** — `records_per_sample` and the cadence belong
  to the Provider (§3, "Row selection is not a reader concern"), so a `select`
  whose time axis is anything but `"all"` is **refused**. A dimension is the time
  axis when a same-named coordinate carries CF `"<step> since <ref>"` units, or
  when it is literally named `time`.
- **`records` is how that axis IS narrowed** (Julia track today) — the refusal
  above says the reader may not *choose* records, not that every record must be
  decoded. `records = {dim: "<name>", indices: [<0-based>, …]}` is a second,
  deliberately separate decode option through which the cadence owner states the
  records it has already chosen, in the file's own numbering. The reader is told
  the records, never the cadence: it does no `mod1`, knows nothing of the
  cadence grid or `records_per_sample`, and materialises exactly the named
  records of `dim` (and of `dim`'s own coordinate) **in the order given**.
  Duplicates are legal — the end-of-data bracket `[last, last]` is one — and an
  index outside `0:len-1`, a `dim` the blob lacks, an empty list, or a `select`
  that narrows the same `dim` are all errors rather than a wrapped or widened
  read. `dim_length(reader, path, dim)` (`registries.md` §2.3) is the metadata
  read that lets an out-of-process caller compute those indices; in process
  `indices` may instead be a `len -> indices` callable resolved inside the
  decode's own open, so the cadence owner does not pay a second open per sample
  (a blob with no such `dim` does not call it and narrows nothing). The gate is
  the same as the window's: a record-selected read must be cell-for-cell
  identical to the full read sliced afterwards.
- **An axis count matching no array is an error**, never a silently ignored
  selection.
- **The decode is unchanged by the window** — CF `scale_factor`/`add_offset` in
  float64, `_FillValue` → NaN, the time axis raw with `units`/`calendar`: a
  windowed read must be **cell-for-cell identical to the full read sliced
  afterwards**, which is exactly how the `era5-window-slice` case states it.

Four more rules are pinned here because three independent implementations of the
same slicing math is exactly where a silent divergence hides. They belong to the
**vocabulary**, so they bind the store-backed `zarr` reader identically:

- **Bounds** — every resolved index must satisfy `0 <= i < dim_len`, for a
  `slice` exactly as for an `indices` list. An over-long or negative
  `[start, stop)` is an **error** naming the index and the dimension length: never
  a silent clamp (which returns fewer cells than the caller asked for) and never a
  negative wrap-around (numpy's `-2` is the second-from-last cell; NCDatasets and
  the Rust `netcdf-reader` slice API refuse it, so a clamp cannot be made to agree
  across the three tracks anyway). Bounds are checked against the FILE dimension
  length, before anything is read.
- **A dimension is never dropped** — a single-index axis (`{indices: [i]}` or
  `{slice: [i, i+1]}`) comes back at **length 1**, kept in `dims`. Rank is a
  function of the array, never of the selection, so a track that squeezed would
  diverge in rank from the other two.
- **An axis may select NOTHING** — `{indices: []}`, and any empty half-open range
  (`{slice: [1, 1]}`, or a reversed `{slice: [2, 0]}`) — and that is **legal**: a
  **zero-length axis**, still present in `dims`, giving an array of the same rank
  with no cells. Not an error, and emphatically not a field whose `shape` says
  `0` while its `data` carries the cells of a bounding slab.
- **Order and repetition are the caller's** — an `indices` list is returned in the
  order given, duplicates included (`[2, 0]` reverses the axis; `[1, 1]` repeats
  a cell). A reader that sorts or deduplicates fails the `permuted-order-tile`
  case.

**Precedence** — identical to the zarr reader's, in all three tracks: a per-call
`select` (`materialize(..., select=…)` / `materialize_with_select`) **overrides**
a baked `reader_options.select` / `DataSource::select` for that call ONLY, and is
a projection *peek* — the cadence/file buffers keep the baked projection, so the
next plain read returns it again. A `select` from either source aimed at a reader
that does not answer `supports_selection` is an error raised **before any fetch**,
whether it arrived per-call or baked.

### Zarr decode notes (store-backed reader)

The `zarr` reader is **store-backed**: a Zarr v2 store is not one blob, so the
Provider hands the reader `(cache, base_url, variables, select)` and the reader
fetches each object it needs — `<base_url>/<array>/.zarray`, `…/.zattrs`
(optional), and only the intersecting `…/<chunk_key>` chunk objects — through the
existing content-addressed cache (each object keyed by `sha256(object_url)`; no
byte-range machinery). Decode contract:

- **Compression** — blosc (`cname` lz4/lz4hc/zlib/zstd/blosclz), zlib, zstd,
  gzip, or none. The blosc container is self-describing (codec + shuffle filter +
  multi-block layout are in its 16-byte header), so a c-blosc-backed library
  (numcodecs / Blosc.jl / the `blosc` crate) undoes the shuffle internally.
- **Chunk unpack** — C-order (or F-order per `.zarray` `order`). Zarr v2 edge
  chunks are stored **full-size, fill-padded**; the padding is sliced off by the
  selection's index math (only valid global indices are copied out).
- **Numeric dtype / endianness** — from the `dtype` typestr: `<f4`/`<f8` →
  **float64** (`<` little-endian, `>` byteswapped); integer zarr dtypes keep
  int32/int64.
- **`fill_value` is NOT mapped to NaN** — a deliberate deviation from the NetCDF
  `_FillValue → NaN` rule: in the pinned ISRM store `fill_value == 0.0` is real
  data. `fill_value` fills only the region of a chunk object that is **absent**
  (a cache/transport miss for that chunk).
- **Dims / coords** — dim names from `.zattrs` `_ARRAY_DIMENSIONS` (synthesized
  `dim_0…` if absent); **no coordinate arrays** are produced (like the CSV
  reader). `variables` is **required** (the store cannot be enumerated without a
  consolidated `.zmetadata`).
- **Selection (lazy, orthogonal)** — `select = {axes: [<axis>, …]}` where each
  `<axis>` is `"all"`, `{indices: [...]}` (an explicit, possibly non-contiguous,
  ordered index list), or `{slice: [start, stop, step?]}`. Applied to each
  requested array whose rank matches the axis count (other-rank arrays read
  whole). For each dim, every requested index `g` maps to chunk `g //
  chunk_len`; the chunk keys fetched are the Cartesian product of the per-dim
  chunk-id **sets** — so the reader fetches only the chunks the selection
  intersects, **never** the whole array (the ISRM linchpin). A store-backed
  case's `objects[]` gives every object its own `cache_key`/`content_sha256`, so
  checks 1+2 (key agreement + integrity) are asserted **per object**.

### GeoTIFF decode notes (raster reader)

The `geotiff` reader decodes a single blob into one field per raster band plus
the grid axes. Every track implements it against a different decoder — rasterio
(GDAL) with a pure-Python `tifffile` fallback in Python, `TiffImages.jl` in
Julia, the `tiff` crate in Rust — so the georeferencing arithmetic below is the
whole parity contract. Pinned by the `landfire-raster-tile` case.

- **Band naming** — bands are keyed `Band1…BandN`, 1-based, GDAL's convention
  (the name a loader's `file_variable: "Band1"` refers to). Band arrays are
  **float64** regardless of the on-disk sample type.
- **Axis names** — from `GTModelTypeGeoKey` (1024) in the GeoKeyDirectoryTag:
  geographic (2) gives `lat`/`lon`, projected (1) gives `y`/`x`. Absent or
  unreadable ⇒ treated as geographic. Band dims are `(<ydim>, <xdim>)`.
  A GDAL-backed reader may instead resolve the declared CRS and ask whether it
  is geographic. For a WELL-FORMED file the two agree, because the CS code and
  the model type agree — but a file declaring `GTModelType = projected` while
  giving a geographic code (4326) under `ProjectedCSTypeGeoKey` is
  self-contradictory, and the two approaches then legitimately differ. Such a
  file is out of contract; do not use one as a fixture.
- **Cell-CENTRE axes** — from `ModelPixelScaleTag` (33550) `(sx, sy)` and
  `ModelTiepointTag` (33922) `(i, j, k, x, y, z)`, which anchors raster point
  `(i, j)` to model point `(x, y)`. Model space is **y-up** while raster rows run
  **downward**, so with a north-up raster:

      lon[c] = x + (c - i + 0.5) * sx        lat[r] = y - (r - j + 0.5) * sy

  The `+ 0.5` is load-bearing: the tiepoint anchors a pixel CORNER and the axes
  are cell CENTRES. `sx` and `sy` are independent — cells need not be square.
- **`GDAL_NODATA` → NaN** — tag 42113, an **ASCII** string (conventionally
  NUL-terminated, e.g. `"-9999"`), parsed to a float; cells **equal** to it
  become NaN. A tag that is absent or does not parse means *no sentinel*, and
  nothing is remapped — a raster whose real values include -9999 keeps them. A
  NaN sentinel is ignored (NaN never compares equal). This is the one rule that
  differs from the zarr reader's `fill_value`, which is deliberately NOT mapped.
- **Georef is optional** — a plain TIFF with no scale/tiepoint decodes to bands
  with **no coordinate arrays** rather than an error; the consumer supplies the
  grid.
- **Out of scope, on purpose** — multi-band interleaving is not pinned. The three
  decoders disagree on what a "band" is for multi-sample and multi-page files
  (page 0 only vs pages-as-bands vs a flat sample count), so the corpus commits a
  single-band raster and a multi-band contract is deferred rather than invented.

---

## 4. The runner (five checks — identical in every language)

The cross-language harness (`esio-9nb.9`, **shipped** —
[`conformance/CROSSLANG.md`](../conformance/CROSSLANG.md)) drives each track's
**provider** over the corpus offline, dumps its native arrays
(`earthsciio/native-dump/v1`), and asserts equality across Python / Julia / Rust
(and against the oracle). Each track performs exactly what
[`verify.py`](../conformance/verify.py) does:

1. **cache-key agreement** — `sha256(resolved_url) == case.cache_key`.
2. **manifest integrity** — `sha256(blob) == manifest.sha256_content ==
   case.content_sha256` and `len(blob) == manifest.bytes == case.bytes`.
3. **format/reader decode** — open the blob with `case.format`'s reader, applying
   §3.
4. **native-array equality** — decoded arrays/coords equal `case.expected`
   (tolerances below).
5. **offline-only** — the run opens no socket; it reads only corpus files.

### Tolerances

- **Raw / unpacked numeric reads**: compared **exactly**.
- **CF-decoded (packed) values, and any unit-affected reads**: compared within
  `atol = 1e-6`, `rtol = 1e-9` (libraries differ at the ULP level).
- **Strings**: exact. **NaN/fill masks**: must match element-for-element
  (`null` ↔ NaN).

---

## 5. Running it

```bash
python3 conformance/verify.py     # offline; validates schemas + all cases, exit 1 on any failure
python3 conformance/generate.py   # deterministically regenerates the corpus (needs numpy + netCDF4 + pyarrow)
./conformance/run_conformance.sh  # offline; run all 3 providers + assert cross-language array equality

# narrow the run to named cases (all three dumpers + the comparator honour it):
ESIO_CONFORMANCE_CASES=moves-rate-table-parquet ./conformance/run_conformance.sh
```

`$ESIO_CONFORMANCE_CASES` exists for an environment where one track's backend
for some *other* format is missing or too old — without it a single unrelated
broken case makes the gate unrunnable and hides real divergences elsewhere. CI
leaves it unset, which is every case.

`generate.py` is committed for provenance and is **byte-deterministic**
(NETCDF3_CLASSIC, fixed data, pinned `fetched_at`) — regenerating does not churn
the committed blobs. The **parquet** blob is the one exception: its bytes carry
the writing pyarrow's `created_by` string and `ARROW:schema` metadata, so
regenerating it on a different pyarrow rewrites it (committed with 21.0.0). Conformance consumers read the **committed** blobs, so no
language track needs Python.

---

## 6. Adding a fixture

1. Add a builder to `generate.py` that returns `(blob_bytes, expected, decode)`
   and an `emit_case(...)` call with the registry triple + a realistic
   `resolved_url`.
2. Run `generate.py` (writes blob + manifest + case + updates `cases.json`).
3. Run `verify.py` — schema validation + the five checks must pass.
4. Keep blobs **tiny** (≤ a few KB) and deterministic; commit directly until the
   binary-hosting decision (git-lfs vs `/projects`) lands for larger real slices.

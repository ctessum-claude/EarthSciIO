# Cross-language conformance harness (esio-9nb.9)

The corpus oracle ([`verify.py`](verify.py)) proves the **Python** decode is
correct. This harness proves the stronger, load-bearing claim: the **Python,
Julia, and Rust providers decode the same cached blob to the _same_ native
arrays** — run fully OFFLINE against the committed [`corpus/`](corpus), CI-gated
([`.github/workflows/conformance.yml`](../.github/workflows/conformance.yml)).

It is the executable form of the architecture decision in
[`spec/conformance.md`](../spec/conformance.md): *idiomatic-per-language against a
shared spec, with cross-language correctness proven by equal arrays on a shared
corpus — so no FFI is needed.*

```
conformance/
  dumpers/dump_python.py        # drives earthsciio.Provider          -> native-dump/v1
  dumpers/dump_julia.jl         # drives EarthSciIO.const_provider     -> native-dump/v1
  dumpers/julia_env.jl          # builds the ONE env the Julia drivers run in
  ../rust/examples/conformance_dump.rs  # drives earthsciio::Provider  -> native-dump/v1
  crosscheck.py                 # asserts equality: vs oracle, pairwise, coverage
  run_conformance.sh            # driver: run all 3 dumpers + crosscheck (the gate)
```

## Run it

```bash
./conformance/run_conformance.sh            # all 3 providers + cross-check, offline
./conformance/run_conformance.sh /tmp/dumps # keep the per-track dumps for inspection
```

Needs the three toolchains: Python
(`pip install -e ".[netcdf,shapefile,zarr,geotiff,test]"` — one extra per corpus
format, `zarr` needs Python ≥3.11), Julia (nothing to do by hand — the driver
runs [`dumpers/julia_env.jl`](dumpers/julia_env.jl) first), Rust (`cargo`). Exit
0 ⇔ every decoding track agrees with the oracle and pairwise with the others.

### The Julia environment

EarthSciIO keeps its `zarr` / `shapefile` / `parquet` / `geotiff` decode backends
in weakdep **extensions**, so `--project=julia` cannot import them and
`Pkg.instantiate()` does not install them.
[`dumpers/julia_env.jl`](dumpers/julia_env.jl) prepares one environment —
gitignored under `conformance/.julia-env`, rebuilt only when
`julia/Project.toml` or the Julia version changes — in which EarthSciIO (deved
from this working tree) **and** every weakdep are resolved in a SINGLE pass, and
every Julia driver runs there.

One pass is the whole point. Resolving a weakdep into its own environment and
pushing that onto `LOAD_PATH` is not equivalent: Julia resolves a stacked
package's dependencies by walking `LOAD_PATH` in order, so the *primary*
environment answers first and the two manifests silently mix. On Julia 1.10 that
built `WeakRefStrings` (which needs `Parsers` 2) against the primary
environment's `Parsers` 3 and the harness died in precompilation — a conflict on
a transitive dependency `julia/Project.toml` names no bound for, so no amount of
`[compat]` copying could have fixed it. `Pkg.test` gets this right for free by
resolving `[targets] test` in one pass; this is the harness's equivalent.

## How it works

Each track runs its **Provider** over every corpus case — resolve the case's
`resolved_url` through an offline cache rooted at the corpus, decode with the
format reader, return native arrays — and emits them as a canonical JSON dump
(schema `earthsciio/native-dump/v1`). [`crosscheck.py`](crosscheck.py) then, per
case:

1. **vs the oracle** — each track's arrays equal the corpus `expected` arrays.
2. **pairwise** — every pair of tracks that decoded the case produced equal
   arrays (same variable/coord set, dtype, dims, shape, values).
3. **coverage** — a track that registers a reader for a case's format **must**
   have decoded it (a silently-dropped case fails); a track with no reader for
   that format may skip, and the gap is logged, never hidden.

Global gates: every case is decoded by ≥1 track, and **≥1 case is decoded by all
three** (so three-way equality is actually demonstrated, not just pairwise on
disjoint subsets).

### The dump format — `earthsciio/native-dump/v1`

```jsonc
{
  "schema": "earthsciio/native-dump/v1",
  "language": "python",                    // | "julia" | "rust"
  "provider": "earthsciio.Provider",
  "readers": ["csv", "netcdf"],            // active format names this track registers
  "cases": {
    "era5-grid-sub-tile": {
      "format": "netcdf",
      "status": "decoded",
      "variables": {
        "t2m": { "dtype": "float64", "dims": ["time","latitude","longitude"],
                 "shape": [2,3,3], "data": [282.5, /* … */, null] },
        "pollutantID": { "dtype": "int32", "dims": ["index"], "shape": [6],
                         "fill_value": -1, "data": [2,3,-1,110,100,31] }
      },
      "coords": {
        "time": { "dtype": "int32", "dims": ["time"], "shape": [2], "data": [0,1],
                  "units": "hours since 2018-11-08 00:00:00", "calendar": "gregorian" }
      }
    },
    "openaq-points-slice": {               // a track with no reader for the format
      "format": "csv", "status": "skipped",
      "reason": "no active reader registered for format 'csv' in the Rust track"
    }
  }
}
```

`data` is the field flattened **row-major (C order)** per `shape` — Julia/Rust
permute their column-major / pre-flattened storage to this one order, so the
arrays line up element-for-element. A masked / `_FillValue` cell is `null`
(== NaN). Strings are verbatim. A case whose format the track can't read is
`status:"skipped"` with a `reason` — **explicit, never omitted** — so a real
coverage gap is distinguishable from a dumper bug.

`fill_value` is present only when a **surviving** fill/missing sentinel exists
(today: a `parquet` `null_int` on an integer column). It is the one place the
dump deliberately **normalises across tracks**: `spec/conformance.md` §3 pins
Rust reporting it in `NativeField::fill_value` and Python/Julia in
`attrs["fill_value"]` — *the same datum in three spellings, not three
decisions* — so each dumper reads its own spelling and writes this one key, and
`crosscheck.py` compares it numerically (Rust's `f64` slot dumps `-1.0` where
the other two dump `-1`). Without it, a track that silently DROPPED a declared
sentinel would cross-check clean, and the loss would surface much later as a
real value standing where a missing one was meant.

## Narrowing a run

`$ESIO_CONFORMANCE_CASES` is a comma-separated case-id list honoured identically
by the three dumpers **and** `crosscheck.py`, so a filtered run is still a
complete cross-check of the cases it names:

```bash
ESIO_CONFORMANCE_CASES=moves-rate-table-parquet ./conformance/run_conformance.sh
```

It exists for an environment where one track's backend for some *other* format
is missing or too old — without it a single unrelated broken case makes the gate
unrunnable and hides real divergences elsewhere. CI leaves it unset, which is
every case.

## Documented tolerance (`spec/conformance.md` §4)

| read kind | comparison |
|---|---|
| CF-decoded (packed) values, any unit-affected read | `|a−b| ≤ atol + rtol·|b|`, **atol = 1e-6**, **rtol = 1e-9** |
| raw / unpacked **integer** reads (e.g. a CF time axis) | **exact** |
| **strings** (CSV/JSON text columns) | **exact** |
| fill / missing (`null` ↔ NaN) | mask must match **element-for-element** |

The float tolerance exists because xarray, NCDatasets, and netcdf-rs differ at the
ULP level when applying `scale_factor`/`add_offset`; everything else is bit-exact.
This is the **only** sanctioned per-language numeric slack, and it is identical to
the oracle [`verify.py`](verify.py).

## Coverage (today)

| case | format | python | julia | rust |
|---|---|---|---|---|
| `era5-grid-sub-tile` | netcdf | ✅ | ✅ | ✅ (three-way) |
| `openaq-points-slice` | csv | ✅ | ✅ | ⊘ no reader |
| `ff10-point-slice` | ff10 | ✅ | ✅ | ✅ (three-way) |
| `ff10-zip-egu-glob` | ff10 | ✅ | ✅ | ✅ (three-way) |
| `isrm-zarr-tile` | zarr | ✅ | ✅ | ✅ (three-way) |
| `permuted-order-tile` | zarr | ✅ | ✅ | ✅ (three-way) |
| `shapefile-polygon-zip` | shapefile | ✅ | ✅ | ✅ (three-way) |
| `moves-rate-table-parquet` | parquet | ✅ | ✅ | ✅ (three-way) |

The parquet case is the columnar one: it is where the reader OPTIONS
(`float_columns`, `null_int`, `null_string`) and the `variables` projection are
cross-checked, and where the `fill_value` normalisation above is exercised.

The Rust track has no `csv` reader yet, so the CSV case is a logged Rust skip
(mirrors [`rust/tests/conformance_decode.rs`](../rust/tests/conformance_decode.rs)),
cross-checked Python↔Julia. Every other case carries the full three-way proof.
When the Rust `csv` reader lands, the dumper reports `csv` in its `readers` and the
coverage gate **automatically requires** it to decode the CSV case — no edit here.

### The `permuted-order-tile` selection gate (Phase 1)

`permuted-order-tile` is a store-backed zarr case that pins **ordered, lazy
orthogonal selection** across all three tracks. Its `select` is
`layer=[0], source=[24,2,9,6] (0-based, NON-CONTIGUOUS and PERMUTED — not sorted),
receptor=all` over an `sr [3,50,4]` array chunked `[1,10,4]`:

* **Laziness** — sources `2,9,6` fall in source-chunk 0 and `24` in chunk 2, so
  only `sr/0.0.0` and `sr/0.2.0` are fetched (never chunk 1, never layers 1/2).
  Proven per-track by [`tests/test_zarr_reader.py`](../tests/test_zarr_reader.py)
  (a `CountingStore` asserting the exact fetched-object set) and
  [`rust/tests/zarr_read_store.rs`](../rust/tests/zarr_read_store.rs) (a poison store).
* **Order preservation** — every track returns the source axis as `[24,2,9,6]` in
  that exact order (a reader that sorted the index list would return `[2,6,9,24]`);
  `crosscheck.py` asserts the three dumps are **byte-identical** here.

The case is data-driven from [`corpus/cases.json`](corpus/cases.json), so
`./run_conformance.sh` runs it in all three tracks and cross-checks it with no
per-case edit — the executable Phase-1 3-way selection acceptance gate.

## Adding a fixture or a reader

The harness is data-driven from [`corpus/cases.json`](corpus/cases.json) and each
track's registered readers — there is nothing per-case to edit:

- **New corpus case** (via [`generate.py`](generate.py)) → every track that has a
  reader for its format decodes it and is cross-checked automatically.
- **New reader in a track** → it appears in that track's dump `readers`, and the
  coverage gate begins requiring it for that format's cases.

---

# Cross-language WRITE conformance (streaming-output-sinks Wave 5)

The harness above proves the three **readers** decode a shared corpus to equal
arrays. This section proves the symmetric claim for the write boundary: the
**Python, Julia, and Rust Zarr v3 sharded WRITERS emit stores that decode to the
same arrays and carry the same structural / CF metadata** — for one shared,
language-neutral input spec.

```
conformance/
  write_spec.json                # shared input spec, `diagnostic` profile
  write_spec_wasm.json           # shared input spec, `wasm` profile (same data)
  gen_write_spec.py              # deterministic regenerator for both spec variants
  dumpers/codec_weakdeps.jl      # loads the Blosc + CodecZstd weakdep extensions
  dumpers/julia_env.jl           # builds the ONE env the Julia drivers run in
  dumpers/write_python.py        # drives earthsciio.backends.zarr_write.ZarrWriter -> store
  dumpers/write_julia.jl         # drives EarthSciIO.ZarrWriter (reference)         -> store
  ../rust/examples/conformance_write.rs  # drives earthsciio::write_zarr_v3         -> store
  dumpers/read_python.py         # reads a store with the Python ZarrReader -> write-native-dump/v1
  dumpers/read_julia.jl          # reads a store with the Julia ZarrReader  -> write-native-dump/v1
  crosscheck_write.py            # asserts: vs spec oracle, pairwise, + store metadata
  run_write_conformance.sh       # driver: run every writer + every reader + crosscheck (the gate)
```

## Run it

```bash
# Python needs >=3.11 with the `zarr` extra (zarr>=3); point PYTHON at it:
PYTHON=/path/to/py311/bin/python conformance/run_write_conformance.sh
conformance/run_write_conformance.sh /tmp/wstores   # keep the stores + dumps
PROFILES=wasm conformance/run_write_conformance.sh  # just one codec profile
```

Each track whose toolchain is present writes a store; each present reader decodes
every produced store; the comparator gates. A missing toolchain is a logged skip,
never a silent pass. The run is **parameterized over codec profiles**
(`PROFILES`, default `diagnostic wasm`): each profile is an independent
write → read → compare round over its own spec variant.

## The shared input spec — `earthsciio/write-conformance-spec/v1`

`write_spec.json` is a tiny deterministic dataset with the values written out in
full (no language re-derives them from a formula), so it is simultaneously the
**writer input** and the **decode oracle**:

* dims `time` (growable) + `lat`(3) + `lon`(4); `lon` split into **2 inner chunks
  per shard**, `time` sharded **2 records/shard** → 4 records = 2 committed shards
  + a streaming time-axis resize;
* CF coordinate variables `lat`/`lon` (`units`/`standard_name`/`axis`) and a
  growable `time` coordinate with CF time attrs;
* two `float64` variables `temperature`/`pressure` over `(time, lat, lon)` carrying
  CF `units`/`standard_name`/`coordinates`; group attrs `title`/`Conventions`;
* a **codec profile** (below), sharding codec (`sharding_indexed`, index
  `[bytes(little), crc32c]`, `index_location:"end"`).

Each record's `vars[name]` is a `[lat][lon]` list (row-major); the full variable
array is `[time][lat][lon]`.

## Codec profiles — and why `wasm` exists

The Zarr v3 codec pipeline is `array -> bytes(endian) -> <compressor>`. Every
profile shares the same dataset, dim order, chunk/shard grid, `dimension_names`,
CF coordinate variables and attrs, `fill_value` 0.0 (never NaN), and output
manifest; **only the inner (per-chunk) compressor differs**.

| profile | spec file | inner pipeline | notes |
|---|---|---|---|
| `diagnostic` | `write_spec.json` | `[bytes(little), blosc(zstd, clevel=5, shuffle)]` | default streaming output |
| `checkpoint` | *(not in the harness)* | `[bytes(little), blosc(zstd, clevel=7, shuffle)]` | lossless durability |
| `wasm` | `write_spec_wasm.json` | `[bytes(little), zstd(level=5, checksum=false)]` | **no Blosc** — browser/wasm-loadable |

**Why `wasm`.** The Blosc *container* — not zstd itself — is what a
WebAssembly/browser Zarr reader cannot decode. The `zarrs` crate's Blosc support
comes from `blosc-src`, whose vendored C sources do **not** build for the
`wasm32-unknown-unknown` target, whereas the standard Zarr v3 `zstd` codec is pure
Rust there. So the `wasm` profile swaps the inner compressor for the plain v3
`zstd` codec and keeps everything else identical. In particular the
`sharding_indexed` outer codec and the crc32c shard index are **retained** — both
are pure Rust and wasm-safe, so sharding's object-count benefits survive.

Blosc's byte-shuffle filter is deliberately **not** replaced: shuffle is not
available as a standalone standard v3 codec, so the chain is plainly
`bytes + zstd`. This costs nothing in practice for smooth model output — zstd
alone compresses it at least as well as blosc-shuffle at the same level — and it
is lossless either way, so decoded values are bit-identical to `diagnostic`.

Selection mirrors each language's existing surface, and an unknown name is
rejected:

| language | profile type | selection |
|---|---|---|
| Julia  | `ZstdProfile <: CodecProfile`, `ZSTD_WASM` | `OutputSchema(; profile = :wasm)` |
| Python | `ZstdProfile` dataclass, `ZSTD_WASM` | `OutputSchema(profile="wasm")` |
| Rust   | `ZstdProfile` / `CodecProfile::Zstd`, `ZSTD_WASM` | `OutputSchema { profile: "wasm".into(), .. }` |

The Julia track's plain-zstd encode/decode rides a weakdep extension
(`CodecZstd` → `EarthSciIOZstdExt`) exactly as Blosc does (`Blosc` →
`EarthSciIOBloscExt`), so a base install stays light.

## How it works — tolerance, not bytes (RFC §16.6)

3rd-party codec builds (Julia Blosc.jl / Python numcodecs / Rust zarrs) legitimately
produce **different compressed bytes** — verified: the three writers' first
`temperature` shard objects have three distinct sha256s. So conformance is on
**decoded arrays within tolerance** + structural metadata, never byte identity.
Two independent checks in [`crosscheck_write.py`](crosscheck_write.py):

1. **Decoded-array agreement.** Each store is read back to
   `earthsciio/write-native-dump/v1` (one dump per (writer, reader) pair) and
   checked (a) against the spec **oracle** and (b) **pairwise** against every other
   dump — same dtype, dims, dim order, shape, and values within
   `|a-b| <= atol + rtol·|b|` (`atol=1e-9`, `rtol=1e-6` for float64). A fill/`null`
   cell matches only `null`; `fill_value` (0.0) is **not** mapped to NaN.
2. **Structural / CF-metadata agreement.** Each store's `zarr.json` objects are read
   **directly** (language-neutral JSON, no reader) and compared pairwise:
   `data_type`, `shape`, `dimension_names`, the shard grid, the sharding inner
   `chunk_shape`, the **ordered inner codec chain** plus the Blosc *and*
   plain-`zstd` params, `fill_value`, and the key CF attributes
   (`units`/`standard_name`/`axis`/`coordinates`/`calendar`) + group attrs. This
   pins dim order, shape, coord metadata and CF attrs even for a store no local
   reader could open — and, per profile, proves every track emitted the *same*
   compressor (a track that silently fell back from `zstd` to `blosc` fails here
   even though the decoded values would still match).

The readback drivers run the **real** store-backed readers (the ones the read
harness proves conformant) over a freshly-written local store: `read_python.py`
via a trivial `<base>/<key>`→file cache shim, `read_julia.jl` via a plain online
`Cache` pointed at a `file://` base URL (a local copy, no network).

## Coverage (executed in-sandbox, 2026-07)

Run for **both** codec profiles (`diagnostic` and `wasm`), identically:

| writer store | python reader | julia reader | structural (zarr.json) |
|---|---|---|---|
| python (`ZarrWriter`)       | ✅ | ✅ | ✅ vs julia, vs rust |
| julia (`ZarrWriter`, ref)   | ✅ | ✅ | ✅ vs python, vs rust |
| rust (`write_zarr_v3`)      | ✅ | ✅ | ✅ vs python, vs julia |

Per profile: all **6** (writer, reader) readbacks agree with the oracle (5 fields
each); all **15** pairwise decoded comparisons agree within tolerance; all **3**
stores agree structurally on every array plus group attrs (18 array + 3 group
pairwise comparisons) — a full three-way write-conformance proof, twice over.

Because both Blosc-zstd and plain zstd are lossless, the decoded float64 values
are in fact bit-exact across languages *and* across profiles (the tolerance is the
sanctioned envelope, not a fudge). The three tracks emit **byte-identical codec
metadata** per profile — verified on disk:

```
diagnostic  python  ['bytes','blosc'] cname=zstd clevel=5 shuffle=shuffle typesize=8
diagnostic  julia   ['bytes','blosc'] cname=zstd clevel=5 shuffle=shuffle typesize=8
diagnostic  rust    ['bytes','blosc'] cname=zstd clevel=5 shuffle=shuffle typesize=8
wasm        python  ['bytes','zstd']  level=5 checksum=False
wasm        julia   ['bytes','zstd']  level=5 checksum=False
wasm        rust    ['bytes','zstd']  level=5 checksum=False
```

### Environment notes

* **Python** requires Python ≥3.11 + `pip install -e ".[zarr]"` (zarr≥3;
  zarr 3.2.1 known-good). The repo's system Python 3.9 cannot run the writer —
  point `PYTHON` at a 3.11+ interpreter.
* **Rust** builds `conformance_write` against the existing crate deps; it needs the
  sibling `netcdf-reader` path dep present (the same dep the whole crate needs). If
  that fork is absent, `cargo` skips with a logged reason and the gate runs on the
  tracks that built. The `wasm` profile needs **no new crate dependency**: `zarrs`'
  default feature set already includes the pure-Rust-friendly `zstd` codec.
* **Julia** runs in the environment `dumpers/julia_env.jl` prepares (see *The
  Julia environment* above), where EarthSciIO and all of its weakdeps are
  resolved in one pass. `dumpers/codec_weakdeps.jl` then just imports the two the
  writer/reader need — `Blosc` (→ `EarthSciIOBloscExt`) for the Blosc profiles
  and `CodecZstd` (→ `EarthSciIOZstdExt`) for `wasm` — and warns, naming the
  environment, if one is missing.

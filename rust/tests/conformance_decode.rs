//! Cross-language decode parity (conformance checks 3 & 4): point the `format`
//! registry's reader at each committed corpus blob — a `$EARTHSCIDATADIR`
//! populated by the **Python** generator — decode it fully offline, and assert
//! the native arrays equal the case's `expected` arrays.
//!
//! This is the half of conformance that component (a)'s `conformance_reuse.rs`
//! explicitly defers to (b): "Decoding the blob into native arrays (checks 3–4)
//! is component (b)." The corpus `expected` arrays are the cross-language oracle
//! — equality here is what "matching the Python and Julia tracks" means.
//!
//! Cases whose format has no Rust reader yet (e.g. `csv`) are skipped with a
//! note, so the test runs every decodable case today and picks up new readers
//! (csv/geotiff/zarr) automatically as they register — no edit here.

// Native-only: this tier drives the cache, the transports and the format
// readers, none of which exist on wasm32 (see the crate's module docs).
#![cfg(not(target_arch = "wasm32"))]

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use earthsciio::{ArrayData, AxisSelect, Coord, DType, Ff10Reader};
use earthsciio::{Cache, FetchRequest, FormatRegistry, NativeField, Selection};
use serde_json::Value;

/// Build a configured [`Ff10Reader`] from an ff10 case's `decode` block —
/// `member` (singular), `members`/`member_glob` (multi-member; sorted-name
/// concat), `skip_header_row` (drop one asserted `country_cd` header line per
/// member). `None` when the case pins none of them (the default reader already
/// decodes the bare blob). Mirrors `rust/examples/conformance_dump.rs`.
fn ff10_reader_from_decode(case: &Value) -> Option<Ff10Reader> {
    let dec = case.get("decode")?;
    let mut reader = Ff10Reader::new();
    let mut configured = false;
    if let Some(m) = dec.get("member").and_then(Value::as_str) {
        reader = reader.member(m);
        configured = true;
    }
    if let Some(ms) = dec.get("members").and_then(Value::as_array) {
        reader = reader.members(ms.iter().filter_map(Value::as_str));
        configured = true;
    }
    if let Some(g) = dec.get("member_glob").and_then(Value::as_str) {
        reader = reader.member_glob(g);
        configured = true;
    }
    if dec.get("skip_header_row").and_then(Value::as_bool).unwrap_or(false) {
        reader = reader.skip_header_row(true);
        configured = true;
    }
    configured.then_some(reader)
}

/// A case's `decode` block narrowed to the named keys, as a loader would declare
/// them in `reader_options` — the input to [`Reader::configured`].
fn declared_options(case: &Value, keys: &[&str]) -> serde_json::Map<String, Value> {
    let mut options = serde_json::Map::new();
    if let Some(dec) = case.get("decode") {
        for k in keys {
            match dec.get(*k) {
                Some(v) if !v.is_null() => {
                    options.insert((*k).to_string(), v.clone());
                }
                _ => {}
            }
        }
    }
    options
}

/// Parse a case's `select.axes` into a `Selection::Orthogonal` (store-backed
/// zarr cases); absent ⇒ `Selection::All`.
fn parse_selection(case: &Value) -> Selection {
    match case.get("select").and_then(|s| s.get("axes")).and_then(Value::as_array) {
        Some(arr) => Selection::Orthogonal(arr.iter().map(parse_axis).collect()),
        None => Selection::All,
    }
}

fn parse_axis(v: &Value) -> AxisSelect {
    if v.as_str() == Some("all") {
        return AxisSelect::All;
    }
    if let Some(idx) = v.get("indices").and_then(Value::as_array) {
        return AxisSelect::Indices(idx.iter().map(|x| x.as_u64().unwrap() as usize).collect());
    }
    if let Some(s) = v.get("slice").and_then(Value::as_array) {
        let g = |i: usize, d: u64| s.get(i).and_then(Value::as_u64).unwrap_or(d) as usize;
        return AxisSelect::Range { start: g(0, 0), stop: g(1, 0), step: g(2, 1) };
    }
    panic!("unrecognized axis selector: {v}")
}

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../conformance/corpus")
}

/// Compared exactly for raw/unpacked reads; CF-decoded (packed) values differ at
/// the ULP level across libraries, so within `atol` (conformance.md §4).
const ATOL: f64 = 1e-6;

#[test]
fn decodes_every_corpus_case_to_match_expected() {
    let corpus = corpus_dir();
    let cache = Arc::new(
        Cache::builder()
            .data_dir(corpus.join("cache"))
            .offline(true)
            .verify_on_read(true)
            .build()
            .expect("offline cache over the corpus"),
    );

    let formats = FormatRegistry::with_builtins();

    let index: Value =
        serde_json::from_slice(&fs::read(corpus.join("cases.json")).unwrap()).unwrap();
    let cases = index["cases"].as_array().expect("cases array");
    assert!(!cases.is_empty(), "corpus must ship at least one case");

    let mut decoded_any = false;
    for entry in cases {
        let case: Value = serde_json::from_slice(
            &fs::read(corpus.join(entry["file"].as_str().unwrap())).unwrap(),
        )
        .unwrap();
        let id = case["id"].as_str().unwrap();
        let format = case["format"].as_str().unwrap();

        let Some(reader) = formats.get(format) else {
            eprintln!("skip case {id}: no Rust reader for format '{format}' yet");
            continue;
        };
        decoded_any = true;

        // Store-backed (zarr): a Zarr store is many objects, not one blob — the
        // reader is handed (cache, base_url, variables, select) and fetches only
        // the intersecting chunk objects itself. Whole-file readers take the
        // single-blob path.
        let ds = if reader.store_backed() {
            let vars: Vec<String> = case["variables"]
                .as_array()
                .expect("store-backed case has a variables array")
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            let sel = parse_selection(&case);
            reader
                .read_store(cache.clone(), case["resolved_url"].as_str().unwrap(), &vars, &sel)
                .unwrap_or_else(|e| panic!("store decode failed for {id}: {e}"))
        } else {
            // Resolve the blob offline (reuses the Python-cached bytes), then decode.
            // An ff10 case whose decode block pins zip member selection / header
            // handling gets a reader CONFIGURED at construction (the Reader trait
            // takes no kwargs) — the same `with_formats`-style seam the dumper uses.
            let blob = cache
                .fetch(&FetchRequest::new(case["resolved_url"].as_str().unwrap()))
                .unwrap_or_else(|e| panic!("offline resolve failed for {id}: {e}"));
            let reader: Arc<dyn earthsciio::Reader> = if format == "ff10" {
                match ff10_reader_from_decode(&case) {
                    Some(configured) => Arc::new(configured),
                    None => reader,
                }
            } else if format == "shapefile" {
                // The shapefile case pins its zip member + the code column the
                // model wants numeric; both reach the reader the way a DOCUMENT
                // delivers them — `Reader::configured`, not a bespoke builder.
                let options = declared_options(&case, &["member", "numeric_columns"]);
                match reader.configured(&options).expect("configured shapefile reader") {
                    Some(configured) => configured,
                    None => reader,
                }
            } else if format == "parquet" {
                // The parquet case pins `float_columns` (a decimal-TEXT rate and
                // an integer measurement) and the two null gates, the same way.
                let options =
                    declared_options(&case, &["float_columns", "null_int", "null_string"]);
                match reader.configured(&options).expect("configured parquet reader") {
                    Some(configured) => configured,
                    None => reader,
                }
            } else {
                reader
            };
            // A single-blob case may still carry `variables` — for parquet that
            // is the PROJECTION, pushed into the decode so only those column
            // chunks are read (and so the decoded field set is exactly the
            // expected one). Every other whole-file case reads everything.
            let vars: Vec<String> = case
                .get("variables")
                .and_then(Value::as_array)
                .map(|a| a.iter().map(|v| v.as_str().unwrap().to_string()).collect())
                .unwrap_or_default();
            // A whole-file reader that honours a `select` (netcdf) gets the
            // case's orthogonal selection too: same blob, same cache key, only
            // the requested hyperslab materialised. `parse_selection` yields
            // `Selection::All` for a case that pins no `axes`.
            let sel = if reader.supports_selection() {
                parse_selection(&case)
            } else {
                Selection::All
            };
            reader
                .read_native(&blob.path, &vars, &sel)
                .unwrap_or_else(|e| panic!("decode failed for {id}: {e}"))
        };

        // Check 4a: data variables.
        let exp_vars = case["expected"]["variables"].as_object().unwrap();
        assert_eq!(
            ds.variables.len(),
            exp_vars.len(),
            "{id}: variable count (got {:?})",
            ds.variables.keys().collect::<Vec<_>>()
        );
        for (name, exp) in exp_vars {
            let got = ds
                .variables
                .get(name)
                .unwrap_or_else(|| panic!("{id}: missing variable {name}"));
            compare_field(id, name, got, exp);
        }

        // Check 4b: coordinates (the corpus pins dtype + values, not dims/shape).
        let exp_coords = case["expected"]["coords"].as_object().unwrap();
        for (name, exp) in exp_coords {
            let got = ds
                .coords
                .get(name)
                .unwrap_or_else(|| panic!("{id}: missing coord {name}"));
            compare_coord(id, name, got, exp);
        }
    }
    assert!(
        decoded_any,
        "no corpus case was decodable — expected ≥1 (netcdf)"
    );
}

/// The same zip case, decoded through a **`Provider` built from a loader that
/// DECLARES its decode options** — no caller-configured reader, no custom
/// registry. This is the path an `.esm` data loader takes (EarthSciAST's
/// `providers_from_document` builds exactly this `DataSource`), and it must
/// land on the corpus expectation the hand-configured reader above produces.
#[test]
fn declared_reader_options_decode_the_zip_case_through_the_provider() {
    let corpus = corpus_dir();
    let case: Value = serde_json::from_slice(
        &fs::read(corpus.join("cases/ff10-zip-egu-glob.json")).unwrap(),
    )
    .unwrap();
    let cache = Arc::new(
        Cache::builder()
            .data_dir(corpus.join("cache"))
            .offline(true)
            .verify_on_read(true)
            .build()
            .expect("offline cache over the corpus"),
    );

    // The case's own decode block, verbatim, as the loader's reader_options.
    let dec = case["decode"].as_object().unwrap();
    let mut options = serde_json::Map::new();
    for k in ["kind", "member_glob", "skip_header_row"] {
        if !dec[k].is_null() {
            options.insert(k.to_string(), dec[k].clone());
        }
    }
    assert_eq!(options.len(), 3, "the zip case pins kind + glob + header row");

    let loader = earthsciio::DataSource::new(
        case["loader"].as_str().unwrap(),
        "ff10",
        case["resolved_url"].as_str().unwrap(),
    )
    .variables(["POLID".to_string(), "ANN_VALUE".to_string()])
    .reader_options(options);
    let mut provider = earthsciio::Provider::new(loader, cache, None).expect("provider");
    let fields = provider.materialize().expect("declared-options decode");

    let exp = &case["expected"]["variables"];
    compare_field(
        "ff10-zip-egu-glob(declared)",
        "POLID",
        &fields["POLID"],
        &exp["POLID"],
    );
    compare_field(
        "ff10-zip-egu-glob(declared)",
        "ANN_VALUE",
        &fields["ANN_VALUE"],
        &exp["ANN_VALUE"],
    );

    // Without the declared options the same loader hits the header line as a
    // data row — i.e. the options are load-bearing, not decoration.
    let bare = earthsciio::DataSource::new("nei2016", "ff10", case["resolved_url"].as_str().unwrap());
    let mut bare = earthsciio::Provider::new(
        bare,
        Arc::new(
            Cache::builder()
                .data_dir(corpus.join("cache"))
                .offline(true)
                .build()
                .unwrap(),
        ),
        None,
    )
    .expect("provider");
    assert!(bare.materialize().is_err(), "no options ⇒ no member selection");
}

fn compare_field(id: &str, name: &str, got: &NativeField, exp: &Value) {
    assert_eq!(
        dtype_str(got.dtype),
        exp["dtype"].as_str().unwrap(),
        "{id}/{name}: dtype"
    );
    let exp_dims: Vec<String> = exp["dims"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(got.dims, exp_dims, "{id}/{name}: dims");
    let exp_shape: Vec<usize> = exp["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(got.shape, exp_shape, "{id}/{name}: shape");
    compare_values(id, name, &got.data, &exp["data"]);
}

fn compare_coord(id: &str, name: &str, got: &Coord, exp: &Value) {
    assert_eq!(
        dtype_str(got.field.dtype),
        exp["dtype"].as_str().unwrap(),
        "{id}/{name}: coord dtype"
    );
    if let Some(units) = exp.get("units").and_then(Value::as_str) {
        assert_eq!(
            got.units.as_deref(),
            Some(units),
            "{id}/{name}: coord units"
        );
    }
    if let Some(cal) = exp.get("calendar").and_then(Value::as_str) {
        assert_eq!(
            got.calendar.as_deref(),
            Some(cal),
            "{id}/{name}: coord calendar"
        );
    }
    compare_values(id, name, &got.field.data, &exp["data"]);
}

/// Compare a decoded array against the corpus's nested `data`: element count and
/// value-by-value (null ↔ NaN; numbers within `ATOL`; strings exact).
fn compare_values(id: &str, name: &str, got: &ArrayData, exp: &Value) {
    match got {
        ArrayData::Str(v) => {
            let expected = flatten_str(exp);
            assert_eq!(v.len(), expected.len(), "{id}/{name}: string len");
            assert_eq!(v, &expected, "{id}/{name}: string values");
        }
        _ => {
            let got_f = to_opt_f64(got);
            let expected = flatten_f64(exp);
            assert_eq!(got_f.len(), expected.len(), "{id}/{name}: element count");
            for (i, (g, e)) in got_f.iter().zip(expected.iter()).enumerate() {
                match (g, e) {
                    (None, None) => {}
                    (Some(a), Some(b)) => assert!(
                        (a - b).abs() <= ATOL,
                        "{id}/{name}[{i}]: {a} != {b} (atol {ATOL})"
                    ),
                    _ => panic!("{id}/{name}[{i}]: fill mask mismatch (got {g:?}, expected {e:?})"),
                }
            }
        }
    }
}

fn dtype_str(d: DType) -> &'static str {
    match d {
        DType::Float64 => "float64",
        DType::Int64 => "int64",
        DType::Int32 => "int32",
        DType::Str => "string",
        DType::Bool => "bool",
    }
}

fn to_opt_f64(data: &ArrayData) -> Vec<Option<f64>> {
    match data {
        ArrayData::F64(v) => v
            .iter()
            .map(|&x| if x.is_nan() { None } else { Some(x) })
            .collect(),
        ArrayData::I64(v) => v.iter().map(|&x| Some(x as f64)).collect(),
        ArrayData::I32(v) => v.iter().map(|&x| Some(x as f64)).collect(),
        ArrayData::Bool(v) => v.iter().map(|&x| Some(x as i64 as f64)).collect(),
        ArrayData::Str(_) => panic!("string array compared as numeric"),
    }
}

/// Flatten a nested JSON array of numbers/null into row-major `Option<f64>`.
fn flatten_f64(v: &Value) -> Vec<Option<f64>> {
    let mut out = Vec::new();
    fn rec(v: &Value, out: &mut Vec<Option<f64>>) {
        match v {
            Value::Array(a) => a.iter().for_each(|x| rec(x, out)),
            Value::Null => out.push(None),
            Value::Number(n) => out.push(Some(n.as_f64().unwrap())),
            // A `bool` field compares numerically (false=0, true=1) — the
            // native-field schema's fifth dtype, from a `.dbf` `L` column.
            Value::Bool(b) => out.push(Some(if *b { 1.0 } else { 0.0 })),
            other => panic!("unexpected value in numeric data: {other}"),
        }
    }
    rec(v, &mut out);
    out
}

/// Flatten a nested JSON array of strings into row-major order.
fn flatten_str(v: &Value) -> Vec<String> {
    let mut out = Vec::new();
    fn rec(v: &Value, out: &mut Vec<String>) {
        match v {
            Value::Array(a) => a.iter().for_each(|x| rec(x, out)),
            Value::String(s) => out.push(s.clone()),
            other => panic!("unexpected value in string data: {other}"),
        }
    }
    rec(v, &mut out);
    out
}

/// A `variables` projection that names something the blob does not hold is an
/// ERROR listing what is present — not a quietly missing array.
///
/// The other two tracks (and this track's `parquet`/`shapefile` readers) have
/// always said so; the `netcdf` reader used to filter with a set membership test
/// and simply return fewer fields, so a typo'd `file_variable` surfaced far from
/// its cause as a `KeyError` in the consumer.
#[test]
fn netcdf_projection_rejects_an_absent_variable_and_keeps_coords() {
    let corpus = corpus_dir();
    let case: Value =
        serde_json::from_slice(&fs::read(corpus.join("cases/era5-grid-sub-tile.json")).unwrap())
            .unwrap();
    let blob = corpus.join(case["blob_path"].as_str().unwrap());
    let reader = FormatRegistry::with_builtins().get("netcdf").unwrap();

    // The projection itself: only `t2m`, with every coordinate still returned.
    let one = reader
        .read_native(&blob, &["t2m".to_string()], &Selection::All)
        .expect("projected decode");
    let mut names: Vec<&str> = one.variables.keys().map(String::as_str).collect();
    names.sort_unstable();
    assert_eq!(names, ["t2m"]);
    assert!(one.coords.contains_key("latitude"));
    assert!(one.coords.contains_key("longitude"));
    assert!(one.coords.contains_key("time"));

    let err = reader
        .read_native(&blob, &["t2m".to_string(), "nope".to_string()], &Selection::All)
        .expect_err("an absent variable must be an error");
    let msg = err.to_string();
    assert!(msg.contains("nope"), "error must name the absent variable: {msg}");
    assert!(msg.contains("sp"), "error must list what IS present: {msg}");
}

/// The whole-file `netcdf` reader honours an orthogonal `Selection` at DECODE
/// time: one blob, one cache key, only the requested hyperslab materialised.
///
/// The gate is that a windowed read equals the FULL read sliced afterwards, cell
/// for cell — anything else is an off-by-one — and that the COORDINATES come back
/// windowed with it, which is the trap a whole-file reader has that the
/// store-backed zarr reader (no coords at all) never had.
#[test]
fn netcdf_select_windows_the_decode_and_slices_the_coords() {
    let corpus = corpus_dir();
    let case: Value =
        serde_json::from_slice(&fs::read(corpus.join("cases/era5-grid-sub-tile.json")).unwrap())
            .unwrap();
    let blob = corpus.join(case["blob_path"].as_str().unwrap());
    let reader = FormatRegistry::with_builtins().get("netcdf").unwrap();
    assert!(reader.supports_selection());
    assert!(!reader.store_backed(), "the fetch is unchanged; only the decode shrinks");

    let full = reader.read_native(&blob, &[], &Selection::All).unwrap();
    let full_t2m = to_opt_f64(&full.variables["t2m"].data);

    // time all, latitude [1,3), longitude {0,2}
    let sel = Selection::Orthogonal(vec![
        AxisSelect::All,
        AxisSelect::Range { start: 1, stop: 3, step: 1 },
        AxisSelect::Indices(vec![0, 2]),
    ]);
    let w = reader.read_native(&blob, &[], &sel).unwrap();
    let t2m = &w.variables["t2m"];
    assert_eq!(t2m.shape, vec![2, 2, 2]);
    assert_eq!(t2m.dims, vec!["time", "latitude", "longitude"]);

    // full[t, 1..3, {0,2}] gathered by hand out of the [2,3,3] row-major array.
    let mut expected: Vec<Option<f64>> = Vec::new();
    for t in 0..2 {
        for y in 1..3 {
            for &x in &[0usize, 2usize] {
                expected.push(full_t2m[t * 9 + y * 3 + x]);
            }
        }
    }
    assert_eq!(to_opt_f64(&t2m.data), expected);

    // Coordinates are windowed WITH the data; the time axis is untouched and
    // keeps its raw values plus units/calendar.
    let full_lat = to_opt_f64(&full.coords["latitude"].field.data);
    let full_lon = to_opt_f64(&full.coords["longitude"].field.data);
    assert_eq!(
        to_opt_f64(&w.coords["latitude"].field.data),
        vec![full_lat[1], full_lat[2]]
    );
    assert_eq!(
        to_opt_f64(&w.coords["longitude"].field.data),
        vec![full_lon[0], full_lon[2]]
    );
    assert_eq!(
        to_opt_f64(&w.coords["time"].field.data),
        to_opt_f64(&full.coords["time"].field.data)
    );
    assert_eq!(w.coords["time"].calendar.as_deref(), Some("gregorian"));

    // An explicit index list comes back in the ORDER GIVEN (the zarr rule): this
    // is the path that reads a bounding slab and gathers out of it.
    let permuted = Selection::Orthogonal(vec![
        AxisSelect::All,
        AxisSelect::All,
        AxisSelect::Indices(vec![2, 0]),
    ]);
    let p = reader.read_native(&blob, &[], &permuted).unwrap();
    assert_eq!(
        to_opt_f64(&p.coords["longitude"].field.data),
        vec![full_lon[2], full_lon[0]]
    );
    let pt = to_opt_f64(&p.variables["t2m"].data);
    for t in 0..2 {
        for y in 0..3 {
            assert_eq!(pt[(t * 3 + y) * 2], full_t2m[t * 9 + y * 3 + 2]);
            assert_eq!(pt[(t * 3 + y) * 2 + 1], full_t2m[t * 9 + y * 3]);
        }
    }
}

/// Time is the Provider's axis (it owns the cadence), and an axis count that
/// matches no array is a mistake — both are refused, not quietly honoured.
#[test]
fn netcdf_select_refuses_a_time_subset_and_a_rank_mismatch() {
    let corpus = corpus_dir();
    let case: Value =
        serde_json::from_slice(&fs::read(corpus.join("cases/era5-grid-sub-tile.json")).unwrap())
            .unwrap();
    let blob = corpus.join(case["blob_path"].as_str().unwrap());
    let reader = FormatRegistry::with_builtins().get("netcdf").unwrap();

    let time_sub = Selection::Orthogonal(vec![
        AxisSelect::Indices(vec![0]),
        AxisSelect::All,
        AxisSelect::All,
    ]);
    let err = reader.read_native(&blob, &[], &time_sub).unwrap_err().to_string();
    assert!(err.contains("time"), "{err}");

    let wrong_rank = Selection::Orthogonal(vec![AxisSelect::All, AxisSelect::All]);
    let err = reader
        .read_native(&blob, &[], &wrong_rank)
        .unwrap_err()
        .to_string();
    assert!(err.contains("rank"), "{err}");
}

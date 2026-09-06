//! Rust track's native-array dumper for the cross-language conformance harness.
//!
//! Drives the **Rust Provider** ([`earthsciio::Provider`]) over every committed
//! corpus case, fully OFFLINE (the cache is rooted at the corpus and refuses the
//! network), and emits the decoded native arrays as a canonical JSON dump in the
//! SAME schema as the Python (`conformance/dumpers/dump_python.py`) and Julia
//! (`conformance/dumpers/dump_julia.jl`) dumpers. The cross-language comparator
//! (`conformance/crosscheck.py`) diffs the three dumps + the corpus oracle to
//! prove native-array equality across all three tracks (`esio-9nb.9`).
//!
//! Dump schema — `earthsciio/native-dump/v1` (see `conformance/CROSSLANG.md`).
//! Rust's [`ArrayData`] is already row-major (C order) per `shape`, so no permute
//! is needed; a masked / `_FillValue` cell is `null` (== NaN); strings verbatim.
//! A case whose `format` has no reader in this track (e.g. `csv` — the Rust track
//! ships `netcdf` only) is `status="skipped"` (explicit, never dropped) so the
//! comparator can tell a real coverage gap from a bug.
//!
//! Usage:  cargo run --example conformance_dump -- [out.json]   # default: stdout

use std::path::{Path, PathBuf};
use std::sync::Arc;

use earthsciio::{
    ArrayData, AxisSelect, Cache, Coord, DType, DataSource, Ff10Reader, FormatRegistry,
    NativeField, Provider, Selection,
};
use serde_json::{json, Map, Value};

/// Build a configured [`Ff10Reader`] from an ff10 case's `decode` block —
/// `member` (singular), `members`/`member_glob` (multi-member; sorted-name
/// concat), `skip_header_row` (drop one asserted `country_cd` header line per
/// member). Returns `None` when the case pins none of them (the default
/// registered reader already decodes the bare blob). The configured reader is
/// injected via `Provider::with_formats` — `FormatRegistry::register` keys by
/// `formats()`, so it REPLACES the default `ff10` registration for that case.
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

fn read_json(path: &Path) -> Value {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
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

/// The flattened (already row-major) values as a JSON array; NaN → `null`.
fn data_to_json(data: &ArrayData) -> Value {
    match data {
        ArrayData::F64(v) => Value::Array(
            v.iter()
                .map(|&x| if x.is_nan() { Value::Null } else { json!(x) })
                .collect(),
        ),
        ArrayData::I64(v) => Value::Array(v.iter().map(|&x| json!(x)).collect()),
        ArrayData::I32(v) => Value::Array(v.iter().map(|&x| json!(x)).collect()),
        ArrayData::Str(v) => Value::Array(v.iter().map(|s| json!(s)).collect()),
        ArrayData::Bool(v) => Value::Array(v.iter().map(|&b| json!(b)).collect()),
    }
}

/// Encode one [`NativeField`] to the dump schema (dtype/dims/shape/data), plus
/// the surviving fill/missing sentinel under the dump's ONE `fill_value` key.
///
/// `spec/conformance.md` §3 pins where each track reports that sentinel: Rust
/// has a dedicated `NativeField::fill_value` field, Python and Julia carry
/// `attrs["fill_value"]`. Those are the SAME datum in three spellings, not
/// three decisions, so every dumper normalises to this one key — otherwise the
/// comparator would read a difference that is not one (or miss one track
/// dropping the sentinel entirely).
fn encode_field(f: &NativeField) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("dtype".into(), json!(dtype_str(f.dtype)));
    m.insert("dims".into(), json!(f.dims));
    m.insert("shape".into(), json!(f.shape));
    if let Some(fv) = f.fill_value {
        m.insert("fill_value".into(), json!(fv));
    }
    m.insert("data".into(), data_to_json(&f.data));
    m
}

/// A coord is a field plus the CF units/calendar it carries (if any).
fn encode_coord(c: &Coord) -> Value {
    let mut m = encode_field(&c.field);
    if let Some(u) = &c.units {
        m.insert("units".into(), json!(u));
    }
    if let Some(cal) = &c.calendar {
        m.insert("calendar".into(), json!(cal));
    }
    Value::Object(m)
}

/// Run the Rust Provider over one corpus case and encode its native arrays. Skips
/// (without error) a case whose format has no reader, so the harness reports the
/// gap instead of failing — the Rust track ships `netcdf` only today.
fn dump_case(corpus: &Path, case: &Value, base_formats: &FormatRegistry) -> Value {
    let fmt = case["format"].as_str().unwrap();
    if base_formats.get(fmt).is_none() {
        return json!({
            "format": fmt,
            "status": "skipped",
            "reason": format!("no active reader registered for format '{fmt}' in the Rust track"),
        });
    }

    // The Rust Reader trait takes no kwargs, so a case whose decode block pins
    // ff10 zip member selection / header handling gets a reader CONFIGURED at
    // construction, injected through the `Provider::with_formats` seam.
    let mut formats = base_formats.clone();
    if fmt == "ff10" {
        if let Some(reader) = ff10_reader_from_decode(case) {
            formats.register(Arc::new(reader));
        }
    }
    let formats = &formats;

    // An OFFLINE cache rooted at the corpus: each case resolves from disk by its
    // sha256(resolved_url) key; verify_on_read checks the blob against its manifest.
    let cache = Cache::builder()
        .data_dir(corpus.join("cache"))
        .offline(true)
        .verify_on_read(true)
        .build()
        .expect("offline cache over the corpus");

    let url = case["resolved_url"].as_str().unwrap();
    let mut loader = DataSource::new(case["loader"].as_str().unwrap(), fmt, url);
    // Columnar table (parquet): `variables` is the loader's PROJECTION, pushed
    // into the reader so only those column chunks come off disk, and the case's
    // decode block pins the three decode options — `float_columns` (an integer
    // measurement, or a column of fixed-decimal TEXT, read as float64) and the
    // two null gates `null_int` / `null_string`. Both reach the reader the way a
    // DOCUMENT delivers them: the loader's `variables` + `reader_options`,
    // resolved by `Reader::configured`.
    if fmt == "parquet" {
        if let Some(vars) = case.get("variables").and_then(Value::as_array) {
            loader = loader.variables(
                vars.iter().map(|v| v.as_str().unwrap().to_string()).collect::<Vec<_>>(),
            );
        }
        let mut options = Map::new();
        if let Some(dec) = case.get("decode") {
            for k in ["float_columns", "null_int", "null_string"] {
                match dec.get(k) {
                    Some(v) if !v.is_null() => {
                        options.insert(k.to_string(), v.clone());
                    }
                    _ => {}
                }
            }
        }
        loader = loader.reader_options(options);
    }
    // Whole-file, but selection-capable (netcdf): a case carrying an orthogonal
    // `select` is a DECODE-TIME window — the same blob under the same cache key,
    // with only the requested hyperslab materialised. `parse_selection` yields
    // `Selection::All` for a case with no `axes`.
    if fmt == "netcdf" {
        loader = loader.select(parse_selection(case));
    }
    // Store-backed (zarr): name the arrays (no .zmetadata to enumerate) + carry
    // the orthogonal selection that drives lazy chunk fetch.
    if formats.get(fmt).map(|r| r.store_backed()).unwrap_or(false) {
        let vars: Vec<String> = case["variables"]
            .as_array()
            .expect("store-backed case has a variables array")
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        loader = loader.variables(vars).select(parse_selection(case));
    }
    // The shapefile case pins its zip member + the code column the model wants
    // numeric. Both reach the reader the way a DOCUMENT delivers them — the
    // loader's `reader_options`, resolved by `Reader::configured`.
    if fmt == "shapefile" {
        let mut options = Map::new();
        if let Some(dec) = case.get("decode") {
            for k in ["member", "numeric_columns"] {
                match dec.get(k) {
                    Some(v) if !v.is_null() => {
                        options.insert(k.to_string(), v.clone());
                    }
                    _ => {}
                }
            }
        }
        loader = loader.reader_options(options);
    }
    let mut provider = Provider::with_formats(loader, Arc::new(cache), None, formats)
        .expect("provider over corpus");
    let buffers = provider.materialize().expect("materialize the corpus blob");

    let mut variables = Map::new();
    for (name, field) in &buffers {
        variables.insert(name.clone(), Value::Object(encode_field(field)));
    }
    let mut coords = Map::new();
    for (name, coord) in provider.coords() {
        coords.insert(name.clone(), encode_coord(coord));
    }

    json!({
        "format": fmt,
        "status": "decoded",
        "variables": Value::Object(variables),
        "coords": Value::Object(coords),
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus = corpus_dir();
    let formats = FormatRegistry::with_builtins();

    let mut active = formats.registered();
    active.sort();

    // `$ESIO_CONFORMANCE_CASES` narrows the corpus to a comma-separated id list,
    // honoured identically by every dumper and by `crosscheck.py`, so a filtered
    // run is still a complete cross-check of the cases it names. It exists for an
    // environment where one track's backend for some OTHER format is missing or
    // too old: without it a single unrelated broken case makes the gate
    // unrunnable and a real divergence elsewhere invisible. Unset => every case.
    let raw = std::env::var("ESIO_CONFORMANCE_CASES").unwrap_or_default();
    let only: Option<Vec<&str>> = match raw.trim() {
        "" => None,
        list => Some(list.split(',').map(str::trim).filter(|s| !s.is_empty()).collect()),
    };

    let index = read_json(&corpus.join("cases.json"));
    let mut cases = Map::new();
    for entry in index["cases"].as_array().expect("cases array") {
        let case = read_json(&corpus.join(entry["file"].as_str().unwrap()));
        let id = case["id"].as_str().unwrap().to_string();
        if let Some(only) = &only {
            if !only.contains(&id.as_str()) {
                continue;
            }
        }
        cases.insert(id, dump_case(&corpus, &case, &formats));
    }

    let out = json!({
        "schema": "earthsciio/native-dump/v1",
        "language": "rust",
        "provider": "earthsciio::Provider",
        "readers": active,
        "cases": Value::Object(cases),
    });

    let text = serde_json::to_string_pretty(&out).expect("serialize dump");
    if let Some(path) = args.get(1) {
        std::fs::write(path, text + "\n").unwrap_or_else(|e| panic!("write {path}: {e}"));
    } else {
        println!("{text}");
    }
}

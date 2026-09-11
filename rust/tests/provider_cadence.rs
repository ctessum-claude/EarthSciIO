//! The cadence-aware `Provider` over the committed ERA5 corpus fixture, fully
//! offline (acceptance: "Provider over the shared fixture returns native arrays
//! matching the tracks; CONST/DISCRETE correct; refresh_times() matches the
//! cadence"). All reads resolve from the Python-populated corpus cache — no
//! network, no per-language data tooling beyond the netcdf reader.

// Native-only: this tier drives the cache, the transports and the format
// readers, none of which exist on wasm32 (see the crate's module docs).
#![cfg(not(target_arch = "wasm32"))]

use std::sync::Arc;

use earthsciio::{ArrayData, Cache, DataSource, Error, NativeField, Provider, SourceTemporal};
use time::macros::datetime;
use time::{Duration, OffsetDateTime};

const ERA5_URL: &str = "https://data.earthsci.dev/era5/2018/11/20181108.nc";
// Resolves the file (day) anchor to ERA5_URL via `time` formatting.
const ERA5_TEMPLATE: &str = "https://data.earthsci.dev/era5/[year]/[month]/[year][month][day].nc";

fn corpus_cache() -> Arc<Cache> {
    let root =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../conformance/corpus/cache");
    Arc::new(
        Cache::builder()
            .data_dir(root)
            .offline(true)
            .verify_on_read(true)
            .build()
            .expect("offline cache over the corpus"),
    )
}

fn f64s(field: &NativeField) -> &[f64] {
    match &field.data {
        ArrayData::F64(v) => v,
        other => panic!("expected f64 data, got {:?}", other.dtype()),
    }
}

// --- DISCRETE (era5 is hourly within a daily file) --------------------------

#[test]
fn discrete_refresh_times_match_the_hourly_cadence() {
    let window = (
        datetime!(2018-11-08 00:00:00 UTC),
        datetime!(2018-11-08 02:00:00 UTC),
    );
    let loader = DataSource::new("era5", "netcdf", ERA5_TEMPLATE).temporal(SourceTemporal::new(
        datetime!(2018-11-08 00:00:00 UTC),
        Duration::hours(1),
        Duration::days(1),
    ));
    let provider = Provider::new(loader, corpus_cache(), Some(window)).unwrap();

    let times = provider.refresh_times();
    // Two hourly tstops in [00:00, 02:00): 00:00 and 01:00.
    assert_eq!(times.len(), 2, "hourly cadence over a 2-hour window");
    assert_eq!(
        times[0],
        datetime!(2018-11-08 00:00:00 UTC).unix_timestamp() as f64
    );
    assert_eq!(times[1] - times[0], 3600.0, "one-hour cadence step");
}

#[test]
fn discrete_refresh_selects_the_right_record_at_each_boundary() {
    let window = (
        datetime!(2018-11-08 00:00:00 UTC),
        datetime!(2018-11-08 02:00:00 UTC),
    );
    let loader = DataSource::new("era5", "netcdf", ERA5_TEMPLATE).temporal(SourceTemporal::new(
        datetime!(2018-11-08 00:00:00 UTC),
        Duration::hours(1),
        Duration::days(1),
    ));
    let mut provider = Provider::new(loader, corpus_cache(), Some(window)).unwrap();

    // materialize() primes the buffer at the first anchor (record 0).
    let buf0 = provider.materialize().unwrap();
    let t2m0 = &buf0["t2m"];
    assert_eq!(
        t2m0.dims,
        vec!["latitude".to_string(), "longitude".to_string()]
    );
    assert_eq!(
        t2m0.shape,
        vec![3, 3],
        "a single hourly slice, time axis dropped"
    );
    assert_eq!(f64s(t2m0)[0], 282.5); // record-0 first cell
    assert_eq!(f64s(t2m0)[8], 284.5); // record-0 last cell (not masked)

    // Within the same hour: no change.
    assert!(
        provider
            .refresh(datetime!(2018-11-08 00:30:00 UTC))
            .unwrap()
            .is_none(),
        "t inside the same cadence interval must not refresh"
    );

    // Next hour: record 1, which carries the masked (NaN) cell.
    let buf1 = provider
        .refresh(datetime!(2018-11-08 01:00:00 UTC))
        .unwrap()
        .expect("crossing the cadence boundary refreshes");
    let t2m1 = &buf1["t2m"];
    assert_eq!(t2m1.shape, vec![3, 3]);
    assert_eq!(f64s(t2m1)[0], 282.6); // record-1 first cell
    assert!(
        f64s(t2m1)[8].is_nan(),
        "record-1 last cell is _FillValue -> NaN"
    );

    // Re-asking for the same hour is idempotent.
    assert!(provider
        .refresh(datetime!(2018-11-08 01:00:00 UTC))
        .unwrap()
        .is_none());

    // Native coordinates are exposed on the loader's grid.
    let coords = provider.coords();
    assert_eq!(f64s(&coords["latitude"].field), &[40.0, 39.5, 39.0]);
    assert_eq!(f64s(&coords["longitude"].field), &[-122.0, -121.5, -121.0]);
    assert_eq!(
        coords["time"].units.as_deref(),
        Some("hours since 2018-11-08 00:00:00"),
        "time axis carries CF units verbatim (calendar decoding stays in ESS)"
    );
}

#[test]
fn discrete_prefetch_warms_the_window_offline() {
    let window = (
        datetime!(2018-11-08 00:00:00 UTC),
        datetime!(2018-11-08 02:00:00 UTC),
    );
    let loader = DataSource::new("era5", "netcdf", ERA5_TEMPLATE).temporal(SourceTemporal::new(
        datetime!(2018-11-08 00:00:00 UTC),
        Duration::hours(1),
        Duration::days(1),
    ));
    let mut provider = Provider::new(loader, corpus_cache(), Some(window)).unwrap();
    // The day's file is in the corpus → prefetch resolves it with no network.
    provider
        .prefetch(window)
        .expect("prefetch the window from the corpus");
}

// --- DISCRETE records_per_sample=2: the 2-record bracket --------------------
//
// With `records_per_sample = 2`, refresh returns the two bracketing records
// (floor + successor) with the time axis RETAINED at length 2, plus a canonical
// 2-element epoch-seconds `time` coordinate, so a downstream model interpolates
// in time. `file_period = 2h` makes each file hold exactly the two hourly records
// the corpus blob carries, so hour 1 is the last record in its file (its
// successor is record 0 of the next file).

fn interp_loader(end: Option<OffsetDateTime>) -> DataSource {
    let mut temporal = SourceTemporal::new(
        datetime!(2018-11-08 00:00:00 UTC),
        Duration::hours(1),
        Duration::hours(2),
    )
    .records_per_sample(2);
    if let Some(e) = end {
        temporal = temporal.end(e);
    }
    DataSource::new("era5", "netcdf", ERA5_TEMPLATE).temporal(temporal)
}

#[test]
fn interp_bracket_returns_two_records_with_time_axis() {
    let window = (
        datetime!(2018-11-08 00:00:00 UTC),
        datetime!(2018-11-08 02:00:00 UTC),
    );
    let mut provider = Provider::new(interp_loader(None), corpus_cache(), Some(window)).unwrap();

    let buf = provider
        .refresh(datetime!(2018-11-08 00:00:00 UTC))
        .unwrap()
        .expect("the first anchor produces a bracket");
    let t2m = &buf["t2m"];
    // The time axis is RETAINED at length 2 (floor + successor), unlike the
    // single-record slice which drops it.
    assert_eq!(
        t2m.dims,
        vec![
            "time".to_string(),
            "latitude".to_string(),
            "longitude".to_string()
        ]
    );
    assert_eq!(t2m.shape, vec![2, 3, 3], "[time=2, lat=3, lon=3] bracket");
    assert_eq!(f64s(t2m)[0], 282.5); // record 0 (hour 0), first cell
    assert_eq!(f64s(t2m)[9], 282.6); // record 1 (hour 1), first cell (offset 3*3)

    // The time coordinate carries the two bracket timestamps as Unix epoch seconds.
    let time = &provider.coords()["time"];
    assert_eq!(
        f64s(&time.field),
        &[
            datetime!(2018-11-08 00:00:00 UTC).unix_timestamp() as f64,
            datetime!(2018-11-08 01:00:00 UTC).unix_timestamp() as f64,
        ]
    );
    assert_eq!(
        time.units.as_deref(),
        Some("seconds since 1970-01-01T00:00:00Z")
    );
}

#[test]
fn interp_bracket_floors_within_interval() {
    let window = (
        datetime!(2018-11-08 00:00:00 UTC),
        datetime!(2018-11-08 02:00:00 UTC),
    );
    let mut provider = Provider::new(interp_loader(None), corpus_cache(), Some(window)).unwrap();

    let at = provider
        .refresh(datetime!(2018-11-08 00:00:00 UTC))
        .unwrap()
        .expect("first anchor brackets");
    let at0 = f64s(&at["t2m"])[0];

    // Flooring within the same hour is the SAME bracket: the anchor is unchanged,
    // so refresh reports no change (the buffer still holds the hour 0 -> 1 bracket).
    assert!(
        provider
            .refresh(datetime!(2018-11-08 00:30:00 UTC))
            .unwrap()
            .is_none(),
        "a t inside the same cadence interval keeps the same bracket"
    );
    // ... and the retained buffer is exactly that bracket.
    assert_eq!(at0, 282.5);
    let time = &provider.coords()["time"];
    assert_eq!(
        f64s(&time.field),
        &[
            datetime!(2018-11-08 00:00:00 UTC).unix_timestamp() as f64,
            datetime!(2018-11-08 01:00:00 UTC).unix_timestamp() as f64,
        ]
    );
}

#[test]
fn interp_bracket_crosses_file_boundary() {
    // No temporal.end and no window => a successor is assumed to exist. Hour 1 is
    // the last record in its 2-record file, so its successor is record 0 of the
    // next file (the same corpus blob resolves for the next day-file anchor here).
    let mut provider = Provider::new(interp_loader(None), corpus_cache(), None).unwrap();

    let buf = provider
        .refresh(datetime!(2018-11-08 01:00:00 UTC))
        .unwrap()
        .expect("hour 1 brackets across the file seam");
    let t2m = &buf["t2m"];
    assert_eq!(t2m.shape, vec![2, 3, 3]);
    assert_eq!(f64s(t2m)[0], 282.6); // this file, record 1
    assert_eq!(f64s(t2m)[9], 282.5); // next file, record 0
    let time = &provider.coords()["time"];
    assert_eq!(
        f64s(&time.field),
        &[
            datetime!(2018-11-08 01:00:00 UTC).unix_timestamp() as f64,
            datetime!(2018-11-08 02:00:00 UTC).unix_timestamp() as f64,
        ]
    );
}

#[test]
fn interp_bracket_end_clamp_degenerates() {
    // temporal.end = hour 2 => hour 1 has no successor; the bracket holds
    // [last, last] with equal timestamps so the downstream weight clamps.
    let window = (
        datetime!(2018-11-08 00:00:00 UTC),
        datetime!(2018-11-08 02:00:00 UTC),
    );
    let end = datetime!(2018-11-08 02:00:00 UTC);
    let mut provider =
        Provider::new(interp_loader(Some(end)), corpus_cache(), Some(window)).unwrap();

    let buf = provider
        .refresh(datetime!(2018-11-08 01:00:00 UTC))
        .unwrap()
        .expect("hour 1 brackets (degenerately) at the end of data");
    let t2m = &buf["t2m"];
    assert_eq!(t2m.shape, vec![2, 3, 3]);
    assert_eq!(f64s(t2m)[0], 282.6);
    assert_eq!(f64s(t2m)[9], 282.6, "successor == floor (held at the endpoint)");
    let time = f64s(&provider.coords()["time"].field);
    assert_eq!(time[0], time[1], "degenerate bracket has equal timestamps");
    assert_eq!(
        time[0],
        datetime!(2018-11-08 01:00:00 UTC).unix_timestamp() as f64
    );
}

#[test]
fn interp_rejects_bad_mode() {
    // An unsupported records-per-sample count is a clean error at refresh, not a panic.
    let loader = DataSource::new("era5", "netcdf", ERA5_TEMPLATE).temporal(
        SourceTemporal::new(
            datetime!(2018-11-08 00:00:00 UTC),
            Duration::hours(1),
            Duration::hours(1),
        )
        .records_per_sample(3),
    );
    let mut provider = Provider::new(loader, corpus_cache(), None).unwrap();
    match provider.refresh(datetime!(2018-11-08 00:00:00 UTC)) {
        Err(Error::Format { detail, .. }) => {
            assert!(detail.contains("records_per_sample"), "detail: {detail}");
        }
        other => panic!("expected a Format error for a bad records_per_sample count, got {other:?}"),
    }
}

// --- CONST (no temporal: read once, never refresh) --------------------------

#[test]
fn const_materialize_reads_whole_file_and_never_refreshes() {
    // A loader with no temporal block is CONST. Pointed at the same ERA5 blob via
    // a literal URL, materialize() returns the full (un-sliced) native arrays.
    let loader = DataSource::new("era5_static", "netcdf", ERA5_URL);
    let mut provider = Provider::new(loader, corpus_cache(), None).unwrap();

    let buf = provider.materialize().unwrap();
    let t2m = &buf["t2m"];
    assert_eq!(
        t2m.dims,
        vec![
            "time".to_string(),
            "latitude".to_string(),
            "longitude".to_string()
        ]
    );
    assert_eq!(t2m.shape, vec![2, 3, 3], "CONST keeps the full file shape");
    assert_eq!(f64s(t2m)[0], 282.5);

    // CONST: refresh is a no-op and the cadence schedule is empty.
    assert!(provider
        .refresh(datetime!(2018-11-08 01:00:00 UTC))
        .unwrap()
        .is_none());
    assert!(provider.refresh_times().is_empty());
}

// --- registry plumbing ------------------------------------------------------

#[test]
fn unknown_format_is_a_clean_error() {
    let loader = DataSource::new("mystery", "grib", ERA5_URL); // no grib reader
                                                               // (Provider holds trait objects and isn't Debug, so match rather than unwrap_err.)
    match Provider::new(loader, corpus_cache(), None) {
        Err(Error::UnknownFormat { name }) => assert_eq!(name, "grib"),
        Err(other) => panic!("expected UnknownFormat, got {other}"),
        Ok(_) => panic!("expected UnknownFormat error for an unregistered format"),
    }
}

#[test]
fn a_records_baked_into_reader_options_never_reaches_the_decode() {
    // `records` is the CADENCE OWNER's option (spec/conformance.md, "One owner
    // per record axis"). A Provider with a `temporal` resolves each anchor to a
    // file-local record itself, so a caller-supplied `records` would narrow that
    // axis BEHIND the resolution: with a one-record `records` every anchor would
    // land on record 0 and `refresh` would hand back the same record forever — a
    // constant field where the model expected to interpolate, with no error
    // anywhere. That is the hole the Julia track had to close after making
    // `records` a declared option there.
    //
    // In Rust it cannot open. A loader states its decode options as
    // `reader_options`, resolved through `Reader::configured`, and the netcdf
    // reader takes none: the pushdown is a per-call argument of
    // `read_native_records`, never a configured property of a reader instance.
    // The default `configured` REFUSES a non-empty option set rather than
    // ignoring it, so this is an error at Provider construction. Pinned here so a
    // later `configured` override on the netcdf reader cannot quietly open it.
    let mut options = serde_json::Map::new();
    options.insert(
        "records".to_string(),
        serde_json::json!({"dim": "time", "indices": [0]}),
    );
    let temporal = SourceTemporal::new(
        datetime!(2018-11-08 00:00:00 UTC),
        Duration::hours(1),
        Duration::days(1),
    );
    let loader = DataSource::new("era5", "netcdf", ERA5_TEMPLATE)
        .temporal(temporal)
        .reader_options(options);
    match Provider::new(loader, corpus_cache(), None) {
        Err(Error::Format { format, detail }) => {
            assert_eq!(format, "netcdf");
            assert!(detail.contains("takes no reader_options"), "{detail}");
            assert!(detail.contains("records"), "{detail}");
        }
        Err(other) => panic!("expected a Format error, got {other}"),
        Ok(_) => panic!("a baked `records` beside a cadence must not be accepted"),
    }
}

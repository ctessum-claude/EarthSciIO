//! Store-backed `zarr` read_store — the load-bearing LAZINESS capability proven
//! against the committed corpus store: a runtime index list fetches ONLY the
//! intersecting chunk objects. Non-selected chunks are overwritten with
//! undecodable "poison" bytes in a scratch copy of the corpus, so any over-fetch
//! blosc-errors instead of silently succeeding.

// Native-only: this tier drives the cache, the transports and the format
// readers, none of which exist on wasm32 (see the crate's module docs).
#![cfg(not(target_arch = "wasm32"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use earthsciio::{
    cache_key, ArrayData, AxisSelect, Cache, DataSource, Provider, Selection,
};

const BASE: &str = "s3://earthsci-fixtures/isrm-mini.zarr";

fn corpus_cache() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../conformance/corpus/cache")
}

/// Recursively copy `src` into `dst`.
fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let to = dst.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &to);
        } else {
            std::fs::copy(entry.path(), &to).unwrap();
        }
    }
}

/// Overwrite the cached blob for `url` (bare-key layout) with `bytes`.
fn poison(root: &Path, url: &str, bytes: &[u8]) {
    let key = cache_key(url);
    let blob = root.join("v1/blobs").join(&key[..2]).join(&key);
    assert!(blob.exists(), "expected corpus blob for {url} at {}", blob.display());
    std::fs::write(&blob, bytes).unwrap();
}

#[test]
fn read_store_is_lazy_never_touching_unselected_chunks() {
    let scratch = tempfile::tempdir().unwrap();
    let cache_root = scratch.path().join("cache");
    copy_dir(&corpus_cache(), &cache_root);

    // The selection layer=[1], y=[1,4], x=all needs ONLY field3d/1.0.0 and
    // field3d/1.2.0 (+ pop1d/0). Poison every other field3d chunk: a non-lazy
    // reader that fetched them would blosc-error.
    for key in ["0.0.0", "0.1.0", "0.2.0", "1.1.0"] {
        poison(&cache_root, &format!("{BASE}/field3d/{key}"), b"\x00POISON-not-blosc\xff");
    }

    // verify_on_read=false so poison is only "seen" if actually decoded (a
    // verify=true read would fail on the poison's integrity before decode — also
    // proving non-fetch, but this isolates the laziness of the reader itself).
    let cache = Cache::builder()
        .data_dir(&cache_root)
        .offline(true)
        .verify_on_read(false)
        .build()
        .unwrap();

    let loader = DataSource::new("isrm", "zarr", BASE)
        .variables(["field3d", "pop1d"])
        .select(Selection::Orthogonal(vec![
            AxisSelect::Indices(vec![1]),
            AxisSelect::Indices(vec![1, 4]),
            AxisSelect::All,
        ]));
    let mut provider = Provider::new(loader, Arc::new(cache), None).unwrap();
    let buffers = provider.materialize().unwrap();

    let f3 = &buffers["field3d"];
    assert_eq!(f3.dims, vec!["layer", "y", "x"]);
    assert_eq!(f3.shape, vec![1, 2, 4]);
    assert_eq!(
        f3.data,
        ArrayData::F64(vec![110.0, 111.0, 112.0, 113.0, 140.0, 141.0, 142.0, 143.0])
    );
    // pop1d (rank 1 != 3 axes) reads whole.
    assert_eq!(
        buffers["pop1d"].data,
        ArrayData::F64(vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
    );
}

#[test]
fn over_selection_hits_poison_and_errors() {
    // Control: a selection that DOES touch a poisoned chunk decode-errors,
    // proving the poison is genuinely undecodable (so the lazy test is meaningful).
    let scratch = tempfile::tempdir().unwrap();
    let cache_root = scratch.path().join("cache");
    copy_dir(&corpus_cache(), &cache_root);
    // Poison the middle y-chunk (rows 2,3) of layer 1.
    poison(&cache_root, &format!("{BASE}/field3d/1.1.0"), b"\x00POISON\xff");

    let cache = Cache::builder()
        .data_dir(&cache_root)
        .offline(true)
        .verify_on_read(false)
        .build()
        .unwrap();
    let loader = DataSource::new("isrm", "zarr", BASE)
        .variables(["field3d"])
        .select(Selection::Orthogonal(vec![
            AxisSelect::Indices(vec![1]),
            AxisSelect::Indices(vec![2]), // row 2 -> the poisoned middle chunk
            AxisSelect::All,
        ]));
    let mut provider = Provider::new(loader, Arc::new(cache), None).unwrap();
    assert!(provider.materialize().is_err());
}

// --- Phase 1: per-call `select` override, ordering, supports_selection, shape --- //

const PBASE: &str = "s3://earthsci-fixtures/permuted-tile.zarr";

#[test]
fn per_call_select_preserves_permuted_order_and_is_lazy() {
    let scratch = tempfile::tempdir().unwrap();
    let cache_root = scratch.path().join("cache");
    copy_dir(&corpus_cache(), &cache_root);

    // select layer=[0], source=[24,2,9,6] (PERMUTED) needs ONLY sr/0.0.0 (sources
    // 2,9,6 -> chunk 0) and sr/0.2.0 (source 24 -> chunk 2). Poison EVERY other sr
    // chunk: a non-lazy read (or one that scanned the whole array) would blosc-error.
    for l in 0..3 {
        for s in 0..5 {
            if !(l == 0 && (s == 0 || s == 2)) {
                poison(&cache_root, &format!("{PBASE}/sr/{l}.{s}.0"), b"\x00POISON-not-blosc\xff");
            }
        }
    }

    let cache = Cache::builder()
        .data_dir(&cache_root)
        .offline(true)
        .verify_on_read(false)
        .build()
        .unwrap();

    // Baked select defaults to All; the PER-CALL override drives the projection.
    let loader = DataSource::new("isrm", "zarr", PBASE).variables(["sr"]);
    let mut provider = Provider::new(loader, Arc::new(cache), None).unwrap();

    let sel = Selection::Orthogonal(vec![
        AxisSelect::Indices(vec![0]),
        AxisSelect::Indices(vec![24, 2, 9, 6]),
        AxisSelect::All,
    ]);
    let buffers = provider.materialize_with_select(Some(&sel)).unwrap();
    let sr = &buffers["sr"];
    assert_eq!(sr.dims, vec!["layer", "source", "receptor"]);
    assert_eq!(sr.shape, vec![1, 4, 4]);
    // Rows follow the GIVEN order 24,2,9,6 (value = source*100 + receptor); a reader
    // that SORTED the index list would return 2,6,9,24 and fail this assertion.
    assert_eq!(
        sr.data,
        ArrayData::F64(vec![
            2400.0, 2401.0, 2402.0, 2403.0, // source 24
            200.0, 201.0, 202.0, 203.0, // source 2
            900.0, 901.0, 902.0, 903.0, // source 9
            600.0, 601.0, 602.0, 603.0, // source 6
        ])
    );
}

#[test]
fn per_call_select_overrides_baked_and_leaves_it_intact() {
    let cache = Cache::builder()
        .data_dir(corpus_cache())
        .offline(true)
        .verify_on_read(true)
        .build()
        .unwrap();
    let baked = Selection::Orthogonal(vec![
        AxisSelect::Indices(vec![0]),
        AxisSelect::Indices(vec![2]),
        AxisSelect::All,
    ]);
    let loader = DataSource::new("isrm", "zarr", PBASE).variables(["sr"]).select(baked);
    let mut provider = Provider::new(loader, Arc::new(cache), None).unwrap();

    // No per-call select ⇒ the baked select applies: source [2] -> [200,201,202,203].
    assert_eq!(
        provider.materialize().unwrap()["sr"].data,
        ArrayData::F64(vec![200.0, 201.0, 202.0, 203.0])
    );

    // A per-call override applies for this call only.
    let over = Selection::Orthogonal(vec![
        AxisSelect::Indices(vec![0]),
        AxisSelect::Indices(vec![24, 2, 9, 6]),
        AxisSelect::All,
    ]);
    let peeked = provider.materialize_with_select(Some(&over)).unwrap();
    assert_eq!(peeked["sr"].shape, vec![1, 4, 4]);
    assert_eq!(
        peeked["sr"].data,
        ArrayData::F64(vec![
            2400.0, 2401.0, 2402.0, 2403.0, 200.0, 201.0, 202.0, 203.0, 900.0, 901.0, 902.0,
            903.0, 600.0, 601.0, 602.0, 603.0,
        ])
    );

    // ... and the baked default is untouched afterwards.
    assert_eq!(
        provider.materialize().unwrap()["sr"].data,
        ArrayData::F64(vec![200.0, 201.0, 202.0, 203.0])
    );
}

#[test]
fn supports_selection_and_array_shape_capability_surface() {
    let cache = Arc::new(
        Cache::builder()
            .data_dir(corpus_cache())
            .offline(true)
            .verify_on_read(true)
            .build()
            .unwrap(),
    );

    // store-backed zarr provider CAN push down; array_shape reads ONLY .zarray.
    let zloader = DataSource::new("isrm", "zarr", PBASE).variables(["sr"]);
    let zprovider = Provider::new(zloader, cache.clone(), None).unwrap();
    assert!(zprovider.supports_selection());
    assert_eq!(zprovider.array_shape("sr").unwrap(), Some(vec![3, 50, 4]));

    assert!(zprovider.store_backed());

    // whole-file (netcdf) reader: it DOES honour a select, at decode time, but it
    // is not store-backed — that pair is how a caller tells "the download shrank"
    // from "only the decode did". Shape is still unknown without a read.
    let nloader =
        DataSource::new("era5", "netcdf", "https://data.earthsci.dev/era5/2018/11/20181108.nc");
    let nprovider = Provider::new(nloader, cache.clone(), None).unwrap();
    assert!(nprovider.supports_selection());
    assert!(!nprovider.store_backed());
    assert_eq!(nprovider.array_shape("t2m").unwrap(), None);

    // a reader that can honour neither
    let floader = DataSource::new("nei2016", "ff10", "https://data.earthsci.dev/ff10/x.csv");
    let fprovider = Provider::new(floader, cache, None).unwrap();
    assert!(!fprovider.supports_selection());
    assert!(!fprovider.store_backed());
}

#[test]
fn per_call_select_on_whole_file_reader_errors() {
    let cache = Cache::builder()
        .data_dir(corpus_cache())
        .offline(true)
        .verify_on_read(true)
        .build()
        .unwrap();
    // `ff10` honours no selection at all (`netcdf` now does, at decode time).
    let loader = DataSource::new("nei2016", "ff10", "https://data.earthsci.dev/ff10/x.csv");
    let mut provider = Provider::new(loader, Arc::new(cache), None).unwrap();
    // A per-call projection on a reader that can't honour it is an error (pre-fetch).
    let sel = Selection::Orthogonal(vec![AxisSelect::All]);
    assert!(provider.materialize_with_select(Some(&sel)).is_err());
}

/// A BAKED `DataSource::select` on a reader that honours none is an error too,
/// not a silently dropped selection — the Julia and Python tracks both refuse it,
/// and a caller who reads the whole array believing it got a window has a wrong
/// number, not a slow one. Raised before any fetch.
#[test]
fn baked_select_on_a_reader_that_honours_none_errors() {
    let cache = Arc::new(
        Cache::builder()
            .data_dir(corpus_cache())
            .offline(true)
            .verify_on_read(true)
            .build()
            .unwrap(),
    );
    let loader = DataSource::new("nei2016", "ff10", "https://data.earthsci.dev/ff10/x.csv")
        .select(Selection::Orthogonal(vec![AxisSelect::All]));
    let mut provider = Provider::new(loader, cache, None).unwrap();
    let err = provider.materialize().unwrap_err().to_string();
    assert!(err.contains("does not support select"), "{err}");
}

/// The whole-file `netcdf` reader honours both, with the per-call `select`
/// OVERRIDING the baked one for that call only — the same precedence the
/// store-backed zarr reader has, and the same the Julia/Python tracks pin.
#[test]
fn netcdf_per_call_select_overrides_the_baked_one() {
    let cache = Arc::new(
        Cache::builder()
            .data_dir(corpus_cache())
            .offline(true)
            .verify_on_read(true)
            .build()
            .unwrap(),
    );
    let baked = Selection::Orthogonal(vec![
        AxisSelect::All,
        AxisSelect::Range { start: 1, stop: 3, step: 1 },
        AxisSelect::Indices(vec![0, 2]),
    ]);
    let loader =
        DataSource::new("era5", "netcdf", "https://data.earthsci.dev/era5/2018/11/20181108.nc")
            .select(baked);
    let mut provider = Provider::new(loader, cache, None).unwrap();
    assert!(provider.supports_selection() && !provider.store_backed());

    // the baked selection is honoured with no per-call argument at all
    assert_eq!(provider.materialize().unwrap()["t2m"].shape, vec![2, 2, 2]);

    // a per-call select wins for this call only...
    let peek = Selection::Orthogonal(vec![
        AxisSelect::All,
        AxisSelect::Indices(vec![0]),
        AxisSelect::All,
    ]);
    let got = provider.materialize_with_select(Some(&peek)).unwrap();
    assert_eq!(got["t2m"].shape, vec![2, 1, 3]);
    // ...and it is a peek: the next plain read is the baked projection again.
    assert_eq!(provider.materialize().unwrap()["t2m"].shape, vec![2, 2, 2]);
}

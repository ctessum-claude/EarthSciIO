//! EarthSciIO — Rust core (component (a)): URL download + a shared
//! content-addressed cache.
//!
//! This is the **first data-loader machinery in the Rust track**. It implements
//! the language-neutral spec under [`../spec`](https://github.com/earthsci/earthsciio/tree/main/spec):
//!
//! - the **shared cache key** `sha256(resolved_url)` ([`cache_key`]) and the
//!   on-disk [`cache-format`](https://github.com/earthsci/earthsciio/blob/main/spec/cache-format.md)
//!   (`v1/blobs/<key[:2]>/<key>.<ext>`, `meta/<key>.json`, `locks/`, `tmp/`);
//! - the [`Manifest`] format, byte-identical to the Python writer, so a blob
//!   fetched by one language is reused and re-validated by the others;
//! - the **transport** and **store** registries
//!   ([`registries`](https://github.com/earthsci/earthsciio/blob/main/spec/registries.md)):
//!   the active `http`/`https`, `file`, and `cds` (Copernicus CDS API:
//!   submit→poll→download) transports and the `local` store — with the ERA5
//!   pressure-level request mapping ([`era5`]) building `cds://` URLs;
//! - `$EARTHSCIDATADIR` resolution ([`data_dir`]), **offline mode**
//!   ([`Cache::is_offline`], [`Error::CacheMiss`]), the ETag/checksum/TTL
//!   validation ladder ([`validate`]), mirror failover, and the pluggable
//!   [`auth`] seam;
//! - the concurrency contract — advisory `flock` + atomic rename — so multiple
//!   processes sharing one `/scratch.local` cache download a URL exactly once.
//!
//! Component (b) builds on this core: the [`FormatRegistry`]'s **readers**
//! decode a cached blob into native-grid arrays ([`NetcdfReader`]), and the
//! cadence-aware [`Provider`] drives `materialize`/`refresh`/`refresh_times`/
//! `prefetch` over them — returning **raw** native arrays (remap/regrid stay
//! upstream/downstream).
//!
//! # Example — fetch (or reuse) a blob
//!
//! (The `cfg` guard is what lets these compile as doctests on the wasm32
//! target, where the cache does not exist — see "Two targets" below.)
//!
//! ```no_run
//! # #[cfg(not(target_arch = "wasm32"))]
//! # fn main() -> Result<(), earthsciio::Error> {
//! use earthsciio::{Cache, FetchRequest};
//!
//! let cache = Cache::from_env()?;                 // $EARTHSCIDATADIR + EARTHSCI_OFFLINE
//! let blob = cache.fetch(&FetchRequest::new("https://data.earthsci.dev/era5/2018/11/20181108.nc")
//!     .loader("era5"))?;
//! println!("cached at {} ({} bytes)", blob.path.display(), blob.manifest.bytes);
//! # Ok(()) }
//! # #[cfg(target_arch = "wasm32")]
//! # fn main() {}
//! ```
//!
//! # Example — offline, cache-only (hermetic)
//!
//! ```no_run
//! # #[cfg(not(target_arch = "wasm32"))]
//! # fn main() -> Result<(), earthsciio::Error> {
//! use earthsciio::{Cache, FetchRequest};
//!
//! let cache = Cache::builder().data_dir("conformance/corpus/cache").offline(true).build()?;
//! let blob = cache.fetch(&FetchRequest::new("https://data.earthsci.dev/era5/2018/11/20181108.nc"))?;
//! // A miss raises Error::CacheMiss naming the url + key — never a silent empty.
//! # Ok(()) }
//! # #[cfg(target_arch = "wasm32")]
//! # fn main() {}
//! ```
//!
//! # Two targets, one writer
//!
//! Everything above describes the **native** crate. The crate also builds for
//! `wasm32-unknown-unknown`, where it is deliberately a much smaller thing: the
//! **output half only** — the codec profiles, the Zarr v3 sharded writer, and
//! (feature `opfs`) a store backed by the browser's Origin Private File System.
//!
//! That exists so earthscilab's browser tier writes a run's output with *this*
//! writer rather than a fourth implementation in JavaScript
//! (`docs/output-handling.md` §4.2/§4.6): one writer, two hosts.
//!
//! The cache, the transports, the store registry, the `Provider` cadence path
//! and every format **reader** are native-only, because each is built on
//! something a browser tab does not have — blocking sockets, `flock`, a temp
//! file on the blob filesystem, a memmap. They are `cfg`-ed out rather than
//! stubbed: a wasm build that calls for them fails to compile, which is a much
//! better outcome than one that links and then returns an empty dataset.
//! `Cargo.toml`'s target tables carry the same split at the dependency level.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

// `opfs`'s dependencies are declared in the wasm32 target table, so on a native
// build the feature would silently activate a module whose imports do not
// resolve. Say so instead.
#[cfg(all(feature = "opfs", not(target_arch = "wasm32")))]
compile_error!(
    "feature `opfs` is wasm32-only (it is a browser filesystem); \
     build it with --target wasm32-unknown-unknown"
);

#[cfg(not(target_arch = "wasm32"))]
pub mod auth;
#[cfg(not(target_arch = "wasm32"))]
mod cache;
mod clock;
#[cfg(not(target_arch = "wasm32"))]
pub mod datadir;
#[cfg(not(target_arch = "wasm32"))]
pub mod era5;
mod error;
pub mod format;
#[cfg(not(target_arch = "wasm32"))]
mod key;
#[cfg(not(target_arch = "wasm32"))]
pub mod manifest;
#[cfg(not(target_arch = "wasm32"))]
mod offline;
#[cfg(not(target_arch = "wasm32"))]
mod provider;
#[cfg(not(target_arch = "wasm32"))]
pub mod s3_config;
#[cfg(not(target_arch = "wasm32"))]
pub mod store;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod test_env;
#[cfg(not(target_arch = "wasm32"))]
pub mod transport;
#[cfg(not(target_arch = "wasm32"))]
pub mod validate;

pub use error::{Error, Result};

// The output half — available on every target.
pub use format::{
    write_all_to_store, BloscProfile, CodecProfile, OutputSchema, WriteCoord, WriteVar,
    ZstdProfile, BLOSC_CHECKPOINT, BLOSC_DIAGNOSTIC, ZSTD_WASM,
};
#[cfg(all(target_arch = "wasm32", feature = "opfs"))]
pub use format::{read_zarr_opfs_array, write_zarr_opfs, OpfsStore};

#[cfg(not(target_arch = "wasm32"))]
pub use cache::{Cache, CacheBuilder, CachedBlob, FetchRequest};
#[cfg(not(target_arch = "wasm32"))]
pub use datadir::{data_dir, default_data_dir, expand_datadir, DATADIR_ENV};
#[cfg(not(target_arch = "wasm32"))]
pub use format::{
    write_zarr_v3, ArrayData, AxisSelect, Coord, DType, Ff10Reader, FormatRegistry, GeoTiffReader,
    NativeDataset, NativeField, NetcdfReader, ParquetReader, Reader, Records, Selection,
    ShapefileReader, ZarrReader,
};
#[cfg(all(feature = "object-store", not(target_arch = "wasm32")))]
pub use format::{
    array_shape_object_store, read_store_options, read_zarr_object_store,
    read_zarr_object_store_with_options, store_options_from_env, write_zarr_object_store,
    write_zarr_object_store_with_options,
};
#[cfg(not(target_arch = "wasm32"))]
pub use key::{cache_key, cache_key_range, sha256_file, sha256_hex};
#[cfg(not(target_arch = "wasm32"))]
pub use manifest::{Manifest, MANIFEST_SCHEMA};
#[cfg(not(target_arch = "wasm32"))]
pub use offline::{is_offline, OFFLINE_ENV};
#[cfg(not(target_arch = "wasm32"))]
pub use provider::{DataSource, Provider, SourceTemporal, StoreAccess, Window, STORE_ACCESS_ENV};
#[cfg(not(target_arch = "wasm32"))]
pub use s3_config::{
    bucket_options_from_env, options_for_bucket, options_for_url, parse_bucket_options,
    stated_region, stated_signing, BUCKET_OPTIONS_ENV,
};

/// The 0.1.1 spelling of [`DataSource`]. From `.esm` 1.0.0 the declaration is a
/// `data_sources` entry, not a `data_loaders` one, and its consumers spell it
/// `DataSource`; this alias keeps 0.1.1 source-compatible and is dropped in
/// 0.2.0.
#[cfg(not(target_arch = "wasm32"))]
#[deprecated(
    since = "0.1.2",
    note = "renamed to `DataSource` (.esm 1.0.0 `data_sources`)"
)]
pub type DataLoader = DataSource;

/// The 0.1.1 spelling of [`SourceTemporal`]. See [`DataLoader`].
#[cfg(not(target_arch = "wasm32"))]
#[deprecated(since = "0.1.2", note = "renamed to `SourceTemporal`")]
pub type LoaderTemporal = SourceTemporal;
#[cfg(not(target_arch = "wasm32"))]
pub use validate::{CacheDecision, Temporal};

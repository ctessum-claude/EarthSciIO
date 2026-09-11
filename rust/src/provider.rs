//! The cadence-aware **Provider** (component (b); plan §4.1–4.6).
//!
//! A `Provider` is bound to one loader and created once per simulation. It holds
//! the parsed loader, a cache handle, the run window, the format reader resolved
//! by the loader's declared format, and a **current native-grid buffer** per
//! variable. The solver reads the buffer as a constant between cadence
//! boundaries; at a boundary the solver's callback calls [`Provider::refresh`].
//!
//! It returns **raw native-grid arrays** keyed by the on-disk `file_variable`
//! name. Variable-name remap and `unit_conversion` stay in ESS; regrid is
//! ESD/C4's job (plan §4.3) — the Provider is the sanctioned impure I/O boundary
//! and nothing more.
//!
//! # CONST vs DISCRETE (plan §4.2)
//!
//! - **CONST** — no `temporal`: [`materialize`](Provider::materialize) reads the
//!   single file once; [`refresh`](Provider::refresh) is a no-op (`None`);
//!   [`refresh_times`](Provider::refresh_times) is empty.
//! - **DISCRETE** — has `temporal`: `refresh(t)` snaps `t` to the loader's
//!   cadence anchor and, when the anchor changed, re-reads the matching record
//!   into the buffer. `refresh_times()` is the cadence schedule over the window —
//!   the solver tstops the callback fires on.
//!
//! # Time type
//!
//! The plan sketches `DateTime<Utc>`, but this crate already standardizes on the
//! `time` crate (see `validate.rs`, the manifest's RFC 3339 stamps); the Provider
//! uses [`time::OffsetDateTime`] for one consistent time type across the core.

use std::collections::HashMap;
use std::sync::Arc;

use time::{Duration, OffsetDateTime};

use crate::cache::{Cache, FetchRequest};
use crate::error::{Error, Result};
use crate::format::{
    ArrayData, Coord, DType, FormatRegistry, NativeDataset, NativeField, Reader, Selection,
};

/// A run window `(start, end)` — half-open `[start, end)`.
pub type Window = (OffsetDateTime, OffsetDateTime);

/// The temporal nature of a loader that refreshes at a cadence (plan §4.2).
///
/// `frequency` is the cadence step (e.g. 1 hour for ERA5) — it drives the
/// refresh tstops and which record within a file is current. `file_period` is
/// the granularity of one file (e.g. 1 day) — it drives URL resolution. A file
/// holds `file_period / frequency` records along `time_dim`.
#[derive(Debug, Clone)]
pub struct SourceTemporal {
    /// The loader's epoch — cadence anchors are aligned to this instant.
    pub start: OffsetDateTime,
    /// Exclusive end of available data, if known (open-ended otherwise).
    pub end: Option<OffsetDateTime>,
    /// Cadence step between successive records.
    pub frequency: Duration,
    /// Time span covered by one file (drives which URL/file is fetched).
    pub file_period: Duration,
    /// Name of the record/time dimension to slice on (default `"time"`).
    pub time_dim: String,
    /// Number of records [`Provider::refresh`] returns per sample. `None` or
    /// `Some(1)` (default) returns the single at-or-before record with `time_dim`
    /// dropped (held piecewise-constant); `Some(2)` returns the two bracketing
    /// records (floor + successor) with `time_dim` retained at length 2 and a
    /// canonical 2-element epoch-seconds `time` coordinate, so a downstream model
    /// interpolates in time. The successor is read across a file boundary when
    /// needed; at the last available record the bracket degenerates to
    /// `[last, last]` so the downstream weight clamps. Only `1` / `2` are
    /// supported (validated in [`Provider::refresh`]); higher-order temporal
    /// stencils are future work.
    pub records_per_sample: Option<u32>,
}

impl SourceTemporal {
    /// A temporal block with `frequency` cadence and `file_period` files,
    /// anchored at `start`, open-ended, slicing on the `time` dimension.
    pub fn new(start: OffsetDateTime, frequency: Duration, file_period: Duration) -> Self {
        Self {
            start,
            end: None,
            frequency,
            file_period,
            time_dim: "time".to_string(),
            records_per_sample: None,
        }
    }

    /// Set the exclusive end of available data.
    pub fn end(mut self, end: OffsetDateTime) -> Self {
        self.end = Some(end);
        self
    }

    /// Override the record/time dimension name (default `"time"`).
    pub fn time_dim(mut self, dim: impl Into<String>) -> Self {
        self.time_dim = dim.into();
        self
    }

    /// Set the number of records returned per sample. `1` (piecewise-constant
    /// single record) or `2` (the 2-record bracket); any other value is
    /// rejected when [`Provider::refresh`] validates the cadence.
    pub fn records_per_sample(mut self, n: u32) -> Self {
        self.records_per_sample = Some(n);
        self
    }
}

/// Environment variable setting the process-wide default for [`StoreAccess`]:
/// `direct` or `cached` (case-insensitive). See [`StoreAccess::from_env`].
pub const STORE_ACCESS_ENV: &str = "EARTHSCIIO_STORE_ACCESS";

/// How a **store-backed** loader (zarr) gets at its objects.
///
/// # Why this is a choice and not a fix
///
/// The content-addressed cache is what makes a *repeated* read cheap: a second
/// run, a second variable out of the same store, a DISCRETE loader stepping back
/// over a file it already has. That is most loaders, which is why
/// [`Cached`](StoreAccess::Cached) is the default and stays the default.
///
/// It earns nothing on a **cold, single-pass, in-region** read, and there it is
/// pure cost: the whole fetched slab is written to disk, read back once, and
/// discarded with the task. The InMAP ISRM source-receptor read is the case in
/// point — ~15–25 GB of gated chunks from a public bucket in the same region as
/// the compute, every chunk touched exactly once. Caching it buys nothing, costs
/// a full write-then-read on the critical path, and on AWS Fargate (20 GiB
/// ephemeral storage by default) may simply not fit.
///
/// # Who decides
///
/// The **caller** does, because the caller is the only one who knows the access
/// pattern. The cache cannot infer it: "will anyone read this object again?" is
/// a fact about the future, and the two cases are indistinguishable at the
/// moment of the first read. Size is not a proxy either — a small store read a
/// thousand times wants the cache more than a huge one read once.
///
/// The alternatives were considered and rejected:
///
/// * **A URL scheme** (`s3+direct://`) — the URL is also the *cache identity*
///   and the document's declared source of truth. A scheme says where bytes
///   live; overloading it to say how we buffer them makes two different URLs
///   name the same object.
/// * **An environment variable alone** — process-global, so a run with one
///   single-pass 20 GB store and one small repeatedly-read store cannot express
///   both. It is still supported ([`from_env`](StoreAccess::from_env)) as a
///   *deployment* override, because a runner that receives documents it did not
///   author has no other way to ask; it is a default, not the decision.
///
/// Precedence is therefore: an explicit [`DataSource::store_access`] wins; then
/// [`STORE_ACCESS_ENV`]; then [`Cached`](StoreAccess::Cached).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StoreAccess {
    /// Every store object is fetched through the content-addressed [`Cache`]:
    /// staged to a temp file, committed to disk, then read back. Integrity
    /// checking, offline replay, mirror failover and cross-process locking all
    /// come with it. **The default** — a warm-cache workflow must never become
    /// mysteriously slow because something changed underneath it.
    #[default]
    Cached,
    /// Store objects are read straight from object storage and decoded in
    /// memory. **Nothing is written to disk.** Selection pushdown is unchanged:
    /// only the chunk objects the selection intersects are fetched.
    ///
    /// The trade is real — no integrity record, no offline replay, no reuse
    /// across runs, and a re-read costs a second download. Choose it when the
    /// read is cold, single-pass, and near the data.
    Direct,
}

impl StoreAccess {
    /// The process-wide default from [`STORE_ACCESS_ENV`] (`direct` / `cached`),
    /// or [`StoreAccess::Cached`] when unset or unrecognised.
    ///
    /// An unrecognised value falls back to the safe default rather than
    /// erroring: a typo'd deployment variable should not take a run down, and
    /// the consequence of ignoring it is a slower run, never a wrong one.
    #[must_use]
    pub fn from_env() -> Self {
        match std::env::var(STORE_ACCESS_ENV) {
            Ok(v) if v.trim().eq_ignore_ascii_case("direct") => StoreAccess::Direct,
            _ => StoreAccess::Cached,
        }
    }
}

/// The minimal parsed `DataSource` the Provider needs: where the bytes are, how
/// to decode them, which variables to read, and the temporal cadence (absent ⇒
/// CONST). This is the I/O-relevant projection of the ESM `DataSource` contract —
/// the full contract (units, remap, grid family) lives upstream in ESS.
#[derive(Debug, Clone)]
pub struct DataSource {
    /// Loader name (provenance, recorded in the cache manifest).
    pub name: String,
    /// Format-registry key selecting the reader (e.g. `"netcdf"`).
    pub format: String,
    /// On-disk `file_variable` names to read; empty ⇒ all data variables.
    pub variables: Vec<String>,
    /// Cadence (absent ⇒ CONST/static).
    pub temporal: Option<SourceTemporal>,
    /// URL template in `time` format-description syntax (e.g.
    /// `".../era5/[year]/[month]/[year][month][day].nc"`). A template with no
    /// `[` placeholders is a literal URL (the CONST case).
    pub url_template: String,
    /// Failover mirrors sharing the same cache identity.
    pub mirrors: Vec<String>,
    /// Auth realm to fetch under (resolved by the cache's auth registry).
    pub auth_realm: Option<String>,
    /// Spatial/orthogonal selection for any reader that answers
    /// [`Reader::supports_selection`] — the store-backed `zarr` one (which fetches
    /// only the intersecting chunk objects) and the whole-file `netcdf` one (which
    /// fetches the same blob under the same cache key and materialises only the
    /// requested hyperslab). Default [`Selection::All`]; ignored by a reader that
    /// can honour neither. The Provider still owns temporal record slicing, so a
    /// selection's time axis must be `All`.
    pub select: Selection,
    /// Format-specific decode options, resolved against the registered reader
    /// by [`Reader::configured`] when the Provider is built — the Rust spelling
    /// of the Python/Julia `reader_kwargs`.
    ///
    /// This is how a *declaration* (an `.esm` data loader) says how its format
    /// is decoded, instead of a caller hand-injecting a configured reader
    /// through [`Provider::with_formats`]: `{"member_glob": "*egu*",
    /// "skip_header_row": true}` on an `ff10` loader is the FF10 zip case that
    /// motivated it. An option the bound reader does not recognise is an error
    /// at construction, never a silently-ignored key.
    pub reader_options: serde_json::Map<String, serde_json::Value>,
    /// Whether a **store-backed** read goes through the cache. `None` ⇒ not
    /// stated, so [`StoreAccess::from_env`] decides (which defaults to
    /// [`StoreAccess::Cached`]). Ignored by whole-file readers, which have no
    /// direct path. See [`StoreAccess`] for who decides and why.
    pub store_access: Option<StoreAccess>,
    /// Backend config for a [`StoreAccess::Direct`] read — `object_store`'s own
    /// option keys (`endpoint`, `region`, `aws_skip_signature`, credentials, …).
    /// Empty ⇒ environment defaults. These override the environment per loader,
    /// which is what a process-global variable cannot express when two loaders
    /// read two different stores.
    ///
    /// They are also the only place a *read* can say it should be **signed**:
    /// an `s3://` read defaults to anonymous no matter what credentials the
    /// process environment is carrying, because a platform-injected role is not
    /// this document asking for anything. So a public bucket needs nothing here,
    /// and a private one needs credentials or `aws_skip_signature=false`. See
    /// `format::zarr_object_store::read_store_options`.
    pub store_options: Vec<(String, String)>,
}

impl DataSource {
    /// A loader named `name`, decoded by `format`, resolving `url_template`.
    /// CONST by default — add [`temporal`](DataSource::temporal) for cadence.
    pub fn new(
        name: impl Into<String>,
        format: impl Into<String>,
        url_template: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            format: format.into(),
            variables: Vec::new(),
            temporal: None,
            url_template: url_template.into(),
            mirrors: Vec::new(),
            auth_realm: None,
            select: Selection::All,
            reader_options: serde_json::Map::new(),
            store_access: None,
            store_options: Vec::new(),
        }
    }

    /// Set the spatial/orthogonal selection for a store-backed reader (zarr).
    pub fn select(mut self, select: Selection) -> Self {
        self.select = select;
        self
    }

    /// Declare the format-specific decode options (see
    /// [`DataSource::reader_options`]). Resolved against the reader when the
    /// [`Provider`] is built, so an unrecognised option fails there.
    pub fn reader_options(mut self, options: serde_json::Map<String, serde_json::Value>) -> Self {
        self.reader_options = options;
        self
    }

    /// State whether this loader's store objects go through the cache,
    /// overriding [`STORE_ACCESS_ENV`]. See [`StoreAccess`].
    ///
    /// [`StoreAccess::Direct`] on a loader whose reader has no direct path is a
    /// named error at [`Provider::new`], never a silent fall back to the cache.
    pub fn store_access(mut self, access: StoreAccess) -> Self {
        self.store_access = Some(access);
        self
    }

    /// Backend config for a [`StoreAccess::Direct`] read (`object_store` option
    /// keys). Overrides the environment for this loader only — an S3-compatible
    /// endpoint, a region, or credentials / `aws_skip_signature=false` to ask for
    /// a **signed** read. An unsigned public read needs nothing here: see
    /// [`DataSource::store_options`](struct.DataSource.html#structfield.store_options).
    pub fn store_options<I, K, V>(mut self, options: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.store_options = options
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();
        self
    }

    /// Restrict to specific on-disk variable names (empty keeps "all").
    pub fn variables<I, S>(mut self, vars: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.variables = vars.into_iter().map(Into::into).collect();
        self
    }

    /// Make the loader DISCRETE with the given cadence.
    pub fn temporal(mut self, temporal: SourceTemporal) -> Self {
        self.temporal = Some(temporal);
        self
    }

    /// Set failover mirror templates.
    pub fn mirrors<I, S>(mut self, mirrors: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.mirrors = mirrors.into_iter().map(Into::into).collect();
        self
    }

    /// Fetch under an auth realm.
    pub fn auth_realm(mut self, realm: impl Into<String>) -> Self {
        self.auth_realm = Some(realm.into());
        self
    }
}

/// A loader-bound provider of native-grid arrays, refreshed at the loader's
/// cadence. See the module-level documentation for the CONST/DISCRETE contract.
pub struct Provider {
    loader: DataSource,
    cache: Arc<Cache>,
    window: Option<Window>,
    reader: Arc<dyn Reader>,
    /// Current native-grid buffer per variable (the slice the solver reads).
    buffers: HashMap<String, NativeField>,
    /// Native coordinates of the current file.
    coords: HashMap<String, Coord>,
    /// The full decode of the currently-open file (re-sliced as the record moves).
    current_dataset: Option<NativeDataset>,
    /// The cadence anchor currently in the buffer (DISCRETE).
    current_anchor: Option<OffsetDateTime>,
    /// The file anchor currently open (avoids re-reading within a file).
    current_file_anchor: Option<OffsetDateTime>,
    /// Tiny LRU of decoded files keyed by file anchor (cap 2), used only by
    /// `records_per_sample = 2`: the successor of the last record in a file
    /// lives in the next file, so the bracket may hold two adjacent files at a
    /// file-period seam. Most-recently-used at the tail.
    files: Vec<(OffsetDateTime, NativeDataset)>,
    /// The store-access decision, resolved ONCE at construction (loader → env →
    /// [`StoreAccess::Cached`]) so it cannot change mid-run and so the
    /// environment is read once rather than per object.
    store_access: StoreAccess,
}

impl Provider {
    /// Bind a provider to `loader`, resolving its format reader from the built-in
    /// [`FormatRegistry`]. Errors with [`Error::UnknownFormat`] if no reader is
    /// registered for the loader's format.
    pub fn new(loader: DataSource, cache: Arc<Cache>, window: Option<Window>) -> Result<Self> {
        Self::with_formats(loader, cache, window, &FormatRegistry::with_builtins())
    }

    /// [`Provider::new`] but resolving the reader from a caller-supplied registry
    /// — the seam that lets a new format plug in without any Provider change.
    pub fn with_formats(
        loader: DataSource,
        cache: Arc<Cache>,
        window: Option<Window>,
        formats: &FormatRegistry,
    ) -> Result<Self> {
        let reader = formats
            .get(&loader.format)
            .ok_or_else(|| Error::UnknownFormat {
                name: loader.format.clone(),
            })?;
        // The loader's declared decode options configure the registered reader
        // (an unrecognised option errors here, not mid-decode).
        let reader = match reader.configured(&loader.reader_options)? {
            Some(configured) => configured,
            None => reader,
        };
        // Resolve the cached/direct decision once, and refuse an impossible one
        // HERE rather than at the first read. Falling back to the cache would
        // silently restore the disk cost the caller explicitly asked to avoid —
        // which on a 20 GiB ephemeral disk is a failure, not a slow success.
        let store_access = loader.store_access.unwrap_or_else(StoreAccess::from_env);
        if store_access == StoreAccess::Direct
            && reader.store_backed()
            && !reader.supports_direct_read()
        {
            return Err(Error::Format {
                format: loader.format.clone(),
                detail: format!(
                    "loader '{}' asked for uncached (direct) store reads, but the '{}' \
                     reader has no direct path in this build — rebuild earthsciio with \
                     the `object-store` feature, or use StoreAccess::Cached",
                    loader.name, loader.format
                ),
            });
        }
        Ok(Self {
            loader,
            cache,
            window,
            reader,
            store_access,
            buffers: HashMap::new(),
            coords: HashMap::new(),
            current_dataset: None,
            current_anchor: None,
            current_file_anchor: None,
            files: Vec::new(),
        })
    }

    /// The native coordinates of the current grid (latitude/longitude/time, …).
    /// Populated after the first [`materialize`](Self::materialize) /
    /// [`refresh`](Self::refresh).
    pub fn coords(&self) -> &HashMap<String, Coord> {
        &self.coords
    }

    /// True when the bound reader can honour an orthogonal `select` **without
    /// materialising the whole array**. A caller uses this to decide whether to
    /// push a projection down (via
    /// [`materialize_with_select`](Self::materialize_with_select)) or read whole
    /// and slice on its own side. Mirrors the Julia/Python `supports_selection`.
    ///
    /// It says nothing about what is FETCHED — pair it with
    /// [`store_backed`](Self::store_backed): `zarr` fetches only the intersecting
    /// chunk objects, so a selection shrinks the download too; `netcdf` fetches
    /// the same blob under the same cache key and only materialises the requested
    /// hyperslab.
    pub fn supports_selection(&self) -> bool {
        self.reader.supports_selection()
    }

    /// True when the bound reader reads a directory-like STORE (a Zarr v2 store,
    /// each object its own fetch) rather than one blob. Read with
    /// [`supports_selection`](Self::supports_selection) it tells a caller whether
    /// a pushed-down `select` shrinks the DOWNLOAD or only the decode.
    #[must_use]
    pub fn store_backed(&self) -> bool {
        self.reader.store_backed()
    }

    /// The full native (dims-order) shape of on-disk array `var`, for a
    /// honour/refuse pushdown decision, or `None`.
    ///
    /// For a store-backed zarr provider this reads ONLY the array's `.zarray`
    /// metadata (never a chunk); `None` for a whole-file reader (whose shape is not
    /// knowable without reading the blob). Mirrors the Julia/Python `array_shape`.
    pub fn array_shape(&self, var: &str) -> Result<Option<Vec<usize>>> {
        if !self.reader.store_backed() {
            return Ok(None);
        }
        let anchor = match &self.loader.temporal {
            None => OffsetDateTime::UNIX_EPOCH,
            Some(t) => t.start,
        };
        let url = self.resolve_url(anchor)?;
        match self.store_access {
            StoreAccess::Direct => {
                self.reader
                    .array_shape_direct(&url, var, &self.loader.store_options)
            }
            StoreAccess::Cached => self.reader.array_shape(self.cache.clone(), &url, var),
        }
    }

    /// How this provider reaches a store-backed loader's objects, after the
    /// loader → environment → default resolution. See [`StoreAccess`].
    #[must_use]
    pub fn store_access(&self) -> StoreAccess {
        self.store_access
    }

    /// Materialize the loader's native arrays into the buffer.
    ///
    /// CONST: reads the single file once. DISCRETE: primes the buffer at the
    /// first cadence anchor of the window (equivalent to `refresh(window.start)`),
    /// so a caller can read an initial state before stepping.
    pub fn materialize(&mut self) -> Result<HashMap<String, NativeField>> {
        self.materialize_with_select(None)
    }

    /// [`materialize`](Self::materialize) with an optional PER-CALL projection
    /// pushdown that OVERRIDES the loader's baked [`DataSource::select`] for this
    /// call only — the seam a caller (EarthSciAST) uses to push a projection down
    /// at sample time without rebuilding the provider. Mirrors the Julia/Python
    /// per-call `select` override.
    ///
    /// Only a store-backed reader that [`supports_selection`](Self::supports_selection)
    /// can honour a per-call `select`; passing one to a whole-file reader is an
    /// error. A per-call `select` is a projection **peek**: it reads fresh and
    /// returns the selected arrays WITHOUT disturbing the cadence buffer the solver
    /// reads (so a subsequent plain `materialize()` restores the baked projection).
    pub fn materialize_with_select(
        &mut self,
        sel: Option<&Selection>,
    ) -> Result<HashMap<String, NativeField>> {
        if sel.is_some() && !self.reader.supports_selection() {
            return Err(Error::Format {
                format: self.loader.format.clone(),
                detail: format!(
                    "reader for format '{}' does not support select/pushdown",
                    self.loader.format
                ),
            });
        }
        match self.loader.temporal.clone() {
            None => {
                let effective = sel.unwrap_or(&self.loader.select);
                let url = self.resolve_url(OffsetDateTime::UNIX_EPOCH)?;
                let ds = self.read_file(url, effective)?;
                if sel.is_none() {
                    self.coords = ds.coords;
                    self.buffers = ds.variables;
                    Ok(self.buffers.clone())
                } else {
                    // Per-call override: a projection peek; leave the buffer intact.
                    Ok(ds.variables)
                }
            }
            Some(t) => {
                let first = self.lower_bound(&t);
                match self.refresh_with_select(first, sel)? {
                    Some(buffers) => Ok(buffers),
                    None => Ok(self.buffers.clone()),
                }
            }
        }
    }

    /// Refresh the buffer to the cadence anchor for time `t`.
    ///
    /// Returns `Some(buffer)` when the anchor changed (the solver must re-read),
    /// `None` when `t` falls in the same cadence interval as the last refresh, and
    /// `None` for a CONST loader (nothing refreshes).
    pub fn refresh(&mut self, t: OffsetDateTime) -> Result<Option<HashMap<String, NativeField>>> {
        self.refresh_with_select(t, None)
    }

    /// [`refresh`](Self::refresh) with an optional PER-CALL projection pushdown
    /// override (see [`materialize_with_select`](Self::materialize_with_select)).
    ///
    /// With `sel = None` this is exactly [`refresh`](Self::refresh): it consults
    /// the file cache, slices the current record, and updates the cadence buffer.
    /// With `sel = Some(_)` it is a cache-bypassing projection **peek**: it reads
    /// the covering file(s) fresh under `sel`, returns `Some(selected_arrays)` for
    /// `t`'s anchor, and does NOT disturb the buffer / file caches. A per-call
    /// `select` on a reader that cannot honour it is an error.
    pub fn refresh_with_select(
        &mut self,
        t: OffsetDateTime,
        sel: Option<&Selection>,
    ) -> Result<Option<HashMap<String, NativeField>>> {
        if sel.is_some() && !self.reader.supports_selection() {
            return Err(Error::Format {
                format: self.loader.format.clone(),
                detail: format!(
                    "reader for format '{}' does not support select/pushdown",
                    self.loader.format
                ),
            });
        }
        let Some(temporal) = self.loader.temporal.clone() else {
            return Ok(None); // CONST: refresh is a no-op
        };
        let freq_s = temporal.frequency.whole_seconds();
        let file_s = temporal.file_period.whole_seconds();
        if freq_s <= 0 || file_s <= 0 {
            return Err(Error::Format {
                format: self.loader.format.clone(),
                detail: "loader cadence frequency/file_period must be positive".to_string(),
            });
        }
        if !matches!(temporal.records_per_sample, None | Some(1) | Some(2)) {
            return Err(Error::Format {
                format: self.loader.format.clone(),
                detail: format!(
                    "records_per_sample must be 1 or 2, got {}",
                    temporal.records_per_sample.unwrap()
                ),
            });
        }

        let anchor = snap_down(temporal.start, t, freq_s);

        // PER-CALL override: a fresh, cache-bypassing projection peek. It returns
        // the selected arrays for `anchor` WITHOUT touching the buffer / caches, so
        // it never short-circuits on `current_anchor` and never evicts a cached file.
        if let Some(select) = sel {
            if temporal.records_per_sample == Some(2) {
                return Ok(Some(self.bracket_peek(&temporal, anchor, freq_s, file_s, select)?));
            }
            let file_anchor = snap_down(temporal.start, anchor, file_s);
            let url = self.resolve_url(file_anchor)?;
            let ds = self.read_file(url, select)?;
            let rec = ((anchor - file_anchor).whole_seconds() / freq_s) as usize;
            return Ok(Some(slice_record_buffers(&ds, temporal.time_dim.as_str(), rec)?));
        }

        // No override: the existing cached path (behavior unchanged).
        if self.current_anchor == Some(anchor) {
            return Ok(None); // same cadence interval — unchanged (bracket too)
        }
        if temporal.records_per_sample == Some(2) {
            return self.refresh_bracket(&temporal, anchor, freq_s, file_s);
        }
        let file_anchor = snap_down(temporal.start, anchor, file_s);

        // Re-read the file only when the anchor crossed a file boundary.
        if self.current_file_anchor != Some(file_anchor) {
            let url = self.resolve_url(file_anchor)?;
            let ds = self.read_file(url, &self.loader.select)?;
            self.coords = ds.coords.clone();
            self.current_dataset = Some(ds);
            self.current_file_anchor = Some(file_anchor);
        }

        let rec = ((anchor - file_anchor).whole_seconds() / freq_s) as usize;
        let ds = self
            .current_dataset
            .as_ref()
            .expect("dataset loaded above for this file anchor");
        let buffers = slice_record_buffers(ds, temporal.time_dim.as_str(), rec)?;
        self.buffers = buffers;
        self.current_anchor = Some(anchor);
        Ok(Some(self.buffers.clone()))
    }

    /// The cadence schedule over the window: the solver tstops the refresh
    /// callback fires on, as **Unix epoch seconds** (`f64`). Empty for a CONST
    /// loader, or when the loader is open-ended and no window bounds it.
    ///
    /// Each value is `anchor.unix_timestamp()`; a solver integrating in
    /// "seconds since window start" subtracts the window's start epoch.
    pub fn refresh_times(&self) -> Vec<f64> {
        let Some(temporal) = &self.loader.temporal else {
            return Vec::new(); // CONST: no refreshes
        };
        let freq_s = temporal.frequency.whole_seconds();
        if freq_s <= 0 {
            return Vec::new();
        }
        let lower = self.lower_bound(temporal);
        let Some(upper) = self.window.map(|(_, b)| b).or(temporal.end) else {
            return Vec::new(); // unbounded — no enumerable schedule
        };

        // First aligned anchor >= lower (lower is already >= start, so ceil ≥ 0).
        let elapsed = (lower - temporal.start).whole_seconds();
        let steps = (elapsed + freq_s - 1) / freq_s;
        let mut anchor = temporal.start + Duration::seconds(steps * freq_s);

        let mut out = Vec::new();
        while anchor < upper {
            out.push(anchor.unix_timestamp() as f64);
            anchor += Duration::seconds(freq_s);
        }
        out
    }

    /// Warm the cache for every file covering `window`, so an offline/HPC run
    /// never blocks on the network mid-integration (plan §4.5). Offline, this
    /// asserts each file is already present (a miss is [`Error::CacheMiss`]).
    pub fn prefetch(&mut self, window: Window) -> Result<()> {
        match self.loader.temporal.clone() {
            None => {
                self.fetch_blob(&self.resolve_url(OffsetDateTime::UNIX_EPOCH)?)?;
                Ok(())
            }
            Some(t) => {
                let file_s = t.file_period.whole_seconds();
                if file_s <= 0 {
                    return Err(Error::Format {
                        format: self.loader.format.clone(),
                        detail: "loader file_period must be positive".to_string(),
                    });
                }
                let start = if window.0 > t.start {
                    window.0
                } else {
                    t.start
                };
                let mut fa = snap_down(t.start, start, file_s);
                while fa < window.1 {
                    let url = self.resolve_url(fa)?;
                    self.fetch_blob(&url)?;
                    fa += Duration::seconds(file_s);
                }
                Ok(())
            }
        }
    }

    // --- internals ----------------------------------------------------------

    /// The effective lower bound: the window start clamped to the loader epoch.
    fn lower_bound(&self, temporal: &SourceTemporal) -> OffsetDateTime {
        match self.window {
            Some((a, _)) if a > temporal.start => a,
            _ => temporal.start,
        }
    }

    /// Resolve a file URL for an anchor. A template without `[` is a literal URL,
    /// as is a `cds://` URL — that scheme carries a fully-resolved CDS request
    /// whose canonical JSON legitimately contains `[...]` arrays (e.g. `area`,
    /// `variable`) that must NOT be read as `time` format components.
    fn resolve_url(&self, anchor: OffsetDateTime) -> Result<String> {
        let tmpl = &self.loader.url_template;
        if !tmpl.contains('[') || tmpl.starts_with("cds://") {
            return Ok(tmpl.clone());
        }
        let desc =
            time::format_description::parse_borrowed::<2>(tmpl).map_err(|e| Error::BadUrl {
                url: tmpl.clone(),
                detail: format!("invalid url_template: {e}"),
            })?;
        anchor.format(&desc).map_err(|e| Error::BadUrl {
            url: tmpl.clone(),
            detail: format!("formatting url_template at {anchor}: {e}"),
        })
    }

    /// Fetch (or reuse) the blob for `url` through the cache, with the loader's
    /// provenance, mirrors, and auth realm.
    fn fetch_blob(&self, url: &str) -> Result<crate::CachedBlob> {
        let mirrors: Vec<&str> = self.loader.mirrors.iter().map(String::as_str).collect();
        let mut req = FetchRequest::new(url)
            .loader(&self.loader.name)
            .mirrors(&mirrors);
        if let Some(realm) = &self.loader.auth_realm {
            req = req.auth_realm(realm);
        }
        self.cache.fetch(&req)
    }

    /// Fetch + decode a file into a native dataset under the effective `select`
    /// (the per-call override, else the loader's baked [`DataSource::select`]).
    fn read_file(&self, url: String, select: &Selection) -> Result<NativeDataset> {
        // Store-backed readers (e.g. zarr) are handed (cache, base_url, variables,
        // select): a Zarr `url` is a directory-like prefix, not a fetchable blob,
        // so the reader fetches individual objects on demand. Additive + default-
        // off: whole-file readers inherit `store_backed() == false` and read whole.
        if self.reader.store_backed() {
            return match self.store_access {
                // Straight from object storage, nothing written to disk. The
                // `select` travels unchanged, so pushdown is identical.
                StoreAccess::Direct => self.reader.read_store_direct(
                    &url,
                    &self.loader.variables,
                    select,
                    &self.loader.store_options,
                ),
                StoreAccess::Cached => {
                    self.reader
                        .read_store(self.cache.clone(), &url, &self.loader.variables, select)
                }
            };
        }
        // Whole-file reader. The `select` reaches it too: a reader that declares
        // `supports_selection` without being store-backed (the `netcdf` one)
        // honours it at DECODE time — the same blob under the same cache key,
        // with only the requested hyperslab materialised. A reader that declares
        // neither is a clear error, raised BEFORE the fetch — the per-call
        // override is already refused at the call site, and a BAKED
        // `DataSource::select` must be refused here rather than silently ignored
        // (that is what the Julia and Python tracks do, and a selection quietly
        // dropped is a caller reading the whole array believing it is a window).
        if !matches!(select, Selection::All) && !self.reader.supports_selection() {
            return Err(Error::Format {
                format: self.loader.format.clone(),
                detail: format!(
                    "reader for format '{}' does not support select/pushdown",
                    self.loader.format
                ),
            });
        }
        let blob = self.fetch_blob(&url)?;
        self.reader
            .read_native(&blob.path, &self.loader.variables, select)
    }

    /// Ensure the file covering `file_anchor` is decoded into the 2-entry LRU
    /// (`self.files`), so a file-period seam decodes each adjacent file at most
    /// once. On a hit the entry is moved to the tail (most-recently-used).
    ///
    /// **Why this Provider does not push its records into the decode.** The
    /// netcdf reader honours a [`Records`](crate::Records) pushdown in every
    /// track (`spec/registries.md` §2.1) and the Julia Provider uses it, because
    /// that one computes its record from the cadence and keeps no decoded-file
    /// buffer: there a whole-file decode per sample is pure waste. Here this LRU
    /// IS the optimisation — one whole-file decode amortised over every tick in
    /// the file — so pushing down would replace one decode per FILE with one per
    /// TICK. Measured on this reader (release build, warm page cache): an
    /// A1-shaped file (24 records, 47 2-D variables) decodes whole in 37.7 ms,
    /// i.e. 1.57 ms per tick, against 8.3 ms for a narrowed 2-record decode —
    /// 5.3x worse per tick; an A3dyn-shaped file (8 records, 6 3-D variables)
    /// 104.0 ms whole = 13.0 ms per tick against 29.5 ms narrowed, 2.3x worse.
    /// The narrowed decode is genuinely 3.5–4.6x cheaper *per decode*, which is
    /// exactly why it is worth having on the reader and not worth wiring in
    /// front of a buffer that already avoids most of them. A caller that wants it
    /// calls [`Reader::read_native_records`](crate::Reader::read_native_records)
    /// directly; a track is not made slower for symmetry.
    fn ensure_file(&mut self, file_anchor: OffsetDateTime) -> Result<()> {
        if let Some(pos) = self.files.iter().position(|(a, _)| *a == file_anchor) {
            let entry = self.files.remove(pos);
            self.files.push(entry);
            return Ok(());
        }
        let url = self.resolve_url(file_anchor)?;
        let ds = self.read_file(url, &self.loader.select)?;
        self.files.push((file_anchor, ds));
        while self.files.len() > 2 {
            self.files.remove(0); // evict the least-recently-used
        }
        Ok(())
    }

    /// The decoded file for `file_anchor` (must have been [`ensure_file`]d).
    fn file_ref(&self, file_anchor: OffsetDateTime) -> &NativeDataset {
        &self
            .files
            .iter()
            .find(|(a, _)| *a == file_anchor)
            .expect("file ensured before file_ref")
            .1
    }

    /// The `records_per_sample = 2` path of [`refresh`](Self::refresh): produce
    /// the two records bracketing `anchor` (floor + successor), retaining
    /// `time_dim` at length 2, plus a canonical 2-element epoch-seconds `time`
    /// coordinate. Handles the cross-file successor (record 0 of the next file
    /// when the floor is the last record) and the end-of-data clamp (a degenerate
    /// `[last, last]` bracket with equal timestamps — never an error).
    fn refresh_bracket(
        &mut self,
        temporal: &SourceTemporal,
        anchor: OffsetDateTime,
        freq_s: i64,
        file_s: i64,
    ) -> Result<Option<HashMap<String, NativeField>>> {
        let time_dim = temporal.time_dim.clone();

        // rec0 = the floor (at-or-before) record in file0, as the single path does.
        let file0_anchor = snap_down(temporal.start, anchor, file_s);
        self.ensure_file(file0_anchor)?;
        let rec0 = ((anchor - file0_anchor).whole_seconds() / freq_s) as usize;
        let file0_len = time_len(self.file_ref(file0_anchor), &time_dim);

        // Successor rec1: rec0+1 in the same file, else record 0 of the next file.
        // None => no reachable successor (end clamp): degenerate [rec0, rec0].
        let next_anchor = anchor + Duration::seconds(freq_s);
        let has_succ = temporal.end.map_or(true, |end| next_anchor < end);
        let mut succ: Option<(OffsetDateTime, usize)> = None;
        if has_succ {
            if rec0 + 1 < file0_len {
                succ = Some((file0_anchor, rec0 + 1)); // successor in the same file
            } else {
                // Successor is record 0 of the next file; read/decode it (a read
                // failure at the seam is treated as end-of-data, not an error).
                let next_file_anchor = snap_down(temporal.start, next_anchor, file_s);
                if self.ensure_file(next_file_anchor).is_ok() {
                    let r1 = ((next_anchor - next_file_anchor).whole_seconds() / freq_s) as usize;
                    if r1 < time_len(self.file_ref(next_file_anchor), &time_dim) {
                        succ = Some((next_file_anchor, r1));
                    }
                }
            }
        }
        let (succ_anchor, rec1, t1_epoch) = match succ {
            Some((a, r)) => (a, r, next_anchor.unix_timestamp() as f64),
            // Degenerate bracket: hold the last record, equal timestamps.
            None => (file0_anchor, rec0, anchor.unix_timestamp() as f64),
        };
        let t0_epoch = anchor.unix_timestamp() as f64;

        // Stack rec0 (file0) and rec1 (successor file) along the leading time axis.
        let file0 = self.file_ref(file0_anchor);
        let file1 = self.file_ref(succ_anchor);
        let mut buffers = HashMap::with_capacity(file0.variables.len());
        for (name, field) in &file0.variables {
            let is_temporal =
                field.dims.first().map(String::as_str) == Some(time_dim.as_str());
            let stacked = if is_temporal {
                let f1 = file1.variables.get(name).unwrap_or(field);
                stack_two_records(field, rec0, f1, rec1)?
            } else {
                field.clone() // non-temporal variables pass through whole
            };
            buffers.insert(name.clone(), stacked);
        }
        // Carry file0's coords, but replace the time coord with the 2-element
        // epoch-seconds bracket timestamps (the seam the ESS adapter reads).
        let mut coords = file0.coords.clone();
        coords.insert(
            time_dim.clone(),
            Coord {
                field: NativeField {
                    dtype: DType::Float64,
                    dims: vec![time_dim.clone()],
                    shape: vec![2],
                    data: ArrayData::F64(vec![t0_epoch, t1_epoch]),
                    fill_value: None,
                },
                units: Some("seconds since 1970-01-01T00:00:00Z".to_string()),
                calendar: Some("standard".to_string()),
            },
        );

        self.coords = coords;
        self.buffers = buffers;
        self.current_anchor = Some(anchor);
        Ok(Some(self.buffers.clone()))
    }

    /// The `records_per_sample = 2` path of a PER-CALL `select` peek: read the
    /// covering file(s) FRESH under `select` (bypassing the LRU) and stack the two
    /// bracketing records, WITHOUT mutating any cache/buffer state. Handles the
    /// cross-file successor and the end-of-data clamp exactly as
    /// [`refresh_bracket`](Self::refresh_bracket), but returns owned buffers.
    fn bracket_peek(
        &self,
        temporal: &SourceTemporal,
        anchor: OffsetDateTime,
        freq_s: i64,
        file_s: i64,
        select: &Selection,
    ) -> Result<HashMap<String, NativeField>> {
        let time_dim = temporal.time_dim.as_str();
        let file0_anchor = snap_down(temporal.start, anchor, file_s);
        let file0 = self.read_file(self.resolve_url(file0_anchor)?, select)?;
        let rec0 = ((anchor - file0_anchor).whole_seconds() / freq_s) as usize;
        let file0_len = time_len(&file0, time_dim);

        let next_anchor = anchor + Duration::seconds(freq_s);
        let has_succ = temporal.end.map_or(true, |end| next_anchor < end);

        // (successor_file_or_none_meaning_file0, rec1). A `None` successor file with
        // rec1 == rec0 is the degenerate end-of-data clamp `[rec0, rec0]`.
        let (succ_file, rec1) = if has_succ && rec0 + 1 < file0_len {
            (None, rec0 + 1) // successor in file0
        } else if has_succ {
            let nfa = snap_down(temporal.start, next_anchor, file_s);
            match self.resolve_url(nfa).and_then(|u| self.read_file(u, select)) {
                Ok(f1) => {
                    let r1 = ((next_anchor - nfa).whole_seconds() / freq_s) as usize;
                    if r1 < time_len(&f1, time_dim) {
                        (Some(f1), r1)
                    } else {
                        (None, rec0) // no valid successor record — clamp
                    }
                }
                Err(_) => (None, rec0), // seam read failed — clamp
            }
        } else {
            (None, rec0) // end-of-data — clamp
        };

        let file1_ref: &NativeDataset = succ_file.as_ref().unwrap_or(&file0);
        let mut buffers = HashMap::with_capacity(file0.variables.len());
        for (name, field) in &file0.variables {
            let is_temporal = field.dims.first().map(String::as_str) == Some(time_dim);
            let stacked = if is_temporal {
                let f1 = file1_ref.variables.get(name).unwrap_or(field);
                stack_two_records(field, rec0, f1, rec1)?
            } else {
                field.clone()
            };
            buffers.insert(name.clone(), stacked);
        }
        Ok(buffers)
    }
}

/// Slice cadence record `rec` from every variable carrying `time_dim` as its
/// leading axis; non-temporal variables pass through whole. Shared by the plain
/// single-record refresh path and the per-call `select` peek.
fn slice_record_buffers(
    ds: &NativeDataset,
    time_dim: &str,
    rec: usize,
) -> Result<HashMap<String, NativeField>> {
    let mut buffers = HashMap::with_capacity(ds.variables.len());
    for (name, field) in &ds.variables {
        let is_temporal = field.dims.first().map(String::as_str) == Some(time_dim);
        let slice = if is_temporal {
            field.select_leading(rec)?
        } else {
            field.clone()
        };
        buffers.insert(name.clone(), slice);
    }
    Ok(buffers)
}

/// Length of `ds` along `time_dim`: from the time coordinate if it carries it,
/// else the first variable that does, else 0.
fn time_len(ds: &NativeDataset, time_dim: &str) -> usize {
    if let Some(coord) = ds.coords.get(time_dim) {
        if let Some(pos) = coord.field.dims.iter().position(|d| d == time_dim) {
            return coord.field.shape[pos];
        }
    }
    for field in ds.variables.values() {
        if let Some(pos) = field.dims.iter().position(|d| d == time_dim) {
            return field.shape[pos];
        }
    }
    0
}

/// Stack record `rec0` of `f0` and record `rec1` of `f1` along the leading axis,
/// keeping that axis (the time dim) at length 2. `f0`/`f1` are the same variable
/// across (possibly) adjacent files, so they share dtype/trailing shape.
fn stack_two_records(
    f0: &NativeField,
    rec0: usize,
    f1: &NativeField,
    rec1: usize,
) -> Result<NativeField> {
    let a = f0.select_leading(rec0)?; // drops the leading dim -> one record
    let b = f1.select_leading(rec1)?;
    let data = concat_array(&a.data, &b.data)?;
    let mut dims = Vec::with_capacity(a.dims.len() + 1);
    dims.push(f0.dims[0].clone());
    dims.extend(a.dims.iter().cloned());
    let mut shape = Vec::with_capacity(a.shape.len() + 1);
    shape.push(2);
    shape.extend(a.shape.iter().cloned());
    Ok(NativeField {
        dtype: f0.dtype,
        dims,
        shape,
        data,
        fill_value: f0.fill_value,
    })
}

/// Concatenate two same-dtype native arrays end-to-end (used to stack two records
/// into a size-2 leading axis).
fn concat_array(a: &ArrayData, b: &ArrayData) -> Result<ArrayData> {
    Ok(match (a, b) {
        (ArrayData::F64(x), ArrayData::F64(y)) => {
            let mut v = x.clone();
            v.extend_from_slice(y);
            ArrayData::F64(v)
        }
        (ArrayData::I64(x), ArrayData::I64(y)) => {
            let mut v = x.clone();
            v.extend_from_slice(y);
            ArrayData::I64(v)
        }
        (ArrayData::I32(x), ArrayData::I32(y)) => {
            let mut v = x.clone();
            v.extend_from_slice(y);
            ArrayData::I32(v)
        }
        (ArrayData::Str(x), ArrayData::Str(y)) => {
            let mut v = x.clone();
            v.extend_from_slice(y);
            ArrayData::Str(v)
        }
        (ArrayData::Bool(x), ArrayData::Bool(y)) => {
            let mut v = x.clone();
            v.extend_from_slice(y);
            ArrayData::Bool(v)
        }
        _ => {
            return Err(Error::Format {
                format: "native".to_string(),
                detail: "cannot stack bracket records of differing dtypes".to_string(),
            })
        }
    })
}

/// `start + floor((t - start) / step) * step` — snap `t` down to the aligned
/// anchor at or before it. `step_s` must be positive.
fn snap_down(start: OffsetDateTime, t: OffsetDateTime, step_s: i64) -> OffsetDateTime {
    let elapsed = (t - start).whole_seconds();
    let steps = elapsed.div_euclid(step_s); // floors toward -∞
    start + Duration::seconds(steps * step_s)
}

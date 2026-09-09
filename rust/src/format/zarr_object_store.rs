//! Direct (non-cache) Zarr read/write over Apache Arrow `object_store`
//! (feature `object-store`).
//!
//! This is the object-storage path that is **backed by `object_store`'s** mature
//! clients rather than any hand-rolled vendor code. A store URL is dispatched by
//! [`object_store::parse_url_opts`], rooted at its path prefix via
//! [`PrefixStore`], wrapped by the [`zarrs_object_store`] adapter (async), and
//! bridged into the crate's **sync** Zarr reader/writer through zarrs'
//! [`AsyncToSyncStorageAdapter`] driven by a `tokio` runtime. The decode/encode
//! logic itself is shared with the cache-backed path
//! ([`super::zarr::read_arrays`] / [`super::zarr_write::write_all_to_store`]) —
//! only the storage backend differs.
//!
//! **No provider is hard-coded.** A store's home is named by its URL scheme and
//! nothing else; adding a provider is a feature flag on `object_store`, not a new
//! code path here.
//!
//! # Which schemes are active
//!
//! With the `object-store` feature on, `object_store` is built with its `aws`,
//! `gcp`, `azure` and `http` features (plus its default `fs`), so **every** scheme
//! `object_store` recognises resolves:
//!
//! | URL | Backend | Gated on |
//! |---|---|---|
//! | `file://…` | local filesystem | `object_store/fs` (its default) |
//! | `memory://…` | in-process, for tests | always — no feature |
//! | `s3://bucket/…`, `s3a://bucket/…` | S3 **and every S3-compatible endpoint** | `object_store/aws` |
//! | `gs://bucket/…` | Google Cloud Storage | `object_store/gcp` |
//! | `az://`, `adl://`, `azure://`, `abfs://`, `abfss://` | Azure Blob / ADLS Gen2 | `object_store/azure` |
//! | `http://`, `https://` | plain HTTP(S) / WebDAV range reads | `object_store/http` |
//!
//! `https://` URLs are additionally re-dispatched *by host* inside
//! `object_store`: `*.amazonaws.com` and `*.r2.cloudflarestorage.com` resolve to
//! the S3 client, `*.blob.core.windows.net`/`*.dfs.core.windows.net` to Azure, and
//! anything else to the plain HTTP store.
//!
//! Any other scheme (`ftp://`, `opfs://`, …) is rejected by
//! [`resolve_backend`] with a `no object_store backend for …` error — a loud
//! failure, never a silent empty read.
//!
//! # Supplying an endpoint override (R2 / MinIO / Backblaze B2 / Ceph)
//!
//! S3-compatible endpoints matter more than new clouds: Cloudflare R2, MinIO,
//! Backblaze B2 and Ceph are all reached through the **`s3://` path plus a
//! configurable endpoint URL**, not through provider-specific code. The endpoint
//! is a config value:
//!
//! ```no_run
//! use earthsciio::{write_zarr_object_store_with_options, OutputSchema};
//! # fn demo(schema: &OutputSchema) -> Result<(), earthsciio::Error> {
//! // Cloudflare R2 — the vendor surface is one option, nothing else.
//! let opts = [
//!     ("endpoint".to_string(), "https://<account>.r2.cloudflarestorage.com".to_string()),
//!     ("region".to_string(), "auto".to_string()),
//! ];
//! write_zarr_object_store_with_options("s3://my-bucket/datasets/run-1.zarr", schema, &opts)?;
//! # Ok(())
//! # }
//! ```
//!
//! Option keys are `object_store`'s own (`endpoint` / `aws_endpoint_url`,
//! `region`, `aws_access_key_id`, `aws_secret_access_key`, `aws_allow_http`,
//! `aws_virtual_hosted_style_request`, `google_service_account_key`,
//! `azure_storage_account_name`, …); unrecognised keys are ignored, so one option
//! list can be handed to any scheme.
//!
//! The zero-option entry points ([`read_zarr_object_store`] /
//! [`write_zarr_object_store`]) default to [`store_options_from_env`], which
//! harvests `AWS_*`, `GOOGLE_*` and `AZURE_*` from the process environment — the
//! same variables each `object_store` builder's own `from_env` reads. So
//! `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and
//! `AWS_REGION` configure an S3-compatible target with no code change at all.
//!
//! A **read** resolves its options through [`read_store_options`] instead, which
//! is that harvest plus one judgement the harvest cannot make on its own: an
//! `AWS_*` variable the platform injected is not the caller saying this read
//! should be signed. See its docs — that distinction is the difference between
//! reading a public bucket and getting a 403 from a role that was never meant
//! for the read.
//!
//! (Before this seam existed the module called `parse_url`, which builds an
//! `AmazonS3Builder::new()` — a builder that reads **no** environment at all and
//! has no way to express an endpoint, so `s3://` only ever worked against real
//! AWS from an instance with an IMDS role.)
//!
//! The content-addressed cache remains the path used by the [`crate::Provider`];
//! this module is for callers that want to read/write a store directly.

use std::sync::Arc;

use object_store::path::Path as ObjectPath;
use object_store::prefix::PrefixStore;
use object_store::ObjectStore;
use url::Url;
use zarrs::storage::storage_adapter::async_to_sync::{
    AsyncToSyncBlockOn, AsyncToSyncStorageAdapter,
};
use zarrs_object_store::AsyncObjectStore;

use super::zarr_store::SanitizedV2;
use super::{AxisSelect, NativeDataset, OutputSchema, Selection};
use crate::error::{Error, Result};
// The S3 option-key vocabulary these decisions are made in, and the per-bucket
// statement that now sits between the environment and the caller. Both live in
// `crate::s3_config` because the `s3://` transport has to reach the same
// conclusion about the same bucket, and two copies of this vocabulary would
// differ silently.
use crate::s3_config::{
    has_any, options_for_url, S3_CREDENTIAL_KEYS, S3_ENDPOINT_KEYS, S3_SIGNING_KEYS,
    S3_STATIC_KEY_KEYS,
};

/// Environment prefixes harvested by [`store_options_from_env`], one per cloud
/// backend, matching what each `object_store` builder's own `from_env` reads.
const ENV_PREFIXES: [&str; 3] = ["AWS_", "GOOGLE_", "AZURE_"];

fn os_err(detail: impl Into<String>) -> Error {
    Error::Format {
        format: "zarr".to_string(),
        detail: detail.into(),
    }
}

/// Backend configuration for a store URL, harvested from the process
/// environment: every `AWS_*`, `GOOGLE_*` and `AZURE_*` variable, lower-cased
/// into `object_store`'s option keys.
///
/// This is what the zero-option entry points use, so `AWS_ENDPOINT_URL` (or
/// `AWS_ENDPOINT`) is all an S3-compatible provider — R2, MinIO, Backblaze B2,
/// Ceph — needs. Keys `object_store` does not recognise for the dispatched
/// scheme are ignored rather than rejected.
///
/// Values are passed straight through to `object_store` and are never logged by
/// this crate.
#[must_use]
pub fn store_options_from_env() -> Vec<(String, String)> {
    std::env::vars()
        .filter(|(k, _)| ENV_PREFIXES.iter().any(|p| k.starts_with(p)))
        .map(|(k, v)| (k.to_ascii_lowercase(), v))
        .collect()
}

/// Is this URL dispatched to the S3 client by scheme?
fn is_s3_url(url_str: &str) -> bool {
    url_str.starts_with("s3://") || url_str.starts_with("s3a://")
}

/// Backend options for a **read** of `url`: the process environment
/// ([`store_options_from_env`]), then whatever
/// [`EARTHSCI_S3_BUCKET_OPTIONS`](crate::BUCKET_OPTIONS_ENV) states about this
/// URL's bucket, then the caller's `explicit` options on top — plus the
/// anonymous-S3 default when nothing among the three has stated that this read
/// should be signed.
///
/// # Why the environment does not get to decide signing
///
/// [`store_options_from_env`] harvests variables; it cannot tell *"the operator
/// configured credentials for this read"* from *"the platform injected a role
/// for something else entirely"*. On a container platform it is reliably the
/// latter. ECS/Fargate sets `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI` in every
/// task that carries a job role, EKS sets `AWS_ROLE_ARN` +
/// `AWS_WEB_IDENTITY_TOKEN_FILE` in every pod with a service-account role, and a
/// runner holds `AWS_ACCESS_KEY_ID` because it *writes its output* somewhere —
/// none of them says anything about the public bucket the document it was handed
/// wants to read. Treated as intent, they leave signing on, and
/// `s3://inmap-model/…` is then signed with a role that deliberately has no
/// `s3:GetObject`: **403**, in production only, because a laptop has none of
/// those variables set.
///
/// So intent has to be *stated*, and exactly four things state it:
///
/// * `aws_skip_signature`, either polarity, from **any** source
///   ([`S3_SIGNING_KEYS`]). Nothing injects it, so `=false` is an unambiguous
///   "sign, I mean it" and `=true` an unambiguous "do not".
/// * A credential option the **caller passed** — [`crate::DataSource::store_options`],
///   or the `options` argument of a `*_with_options` entry point. That is a
///   document or a program describing this read, not an ambient variable.
/// * A credential **named against this bucket** in
///   [`EARTHSCI_S3_BUCKET_OPTIONS`](crate::BUCKET_OPTIONS_ENV). A variable that
///   names the bucket is about the bucket, which is the whole difference between
///   it and the ambient harvest above. Naming a bucket only to pin its region
///   states nothing and leaves a public read public — see
///   [`crate::stated_signing`].
/// * Static keys in the environment **next to an endpoint override**
///   ([`S3_ENDPOINT_KEYS`]): the R2 / MinIO / Backblaze B2 / Ceph deployment.
///   An endpoint is never injected either, so credentials beside one belong to
///   the same deliberate configuration. This exception cannot reopen the case
///   above, because a *process-wide* endpoint override already sends every
///   `s3://` read to that one deployment — a public bucket somewhere else is
///   unreachable through it whatever the signing decision is, so there is no
///   configuration in which this rule is what breaks the read.
///
/// Anything else reads anonymously — the meaning `s3://` already has on the
/// cache-backed path, which is the point of the whole exercise. Credential
/// options are still passed through, so a caller who states
/// `aws_skip_signature=false` authenticates with whatever the environment
/// provides, ambient role included.
///
/// This is a **read** default and the write entry points do not apply it: a
/// process that reads a public store and writes its output to a private one has
/// to sign the write, so the two cannot share one process-wide switch. That is
/// also why `AWS_SKIP_SIGNATURE=true` is not the deployment fix it looks like.
///
/// # What this changed, for anyone it changed it for
///
/// `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` in the environment used to be
/// enough to sign a **read**, and no longer is. If a deployment was relying on
/// that to reach its own private bucket through this path, the two ways to say
/// so are `AWS_SKIP_SIGNATURE=false` in the same environment — one variable,
/// no code — or `aws_skip_signature=false` on the loader
/// ([`crate::DataSource::store_options`]) when only some of its loaders are
/// private. Both keep authenticating with exactly the credentials the
/// environment already provides; only the *decision* moved.
///
/// A read of a private bucket that says neither now fails as `403`/`AccessDenied`
/// on the first object rather than succeeding, which is a loud failure at the
/// first attempt and the reason this is a default worth having.
///
/// # One boundary, deliberately
///
/// This looks at the URL's **scheme**, so it covers `s3://` and `s3a://` and not
/// the `https://…amazonaws.com` / `…r2.cloudflarestorage.com` spellings that
/// `object_store` re-dispatches to the S3 client by *host* (see the module
/// docs). Reproducing that host table here is exactly the hard-coded vendor
/// knowledge this module is built to avoid, and getting it subtly out of step
/// with `object_store`'s own would be worse than not having it: prefer `s3://`
/// for a store that should read anonymously.
#[must_use]
pub fn read_store_options(url: &str, explicit: &[(String, String)]) -> Vec<(String, String)> {
    // Ambient first, then the bucket somebody named, then this caller: each
    // layer is more deliberate and more specific than the one under it.
    let per_bucket = options_for_url(url);
    let mut merged = store_options_from_env();
    for (k, v) in per_bucket.iter().chain(explicit.iter()) {
        merged.retain(|(mk, _)| mk != k);
        merged.push((k.clone(), v.clone()));
    }
    if !is_s3_url(url) || has_any(&merged, &S3_SIGNING_KEYS) {
        return merged;
    }
    let stated_for_this_read =
        has_any(explicit, &S3_CREDENTIAL_KEYS) || has_any(&per_bucket, &S3_CREDENTIAL_KEYS);
    let configured_s3_compatible_endpoint =
        has_any(&merged, &S3_ENDPOINT_KEYS) && has_any(&merged, &S3_STATIC_KEY_KEYS);
    if !stated_for_this_read && !configured_s3_compatible_endpoint {
        merged.push(("aws_skip_signature".to_string(), "true".to_string()));
    }
    merged
}

/// Give an `s3://` URL the SAME meaning it has on the cache-backed path:
/// **anonymous, regional, no credentials**.
///
/// This crate's own `s3://` transport (`transport::s3`) is documented as "a
/// public bucket needs no AWS SDK, no SigV4, no credentials", and resolves the
/// region as `$EARTHSCI_S3_REGION` → `$AWS_REGION` → `us-east-2`. `object_store`
/// makes the opposite default choice: it signs, and it has no region fallback.
/// Left alone, the same `s3://inmap-model/isrm_v1.2.1.zarr/` URL would read fine
/// through the cache and fail on a credential lookup without it — a silent
/// divergence between two paths that are supposed to differ only in whether
/// bytes touch the disk.
///
/// So: for `s3://`/`s3a://`, when `options` state no way to authenticate
/// ([`S3_SIGNING_KEYS`], [`S3_CREDENTIAL_KEYS`]), signing is switched off; and a
/// missing region is filled from the same chain the cached transport uses.
/// Anything stated is left exactly as given — including
/// `aws_skip_signature=false`, which is how a caller asks for signed access
/// explicitly.
///
/// `options` here are read as **the caller's own**, which is what the
/// `*_with_options` entry points document them to be. Options that came from the
/// environment have already been through [`read_store_options`], whose decision
/// arrives as an explicit `aws_skip_signature` and is left alone below.
fn apply_s3_defaults(url_str: &str, options: &mut Vec<(String, String)>) {
    if !is_s3_url(url_str) {
        return;
    }
    if !has_any(options, &S3_SIGNING_KEYS) && !has_any(options, &S3_CREDENTIAL_KEYS) {
        options.push(("aws_skip_signature".to_string(), "true".to_string()));
    }
    if !has_any(options, &["region", "aws_region"]) {
        options.push((
            "aws_region".to_string(),
            crate::transport::resolve_region(None),
        ));
    }
}

/// A `tokio` `block_on` bridge so an async `object_store` store can be used from
/// the crate's synchronous Zarr API (via zarrs' [`AsyncToSyncStorageAdapter`]).
struct TokioBlockOn(tokio::runtime::Handle);

impl AsyncToSyncBlockOn for TokioBlockOn {
    fn block_on<F: core::future::Future>(&self, future: F) -> F::Output {
        self.0.block_on(future)
    }
}

/// The sync storage type: an object_store store, prefix-rooted, adapted to sync.
type SyncObjectStore =
    AsyncToSyncStorageAdapter<AsyncObjectStore<PrefixStore<Box<dyn ObjectStore>>>, TokioBlockOn>;

/// Build a multi-thread `tokio` runtime for driving `object_store` I/O.
///
/// `pub(crate)` because the signed `s3://` transport
/// ([`crate::transport::S3Transport`]) drives the same async client from the
/// same synchronous call stack, and two runtimes built two ways would be two
/// sets of behaviour to keep in step.
pub(crate) fn runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| os_err(format!("build tokio runtime for object_store: {e}")))
}

/// Dispatch a store URL to an `object_store` backend by **scheme alone**,
/// returning the store and the path prefix inside it.
///
/// This is the one place a provider is chosen. `options` are `object_store`'s
/// own config keys (see the module docs); an endpoint override travels here.
pub(crate) fn resolve_backend(
    url_str: &str,
    options: &[(String, String)],
) -> Result<(Box<dyn ObjectStore>, ObjectPath)> {
    let url =
        Url::parse(url_str).map_err(|e| os_err(format!("invalid store URL '{url_str}': {e}")))?;
    object_store::parse_url_opts(&url, options.iter().map(|(k, v)| (k.as_str(), v.as_str())))
        .map_err(|e| os_err(format!("no object_store backend for '{url_str}': {e}")))
}

/// Resolve a store URL to a sync-usable `zarrs` storage rooted at the store.
fn build_sync_store(
    url_str: &str,
    options: &[(String, String)],
    handle: tokio::runtime::Handle,
) -> Result<Arc<SyncObjectStore>> {
    let (store, prefix) = resolve_backend(url_str, options)?;
    let prefixed = PrefixStore::new(store, prefix);
    let async_store = Arc::new(AsyncObjectStore::new(prefixed));
    Ok(Arc::new(AsyncToSyncStorageAdapter::new(
        async_store,
        TokioBlockOn(handle),
    )))
}

/// The read-side store: [`build_sync_store`] plus the two things a *read* of a
/// foreign store needs and a write of our own does not — anonymous-S3 defaults
/// ([`apply_s3_defaults`]) and Zarr v2 `.zarray` normalization
/// ([`SanitizedV2`]). Together these make a direct read of a given URL behave
/// like the cache-backed read of the same URL.
fn build_read_store(
    url_str: &str,
    options: &[(String, String)],
    handle: tokio::runtime::Handle,
) -> Result<Arc<SanitizedV2<SyncObjectStore>>> {
    let mut opts = options.to_vec();
    apply_s3_defaults(url_str, &mut opts);
    let inner = build_sync_store(url_str, &opts, handle)?;
    Ok(Arc::new(SanitizedV2::new(inner)))
}

/// The full (dims-order) shape of array `var` in the store at `url`, read from
/// ONLY that array's metadata object — never a chunk.
///
/// The direct-read twin of [`super::Reader::array_shape`]: the same
/// honour/refuse probe a caller makes before pushing a projection down, with no
/// cache and so no bytes written to disk.
///
/// `options` are the caller's own, with the same caveat as
/// [`read_zarr_object_store_with_options`]: pass a loader's options or
/// [`read_store_options`]'s result, not a raw environment harvest.
///
/// # Errors
/// Returns [`Error::Format`] if the URL has no `object_store` backend or the
/// array's metadata cannot be opened.
pub fn array_shape_object_store(
    url: &str,
    var: &str,
    options: &[(String, String)],
) -> Result<Vec<usize>> {
    let rt = runtime()?;
    let store = build_read_store(url, options, rt.handle().clone())?;
    let array = super::zarr::open_array(store, var)?;
    Ok(array.shape().iter().map(|&s| s as usize).collect())
}

/// Read `variables` from a Zarr store at `url` directly through `object_store`,
/// applying the orthogonal `select` lazily.
///
/// Backend options default to [`read_store_options`] — the environment harvest
/// plus the anonymous-S3 default, so a public bucket reads with no
/// configuration whatever the platform injected. Use
/// [`read_zarr_object_store_with_options`] to pass them explicitly. See the
/// module docs for the active schemes.
///
/// # Errors
/// Returns [`Error::Format`] if the URL has no `object_store` backend, the store
/// cannot be opened, or a decode fails.
pub fn read_zarr_object_store(
    url: &str,
    variables: &[String],
    select: &Selection,
) -> Result<NativeDataset> {
    read_zarr_object_store_with_options(url, variables, select, &read_store_options(url, &[]))
}

/// [`read_zarr_object_store`] with explicit backend `options` (`object_store`
/// config keys — `endpoint`, `region`, credentials, …) instead of the
/// environment-derived defaults.
///
/// `options` are read as **the caller's own** — a credential among them means
/// "sign this read". So do not hand it `store_options_from_env()`: that hands
/// the environment the caller's authority and is precisely how a public bucket
/// ends up signed with a role a container platform injected for something else.
/// [`read_store_options`] is the function that merges the two while keeping them
/// distinguishable, and it is what [`read_zarr_object_store`] and the
/// [`crate::Provider`] direct path both use.
///
/// # Errors
/// Returns [`Error::Format`] if the URL has no `object_store` backend, the store
/// cannot be opened, or a decode fails.
pub fn read_zarr_object_store_with_options(
    url: &str,
    variables: &[String],
    select: &Selection,
    options: &[(String, String)],
) -> Result<NativeDataset> {
    if variables.is_empty() {
        return Err(os_err(
            "object-store zarr read requires an explicit list of variables",
        ));
    }
    // The runtime must outlive the synchronous decode (the adapter blocks on it).
    let rt = runtime()?;
    let store = build_read_store(url, options, rt.handle().clone())?;
    let axes: Option<&[AxisSelect]> = match select {
        Selection::Orthogonal(a) => Some(a.as_slice()),
        _ => None,
    };
    // Selection pushdown is NOT re-implemented here: `read_arrays` is the same
    // function the cache-backed reader calls, so `Selection::Orthogonal` fetches
    // exactly the intersecting chunk objects on this path too.
    super::zarr::read_arrays(store, variables, axes)
}

/// Write a sharded Zarr **v3** store to `url` through `object_store`, following
/// `schema` (same layout as [`super::write_zarr_v3`]).
///
/// Backend options default to [`store_options_from_env`]; use
/// [`write_zarr_object_store_with_options`] to pass them explicitly.
///
/// # Errors
/// Returns [`Error::Format`] on schema inconsistency, a missing backend, or a
/// store write error.
pub fn write_zarr_object_store(url: &str, schema: &OutputSchema) -> Result<()> {
    write_zarr_object_store_with_options(url, schema, &store_options_from_env())
}

/// [`write_zarr_object_store`] with explicit backend `options` (`object_store`
/// config keys — `endpoint`, `region`, credentials, …) instead of the
/// environment-derived defaults.
///
/// # Errors
/// Returns [`Error::Format`] on schema inconsistency, a missing backend, or a
/// store write error.
pub fn write_zarr_object_store_with_options(
    url: &str,
    schema: &OutputSchema,
    options: &[(String, String)],
) -> Result<()> {
    let rt = runtime()?;
    let store = build_sync_store(url, options, rt.handle().clone())?;
    super::zarr_write::write_all_to_store(store, url, schema)
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::ObjectStoreExt;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// Options that keep a backend build hermetic: no credential lookup, no
    /// metadata-server probe, no ambient config file. These are *not* secrets —
    /// they switch signing off, which is what a dispatch test wants.
    fn hermetic_opts() -> Vec<(String, String)> {
        [
            ("aws_skip_signature", "true"),
            ("aws_region", "auto"),
            ("google_skip_signature", "true"),
            ("azure_skip_signature", "true"),
            ("azure_storage_account_name", "dispatchtest"),
        ]
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
    }

    /// Every scheme the module claims is active must actually dispatch to a
    /// backend, with the in-store path prefix parsed off the URL. Pure URL
    /// parsing + client construction: no network, no credentials.
    #[test]
    fn every_claimed_scheme_dispatches_to_a_backend() {
        let opts = hermetic_opts();
        let cases: &[(&str, &str)] = &[
            ("file:///tmp/some/store.zarr", "tmp/some/store.zarr"),
            ("memory:///datasets/run-1.zarr", "datasets/run-1.zarr"),
            ("s3://bucket/datasets/run-1.zarr", "datasets/run-1.zarr"),
            ("s3a://bucket/datasets/run-1.zarr", "datasets/run-1.zarr"),
            ("gs://bucket/datasets/run-1.zarr", "datasets/run-1.zarr"),
            ("az://container/datasets/run-1.zarr", "datasets/run-1.zarr"),
            (
                "abfs://container/datasets/run-1.zarr",
                "datasets/run-1.zarr",
            ),
            (
                "abfss://container/datasets/run-1.zarr",
                "datasets/run-1.zarr",
            ),
            ("adl://container/datasets/run-1.zarr", "datasets/run-1.zarr"),
            (
                "azure://container/datasets/run-1.zarr",
                "datasets/run-1.zarr",
            ),
            (
                "http://example.org/datasets/run-1.zarr",
                "datasets/run-1.zarr",
            ),
            (
                "https://example.org/datasets/run-1.zarr",
                "datasets/run-1.zarr",
            ),
            // https is re-dispatched by host inside object_store: these are the
            // S3 and Azure clients, not the plain HTTP one.
            (
                "https://bucket.s3.us-east-1.amazonaws.com/datasets/run-1.zarr",
                "datasets/run-1.zarr",
            ),
            (
                "https://acct.r2.cloudflarestorage.com/bucket/datasets/run-1.zarr",
                "datasets/run-1.zarr",
            ),
            (
                "https://acct.blob.core.windows.net/container/datasets/run-1.zarr",
                "datasets/run-1.zarr",
            ),
        ];
        for (url, want_prefix) in cases {
            let (_store, prefix) = resolve_backend(url, &opts)
                .unwrap_or_else(|e| panic!("scheme dispatch failed for {url}: {e}"));
            assert_eq!(prefix.as_ref(), *want_prefix, "path prefix for {url}");
        }
    }

    /// A scheme with no backend fails loudly and names the URL — it must never
    /// fall through to some default store.
    #[test]
    fn an_unsupported_scheme_is_a_named_error() {
        for url in ["ftp://host/store.zarr", "opfs:///datasets/run-1.zarr"] {
            let err = resolve_backend(url, &[]).expect_err("should have no backend");
            let msg = err.to_string();
            assert!(msg.contains(url), "error should name the URL: {msg}");
        }
    }

    /// The whole of S3-compatible provider support: a configurable endpoint URL.
    /// Points the `s3://` path at a loopback listener and asserts the request
    /// actually goes there, path-style, with the bucket and key intact — which is
    /// what makes Cloudflare R2 / MinIO / Backblaze B2 / Ceph reachable without a
    /// vendor-specific code path. Loopback only: no network, no credentials
    /// (signing is switched off).
    #[test]
    fn an_s3_endpoint_override_redirects_the_request() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().expect("accept");
            let mut buf = [0u8; 4096];
            let n = sock.read(&mut buf).expect("read request");
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            sock.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nETag: \"x\"\r\n\
                  Last-Modified: Thu, 01 Jan 1970 00:00:00 GMT\r\n\r\nhello",
            )
            .expect("write response");
            let _ = sock.flush();
            req
        });

        let endpoint = format!("http://127.0.0.1:{port}");
        let opts: Vec<(String, String)> = [
            ("endpoint", endpoint.as_str()),
            ("region", "auto"),
            ("aws_allow_http", "true"),
            ("aws_skip_signature", "true"),
        ]
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();

        let (store, prefix) =
            resolve_backend("s3://my-bucket/datasets/run-1.zarr", &opts).expect("s3 dispatch");
        assert_eq!(prefix.as_ref(), "datasets/run-1.zarr");

        let rt = runtime().expect("runtime");
        let body = rt
            .block_on(async move {
                let key = prefix.join("zarr.json");
                store.get(&key).await?.bytes().await
            })
            .expect("get through the overridden endpoint");
        assert_eq!(&body[..], b"hello");

        let req = server.join().expect("server thread");
        let first_line = req.lines().next().unwrap_or_default().to_string();
        assert!(
            first_line.starts_with("GET /my-bucket/datasets/run-1.zarr/zarr.json "),
            "request should be path-style against the custom endpoint, got: {first_line}"
        );
        assert!(
            req.to_ascii_lowercase()
                .contains(&format!("host: 127.0.0.1:{port}")),
            "request should be addressed to the override host"
        );
    }

    /// An `s3://` URL means the same thing here as it does on the cache-backed
    /// path: anonymous, regional, no credentials. That is what makes
    /// `s3://inmap-model/...` — a public bucket read by a runner that
    /// deliberately holds no `s3:GetObject` — work through both paths with no
    /// configuration at all.
    #[test]
    fn an_s3_read_defaults_to_unsigned_and_regional() {
        let mut opts = Vec::new();
        apply_s3_defaults("s3://inmap-model/isrm_v1.2.1.zarr/", &mut opts);
        assert_eq!(
            opts.iter().find(|(k, _)| k == "aws_skip_signature"),
            Some(&("aws_skip_signature".to_string(), "true".to_string())),
            "an unauthenticated s3:// read must not try to sign"
        );
        assert!(
            opts.iter().any(|(k, v)| k == "aws_region" && !v.is_empty()),
            "object_store has no region fallback; the cached transport's chain supplies one"
        );
    }

    /// Whatever the caller stated is left alone — including asking for signed
    /// access explicitly, and including a non-S3 scheme, which is untouched.
    #[test]
    fn stated_credentials_and_other_schemes_are_left_alone() {
        for stated in [
            ("aws_access_key_id", "AKIAEXAMPLE"),
            ("aws_skip_signature", "false"),
        ] {
            let mut opts = vec![(stated.0.to_string(), stated.1.to_string())];
            apply_s3_defaults("s3://private-bucket/store.zarr", &mut opts);
            assert_eq!(
                opts.iter()
                    .filter(|(k, _)| k == "aws_skip_signature")
                    .count(),
                usize::from(stated.0 == "aws_skip_signature"),
                "stating {} must not add a signing default",
                stated.0
            );
        }

        let mut opts = Vec::new();
        apply_s3_defaults("https://example.org/store.zarr", &mut opts);
        assert!(opts.is_empty(), "non-s3 schemes get no S3 defaults");
    }

    /// `store_options_from_env` picks up exactly the cloud-config prefixes and
    /// lower-cases them into `object_store` option keys.
    #[test]
    fn env_options_are_prefix_scoped_and_lowercased() {
        // A synthetic, non-secret value; asserted on the key, not the value.
        let _env = EnvScope::new(&[
            ("AWS_ENDPOINT_URL", "http://127.0.0.1:9000"),
            ("EARTHSCIIO_NOT_A_STORE_OPTION", "ignored"),
        ]);
        let opts = store_options_from_env();
        assert!(opts.iter().any(|(k, _)| k == "aws_endpoint_url"));
        assert!(
            !opts.iter().any(|(k, _)| k.contains("earthsciio")),
            "only AWS_/GOOGLE_/AZURE_ variables are harvested"
        );
    }

    // -----------------------------------------------------------------------
    // Who gets to say a read is signed.
    //
    // The environment states a great deal that has nothing to do with the store
    // a document names: a container platform injects a role for the task, and a
    // runner carries write credentials for its own output bucket. Harvested and
    // read as intent, either one signs a request for somebody else's PUBLIC
    // bucket with an identity that has no business reading it, and S3 answers
    // 403 — in the deployment only, never on the laptop where the variables do
    // not exist. These pin which of the two it is.
    // -----------------------------------------------------------------------

    use crate::test_env::EnvScope;

    fn value<'a>(options: &'a [(String, String)], key: &str) -> Option<&'a str> {
        options
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// The options a direct read is actually dispatched with: the environment,
    /// the caller's own on top, and the defaults [`build_read_store`] applies.
    /// Asserted at this level rather than on either half, because the question
    /// is only ever what reaches `object_store`.
    fn resolved_read_options(url: &str, explicit: &[(String, String)]) -> Vec<(String, String)> {
        let mut options = read_store_options(url, explicit);
        apply_s3_defaults(url, &mut options);
        options
    }

    const PUBLIC_STORE: &str = "s3://inmap-model/isrm_v1.2.1.zarr/";

    /// A synthetic ECS task-role pointer. The value is never dereferenced by
    /// any test here — its mere presence is what used to change the decision.
    const ECS_ROLE_URI: &str = "/v2/credentials/00000000-0000-0000-0000-000000000000";

    /// **The bug.** ECS/Fargate injects `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI`
    /// into every task that carries a `jobRoleArn`. Counted as the caller having
    /// "a way to authenticate", it leaves signing on, and the public
    /// `inmap-model` bucket is then requested with a role that deliberately holds
    /// no `s3:GetObject`. The platform setting a variable for the task's *output*
    /// is not the document stating anything about its *input*.
    #[test]
    fn an_injected_container_role_does_not_sign_a_public_read() {
        let _env = EnvScope::new(&[("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", ECS_ROLE_URI)]);
        let opts = resolved_read_options(PUBLIC_STORE, &[]);
        assert_eq!(
            value(&opts, "aws_skip_signature"),
            Some("true"),
            "a role the platform injected must not sign a public read"
        );
        assert!(
            value(&opts, "aws_container_credentials_relative_uri").is_some(),
            "the pointer is still passed through, so a caller who asks for \
             signing authenticates with the ambient role"
        );
    }

    /// The same shape on the other container platform, because the reasoning was
    /// never about ECS: EKS injects `AWS_ROLE_ARN` +
    /// `AWS_WEB_IDENTITY_TOKEN_FILE` into every pod with a service-account role.
    /// Pinned so that "which variables a platform injects" can grow without the
    /// rule having to be rediscovered — an ambient role is an ambient role.
    #[test]
    fn an_injected_web_identity_role_does_not_sign_a_public_read_either() {
        let _env = EnvScope::new(&[
            (
                "AWS_ROLE_ARN",
                "arn:aws:iam::000000000000:role/not-a-real-role",
            ),
            (
                "AWS_WEB_IDENTITY_TOKEN_FILE",
                "/var/run/secrets/eks.amazonaws.com/serviceaccount/token",
            ),
        ]);
        let opts = resolved_read_options(PUBLIC_STORE, &[]);
        assert_eq!(
            value(&opts, "aws_skip_signature"),
            Some("true"),
            "a service-account role the platform injected is not a stated intent"
        );
    }

    /// The credential list must name every spelling `object_store` parses. A
    /// missing one is silent and backwards: the caller states a credential, this
    /// module does not recognise it, and the read they asked to sign goes out
    /// anonymous — a 403 they cannot explain from the options they passed.
    ///
    /// Asserted against `object_store`'s own parser rather than a copy of its
    /// table, so the day it gains an alias this fails instead of drifting.
    #[test]
    fn every_credential_and_endpoint_spelling_is_one_object_store_parses() {
        use std::str::FromStr;
        for key in S3_CREDENTIAL_KEYS
            .iter()
            .chain(S3_ENDPOINT_KEYS.iter())
            .chain(S3_SIGNING_KEYS.iter())
            .chain(S3_STATIC_KEY_KEYS.iter())
        {
            assert!(
                object_store::aws::AmazonS3ConfigKey::from_str(key).is_ok(),
                "'{key}' is not an option key object_store understands, so \
                 naming it here can only ever be a no-op"
            );
        }
        // And the converse for the one that matters most: a caller who states
        // the unprefixed spelling is stating intent just as much.
        for stated in ["container_credentials_full_uri", "token", "role_arn"] {
            assert!(
                S3_CREDENTIAL_KEYS.contains(&stated),
                "'{stated}' is a credential object_store accepts and this list must see it"
            );
        }
    }

    /// The same door, on the other runner tier: a process holding
    /// `AWS_ACCESS_KEY_ID` because it WRITES its output dataset somewhere. Those
    /// keys say nothing about the public store the document reads, and reading
    /// them as intent is how a Fly-dispatched run would 403 on exactly the URL
    /// that works from a laptop.
    #[test]
    fn the_processs_own_write_credentials_do_not_sign_a_public_read() {
        let _env = EnvScope::new(&[
            ("AWS_ACCESS_KEY_ID", "AKIAEXAMPLENOTREAL"),
            ("AWS_SECRET_ACCESS_KEY", "not-a-real-secret-for-tests"),
        ]);
        let opts = resolved_read_options(PUBLIC_STORE, &[]);
        assert_eq!(
            value(&opts, "aws_skip_signature"),
            Some("true"),
            "ambient write credentials must not sign a public read"
        );
    }

    /// A caller that genuinely wants signed access still gets it, three ways:
    /// `aws_skip_signature=false` ("sign, I mean it"), credentials stated on the
    /// load, and `AWS_SKIP_SIGNATURE=false` in the environment — which nothing
    /// injects, so it can only be a deployment saying so. In the first case the
    /// ambient role is still on hand to sign WITH.
    #[test]
    fn a_caller_that_asks_for_signed_access_gets_it() {
        let _env = EnvScope::new(&[("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", ECS_ROLE_URI)]);

        const PRIVATE_STORE: &str = "s3://private-bucket/store.zarr";

        let stated = [("aws_skip_signature".to_string(), "false".to_string())];
        let opts = resolved_read_options(PRIVATE_STORE, &stated);
        assert_eq!(
            value(&opts, "aws_skip_signature"),
            Some("false"),
            "no default may overturn a stated intent"
        );
        assert!(
            value(&opts, "aws_container_credentials_relative_uri").is_some(),
            "asking for signing must leave something to sign with"
        );

        let stated = [
            (
                "aws_access_key_id".to_string(),
                "AKIAEXAMPLENOTREAL".to_string(),
            ),
            (
                "aws_secret_access_key".to_string(),
                "not-a-real-secret-for-tests".to_string(),
            ),
        ];
        let opts = resolved_read_options(PRIVATE_STORE, &stated);
        assert_eq!(
            value(&opts, "aws_skip_signature"),
            None,
            "credentials the caller stated ARE the caller stating intent"
        );

        drop(_env);
        let _env = EnvScope::new(&[("AWS_SKIP_SIGNATURE", "false")]);
        let opts = resolved_read_options(PUBLIC_STORE, &[]);
        assert_eq!(
            value(&opts, "aws_skip_signature"),
            Some("false"),
            "AWS_SKIP_SIGNATURE is not injected by anything: it is a statement"
        );
    }

    /// The R2 / MinIO / Backblaze B2 / Ceph story is unchanged. An endpoint
    /// override is never injected by a platform either, so credentials sitting
    /// next to one in the environment belong to the same deliberate
    /// configuration and still sign. An endpoint with no credentials is an
    /// anonymous S3-compatible read and still does not.
    #[test]
    fn an_env_configured_s3_compatible_endpoint_still_signs() {
        {
            let _env = EnvScope::new(&[
                (
                    "AWS_ENDPOINT_URL",
                    "https://account.r2.cloudflarestorage.com",
                ),
                ("AWS_ACCESS_KEY_ID", "AKIAEXAMPLENOTREAL"),
                ("AWS_SECRET_ACCESS_KEY", "not-a-real-secret-for-tests"),
            ]);
            let opts = resolved_read_options("s3://my-bucket/store.zarr", &[]);
            assert_eq!(
                value(&opts, "aws_skip_signature"),
                None,
                "an operator-configured endpoint plus keys is a configured provider"
            );
        }
        let _env = EnvScope::new(&[("AWS_ENDPOINT_URL", "http://127.0.0.1:9000")]);
        let opts = resolved_read_options("s3://my-bucket/store.zarr", &[]);
        assert_eq!(
            value(&opts, "aws_skip_signature"),
            Some("true"),
            "an endpoint alone is still an anonymous read"
        );
    }

    // --- the per-bucket statement ------------------------------------------
    //
    // `EARTHSCI_S3_BUCKET_OPTIONS` exists because a process can need two
    // different answers about two different buckets at once: our own private
    // store beside somebody else's requester-pays one. These pin the precedence
    // it sits at, and — the property that matters most — that naming a bucket
    // for one reason does not quietly change the answer to the other question.
    // -----------------------------------------------------------------------

    /// The spec for the case it was written for.
    const REQUESTER_PAYS: &str = r#"{"inmap-model":{"aws_skip_signature":"false","aws_request_payer":"true","aws_region":"us-east-2"}}"#;

    /// A bucket's own statement outranks an ambient variable, because the
    /// ambient one describes a process and this one describes a bucket.
    #[test]
    fn a_named_bucket_outranks_the_environment() {
        let _env = EnvScope::new(&[
            ("AWS_REGION", "eu-west-1"),
            (crate::BUCKET_OPTIONS_ENV, REQUESTER_PAYS),
        ]);
        let opts = resolved_read_options(PUBLIC_STORE, &[]);
        assert_eq!(value(&opts, "aws_region"), Some("us-east-2"));
        assert_eq!(value(&opts, "aws_request_payer"), Some("true"));
        assert_eq!(
            value(&opts, "aws_skip_signature"),
            Some("false"),
            "naming the bucket with a signing statement is stating intent"
        );
    }

    /// …and loses to the caller, who is describing one loader rather than one
    /// deployment.
    #[test]
    fn a_caller_still_outranks_a_named_bucket() {
        let _env = EnvScope::new(&[(crate::BUCKET_OPTIONS_ENV, REQUESTER_PAYS)]);
        let stated = [("aws_region".to_string(), "us-west-2".to_string())];
        let opts = resolved_read_options(PUBLIC_STORE, &stated);
        assert_eq!(value(&opts, "aws_region"), Some("us-west-2"));
        assert_eq!(
            value(&opts, "aws_request_payer"),
            Some("true"),
            "overriding one key must not discard the rest of the bucket's options"
        );
    }

    /// The guard rail. A bucket named only to pin its region says NOTHING about
    /// signing, so the anonymous default still applies — otherwise the mechanism
    /// would have reintroduced, through its own front door, the exact failure it
    /// was built beside: a public read signed with a role that cannot read it.
    #[test]
    fn naming_a_bucket_to_pin_its_region_leaves_the_read_anonymous() {
        let _env = EnvScope::new(&[
            ("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", ECS_ROLE_URI),
            (
                crate::BUCKET_OPTIONS_ENV,
                r#"{"inmap-model":{"aws_region":"us-east-2"}}"#,
            ),
        ]);
        let opts = resolved_read_options(PUBLIC_STORE, &[]);
        assert_eq!(value(&opts, "aws_region"), Some("us-east-2"));
        assert_eq!(
            value(&opts, "aws_skip_signature"),
            Some("true"),
            "a region is not a statement about signing"
        );
    }

    /// A credential named against a bucket IS a statement, unlike the same
    /// credential harvested from the environment. The difference is that this
    /// one names the bucket it is for.
    #[test]
    fn a_credential_named_against_a_bucket_signs_that_bucket() {
        let _env = EnvScope::new(&[(
            crate::BUCKET_OPTIONS_ENV,
            r#"{"private-bucket":{"aws_access_key_id":"AKIAEXAMPLENOTREAL","aws_secret_access_key":"not-a-real-secret-for-tests"}}"#,
        )]);
        let opts = resolved_read_options("s3://private-bucket/store.zarr", &[]);
        assert_eq!(value(&opts, "aws_skip_signature"), None);
        assert_eq!(value(&opts, "aws_access_key_id"), Some("AKIAEXAMPLENOTREAL"));

        // …and only that bucket. The point of the whole exercise.
        let opts = resolved_read_options(PUBLIC_STORE, &[]);
        assert_eq!(value(&opts, "aws_skip_signature"), Some("true"));
        assert_eq!(value(&opts, "aws_access_key_id"), None);
    }

    /// A bucket nobody named reads exactly as it did before.
    #[test]
    fn an_unnamed_bucket_is_untouched() {
        let _env = EnvScope::new(&[(crate::BUCKET_OPTIONS_ENV, REQUESTER_PAYS)]);
        let opts = resolved_read_options("s3://some-other-bucket/store.zarr", &[]);
        assert_eq!(value(&opts, "aws_request_payer"), None);
        assert_eq!(value(&opts, "aws_skip_signature"), Some("true"));
    }

    /// `aws_request_payer` has to be a key `object_store` actually parses, or
    /// the whole mechanism is an option nobody reads and a 403 nobody predicted.
    /// Asserted the way
    /// `every_credential_and_endpoint_spelling_is_one_object_store_parses`
    /// asserts the rest of the vocabulary: by building the backend.
    #[test]
    fn request_payer_is_a_key_object_store_parses() {
        let opts: Vec<(String, String)> = [
            ("aws_request_payer", "true"),
            ("aws_region", "us-east-2"),
            ("aws_skip_signature", "false"),
        ]
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
        assert!(
            resolve_backend("s3://inmap-model/k", &opts).is_ok(),
            "object_store must recognise aws_request_payer"
        );
    }

    /// A spec nobody can parse must not read as "no options for this bucket" —
    /// that is a read that goes out anonymous and fails much later, against
    /// whichever bucket needed the statement. The option path has nowhere to put
    /// the failure, so it yields nothing here and the TRANSPORT refuses the
    /// fetch (see `transport::s3`'s
    /// `a_malformed_bucket_options_spec_refuses_the_fetch`).
    #[test]
    fn a_malformed_spec_yields_no_options_here_and_is_refused_at_the_transport() {
        let _env = EnvScope::new(&[(crate::BUCKET_OPTIONS_ENV, "{not json")]);
        let opts = resolved_read_options(PUBLIC_STORE, &[]);
        assert_eq!(value(&opts, "aws_request_payer"), None);
        assert!(crate::bucket_options_from_env().is_err());
    }

    /// The read default is a READ default. A runner reads a public store and
    /// writes its output to a private one **in the same process**, so the write
    /// must still sign — which is why this is not `AWS_SKIP_SIGNATURE=true` in a
    /// job definition. The write's option source is the raw harvest, and nothing
    /// here may put a signing decision into it.
    #[test]
    fn the_read_default_is_not_applied_to_a_write() {
        let _env = EnvScope::new(&[
            ("AWS_ACCESS_KEY_ID", "AKIAEXAMPLENOTREAL"),
            ("AWS_SECRET_ACCESS_KEY", "not-a-real-secret-for-tests"),
        ]);
        let write_opts = store_options_from_env();
        assert_eq!(
            value(&write_opts, "aws_skip_signature"),
            None,
            "the write path signs with the process's own credentials"
        );
        assert_eq!(
            value(
                &resolved_read_options(PUBLIC_STORE, &[]),
                "aws_skip_signature"
            ),
            Some("true"),
            "the read of a public store in the same process does not"
        );
    }

    /// The same thing over the wire, which is where a 403 actually happens: with
    /// an ECS role in the environment, a direct read reaches the store carrying
    /// **no `Authorization` header at all**. Loopback endpoint, so this is the
    /// real `object_store` client and the real request bytes, with no network.
    #[test]
    fn an_injected_role_sends_an_unsigned_request() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().expect("accept");
            let mut buf = [0u8; 4096];
            let n = sock.read(&mut buf).expect("read request");
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            sock.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nETag: \"x\"\r\n\
                  Last-Modified: Thu, 01 Jan 1970 00:00:00 GMT\r\n\r\nhello",
            )
            .expect("write response");
            let _ = sock.flush();
            req
        });

        let endpoint = format!("http://127.0.0.1:{port}");
        let _env = EnvScope::new(&[
            ("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", ECS_ROLE_URI),
            ("AWS_ENDPOINT_URL", &endpoint),
            ("AWS_ALLOW_HTTP", "true"),
        ]);

        // Exactly what `build_read_store` does with a loader's (empty) options.
        let url = "s3://inmap-model/isrm_v1.2.1.zarr/";
        let mut opts = read_store_options(url, &[]);
        apply_s3_defaults(url, &mut opts);
        let (store, prefix) = resolve_backend(url, &opts).expect("s3 dispatch");

        let rt = runtime().expect("runtime");
        let body = rt
            .block_on(async move {
                let key = prefix.join(".zarray");
                store.get(&key).await?.bytes().await
            })
            .expect("an anonymous GET, with no credential lookup on the way");
        assert_eq!(&body[..], b"hello");

        let req = server.join().expect("server thread").to_ascii_lowercase();
        assert!(
            !req.contains("authorization:"),
            "a public read must go out unsigned; got:\n{req}"
        );
        assert!(
            !req.contains("x-amz-security-token"),
            "no session credential should have been fetched; got:\n{req}"
        );
    }
}

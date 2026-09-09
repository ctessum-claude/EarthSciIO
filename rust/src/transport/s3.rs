//! The `s3` transport — an anonymous `s3://` → regional-HTTPS URL rewriter over
//! the [`HttpTransport`] (`spec/registries.md` §1).
//!
//! The canonical resolved URL stays `s3://<bucket>/<key…>` (kept verbatim in the
//! cache key + `manifest.url`, exactly like `cds://`). [`S3Transport::fetch`]
//! rewrites it to **virtual-hosted HTTPS** —
//! `https://<bucket>.s3.<region>.amazonaws.com/<key>` — and delegates a plain
//! **anonymous** GET to a held `HttpTransport`. A public bucket needs **no AWS
//! SDK, no SigV4, no credentials**; streaming, conditional GET (S3 returns
//! ETags), redirect following, and mirror failover all come from the HTTP
//! delegate. Region defaults to `us-east-2` (the pinned InMAP ISRM bucket),
//! overridable via `$EARTHSCI_S3_REGION` (fallback `$AWS_REGION`) or
//! [`S3Transport::with_region`]. The `auth` resolver threads through unchanged so
//! a future requester-pays resolver plugs in with no transport edit.
//!
//! ## Signing, for the buckets that are *named*
//!
//! Anonymous is the right default and stays the default: the stores this crate
//! was built to read are public, and the [`format::zarr_object_store`] path
//! already learned the hard way that inferring "we can authenticate" from
//! ambient credentials signs a *public* read and turns it into a 403
//! (`read_store_options`, and the U6 note it carries). Nothing here reads
//! `AWS_ACCESS_KEY_ID` and concludes anything.
//!
//! But a private bucket then has no path at all through the cache, and that is
//! a real gap rather than a policy: every whole-file format — `shapefile`,
//! `ff10`, `netcdf`, `geotiff` — reads through this transport, so a `.zip` in a
//! private bucket is unreadable no matter what the reader's IAM role grants.
//! [`DataSource::store_access`](crate::DataSource::store_access) and
//! [`store_options`](crate::DataSource::store_options), which is where a loader
//! *would* ask for a signed read, are store-backed (Zarr) only.
//!
//! So: **a bucket named in [`SIGNED_BUCKETS_ENV`] (or through
//! [`S3Transport::signing_buckets`]) is fetched signed**, through
//! `object_store`'s AWS client — the same mature client, credential chain and
//! SigV4 the direct Zarr path uses, rather than any hand-rolled signing here.
//! Every other bucket takes the anonymous path unchanged, byte for byte.
//!
//! Naming a bucket is the *statement of intent* this crate keeps asking for. It
//! cannot be arrived at by accident, it cannot widen to a bucket nobody listed,
//! and — the property that matters most — **it cannot make a public read start
//! signing**, which is the failure mode that has already cost this codebase two
//! rounds of debugging. There is deliberately no `*`.

use std::collections::BTreeSet;
use std::path::Path;

use super::{Conditional, FetchResult, HttpTransport, Transport};
use crate::auth::AuthResolver;
use crate::error::{Error, Result};

/// Default region — the pinned InMAP ISRM bucket lives in `us-east-2`.
pub const DEFAULT_S3_REGION: &str = "us-east-2";

/// Comma-separated buckets whose objects are fetched **signed** rather than
/// anonymously (see the module note).
///
/// A deployment naming its own bucket here is stating that reads of it should
/// authenticate with whatever credentials the process already has — an ECS task
/// role, an instance profile, static keys. Bucket names only: an entry is
/// matched against the `s3://<bucket>/…` host, so it cannot accidentally
/// describe some other origin, and there is no wildcard.
pub const SIGNED_BUCKETS_ENV: &str = "EARTHSCI_S3_SIGNED_BUCKETS";

/// The buckets [`SIGNED_BUCKETS_ENV`] names, empty when unset or blank.
///
/// Blank is treated as unset for the same reason `store_options`' readers do it:
/// an empty string is how a secrets manager spells "not configured", and reading
/// it as a value would be a list with one nameless bucket in it.
fn signed_buckets_from_env() -> BTreeSet<String> {
    std::env::var(SIGNED_BUCKETS_ENV)
        .ok()
        .map(|spec| parse_signed_buckets(&spec))
        .unwrap_or_default()
}

/// Split a comma-separated bucket list, dropping blanks and lowercasing.
///
/// S3 bucket names are already lowercase, so folding case cannot merge two
/// distinct buckets; it only stops a mis-cased entry from silently matching
/// nothing, which would present as an unexplained 403.
fn parse_signed_buckets(spec: &str) -> BTreeSet<String> {
    spec.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_ascii_lowercase)
        .collect()
}

/// The bucket an `s3://` URL addresses.
fn bucket_of(s3_url: &str) -> Result<&str> {
    let rest = s3_url.strip_prefix("s3://").ok_or_else(|| Error::BadUrl {
        url: s3_url.to_string(),
        detail: "not an s3:// URL".to_string(),
    })?;
    let bucket = rest.split('/').next().unwrap_or("");
    if bucket.is_empty() {
        return Err(Error::BadUrl {
            url: s3_url.to_string(),
            detail: "s3:// URL has an empty bucket".to_string(),
        });
    }
    Ok(bucket)
}

/// Refuse a fetch when [`BUCKET_OPTIONS_ENV`](crate::BUCKET_OPTIONS_ENV) is set
/// to something unparseable.
///
/// Checked on every fetch rather than once at construction because the variable
/// is read on every fetch, and because a transport is built long before a
/// deployment's environment is interesting. The cost is one small JSON parse
/// against a variable that is usually unset.
fn check_bucket_options() -> Result<()> {
    crate::s3_config::bucket_options_from_env()
        .map(|_| ())
        .map_err(|detail| Error::BadConfig {
            var: crate::s3_config::BUCKET_OPTIONS_ENV.to_string(),
            detail,
        })
}

/// Resolve the S3 region for `bucket`: explicit arg → the bucket's own
/// `aws_region` in [`BUCKET_OPTIONS_ENV`](crate::BUCKET_OPTIONS_ENV) →
/// `$EARTHSCI_S3_REGION` → `$AWS_REGION` → [`DEFAULT_S3_REGION`].
///
/// A per-bucket statement sits between the two because it is more specific than
/// any process-wide variable and less specific than an argument a caller passed
/// in code. This is the resolution that makes two stores in two regions
/// readable by one process: `$EARTHSCI_S3_REGION` can only ever have one value,
/// and getting it wrong does not fail — it reads the right bytes across a
/// region boundary and bills for the transfer.
pub fn resolve_region_for_bucket(bucket: &str, explicit: Option<&str>) -> String {
    if let Some(r) = explicit {
        return r.to_string();
    }
    let per_bucket = crate::s3_config::options_for_bucket(bucket);
    if let Some(region) = crate::s3_config::stated_region(&per_bucket) {
        return region.to_string();
    }
    resolve_region(None)
}

/// Resolve the S3 region without reference to a bucket: explicit arg →
/// `$EARTHSCI_S3_REGION` → `$AWS_REGION` → [`DEFAULT_S3_REGION`].
///
/// Prefer [`resolve_region_for_bucket`] wherever the bucket is known.
pub fn resolve_region(explicit: Option<&str>) -> String {
    if let Some(r) = explicit {
        return r.to_string();
    }
    for var in ["EARTHSCI_S3_REGION", "AWS_REGION"] {
        if let Ok(v) = std::env::var(var) {
            if !v.is_empty() {
                return v;
            }
        }
    }
    DEFAULT_S3_REGION.to_string()
}

/// Rewrite `s3://<bucket>/<key…>` to regional virtual-hosted HTTPS.
pub fn s3_https_url(s3_url: &str, region: &str) -> Result<String> {
    let bad = |detail: String| Error::BadUrl {
        url: s3_url.to_string(),
        detail,
    };
    let rest = s3_url
        .strip_prefix("s3://")
        .ok_or_else(|| bad("not an s3:// URL".to_string()))?;
    let (bucket, key) = rest
        .split_once('/')
        .ok_or_else(|| bad("s3:// URL has no object key".to_string()))?;
    if bucket.is_empty() {
        return Err(bad("s3:// URL has an empty bucket".to_string()));
    }
    Ok(format!("https://{bucket}.s3.{region}.amazonaws.com/{key}"))
}

/// `s3://` transport: anonymous regional-HTTPS rewrite, or a signed read for the
/// buckets that were named (see the module note).
pub struct S3Transport {
    http: HttpTransport,
    region: Option<String>,
    signed: BTreeSet<String>,
    /// Whether the environment still gets a say in which buckets are signed.
    /// Cleared by [`S3Transport::signing_buckets`], for the reason that method
    /// replaces rather than extends: a caller stating a list gets exactly that
    /// list, and an ambient variable adding a bucket to it would be the accident
    /// this whole mechanism exists to avoid.
    env_signing: bool,
}

impl S3Transport {
    /// A transport resolving the region **and** the signed-bucket list from the
    /// environment ([`SIGNED_BUCKETS_ENV`]; empty by default, so this is the
    /// anonymous transport it has always been unless a deployment says otherwise).
    pub fn new() -> Self {
        Self {
            http: HttpTransport::new(),
            region: None,
            signed: signed_buckets_from_env(),
            env_signing: true,
        }
    }

    /// A transport pinned to `region` (overrides the environment).
    pub fn with_region(region: impl Into<String>) -> Self {
        Self {
            http: HttpTransport::new(),
            region: Some(region.into()),
            signed: signed_buckets_from_env(),
            env_signing: true,
        }
    }

    /// Fetch these buckets signed, **replacing** anything the environment named.
    ///
    /// The programmatic spelling of [`SIGNED_BUCKETS_ENV`], for a library caller
    /// that knows which stores are its own and would rather say so in code than
    /// through the environment. Replaces rather than extends so that a caller
    /// stating a list gets exactly that list — an ambient variable silently
    /// adding a bucket to an explicit set is the accident this whole mechanism
    /// exists to avoid.
    #[must_use]
    pub fn signing_buckets<I, S>(mut self, buckets: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.signed = buckets
            .into_iter()
            .map(|b| b.as_ref().trim().to_ascii_lowercase())
            .filter(|b| !b.is_empty())
            .collect();
        self.env_signing = false;
        self
    }

    /// Whether reads of `bucket` are signed: because it was **named** in
    /// [`SIGNED_BUCKETS_ENV`] (or through [`S3Transport::signing_buckets`]), or
    /// because [`BUCKET_OPTIONS_ENV`](crate::BUCKET_OPTIONS_ENV) states signing
    /// for it — a credential, or `aws_skip_signature=false`.
    ///
    /// A bucket named in the options map only to pin its region states nothing
    /// about signing and reads exactly as anonymously as before
    /// ([`crate::stated_signing`]); a bucket named with
    /// `aws_skip_signature=true` is anonymous even if the signed list names it,
    /// because that is the more specific statement about it.
    ///
    /// A malformed options spec reads here as "no statement". It is not
    /// swallowed — [`Transport::fetch`] refuses the fetch outright — but this
    /// predicate has nowhere to put the failure and must not answer `true` on a
    /// spec nobody could parse.
    #[must_use]
    pub fn signs(&self, bucket: &str) -> bool {
        let bucket = bucket.to_ascii_lowercase();
        if self.env_signing {
            if let Some(stated) =
                crate::s3_config::stated_signing(&crate::s3_config::options_for_bucket(&bucket))
            {
                return stated;
            }
        }
        self.signed.contains(&bucket)
    }

    /// The signed read, for a bucket this transport was told to sign for.
    ///
    /// Delegated to `object_store`'s AWS client — the same one
    /// [`crate::read_zarr_object_store`] uses — so SigV4, the credential chain
    /// (static keys, an instance profile, an ECS task role) and the retry policy
    /// are its concern and not this crate's. `read_store_options` merges the
    /// environment underneath an explicit `aws_skip_signature=false`, which is
    /// how a *caller* states signing in the vocabulary that path already has.
    ///
    /// The object is HEADed first and then pulled in ranges. That buys three
    /// things a single `get` would not: revalidation without a download when the
    /// ETag still matches, a bounded memory profile on a file that may be
    /// gigabytes, and a size to write against.
    #[cfg(all(feature = "object-store", not(target_arch = "wasm32")))]
    fn fetch_signed(&self, url: &str, dest: &Path, conditional: &Conditional) -> Result<FetchResult> {
        use std::io::Write;

        // `head` / `get_range` are convenience methods on `ObjectStoreExt`; the
        // object-safe `ObjectStore` trait itself carries only `get_opts`.
        use object_store::ObjectStoreExt;

        use crate::format::{resolve_backend, runtime};

        /// One ranged GET. Large enough that a 37 KB shapefile is one request
        /// and small enough that a 2 GiB NetCDF never sits in memory.
        const CHUNK: u64 = 8 * 1024 * 1024;

        let region = resolve_region_for_bucket(bucket_of(url)?, self.region.as_deref());
        let options = crate::read_store_options(
            url,
            &[
                ("aws_skip_signature".to_string(), "false".to_string()),
                ("aws_region".to_string(), region),
            ],
        );
        let (store, path) = resolve_backend(url, &options)?;

        let rt = runtime()?;
        let transport_err = |e: object_store::Error| match e {
            object_store::Error::NotFound { .. } => Error::NotFound {
                url: url.to_string(),
                detail: "signed s3 read: no such key".to_string(),
            },
            other => Error::Transport {
                url: url.to_string(),
                detail: format!("signed s3 read: {other}"),
            },
        };

        let meta = rt.block_on(store.head(&path)).map_err(transport_err)?;

        // Revalidation is by ETag alone, and `last_modified` is deliberately not
        // reported back: the anonymous path stores an HTTP-date string and would
        // send it as `If-Modified-Since`, and handing it a differently formatted
        // timestamp from here would produce a validator that silently never
        // matches. An ETag means the same thing on both paths.
        let unchanged = match (conditional.etag.as_deref(), meta.e_tag.as_deref()) {
            (Some(have), Some(now)) => etag_eq(have, now),
            _ => false,
        };
        if unchanged {
            return Ok(FetchResult {
                status: super::FetchStatus::NotModified,
                etag: meta.e_tag,
                last_modified: None,
                bytes_written: 0,
            });
        }

        let mut file = std::fs::File::create(dest)?;
        let mut offset = 0u64;
        while offset < meta.size {
            let end = (offset + CHUNK).min(meta.size);
            let bytes = rt
                .block_on(store.get_range(&path, offset..end))
                .map_err(transport_err)?;
            file.write_all(&bytes)?;
            offset = end;
        }
        file.flush()?;

        Ok(FetchResult {
            status: super::FetchStatus::Downloaded,
            etag: meta.e_tag,
            last_modified: None,
            bytes_written: meta.size,
        })
    }

    /// Without the `object-store` feature there is no signing client to reach.
    ///
    /// This **errors rather than falling back to an anonymous fetch**, following
    /// [`StoreAccess::Direct`](crate::StoreAccess::Direct), which "refuses at
    /// construction ... rather than silently falling back". A quiet anonymous
    /// read of a bucket someone asked to sign is a 403 at best, and at worst a
    /// different object than the one they meant.
    #[cfg(not(all(feature = "object-store", not(target_arch = "wasm32"))))]
    fn fetch_signed(&self, url: &str, _d: &Path, _c: &Conditional) -> Result<FetchResult> {
        Err(Error::Transport {
            url: url.to_string(),
            detail: format!(
                "this bucket is named in {SIGNED_BUCKETS_ENV}, but signing needs the \
                 `object-store` feature, which this build does not have"
            ),
        })
    }
}

/// Compare two ETags ignoring the quoting and any weak marker.
///
/// One path takes the header verbatim and the other takes `object_store`'s
/// parse of it; `"abc"` and `abc` are the same validator, and treating them as
/// different would re-download every object on every run.
fn etag_eq(a: &str, b: &str) -> bool {
    fn norm(s: &str) -> &str {
        s.trim().trim_start_matches("W/").trim_matches('"')
    }
    norm(a) == norm(b)
}

impl Default for S3Transport {
    fn default() -> Self {
        Self::new()
    }
}

impl Transport for S3Transport {
    fn schemes(&self) -> &'static [&'static str] {
        &["s3"]
    }

    fn fetch(
        &self,
        url: &str,
        dest: &Path,
        conditional: &Conditional,
        auth: Option<&dyn AuthResolver>,
    ) -> Result<FetchResult> {
        let bucket = bucket_of(url)?;
        check_bucket_options()?;
        if self.signs(bucket) {
            return self.fetch_signed(url, dest, conditional);
        }
        let region = resolve_region_for_bucket(bucket, self.region.as_deref());
        let https = s3_https_url(url, &region)?;
        self.http.fetch(&https, dest, conditional, auth)
    }

    fn fetch_hashed(
        &self,
        url: &str,
        dest: &Path,
        conditional: &Conditional,
        auth: Option<&dyn AuthResolver>,
    ) -> Result<(FetchResult, Option<String>)> {
        // `None` for the signed branch: the hash is the caller's to take off the
        // staged file, which is this method's documented default. Hashing in
        // transit would be free here, but it would be a second implementation of
        // integrity to keep in step with the one the cache already runs.
        let bucket = bucket_of(url)?;
        check_bucket_options()?;
        if self.signs(bucket) {
            return Ok((self.fetch_signed(url, dest, conditional)?, None));
        }
        let region = resolve_region_for_bucket(bucket, self.region.as_deref());
        let https = s3_https_url(url, &region)?;
        self.http.fetch_hashed(&https, dest, conditional, auth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_env::EnvScope;

    #[test]
    fn scheme_is_s3() {
        assert_eq!(S3Transport::new().schemes(), &["s3"]);
    }

    #[test]
    fn rewrite_default_and_explicit_region() {
        assert_eq!(
            s3_https_url(
                "s3://inmap-model/isrm_v1.2.1.zarr/PrimaryPM25/0.5.0",
                "us-east-2"
            )
            .unwrap(),
            "https://inmap-model.s3.us-east-2.amazonaws.com/isrm_v1.2.1.zarr/PrimaryPM25/0.5.0"
        );
        assert_eq!(
            s3_https_url("s3://b/k/o", "eu-west-1").unwrap(),
            "https://b.s3.eu-west-1.amazonaws.com/k/o"
        );
    }

    #[test]
    fn rewrite_rejects_bad_urls() {
        assert!(s3_https_url("https://x/y", "us-east-2").is_err());
        assert!(s3_https_url("s3://bucket-only", "us-east-2").is_err());
        assert!(s3_https_url("s3:///key", "us-east-2").is_err());
    }

    // --- signing -----------------------------------------------------------

    /// The property the whole mechanism rests on: with nothing named, this is
    /// the anonymous transport it has always been. A regression here signs a
    /// public read and turns it into a 403.
    #[test]
    fn nothing_is_signed_by_default() {
        let t = S3Transport::new().signing_buckets(Vec::<String>::new());
        assert!(!t.signs("inmap-model"));
        assert!(!t.signs("anything-at-all"));
    }

    #[test]
    fn only_the_named_buckets_are_signed() {
        let t = S3Transport::new().signing_buckets(["my-private-bucket"]);
        assert!(t.signs("my-private-bucket"));
        assert!(!t.signs("inmap-model"), "a public bucket must not start signing");
        assert!(!t.signs("my-private-bucket-2"), "prefixes are not matches");
    }

    /// Case folding is a convenience that cannot merge two real buckets, since
    /// S3 bucket names are lowercase to begin with.
    #[test]
    fn bucket_matching_folds_case() {
        let t = S3Transport::new().signing_buckets(["My-Bucket"]);
        assert!(t.signs("my-bucket"));
        assert!(t.signs("MY-BUCKET"));
    }

    /// An explicit list REPLACES the environment, so a caller that states its
    /// stores gets exactly those and cannot have one added underneath it.
    #[test]
    fn an_explicit_list_replaces_the_environment() {
        let t = S3Transport::new()
            .signing_buckets(["from-env-maybe"])
            .signing_buckets(["only-this-one"]);
        assert!(t.signs("only-this-one"));
        assert!(!t.signs("from-env-maybe"));
    }

    // --- the per-bucket statement ------------------------------------------

    /// A bucket that states signing in `EARTHSCI_S3_BUCKET_OPTIONS` is signed
    /// without having to appear in the signed list as well. One statement, not
    /// two things to keep in step.
    #[test]
    fn a_named_bucket_states_its_own_signing() {
        let _env = EnvScope::new(&[(
            crate::BUCKET_OPTIONS_ENV,
            r#"{"inmap-model":{"aws_skip_signature":"false"}}"#,
        )]);
        assert!(S3Transport::new().signs("inmap-model"));
        assert!(!S3Transport::new().signs("some-other-bucket"));
    }

    /// The guard rail, at the transport this time: naming a bucket to pin its
    /// region must not start signing it. A signed read of a public bucket is a
    /// 403, and it is a 403 that only appears in the deployment.
    #[test]
    fn naming_a_bucket_for_its_region_does_not_start_signing_it() {
        let _env = EnvScope::new(&[(
            crate::BUCKET_OPTIONS_ENV,
            r#"{"inmap-model":{"aws_region":"us-east-2"}}"#,
        )]);
        assert!(!S3Transport::new().signs("inmap-model"));
    }

    /// The more specific statement wins in both directions: a bucket the signed
    /// list names can be turned back off by naming it with
    /// `aws_skip_signature=true`.
    #[test]
    fn a_bucket_can_state_anonymous_against_the_signed_list() {
        let _env = EnvScope::new(&[
            (SIGNED_BUCKETS_ENV, "inmap-model"),
            (
                crate::BUCKET_OPTIONS_ENV,
                r#"{"inmap-model":{"aws_skip_signature":"true"}}"#,
            ),
        ]);
        assert!(!S3Transport::new().signs("inmap-model"));
    }

    /// `signing_buckets` replaces the environment's say entirely — including the
    /// options map's say about signing. A caller stating its stores in code gets
    /// exactly those, which is the whole reason that method replaces rather than
    /// extends.
    #[test]
    fn an_explicit_list_also_replaces_the_options_maps_say() {
        let _env = EnvScope::new(&[(
            crate::BUCKET_OPTIONS_ENV,
            r#"{"inmap-model":{"aws_skip_signature":"false"}}"#,
        )]);
        let t = S3Transport::new().signing_buckets(["only-this-one"]);
        assert!(t.signs("only-this-one"));
        assert!(!t.signs("inmap-model"));
    }

    /// A bucket's own region beats the process-wide pin. This is what makes two
    /// stores in two regions readable by one process: `EARTHSCI_S3_REGION` has
    /// one value, and the wrong one does not fail — it reads the right bytes
    /// across a region boundary and bills for the transfer.
    #[test]
    fn a_buckets_own_region_beats_the_process_wide_pin() {
        let _env = EnvScope::new(&[
            ("EARTHSCI_S3_REGION", "eu-west-1"),
            (
                crate::BUCKET_OPTIONS_ENV,
                r#"{"inmap-model":{"aws_region":"us-east-2"}}"#,
            ),
        ]);
        assert_eq!(resolve_region_for_bucket("inmap-model", None), "us-east-2");
        assert_eq!(resolve_region_for_bucket("other", None), "eu-west-1");
        assert_eq!(
            resolve_region_for_bucket("inmap-model", Some("ap-south-1")),
            "ap-south-1",
            "an argument passed in code is the most local statement of all"
        );
    }

    /// A spec nobody can parse fails the fetch, rather than reading as "no
    /// statement" and going out with the wrong identity. Checked before any
    /// network, so this costs nothing and names the variable.
    #[test]
    fn a_malformed_bucket_options_spec_refuses_the_fetch() {
        let _env = EnvScope::new(&[(crate::BUCKET_OPTIONS_ENV, "{not json")]);
        let dest = std::env::temp_dir().join("earthsciio-bucket-options-test");
        let err = S3Transport::new()
            .fetch("s3://inmap-model/k", &dest, &Conditional::default(), None)
            .unwrap_err();
        match err {
            Error::BadConfig { var, .. } => assert_eq!(var, crate::BUCKET_OPTIONS_ENV),
            other => panic!("expected BadConfig, got {other}"),
        }
        assert!(!dest.exists(), "a refused fetch writes nothing");
    }

    #[test]
    fn the_env_list_drops_blanks_and_is_never_wild() {
        let parsed = parse_signed_buckets(" a , ,b,, ");
        assert_eq!(parsed.len(), 2);
        assert!(parsed.contains("a") && parsed.contains("b"));
        // There is deliberately no wildcard: `*` is a bucket name, not "all".
        let star = parse_signed_buckets("*");
        assert!(star.contains("*"));
        assert!(!S3Transport::new().signing_buckets(["*"]).signs("inmap-model"));
    }

    #[test]
    fn bucket_of_reads_the_host() {
        assert_eq!(bucket_of("s3://b/k/o").unwrap(), "b");
        assert_eq!(bucket_of("s3://b/").unwrap(), "b");
        assert!(bucket_of("s3://").is_err());
        assert!(bucket_of("https://b/k").is_err());
    }

    /// A bucket with no key is not fetchable, but it must fail as a bad URL
    /// rather than by picking a transport path — both branches see the same
    /// error, so the routing decision cannot change what an invalid URL means.
    #[test]
    fn a_keyless_url_fails_the_same_way_signed_or_not() {
        assert!(s3_https_url("s3://bucket-only", "us-east-2").is_err());
        assert_eq!(bucket_of("s3://bucket-only").unwrap(), "bucket-only");
    }

    #[test]
    fn etags_compare_through_quotes_and_weak_markers() {
        assert!(etag_eq("\"abc\"", "abc"));
        assert!(etag_eq("W/\"abc\"", "\"abc\""));
        assert!(!etag_eq("\"abc\"", "\"abd\""));
    }
}

//! Error type for the EarthSciIO cache/transport/store machinery.

use std::path::PathBuf;

/// Convenience alias for fallible EarthSciIO operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors raised by the cache, transports, and stores.
///
/// `CacheMiss` is load-bearing for the offline contract: it carries both the
/// resolved URL and its cache key so a failure names exactly which blob the
/// corpus/cache is missing (see `spec/offline-mode.md` §2).
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A cache-only read (offline mode, or an offline store) found no blob for
    /// the resolved URL's key. Never a silent empty result, never a fallback
    /// fetch (`spec/offline-mode.md` §2).
    CacheMiss {
        /// The resolved URL whose blob is absent.
        url: String,
        /// The cache key (`sha256(url)`) that was looked up.
        key: String,
    },

    /// An on-disk blob failed its stored `sha256_content` or byte-length check.
    Integrity {
        /// The cache key of the failing blob.
        key: String,
        /// What mismatched (size or hash, with both values).
        detail: String,
    },

    /// No transport is registered for a URL scheme — a registration gap, not a
    /// Provider change (`spec/registries.md` §1).
    UnknownScheme {
        /// The unhandled URL scheme.
        scheme: String,
        /// The URL whose scheme had no transport.
        url: String,
    },

    /// No store is registered under the configured name (`spec/registries.md` §3).
    UnknownStore {
        /// The configured store name that was not found.
        name: String,
    },

    /// A resolved URL could not be parsed or had no usable scheme.
    BadUrl {
        /// The offending URL.
        url: String,
        /// Why it could not be used.
        detail: String,
    },

    /// A transport-level failure (network error, non-success HTTP status, a
    /// missing local `file://` source, …).
    Transport {
        /// The URL being fetched.
        url: String,
        /// The underlying transport failure.
        detail: String,
    },

    /// The source answered, and the answer is **the object does not exist**
    /// (HTTP 404/410, a missing local `file://` path).
    ///
    /// Distinct from [`Error::Transport`] because a *probing* consumer must be
    /// able to tell absence from a failure to find out. A zarr store has to
    /// report a missing key as `Ok(None)` — zarr opens an array by probing the
    /// v3 `zarr.json` before falling back to the v2 `.zarray`, so treating that
    /// 404 as an error makes every v2 store unreadable. Reporting "absent" for
    /// a timeout or a 5xx would be the opposite mistake: a live store would
    /// silently read as empty.
    NotFound {
        /// The URL that does not exist.
        url: String,
        /// How the absence was reported (e.g. `HTTP 404`).
        detail: String,
    },

    /// Every source (the primary URL plus any failover mirrors) failed.
    AllMirrorsFailed {
        /// The canonical resolved URL.
        url: String,
        /// The last underlying failure encountered.
        detail: String,
        /// True only when EVERY source reported a definitive absence. One
        /// transient failure among the mirrors makes the outcome unknown.
        not_found: bool,
    },

    /// An authenticated realm was requested but no resolver is registered for it.
    MissingAuth {
        /// The realm with no registered resolver.
        realm: String,
    },

    /// An I/O error, tagged with the path that triggered it when known.
    Io {
        /// The path the I/O error concerns, if known.
        path: Option<PathBuf>,
        /// The underlying I/O error.
        source: std::io::Error,
    },

    /// Manifest JSON could not be (de)serialized.
    Manifest {
        /// The (de)serialization failure detail.
        detail: String,
    },

    /// No reader is registered under the configured format name — a registration
    /// gap, not a Provider change (`spec/registries.md` §2).
    UnknownFormat {
        /// The configured format name that was not found.
        name: String,
    },

    /// A configuration variable this crate reads is set to something it cannot
    /// use.
    ///
    /// Deliberately an error and not a fallback to the unconfigured default: the
    /// variables that reach here state how a store is to be *reached*, so
    /// ignoring a malformed one produces a read that goes out with the wrong
    /// identity and fails much later, somewhere that names neither the variable
    /// nor the mistake.
    BadConfig {
        /// The environment variable at fault.
        var: String,
        /// Why its value could not be used.
        detail: String,
    },

    /// A format reader failed to decode a cached blob (bad magic, truncated file,
    /// an unsupported on-disk type, …). Decode parity is `spec/conformance.md` §3.
    Format {
        /// The reader's format name (e.g. `netcdf`).
        format: String,
        /// What went wrong while decoding.
        detail: String,
    },
}

impl Error {
    /// Wrap an I/O error together with the path it concerns.
    pub fn io(path: Option<PathBuf>, source: std::io::Error) -> Self {
        Error::Io { path, source }
    }

    /// True when this is a [`Error::CacheMiss`] — the offline "absent" signal.
    pub fn is_cache_miss(&self) -> bool {
        matches!(self, Error::CacheMiss { .. })
    }

    /// True when the source DEFINITIVELY reported that the object does not
    /// exist — a 404/410, a missing local file, or an all-mirrors failure in
    /// which every source said so.
    ///
    /// A probing consumer (a zarr store answering "is this key present?")
    /// treats this as absence; every other failure stays an error, because
    /// there existence is unknown.
    pub fn is_not_found(&self) -> bool {
        match self {
            Error::NotFound { .. } => true,
            Error::AllMirrorsFailed { not_found, .. } => *not_found,
            // A missing local path surfaces as an ordinary I/O error.
            Error::Io { source, .. } => source.kind() == std::io::ErrorKind::NotFound,
            _ => false,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::CacheMiss { url, key } => {
                write!(f, "cache miss: no blob for key {key} (url {url})")
            }
            Error::Integrity { key, detail } => {
                write!(f, "integrity failure for key {key}: {detail}")
            }
            Error::UnknownScheme { scheme, url } => {
                write!(
                    f,
                    "no transport registered for scheme '{scheme}' (url {url})"
                )
            }
            Error::UnknownStore { name } => write!(f, "no store registered as '{name}'"),
            Error::BadUrl { url, detail } => write!(f, "bad url '{url}': {detail}"),
            Error::Transport { url, detail } => write!(f, "transport error for {url}: {detail}"),
            Error::NotFound { url, detail } => write!(f, "not found: {url} ({detail})"),
            Error::AllMirrorsFailed { url, detail, .. } => {
                write!(f, "all sources failed for {url}: {detail}")
            }
            Error::MissingAuth { realm } => {
                write!(f, "no auth resolver registered for realm '{realm}'")
            }
            Error::Io { path, source } => match path {
                Some(p) => write!(f, "io error at {}: {source}", p.display()),
                None => write!(f, "io error: {source}"),
            },
            Error::Manifest { detail } => write!(f, "manifest error: {detail}"),
            Error::BadConfig { var, detail } => write!(f, "{var} is unusable: {detail}"),
            Error::UnknownFormat { name } => write!(f, "no reader registered for format '{name}'"),
            Error::Format { format, detail } => {
                write!(f, "{format} decode error: {detail}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(source: std::io::Error) -> Self {
        Error::Io { path: None, source }
    }
}

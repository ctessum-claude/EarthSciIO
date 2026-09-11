"""EarthSciIO — Python binding of the cross-language data-provider spec.

This package exposes:

* the **extensibility seam** — the three registries
  (:data:`transport_registry`, :data:`format_registry`, :data:`store_registry`),
  their interfaces (:class:`Transport`, :class:`Reader`, :class:`Store`), and the
  registry error types (:class:`Unsupported`, :class:`BackendNotRegistered`);
* the **cache core** (``esio-9nb.2``) — :class:`Cache` + :class:`CacheEntry`, the
  URL→content-addressed-blob fetcher behind the ESS opener/fetcher seam, with its
  runtime errors (:class:`CacheMiss`, :class:`IntegrityError`, …), the
  :class:`Manifest`, the :func:`cache_key`, the :class:`~earthsciio.validate.Temporal`
  freshness policy, and the pluggable auth resolvers.

On import it registers the cloud **stubs** (S3 transport+store, Zarr reader) and
the **active** core backends (``http``/``file`` transports, ``local`` store) into
the three registries — additively, without changing the Provider API (the whole
point of the seam, ``spec/registries.md`` §4).
"""

from __future__ import annotations

from .errors import (
    AuthError,
    BackendNotRegistered,
    CacheMiss,
    EarthSciIOError,
    FetchError,
    IntegrityError,
    OfflineError,
    TransportError,
    UnknownReaderOption,
    Unsupported,
)
from .registry import (
    Reader,
    Registry,
    RegistryEntry,
    Store,
    Transport,
    all_registries,
    dim_length,
    format_registry,
    store_registry,
    supports_selection,
    transport_registry,
)
from .cachekey import cache_key, range_keyed_url, sha256_bytes, sha256_file
from .config import (
    CACHE_FORMAT_VERSION,
    default_cache_root,
    resolve_cache_root,
    resolve_offline,
)
from .manifest import Manifest, parse_rfc3339, utc_now_rfc3339
from .validate import Temporal
from .auth import AuthRegistry, AuthResolver, StaticHeaderAuth
from .cache import Cache, CacheEntry
from .native import NativeDataset, NativeField
from .readers import (
    CSVReader,
    FF10Reader,
    GeoTIFFReader,
    ParquetReader,
    ShapefileReader,
    NetCDFReader,
    register_format_readers,
)
from .provider import (
    DataLoader,  # 0.1.1 alias of DataSource; dropped in 0.2.0
    DataSource,
    LoaderTemporal,  # 0.1.1 alias of SourceTemporal; dropped in 0.2.0
    Provider,
    SourceTemporal,
    Window,
    reader_option_names,
)
from .backends.cds import (
    CdsTransport,
    cds_api_key,
    cds_api_url,
    cds_auth,
    decode_cds_url,
    encode_cds_url,
)
from . import era5

# Register backends on import (idempotent). Stubs first (esio-9nb.8), then the
# active core backends (esio-9nb.2); order is irrelevant — names are orthogonal.
from . import backends

backends.register_stub_backends()
backends.register_active_backends()

# Register the active netcdf/csv format readers (esio-9nb.3) — the decode half
# the Provider reads through. Idempotent; orthogonal to the zarr stub.
register_format_readers()

__all__ = [
    # errors — registry seam
    "EarthSciIOError",
    "BackendNotRegistered",
    "Unsupported",
    "UnknownReaderOption",
    # errors — cache core
    "CacheMiss",
    "IntegrityError",
    "TransportError",
    "FetchError",
    "OfflineError",
    "AuthError",
    # registry mechanism + interfaces
    "Registry",
    "RegistryEntry",
    "Transport",
    "Reader",
    "Store",
    # the three registry singletons
    "transport_registry",
    "format_registry",
    "store_registry",
    "all_registries",
    "supports_selection",
    "dim_length",
    # cache core
    "Cache",
    "CacheEntry",
    "Manifest",
    "Temporal",
    "cache_key",
    "range_keyed_url",
    "sha256_bytes",
    "sha256_file",
    "utc_now_rfc3339",
    "parse_rfc3339",
    # native arrays + format readers (esio-9nb.3)
    "NativeField",
    "NativeDataset",
    "NetCDFReader",
    "CSVReader",
    "GeoTIFFReader",
    "FF10Reader",
    "ShapefileReader",
    "ParquetReader",
    "register_format_readers",
    # provider API (esio-9nb.3)
    "Provider",
    "DataSource",
    "SourceTemporal",
    "Window",
    "reader_option_names",
    # 0.1.1 spellings, kept source-compatible; dropped in 0.2.0
    "DataLoader",
    "LoaderTemporal",
    # config
    "CACHE_FORMAT_VERSION",
    "resolve_cache_root",
    "default_cache_root",
    "resolve_offline",
    # auth
    "AuthRegistry",
    "AuthResolver",
    "StaticHeaderAuth",
    # CDS transport + ERA5 request mapping
    "CdsTransport",
    "cds_api_key",
    "cds_api_url",
    "cds_auth",
    "encode_cds_url",
    "decode_cds_url",
    "era5",
    # backend package
    "backends",
]

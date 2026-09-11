"""The three EarthSciIO extensibility registries (Python binding).

This is the Python realization of the language-neutral contract in
``spec/registries.md`` / ``spec/registries.json``. EarthSciIO is **extensible
by construction** through three registries, each a *name → implementation*
lookup:

* :data:`transport_registry` — keyed by URL **scheme** (``http``/``file``/``s3``).
* :data:`format_registry` — keyed by **format name** (``netcdf``/``csv``/``zarr``).
* :data:`store_registry` — keyed by **store name** (``local``/``s3``).

The single load-bearing rule (spec §4): **a new backend registers under a new
name WITHOUT touching the Provider API.** The Provider depends only on the three
*interfaces* below — :class:`Transport`, :class:`Reader`, :class:`Store` — and
resolves the concrete implementation by name at runtime::

    transport = transport_registry.create(scheme_of(url))   # resolved by name
    store     = store_registry.create(config.store)          # resolved by name
    reader    = format_registry.create(loader.format)        # resolved by name

Adding S3 transport, a Zarr reader, or an object-store backend is therefore a
*registration*, never a Provider edit. ``esio-9nb.8`` proves this by registering
and exercising the S3 + Zarr **stubs** through exactly these lookups (see
:mod:`earthsciio.backends`).

The active backends (``http``/``file`` transport, ``netcdf``/``csv`` readers,
``local`` store) are contributed by the language-core work (``esio-9nb.2``);
this module only defines the seam and the spec-faithful interfaces they bind to.
"""

from __future__ import annotations

import importlib.util

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)

from .errors import BackendNotRegistered

__all__ = [
    "Transport",
    "Reader",
    "Store",
    "RegistryEntry",
    "Registry",
    "transport_registry",
    "format_registry",
    "store_registry",
    "all_registries",
    "supports_selection",
    "dim_length",
]


def dim_length(reader: Any, path: Any, dim: str) -> Optional[int]:
    """The length of dimension ``dim`` in the blob at ``path``, or ``None``.

    Read from the blob's METADATA — the header, never an array
    (``spec/registries.md`` §2.3). ``None`` when the reader cannot answer (the
    default every reader inherits) or when the blob has no such dimension; a
    caller must treat that as "read whole and slice on my own side" rather than
    as an error.

    This is the whole-file counterpart of the store-backed ``array_shape``, and it
    is what makes a ``records`` pushdown (``conformance.md``, "NetCDF decode
    notes") expressible out of process: the records a cadence owner wants are a
    function of the file's length along the record axis, so that length has to be
    known before the decode the selection is meant to narrow. The Python
    realization of the Julia ``dim_length`` generic and the Rust
    ``Reader::dim_length``.
    """
    fn = getattr(reader, "dim_length", None)
    return None if fn is None else fn(path, dim)


def supports_selection(reader: Any) -> bool:
    """Orthogonal-selection (projection-pushdown) capability trait — default-off.

    ``True`` when ``reader`` can honour a per-axis ``select`` at read time, fetching
    only the intersecting chunk objects. A store-backed reader that supports it
    declares a class attribute ``supports_selection = True`` (like ``store_backed``);
    whole-file readers inherit the ``False`` default. This is the Python realization
    of the Julia ``supports_selection`` trait and the Rust ``Reader::supports_selection``
    — the seam a caller (EarthSciAST) uses to decide whether to push a projection
    down (``Provider.materialize(..., select=...)``) or read whole and slice itself.
    """
    return bool(getattr(reader, "supports_selection", False))


# --------------------------------------------------------------------------- #
# Interfaces (spec/registries.md §1–§3), as runtime-checkable Protocols.
#
# Pseudo-signatures from the spec bound to Python idiom. They are
# ``runtime_checkable`` so a backend can be asserted interface-conformant with
# ``isinstance(impl, Transport)`` (structural: presence of the methods).
# --------------------------------------------------------------------------- #


@runtime_checkable
class Transport(Protocol):
    """Fetches a resolved URL's bytes into the cache. Keyed by URL scheme.

    Never constructed in offline mode (the transport is bypassed entirely).
    """

    def schemes(self) -> List[str]:
        """URL schemes this transport serves, e.g. ``["http", "https"]``."""
        ...

    def fetch(
        self,
        resolved_url: str,
        dest: Any,
        conditional: Optional[Dict[str, Any]] = None,
        auth: Optional[Any] = None,
    ) -> Any:
        """Download ``resolved_url`` to the staging path ``dest``.

        ``conditional`` carries ``etag``/``last_modified`` for revalidation;
        ``auth`` is an optional pluggable auth resolver. Returns a
        ``FetchResult`` (``status``/``etag``/``last_modified``/``bytes_written``).
        """
        ...


@runtime_checkable
class Reader(Protocol):
    """Opens a cached blob and returns CF-decoded native-grid arrays.

    Keyed by format name. Returns arrays keyed by the on-disk ``file_variable``
    name; does **not** remap variable names or apply ``unit_conversion`` (those
    are ESS contract semantics — spec §2, Risk R3).
    """

    def formats(self) -> List[str]:
        """Format names this reader handles, e.g. ``["netcdf"]``."""
        ...

    def extensions(self) -> List[str]:
        """Filename-extension sniff hints, e.g. ``["nc", "nc4"]``."""
        ...

    def open(self, blob_path: Any) -> Any:
        """Open the cached blob, returning an opaque handle."""
        ...

    def read_native(
        self,
        handle: Any,
        variables: List[str],
        select: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Read ``variables`` (file_variable names) into native fields + coords."""
        ...

    # --- optional: the store-backed capability (additive; default-off) ----- #
    #
    # A reader whose source is NOT one fetchable blob but a directory-like store
    # (a Zarr v2 store, whose ``.zarray``/``.zattrs``/chunks are each their own
    # object) may declare itself **store-backed** by exposing::
    #
    #     store_backed: bool = True
    #     def read_store(self, cache, base_url, variables, select=None, **kwargs)
    #             -> NativeDataset: ...
    #
    # The Provider gates on ``getattr(reader, "store_backed", False)`` and, when
    # true, calls ``read_store(cache, base_url, variables, **reader_kwargs)`` —
    # handing the reader the cache + the base URL so it can fetch individual
    # objects on demand — instead of pre-fetching one blob and calling
    # ``read_native`` (``earthsciio.provider.Provider._read_file``). Active
    # whole-file readers define **neither** attribute and are wholly unaffected.


@runtime_checkable
class Store(Protocol):
    """The physical home of the content-addressed cache. Keyed by store name.

    Realizes the cache layout in ``spec/cache-format.md``; the cache **key**
    (``sha256(resolved_url)``) is store-independent.
    """

    def name(self) -> str:
        """The store's registry name, e.g. ``"local"``."""
        ...

    def exists(self, key: str) -> bool:
        """Whether a valid blob is present for ``key``."""
        ...

    def get_blob(self, key: str) -> Any:
        """Return the blob (path/bytes) for ``key``, or ``None`` on a miss."""
        ...

    def put_blob(self, key: str, staged: Any) -> None:
        """Atomically commit a staged blob into the cache under ``key``."""
        ...

    def get_meta(self, key: str) -> Any:
        """Return the manifest for ``key``, or ``None`` if absent."""
        ...

    def put_meta(self, key: str, manifest: Any) -> None:
        """Persist the manifest for ``key``."""
        ...

    def lock(self, key: str) -> Any:
        """Acquire the per-blob advisory lock (scope = one blob fetch)."""
        ...


T = TypeVar("T")


# --------------------------------------------------------------------------- #
# The registry mechanism.
# --------------------------------------------------------------------------- #


def _module_available(module: str) -> bool:
    """Is ``module`` importable, without importing it?

    ``find_spec`` does the search but not the execution, so probing a heavy
    optional backend (rasterio pulls GDAL) costs a filesystem walk rather than a
    module init. It raises rather than returning None when a *parent* package is
    missing, and can raise ValueError for a half-initialized module, so both are
    treated as "not available".
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # missing parent package, or __spec__ is None
        return False


class RegistryEntry(Generic[T]):
    """One registered backend: its name, factory, status, lookup keys, metadata."""

    __slots__ = ("name", "factory", "status", "keys", "meta")

    def __init__(
        self,
        name: str,
        factory: Callable[..., T],
        status: str,
        keys: Tuple[str, ...],
        meta: Dict[str, Any],
    ) -> None:
        self.name = name
        self.factory = factory
        self.status = status
        self.keys = keys
        self.meta = meta

    @property
    def is_stub(self) -> bool:
        return self.status == "stub"

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"RegistryEntry(name={self.name!r}, status={self.status!r}, "
            f"keys={list(self.keys)!r})"
        )


class Registry(Generic[T]):
    """A name → implementation lookup for one extensibility seam.

    Mechanically a string-keyed map; the key *semantics* differ per registry
    (URL scheme / format name / store name — see :attr:`keyed_by`). One
    implementation may answer to several keys (e.g. an ``http`` transport serves
    both ``http`` and ``https``).

    Registration is idempotent for the same factory and refuses to silently
    reassign a name or key to a *different* implementation — so two concurrent
    contributions that both register ``s3`` fail loud rather than shadowing each
    other.
    """

    def __init__(self, kind: str, keyed_by: str) -> None:
        self.kind = kind
        self.keyed_by = keyed_by
        self._entries: Dict[str, RegistryEntry[T]] = {}
        self._by_key: Dict[str, RegistryEntry[T]] = {}

    # -- registration ------------------------------------------------------- #

    def register(
        self,
        name: str,
        factory: Callable[..., T],
        *,
        keys: Optional[Iterable[str]] = None,
        status: str = "active",
        **meta: Any,
    ) -> RegistryEntry[T]:
        """Register ``factory`` under ``name`` (and ``keys``, default ``[name]``).

        ``status`` is ``"active"`` or ``"stub"``. Extra keyword metadata
        (``extensions``, ``content_types``, ``tracking``, ``notes`` …) is kept
        on the entry. Returns the entry; idempotent if the same factory is
        re-registered (safe under repeated imports).

        ``requires`` declares the third-party modules a backend needs at decode
        time, as a tuple of ALTERNATIVE GROUPS: every group must be satisfied,
        and any one module within a group satisfies it. The geotiff reader's
        ``(("rasterio", "tifffile"),)`` reads "rasterio OR tifffile". Status and
        availability are deliberately independent axes — an active backend with
        an uninstalled optional stack stays registered, so asking for it raises
        the reader's own "install the extra" error rather than
        :class:`BackendNotRegistered`. Callers that need to know what this
        environment can actually decode ask :meth:`available`.
        """
        if status not in ("active", "stub"):
            raise ValueError(
                f"{self.kind} registry: status must be 'active' or 'stub', "
                f"got {status!r}"
            )
        key_tuple: Tuple[str, ...] = tuple(keys) if keys is not None else (name,)

        existing = self._entries.get(name)
        if existing is not None:
            if existing.factory is factory:
                return existing  # idempotent re-registration
            raise ValueError(
                f"{self.kind} registry: name {name!r} is already registered to a "
                f"different implementation"
            )
        for k in key_tuple:
            other = self._by_key.get(k)
            if other is not None and other.factory is not factory:
                raise ValueError(
                    f"{self.kind} registry: {self.keyed_by}={k!r} already maps to "
                    f"backend {other.name!r}"
                )

        entry: RegistryEntry[T] = RegistryEntry(
            name, factory, status, key_tuple, dict(meta)
        )
        self._entries[name] = entry
        for k in key_tuple:
            self._by_key[k] = entry
        return entry

    def unregister(self, name: str) -> None:
        """Remove a backend by name (used by tests; not a runtime path)."""
        entry = self._entries.pop(name, None)
        if entry is None:
            return
        for k in entry.keys:
            if self._by_key.get(k) is entry:
                del self._by_key[k]

    # -- lookup ------------------------------------------------------------- #

    def __contains__(self, key: object) -> bool:
        return key in self._by_key

    def is_registered(self, key: str) -> bool:
        return key in self._by_key

    def entry(self, key: str) -> RegistryEntry[T]:
        """Resolve ``key`` to its entry, or raise :class:`BackendNotRegistered`."""
        try:
            return self._by_key[key]
        except KeyError:
            raise BackendNotRegistered(
                self.kind, self.keyed_by, key, self._by_key.keys()
            ) from None

    def factory(self, key: str) -> Callable[..., T]:
        """The factory (class/callable) registered for ``key``."""
        return self.entry(key).factory

    def status(self, key: str) -> str:
        """``"active"`` or ``"stub"`` for ``key`` (raises if unregistered)."""
        return self.entry(key).status

    def is_stub(self, key: str) -> bool:
        return self.entry(key).status == "stub"

    def missing_requirements(self, key: str) -> List[str]:
        """The unsatisfied ``requires`` groups for ``key``, as ``"a or b"`` strings.

        Empty when the backend can run here. See :meth:`register` for the shape
        of ``requires``.
        """
        groups = self.entry(key).meta.get("requires") or ()
        return [
            " or ".join(group)
            for group in groups
            if not any(_module_available(m) for m in group)
        ]

    def available(self, key: str) -> bool:
        """Can ``key`` actually decode in THIS environment?

        ``status`` says what the library implements; this says what the installed
        dependencies permit. A backend that declares no ``requires`` is always
        available. Registration is not conditioned on this on purpose — see
        :meth:`register`.
        """
        return not self.missing_requirements(key)

    def create(self, key: str, *args: Any, **kwargs: Any) -> T:
        """Resolve ``key`` and construct the implementation.

        This is the dispatch the Provider uses. For a *stub* backend the
        construction succeeds (it is interface-conformant); calling one of its
        operations is what raises :class:`~earthsciio.errors.Unsupported`.
        """
        return self.entry(key).factory(*args, **kwargs)

    # -- introspection ------------------------------------------------------ #

    def names(self) -> List[str]:
        """Registered implementation names, sorted."""
        return sorted(self._entries)

    def keys(self) -> List[str]:
        """All lookup keys (schemes/formats/store-names), sorted."""
        return sorted(self._by_key)

    def describe(self) -> Dict[str, Dict[str, Any]]:
        """``{name: {status, keys, meta}}`` — a snapshot for tests/diagnostics."""
        return {
            name: {
                "status": e.status,
                "available": self.available(e.keys[0]),
                "keys": list(e.keys),
                "meta": dict(e.meta),
            }
            for name, e in sorted(self._entries.items())
        }

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"Registry(kind={self.kind!r}, keyed_by={self.keyed_by!r}, "
            f"backends={self.names()!r})"
        )


# --------------------------------------------------------------------------- #
# The three registry singletons (match spec/registries.json `keyed_by`).
# Active backends register from the core track; stubs from earthsciio.backends.
# --------------------------------------------------------------------------- #

transport_registry: "Registry[Transport]" = Registry("transport", keyed_by="url_scheme")
format_registry: "Registry[Reader]" = Registry("format", keyed_by="format_name")
store_registry: "Registry[Store]" = Registry("store", keyed_by="store_name")


def all_registries() -> Dict[str, Registry]:
    """Map of registry kind → the registry singleton (for spec-parity checks)."""
    return {
        "transport": transport_registry,
        "format": format_registry,
        "store": store_registry,
    }

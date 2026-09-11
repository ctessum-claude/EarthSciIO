"""The cadence-aware **Provider** (component (b)) — the sanctioned impure I/O
boundary for one ESS ``DataSource`` node.

A :class:`Provider` is bound to one loader and created once per simulation. It
resolves a URL (per cadence anchor, for a time-varying source), fetches it
through the content-addressed cache (component (a), :mod:`earthsciio.cache`),
decodes it with the loader's :data:`~earthsciio.registry.format_registry` reader,
and returns RAW native-grid arrays (:class:`~earthsciio.native.NativeDataset`).
Variable-name remap / ``unit_conversion`` stay in ESS; regrid is ESD/C4's job
(Risk R3) — the Provider returns native arrays and nothing more.

It provides DATA, not a solver. The library EXPOSES the provider and its
:meth:`Provider.refresh_times`; the **user/solver** drives the discrete-cadence
update (e.g. a ``DifferentialEquations`` ``PresetTimeCallback`` /
``diffeqpy``/SciPy event that calls :meth:`Provider.refresh` at each anchor). No
solver is embedded here. The refresh is wired to fire **once per cadence
boundary** in the solver callback — NEVER per-RHS — exactly matching campfire C1
(``ess-06y``) terminal-event segmentation; downstream is C1 (the cadence
callback) + C4 (the regrid handoff), integration verified campfire-side.

CONST vs DISCRETE (the cadence contract):

* **CONST** — ``loader.temporal is None``: :meth:`materialize` reads the single
  file once; :meth:`refresh_times` is empty; :meth:`refresh` returns that same
  constant data.
* **DISCRETE** — ``loader.temporal`` set: :meth:`refresh` snaps a time to the
  loader's cadence **anchor** and returns the matching record's native arrays;
  :meth:`refresh_times` is the cadence schedule over the run window (the solver
  tstops). A multi-record file is sliced on its ``time_dim`` axis; a file-per-tick
  source resolves a new URL per file period.

This mirrors the peer Rust ``Provider`` (``rust/src/provider.rs``) and Julia
``Provider`` (``julia/src/provider.jl``); the decoded native arrays are equal
across all three tracks (conformance ``esio-9nb.9``).
"""

from __future__ import annotations

import datetime as _dt
import inspect
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from .cache import Cache, CacheEntry
from .errors import UnknownReaderOption
from .native import NativeDataset, NativeField
from .registry import Registry, format_registry, supports_selection

__all__ = ["SourceTemporal", "DataSource", "Provider", "Window",
           "LoaderTemporal", "DataLoader"]

#: A run window ``(start, end)`` — half-open ``[start, end)``.
Window = Tuple[_dt.datetime, _dt.datetime]

#: A URL spec: a literal URL, a ``strftime`` template (contains ``%``), or a
#: callable ``anchor -> url`` (the most general, per-file-anchor resolver).
UrlSpec = Union[str, Callable[[_dt.datetime], str]]

_EPOCH = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
_ZERO = _dt.timedelta(0)


def _snap_down(start: _dt.datetime, t: _dt.datetime, step: _dt.timedelta) -> _dt.datetime:
    """The largest aligned anchor ``start + k*step <= t`` (``k`` an integer).

    Timedelta floor-division floors toward -infinity, so this snaps ``t`` *down*
    to its cadence anchor even before ``start`` (caller validates the range).
    """
    return start + ((t - start) // step) * step


@dataclass(frozen=True)
class SourceTemporal:
    """The temporal cadence of a DISCRETE loader (absent ⇒ CONST).

    ``frequency`` is the cadence step (e.g. 1 hour for ERA5) — it drives the
    refresh tstops and which record within a file is current. ``file_period`` is
    the span of one file (e.g. 1 day) — it drives URL resolution; a file holds
    ``file_period / frequency`` records along ``time_dim``. Anchors are aligned
    to ``start`` (the loader epoch). ``end`` (optional) is the exclusive end of
    available data and bounds :meth:`Provider.refresh_times` when no run window
    is given.

    ``records_per_sample`` is HOW MANY time records :meth:`Provider.refresh`
    returns per query time — pure I/O; the loader does not interpolate. ``None``
    or ``1`` (default) returns the single at-or-before record with ``time_dim``
    dropped (held piecewise-constant); ``2`` returns the two bracketing records
    (floor + successor) with ``time_dim`` retained at length 2 and a canonical
    2-element ``time_dim`` coordinate of Unix epoch seconds, so a downstream model
    can interpolate in time. The successor is read across a file boundary when
    needed; at the last available record the bracket degenerates to ``[last, last]``
    so the downstream weight clamps. Only 1 and 2 are supported; higher-order
    temporal stencils are future work. (Distinct from a per-file record count.)
    """

    start: _dt.datetime
    frequency: _dt.timedelta
    file_period: _dt.timedelta
    end: Optional[_dt.datetime] = None
    time_dim: str = "time"
    records_per_sample: Optional[int] = None

    def __post_init__(self) -> None:
        if self.frequency <= _ZERO:
            raise ValueError(f"frequency must be positive, got {self.frequency!r}")
        if self.file_period <= _ZERO:
            raise ValueError(f"file_period must be positive, got {self.file_period!r}")
        if self.records_per_sample not in (None, 1, 2):
            raise ValueError(
                f"records_per_sample must be 1 or 2, got {self.records_per_sample!r}"
            )


@dataclass
class DataSource:
    """The I/O-relevant projection of an ESS ``DataSource`` the Provider needs.

    The full ESM contract (units, variable remap, grid family) lives upstream in
    ESS; this is only what resolves bytes and decodes them.

    Parameters
    ----------
    name:
        Loader name — provenance, recorded in the cache manifest.
    format:
        Format-registry key selecting the reader (e.g. ``"netcdf"``, ``"csv"``).
    url:
        A literal URL (CONST), a ``strftime`` template resolved at each file
        anchor (e.g. ``".../era5/%Y/%m/%Y%m%d.nc"``), or a callable
        ``anchor -> url``.
    variables:
        On-disk ``file_variable`` names to read; empty ⇒ all data variables
        (coordinates are always kept).
    temporal:
        The cadence (``None`` ⇒ CONST/static).
    mirrors:
        Failover URL candidates sharing the same cache identity.
    auth_realm:
        Auth realm to fetch under (resolved by the cache's auth registry).
    reader_kwargs:
        Extra keywords forwarded to the reader's ``read_native`` (e.g. the CSV
        reader's ``numeric_columns``).
    """

    name: str
    format: str
    url: UrlSpec
    variables: Sequence[str] = ()
    temporal: Optional[SourceTemporal] = None
    mirrors: Sequence[str] = ()
    auth_realm: Optional[str] = None
    reader_kwargs: Dict[str, object] = field(default_factory=dict)

    @property
    def is_const(self) -> bool:
        """True if the loader is time-invariant (no :class:`SourceTemporal`)."""
        return self.temporal is None

    def resolve_url(self, anchor: _dt.datetime) -> str:
        """Resolve the file URL for a cadence/file anchor."""
        u = self.url
        if callable(u):
            return u(anchor)
        if "%" in u:
            return anchor.strftime(u)
        return u


#: Decode-entry-point arguments that are NOT reader options: the blob/store
#: handle and the variable list, which every caller supplies from the loader.
_NOT_AN_OPTION = frozenset({"self", "handle", "cache", "base_url", "variables"})


def reader_option_names(reader: Any) -> Set[str]:
    """The decode-option names ``reader`` recognises (``spec/registries.md`` §2.1).

    A Python reader **declares** its options as the keyword parameters of its
    decode entry point — ``read_store`` for a store-backed reader, else
    ``read_native`` — so that signature is the single source of truth and an
    option list cannot drift from the code that consumes it. (This is where the
    Python idiom parts company with Rust's ``Reader::configured``, which needs an
    explicit per-reader list because a trait method has no introspectable kwargs;
    the accepted SET is the same either way, which is what conformance compares.)

    The ``**_`` catch-all every reader carries is deliberately NOT read as
    "accepts anything": swallowing an unknown key is precisely the failure mode
    §2.1 forbids. A reader whose signature cannot be introspected at all (a C
    extension, a ``functools.partial``) reports every declared option as
    accepted rather than inventing a rejection it cannot justify.
    """
    fn = getattr(
        reader,
        "read_store" if getattr(reader, "store_backed", False) else "read_native",
        None,
    )
    if fn is None:
        return set()
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # unintrospectable — cannot police it
        return set()
    return {
        name
        for name, p in params.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and name not in _NOT_AN_OPTION
    }


def check_reader_options(reader: Any, loader: "DataSource") -> None:
    """Reject a ``reader_kwargs`` key the bound ``reader`` does not recognise.

    Called once, when the :class:`Provider` is built, so a mis-typed decode
    option fails there rather than quietly selecting nothing (the Rust binding
    performs the same check in ``Reader::configured``; ``spec/registries.md``
    §2.1 requires it of every binding).
    """
    options = loader.reader_kwargs or {}
    if not options:
        return
    accepted = reader_option_names(reader)
    unknown = [k for k in options if k not in accepted]
    if unknown:
        raise UnknownReaderOption(loader.format, unknown, accepted)


class Provider:
    """A loader-bound provider of native-grid arrays, refreshed at the loader's
    cadence. See the module docstring for the CONST/DISCRETE contract.

    Parameters
    ----------
    loader:
        The :class:`DataSource` this provider serves.
    cache:
        The content-addressed :class:`~earthsciio.cache.Cache` (component (a)).
    window:
        Optional run window ``(start, end)`` bounding :meth:`refresh_times` /
        :meth:`prefetch` and priming :meth:`materialize` for a DISCRETE loader.
    formats:
        The format registry to resolve the reader from (defaults to the global
        :data:`~earthsciio.registry.format_registry`) — the seam that lets a new
        format plug in with no Provider change.

    Raises :class:`~earthsciio.errors.BackendNotRegistered` if no reader is
    registered for ``loader.format``.
    """

    def __init__(
        self,
        loader: DataSource,
        cache: Cache,
        window: Optional[Window] = None,
        *,
        formats: Optional[Registry] = None,
    ) -> None:
        self.loader = loader
        self.cache = cache
        self.window = window
        registry = formats if formats is not None else format_registry
        # Resolve (and construct) the reader by name now, so an unknown format
        # fails at construction, not mid-solve.
        self._reader = registry.create(loader.format)
        # ... and resolve the loader's DECLARED decode options against it, for
        # the same reason: an unrecognised reader_kwarg is an error here, never
        # a key the decode silently ignores (spec/registries.md §2.1).
        check_reader_options(self._reader, loader)
        # Current state — a file-read cache so stepping within one file decodes
        # it once, and the buffer the solver reads between cadence boundaries.
        self._current: Optional[NativeDataset] = None
        self._current_anchor: Optional[_dt.datetime] = None
        self._current_file_anchor: Optional[_dt.datetime] = None
        self._current_file: Optional[NativeDataset] = None
        # Tiny LRU of decoded files keyed by file anchor — bracket mode may need
        # two adjacent files open at a file-period seam (the "after" record of the
        # last record in a file lives in the next file). Cap 2.
        self._files: "OrderedDict[_dt.datetime, NativeDataset]" = OrderedDict()

    # -- introspection ------------------------------------------------------ #

    @property
    def is_const(self) -> bool:
        """True if the bound loader is time-invariant (CONST)."""
        return self.loader.temporal is None

    @property
    def coords(self) -> Dict[str, NativeField]:
        """Native coordinates of the current buffer (lat/lon/…); empty until the
        first :meth:`materialize`/:meth:`refresh`."""
        return self._current.coords if self._current is not None else {}

    # -- materialize / refresh --------------------------------------------- #

    def materialize(self, select: Optional[Any] = None) -> NativeDataset:
        """Materialize the loader's native arrays into the buffer and return them.

        CONST: reads the single file once. DISCRETE: primes the buffer at the
        first cadence anchor of the window (≡ ``refresh(window.start)``), so a
        caller can read an initial state before stepping.

        ``select`` is an optional PER-CALL projection pushdown (in EarthSciIO's
        native select shape, e.g. ``{"axes": [...]}`` with 0-based indices). When
        supplied it OVERRIDES any baked ``reader_kwargs["select"]`` for this call
        only, letting a caller (EarthSciAST) push a projection down at sample time
        without rebuilding the provider. Any reader that answers
        :attr:`supports_selection` can honour it — ``zarr`` by fetching only the
        intersecting chunk objects, ``netcdf`` by materialising only the requested
        hyperslab of the blob it already fetched; passing ``select`` to a reader
        that cannot is a :class:`ValueError`. A per-call ``select`` is a projection
        **peek** — it does NOT disturb the cadence buffer / file cache the solver
        reads.
        """
        if self.loader.temporal is None:
            ds = self._read_file(self.loader.resolve_url(_EPOCH), select=select)
            if select is None:
                self._current = ds
            return ds
        return self.refresh(self._lower_bound(), select=select)

    def refresh(self, t: _dt.datetime, select: Optional[Any] = None) -> NativeDataset:
        """Refresh the buffer to the cadence anchor for time ``t`` and return it.

        Snaps ``t`` down to the loader's cadence anchor, re-reads the covering
        file only when the **file** anchor changed (stepping within one file
        decodes it once), slices the current record on ``time_dim``, and returns
        the native arrays. For a CONST loader this returns the constant data
        (materializing once if needed). Raises :class:`ValueError` if ``t``
        precedes the loader epoch, :class:`IndexError` if the snapped record is
        absent from its file.

        ``select`` is the per-call projection pushdown override (see
        :meth:`materialize`) — a peek that reads fresh and leaves the buffer intact.
        """
        temporal = self.loader.temporal
        if temporal is None:
            if select is None and self._current is not None:
                return self._current
            return self.materialize(select=select)

        anchor = _snap_down(temporal.start, t, temporal.frequency)
        if anchor < temporal.start:
            raise ValueError(
                f"t={t!r} precedes the loader start {temporal.start!r}"
            )
        if temporal.records_per_sample == 2:
            return self._refresh_bracket(temporal, anchor, select=select)
        file_anchor = _snap_down(temporal.start, anchor, temporal.file_period)
        # A per-call `select` override reads fresh and does NOT disturb the cadence
        # buffer / file cache (it is a projection peek for this call only).
        if select is not None:
            file_ds = self._read_file(
                self.loader.resolve_url(file_anchor), select=select
            )
            rec = _record_index(file_ds, temporal, anchor, file_anchor)
            return _slice_record(file_ds, temporal.time_dim, rec)
        if self._current_file is None or self._current_file_anchor != file_anchor:
            self._current_file = self._read_file(self.loader.resolve_url(file_anchor))
            self._current_file_anchor = file_anchor

        rec = _record_index(self._current_file, temporal, anchor, file_anchor)
        sliced = _slice_record(self._current_file, temporal.time_dim, rec)
        self._current = sliced
        self._current_anchor = anchor
        return sliced

    # -- projection-pushdown capability ------------------------------------ #

    @property
    def supports_selection(self) -> bool:
        """True when the bound reader can honour an orthogonal ``select``
        **without materialising the whole array**. A caller uses this to decide
        whether to push a projection down (``materialize(..., select=…)``) or to
        read whole and slice on its own side.

        It says nothing about what is FETCHED — pair it with
        :attr:`store_backed` for that: ``zarr`` (store-backed) fetches only the
        intersecting chunk objects, so a selection shrinks the download too;
        ``netcdf`` (whole-file) fetches the same blob under the same cache key
        and only materialises the requested hyperslab."""
        return supports_selection(self._reader)

    @property
    def store_backed(self) -> bool:
        """True when the bound reader reads a directory-like STORE (a Zarr v2
        store, each object its own fetch) rather than one blob. Read with
        :attr:`supports_selection` it tells a caller whether a pushed-down
        ``select`` shrinks the DOWNLOAD or only the decode."""
        return bool(getattr(self._reader, "store_backed", False))

    def array_shape(self, var: str) -> Optional[Tuple[int, ...]]:
        """The full native (dims-order) shape of on-disk array ``var``, for a
        honour/refuse pushdown decision, or ``None``.

        For a store-backed zarr provider this reads ONLY the array's ``.zarray``
        metadata (never a chunk); ``None`` for a whole-file reader (whose shape is
        not knowable without reading the blob). Mirrors the Julia/Rust
        ``array_shape``."""
        reader = self._reader
        if not getattr(reader, "store_backed", False):
            return None
        shape_fn = getattr(reader, "array_shape", None)
        if shape_fn is None:
            return None
        anchor = _EPOCH if self.loader.temporal is None else self.loader.temporal.start
        return shape_fn(self.cache, self.loader.resolve_url(anchor), var)

    def refresh_times(self) -> List[_dt.datetime]:
        """The cadence anchors at which the data changes and the solver must
        :meth:`refresh` — the solver tstops.

        Empty for a CONST loader, or for an unbounded DISCRETE loader with no run
        window and no ``temporal.end``. Otherwise the aligned anchors in
        ``[lower, upper)`` where ``lower`` is the window start clamped to the
        loader epoch and ``upper`` is the window end (else ``temporal.end``).
        Each is a timezone-aware :class:`datetime.datetime`; a solver integrating
        in epoch seconds takes ``t.timestamp()``.
        """
        temporal = self.loader.temporal
        if temporal is None:
            return []
        upper = self.window[1] if self.window is not None else temporal.end
        if upper is None:
            return []  # unbounded — no enumerable schedule
        lower = self._lower_bound()
        # First aligned anchor >= lower (ceil of the elapsed step count).
        steps = -((temporal.start - lower) // temporal.frequency)
        anchor = temporal.start + steps * temporal.frequency
        out: List[_dt.datetime] = []
        while anchor < upper:
            out.append(anchor)
            anchor = anchor + temporal.frequency
        return out

    # -- prefetch ----------------------------------------------------------- #

    def prefetch(self, window: Optional[Window] = None) -> List[CacheEntry]:
        """Warm the cache for every file the provider will need, WITHOUT decoding.

        CONST: the single file. DISCRETE: each unique file covering the window
        (``window`` arg, else the provider's window). Lets a caller pull all
        blobs up front (e.g. before a solve, or while online) so later
        :meth:`materialize`/:meth:`refresh` hit a warm, offline-readable cache.
        Returns the :class:`~earthsciio.cache.CacheEntry` for each unique file.
        """
        temporal = self.loader.temporal
        if temporal is None:
            return [self._fetch(self.loader.resolve_url(_EPOCH))]

        win = window if window is not None else self.window
        upper = win[1] if win is not None else temporal.end
        if upper is None:
            raise ValueError(
                "prefetch needs a bounded window: pass window=(start, end) or set "
                "SourceTemporal.end"
            )
        start = win[0] if (win is not None and win[0] > temporal.start) else temporal.start
        entries: List[CacheEntry] = []
        seen: set = set()
        file_anchor = _snap_down(temporal.start, start, temporal.file_period)
        while file_anchor < upper:
            url = self.loader.resolve_url(file_anchor)
            if url not in seen:
                seen.add(url)
                entries.append(self._fetch(url))
            file_anchor = file_anchor + temporal.file_period
        return entries

    # -- internals ---------------------------------------------------------- #

    def _lower_bound(self) -> _dt.datetime:
        """The effective lower bound: the window start clamped to the loader
        epoch (a DISCRETE provider always has a temporal)."""
        temporal = self.loader.temporal
        assert temporal is not None
        if self.window is not None and self.window[0] > temporal.start:
            return self.window[0]
        return temporal.start

    def _fetch(self, url: str) -> CacheEntry:
        return self.cache.fetch(
            url,
            source_loader=self.loader.name,
            auth_realm=self.loader.auth_realm,
            mirrors=tuple(self.loader.mirrors),
        )

    def _read_file(self, url: str, select: Optional[Any] = None) -> NativeDataset:
        variables = list(self.loader.variables) if self.loader.variables else None
        reader_kwargs = self.loader.reader_kwargs
        # Store-backed readers (e.g. zarr) are handed (cache, base_url, variables,
        # select) — a Zarr `url` is a directory-like prefix, not a fetchable blob,
        # so the reader fetches individual objects on demand. Additive + default-
        # off: active whole-file readers never define `store_backed` and take the
        # existing path.
        #
        # Effective select: a per-call `select` OVERRIDES the baked
        # reader_kwargs["select"] for this call only. Forward it explicitly and
        # splat the REST of reader_kwargs (with "select" removed, so `select` is
        # never passed twice).
        if getattr(self._reader, "store_backed", False):
            effective = select if select is not None else reader_kwargs.get("select")
            rest = {k: v for k, v in reader_kwargs.items() if k != "select"}
            return self._reader.read_store(
                self.cache, url, variables, select=effective, **rest
            )
        # Whole-file reader. A `select` — per-call, else the baked one — reaches
        # it only if it declares `supports_selection`; the netcdf reader does, and
        # honours it at DECODE time (same blob, same cache key, only the requested
        # hyperslab materialised). A reader that cannot is a clear error (the
        # fetch-full fallback belongs to the EarthSciAST caller, not the reader),
        # raised before any fetch.
        effective = select if select is not None else reader_kwargs.get("select")
        if effective is not None and not supports_selection(self._reader):
            raise ValueError(
                f"reader {type(self._reader).__name__} does not support "
                "select/pushdown"
            )
        rest = {k: v for k, v in reader_kwargs.items() if k != "select"}
        entry = self._fetch(url)
        handle = self._reader.open(entry.path)
        if effective is None:
            return self._reader.read_native(handle, variables, **rest)
        return self._reader.read_native(handle, variables, select=effective, **rest)

    def _file_for(self, file_anchor: _dt.datetime,
                  select: Optional[Any] = None) -> NativeDataset:
        """Decode the file covering ``file_anchor``, via a 2-entry LRU so a
        file-period seam (two adjacent files) decodes each at most once.

        A per-call ``select`` override reads fresh and bypasses the decoded-file
        LRU (the projected decode must not be cached as the plain buffer)."""
        if select is not None:
            return self._read_file(self.loader.resolve_url(file_anchor), select=select)
        ds = self._files.get(file_anchor)
        if ds is None:
            ds = self._read_file(self.loader.resolve_url(file_anchor))
            self._files[file_anchor] = ds
            while len(self._files) > 2:
                self._files.popitem(last=False)
        else:
            self._files.move_to_end(file_anchor)
        return ds

    def _refresh_bracket(self, temporal: SourceTemporal,
                         anchor: _dt.datetime,
                         select: Optional[Any] = None) -> NativeDataset:
        """Return the two records bracketing ``anchor`` (floor + successor) with
        ``time_dim`` retained at length 2 and a canonical epoch-seconds ``time``
        coordinate. Handles the cross-file successor and the end-of-data clamp.

        A per-call ``select`` override reads fresh and does NOT disturb the cadence
        buffer (a projection peek for this call only)."""
        time_dim = temporal.time_dim
        file0_anchor = _snap_down(temporal.start, anchor, temporal.file_period)
        file0 = self._file_for(file0_anchor, select=select)
        rec0 = _record_index(file0, temporal, anchor, file0_anchor)

        next_anchor = anchor + temporal.frequency
        has_succ = temporal.end is None or next_anchor < temporal.end
        succ: Optional[Tuple[NativeDataset, int]] = None
        if has_succ:
            try:
                if rec0 + 1 < _time_len(file0, time_dim):
                    succ = (file0, rec0 + 1)  # successor in the same file
                else:  # successor is record 0 of the next file
                    n_anchor = _snap_down(temporal.start, next_anchor,
                                          temporal.file_period)
                    file1 = self._file_for(n_anchor, select=select)
                    r1 = _record_index(file1, temporal, next_anchor, n_anchor)
                    if 0 <= r1 < _time_len(file1, time_dim):
                        succ = (file1, r1)
            except Exception:
                succ = None  # no reachable successor — clamp below
        if succ is None:  # end-of-data: degenerate bracket, hold the last record
            next_anchor = anchor
            succ = (file0, rec0)

        sliced = _bracket_record(
            file0, rec0, succ[0], succ[1], time_dim,
            _epoch_seconds(anchor), _epoch_seconds(next_anchor),
        )
        if select is None:
            self._current = sliced
            self._current_anchor = anchor
        return sliced


def _slice_record(ds: NativeDataset, time_dim: str, rec: int) -> NativeDataset:
    """Slice cadence record ``rec`` out of every field carrying ``time_dim``.

    Non-temporal fields pass through whole; the sliced ``time_dim`` coordinate is
    dropped (its value for this record is implied by the anchor). Mirrors the
    Julia ``_slice_dim`` / Rust ``select_leading``.
    """
    variables = {name: _slice_field(f, time_dim, rec) for name, f in ds.variables.items()}
    coords = {
        name: _slice_field(f, time_dim, rec)
        for name, f in ds.coords.items()
        if name != time_dim
    }
    return NativeDataset(variables, coords)


def _slice_field(f: NativeField, time_dim: str, rec: int) -> NativeField:
    if time_dim not in f.dims:
        return f  # non-temporal field unchanged
    axis = f.dims.index(time_dim)
    length = f.data.shape[axis]
    if rec < 0 or rec >= length:
        raise IndexError(
            f"cadence record {rec} out of range for dimension {time_dim!r} "
            f"(length {length})"
        )
    sliced = np.take(f.data, rec, axis=axis)
    new_dims = tuple(d for d in f.dims if d != time_dim)
    return NativeField(sliced, new_dims, f.attrs)


def _time_len(ds: NativeDataset, time_dim: str) -> int:
    """Length of ``ds`` along ``time_dim`` (from the time coord, else the first
    variable carrying it); 0 if no field carries it."""
    coord = ds.coords.get(time_dim)
    if coord is not None and time_dim in coord.dims:
        return int(coord.data.shape[coord.dims.index(time_dim)])
    for f in ds.variables.values():
        if time_dim in f.dims:
            return int(f.data.shape[f.dims.index(time_dim)])
    return 0


def _stack_two(f0: NativeField, f1: NativeField, time_dim: str,
               rec0: int, rec1: int) -> NativeField:
    """Stack record ``rec0`` of ``f0`` and ``rec1`` of ``f1`` along ``time_dim``,
    keeping ``time_dim`` at length 2. Non-temporal fields pass through unchanged."""
    if time_dim not in f0.dims:
        return f0
    axis = f0.dims.index(time_dim)
    a = np.take(f0.data, rec0, axis=axis)
    b = np.take(f1.data, rec1, axis=axis)
    return NativeField(np.stack([a, b], axis=axis), f0.dims, f0.attrs)


def _bracket_record(file0: NativeDataset, rec0: int, file1: NativeDataset, rec1: int,
                    time_dim: str, t0_epoch: float, t1_epoch: float) -> NativeDataset:
    """Assemble the 2-record bracket dataset: every temporal variable stacked to a
    size-2 ``time_dim`` axis, non-temporal coords passed through, and a canonical
    2-element ``time_dim`` coordinate of Unix epoch seconds ``[t0, t1]``."""
    variables = {
        name: _stack_two(f, file1.variables[name], time_dim, rec0, rec1)
        for name, f in file0.variables.items()
    }
    coords = {
        name: f for name, f in file0.coords.items() if name != time_dim
    }
    coords[time_dim] = NativeField(
        np.array([t0_epoch, t1_epoch], dtype=np.float64),
        (time_dim,),
        {"units": "seconds since 1970-01-01T00:00:00Z", "calendar": "standard"},
    )
    return NativeDataset(variables, coords)


_EPOCH_NAIVE = _dt.datetime(1970, 1, 1)


def _epoch_seconds(d: _dt.datetime) -> float:
    """Unix epoch seconds for a datetime (tz-aware normalized to UTC; naive
    assumed UTC — matching :func:`_record_index`'s comparison convention)."""
    return (_to_naive_utc(d) - _EPOCH_NAIVE).total_seconds()


def _to_naive_utc(d: _dt.datetime) -> _dt.datetime:
    """Drop tz to naive UTC so decoded file times and cadence anchors compare."""
    if getattr(d, "tzinfo", None) is not None:
        return d.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    return d


def _decode_file_times(coord: NativeField) -> Optional[List[_dt.datetime]]:
    """Decode a raw CF time coordinate to naive-UTC datetimes, or ``None``.

    The netcdf reader returns the time axis undecoded (raw integers + a CF
    ``units`` like ``"seconds since 1970-01-01"``); decode it here so the record
    is selected by real timestamp. ``None`` for an undecodable axis (no
    ``units``, or a non-standard cftime calendar) — the caller falls back to the
    cadence-grid offset."""
    attrs = getattr(coord, "attrs", None) or {}
    units = attrs.get("units")
    if not units or "since" not in str(units):
        return None
    try:
        from xarray.coding.times import decode_cf_datetime

        decoded = np.asarray(
            decode_cf_datetime(np.asarray(coord.data), str(units),
                               attrs.get("calendar", "standard"))
        )
        if not np.issubdtype(decoded.dtype, np.datetime64):
            return None  # cftime (non-standard calendar) — fall back
        objs = decoded.astype("datetime64[s]").astype(object).ravel().tolist()
        return [_to_naive_utc(d) for d in objs]
    except Exception:
        return None


def _record_at_or_before(times: List[_dt.datetime], anchor: _dt.datetime) -> Optional[int]:
    """Index of the latest record at or before ``anchor`` (clamped to the
    earliest record when ``anchor`` precedes them all). Order-independent."""
    if not times:
        return None
    a = _to_naive_utc(anchor)
    best: Optional[int] = None
    for i, ti in enumerate(times):
        if ti <= a and (best is None or ti > times[best]):
            best = i
    if best is not None:
        return best
    return min(range(len(times)), key=lambda i: times[i])


def _record_index(file_ds: NativeDataset, temporal: Any, anchor: _dt.datetime,
                  file_anchor: _dt.datetime) -> int:
    """The record index for ``anchor`` within ``file_ds``.

    Prefer the file's REAL time axis (``temporal.time_dim`` coordinate): robust to
    a trimmed file (fewer records than ``file_period / frequency``) and to a
    ``file_period`` that only approximates a calendar month (so the computed
    offset can land in the wrong file / out of range — e.g. ERA5's ``P1M`` from a
    1940 epoch). Fall back to the cadence-grid offset when the axis can't be
    decoded."""
    coord = file_ds.coords.get(temporal.time_dim)
    if coord is not None:
        times = _decode_file_times(coord)
        if times:
            a = _to_naive_utc(anchor)
            # Use the real axis only when the anchor falls within the file's
            # actual span; an anchor beyond it is genuinely absent, so fall
            # through to the cadence-grid offset (which raises — the documented
            # out-of-range contract).
            if min(times) <= a <= max(times):
                idx = _record_at_or_before(times, a)
                if idx is not None:
                    return int(idx)
    return (anchor - file_anchor) // temporal.frequency


# --- 0.1.1 compatibility -------------------------------------------------
# From `.esm` 1.0.0 the declaration is a `data_sources` entry, not a
# `data_loaders` one, and the consumers spell it `DataSource`. These aliases
# keep 0.1.1 source-compatible; they will be dropped in 0.2.0.
DataLoader = DataSource
LoaderTemporal = SourceTemporal

"""The active, store-backed ``zarr`` reader (Zarr v2 chunked arrays).

Covers the chunk math, orthogonal selection, the partial edge chunk, the
fill-for-absent-chunk rule, the ``fill_value != NaN`` deviation, and — the
load-bearing capability — **laziness**: a runtime index list on a dimension
fetches ONLY the intersecting chunk objects, never the whole array. Laziness is
proven with a "poison" store: chunks that must NOT be read hold undecodable
garbage, so any over-fetch decode-errors instead of silently succeeding.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pytest

numcodecs = pytest.importorskip("numcodecs")  # builds the v2 test fixtures
pytest.importorskip("zarr")  # the reader is now built on zarr-python 3.x

from earthsciio import (
    Cache,
    CSVReader,
    DataSource,
    FF10Reader,
    Provider,
    cache_key,
    format_registry,
    supports_selection,
)
from earthsciio.backends.local import LocalStore
from earthsciio.backends.zarr import (
    ZarrReader,
    _parse_axis,
    _resolve_axis_indices,
)
from earthsciio.cachekey import sha256_bytes
from earthsciio.manifest import Manifest

BASE = "s3://earthsci-fixtures/mini.zarr"


def _blosc():
    return numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE, blocksize=0)


def _zarray(shape, chunks, dtype="<f4"):
    import json

    return json.dumps({
        "zarr_format": 2, "shape": list(shape), "chunks": list(chunks), "dtype": dtype,
        "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0},
        "fill_value": 0.0, "order": "C", "filters": None, "dimension_separator": None,
    }).encode()


def _encode_chunk(chunk):
    return bytes(_blosc().encode(np.ascontiguousarray(chunk)))


def _populate(root, objects):
    """Write ``{url: bytes}`` as offline cache blobs+manifests keyed by sha256(url)."""
    store = LocalStore(root)
    for url, data in objects.items():
        key = cache_key(url)
        staged = store.staging_path()
        staged.write_bytes(data)
        store.put_blob(key, staged, "")
        store.put_meta(key, Manifest(
            url=url, sha256_content=sha256_bytes(data), bytes=len(data),
            fetched_at="2026-06-26T00:00:00Z",
        ))


# --------------------------------------------------------------------------- #
# Selector parsing (the EarthSciIO ``select`` shape → zarr oindex key).
# --------------------------------------------------------------------------- #


def test_resolve_axis_slice_and_indices():
    assert _resolve_axis_indices(_parse_axis("all"), 4) == [0, 1, 2, 3]
    assert _resolve_axis_indices(_parse_axis({"indices": [3, 0, 1]}), 4) == [3, 0, 1]
    assert _resolve_axis_indices(_parse_axis({"slice": [1, 8, 2]}), 10) == [1, 3, 5, 7]
    with pytest.raises(IndexError):
        _resolve_axis_indices(_parse_axis({"indices": [9]}), 4)


# --------------------------------------------------------------------------- #
# read_store over a small file-backed store (offline cache).
# --------------------------------------------------------------------------- #


def _mini_store(root):
    """field3d [2,5,4] chunks [1,2,4]; value = layer*100 + y*10 + x."""
    import json

    objs = {
        f"{BASE}/field3d/.zarray": _zarray((2, 5, 4), (1, 2, 4), "<f4"),
        f"{BASE}/field3d/.zattrs": json.dumps({"_ARRAY_DIMENSIONS": ["layer", "y", "x"]}).encode(),
    }
    for c0 in range(2):
        for c1 in range(3):
            chunk = np.zeros((1, 2, 4), dtype="<f4")
            for b in range(2):
                for c in range(4):
                    y, x = c1 * 2 + b, c
                    if y < 5:
                        chunk[0, b, c] = c0 * 100 + y * 10 + x
            objs[f"{BASE}/field3d/{c0}.{c1}.0"] = _encode_chunk(chunk)
    _populate(root, objs)


def test_read_store_orthogonal_selection(tmp_path):
    _mini_store(tmp_path)
    cache = Cache(root=tmp_path, offline=True, verify=True)
    nds = ZarrReader().read_store(
        cache, BASE, ["field3d"],
        select={"axes": [{"indices": [1]}, {"indices": [1, 4]}, "all"]},
    )
    f = nds.variables["field3d"]
    assert f.dims == ("layer", "y", "x")
    assert f.shape == (1, 2, 4)
    assert f.data.dtype == np.float64
    np.testing.assert_array_equal(
        f.data, np.array([[[110, 111, 112, 113], [140, 141, 142, 143]]], dtype="f8")
    )


def test_read_store_all_and_partial_edge_chunk(tmp_path):
    _mini_store(tmp_path)
    cache = Cache(root=tmp_path, offline=True, verify=True)
    nds = ZarrReader().read_store(cache, BASE, ["field3d"], select=None)
    f = nds.variables["field3d"]
    assert f.shape == (2, 5, 4)  # full array; edge chunk row 4 present, pad row dropped
    # row 4 (the partial edge chunk) decodes to real values, not pad
    np.testing.assert_array_equal(f.data[1, 4], np.array([140, 141, 142, 143], dtype="f8"))


def test_fill_value_zero_is_not_nan(tmp_path):
    """A stored 0.0 is real data; the reader must NOT map fill_value to NaN."""
    import json

    objs = {
        f"{BASE}/z/.zarray": _zarray((4,), (4,), "<f8"),
        f"{BASE}/z/.zattrs": json.dumps({"_ARRAY_DIMENSIONS": ["c"]}).encode(),
        f"{BASE}/z/0": _encode_chunk(np.array([0.0, 1.0, 0.0, 2.0], dtype="<f8")),
    }
    _populate(tmp_path, objs)
    cache = Cache(root=tmp_path, offline=True, verify=True)
    f = ZarrReader().read_store(cache, BASE, ["z"]).variables["z"]
    assert not np.isnan(f.data).any()
    np.testing.assert_array_equal(f.data, [0.0, 1.0, 0.0, 2.0])


def test_absent_chunk_object_fills_with_fill_value(tmp_path):
    """A missing chunk object fills its region with fill_value (0.0 here)."""
    import json

    objs = {
        f"{BASE}/g/.zarray": _zarray((4,), (2,), "<f8"),  # 2 chunks
        f"{BASE}/g/.zattrs": json.dumps({"_ARRAY_DIMENSIONS": ["c"]}).encode(),
        f"{BASE}/g/0": _encode_chunk(np.array([5.0, 6.0], dtype="<f8")),
        # chunk "1" (cells 2,3) intentionally omitted → filled with 0.0
    }
    _populate(tmp_path, objs)
    cache = Cache(root=tmp_path, offline=True, verify=True)
    f = ZarrReader().read_store(cache, BASE, ["g"]).variables["g"]
    np.testing.assert_array_equal(f.data, [5.0, 6.0, 0.0, 0.0])


def test_synthesized_dims_without_zattrs(tmp_path):
    objs = {f"{BASE}/n/.zarray": _zarray((3,), (3,), "<f8"),
            f"{BASE}/n/0": _encode_chunk(np.array([1.0, 2.0, 3.0], dtype="<f8"))}
    _populate(tmp_path, objs)
    cache = Cache(root=tmp_path, offline=True, verify=True)
    f = ZarrReader().read_store(cache, BASE, ["n"]).variables["n"]
    assert f.dims == ("dim_0",)


# --------------------------------------------------------------------------- #
# Laziness — the load-bearing capability, proven with a poison store.
# --------------------------------------------------------------------------- #


def test_laziness_never_touches_unselected_chunks(tmp_path):
    """Non-selected chunks hold undecodable garbage; a lazy read never touches
    them, so the selection decodes cleanly. An over-fetch would blosc-error."""
    import json

    objs = {
        f"{BASE}/sr/.zarray": _zarray((3, 500, 1), (1, 100, 1), "<f4"),
        f"{BASE}/sr/.zattrs": json.dumps(
            {"_ARRAY_DIMENSIONS": ["layer", "source", "receptor"]}).encode(),
    }
    # 3 layers x 5 source-chunks x 1 = 15 chunks. Only layer 0, source-chunks {0,3}
    # are valid; every other chunk is poison (garbage that fails blosc decode).
    want_layers, want_source_chunks = {0}, {0, 3}
    for c0 in range(3):
        for c1 in range(5):
            key = f"{BASE}/sr/{c0}.{c1}.0"
            if c0 in want_layers and c1 in want_source_chunks:
                chunk = np.full((1, 100, 1), float(c0 * 1000 + c1), dtype="<f4")
                objs[key] = _encode_chunk(chunk)
            else:
                objs[key] = b"\x00POISON-not-a-blosc-container\xff"
    _populate(tmp_path, objs)
    cache = Cache(root=tmp_path, offline=True, verify=True)

    # select layer=[0], source=[5, 12, 305, 340] (chunks {0, 3}), receptor=all.
    nds = ZarrReader().read_store(
        cache, BASE, ["sr"],
        select={"axes": [{"indices": [0]},
                         {"indices": [5, 12, 305, 340]},
                         "all"]},
    )
    f = nds.variables["sr"]
    assert f.shape == (1, 4, 1)
    # sources 5,12 -> chunk 0 (value 0); sources 305,340 -> chunk 3 (value 3)
    np.testing.assert_array_equal(f.data.ravel(), [0.0, 0.0, 3.0, 3.0])


def test_over_selection_would_hit_poison(tmp_path):
    """Control: selecting a source in a POISON chunk DOES decode-error — proving
    the poison is genuinely undecodable, so the lazy test above is meaningful."""
    import json

    objs = {
        f"{BASE}/sr/.zarray": _zarray((1, 500, 1), (1, 100, 1), "<f4"),
        f"{BASE}/sr/.zattrs": json.dumps(
            {"_ARRAY_DIMENSIONS": ["layer", "source", "receptor"]}).encode(),
        f"{BASE}/sr/0.0.0": _encode_chunk(np.zeros((1, 100, 1), dtype="<f4")),
    }
    for c1 in range(1, 5):
        objs[f"{BASE}/sr/0.{c1}.0"] = b"\x00POISON\xff"
    _populate(tmp_path, objs)
    cache = Cache(root=tmp_path, offline=True, verify=True)
    with pytest.raises(Exception):
        ZarrReader().read_store(
            cache, BASE, ["sr"],
            select={"axes": [{"indices": [0]}, {"indices": [150]}, "all"]},  # chunk 1 = poison
        )


# --------------------------------------------------------------------------- #
# Registry dispatch + Provider store-backed seam.
# --------------------------------------------------------------------------- #


def test_zarr_registered_active_and_store_backed():
    assert format_registry.status("zarr") == "active"
    reader = format_registry.create("zarr")
    assert getattr(reader, "store_backed", False) is True


def test_provider_routes_store_backed(tmp_path):
    _mini_store(tmp_path)
    cache = Cache(root=tmp_path, offline=True, verify=True)
    loader = DataSource(
        name="isrm", format="zarr", url=BASE, variables=["field3d"],
        reader_kwargs={"select": {"axes": [{"indices": [1]}, {"indices": [1, 4]}, "all"]}},
    )
    nds = Provider(loader, cache).materialize()
    assert nds.variables["field3d"].shape == (1, 2, 4)


def test_read_native_is_store_backed_error():
    reader = ZarrReader()
    from earthsciio.errors import Unsupported

    with pytest.raises(Unsupported):
        reader.open("/tmp/x")
    with pytest.raises(Unsupported):
        reader.read_native(object(), ["field3d"])


def test_variables_required():
    reader = ZarrReader()
    with pytest.raises(ValueError):
        reader.read_store(object(), BASE, None)


# --------------------------------------------------------------------------- #
# Phase 1: per-call `select` pushdown, supports_selection, array_shape.
# (mirrors julia/test/test_zarr.jl's Phase-1a tests.)
# --------------------------------------------------------------------------- #


class _CountingStore:
    """Wraps a :class:`LocalStore` and records every ``get_blob`` KEY, so a test can
    prove the reader fetched ONLY the objects it needed (each on-demand object fetch
    is exactly one ``get_blob`` on the offline path). Mirrors the Julia
    ``CountingStore``; everything else forwards to the wrapped store."""

    def __init__(self, inner: LocalStore) -> None:
        self.inner = inner
        self.gets: list = []

    def name(self):  # pragma: no cover - trivial delegation
        return self.inner.name()

    def get_blob(self, key):
        self.gets.append(key)
        return self.inner.get_blob(key)

    def exists(self, key):  # pragma: no cover - trivial delegation
        return self.inner.exists(key)

    def get_meta(self, key):
        return self.inner.get_meta(key)

    def put_blob(self, key, staged, ext=""):  # pragma: no cover - not used offline
        return self.inner.put_blob(key, staged, ext)

    def put_meta(self, key, manifest):  # pragma: no cover - not used offline
        return self.inner.put_meta(key, manifest)

    def staging_path(self, ext="part"):  # pragma: no cover - not used offline
        return self.inner.staging_path(ext)

    def lock(self, key):  # pragma: no cover - not used offline
        return self.inner.lock(key)


ZSR = "s3://earthsci-fixtures/sr-mini.zarr"


def _sr_store(root):
    """A VALID `sr` store: shape (3,500,1), chunks (1,100,1). Element at global
    (layer, source, 0) encodes its indices: value = layer*1_000_000 + source (exact
    in float32 for these ranges), so a selection's values are self-checking."""
    import json

    objs = {
        f"{ZSR}/sr/.zarray": _zarray((3, 500, 1), (1, 100, 1), "<f4"),
        f"{ZSR}/sr/.zattrs": json.dumps(
            {"_ARRAY_DIMENSIONS": ["layer", "source", "receptor"]}
        ).encode(),
    }
    for c0 in range(3):
        for c1 in range(5):
            chunk = np.zeros((1, 100, 1), dtype="<f4")
            for j in range(100):
                chunk[0, j, 0] = float(c0 * 1_000_000 + (c1 * 100 + j))
            objs[f"{ZSR}/sr/{c0}.{c1}.0"] = _encode_chunk(chunk)
    _populate(root, objs)


def test_per_call_select_pushes_down_and_fetches_only_needed_chunks(tmp_path):
    _sr_store(tmp_path)
    store = _CountingStore(LocalStore(tmp_path))
    cache = Cache(store, offline=True, verify=False)
    p = Provider(DataSource(name="isrm", format="zarr", url=ZSR, variables=["sr"]), cache)

    # layer 1, sources {5,12}∈chunk0 and {305,340}∈chunk3, all receptors.
    sel = {"axes": [{"indices": [1]}, {"indices": [5, 12, 305, 340]}, "all"]}
    nds = p.materialize(select=sel)
    f = nds.variables["sr"]
    assert f.dims == ("layer", "source", "receptor")
    assert f.shape == (1, 4, 1)
    np.testing.assert_array_equal(
        f.data.ravel(), [1_000_005, 1_000_012, 1_000_305, 1_000_340]
    )

    # Laziness: the two needed chunk objects (1,0,0) and (1,3,0) WERE fetched, and
    # NONE of the 13 other chunk objects was (layers 0/2, source-chunks 1/2/4).
    # (zarr-python may also probe metadata objects — zarr.json/.zarray/.zattrs —
    # which is irrelevant to chunk-level laziness, so we assert on chunk keys only.)
    got = set(store.gets)
    for k in ("1.0.0", "1.3.0"):
        assert cache_key(f"{ZSR}/sr/{k}") in got, f"needed chunk {k} not fetched"
    for c0 in range(3):
        for c1 in range(5):
            if (c0, c1) in ((1, 0), (1, 3)):
                continue
            assert cache_key(f"{ZSR}/sr/{c0}.{c1}.0") not in got, (
                f"over-fetched chunk {c0}.{c1}.0"
            )


def test_per_call_select_preserves_permuted_order(tmp_path):
    """A NON-CONTIGUOUS PERMUTED index list returns rows in the GIVEN order, not
    sorted — the load-bearing ordering contract for the 3-way conformance case."""
    _sr_store(tmp_path)
    cache = Cache(LocalStore(tmp_path), offline=True, verify=False)
    p = Provider(DataSource(name="isrm", format="zarr", url=ZSR, variables=["sr"]), cache)
    sel = {"axes": [{"indices": [0]}, {"indices": [340, 5, 305, 12]}, "all"]}
    f = p.materialize(select=sel).variables["sr"]
    # Order preserved exactly (a reader that sorted would give 5,12,305,340).
    np.testing.assert_array_equal(f.data.ravel(), [340, 5, 305, 12])


def test_per_call_select_overrides_baked(tmp_path):
    _sr_store(tmp_path)
    cache = Cache(LocalStore(tmp_path), offline=True, verify=False)
    baked = {"axes": [{"indices": [0]}, {"indices": [7]}, "all"]}
    p = Provider(
        DataSource(name="isrm", format="zarr", url=ZSR, variables=["sr"],
                   reader_kwargs={"select": baked}),
        cache,
    )
    # No per-call select ⇒ the baked select still applies (regression).
    np.testing.assert_array_equal(p.materialize().variables["sr"].data.ravel(), [7])
    # A per-call select OVERRIDES the baked one for this call only.
    over = {"axes": [{"indices": [2]}, {"indices": [7]}, "all"]}
    np.testing.assert_array_equal(
        p.materialize(select=over).variables["sr"].data.ravel(), [2_000_007]
    )
    # ... and the baked default is untouched afterwards.
    np.testing.assert_array_equal(p.materialize().variables["sr"].data.ravel(), [7])


def test_array_shape_reads_only_zarray(tmp_path):
    _sr_store(tmp_path)
    store = _CountingStore(LocalStore(tmp_path))
    cache = Cache(store, offline=True, verify=False)
    p = Provider(DataSource(name="isrm", format="zarr", url=ZSR, variables=["sr"]), cache)

    assert p.array_shape("sr") == (3, 500, 1)
    # array_shape reads ONLY metadata — never a chunk object. (zarr-python may
    # probe both zarr.json and .zarray; the invariant is that no chunk was read.)
    for c0 in range(3):
        for c1 in range(5):
            assert cache_key(f"{ZSR}/sr/{c0}.{c1}.0") not in set(store.gets)


def test_supports_selection_and_array_shape_capability_surface(tmp_path):
    _sr_store(tmp_path)
    cache = Cache(LocalStore(tmp_path), offline=True, verify=False)

    # store-backed zarr provider CAN push down
    pz = Provider(DataSource(name="isrm", format="zarr", url=ZSR, variables=["sr"]), cache)
    assert supports_selection(ZarrReader()) is True
    assert pz.supports_selection is True

    assert pz.store_backed is True

    # a whole-file reader that honours `select` at DECODE time: it answers
    # supports_selection but NOT store_backed, which is how a caller tells "the
    # download shrank" from "only the decode did". array_shape is still None (a
    # blob's shape is not knowable without reading it).
    pn = Provider(DataSource(name="era5", format="netcdf", url="file:///dev/null"), cache)
    assert pn.supports_selection is True
    assert pn.store_backed is False
    assert pn.array_shape("anything") is None

    # readers that cannot honour a select at all
    for fmt in ("csv", "ff10"):
        pw = Provider(DataSource(name="x", format=fmt, url="file:///dev/null"), cache)
        assert pw.supports_selection is False
        assert pw.store_backed is False
        assert pw.array_shape("anything") is None
    assert supports_selection(CSVReader()) is False
    assert supports_selection(FF10Reader()) is False


def test_per_call_select_on_non_store_reader_raises(tmp_path):
    cache = Cache(LocalStore(tmp_path), offline=True, verify=False)
    pw = Provider(DataSource(name="x", format="csv", url="file:///dev/null"), cache)
    # raised before any fetch — the reader can't honour a projection pushdown
    with pytest.raises(ValueError):
        pw.materialize(select={"axes": ["all"]})
    with pytest.raises(ValueError):
        pw.refresh(_dt.datetime(2020, 1, 1), select={"axes": ["all"]})


# --------------------------------------------------------------------------- #
# Peak-memory regression: the reader must STREAM chunks into the output, never
# hold every intersected chunk at once.
#
# The Julia and Rust zarr readers decoded *every* chunk a selection intersects
# into a map and only then assembled the result, so their peak memory scaled
# with the total decompressed chunk volume instead of with the answer: on the
# real ISRM source-receptor array that is 416 chunks x ~21 MB = ~8.7 GB held
# simultaneously to produce a 0.59 GB result (~15x), which OOM-killed a
# production run.
#
# The Python reader does NOT have that shape: it delegates chunk iteration to
# zarr-python's ``oindex``, which scatters each decoded chunk straight into a
# preallocated output and drops it, keeping at most ``async.concurrency`` chunks
# live. This test PINS that property so a future refactor (e.g. hand-rolling the
# chunk loop again, or collecting chunks into a dict "for clarity") cannot
# silently reintroduce the amplification.
#
# Measured with ``tracemalloc``, which counts peak SIMULTANEOUS live bytes and
# includes numpy's data allocations. Deliberately NOT measured with RSS: freed
# chunk buffers are retained in the glibc arena (reusable, not live), so RSS
# over-reports by 5-10x here and is not a liveness signal.
#
# See ``bench/zarr_peak_memory.py`` for the full sweep this test distils.
# --------------------------------------------------------------------------- #

_STREAM_CHUNK_ROWS = 256
_STREAM_NCOLS = 1024
_STREAM_CHUNK_BYTES = _STREAM_CHUNK_ROWS * _STREAM_NCOLS * 4  # 1 MiB decompressed
_STREAM_BASE = "s3://earthsci-fixtures/stream.zarr"


def _stream_store(root, n_chunks):
    """A 2-D ``(n_chunks*256, 1024)`` float32 v2 array chunked ``(256, 1024)``.

    Payloads are low-entropy (so the on-disk fixture stays tiny) but every chunk
    still decompresses to a full 1 MiB — the ISRM situation of a compressed
    object store with fat decompressed chunks.
    """
    import json

    nrows = n_chunks * _STREAM_CHUNK_ROWS
    objs = {
        f"{_STREAM_BASE}/field/.zarray": _zarray(
            (nrows, _STREAM_NCOLS), (_STREAM_CHUNK_ROWS, _STREAM_NCOLS), "<f4"
        ),
        f"{_STREAM_BASE}/field/.zattrs": json.dumps(
            {"_ARRAY_DIMENSIONS": ["source", "receptor"]}
        ).encode(),
    }
    col = (np.arange(_STREAM_NCOLS, dtype=np.float32) % 97.0) * 0.25
    for c in range(n_chunks):
        chunk = np.empty((_STREAM_CHUNK_ROWS, _STREAM_NCOLS), dtype="<f4")
        for r in range(_STREAM_CHUNK_ROWS):
            chunk[r] = col + np.float32(c * _STREAM_CHUNK_ROWS + r)
        objs[f"{_STREAM_BASE}/field/{c}.0"] = _encode_chunk(chunk)
        del chunk
    _populate(root, objs)
    return nrows


def _stream_expected_row(global_row):
    col = (np.arange(_STREAM_NCOLS, dtype=np.float32) % 97.0) * 0.25
    return (col + np.float32(global_row)).astype(np.float64)


def _peak_live_bytes(fn):
    """Peak SIMULTANEOUS traced bytes allocated by ``fn`` (numpy data included)."""
    import tracemalloc

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


def _numpy_allocations_are_traced():
    """tracemalloc only measures liveness here if numpy routes its data
    allocations through it (numpy >= 1.17 does). Verify rather than assume —
    otherwise the assertions below would pass vacuously."""
    _, peak = _peak_live_bytes(lambda: np.ones(4 * 1024 * 1024, dtype=np.float64))
    return peak > 24 * 1024 * 1024


def _read_one_row_per_chunk(tmp_path, n_chunks):
    """Read exactly one row out of every chunk: a tiny answer whose selection
    nevertheless touches the whole array."""
    root = tmp_path / f"store{n_chunks}"
    root.mkdir()
    _stream_store(root, n_chunks)
    cache = Cache(LocalStore(root), offline=True, verify=False)
    idx = [c * _STREAM_CHUNK_ROWS + 7 for c in range(n_chunks)]
    select = {"axes": [{"indices": idx}, "all"]}

    def _go():
        nds = ZarrReader().read_store(cache, _STREAM_BASE, ["field"], select=select)
        return nds.variables["field"].data

    data, peak = _peak_live_bytes(_go)
    assert data.shape == (n_chunks, _STREAM_NCOLS)
    assert data.dtype == np.float64
    # The streaming path must still be exactly correct: every output row is
    # written by exactly one chunk, so spot-check the ends and the middle.
    for probe in (0, n_chunks // 2, n_chunks - 1):
        np.testing.assert_array_equal(data[probe], _stream_expected_row(idx[probe]))
    return peak


def test_selection_streams_chunks_and_does_not_buffer_them_all(tmp_path):
    """Peak LIVE memory must be bounded by (output + a few chunks), NOT by the
    total decompressed volume of the chunks the selection intersects."""
    zarr = pytest.importorskip("zarr")
    if not _numpy_allocations_are_traced():  # pragma: no cover - platform guard
        pytest.skip("tracemalloc does not see numpy data allocations here")

    n_chunks = 128
    volume = n_chunks * _STREAM_CHUNK_BYTES  # 128 MiB if every chunk were held

    # Pin zarr's in-flight chunk budget so the bound under test is a property of
    # the reader, not of whatever default the installed zarr happens to ship.
    concurrency = 8
    with zarr.config.set({"async.concurrency": concurrency}):
        peak = _read_one_row_per_chunk(tmp_path, n_chunks)

    # A buffer-everything reader peaks at >= `volume`; a streaming one peaks at
    # roughly `concurrency` chunks plus the (here negligible) output.
    budget = (concurrency + 8) * _STREAM_CHUNK_BYTES
    assert peak < budget, (
        f"peak live memory {peak/1e6:.1f} MB exceeds the streaming budget "
        f"{budget/1e6:.1f} MB while the intersected chunk volume is "
        f"{volume/1e6:.1f} MB — the reader looks like it is buffering chunks "
        f"instead of scattering each one into the output"
    )
    assert peak < volume / 4


def test_peak_memory_is_flat_in_the_number_of_chunks(tmp_path):
    """The load-bearing invariant: growing the intersected chunk volume 8x at a
    near-constant answer size must NOT grow peak memory 8x."""
    zarr = pytest.importorskip("zarr")
    if not _numpy_allocations_are_traced():  # pragma: no cover - platform guard
        pytest.skip("tracemalloc does not see numpy data allocations here")

    with zarr.config.set({"async.concurrency": 8}):
        small = _read_one_row_per_chunk(tmp_path, 16)   # 16 MiB chunk volume
        large = _read_one_row_per_chunk(tmp_path, 128)  # 128 MiB chunk volume

    # Buffering would make this ratio ~8; streaming keeps it near 1 (the only
    # growth is the output itself, 128 KiB -> 1 MiB).
    assert large < 2.0 * small, (
        f"peak live memory grew from {small/1e6:.1f} MB to {large/1e6:.1f} MB when the "
        f"intersected chunk volume grew 8x at a nearly fixed answer size — peak is "
        f"tracking chunk volume, not output"
    )

"""Active format readers decode the corpus to the oracle's native arrays.

The reader is the cross-language parity surface (``spec/conformance.md`` §3): the
``netcdf``/``csv`` readers must decode the committed corpus blobs to arrays
equal to ``expected`` in each ``conformance/corpus/cases/*.json`` — the same
verdict the Python oracle (``conformance/verify.py``) and the Julia/Rust readers
reach (conformance checks 3–4, run OFFLINE). These tests exercise that decode
through the *active registry readers* this bead adds.
"""

from __future__ import annotations

import csv as _csv
import io
import json
import math
import pathlib
import zipfile

import numpy as np
import pytest

from earthsciio import Cache, CSVReader, FF10Reader, NetCDFReader
from earthsciio.native import NativeDataset
from earthsciio.readers import FF10_POINT_COLUMNS, FF10_POINT_NUMERIC
from earthsciio.provider import reader_option_names
from earthsciio.registry import dim_length as registry_dim_length
from earthsciio.registry import format_registry

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / "conformance" / "corpus"

ERA5_URL = "https://data.earthsci.dev/era5/2018/11/20181108.nc"
OPENAQ_URL = (
    "https://openaq-data-archive.s3.amazonaws.com/records/openaq/"
    "locationid=1/2018-11-08.csv"
)
FF10_URL = "https://gaftp.epa.gov/air/emismod/2016/v1/2016fd/point/ff10_point.csv"

ATOL, RTOL = 1e-6, 1e-9


def _case(case_id: str) -> dict:
    return json.loads((CORPUS / "cases" / f"{case_id}.json").read_text())


def _flat(x):
    if isinstance(x, list):
        for e in x:
            yield from _flat(e)
    else:
        yield x


def _assert_numeric(field, expected_nested, label: str) -> None:
    exp = np.array(
        [math.nan if v is None else float(v) for v in _flat(expected_nested)],
        dtype="f8",
    )
    got = np.asarray(field.data, dtype="f8").reshape(-1)
    assert got.shape == exp.shape, f"{label}: shape {got.shape} != {exp.shape}"
    gn, en = np.isnan(got), np.isnan(exp)
    assert np.array_equal(gn, en), f"{label}: NaN/fill mask mismatch"
    assert np.allclose(got[~gn], exp[~en], atol=ATOL, rtol=RTOL), f"{label}: value mismatch"


def _assert_string(field, expected_nested, label: str) -> None:
    got = [str(v) for v in _flat(field.data)]
    exp = [None if v is None else str(v) for v in _flat(expected_nested)]
    assert got == [e for e in exp], f"{label}: string mismatch {got} != {exp}"


@pytest.fixture
def offline_cache():
    """A read-only offline cache rooted at the conformance corpus (no network)."""
    return Cache(root=CORPUS / "cache", offline=True, verify=True)


# --------------------------------------------------------------------------- #
# Registration: the active readers are wired into the format registry.
# --------------------------------------------------------------------------- #


def test_active_readers_registered():
    assert "netcdf" in format_registry and "csv" in format_registry
    assert format_registry.status("netcdf") == "active"
    assert format_registry.status("csv") == "active"
    # constructed by name through the seam the Provider uses
    assert isinstance(format_registry.create("netcdf"), NetCDFReader)
    assert isinstance(format_registry.create("csv"), CSVReader)
    assert NetCDFReader().formats() == ["netcdf"]
    assert "nc" in NetCDFReader().extensions()
    assert CSVReader().formats() == ["csv"]


# --------------------------------------------------------------------------- #
# NetCDF decode (CF scale/offset + fill->NaN + raw time), vs the corpus oracle.
# --------------------------------------------------------------------------- #


@pytest.mark.needs_format("netcdf")
def test_netcdf_decodes_to_oracle_arrays(offline_cache):
    case = _case("era5-grid-sub-tile")
    blob = offline_cache.fetch(ERA5_URL)
    reader = NetCDFReader()
    nds = reader.read_native(reader.open(blob.path))

    assert isinstance(nds, NativeDataset)
    assert nds.variable_names() == ["sp", "t2m"]
    assert nds.coord_names() == ["latitude", "longitude", "time"]
    for name, spec in case["expected"]["variables"].items():
        _assert_numeric(nds[name], spec["data"], name)
        assert list(nds[name].dims) == spec["dims"]
        assert list(nds[name].shape) == spec["shape"]
        assert nds[name].data.dtype == np.float64  # packed + plain both -> float64
    for name, spec in case["expected"]["coords"].items():
        _assert_numeric(nds[name], spec["data"], f"coord {name}")


@pytest.mark.needs_format("netcdf")
def test_netcdf_time_axis_is_raw_with_calendar(offline_cache):
    blob = offline_cache.fetch(ERA5_URL)
    reader = NetCDFReader()
    nds = reader.read_native(reader.open(blob.path))
    time = nds["time"]
    # raw integer time (decode_times=False) — int kept, NOT upcast to float
    assert np.issubdtype(time.data.dtype, np.integer)
    assert time.data.dtype == np.int32
    assert list(time.data) == [0, 1]
    # units + calendar carried for ESS; calendar decoding is NOT the reader's job
    assert time.attrs["units"] == "hours since 2018-11-08 00:00:00"
    assert time.attrs["calendar"] == "gregorian"


@pytest.mark.needs_format("netcdf")
def test_netcdf_variable_selection_keeps_coords(offline_cache):
    blob = offline_cache.fetch(ERA5_URL)
    reader = NetCDFReader()
    nds = reader.read_native(reader.open(blob.path), ["t2m"])
    assert nds.variable_names() == ["t2m"]  # sp dropped
    assert nds.coord_names() == ["latitude", "longitude", "time"]  # coords always kept


@pytest.mark.needs_format("netcdf")
def test_netcdf_absent_variable_raises(offline_cache):
    blob = offline_cache.fetch(ERA5_URL)
    reader = NetCDFReader()
    with pytest.raises(KeyError):
        reader.read_native(reader.open(blob.path), ["nope"])


# --------------------------------------------------------------------------- #
# NetCDF decode-time `select`: one blob, one cache key, only the requested
# hyperslab materialised (spec/conformance.md "NetCDF decode notes").
# --------------------------------------------------------------------------- #


@pytest.mark.needs_format("netcdf")
def test_netcdf_select_windows_the_decode_and_slices_coords(offline_cache):
    blob = offline_cache.fetch(ERA5_URL)
    reader = NetCDFReader()
    full = reader.read_native(reader.open(blob.path))

    sel = {"axes": ["all", {"slice": [1, 3]}, {"indices": [0, 2]}]}
    w = reader.read_native(reader.open(blob.path), None, sel)

    # The gate: a windowed read equals the FULL read sliced afterwards.
    for name in ("t2m", "sp"):
        expected = full[name].data[:, 1:3][:, :, [0, 2]]
        assert np.array_equal(w[name].data, expected, equal_nan=True)
        assert list(w[name].dims) == list(full[name].dims)  # dims are NAMES
        assert w[name].attrs == full[name].attrs

    # Coordinates are sliced WITH the data — a windowed variable beside a
    # full-length lon/lat would be a silent trap.
    assert np.array_equal(w["latitude"].data, full["latitude"].data[1:3])
    assert np.array_equal(w["longitude"].data, full["longitude"].data[[0, 2]])
    # the time axis is untouched, and still raw with its units/calendar
    assert np.array_equal(w["time"].data, full["time"].data)
    assert w["time"].attrs == full["time"].attrs


@pytest.mark.needs_format("netcdf")
def test_netcdf_select_preserves_the_requested_index_order(offline_cache):
    blob = offline_cache.fetch(ERA5_URL)
    reader = NetCDFReader()
    full = reader.read_native(reader.open(blob.path))
    p = reader.read_native(reader.open(blob.path), None,
                           {"axes": ["all", "all", {"indices": [2, 0]}]})
    assert np.array_equal(p["t2m"].data, full["t2m"].data[:, :, [2, 0]], equal_nan=True)
    assert np.array_equal(p["longitude"].data, full["longitude"].data[[2, 0]])


@pytest.mark.needs_format("netcdf")
def test_netcdf_select_composes_with_variables(offline_cache):
    blob = offline_cache.fetch(ERA5_URL)
    reader = NetCDFReader()
    full = reader.read_native(reader.open(blob.path))
    both = reader.read_native(reader.open(blob.path), ["t2m"],
                              {"axes": ["all", {"slice": [1, 3]}, "all"]})
    assert both.variable_names() == ["t2m"]
    assert np.array_equal(both["t2m"].data, full["t2m"].data[:, 1:3], equal_nan=True)


@pytest.mark.needs_format("netcdf")
def test_netcdf_select_refuses_a_time_subset(offline_cache):
    """Record selection is the Provider's — it owns the cadence, not the reader."""
    blob = offline_cache.fetch(ERA5_URL)
    reader = NetCDFReader()
    with pytest.raises(ValueError, match="time"):
        reader.read_native(reader.open(blob.path), None,
                           {"axes": [{"indices": [0]}, "all", "all"]})


@pytest.mark.needs_format("netcdf")
def test_netcdf_select_refuses_an_axis_count_matching_nothing(offline_cache):
    blob = offline_cache.fetch(ERA5_URL)
    reader = NetCDFReader()
    with pytest.raises(ValueError, match="rank"):
        reader.read_native(reader.open(blob.path), None, {"axes": ["all", "all"]})


def test_cf_time_units_detection_is_case_insensitive_and_token_wise():
    """"hours SINCE 1900-01-01" must not be a time axis in one track and a
    selectable spatial axis in another: the rule is a whitespace-separated
    ``since`` token, case-insensitive, matching the Rust track exactly."""
    from earthsciio.readers import _is_cf_time

    for units in ("hours since 1900-01-01", "hours SINCE 1900-01-01",
                  "days since 2018-11-08 00:00:00", "since"):
        assert _is_cf_time({"units": units}) is True, units
    for units in ("degrees_north", "Pa", "kg m-2 s-1", "sincerity", ""):
        assert _is_cf_time({"units": units}) is False, units
    assert _is_cf_time({}) is False


@pytest.mark.needs_format("netcdf")
def test_netcdf_full_extent_select_is_identical_to_no_select(offline_cache):
    """A selection covering every index must be the no-selection read, exactly.

    Both spellings of "everything" — an explicit half-open slice and an explicit
    index list — go down the windowed code path, so this is the read that catches
    an off-by-one in the slab planner on the happy axis.
    """
    blob = offline_cache.fetch(ERA5_URL)
    reader = NetCDFReader()
    full = reader.read_native(reader.open(blob.path))
    for axes in (["all", {"slice": [0, 3]}, {"slice": [0, 3]}],
                 ["all", {"indices": [0, 1, 2]}, {"indices": [0, 1, 2]}]):
        got = reader.read_native(reader.open(blob.path), None, {"axes": axes})
        for name in ("t2m", "sp"):
            assert got[name].shape == full[name].shape
            assert np.array_equal(got[name].data, full[name].data, equal_nan=True)
        for name in ("latitude", "longitude", "time"):
            assert np.array_equal(got[name].data, full[name].data)


@pytest.mark.needs_format("netcdf")
def test_netcdf_select_keeps_a_single_element_dimension(offline_cache):
    """This vocabulary NEVER drops a dimension: a one-index axis comes back with
    length 1, not squeezed away. A track that squeezed would diverge in rank."""
    blob = offline_cache.fetch(ERA5_URL)
    reader = NetCDFReader()
    full = reader.read_native(reader.open(blob.path))
    for axis in ({"indices": [1]}, {"slice": [1, 2]}):
        got = reader.read_native(reader.open(blob.path), None,
                                 {"axes": ["all", axis, "all"]})
        assert got["t2m"].shape == (2, 1, 3)
        assert list(got["t2m"].dims) == ["time", "latitude", "longitude"]
        assert got["latitude"].shape == (1,)
        assert np.array_equal(got["t2m"].data, full["t2m"].data[:, 1:2], equal_nan=True)


@pytest.mark.needs_format("netcdf")
def test_netcdf_select_empty_axis_is_a_zero_length_window(offline_cache):
    """An axis may legally resolve to NOTHING — a length-0 axis, KEPT in dims.

    The trap is a field whose ``shape`` contradicts its own ``dims`` and the
    coordinates beside it: xarray's lazy indexer materialises an empty selection
    to (1, 0, 1) rather than (2, 0, 3), and shipping that is silent corruption.
    """
    blob = offline_cache.fetch(ERA5_URL)
    reader = NetCDFReader()
    for axis in ({"indices": []}, {"slice": [1, 1]}, {"slice": [2, 0]}):
        got = reader.read_native(reader.open(blob.path), None,
                                 {"axes": ["all", axis, "all"]})
        for name in ("t2m", "sp"):
            assert got[name].shape == (2, 0, 3), axis
            assert list(got[name].dims) == ["time", "latitude", "longitude"]
            assert np.asarray(got[name].data).size == 0
        assert got["latitude"].shape == (0,)
        # ...and the axes NOT selected keep their full length.
        assert got["longitude"].shape == (3,)
        assert got["time"].shape == (2,)


@pytest.mark.needs_format("netcdf")
def test_netcdf_select_refuses_an_out_of_range_slice(offline_cache):
    """A slice is bounds-checked like an `indices` list: an over-long window is an
    error, never a silent clamp, and a negative bound never wraps around."""
    blob = offline_cache.fetch(ERA5_URL)
    reader = NetCDFReader()
    with pytest.raises(IndexError, match="out of range"):
        reader.read_native(reader.open(blob.path), None,
                           {"axes": ["all", {"slice": [1, 99]}, "all"]})
    with pytest.raises(IndexError, match="out of range"):
        reader.read_native(reader.open(blob.path), None,
                           {"axes": ["all", {"slice": [-2, 3]}, "all"]})
    with pytest.raises(IndexError, match="out of range"):
        reader.read_native(reader.open(blob.path), None,
                           {"axes": ["all", {"indices": [5]}, "all"]})


# --------------------------------------------------------------------------- #
# NetCDF `records` pushdown — the cadence owner's record axis.
#
# The other half of the refused-time-axis rule above: a `select` may not CHOOSE
# records, but the owner of the cadence may say which ones it wants. The corpus
# blob holds two records, where a two-record bracket IS the whole axis and a
# narrowed decode cannot be told from a full one, so the equivalence sweep below
# runs on a synthetic FOUR-record file where every selection is a real subset.
# Mirrors `julia/test/test_readers.jl` and the Rust `format::netcdf::tests`.
# --------------------------------------------------------------------------- #


def _write_cadence_nc(path, nrec: int) -> pathlib.Path:
    """A 4-record cadence file: `t2m(time, lat, lon)` whose every cell is
    distinct in the record AND in both spatial axes (so a record read out of
    order, de-duplicated or off by one is a wrong NUMBER, not a wrong shape),
    beside an `orog(lat, lon)` that carries no record axis at all."""
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(str(path), "w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("time", nrec)
        ds.createDimension("lat", 2)
        ds.createDimension("lon", 3)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = "hours since 2018-11-09 00:00:00"
        t.calendar = "standard"
        t[:] = np.arange(nrec, dtype="f8")
        la = ds.createVariable("lat", "f8", ("lat",))
        la[:] = [40.0, 41.0]
        lo = ds.createVariable("lon", "f8", ("lon",))
        lo[:] = [-90.0, -89.0, -88.0]
        v = ds.createVariable("t2m", "f8", ("time", "lat", "lon"))
        r, i, j = np.meshgrid(np.arange(nrec), np.arange(2), np.arange(3), indexing="ij")
        v[:] = 100 * (r + 1) + 10 * i + j
        o = ds.createVariable("orog", "f8", ("lat", "lon"))
        o[:] = [[7.0, 8.0, 9.0], [8.0, 9.0, 10.0]]
    return path


@pytest.fixture
def cadence_blob(tmp_path):
    return _write_cadence_nc(tmp_path / "cadence.nc", 4)


@pytest.mark.needs_format("netcdf")
def test_netcdf_dim_length_reads_the_header(cadence_blob, offline_cache):
    """The metadata-only half of the pushdown (``spec/registries.md`` §2.3): the
    records a cadence owner wants are a function of the file's length along the
    record axis, so that length has to be knowable BEFORE the decode the
    selection is meant to narrow. It reads the header, never an array."""
    reader = NetCDFReader()
    assert reader.dim_length(cadence_blob, "time") == 4
    assert reader.dim_length(cadence_blob, "lon") == 3
    assert reader.dim_length(cadence_blob, "no-such-dim") is None
    blob = offline_cache.fetch(ERA5_URL)
    assert reader.dim_length(blob.path, "time") == 2
    # the registry generic answers for a reader that implements it...
    assert registry_dim_length(reader, cadence_blob, "time") == 4
    # ...and "I cannot answer that" for every reader that does not.
    assert registry_dim_length(CSVReader(), cadence_blob, "time") is None
    # `records` is a DECLARED option of this reader, which is what tells a
    # caller (and the cross-track parity check) that it is real in this track.
    assert "records" in reader_option_names(reader)


@pytest.mark.needs_format("netcdf")
def test_netcdf_records_equal_the_whole_file_read(cadence_blob):
    """THE equivalence gate: a record-selected read is cell-for-cell the
    whole-file read gathered afterwards, for every selection a cadence owner can
    produce — a single record, an interior bracket, a reversed pair, the
    degenerate end-of-data ``[last, last]``, the whole axis, a permutation."""
    reader = NetCDFReader()
    whole = reader.read_native(reader.open(cadence_blob))
    w_t2m = np.asarray(whole["t2m"].data)
    w_time = np.asarray(whole["time"].data)

    sweeps = [[a] for a in range(4)]
    sweeps += [[a, b] for a in range(4) for b in range(4)]
    sweeps += [[0, 1, 2, 3], [3, 1, 2, 0]]
    for idx in sweeps:
        got = reader.read_native(
            reader.open(cadence_blob), records={"dim": "time", "indices": idx}
        )
        # The axis is RETAINED at the requested length and never dropped.
        assert list(got["t2m"].dims) == ["time", "lat", "lon"], idx
        assert np.asarray(got["t2m"].data).shape == (len(idx), 2, 3), idx
        assert np.array_equal(np.asarray(got["t2m"].data), w_t2m[idx]), idx
        # the record axis's own coordinate moves WITH the data...
        assert np.array_equal(np.asarray(got["time"].data), w_time[idx]), idx
        assert got["time"].attrs == whole["time"].attrs, idx
        # ...untouched axes stay whole, and a field with no record axis passes
        # through unchanged.
        assert np.array_equal(np.asarray(got["lat"].data), np.asarray(whole["lat"].data))
        assert np.array_equal(np.asarray(got["orog"].data), np.asarray(whole["orog"].data))


@pytest.mark.needs_format("netcdf")
def test_netcdf_records_are_absolute_ordered_and_may_repeat(cadence_blob):
    """Pinned ABSOLUTE values, so the sweep above cannot pass with both paths
    reading the wrong records; and the order given is the order returned, with
    duplicates legal (the end-of-data bracket ``[last, last]`` is one)."""
    reader = NetCDFReader()
    one = reader.read_native(
        reader.open(cadence_blob), records={"dim": "time", "indices": [2]}
    )
    assert np.asarray(one["t2m"].data).ravel().tolist() == [
        300.0, 301.0, 302.0, 310.0, 311.0, 312.0
    ]
    assert np.asarray(one["time"].data).tolist() == [2.0]

    rev = reader.read_native(
        reader.open(cadence_blob), records={"dim": "time", "indices": [3, 1]}
    )
    assert np.asarray(rev["t2m"].data)[:, 0, 0].tolist() == [400.0, 200.0]
    assert np.asarray(rev["time"].data).tolist() == [3.0, 1.0]

    dup = reader.read_native(
        reader.open(cadence_blob), records={"dim": "time", "indices": [3, 3]}
    )
    assert np.asarray(dup["t2m"].data).shape == (2, 2, 3)
    assert np.asarray(dup["t2m"].data)[:, 0, 0].tolist() == [400.0, 400.0]
    assert np.asarray(dup["time"].data).tolist() == [3.0, 3.0]


@pytest.mark.needs_format("netcdf")
def test_netcdf_records_compose_with_variables_and_select(cadence_blob):
    """`variables`, `select` and `records` are three independent narrowings of
    one decode and must compose: project, window, then records."""
    reader = NetCDFReader()
    got = reader.read_native(
        reader.open(cadence_blob),
        ["t2m"],
        {"axes": ["all", {"indices": [1]}, {"slice": [0, 3, 2]}]},
        records={"dim": "time", "indices": [2, 0]},
    )
    assert got.variable_names() == ["t2m"]
    assert np.asarray(got["t2m"].data).shape == (2, 1, 2)
    assert np.asarray(got["t2m"].data).ravel().tolist() == [310.0, 312.0, 110.0, 112.0]
    assert np.asarray(got["time"].data).tolist() == [2.0, 0.0]
    assert np.asarray(got["lat"].data).tolist() == [41.0]
    assert np.asarray(got["lon"].data).tolist() == [-90.0, -88.0]


@pytest.mark.needs_format("netcdf")
def test_netcdf_records_moves_no_boundary_the_select_refusal_drew(cadence_blob):
    """A `select` whose time axis is not "all" is still REFUSED — `records` says
    only that the reader may be TOLD records, never that it may choose them."""
    reader = NetCDFReader()
    with pytest.raises(ValueError, match="record selection is the Provider's"):
        reader.read_native(
            reader.open(cadence_blob), None, {"axes": [{"indices": [0]}, "all", "all"]}
        )
    with pytest.raises(ValueError, match="record selection is the Provider's"):
        reader.read_native(
            reader.open(cadence_blob),
            None,
            {"axes": [{"indices": [0]}, "all", "all"]},
            records={"dim": "time", "indices": [0]},
        )


@pytest.mark.needs_format("netcdf")
def test_netcdf_bad_records_is_an_error_not_a_wrapped_or_widened_read(cadence_blob):
    """The reader is told the records and never the cadence: it does no `mod1`,
    so an out-of-range index is an error, and an empty list is a caller bug
    rather than a licence to read every record."""
    reader = NetCDFReader()
    for bad in ([4], [-1], [0, 9]):
        with pytest.raises(IndexError, match="outside 0:3"):
            reader.read_native(
                reader.open(cadence_blob), records={"dim": "time", "indices": bad}
            )
    with pytest.raises(ValueError, match="which the blob does not have"):
        reader.read_native(
            reader.open(cadence_blob), records={"dim": "nope", "indices": [0]}
        )
    with pytest.raises(ValueError, match="empty selection is an error"):
        reader.read_native(
            reader.open(cadence_blob), records={"dim": "time", "indices": []}
        )
    with pytest.raises(ValueError, match='must be a mapping with "dim" and "indices"'):
        reader.read_native(reader.open(cadence_blob), records={"dim": "time"})
    with pytest.raises(ValueError, match="both narrow dimension"):
        reader.read_native(
            reader.open(cadence_blob),
            None,
            {"axes": ["all", {"indices": [0]}, "all"]},
            records={"dim": "lat", "indices": [0]},
        )
    # The `len -> indices` CALLABLE is a Julia in-process convenience, not the
    # wire form. Refused by name rather than dying inside `int()`, so a caller
    # porting Julia code is told what the portable spelling is.
    with pytest.raises(ValueError, match="must be a list of 0-based indices"):
        reader.read_native(
            reader.open(cadence_blob),
            records={"dim": "time", "indices": lambda n: [n - 1]},
        )


# --------------------------------------------------------------------------- #
# CSV decode (numeric_columns -> float64, others -> string) — the 2nd format.
# --------------------------------------------------------------------------- #


def test_csv_decodes_to_oracle_arrays(offline_cache):
    case = _case("openaq-points-slice")
    blob = offline_cache.fetch(OPENAQ_URL)
    reader = CSVReader()
    nds = reader.read_native(
        reader.open(blob.path),
        numeric_columns=case["decode"]["numeric_columns"],
    )
    for name, spec in case["expected"]["variables"].items():
        assert list(nds[name].dims) == ["index"]
        if spec["dtype"] == "string":
            _assert_string(nds[name], spec["data"], name)
            assert isinstance(nds[name].data, list)
        else:
            _assert_numeric(nds[name], spec["data"], name)
            assert nds[name].data.dtype == np.float64


def test_csv_digit_text_stays_string_only_when_declared(offline_cache):
    blob = offline_cache.fetch(OPENAQ_URL)
    reader = CSVReader()
    # location_id is digit-only ("1","2") but NOT in numeric_columns => string
    nds = reader.read_native(reader.open(blob.path), numeric_columns=["value"])
    assert nds["location_id"].data == ["1", "1", "2", "2"]
    assert nds["value"].data.dtype == np.float64


def test_csv_variable_selection(offline_cache):
    blob = offline_cache.fetch(OPENAQ_URL)
    reader = CSVReader()
    nds = reader.read_native(
        reader.open(blob.path),
        ["value", "location_id"],
        numeric_columns=["value"],
    )
    assert nds.variable_names() == ["location_id", "value"]


# --------------------------------------------------------------------------- #
# FF10 decode — the RAW long-format point table; a NEW reader (77-col schema,
# '#' header skip, RFC-4180 quotes, zip member) — reader-only (no pivot/convert).
# --------------------------------------------------------------------------- #


def _ff10_fixture_text() -> str:
    """A tiny FF10 point blob: a '#' header block + 3 data rows. Rows 1 & 2
    (NOX/SO2) share ONE stack (F001/U1/R1/P1 + stack params + lon/lat), differing
    only in POLID/ANN_VALUE. Row 1 has a quoted-comma FACILITY_NAME and a blank
    DESIGN_CAPACITY (numeric -> NaN)."""
    idx = {n: j for j, n in enumerate(FF10_POINT_COLUMNS)}

    def mkrow(**over):
        r = [""] * len(FF10_POINT_COLUMNS)
        for k, v in over.items():
            r[idx[k]] = v
        return r

    stack = dict(
        COUNTRY_CD="US", REGION_CD="01001", FACILITY_ID="F001", UNIT_ID="U1",
        REL_POINT_ID="R1", PROCESS_ID="P1", SCC="0030700101",
        FACILITY_NAME="Autauga Plant, Unit 1", STKHGT="100.0", STKTEMP="500.0",
        LONGITUDE="-86.51045", LATITUDE="32.43878", ZIPCODE="00000",
    )
    rows = [
        mkrow(**stack, POLID="NOX", ANN_VALUE="123.45"),
        mkrow(**stack, POLID="SO2", ANN_VALUE="67.89"),
        mkrow(COUNTRY_CD="US", REGION_CD="01001", FACILITY_ID="F002",
              POLID="PM25", ANN_VALUE="4.2", FACILITY_NAME="Plain Name"),
    ]
    sio = io.StringIO()
    sio.write("#FORMAT=FF10_POINT\n#COUNTRY US\n\n")  # header block + blank line
    w = _csv.writer(sio, lineterminator="\n")
    for r in rows:
        w.writerow(r)
    return sio.getvalue()


def test_ff10_registered():
    assert "ff10" in format_registry
    assert format_registry.status("ff10") == "active"
    assert isinstance(format_registry.create("ff10"), FF10Reader)
    assert FF10Reader().formats() == ["ff10"]
    assert "csv" in FF10Reader().extensions()
    assert len(FF10_POINT_COLUMNS) == 77
    assert len(FF10_POINT_NUMERIC) == 42


def test_ff10_decodes_to_oracle_arrays(offline_cache):
    case = json.loads((CORPUS / "cases" / "ff10-point-slice.json").read_text())
    blob = offline_cache.fetch(FF10_URL)
    reader = FF10Reader()
    nds = reader.read_native(
        reader.open(blob.path),
        numeric_columns=case["decode"]["numeric_columns"],
    )
    assert isinstance(nds, NativeDataset)
    assert len(nds.variables) == 77
    assert nds.coord_names() == []  # points table: no gridded axis
    for name, spec in case["expected"]["variables"].items():
        assert list(nds[name].dims) == ["index"]
        if spec["dtype"] == "string":
            _assert_string(nds[name], spec["data"], name)
            assert isinstance(nds[name].data, list)
        else:
            _assert_numeric(nds[name], spec["data"], name)
            assert nds[name].data.dtype == np.float64


def test_ff10_header_quote_empty(tmp_path):
    p = tmp_path / "ff10_point.csv"
    p.write_text(_ff10_fixture_text())
    nds = FF10Reader().read_native(str(p))

    assert len(nds.variables) == 77 and nds.coord_names() == []
    assert len(nds["POLID"].data) == 3  # '#' header + blank line skipped
    # numeric vs string typing
    assert nds["ANN_VALUE"].data.dtype == np.float64
    assert list(nds["ANN_VALUE"].data) == [123.45, 67.89, 4.2]
    assert nds["POLID"].data == ["NOX", "SO2", "PM25"]
    # leading-zero codes stay strings
    assert nds["REGION_CD"].data == ["01001", "01001", "01001"]
    assert nds["SCC"].data[0] == "0030700101"
    assert nds["ZIPCODE"].data[0] == "00000"
    # quoted comma preserved verbatim (quotes stripped)
    assert nds["FACILITY_NAME"].data[0] == "Autauga Plant, Unit 1"
    assert nds["FACILITY_NAME"].data[2] == "Plain Name"
    # blank numeric -> NaN; blank string -> ""
    assert math.isnan(nds["DESIGN_CAPACITY"].data[0])
    assert nds["TRIBAL_CODE"].data[0] == ""


def test_ff10_multi_pollutant_same_stack(tmp_path):
    p = tmp_path / "ff10_point.csv"
    p.write_text(_ff10_fixture_text())
    nds = FF10Reader().read_native(str(p))
    # rows 1 & 2 share the stack, differ only in POLID/ANN_VALUE (no pivot)
    assert nds["FACILITY_ID"].data[0] == nds["FACILITY_ID"].data[1] == "F001"
    assert nds["STKHGT"].data[0] == nds["STKHGT"].data[1] == 100.0
    assert nds["POLID"].data[:2] == ["NOX", "SO2"]
    assert list(nds["ANN_VALUE"].data[:2]) == [123.45, 67.89]
    # native units retained (feet / °F), not converted downstream
    assert nds["STKHGT"].data[0] == 100.0
    assert nds["STKTEMP"].data[0] == 500.0


def test_ff10_zip_member_equals_bare(tmp_path):
    text = _ff10_fixture_text()
    # write the bare CSV and decode it
    bare_path = tmp_path / "point.csv"
    bare_path.write_text(text)
    bare = FF10Reader().read_native(str(bare_path))
    # write a zip holding the CSV as member 'inv/point.csv'
    zpath = tmp_path / "2016fd_inputs_point.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("inv/point.csv", text)
    zipped = FF10Reader().read_native(str(zpath), member="inv/point.csv")
    assert list(zipped["ANN_VALUE"].data) == list(bare["ANN_VALUE"].data)
    assert zipped["POLID"].data == bare["POLID"].data
    assert zipped["FACILITY_NAME"].data == bare["FACILITY_NAME"].data
    # a missing member is a clear error
    with pytest.raises(KeyError):
        FF10Reader().read_native(str(zpath), member="nope.csv")


def test_ff10_absent_variable_raises(tmp_path):
    p = tmp_path / "ff10_point.csv"
    p.write_text(_ff10_fixture_text())
    with pytest.raises(KeyError):
        FF10Reader().read_native(str(p), ["NOT_A_COLUMN"])


# --------------------------------------------------------------------------- #
# FF10 zip member selection (members/member_glob) + skip_header_row — the EPA
# 2016fd zip shape: members carry a non-comment `country_cd,…` header line.
# --------------------------------------------------------------------------- #


def _ff10_header_line() -> str:
    """The lowercase 77-field `country_cd,…` header line 2016fd members carry."""
    return ",".join(c.lower() for c in FF10_POINT_COLUMNS)


def _ff10_tiny_row(fac: str, polid: str, ann: str) -> str:
    idx = {n: j for j, n in enumerate(FF10_POINT_COLUMNS)}
    r = [""] * len(FF10_POINT_COLUMNS)
    r[idx["COUNTRY_CD"]] = "US"
    r[idx["REGION_CD"]] = "01001"
    r[idx["FACILITY_ID"]] = fac
    r[idx["POLID"]] = polid
    r[idx["ANN_VALUE"]] = ann
    return ",".join(r)


def _ff10_egu_zip(tmp_path) -> str:
    """A zip with two `*egu*` members + one excluded member, each carrying a `#`
    comment block and the `country_cd` header line. Written in NON-sorted order
    to prove the read order is sorted-name."""
    def member(rows):
        return "#FORMAT=FF10_POINT\n" + _ff10_header_line() + "\n" + "\n".join(rows) + "\n"

    zpath = tmp_path / "2016fd_inputs_point.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        # a glob-matching DIRECTORY placeholder entry (like the real 2016fd
        # `…/ptegu/`) — selection must ignore it (file members only).
        zf.writestr("point_egu/", b"")
        zf.writestr("point/egu_beta.csv", member([_ff10_tiny_row("F202", "NOX", "333.3")]))
        zf.writestr("point/egu_alpha.csv", member([
            _ff10_tiny_row("F101", "NOX", "111.1"),
            _ff10_tiny_row("F101", "SO2", "22.2"),
        ]))
        zf.writestr("point/ptnonipm.csv", member([_ff10_tiny_row("F999", "NOX", "999.9")]))
    return str(zpath)


def test_ff10_member_glob_concatenates_sorted(tmp_path):
    zpath = _ff10_egu_zip(tmp_path)
    nds = FF10Reader().read_native(zpath, member_glob="*egu*", skip_header_row=True)
    # alpha (2 rows) sorts before beta (1 row); ptnonipm excluded.
    assert nds["FACILITY_ID"].data == ["F101", "F101", "F202"]
    assert list(nds["ANN_VALUE"].data) == [111.1, 22.2, 333.3]
    assert "F999" not in nds["FACILITY_ID"].data


def test_ff10_member_glob_zero_matches_raises(tmp_path):
    zpath = _ff10_egu_zip(tmp_path)
    with pytest.raises(ValueError, match="matched no members"):
        FF10Reader().read_native(zpath, member_glob="*nope*")


def test_ff10_members_union_glob_and_missing_name(tmp_path):
    zpath = _ff10_egu_zip(tmp_path)
    # explicit list ∪ glob, deduplicated, sorted-name concatenation order.
    nds = FF10Reader().read_native(
        zpath,
        members=["point/ptnonipm.csv", "point/egu_alpha.csv"],
        member_glob="*egu*",
        skip_header_row=True,
    )
    assert nds["FACILITY_ID"].data == ["F101", "F101", "F202", "F999"]
    # an explicit member absent from the archive is an error.
    with pytest.raises(KeyError, match="not found"):
        FF10Reader().read_native(zpath, members=["point/absent.csv"],
                                 skip_header_row=True)


def test_ff10_header_row_skipped_exactly_once_per_member(tmp_path):
    zpath = _ff10_egu_zip(tmp_path)
    # WITHOUT skip_header_row, the header line is a 77-field data row that dies
    # at the numeric parse of ann_value — it is never silently accepted.
    with pytest.raises(ValueError):
        FF10Reader().read_native(zpath, member_glob="*egu*")
    # singular member + skip_header_row composes.
    nds = FF10Reader().read_native(zpath, member="point/egu_beta.csv",
                                   skip_header_row=True)
    assert nds["FACILITY_ID"].data == ["F202"]


def test_ff10_skip_header_row_asserts_header_present(tmp_path):
    # asserting a header on an input that has none errors — never a silently
    # dropped data row.
    p = tmp_path / "noheader.csv"
    p.write_text(_ff10_fixture_text())
    with pytest.raises(ValueError, match="refusing to drop"):
        FF10Reader().read_native(str(p), skip_header_row=True)


def test_ff10_member_exclusive_with_members(tmp_path):
    zpath = _ff10_egu_zip(tmp_path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        FF10Reader().read_native(zpath, member="point/egu_beta.csv",
                                 member_glob="*egu*")


def test_ff10_zip_case_decodes_to_oracle_arrays(offline_cache):
    """The committed ff10-zip-egu-glob case decodes through the reader's own
    zip path (member_glob + skip_header_row) to the oracle arrays."""
    case = json.loads((CORPUS / "cases" / "ff10-zip-egu-glob.json").read_text())
    blob = offline_cache.fetch(case["resolved_url"])
    dec = case["decode"]
    nds = FF10Reader().read_native(
        blob.path,
        numeric_columns=dec["numeric_columns"],
        member_glob=dec["member_glob"],
        skip_header_row=dec["skip_header_row"],
    )
    assert len(nds.variables) == 77
    for name, spec in case["expected"]["variables"].items():
        if spec["dtype"] == "string":
            _assert_string(nds[name], spec["data"], name)
        else:
            _assert_numeric(nds[name], spec["data"], name)

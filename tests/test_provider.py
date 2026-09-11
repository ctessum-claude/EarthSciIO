"""The cadence Provider over the shared corpus — CONST/DISCRETE, OFFLINE.

Acceptance (``esio-9nb.3``): "A Provider over a fixture loader returns correct
native arrays for CONST (materialize once) and DISCRETE (refresh at each anchor);
refresh_times() matches the temporal cadence; unit-tested standalone (no campfire
dependency to pass)." Drives the full (a)+(b) pipeline offline: cache (shared
corpus) → format reader → native arrays, plus the cadence surface the solver
consumes (materialize/refresh/refresh_times/prefetch). Mirrors the peer
``julia/test/test_provider.jl`` and ``rust/tests/provider_cadence.rs``.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import pathlib

import numpy as np
import pytest

from earthsciio import (
    BackendNotRegistered,
    Cache,
    DataSource,
    SourceTemporal,
    Provider,
    UnknownReaderOption,
    format_registry,
    reader_option_names,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / "conformance" / "corpus"

ERA5_URL = "https://data.earthsci.dev/era5/2018/11/20181108.nc"
OPENAQ_URL = (
    "https://openaq-data-archive.s3.amazonaws.com/records/openaq/"
    "locationid=1/2018-11-08.csv"
)

START = dt.datetime(2018, 11, 8, tzinfo=dt.timezone.utc)
HOUR = dt.timedelta(hours=1)
DAY = dt.timedelta(days=1)


def _utc(h: int) -> dt.datetime:
    return START + h * HOUR


@pytest.fixture
def cache():
    """A read-only offline cache rooted at the conformance corpus (no network)."""
    return Cache(root=CORPUS / "cache", offline=True, verify=True)


def _era5_case() -> dict:
    return json.loads((CORPUS / "cases" / "era5-grid-sub-tile.json").read_text())


def _flat(x):
    if isinstance(x, list):
        for e in x:
            yield from _flat(e)
    else:
        yield x


def _match(field, expected_nested) -> bool:
    exp = np.array(
        [math.nan if v is None else float(v) for v in _flat(expected_nested)],
        dtype="f8",
    )
    got = np.asarray(field.data, dtype="f8").reshape(-1)
    if got.shape != exp.shape:
        return False
    gn, en = np.isnan(got), np.isnan(exp)
    return bool(np.array_equal(gn, en) and np.allclose(got[~gn], exp[~en], atol=1e-6, rtol=1e-9))


# --------------------------------------------------------------------------- #
# CONST grid: empty cadence, materialize once, native arrays match the oracle.
# --------------------------------------------------------------------------- #


@pytest.mark.needs_format("netcdf")
def test_const_provider_materializes_oracle_arrays(cache):
    case = _era5_case()
    p = Provider(DataSource("era5", "netcdf", ERA5_URL), cache)
    assert p.is_const
    assert p.refresh_times() == []  # CONST => never refreshes

    nds = p.materialize()
    assert nds.variable_names() == ["sp", "t2m"]
    assert nds.coord_names() == ["latitude", "longitude", "time"]
    # full native-array equality vs the oracle (checks 3–4 through the Provider)
    for group in ("variables", "coords"):
        for name, spec in case["expected"][group].items():
            assert _match(nds[name], spec["data"]), name
    # the raw time axis is undecoded with its calendar carried for ESS
    assert nds["time"].data.dtype == np.int32
    assert nds["time"].attrs["calendar"] == "gregorian"
    # coords property exposes the current grid
    assert set(p.coords) == {"latitude", "longitude", "time"}


@pytest.mark.needs_format("netcdf")
def test_netcdf_per_call_select_overrides_the_baked_one(cache):
    """The whole-file netcdf reader honours a `select` at decode time, and the
    per-call value OVERRIDES a baked ``reader_kwargs["select"]`` — the same
    precedence the store-backed zarr reader has (``tests/test_zarr_reader.py``).
    Neither changes the URL or the cache key: the same blob is read either way.
    """
    baked = {"axes": ["all", {"slice": [1, 3]}, {"indices": [0, 2]}]}
    p = Provider(
        DataSource("era5", "netcdf", ERA5_URL, reader_kwargs={"select": baked}), cache
    )
    assert p.supports_selection is True
    assert p.store_backed is False

    # the baked selection is honoured with no per-call argument at all
    assert p.materialize()["t2m"].shape == (2, 2, 2)

    # a per-call select wins over the baked one for this call only...
    peek = p.materialize(select={"axes": ["all", {"indices": [0]}, "all"]})
    assert peek["t2m"].shape == (2, 1, 3)
    assert peek["latitude"].shape == (1,)
    # ...and it is a peek: the baked projection is what the next plain read gives
    assert p.materialize()["t2m"].shape == (2, 2, 2)

    # a per-call select on a provider with NO baked one is still honoured
    q = Provider(DataSource("era5", "netcdf", ERA5_URL), cache)
    assert q.materialize()["t2m"].shape == (2, 3, 3)
    assert q.materialize(select=baked)["t2m"].shape == (2, 2, 2)
    assert q.materialize()["t2m"].shape == (2, 3, 3)


@pytest.mark.needs_format("netcdf")
def test_const_refresh_returns_constant_data(cache):
    p = Provider(DataSource("era5", "netcdf", ERA5_URL), cache)
    a = p.refresh(_utc(0))  # CONST: refresh is the constant data (materialize once)
    b = p.refresh(_utc(99))
    assert np.array_equal(a["sp"].data, b["sp"].data)
    assert a["t2m"].shape == (2, 3, 3)


# --------------------------------------------------------------------------- #
# DISCRETE grid: refresh_times match cadence; per-anchor record slice.
# --------------------------------------------------------------------------- #


def _discrete_internal_axis() -> DataSource:
    # one file holds the day's hourly records; cadence slices the internal axis
    temporal = SourceTemporal(start=START, frequency=HOUR, file_period=2 * HOUR)
    return DataSource("era5", "netcdf", ERA5_URL, temporal=temporal)


def test_discrete_refresh_times_match_cadence(cache):
    p = Provider(_discrete_internal_axis(), cache, window=(_utc(0), _utc(2)))
    assert not p.is_const
    assert p.refresh_times() == [_utc(0), _utc(1)]  # the hourly cadence


@pytest.mark.needs_format("netcdf")
def test_discrete_refresh_slices_record_per_anchor(cache):
    p = Provider(_discrete_internal_axis(), cache, window=(_utc(0), _utc(2)))
    s0 = p.refresh(_utc(0))
    s1 = p.refresh(_utc(1))
    # the time record is sliced out: (time, lat, lon) -> (lat, lon)
    assert list(s0["t2m"].dims) == ["latitude", "longitude"]
    assert s0["t2m"].shape == (3, 3)
    assert s0["t2m"].data[0, 0] == pytest.approx(282.5)
    assert s1["t2m"].data[0, 0] == pytest.approx(282.6)  # a different record per tick
    assert math.isnan(s1["t2m"].data[2, 2])  # the masked cell survives the slice
    assert "time" not in s0  # the sliced dim's coordinate is dropped
    assert set(p.coords) == {"latitude", "longitude"}  # time dropped from coords too


@pytest.mark.needs_format("netcdf")
def test_discrete_materialize_primes_first_anchor(cache):
    p = Provider(_discrete_internal_axis(), cache, window=(_utc(0), _utc(2)))
    primed = p.materialize()
    assert np.array_equal(primed["t2m"].data, p.refresh(_utc(0))["t2m"].data)


@pytest.mark.needs_format("netcdf")
def test_discrete_between_anchors_uses_active_record(cache):
    p = Provider(_discrete_internal_axis(), cache, window=(_utc(0), _utc(2)))
    at_anchor = p.refresh(_utc(0))["t2m"].data.copy()
    between = p.refresh(START + dt.timedelta(minutes=30))  # snaps down to hour 0
    assert np.array_equal(between["t2m"].data, at_anchor)


@pytest.mark.needs_format("netcdf")
def test_discrete_refresh_is_idempotent_within_interval(cache):
    p = Provider(_discrete_internal_axis(), cache, window=(_utc(0), _utc(2)))
    a = p.refresh(_utc(1))["sp"].data
    b = p.refresh(_utc(1))["sp"].data
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# DISCRETE records_per_sample=2: the 2-record bracket for downstream time
# interpolation (floor + successor, time axis retained, epoch-seconds time coord).
# --------------------------------------------------------------------------- #


def _interp_loader(end=None, file_period=2 * HOUR) -> DataSource:
    temporal = SourceTemporal(start=START, frequency=HOUR, file_period=file_period,
                              end=end, records_per_sample=2)
    return DataSource("era5", "netcdf", ERA5_URL, temporal=temporal)


@pytest.mark.needs_format("netcdf")
def test_interp_bracket_returns_two_records_with_time_axis(cache):
    p = Provider(_interp_loader(), cache, window=(_utc(0), _utc(2)))
    b = p.refresh(_utc(0))
    # time axis is RETAINED at length 2 (floor + successor), unlike the single slice
    assert list(b["t2m"].dims) == ["time", "latitude", "longitude"]
    assert b["t2m"].shape == (2, 3, 3)
    assert b["t2m"].data[0, 0, 0] == pytest.approx(282.5)  # record 0 (hour 0)
    assert b["t2m"].data[1, 0, 0] == pytest.approx(282.6)  # record 1 (hour 1)
    # the time coordinate carries the two bracket timestamps as Unix epoch seconds
    assert "time" in b
    assert list(b["time"].data) == pytest.approx([START.timestamp(), _utc(1).timestamp()])
    assert b["time"].attrs["units"] == "seconds since 1970-01-01T00:00:00Z"


@pytest.mark.needs_format("netcdf")
def test_interp_bracket_floors_within_interval(cache):
    p = Provider(_interp_loader(), cache, window=(_utc(0), _utc(2)))
    at = p.refresh(_utc(0))["t2m"].data
    between = p.refresh(START + dt.timedelta(minutes=30))["t2m"].data  # same bracket
    assert np.array_equal(at, between, equal_nan=True)  # masked cell in record 1
    # ... but the bracket timestamps still describe hour 0 -> hour 1
    b = p.refresh(START + dt.timedelta(minutes=30))
    assert list(b["time"].data) == pytest.approx([START.timestamp(), _utc(1).timestamp()])


@pytest.mark.needs_format("netcdf")
def test_interp_bracket_crosses_file_boundary(cache):
    # No temporal.end => a successor is assumed to exist; hour 1 is the last record
    # in the 2-record file, so its "after" record is record 0 of the next file
    # (the same corpus blob resolves for every anchor here).
    p = Provider(_interp_loader(), cache)
    b = p.refresh(_utc(1))
    assert b["t2m"].shape == (2, 3, 3)
    assert b["t2m"].data[0, 0, 0] == pytest.approx(282.6)  # this file, record 1
    assert b["t2m"].data[1, 0, 0] == pytest.approx(282.5)  # next file, record 0
    assert list(b["time"].data) == pytest.approx([_utc(1).timestamp(), _utc(2).timestamp()])


@pytest.mark.needs_format("netcdf")
def test_interp_bracket_end_clamp_degenerates(cache):
    # temporal.end=hour 2 => hour 1 has no successor; the bracket holds [last, last]
    p = Provider(_interp_loader(end=_utc(2)), cache, window=(_utc(0), _utc(2)))
    b = p.refresh(_utc(1))
    assert b["t2m"].data[0, 0, 0] == pytest.approx(282.6)
    assert b["t2m"].data[1, 0, 0] == pytest.approx(282.6)  # successor == floor (held)
    t0, t1 = b["time"].data
    assert t0 == pytest.approx(t1)  # degenerate bracket => downstream weight clamps


def test_interp_rejects_bad_records_per_sample():
    with pytest.raises(ValueError):
        SourceTemporal(start=START, frequency=HOUR, file_period=HOUR, records_per_sample=3)


# --------------------------------------------------------------------------- #
# refresh_times bounds: window end, temporal.end, and the unbounded case.
# --------------------------------------------------------------------------- #


def test_refresh_times_bounded_by_temporal_end_without_window(cache):
    temporal = SourceTemporal(start=START, frequency=HOUR, file_period=2 * HOUR, end=_utc(3))
    p = Provider(DataSource("era5", "netcdf", ERA5_URL, temporal=temporal), cache)
    assert p.refresh_times() == [_utc(0), _utc(1), _utc(2)]


def test_refresh_times_empty_when_unbounded(cache):
    temporal = SourceTemporal(start=START, frequency=HOUR, file_period=2 * HOUR)
    p = Provider(DataSource("era5", "netcdf", ERA5_URL, temporal=temporal), cache)
    assert p.refresh_times() == []  # no window, no end => no enumerable schedule


def test_refresh_times_window_start_clamped_to_epoch(cache):
    # window starts mid-cadence; the first tstop is the aligned anchor >= start
    temporal = SourceTemporal(start=START, frequency=HOUR, file_period=DAY)
    p = Provider(
        DataSource("era5", "netcdf", ERA5_URL, temporal=temporal),
        cache,
        window=(START + dt.timedelta(minutes=30), _utc(3)),
    )
    assert p.refresh_times() == [_utc(1), _utc(2)]


# --------------------------------------------------------------------------- #
# Per-file URL resolution (strftime template + callable) and prefetch.
# --------------------------------------------------------------------------- #


def test_strftime_url_template_resolves_per_anchor():
    loader = DataSource("era5", "netcdf", "https://data.earthsci.dev/era5/%Y/%m/%Y%m%d.nc")
    assert loader.resolve_url(START) == ERA5_URL


def test_prefetch_const_warms_single_file(cache):
    p = Provider(DataSource("era5", "netcdf", ERA5_URL), cache)
    entries = p.prefetch()
    assert len(entries) == 1
    assert entries[0].status == "hit"  # offline corpus hit, no decode


def test_prefetch_strftime_window_hits_corpus(cache):
    temporal = SourceTemporal(start=START, frequency=HOUR, file_period=DAY)
    loader = DataSource("era5", "netcdf", "https://data.earthsci.dev/era5/%Y/%m/%Y%m%d.nc",
                        temporal=temporal)
    p = Provider(loader, cache, window=(START, START + DAY))
    entries = p.prefetch()
    assert len(entries) == 1 and entries[0].status == "hit"


def test_prefetch_enumerates_file_anchors_and_dedups(cache):
    seen = []

    def url_for(anchor):
        seen.append(anchor)
        return ERA5_URL  # every file anchor collapses to the one corpus blob

    temporal = SourceTemporal(start=START, frequency=HOUR, file_period=HOUR)
    p = Provider(DataSource("era5", "netcdf", url_for, temporal=temporal), cache,
                 window=(_utc(0), _utc(2)))
    entries = p.prefetch()
    assert seen == [_utc(0), _utc(1)]  # one anchor per file period across the window
    assert len(entries) == 1  # collapsed to a single unique fetch
    assert entries[0].status == "hit"


def test_prefetch_unbounded_raises(cache):
    temporal = SourceTemporal(start=START, frequency=HOUR, file_period=HOUR)
    p = Provider(DataSource("era5", "netcdf", ERA5_URL, temporal=temporal), cache)
    with pytest.raises(ValueError):
        p.prefetch()  # no window, no temporal.end


# --------------------------------------------------------------------------- #
# CSV points provider + variable selection (the 2nd format, end to end).
# --------------------------------------------------------------------------- #


def test_csv_provider_with_variable_selection(cache):
    loader = DataSource(
        "openaq",
        "csv",
        OPENAQ_URL,
        variables=["value", "location_id"],
        reader_kwargs={"numeric_columns": ["latitude", "longitude", "value"]},
    )
    nds = Provider(loader, cache).materialize()
    assert nds.variable_names() == ["location_id", "value"]  # restricted
    assert list(nds["value"].data) == [152.3, 168.7, 98.1, 110.4]
    assert nds["value"].data.dtype == np.float64
    assert nds["location_id"].data == ["1", "1", "2", "2"]  # digit text stays string


# --------------------------------------------------------------------------- #
# Construction + use guards.
# --------------------------------------------------------------------------- #


def test_unknown_format_raises_at_construction(cache):
    with pytest.raises(BackendNotRegistered):
        Provider(DataSource("x", "nonesuch", ERA5_URL), cache)


# --------------------------------------------------------------------------- #
# Reader options (spec/registries.md §2.1): a loader DECLARES how its format
# decodes, and an option the bound reader does not know is an ERROR at
# construction — the same rule the Rust `Reader::configured` enforces, so a
# document that decodes correctly in one binding decodes correctly in all.
# --------------------------------------------------------------------------- #


def test_unrecognised_reader_option_raises_at_construction(cache):
    """A mis-typed decode option must not be swallowed by the reader's ``**_``
    catch-all: an ignored option reads back much later as an empty selection or
    a mis-parsed table, arbitrarily far from its cause."""
    loader = DataSource(
        "openaq",
        "csv",
        OPENAQ_URL,
        reader_kwargs={"numeric_colums": ["value"]},  # sic — one letter off
    )
    with pytest.raises(UnknownReaderOption) as e:
        Provider(loader, cache)
    assert "numeric_colums" in str(e.value)
    assert "numeric_columns" in str(e.value)  # names what it DOES accept


def test_recognised_reader_options_are_the_reader_signature(cache):
    """The accepted set is the decode entry point's own keyword parameters, so
    it cannot drift from the code that consumes them."""
    accepted = reader_option_names(format_registry.create("csv"))
    assert {"numeric_columns", "delimiter", "header_row", "select"} <= accepted
    assert "handle" not in accepted and "variables" not in accepted
    # An empty option set never errors, whatever the reader accepts.
    Provider(DataSource("openaq", "csv", OPENAQ_URL), cache)


@pytest.mark.needs_format("netcdf")
def test_records_is_a_declared_netcdf_option(cache):
    """The `records` pushdown is a real decode option of THIS track's netcdf
    reader (``spec/registries.json`` ``format.netcdf.reader_options``), not a
    spec-only promise a Python caller discovers as a ``TypeError``."""
    accepted = reader_option_names(format_registry.create("netcdf"))
    assert {"select", "records"} <= accepted


@pytest.mark.needs_format("netcdf")
def test_baked_records_beside_a_cadence_is_refused_not_honoured(cache):
    """Making `records` a declared option would otherwise let a caller bake one
    into ``reader_kwargs`` beside a cadence, narrowing the record axis BEHIND the
    Provider's own anchor→record resolution.

    The Provider then locates the anchor inside an already-narrowed axis: with a
    one-record `records` the file's time coordinate has a single value, every
    anchor resolves to record 0, and ``refresh`` hands back the same record
    forever — a constant field where the model expected to interpolate, with no
    error anywhere. Refused at construction (``spec/registries.md`` §2.1); the
    Julia track refuses the same shape for the same reason.
    """
    temporal = SourceTemporal(start=START, frequency=HOUR, file_period=2 * HOUR)
    baked = {"dim": "time", "indices": [0]}
    with pytest.raises(ValueError, match="`records` may not be baked"):
        Provider(
            DataSource("era5", "netcdf", ERA5_URL, temporal=temporal,
                       reader_kwargs={"records": baked}),
            cache,
        )
    # Left unrefused, that is what it would have done: every anchor the same
    # record. Shown through the reader the Provider would have called, so the
    # assertion pins the wrong NUMBER the refusal prevents rather than merely
    # restating the rule.
    reader = format_registry.create("netcdf")
    blob = cache.fetch(ERA5_URL)
    narrowed = reader.read_native(reader.open(blob.path), None, records=baked)
    assert len(narrowed["time"].data) == 1  # ...so `mod1(anchor, 1)` is always 0
    whole = reader.read_native(reader.open(blob.path))
    assert len(whole["time"].data) == 2
    p = Provider(DataSource("era5", "netcdf", ERA5_URL, temporal=temporal), cache)
    assert p.refresh(_utc(0))["t2m"].data[0, 0] != pytest.approx(
        p.refresh(_utc(1))["t2m"].data[0, 0]
    )

    # `records = None` stays legal — it states the whole record axis, which is
    # what this track reads — and so does a `records` on a CONST loader, which
    # owns no cadence and therefore no record axis to narrow behind.
    Provider(
        DataSource("era5", "netcdf", ERA5_URL, temporal=temporal,
                   reader_kwargs={"records": None}),
        cache,
    )
    const = Provider(
        DataSource("era5", "netcdf", ERA5_URL, reader_kwargs={"records": baked}), cache
    )
    assert len(const.materialize()["time"].data) == 1


def test_a_store_backed_reader_declares_its_options_on_read_store():
    """A store-backed reader's decode entry point is ``read_store``, so that is
    the signature the check reads (the zarr reader takes only ``select``)."""
    accepted = reader_option_names(format_registry.create("zarr"))
    assert "select" in accepted
    assert "cache" not in accepted and "base_url" not in accepted


def test_loader_temporal_rejects_nonpositive_cadence():
    with pytest.raises(ValueError):
        SourceTemporal(start=START, frequency=dt.timedelta(0), file_period=HOUR)
    with pytest.raises(ValueError):
        SourceTemporal(start=START, frequency=HOUR, file_period=dt.timedelta(0))


def test_refresh_before_start_raises(cache):
    p = Provider(_discrete_internal_axis(), cache, window=(_utc(0), _utc(2)))
    with pytest.raises(ValueError):
        p.refresh(START - HOUR)


@pytest.mark.needs_format("netcdf")
def test_refresh_record_out_of_range_raises(cache):
    # file_period=DAY claims 24 hourly records, but the fixture file holds 2
    temporal = SourceTemporal(start=START, frequency=HOUR, file_period=DAY)
    p = Provider(DataSource("era5", "netcdf", ERA5_URL, temporal=temporal), cache)
    with pytest.raises(IndexError):
        p.refresh(_utc(5))  # record 5 absent from the 2-record file


@pytest.mark.needs_format("netcdf")
def test_absent_variable_raises(cache):
    p = Provider(DataSource("era5", "netcdf", ERA5_URL, variables=["nope"]), cache)
    with pytest.raises(KeyError):
        p.materialize()

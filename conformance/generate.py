#!/usr/bin/env python3
"""Reproducible generator for the EarthSciIO conformance corpus.

This script is the single source of truth for the golden fixtures under
``conformance/corpus/``. It writes, deterministically:

  * the cached *blobs* (a tiny real NetCDF-3 grid file + a CSV points file),
    laid out exactly as a populated ``$EARTHSCIDATADIR`` cache
    (``cache/v1/blobs/<key[:2]>/<key>.<ext>``) so a provider in offline mode
    can be pointed straight at ``corpus/cache`` and find every blob by hashing
    its resolved URL;
  * the per-blob *manifests* (``cache/v1/meta/<key>.json``);
  * the language-neutral conformance *cases* (``corpus/cases/*.json``) carrying
    the expected CF-decoded native arrays + coordinates;
  * the case index (``corpus/cases.json``).

Determinism: the NetCDF blob is written as ``NETCDF3_CLASSIC`` (no embedded
HDF5 timestamps/UUIDs), all data values are fixed, and ``fetched_at`` in the
manifests is a pinned constant. Re-running this script on the same numpy /
netCDF4 stack reproduces byte-identical blobs. Conformance readers consume the
*committed* blobs, so other language tracks need no Python at all.

Run from anywhere:  ``python3 conformance/generate.py``

Spec references: ../spec/cache-format.md, ../spec/conformance.md,
../spec/schemas/manifest.schema.json, ../spec/schemas/cache-case.schema.json.
"""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import os
import pathlib
import zipfile

import numpy as np

# --- spec constants (keep in sync with ../spec/cache-format.md) ---------------
CACHE_FORMAT_VERSION = "v1"
# Pinned so manifests are byte-stable across regenerations (never "now").
FIXED_FETCHED_AT = "2026-06-26T00:00:00Z"

HERE = pathlib.Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
CACHE_ROOT = CORPUS / "cache" / CACHE_FORMAT_VERSION
CASES_DIR = CORPUS / "cases"


def cache_key(resolved_url: str) -> str:
    """The shared cache key: sha256 of the resolved URL, lowercase hex.

    The URL is encoded as UTF-8 with no trailing newline, exactly as resolved
    (after time-anchor + parameter expansion). This MUST be identical across
    Python / Julia / Rust so a file fetched by one language is reused by the
    others. See ../spec/cache-format.md#1-cache-key.
    """
    return hashlib.sha256(resolved_url.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def blob_relpath(key: str, ext: str) -> str:
    # A bare key (no extension, e.g. a Zarr chunk/metadata object) is stored
    # WITHOUT a trailing dot, matching the real LocalStore (glob lookup is by
    # <key>*, so the suffix is human-debug only).
    suffix = f".{ext}" if ext else ""
    return f"cache/{CACHE_FORMAT_VERSION}/blobs/{key[:2]}/{key}{suffix}"


def meta_relpath(key: str) -> str:
    return f"cache/{CACHE_FORMAT_VERSION}/meta/{key}.json"


def write_blob(key: str, ext: str, data: bytes) -> str:
    rel = blob_relpath(key, ext)
    path = CORPUS / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return rel


def write_manifest(key: str, manifest: dict) -> str:
    rel = meta_relpath(key)
    path = CORPUS / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return rel


def write_json(path: pathlib.Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


# -----------------------------------------------------------------------------
# Fixture 1 — ERA5-like NetCDF grid sub-tile (transport=file, format=netcdf).
#
# Exercises the cross-language CF-decode parity risk directly: one *packed*
# variable (int16 + scale_factor/add_offset/_FillValue) and one plain float64
# variable, a CF time axis (hours since ... + calendar). The "native array" a
# conformant reader returns is the CF-DECODED value as float64, keyed by the
# on-disk file_variable name. Variable-name remap + unit_conversion are NOT the
# reader's job (they stay in ESS) — see ../spec/conformance.md#decode.
# -----------------------------------------------------------------------------
def build_era5_netcdf() -> tuple[bytes, dict, dict]:
    from netCDF4 import Dataset  # lazy: only the netcdf fixture needs it

    lat = np.array([40.0, 39.5, 39.0], dtype="f8")        # N->S, ERA5 order
    lon = np.array([-122.0, -121.5, -121.0], dtype="f8")  # Camp Fire vicinity
    time = np.array([0, 1], dtype="i4")  # hours since 2018-11-08 00:00:00

    # t2m: target DECODED values (Kelvin). Packed as int16 with these CF attrs.
    scale_factor = 0.01
    add_offset = 280.0
    fill_short = np.int16(-32767)
    t2m_decoded = np.array(
        [
            [[282.50, 282.75, 283.00],
             [283.25, 283.50, 283.75],
             [284.00, 284.25, 284.50]],
            [[282.60, 282.85, 283.10],
             [283.35, 283.60, 283.85],
             [284.10, 284.35, np.nan]],  # one masked cell -> _FillValue on disk
        ],
        dtype="f8",
    )
    # Pack to int16 exactly as CF specifies: raw = round((value - off) / scale).
    raw = np.empty(t2m_decoded.shape, dtype="i2")
    mask = np.isnan(t2m_decoded)
    raw[~mask] = np.round((t2m_decoded[~mask] - add_offset) / scale_factor).astype("i2")
    raw[mask] = fill_short

    # sp: plain float64 surface pressure (Pa), no packing, no fills.
    sp = np.array(
        [
            [[100000.0, 100100.0, 100200.0],
             [100300.0, 100400.0, 100500.0],
             [100600.0, 100700.0, 100800.0]],
            [[100050.0, 100150.0, 100250.0],
             [100350.0, 100450.0, 100550.0],
             [100650.0, 100750.0, 100850.0]],
        ],
        dtype="f8",
    )

    buf = io.BytesIO()
    # netCDF4 needs a filename; write to a temp path then read bytes back so the
    # committed artifact is exactly what lands on disk.
    tmp = CORPUS / ".tmp_era5.nc"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    ds = Dataset(tmp, "w", format="NETCDF3_CLASSIC")
    ds.createDimension("time", None)  # record dim
    ds.createDimension("latitude", lat.size)
    ds.createDimension("longitude", lon.size)

    vlat = ds.createVariable("latitude", "f8", ("latitude",))
    vlat.units = "degrees_north"
    vlat.standard_name = "latitude"
    vlat[:] = lat

    vlon = ds.createVariable("longitude", "f8", ("longitude",))
    vlon.units = "degrees_east"
    vlon.standard_name = "longitude"
    vlon[:] = lon

    vtime = ds.createVariable("time", "i4", ("time",))
    vtime.units = "hours since 2018-11-08 00:00:00"
    vtime.calendar = "gregorian"
    vtime.standard_name = "time"
    vtime[:] = time

    vt2m = ds.createVariable("t2m", "i2", ("time", "latitude", "longitude"),
                             fill_value=fill_short)
    # Write the already-packed int16 verbatim: disable netCDF4's auto pack/mask
    # so our hand-computed raw values (and the -32767 fill cell) land as-is. CF
    # decoding happens on READ (xarray / NCDatasets / netcdf-rs), not on write.
    vt2m.set_auto_maskandscale(False)
    vt2m.scale_factor = scale_factor
    vt2m.add_offset = add_offset
    vt2m.units = "K"
    vt2m.long_name = "2 metre temperature"
    vt2m[:] = raw

    vsp = ds.createVariable("sp", "f8", ("time", "latitude", "longitude"))
    vsp.units = "Pa"
    vsp.long_name = "Surface pressure"
    vsp[:] = sp

    ds.close()
    data = tmp.read_bytes()
    tmp.unlink()
    buf.write(data)

    # Expected native arrays = CF-decoded float64 (fill -> null/NaN).
    def f64(a):
        return [[[None if np.isnan(v) else round(float(v), 10) for v in row]
                 for row in slab] for slab in a]

    expected = {
        "variables": {
            "t2m": {
                "dtype": "float64",
                "dims": ["time", "latitude", "longitude"],
                "shape": [2, 3, 3],
                "fill_value": None,
                "data": f64(t2m_decoded),
            },
            "sp": {
                "dtype": "float64",
                "dims": ["time", "latitude", "longitude"],
                "shape": [2, 3, 3],
                "fill_value": None,
                "data": f64(sp),
            },
        },
        "coords": {
            "latitude": {"dtype": "float64", "data": [round(float(v), 10) for v in lat]},
            "longitude": {"dtype": "float64", "data": [round(float(v), 10) for v in lon]},
            "time": {
                "dtype": "int32",
                "units": "hours since 2018-11-08 00:00:00",
                "calendar": "gregorian",
                "data": [int(v) for v in time],
            },
        },
    }
    decode = {
        "scale_factor_offset": True,
        "fill_to_nan": True,
        "time_decoded": False,  # raw hours retained; calendar decoding is ESS's job
    }
    return data, expected, decode



# -----------------------------------------------------------------------------
# Fixture 1b — the SAME ERA5 blob read through a decode-time `select`
# (transport=file, format=netcdf).
#
# A whole-file reader that honours an orthogonal selection still fetches one blob
# under one cache key; only the requested hyperslab is materialised. The case
# pins the two halves a windowed netCDF read can get wrong and zarr never could:
# the array window itself, and the COORDINATES that must be sliced with it (the
# zarr reader returns no coords, so this question has never been asked of the
# corpus). The time axis stays "all" — record selection is the Provider's.
# -----------------------------------------------------------------------------
ERA5_WINDOW = {"axes": ["all", {"slice": [1, 3]}, {"indices": [0, 2]}]}


def window_era5_expected(expected: dict) -> dict:
    """Slice the full ERA5 expectation by ERA5_WINDOW — the oracle stated as
    "the full read, sliced afterwards", which is exactly the acceptance gate."""
    lat = [1, 2]      # {"slice": [1, 3]}
    lon = [0, 2]      # {"indices": [0, 2]}
    out_vars = {}
    for name, spec in expected["variables"].items():
        data = [[[slab[y][x] for x in lon] for y in lat] for slab in spec["data"]]
        out_vars[name] = {
            "dtype": spec["dtype"],
            "dims": list(spec["dims"]),
            "shape": [len(data), len(lat), len(lon)],
            "fill_value": spec["fill_value"],
            "data": data,
        }
    coords = expected["coords"]
    out_coords = {
        "latitude": {"dtype": coords["latitude"]["dtype"],
                     "data": [coords["latitude"]["data"][y] for y in lat]},
        "longitude": {"dtype": coords["longitude"]["dtype"],
                      "data": [coords["longitude"]["data"][x] for x in lon]},
        # the time axis is untouched: raw, with its units + calendar
        "time": dict(coords["time"]),
    }
    return {"variables": out_vars, "coords": out_coords}


# -----------------------------------------------------------------------------
# Fixture 2 — OpenAQ-like CSV points slice (transport=file, format=csv).
#
# Demonstrates a SECOND reader plugging into the FORMAT registry and yielding
# native 1-D arrays. Contract: numeric columns -> float64 arrays keyed by
# column (file_variable) name; non-numeric columns -> string arrays. Row
# filtering / variable remap are higher layers (ESS), not the reader.
# -----------------------------------------------------------------------------
def build_openaq_csv() -> tuple[bytes, dict, dict]:
    rows = [
        ("location_id", "datetime", "latitude", "longitude", "parameter", "value", "unit"),
        ("1", "2018-11-08T00:00:00Z", "39.76", "-121.62", "pm25", "152.3", "ug/m3"),
        ("1", "2018-11-08T01:00:00Z", "39.76", "-121.62", "pm25", "168.7", "ug/m3"),
        ("2", "2018-11-08T00:00:00Z", "39.50", "-121.50", "pm25", "98.1", "ug/m3"),
        ("2", "2018-11-08T01:00:00Z", "39.50", "-121.50", "pm25", "110.4", "ug/m3"),
    ]
    sio = io.StringIO()
    w = csv.writer(sio, lineterminator="\n")
    for r in rows:
        w.writerow(r)
    data = sio.getvalue().encode("utf-8")

    numeric = {"latitude", "longitude", "value"}
    header = rows[0]
    body = rows[1:]
    variables = {}
    for j, col in enumerate(header):
        vals = [r[j] for r in body]
        if col in numeric:
            variables[col] = {
                "dtype": "float64",
                "dims": ["index"],
                "shape": [len(body)],
                "fill_value": None,
                "data": [round(float(v), 10) for v in vals],
            }
        else:
            variables[col] = {
                "dtype": "string",
                "dims": ["index"],
                "shape": [len(body)],
                "fill_value": None,
                "data": list(vals),
            }
    expected = {"variables": variables, "coords": {}}
    decode = {"delimiter": ",", "header_row": 0, "numeric_columns": sorted(numeric)}
    return data, expected, decode


# -----------------------------------------------------------------------------
# Fixture 3 — FF10 point long-format slice (transport=file, format=ff10).
#
# A NEW reader (the generic CSV reader skips only empty lines, not '#' comments).
# Pins: the '#' header block is skipped; the fixed 77-column FF10_POINT schema is
# applied positionally (data rows carry no clean header row); the 42 numeric
# columns -> float64 (blank -> NaN), the other 35 ids/codes/free-text -> string
# (blank -> ""); RFC-4180 quoting so a FACILITY_NAME can embed a comma. It is
# READER-ONLY: no pollutant pivot (3 rows share one stack, differing only in
# POLID/ANN_VALUE), no unit conversion (STKHGT stays feet, STKTEMP °F), no
# FIPS/SCC normalization, no EGU filter — those move downstream into the .esm.
#
# Column names copied from Emissions.jl `src/ff10.jl` `FF10_POINT_COLUMNS`, with
# the SMOKE FF10_POINT spec names COUNTRY_CD / REGION_CD for the first two
# (Emissions.jl: COUNTRY / FIPS — identical values, positional alias).
# -----------------------------------------------------------------------------
FF10_POINT_COLUMNS = [
    "COUNTRY_CD", "REGION_CD", "TRIBAL_CODE", "FACILITY_ID",
    "UNIT_ID", "REL_POINT_ID", "PROCESS_ID", "AGY_FACILITY_ID",
    "AGY_UNIT_ID", "AGY_REL_POINT_ID", "AGY_PROCESS_ID", "SCC",
    "POLID", "ANN_VALUE", "ANN_PCT_RED", "FACILITY_NAME",
    "ERPTYPE", "STKHGT", "STKDIAM", "STKTEMP",
    "STKFLOW", "STKVEL", "NAICS", "LONGITUDE",
    "LATITUDE", "LL_DATUM", "HORIZ_COLL_MTHD", "DESIGN_CAPACITY",
    "DESIGN_CAPACITY_UNITS", "REG_CODES", "FAC_SOURCE_TYPE", "UNIT_TYPE_CODE",
    "CONTROL_IDS", "CONTROL_MEASURES", "CURRENT_COST", "CUMULATIVE_COST",
    "PROJECTION_FACTOR", "SUBMITTER_FAC_ID", "CALC_METHOD", "DATA_SET_ID",
    "FACIL_CATEGORY_CODE", "ORIS_FACILITY_CODE", "ORIS_BOILER_ID", "IPM_YN",
    "CALC_YEAR", "DATE_UPDATED", "FUG_HEIGHT", "FUG_WIDTH_XDIM",
    "FUG_LENGTH_YDIM", "FUG_ANGLE", "ZIPCODE", "ANNUAL_AVG_HOURS_PER_YEAR",
    "JAN_VALUE", "FEB_VALUE", "MAR_VALUE", "APR_VALUE",
    "MAY_VALUE", "JUN_VALUE", "JUL_VALUE", "AUG_VALUE",
    "SEP_VALUE", "OCT_VALUE", "NOV_VALUE", "DEC_VALUE",
    "JAN_PCTRED", "FEB_PCTRED", "MAR_PCTRED", "APR_PCTRED",
    "MAY_PCTRED", "JUN_PCTRED", "JUL_PCTRED", "AUG_PCTRED",
    "SEP_PCTRED", "OCT_PCTRED", "NOV_PCTRED", "DEC_PCTRED",
    "COMMENT",
]
FF10_POINT_NUMERIC = {
    "ANN_VALUE", "ANN_PCT_RED", "STKHGT", "STKDIAM", "STKTEMP", "STKFLOW",
    "STKVEL", "LONGITUDE", "LATITUDE", "DESIGN_CAPACITY", "CURRENT_COST",
    "CUMULATIVE_COST", "PROJECTION_FACTOR", "FUG_HEIGHT", "FUG_WIDTH_XDIM",
    "FUG_LENGTH_YDIM", "FUG_ANGLE", "ANNUAL_AVG_HOURS_PER_YEAR",
    "JAN_VALUE", "FEB_VALUE", "MAR_VALUE", "APR_VALUE", "MAY_VALUE", "JUN_VALUE",
    "JUL_VALUE", "AUG_VALUE", "SEP_VALUE", "OCT_VALUE", "NOV_VALUE", "DEC_VALUE",
    "JAN_PCTRED", "FEB_PCTRED", "MAR_PCTRED", "APR_PCTRED", "MAY_PCTRED",
    "JUN_PCTRED", "JUL_PCTRED", "AUG_PCTRED", "SEP_PCTRED", "OCT_PCTRED",
    "NOV_PCTRED", "DEC_PCTRED",
}


def build_ff10_point() -> tuple[bytes, dict, dict]:
    def row(**over) -> list:
        r = {c: "" for c in FF10_POINT_COLUMNS}
        r.update(over)
        return [r[c] for c in FF10_POINT_COLUMNS]

    # One stack (facility F001, unit U1, point R1, process P1) emitting THREE
    # pollutants (NOX/SO2/PM25) — identical stack params, distinct POLID/ANN_VALUE
    # (the reader must NOT pivot/aggregate). Row 1 has a quoted-comma FACILITY_NAME
    # and one non-blank monthly value; DESIGN_CAPACITY is blank (numeric -> NaN).
    stack = dict(
        COUNTRY_CD="US", REGION_CD="01001", FACILITY_ID="F001", UNIT_ID="U1",
        REL_POINT_ID="R1", PROCESS_ID="P1", SCC="0030700101",
        FACILITY_NAME="Autauga Plant, Unit 1", ERPTYPE="01",
        STKHGT="100.0", STKDIAM="5.0", STKTEMP="500.0", STKFLOW="25.0",
        STKVEL="12.5", NAICS="221112", LONGITUDE="-86.51045", LATITUDE="32.43878",
        LL_DATUM="NAD83", ZIPCODE="36066",
    )
    rows = [
        row(**stack, POLID="NOX", ANN_VALUE="123.45", JAN_VALUE="10.0",
            CALC_YEAR="2016", DATE_UPDATED="20130210"),
        row(**stack, POLID="SO2", ANN_VALUE="67.89"),
        row(**stack, POLID="PM25", ANN_VALUE="4.2"),
        # A second facility: plain (unquoted) FACILITY_NAME, ZIPCODE "00000".
        row(COUNTRY_CD="US", REGION_CD="01001", FACILITY_ID="F002", UNIT_ID="U9",
            REL_POINT_ID="R9", PROCESS_ID="P9", SCC="0030700201",
            POLID="NOX", ANN_VALUE="8.0", FACILITY_NAME="Prattville Facility",
            ERPTYPE="01", STKHGT="50.0", STKDIAM="2.5", STKTEMP="350.0",
            STKFLOW="10.0", STKVEL="6.0", NAICS="221112", LONGITUDE="-86.40000",
            LATITUDE="32.50000", LL_DATUM="NAD83", ZIPCODE="00000"),
    ]

    header_lines = ["#FORMAT=FF10_POINT", "#COUNTRY US", "#YEAR 2016",
                    "#DESC synthetic conformance fixture"]
    sio = io.StringIO()
    w = csv.writer(sio, lineterminator="\n")  # quotes the comma-bearing name
    for r in rows:
        w.writerow(r)
    data = ("".join(ln + "\n" for ln in header_lines) + sio.getvalue()).encode("utf-8")

    nrows = len(rows)
    variables = {}
    for j, col in enumerate(FF10_POINT_COLUMNS):
        vals = [r[j] for r in rows]
        if col in FF10_POINT_NUMERIC:
            variables[col] = {
                "dtype": "float64", "dims": ["index"], "shape": [nrows],
                "fill_value": None,
                "data": [None if str(v).strip() == "" else round(float(v), 10)
                         for v in vals],
            }
        else:
            variables[col] = {
                "dtype": "string", "dims": ["index"], "shape": [nrows],
                "fill_value": None, "data": [str(v) for v in vals],
            }
    expected = {"variables": variables, "coords": {}}
    decode = {"kind": "point", "member": None, "delimiter": ",", "comment": "#",
              "numeric_columns": sorted(FF10_POINT_NUMERIC)}
    return data, expected, decode


# -----------------------------------------------------------------------------
# Fixture 3b — FF10 point members inside a zip (transport=file, format=ff10).
#
# The EPA 2016fd zip shape: a `.zip` blob holding several FF10 member CSVs, each
# opening with a `#` comment block AND a `country_cd,region_cd,…` column-header
# line that is NOT a comment (77 fields, so it passes the arity check and would
# die at the numeric parse of ann_value if treated as data). Pins the two reader
# options that make the zip directly readable — no pre-extraction to temp files:
#
#   * `member_glob` — `*egu*` selects the TWO egu members (a third, non-matching
#     member is present and MUST be excluded); rows concatenate in ascending
#     lexicographic member-name order (alpha before beta);
#   * `skip_header_row` — drops exactly ONE asserted `country_cd` header line
#     per member (errors, never drops data, if the header is absent).
#
# Determinism: members are STORED (no compression, so no zlib variance), with a
# pinned DOS timestamp (1980-01-01), pinned create_system/external_attr, and no
# extra fields — regeneration is byte-identical on any Python.
# -----------------------------------------------------------------------------
def build_ff10_zip() -> tuple[bytes, dict, dict]:
    def row(**over) -> list:
        r = {c: "" for c in FF10_POINT_COLUMNS}
        r.update(over)
        return [r[c] for c in FF10_POINT_COLUMNS]

    def member_text(rows) -> str:
        sio = io.StringIO()
        w = csv.writer(sio, lineterminator="\n")
        for r in rows:
            w.writerow(r)
        header = ",".join(c.lower() for c in FF10_POINT_COLUMNS)
        return (
            "#FORMAT=FF10_POINT\n#COUNTRY US\n#YEAR 2016\n"
            + header + "\n" + sio.getvalue()
        )

    # Two `*egu*` members (mirrors the real 2016fd zip, which has exactly two)
    # + one member the glob must EXCLUDE. egu_alpha sorts before egu_beta.
    alpha_rows = [
        row(COUNTRY_CD="US", REGION_CD="01001", FACILITY_ID="F101", UNIT_ID="U1",
            REL_POINT_ID="R1", PROCESS_ID="P1", SCC="0030700101", POLID="NOX",
            ANN_VALUE="111.1", FACILITY_NAME="Alpha Station, Unit 1",
            STKHGT="100.0", STKTEMP="500.0", LONGITUDE="-86.51045",
            LATITUDE="32.43878", ZIPCODE="00000"),
        row(COUNTRY_CD="US", REGION_CD="01001", FACILITY_ID="F101", UNIT_ID="U1",
            REL_POINT_ID="R1", PROCESS_ID="P1", SCC="0030700101", POLID="SO2",
            ANN_VALUE="22.2", FACILITY_NAME="Alpha Station, Unit 1",
            STKHGT="100.0", STKTEMP="500.0", LONGITUDE="-86.51045",
            LATITUDE="32.43878", ZIPCODE="00000"),
    ]
    beta_rows = [
        row(COUNTRY_CD="US", REGION_CD="17031", FACILITY_ID="F202", UNIT_ID="U2",
            REL_POINT_ID="R2", PROCESS_ID="P2", SCC="0030700201", POLID="NOX",
            ANN_VALUE="333.3", FACILITY_NAME="Beta Plant",
            STKHGT="75.0", STKTEMP="400.0", LONGITUDE="-87.65000",
            LATITUDE="41.85000"),  # blank DESIGN_CAPACITY et al -> NaN
    ]
    other_rows = [
        row(COUNTRY_CD="US", REGION_CD="99999", FACILITY_ID="F999", POLID="NOX",
            ANN_VALUE="999.9", FACILITY_NAME="Excluded Nonpoint-ish"),
    ]
    members = {
        "point/egu_2016fd_alpha.csv": member_text(alpha_rows),
        "point/egu_2016fd_beta.csv": member_text(beta_rows),
        "point/ptnonipm_2016fd.csv": member_text(other_rows),
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        # A directory placeholder entry whose name MATCHES the glob (the real
        # 2016fd zip has `…/ptegu/`): selection must ignore it (file members
        # only), pinned by this fixture.
        zdir = zipfile.ZipInfo("point_egu/", date_time=(1980, 1, 1, 0, 0, 0))
        zdir.create_system = 3
        zdir.external_attr = (0o755 << 16) | 0x10  # dir mode + MS-DOS dir bit
        zdir.compress_type = zipfile.ZIP_STORED
        zf.writestr(zdir, b"")
        for name in sorted(members):
            zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            zi.create_system = 3            # pinned (else platform-dependent)
            zi.external_attr = 0o644 << 16  # pinned unix mode
            zi.compress_type = zipfile.ZIP_STORED  # no zlib version variance
            zf.writestr(zi, members[name])
    data = buf.getvalue()

    # Expected: `*egu*` selects alpha+beta (NOT ptnonipm); concatenation is in
    # sorted member-name order (alpha's 2 rows, then beta's 1); each member's
    # `country_cd` header line is skipped exactly once.
    sel_rows = alpha_rows + beta_rows
    nrows = len(sel_rows)
    variables = {}
    for j, col in enumerate(FF10_POINT_COLUMNS):
        vals = [r[j] for r in sel_rows]
        if col in FF10_POINT_NUMERIC:
            variables[col] = {
                "dtype": "float64", "dims": ["index"], "shape": [nrows],
                "fill_value": None,
                "data": [None if str(v).strip() == "" else round(float(v), 10)
                         for v in vals],
            }
        else:
            variables[col] = {
                "dtype": "string", "dims": ["index"], "shape": [nrows],
                "fill_value": None, "data": [str(v) for v in vals],
            }
    expected = {"variables": variables, "coords": {}}
    decode = {"kind": "point", "container": "zip", "member": None,
              "member_glob": "*egu*", "skip_header_row": True,
              "delimiter": ",", "comment": "#",
              "numeric_columns": sorted(FF10_POINT_NUMERIC)}
    return data, expected, decode


# -----------------------------------------------------------------------------
# Fixture 5 — synthetic ESRI shapefile in a zip (transport=file, format=shapefile).
#
# A shapefile is a FILE SET but the content-addressed cache holds ONE blob, so
# the fetchable form is a `.zip` of `.shp`/`.shx`/`.dbf`/`.prj`. The fixture pins
# every branch of the reader contract in one layer:
#
#   * ONE ROW PER PART — record 1 is a mainland + an island, so 4 live records
#     become 5 rows and its `.dbf` attributes are REPLICATED across both.
#   * PADDING — rings are 4 and 5 vertices, so the short ones are right-padded by
#     REPEATING their final vertex (esm-spec §8.6.1), never by NaN.
#   * DELETION — record 3's `.dbf` flag byte is `*`: the row AND its shape are
#     dropped. Record 4's flag byte is NUL, which is NOT deletion, so it is kept
#     (pyshp treats any non-space flag as deleted; the Python reader normalizes).
#   * DTYPES — `NAME` (C) -> string, `EMIS` (N) -> float64 with a blank -> NaN,
#     `FLAG` (L) -> bool, and `CODE` (C, a FIPS-shaped code) forced to float64 by
#     the `numeric_columns` reader option.
#   * The `.prj` WKT rides as the one-element `crs_wkt` field; `shape_type` names
#     the layer's geometry.
#
# The `.shp`/`.dbf` bytes are written with pyshp (the same third-party decoder
# the Python reader uses), then two bytes are patched to plant the deletion
# flags — there is no writer API for a deleted row — and the DBF's
# last-updated date is pinned to 1980-01-01 so the blob is byte-deterministic.
# -----------------------------------------------------------------------------

SHP_PRJ = (
    'GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",'
    'SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],'
    'UNIT["Degree",0.0174532925199433]]'
)

# (rings, NAME, EMIS, CODE, FLAG, deletion-flag byte)
SHP_RECORDS = [
    ([[(0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0), (0.0, 0.0)]],
     "Alpha", 1.5, "01001", True, b" "),
    ([[(4.0, 0.0), (4.0, 2.0), (6.0, 2.0), (6.0, 0.0), (4.0, 0.0)],
      [(7.0, 0.0), (7.0, 1.0), (8.0, 1.0), (7.0, 0.0)]],
     "Bravo", 2.25, "17031", False, b" "),
    ([[(0.0, 4.0), (0.0, 6.0), (2.0, 6.0), (0.0, 4.0)]],
     "Charlie", None, "06037", True, b" "),
    ([[(10.0, 10.0), (10.0, 11.0), (11.0, 11.0), (10.0, 10.0)]],
     "Deleted", 999.0, "99999", False, b"*"),
    ([[(0.0, 8.0), (0.0, 9.0), (1.0, 9.0), (1.0, 8.0), (0.0, 8.0)]],
     "Echo", -3.5, "36061", True, b"\x00"),
]


def _patch_dbf(raw: bytes) -> bytes:
    """Pin the DBF last-updated date to 1980-01-01 (byte-determinism — pyshp
    stamps *today*) and plant the fixture's per-record deletion flags. The date
    must stay a VALID one: Julia's DBFTables parses it into a `Date`, so a zeroed
    month is a hard error, not an ignored field."""
    buf = bytearray(raw)
    buf[1:4] = bytes((80, 1, 1))
    hdr = int.from_bytes(buf[8:10], "little")
    rec = int.from_bytes(buf[10:12], "little")
    for i, (_r, _n, _e, _c, _f, flag) in enumerate(SHP_RECORDS):
        buf[hdr + i * rec] = flag[0]
    return bytes(buf)


def build_shapefile_zip() -> tuple[bytes, dict, dict]:
    import shapefile as pyshp  # the `shapefile` extra; writer side only

    shp_io, shx_io, dbf_io = io.BytesIO(), io.BytesIO(), io.BytesIO()
    with pyshp.Writer(shp=shp_io, shx=shx_io, dbf=dbf_io) as w:
        w.field("NAME", "C", 12)
        w.field("EMIS", "N", 12, 4)
        w.field("CODE", "C", 5)
        w.field("FLAG", "L", 1)
        for rings, name, emis, code, flag, _del in SHP_RECORDS:
            w.poly(rings)
            w.record(name, emis, code, flag)
    members = {
        "layer/emis_polygons.shp": shp_io.getvalue(),
        "layer/emis_polygons.shx": shx_io.getvalue(),
        "layer/emis_polygons.dbf": _patch_dbf(dbf_io.getvalue()),
        "layer/emis_polygons.prj": SHP_PRJ.encode(),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in sorted(members):
            zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            zi.create_system = 3            # pinned (else platform-dependent)
            zi.external_attr = 0o644 << 16  # pinned unix mode
            zi.compress_type = zipfile.ZIP_STORED  # no zlib version variance
            zf.writestr(zi, members[name])
    data = buf.getvalue()

    # Expected arrays, by construction: explode the LIVE records to parts.
    rings, shape_ix, part_ix, nparts, boxes, rows = [], [], [], [], [], []
    for si, (parts, name, emis, code, flag, delflag) in enumerate(SHP_RECORDS):
        if delflag == b"*":
            continue
        xs = [p[0] for r in parts for p in r]
        ys = [p[1] for r in parts for p in r]
        for pi, ring in enumerate(parts):
            rings.append(ring)
            shape_ix.append(si)
            part_ix.append(pi)
            nparts.append(len(parts))
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
            rows.append((name, emis, code, flag))
    n = len(rings)
    nvert = max(len(r) for r in rings)
    geom = []
    for ring in rings:
        padded = list(ring) + [ring[-1]] * (nvert - len(ring))
        geom.extend([c for pt in padded for c in pt])

    def col(dtype, data):
        return {"dtype": dtype, "dims": ["index"], "shape": [n],
                "fill_value": None, "data": data}

    variables = {
        "geometry": {"dtype": "float64", "dims": ["index", "vertex", "xy"],
                     "shape": [n, nvert, 2], "fill_value": None, "data": geom},
        "shape_type": {"dtype": "string", "dims": ["meta"], "shape": [1],
                       "fill_value": None, "data": ["Polygon"]},
        "crs_wkt": {"dtype": "string", "dims": ["meta"], "shape": [1],
                    "fill_value": None, "data": [SHP_PRJ]},
        "n_vertices": col("int64", [len(r) for r in rings]),
        "shape_index": col("int64", shape_ix),
        "part_index": col("int64", part_ix),
        "n_parts": col("int64", nparts),
        "xmin": col("float64", [b[0] for b in boxes]),
        "ymin": col("float64", [b[1] for b in boxes]),
        "xmax": col("float64", [b[2] for b in boxes]),
        "ymax": col("float64", [b[3] for b in boxes]),
        "NAME": col("string", [r[0] for r in rows]),
        "EMIS": col("float64", [None if r[1] is None else float(r[1]) for r in rows]),
        "CODE": col("float64", [float(r[2]) for r in rows]),
        "FLAG": {"dtype": "bool", "dims": ["index"], "shape": [n],
                 "fill_value": None, "data": [bool(r[3]) for r in rows]},
    }
    expected = {"variables": variables, "coords": {}}
    decode = {"container": "zip", "member": "layer/emis_polygons.shp",
              "numeric_columns": ["CODE"]}
    return data, expected, decode

# -----------------------------------------------------------------------------
# Fixture 6 — georeferenced GeoTIFF raster (transport=file, format=geotiff).
#
# The shape the ESS raster loaders actually fetch: a single-band north-up
# geographic raster, the ArcGIS ImageServer `exportImage` / USGS 3DEP / LANDFIRE
# response. Closes the gap `spec/conformance.md` reserved and that
# tests/test_geotiff_reader.py names in its docstring — until now there was no
# committed binary GeoTIFF in the corpus, so the three tracks' geotiff readers
# had never been compared against each other on the same bytes.
#
# What it pins, deliberately, in one small blob:
#   * `Band1` — GDAL's 1-based band naming (the LANDFIRE loader's file_variable)
#   * lat/lon axis NAMES, from GTModelTypeGeoKey=2 (geographic). A projected
#     raster would be y/x; getting this branch wrong is silent and survives
#     every single-track test.
#   * CELL-CENTRE axes, not corners: the tiepoint anchors the top-left CORNER,
#     so every coordinate carries the +0.5-cell offset.
#   * NON-SQUARE cells (sx=0.5, sy=0.25). A reader that assumes square pixels,
#     or that reads ModelPixelScale[1] as the x scale, still passes a square
#     fixture — this one catches it.
#   * y-DOWN row order against y-UP model space: lat DECREASES with row index.
#   * GDAL_NODATA -> NaN (spec/conformance.md §3), on an interior cell so a
#     reader that drops an edge row cannot accidentally pass.
#   * int16 storage -> float64 output, the native-array dtype contract.
#
# SINGLE band on purpose. Multi-band is genuinely ambiguous across the three
# decoders — tifffile reads page 0 only, rasterio counts pages AND samples as
# bands, and the Rust reader derives a band count from a flat sample buffer, so
# a 2-band fixture would encode an interleaving convention nothing has agreed
# on yet. That is a real question, but it is not this case's question.
# -----------------------------------------------------------------------------
# Top-left CORNER of the raster in model space, and the cell size.
TIF_LON0, TIF_LAT0 = -121.5, 40.0
TIF_SX, TIF_SY = 0.5, 0.25
TIF_NODATA = -9999.0
TIF_URL = "https://landfire.gov/arcgis/rest/services/Landfire/US_230/ImageServer/exportImage?bbox=-121.5,39.25,-119.5,40.0&imageSR=4326&format=tiff"


def build_geotiff() -> tuple[bytes, dict, dict]:
    """A 3(lat) x 4(lon) single-band geographic GeoTIFF + its expected arrays."""
    import tifffile  # authoring side only; also the reader's pure-Python backend

    nlat, nlon = 3, 4
    # int16 counts (a LANDFIRE fuel-model code raster), with one interior NODATA.
    values = np.arange(nlat * nlon, dtype="int16").reshape(nlat, nlon) * 7 + 100
    values[1, 2] = int(TIF_NODATA)

    geokeys = (
        1, 1, 0, 3,          # KeyDirectoryVersion, Revision, MinorRevision, NumberOfKeys
        1024, 0, 1, 2,       # GTModelTypeGeoKey = 2 (geographic)
        1025, 0, 1, 1,       # GTRasterTypeGeoKey = 1 (PixelIsArea)
        2048, 0, 1, 4326,    # GeographicTypeGeoKey = WGS 84
    )
    # GDAL writes GDAL_NODATA as ASCII, integral for an integer band ("-9999",
    # not "-9999.0") — and tifffile parses the tag as an int, warning loudly on
    # anything else. Match real GDAL output so the committed blob is faithful and
    # decodes without a warning in any of the three tracks.
    sentinel = f"{int(TIF_NODATA)}\x00"
    extratags = [
        (33550, "d", 3, (TIF_SX, TIF_SY, 0.0)),                        # ModelPixelScale
        (33922, "d", 6, (0.0, 0.0, 0.0, TIF_LON0, TIF_LAT0, 0.0)),     # ModelTiepoint
        (34735, "H", len(geokeys), geokeys),                           # GeoKeyDirectory
        (42113, "s", len(sentinel), sentinel.encode()),                # GDAL_NODATA
    ]
    buf = io.BytesIO()
    # `software=False` and no datetime keep the bytes reproducible: a committed
    # corpus blob is compared by sha256, so a version string baked into the
    # header would churn the manifest on every tifffile upgrade.
    tifffile.imwrite(buf, values, extratags=extratags, software=False)
    data = buf.getvalue()

    # The expected native arrays, derived from the georef the same way the spec
    # says a reader must: centres, y-down, nodata -> NaN, float64.
    band = values.astype("float64")
    band[band == TIF_NODATA] = np.nan
    lon = TIF_LON0 + (np.arange(nlon, dtype="float64") + 0.5) * TIF_SX
    lat = TIF_LAT0 - (np.arange(nlat, dtype="float64") + 0.5) * TIF_SY

    def field(arr, dims):
        flat = np.asarray(arr, dtype="float64").ravel(order="C")
        return {
            "dtype": "float64",
            "dims": list(dims),
            "shape": list(np.shape(arr)),
            "fill_value": None,
            "data": [None if np.isnan(v) else float(v) for v in flat],
        }

    expected = {
        "variables": {"Band1": field(band, ("lat", "lon"))},
        "coords": {"lon": field(lon, ("lon",)), "lat": field(lat, ("lat",))},
    }
    return data, expected, {"nodata": TIF_NODATA, "band_names": None}


# -----------------------------------------------------------------------------
# Fixture 4 — synthetic Zarr v2 store (transport=s3, format=zarr, store=local).
#
# A tiny multi-chunk Zarr v2 store that pins the load-bearing capability: LAZY
# orthogonal selection on an arbitrary dimension driven by a runtime index list —
# fetch ONLY the chunk objects the selection intersects, never the whole array
# (the ISRM workflow depends on this). `field3d` is [2,5,4] chunked [1,2,4] so
# dim1 has a PARTIAL EDGE CHUNK (5 % 2 = 1, fill-padded), and `pop1d` is a 1-D
# single-chunk array. All arrays: blosc {cname lz4, clevel 5, shuffle 1}, order
# C, fill_value 0.0, zarr_format 2, dimension_separator null (-> "."). Each
# .zarray/.zattrs/chunk is its OWN object with its OWN URL, keyed by
# sha256(object_url) — so "lazy partial read" is just fetching a subset of small
# whole objects through the existing content-addressed cache.
#
# The committed blosc bytes are produced with the same c-blosc (numcodecs) the
# readers decode with, so they decode identically in all three tracks.
# -----------------------------------------------------------------------------
ZARR_BASE_URL = "s3://earthsci-fixtures/isrm-mini.zarr"

# The orthogonal selection the tile case exercises (a single selection applied to
# each array whose rank matches its axis count; other-rank arrays read whole):
#   field3d (ndim 3): layer=[1], y=[1,4], x=all -> only chunks {1}x{0,2}x{0}
#     = field3d/1.0.0 and field3d/1.2.0 (2 of 6). Skips ALL layer-0 chunks and
#     the middle y-chunk field3d/1.1.0 — the laziness contract, verified.
#   pop1d   (ndim 1): rank != 3 -> read whole.
ZARR_SELECT = {"axes": [{"indices": [1]}, {"indices": [1, 4]}, "all"]}


# --------------------------------------------------------------------------- #
# `parquet` — a MOVES-shaped rate table (the columnar cross-language case)
# --------------------------------------------------------------------------- #

#: `float_columns` for the parquet case: a rate stored as fixed-decimal TEXT
#: (how the MOVES snapshots keep floats byte-reproducible) and an integer
#: measurement whose missing cells must be NaN rather than a sentinel.
PQ_FLOAT_COLUMNS = ["meanBaseRate", "modelYearID"]
#: The two null gates. Declared, so an integer/string null is substituted
#: instead of refused — and `null_int` survives into `fill_value`.
PQ_NULL_INT = -1
PQ_NULL_STRING = "(unknown)"
#: 6 rows written as 2 row groups, so every track's decode CONCATENATES two
#: batches and the row order across the seam is part of what is pinned.
PQ_ROWS_PER_GROUP = 3
#: The columns a document asks for. `internalKey` (binary, no rank-1 reading)
#: and `ignoredNote` (a perfectly readable string column) are deliberately left
#: out: the projection must narrow the result to exactly this list.
PQ_PROJECTION = [
    "sourceTypeID", "roadTypeID", "linkID", "zoneID", "pollutantID",
    "fuelTypeDesc", "countyName", "isRamp", "meanBaseRate", "emissionQuant",
    "energyRate", "massFraction", "modelYearID", "startDate", "updateTime",
    "startTime", "microTime", "notApplicable",
]


def build_parquet_moves() -> tuple[bytes, dict, dict]:
    """A 6-row MOVES-shaped Parquet rate table + its expected native arrays.

    One column per supported Arrow family that all THREE backends can be asked
    for, so the case pins ``spec/conformance.md`` §3's "Parquet decode notes"
    table cross-language rather than per-track: the narrow/wide integer split,
    a categorical expanded, temporal columns carried as their RAW stored
    integer, a `Decimal` as unscaled ÷ 10^scale, an all-`Null` column as
    float64/all-NaN, the null policy and its two declared gates, and
    ``float_columns`` doing both of its jobs.

    Three shapes are deliberately ABSENT, each because a Parquet2.jl limit
    recorded in ``spec/conformance.md`` §3 makes it unenforceable across the
    three tracks rather than because the contract is unclear:

    * **no nested column** — Parquet2.jl cannot OPEN a file that carries one at
      all, so a shared fixture with a list/struct/map column would decode in
      two tracks and fail to open in the third. The "unrequested, it is simply
      not a field" rule stays normative and is exercised per-track
      (``rust/tests/parquet_reader.rs``, ``julia/test/test_parquet_reader.jl``,
      ``tests/test_parquet_reader.py``). A **binary** column IS here, because
      that one Parquet2.jl opens happily — so the unsupported-column rule is
      pinned cross-language for the case it can be;
    * **millisecond timestamps only** — Parquet2.jl decodes a timestamp to a
      `DateTime` before the reader sees it, so a MICROS/NANOS column's raw
      integer is not recoverable. `updateTime` is `timestamp[ms]`. (A
      `time64[us]` column IS here: `Dates.Time` is nanosecond-valued, so a
      sub-second time-of-day survives where a timestamp does not.)
    * **decimals inside `Dec64`'s exact range** — `massFraction` is
      `decimal128(18, 6)` with every |unscaled| < 2^53 and scale ≤ 22, where
      Julia's `Dec64 → Float64` and the other two tracks' `f64(unscaled) /
      10^scale` are bit-identical.

    Returns ``(blob_bytes, expected, decode)``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = 6
    # (name, arrow array, expected native dtype, expected native data). The
    # stored values and the values every track must return sit side by side, so
    # the fixture and its oracle cannot drift apart.
    cols: list[tuple[str, object, str, list]] = [
        # --- the narrow/wide integer split (the NetCDF reader's, verbatim) ---
        ("sourceTypeID", pa.array([11, 21, 31, 32, 52, 62], pa.int16()),
         "int32", [11, 21, 31, 32, 52, 62]),
        # uint8 is narrow -> int32; 255 proves it is read UNSIGNED.
        ("roadTypeID", pa.array([1, 2, 3, 4, 5, 255], pa.uint8()),
         "int32", [1, 2, 3, 4, 5, 255]),
        ("linkID", pa.array([4294967296, -4294967296, 3, 4, 5, 6], pa.int64()),
         "int64", [4294967296, -4294967296, 3, 4, 5, 6]),
        # uint32 is WIDE -> int64; 4294967295 does not fit an int32 and proves it.
        ("zoneID", pa.array([4294967295, 1, 2, 3, 4, 5], pa.uint32()),
         "int64", [4294967295, 1, 2, 3, 4, 5]),
        # --- the null policy: an integer null takes the declared sentinel, and
        # the sentinel is REPORTED BACK in fill_value ---
        ("pollutantID", pa.array([2, 3, None, 110, 100, 31], pa.int32()),
         "int32", [2, 3, PQ_NULL_INT, 110, 100, 31]),
        # --- a text null takes the declared substitute, and is then
        # indistinguishable from a real cell holding it. Row 3 holds a real
        # EMPTY string, which is NOT a null and must survive as "" ---
        ("fuelTypeDesc",
         pa.array(["Gasoline", None, "Diesel", "", "Electricity", "Gasoline"],
                  pa.string()),
         "string", ["Gasoline", PQ_NULL_STRING, "Diesel", "", "Electricity",
                    "Gasoline"]),
        # --- a categorical is EXPANDED to one value per row; the key encoding
        # is storage and never reaches the native array. Its null takes
        # null_string too (a null key and a key pointing at a null value are the
        # same null) ---
        ("countyName",
         pa.array(["Cook", "Cook", "DuPage", None, "Cook", "DuPage"]).dictionary_encode(),
         "string", ["Cook", "Cook", "DuPage", PQ_NULL_STRING, "Cook", "DuPage"]),
        ("isRamp", pa.array([False, True, False, False, True, True], pa.bool_()),
         "bool", [False, True, False, False, True, True]),
        # --- float_columns, job 2: floats stored as fixed-decimal TEXT (the
        # MOVES snapshots' `meanBaseRate`). Trimmed and parsed; an
        # all-whitespace cell is NaN (the FF10/shapefile blank->NaN rule).
        # Without the option this column stays `string` ---
        ("meanBaseRate",
         pa.array(["261.000000000000", "-1.500000000000", "0.000000000000",
                   "   ", "1e3", "0.062500000000"], pa.string()),
         "float64", [261.0, -1.5, 0.0, None, 1000.0, 0.0625]),
        # --- every float width is float64; 1e10 and 0.125 are exact in binary32
        # so the widening is bit-exact ---
        ("emissionQuant",
         pa.array([1.5, 2.5, -3.25, 0.0, 1.0e10, 0.125], pa.float32()),
         "float64", [1.5, 2.5, -3.25, 0.0, 1.0e10, 0.125]),
        # --- a null in a FLOAT column is NaN, and no sentinel survives ---
        ("energyRate",
         pa.array([3.14159, None, -0.5, 6.02214076e23, 0.0, 2.5], pa.float64()),
         "float64", [3.14159, None, -0.5, 6.02214076e23, 0.0, 2.5]),
        # --- Decimal128 -> float64 as unscaled / 10^scale. Max |unscaled| here
        # is 1e12 (< 2^53) and the scale is 6 (<= 22), which is exactly the
        # range where Julia's Dec64 and the other two tracks' division agree
        # bit for bit ---
        ("massFraction",
         pa.array(["0.123456", "-9.000000", "0.031250", "1000000.000000",
                   "0.500000", "-0.000123"]).cast(pa.decimal128(18, 6)),
         "float64", [0.123456, -9.0, 0.03125, 1000000.0, 0.5, -0.000123]),
        # --- float_columns, job 1: an INTEGER column that is really a
        # measurement. Its nulls become NaN, NOT the declared null_int, and no
        # fill_value survives — a float carries its missing values itself ---
        ("modelYearID", pa.array([1995, 2000, None, 2010, 2020, None], pa.int32()),
         "float64", [1995.0, 2000.0, None, 2010.0, 2020.0, None]),
        # --- temporal columns ride as their RAW stored integer at their stored
        # width, undecoded: the unit and any timezone are not applied and not
        # reported, because an epoch offset -> instant is ESS's job (R3) ---
        ("startDate", pa.array([19000, 19001, 0, -1, 19723, 12345], pa.date32()),
         "int32", [19000, 19001, 0, -1, 19723, 12345]),
        ("updateTime",
         pa.array([1700000000000, 0, -5, 1, 1234567890123, -86400000],
                  pa.timestamp("ms")),
         "int64", [1700000000000, 0, -5, 1, 1234567890123, -86400000]),
        ("startTime",
         pa.array([0, 3600000, 43200000, 86399999, 1000, 2000], pa.time32("ms")),
         "int32", [0, 3600000, 43200000, 86399999, 1000, 2000]),
        ("microTime",
         pa.array([0, 1, 999999, 86399999999, 123456, 7], pa.time64("us")),
         "int64", [0, 1, 999999, 86399999999, 123456, 7]),
        # --- an all-null column has no type of its own: float64, every cell NaN ---
        ("notApplicable", pa.array([None] * n, pa.null()),
         "float64", [None] * n),
        # --- NOT in the projection. A binary column has no rank-1 reading, so
        # unrequested it is simply not a field (never an error) ---
        ("internalKey",
         pa.array([b"\x00", b"\x01", b"\x02", b"\x03", b"\x04", b"\x05"],
                  pa.binary()),
         None, None),
        # --- NOT in the projection, and perfectly readable: the projection must
        # narrow the result, not just drop what it could not read ---
        ("ignoredNote", pa.array(["a", "b", "c", "d", "e", "f"], pa.string()),
         None, None),
    ]

    table = pa.table({name: arr for name, arr, _dt, _exp in cols})
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf, compression="snappy",
                   row_group_size=PQ_ROWS_PER_GROUP)
    data = buf.getvalue().to_pybytes()

    forced = set(PQ_FLOAT_COLUMNS)
    variables = {}
    for name, _arr, dtype, values in cols:
        if name not in PQ_PROJECTION:
            continue
        field = {"dtype": dtype, "dims": ["index"], "shape": [n], "data": values}
        # A NaN-folded float carries no surviving sentinel; a DECLARED integer
        # sentinel does, and every track reports it (Rust in the NativeField's
        # own `fill_value`, Python and Julia in `attrs["fill_value"]` — the same
        # datum in three spellings, normalised by the dumpers).
        if dtype in ("int32", "int64") and name not in forced:
            field["fill_value"] = PQ_NULL_INT
        variables[name] = field

    # A table produces NO coordinates (like the CSV and Zarr readers) — spelled
    # out as an empty map rather than omitted, so every case has the same shape.
    expected = {"variables": variables, "coords": {}}
    decode = {
        "float_columns": PQ_FLOAT_COLUMNS,
        "null_int": PQ_NULL_INT,
        "null_string": PQ_NULL_STRING,
        "row_groups": -(-n // PQ_ROWS_PER_GROUP),
        "rows_per_group": PQ_ROWS_PER_GROUP,
        "compression": "snappy",
        "projected_out": ["internalKey", "ignoredNote"],
        "no_nested_columns": True,
        "timestamps_ms_aligned": True,
        "decimal_scale": 6,
    }
    return data, expected, decode


def _blosc_codec():
    """The pinned numcodecs Blosc codec (matches the .zarray compressor)."""
    import numcodecs

    return numcodecs.Blosc(
        cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE, blocksize=0
    )


def _zarr_compressor():
    return {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0}


def _zarray_meta(shape, chunks, dtype):
    return {
        "zarr_format": 2,
        "shape": list(shape),
        "chunks": list(chunks),
        "dtype": dtype,
        "compressor": _zarr_compressor(),
        "fill_value": 0.0,
        "order": "C",
        "filters": None,
        "dimension_separator": None,
    }


def _zarr_chunks(shape, chunks, np_dtype, value_fn, fill=0.0):
    """Every chunk of an array as ``(chunk_key, encoded_bytes)``.

    Edge chunks are stored FULL-SIZE, fill-padded (the Zarr v2 contract) so the
    decompressed length is always ``prod(chunks)``. Bytes are C-order, blosc-lz4
    with the shuffle filter, exactly what the readers decode.
    """
    codec = _blosc_codec()
    ndim = len(shape)
    nchunks = [-(-shape[d] // chunks[d]) for d in range(ndim)]  # ceil-div
    out = []
    for cidx in itertools.product(*[range(n) for n in nchunks]):
        chunk = np.full(chunks, fill, dtype=np_dtype)
        for local in itertools.product(*[range(chunks[d]) for d in range(ndim)]):
            g = tuple(cidx[d] * chunks[d] + local[d] for d in range(ndim))
            if all(g[d] < shape[d] for d in range(ndim)):
                chunk[local] = value_fn(g)
        enc = bytes(codec.encode(np.ascontiguousarray(chunk)))
        out.append((".".join(str(c) for c in cidx), enc))
    return out


def build_zarr_store():
    """Return ``(objects, expected)`` for the synthetic Zarr v2 store.

    ``objects`` is a list of ``(relative_object_path, bytes)`` where the path is
    ``<array>/<name>`` (``.zarray``/``.zattrs``/``<chunk_key>``). ``expected`` is
    the sub-selected native arrays the tile case pins.
    """
    objects = []

    # field3d: [2,5,4] chunked [1,2,4], <f4. value = layer*100 + y*10 + x.
    f3_shape, f3_chunks = (2, 5, 4), (1, 2, 4)
    objects.append(("field3d/.zarray",
                    json.dumps(_zarray_meta(f3_shape, f3_chunks, "<f4"),
                               sort_keys=True).encode("utf-8")))
    objects.append(("field3d/.zattrs",
                    json.dumps({"_ARRAY_DIMENSIONS": ["layer", "y", "x"]},
                               sort_keys=True).encode("utf-8")))
    for key, enc in _zarr_chunks(f3_shape, f3_chunks, np.dtype("<f4"),
                                 lambda g: g[0] * 100 + g[1] * 10 + g[2]):
        objects.append((f"field3d/{key}", enc))

    # pop1d: [8] chunked [8], <f8. value = 2*i + 1.
    p_shape, p_chunks = (8,), (8,)
    objects.append(("pop1d/.zarray",
                    json.dumps(_zarray_meta(p_shape, p_chunks, "<f8"),
                               sort_keys=True).encode("utf-8")))
    objects.append(("pop1d/.zattrs",
                    json.dumps({"_ARRAY_DIMENSIONS": ["cell"]},
                               sort_keys=True).encode("utf-8")))
    for key, enc in _zarr_chunks(p_shape, p_chunks, np.dtype("<f8"),
                                 lambda g: 2 * g[0] + 1):
        objects.append((f"pop1d/{key}", enc))

    # Expected sub-selected arrays (float64; fill_value 0.0 is NOT mapped to NaN).
    expected = {
        "variables": {
            "field3d": {
                "dtype": "float64",
                "dims": ["layer", "y", "x"],
                "shape": [1, 2, 4],
                "fill_value": None,
                "data": [[[110.0, 111.0, 112.0, 113.0],
                          [140.0, 141.0, 142.0, 143.0]]],
            },
            "pop1d": {
                "dtype": "float64",
                "dims": ["cell"],
                "shape": [8],
                "fill_value": None,
                "data": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
            },
        },
        "coords": {},
    }
    return objects, expected


# -----------------------------------------------------------------------------
# Fixture 5 — synthetic Zarr v2 store pinning ORDERED lazy orthogonal selection.
#
# `sr` is [3,50,4] chunked [1,10,4]: dim1 (source) spans 5 chunks of width 10. The
# tile case selects layer=[0], source=[24,2,9,6] (0-based, NON-CONTIGUOUS AND
# PERMUTED — deliberately NOT sorted), receptor=all. This pins two load-bearing
# guarantees the ISRM workflow depends on, that all three tracks must reproduce
# BYTE-FOR-BYTE (the 3-way conformance acceptance gate):
#   * LAZINESS — source 24 -> chunk 2 and {2,9,6} -> chunk 0, so ONLY sr/0.0.0 and
#     sr/0.2.0 are fetched (2 of 15 chunks); chunk 1 (sources 10..19) and every
#     layer-1/2 chunk are never touched.
#   * ORDER PRESERVATION — the returned source axis is [24,2,9,6] in that EXACT
#     order; a reader that sorted the index list would return [2,6,9,24] and fail.
# value(layer,source,receptor) = layer*100000 + source*100 + receptor (exact in
# float32), so each cell self-encodes its indices and the permuted order is checkable.
# -----------------------------------------------------------------------------
ZARR_PERMUTED_URL = "s3://earthsci-fixtures/permuted-tile.zarr"
ZARR_PERMUTED_SELECT = {"axes": [{"indices": [0]}, {"indices": [24, 2, 9, 6]}, "all"]}


def build_zarr_permuted():
    """Return ``(objects, expected)`` for the ordered-selection Zarr v2 store."""
    objects = []
    shape, chunks = (3, 50, 4), (1, 10, 4)
    objects.append(("sr/.zarray",
                    json.dumps(_zarray_meta(shape, chunks, "<f4"),
                               sort_keys=True).encode("utf-8")))
    objects.append(("sr/.zattrs",
                    json.dumps({"_ARRAY_DIMENSIONS": ["layer", "source", "receptor"]},
                               sort_keys=True).encode("utf-8")))
    for key, enc in _zarr_chunks(shape, chunks, np.dtype("<f4"),
                                 lambda g: g[0] * 100000 + g[1] * 100 + g[2]):
        objects.append((f"sr/{key}", enc))

    # Expected: layer 0, sources [24,2,9,6] (permuted), all 4 receptors — rows in
    # the GIVEN order, nested C-order per shape [1,4,4]. value = 0*100000 + s*100 + r.
    def row(src):
        return [float(src * 100 + r) for r in range(4)]

    expected = {
        "variables": {
            "sr": {
                "dtype": "float64",
                "dims": ["layer", "source", "receptor"],
                "shape": [1, 4, 4],
                "fill_value": None,
                "data": [[row(24), row(2), row(9), row(6)]],
            },
        },
        "coords": {},
    }
    return objects, expected


def emit_zarr_case(case_id, *, loader, base_url, objects, variables, expected,
                   decode, select, notes):
    """Write every store object as a cache blob + manifest, then the case JSON.

    A Zarr case's ``blob_path``/``content_sha256``/``bytes``/``cache_key`` anchor
    on the primary array's ``.zarray`` object; the full per-object key/integrity
    table is the additive ``objects`` array (each ``{url, cache_key, blob_path,
    content_sha256, bytes}``), which the runner verifies per object.
    """
    obj_records = []
    primary = None
    for rel, data in objects:
        url = f"{base_url}/{rel}"
        key = cache_key(url)
        content_sha = sha256_bytes(data)
        blob_rel = write_blob(key, "", data)  # bare-key blob (found by <key> glob)
        manifest = {
            "schema": "earthsciio/manifest/v1",
            "url": url,
            "etag": None,
            "last_modified": None,
            "sha256_content": content_sha,
            "bytes": len(data),
            "fetched_at": FIXED_FETCHED_AT,
            "source_loader": loader,
            "auth_realm": None,
        }
        write_manifest(key, manifest)
        rec = {"url": url, "cache_key": key, "blob_path": blob_rel,
               "content_sha256": content_sha, "bytes": len(data)}
        obj_records.append(rec)
        if rel.endswith(f"{variables[0]}/.zarray"):
            primary = rec

    assert primary is not None, "primary .zarray object not found among store objects"
    case = {
        "schema": "earthsciio/cache-case/v1",
        "id": case_id,
        "loader": loader,
        "kind": "grid",
        "format": "zarr",
        "transport": "s3",
        "store": "local",
        "resolved_url": base_url,
        "cache_key": primary["cache_key"],
        "blob_path": primary["blob_path"],
        "manifest_path": meta_relpath(primary["cache_key"]),
        "content_sha256": primary["content_sha256"],
        "bytes": primary["bytes"],
        "variables": list(variables),
        "objects": obj_records,
        "select": select,
        "decode": decode,
        "expected": expected,
        "notes": notes,
    }
    write_json(CASES_DIR / f"{case_id}.json", case)
    return (primary["cache_key"], primary["content_sha256"], primary["bytes"],
            primary["blob_path"])


def emit_case(case_id, *, loader, kind, fmt, transport, store, resolved_url,
              ext, data, expected, decode, select, notes, variables=None):
    key = cache_key(resolved_url)
    content_sha = sha256_bytes(data)
    blob_rel = write_blob(key, ext, data)
    manifest = {
        "schema": "earthsciio/manifest/v1",
        "url": resolved_url,
        "etag": None,
        "last_modified": None,
        "sha256_content": content_sha,
        "bytes": len(data),
        "fetched_at": FIXED_FETCHED_AT,
        "source_loader": loader,
        "auth_realm": None,
    }
    meta_rel = write_manifest(key, manifest)
    case = {
        "schema": "earthsciio/cache-case/v1",
        "id": case_id,
        "loader": loader,
        "kind": kind,
        "format": fmt,
        "transport": transport,
        "store": store,
        "resolved_url": resolved_url,
        "cache_key": key,
        "blob_path": blob_rel,
        "manifest_path": meta_rel,
        "content_sha256": content_sha,
        "bytes": len(data),
        "select": select,
        "decode": decode,
        "expected": expected,
        "notes": notes,
    }
    # A single-blob case may still carry `variables`: the parquet reader takes
    # them as a PROJECTION pushed into the decode (only those column chunks come
    # off disk), the same field the store-backed zarr cases use to name arrays.
    if variables is not None:
        case["variables"] = list(variables)
    write_json(CASES_DIR / f"{case_id}.json", case)
    return key, content_sha, len(data), blob_rel


def main() -> None:
    summary = []

    nc_data, nc_expected, nc_decode = build_era5_netcdf()
    summary.append(("era5-grid-sub-tile",) + emit_case(
        "era5-grid-sub-tile",
        loader="era5", kind="grid", fmt="netcdf", transport="file", store="local",
        resolved_url="https://data.earthsci.dev/era5/2018/11/20181108.nc",
        ext="nc", data=nc_data, expected=nc_expected, decode=nc_decode,
        select={"all_records": True},
        notes=("ERA5-like 2x3x3 sub-tile. t2m is int16-packed (scale_factor/"
               "add_offset/_FillValue) -> decoded float64; one masked cell. sp "
               "is plain float64. Pins CF scale/offset/fill decode parity."),
    ))

    summary.append(("era5-window-slice",) + emit_case(
        "era5-window-slice",
        loader="era5", kind="grid", fmt="netcdf", transport="file", store="local",
        resolved_url="https://data.earthsci.dev/era5/2018/11/20181108.nc",
        ext="nc", data=nc_data,
        expected=window_era5_expected(nc_expected),
        decode=dict(nc_decode, decode_time_select=True),
        select=ERA5_WINDOW,
        notes=("The SAME blob and cache key as era5-grid-sub-tile, read through a "
               "decode-time orthogonal `select`: lat {slice:[1,3]}, lon "
               "{indices:[0,2]}, time \"all\". Pins that a whole-file reader "
               "materialises only the hyperslab and returns arrays identical to "
               "the full read sliced afterwards, that the lat/lon COORDINATES are "
               "sliced with the data (a windowed variable beside a full-length "
               "axis is the trap), that the masked cell still decodes to NaN "
               "inside the window, and that the time axis is untouched and raw. "
               "One blob, one sha256(url) key: a selection changes the decode, "
               "never the fetch."),
    ))

    csv_data, csv_expected, csv_decode = build_openaq_csv()
    summary.append(("openaq-points-slice",) + emit_case(
        "openaq-points-slice",
        loader="openaq", kind="points", fmt="csv", transport="file", store="local",
        resolved_url="https://openaq-data-archive.s3.amazonaws.com/records/openaq/locationid=1/2018-11-08.csv",
        ext="csv", data=csv_data, expected=csv_expected, decode=csv_decode,
        select={"all_rows": True},
        notes=("OpenAQ-like points CSV. Numeric columns -> float64 1-D arrays; "
               "others -> string arrays. Second reader behind the FORMAT "
               "registry; proves a non-NetCDF format plugs in unchanged."),
    ))

    ff10_data, ff10_expected, ff10_decode = build_ff10_point()
    summary.append(("ff10-point-slice",) + emit_case(
        "ff10-point-slice",
        loader="nei2016", kind="points", fmt="ff10", transport="file", store="local",
        resolved_url="https://gaftp.epa.gov/air/emismod/2016/v1/2016fd/point/ff10_point.csv",
        ext="csv", data=ff10_data, expected=ff10_expected, decode=ff10_decode,
        select={"all_rows": True},
        notes=("FF10 point long-format slice (NEI 2016). '#' header skipped; fixed "
               "77-col FF10_POINT schema applied positionally; 42 numeric cols -> "
               "float64 (blank->NaN), 35 ids/codes/text -> string; RFC-4180 quoted "
               "FACILITY_NAME with an embedded comma. 3 rows share one stack, "
               "differing only in POLID/ANN_VALUE (reader-only: no pivot/convert/"
               "filter). member=null decodes the bare extracted CSV member."),
    ))

    zf_data, zf_expected, zf_decode = build_ff10_zip()
    summary.append(("ff10-zip-egu-glob",) + emit_case(
        "ff10-zip-egu-glob",
        loader="nei2016", kind="points", fmt="ff10", transport="file", store="local",
        resolved_url="https://gaftp.epa.gov/air/emismod/2016/v1/2016fd/point/2016fd_inputs_point_mini.zip",
        ext="zip", data=zf_data, expected=zf_expected, decode=zf_decode,
        select={"all_rows": True},
        notes=("EPA-2016fd-shaped zip of FF10 point members: two `*egu*` members "
               "+ one excluded member + a glob-matching DIRECTORY placeholder "
               "entry (`point_egu/`, ignored — file members only), each member "
               "with a `#` comment block AND a non-comment `country_cd,…` "
               "column-header line (77 fields). Pins member_glob selection "
               "(glob `*egu*`, exclusion of the third member, sorted "
               "member-name concatenation: alpha rows then beta) and "
               "skip_header_row (exactly one asserted header line dropped per "
               "member). The blob is the WHOLE zip; member selection is reader "
               "config, never part of the cache key."),
    ))

    zarr_objects, zarr_expected = build_zarr_store()
    summary.append(("isrm-zarr-tile",) + emit_zarr_case(
        "isrm-zarr-tile",
        loader="isrm", base_url=ZARR_BASE_URL, objects=zarr_objects,
        variables=["field3d", "pop1d"], expected=zarr_expected,
        decode={"compressor": "blosc-lz4-shuffle", "fill_to_nan": False,
                "order": "C", "zarr_format": 2},
        select=ZARR_SELECT,
        notes=("Synthetic Zarr v2 store. field3d [2,5,4] chunked [1,2,4] (partial "
               "edge chunk on dim1); pop1d [8] chunked [8]. Orthogonal selection "
               "layer=[1], y=[1,4], x=all fetches ONLY field3d/1.0.0 + field3d/"
               "1.2.0 (2 of 6 chunks) — never layer 0, never the middle y-chunk "
               "field3d/1.1.0; pop1d (rank 1) reads whole. fill_value 0.0 is real "
               "data, NOT mapped to NaN. No coordinate arrays."),
    ))

    zp_objects, zp_expected = build_zarr_permuted()
    summary.append(("permuted-order-tile",) + emit_zarr_case(
        "permuted-order-tile",
        loader="isrm", base_url=ZARR_PERMUTED_URL, objects=zp_objects,
        variables=["sr"], expected=zp_expected,
        decode={"compressor": "blosc-lz4-shuffle", "fill_to_nan": False,
                "order": "C", "zarr_format": 2},
        select=ZARR_PERMUTED_SELECT,
        notes=("Synthetic Zarr v2 store pinning ORDERED lazy orthogonal selection. "
               "sr [3,50,4] chunked [1,10,4]; select layer=[0], source=[24,2,9,6] "
               "(NON-CONTIGUOUS, PERMUTED — not sorted), receptor=all fetches ONLY "
               "sr/0.0.0 + sr/0.2.0 (2 of 15 chunks) and returns the source axis in "
               "[24,2,9,6] order EXACTLY (a reader that sorted the indices would "
               "return [2,6,9,24] and fail). Proven byte-identical across "
               "Python/Julia/Rust — the Phase-1 3-way selection conformance gate."),
    ))

    shp_data, shp_expected, shp_decode = build_shapefile_zip()
    summary.append(("shapefile-polygon-zip",) + emit_case(
        "shapefile-polygon-zip",
        loader="emis_polygons", kind="points", fmt="shapefile", transport="file",
        store="local",
        resolved_url="https://data.earthsci.dev/fixtures/emis_polygons.zip",
        ext="zip", data=shp_data, expected=shp_expected, decode=shp_decode,
        select={"all_records": True},
        notes=("ESRI shapefile (Polygon) zipped with its .shx/.dbf/.prj sidecars. "
               "Pins the whole reader contract in one layer: ONE ROW PER PART "
               "(record 1 is a mainland + an island, so 4 live records decode to "
               "5 rows with the .dbf attributes replicated); the esm-spec 8.6.1 "
               "padding (a 4-vertex ring in a 5-vertex stack repeats its FINAL "
               "vertex, never NaN); a `*` deletion flag dropping the row AND its "
               "shape while a NUL flag byte keeps it (pyshp treats any non-space "
               "flag as deleted — the Python reader normalizes); the stored "
               "per-record bbox replicated to parts; and the dtype rules — C -> "
               "string, N -> float64 with a blank -> NaN, L -> bool, and a C code "
               "column forced to float64 by numeric_columns. shape_type/crs_wkt "
               "are one-element `meta` string fields, not field attrs, because "
               "the Rust NativeField has none and equality compares fields."),
    ))

    pq_data, pq_expected, pq_decode = build_parquet_moves()
    summary.append(("moves-rate-table-parquet",) + emit_case(
        "moves-rate-table-parquet",
        loader="moves", kind="points", fmt="parquet", transport="file",
        store="local",
        resolved_url="https://data.earthsci.dev/moves/2024/MOVESExecution/emissionrate.parquet",
        ext="parquet", data=pq_data, expected=pq_expected, decode=pq_decode,
        select={"all_rows": True}, variables=PQ_PROJECTION,
        notes=("MOVES-shaped Parquet rate table, 6 rows in 2 ROW GROUPS (so every "
               "track concatenates two batches and the row order across the seam "
               "is pinned), snappy-compressed. One column per supported Arrow "
               "family: the narrow/wide integer split (uint8 255 read unsigned, "
               "uint32 4294967295 not fitting an int32), a categorical EXPANDED "
               "to one string per row, float32 widened bit-exactly, Decimal128 "
               "as unscaled/10^scale, temporal columns as their RAW stored "
               "integer at their stored width (date32/time32[ms] -> int32; "
               "timestamp[ms]/time64[us] -> int64), and an all-Null column as "
               "float64/all-NaN. Null policy: a float null -> NaN with no "
               "surviving sentinel; a declared null_int=-1 fills the integer "
               "nulls AND is reported back in fill_value on every int column; "
               "null_string fills a text null (and a DICTIONARY null) while a "
               "real empty string stays \"\". float_columns does both of its "
               "jobs at once — `meanBaseRate` is fixed-decimal TEXT (blank -> "
               "NaN) and `modelYearID` is an integer measurement whose nulls "
               "become NaN rather than the sentinel, so no fill_value survives "
               "it. `variables` is the PROJECTION: `ignoredNote` (readable) and "
               "`internalKey` (binary, no rank-1 reading) are both left out. "
               "Deliberately carries NO nested column, ms-aligned timestamps "
               "only, and decimals inside Dec64's exact range — the three "
               "Parquet2.jl limits spec/conformance.md §3 records."),
    ))

    tif_data, tif_expected, tif_decode = build_geotiff()
    summary.append(("landfire-raster-tile",) + emit_case(
        "landfire-raster-tile",
        loader="landfire", kind="grid", fmt="geotiff", transport="file",
        store="local",
        resolved_url=TIF_URL,
        ext="tif", data=tif_data, expected=tif_expected, decode=tif_decode,
        select={"all_cells": True},
        notes=("Single-band north-up GEOGRAPHIC GeoTIFF, 3(lat) x 4(lon), the "
               "shape the LANDFIRE / USGS 3DEP ImageServer `exportImage` "
               "responses have. Pins the georef contract the three readers each "
               "implement separately and had never been compared on: Band1 "
               "naming; lat/lon axis names from GTModelTypeGeoKey=2 (a projected "
               "raster would be y/x); CELL-CENTRE axes offset half a cell from "
               "the tiepoint's top-left CORNER; NON-SQUARE cells (sx=0.5, "
               "sy=0.25) so a reader assuming square pixels fails; lat "
               "DECREASING with row index (y-up model space, y-down rows); "
               "GDAL_NODATA -> NaN on an INTERIOR cell; and int16 storage "
               "widened to float64. Single-band on purpose — multi-band "
               "interleaving is not agreed across tifffile/rasterio/the Rust "
               "tiff crate, so pinning it here would invent a convention."),
    ))

    index = {
        "schema": "earthsciio/cases-index/v1",
        "cache_format_version": CACHE_FORMAT_VERSION,
        "cache_root": f"cache/{CACHE_FORMAT_VERSION}",
        "cases": [
            {"id": cid, "file": f"cases/{cid}.json", "cache_key": key,
             "blob_path": blob_rel}
            for (cid, key, _sha, _n, blob_rel) in summary
        ],
    }
    write_json(CORPUS / "cases.json", index)

    print(f"cache format: {CACHE_FORMAT_VERSION}   fetched_at(pinned): {FIXED_FETCHED_AT}")
    for cid, key, sha, nbytes, blob_rel in summary:
        print(f"  {cid:24s} key={key}  content_sha256={sha[:16]}…  {nbytes:>6d} B")
        print(f"  {'':24s} blob={blob_rel}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reference conformance runner for the EarthSciIO corpus (the oracle).

This is the executable definition of "passes conformance", in Python, run
fully OFFLINE against the committed corpus. Every language track's harness
(esio-9nb.9) performs the same five checks and must reach the same verdict:

  1. cache-key agreement     sha256(resolved_url) == case.cache_key
  2. manifest integrity      sha256(blob) == manifest.sha256_content == case.content_sha256
                             and len(blob) == manifest.bytes == case.bytes
  3. format/reader decode    open the blob with the format's reader, CF-decode
  4. native-array equality   decoded arrays == case.expected (tolerance below)
  5. offline-only            no network access (this runner only reads files)

Tolerances: raw/unpacked numeric reads compare exactly; CF-decoded (packed)
values and unit-affected reads compare within ATOL/RTOL (libraries differ at
ULP level — xarray vs NCDatasets vs netcdf-rs). String arrays compare exactly.
``null`` in expected.data maps to NaN (numeric) / missing (string).

Usage:  python3 conformance/verify.py        # verifies every case, exit 1 on failure
Spec:   ../spec/conformance.md
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import pathlib
import sys

import numpy as np

ATOL = 1e-6
RTOL = 1e-9

HERE = pathlib.Path(__file__).resolve().parent
CORPUS = HERE / "corpus"


def _flat(x):
    if isinstance(x, list):
        for e in x:
            yield from _flat(e)
    else:
        yield x


def _cmp_numeric(got: np.ndarray, expected_nested, label: str, errs: list):
    exp = np.array(
        [math.nan if v is None else float(v) for v in _flat(expected_nested)],
        dtype="f8",
    )
    g = np.asarray(got, dtype="f8").reshape(-1)
    if g.shape != exp.shape:
        errs.append(f"{label}: shape {g.shape} != expected {exp.shape}")
        return
    gn, en = np.isnan(g), np.isnan(exp)
    if not np.array_equal(gn, en):
        errs.append(f"{label}: NaN/fill mask mismatch")
        return
    ok = np.allclose(g[~gn], exp[~en], atol=ATOL, rtol=RTOL)
    if not ok:
        d = np.max(np.abs(g[~gn] - exp[~en])) if np.any(~gn) else 0.0
        errs.append(f"{label}: value mismatch (max abs diff {d:g} > atol {ATOL:g})")


def _cmp_string(got, expected_nested, label: str, errs: list):
    g = [str(v) for v in _flat(got)]
    e = [None if v is None else str(v) for v in _flat(expected_nested)]
    if g != e:
        errs.append(f"{label}: string mismatch {g} != {e}")


def read_netcdf(path, expected, select=None):
    """CF-decode via xarray: scale/offset + fill->NaN; time NOT decoded.

    A case carrying an orthogonal ``select`` (``{"axes": [...]}``) is the
    decode-time window: the oracle reads the array WHOLE and slices it
    afterwards, deliberately, because "the full read, sliced" is exactly the
    contract a windowed reader has to reproduce. The axes are positional over the
    file-order dims of the arrays whose rank matches, which induces the
    dimension -> indices map applied here to every array AND every coordinate.
    """
    import xarray as xr

    axes_spec = (select or {}).get("axes")
    out = {}
    with xr.open_dataset(path, decode_times=False, mask_and_scale=True) as ds:
        take = {}
        if axes_spec is not None:
            for _, da in ds.variables.items():
                if len(da.dims) != len(axes_spec):
                    continue
                for spec, dim in zip(axes_spec, da.dims):
                    take[str(dim)] = _zarr_resolve_axis(spec, int(ds.sizes[dim]))

        def gather(da):
            idx = {d: take[d] for d in map(str, da.dims) if d in take}
            values = da.values
            for axis, dim in enumerate(map(str, da.dims)):
                if dim in idx:
                    values = np.take(values, idx[dim], axis=axis)
            return values

        for name in expected["variables"]:
            out[name] = gather(ds[name])
        coords = {}
        for name in expected.get("coords", {}):
            coords[name] = gather(ds[name])
    return out, coords


def read_csv(path, expected):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    cols = {h: [r[j] for r in body] for j, h in enumerate(header)}
    out = {}
    for name, spec in expected["variables"].items():
        vals = cols[name]
        if spec["dtype"] == "string":
            out[name] = vals
        else:
            out[name] = np.array([float(v) for v in vals], dtype="f8")
    return out, {}


# FF10 point column schema — copied from Emissions.jl `src/ff10.jl`
# `FF10_POINT_COLUMNS` (SMOKE names COUNTRY_CD/REGION_CD for the first two).
_FF10_POINT_COLUMNS = [
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


def read_ff10(path, expected, decode=None):
    """FF10 point decode oracle: resolve the case's zip member selection (if
    any), skip '#'/blank lines, drop the asserted ``country_cd`` header line per
    member (``skip_header_row``), RFC-4180 split, assign the 77-column schema
    positionally, then type each column per the case's expected dtype (blank ->
    NaN in a float64 column, str otherwise). Member-selection semantics mirror
    the readers: ``members`` (explicit list) union ``member_glob`` (fnmatch,
    case-sensitive) matches, read in ascending lexicographic member-name order;
    an absent explicit member or a zero-match glob is an error; directory
    placeholder entries (names ending in ``/``) are never selected."""
    import fnmatch
    import zipfile

    decode = decode or {}
    member = decode.get("member")
    members = decode.get("members")
    member_glob = decode.get("member_glob")
    skip_header = bool(decode.get("skip_header_row", False))

    if members is not None or member_glob is not None:
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            selected = set()
            if members is not None:
                missing = [m for m in members if m not in names]
                if missing:
                    raise ValueError(f"zip members {missing!r} not in archive")
                selected.update(members)
            if member_glob is not None:
                hits = [n for n in names if fnmatch.fnmatchcase(n, member_glob)]
                if not hits:
                    raise ValueError(f"member_glob {member_glob!r} matched no members")
                selected.update(hits)
            texts = [zf.read(n).decode("utf-8") for n in sorted(selected)]
    elif member is not None:
        with zipfile.ZipFile(path) as zf:
            texts = [zf.read(member).decode("utf-8")]
    else:
        with open(path, newline="") as fh:
            texts = [fh.read()]

    index = {name: j for j, name in enumerate(_FF10_POINT_COLUMNS)}
    rows = []
    for text in texts:
        lines = [
            ln for ln in text.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        if skip_header:
            if not lines or lines[0].split(",", 1)[0].strip().lower() != "country_cd":
                raise ValueError(
                    "skip_header_row: asserted 'country_cd' header line missing"
                )
            lines = lines[1:]
        rows.extend(csv.reader(lines))
    ncol = len(_FF10_POINT_COLUMNS)
    for r in rows:
        if len(r) != ncol:
            raise ValueError(f"FF10 row has {len(r)} fields, expected {ncol}: {r!r}")
    out = {}
    for name, spec in expected["variables"].items():
        vals = [r[index[name]] for r in rows]
        if spec["dtype"] == "string":
            out[name] = vals
        else:
            out[name] = np.array(
                [math.nan if v.strip() == "" else float(v) for v in vals],
                dtype="f8",
            )
    return out, {}


def _numcodecs_codec(name):
    """The named ``numcodecs`` codec, or an actionable error naming the extra.

    The oracle decodes zarr chunks with numcodecs *on purpose* — independently of
    the production reader's zarr-python stack (spec/conformance.md). It ships in
    the ``zarr`` extra; without it the corpus self-check cannot run at all, so say
    so rather than surfacing a bare ``ModuleNotFoundError`` from six frames down.
    """
    try:
        import numcodecs
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            f"conformance: the corpus's zarr cases need the '{name}' codec from "
            "numcodecs, which is not installed. Install the extras every corpus "
            'format needs: pip install -e ".[netcdf,shapefile,zarr,test]" '
            "(the zarr extra requires Python >=3.11)."
        ) from exc
    return getattr(numcodecs, name)


def _zarr_decompress(compressor, raw):
    """Independent decode of one chunk object's bytes (the zarr oracle codec)."""
    if compressor is None:
        return bytes(raw)
    cid = str(compressor.get("id", "")).lower()
    if cid == "blosc":
        return bytes(_numcodecs_codec("Blosc")().decode(raw))
    if cid == "zlib":
        return bytes(_numcodecs_codec("Zlib")().decode(raw))
    if cid == "zstd":
        return bytes(_numcodecs_codec("Zstd")().decode(raw))
    if cid in ("", "none"):
        return bytes(raw)
    raise ValueError(f"unsupported zarr compressor id {cid!r}")


def _zarr_resolve_axis(spec, dim_len):
    """Resolve one axis selector (from ``select.axes``) to a global index list."""
    if spec is None or spec == "all":
        return list(range(dim_len))
    if isinstance(spec, dict) and "indices" in spec:
        return [int(i) for i in spec["indices"]]
    if isinstance(spec, dict) and "slice" in spec:
        s = spec["slice"]
        step = int(s[2]) if len(s) > 2 else 1
        return list(range(int(s[0]), int(s[1]), step))
    if isinstance(spec, (list, tuple)):
        return [int(i) for i in spec]
    raise ValueError(f"unrecognized axis selector: {spec!r}")


def read_zarr(corpus, case):
    """Reference (oracle) Zarr v2 reader: reconstruct each array from its committed
    chunk objects and apply the case's orthogonal selection.

    Independent of the production reader's chunk math — it rebuilds the FULL array
    from every chunk object, then gathers ``full[np.ix_(*sel)]`` — so agreement is
    a real cross-check, not a tautology. Runs offline against the committed blobs.
    """
    objmap = {o["url"]: o for o in case.get("objects", [])}
    base = case["resolved_url"]
    axes_spec = (case.get("select") or {}).get("axes")

    def obj_bytes(url):
        o = objmap.get(url)
        return None if o is None else (corpus / o["blob_path"]).read_bytes()

    out = {}
    for array in case["variables"]:
        zmeta = json.loads(obj_bytes(f"{base}/{array}/.zarray").decode("utf-8"))
        shape = [int(s) for s in zmeta["shape"]]
        chunks = [int(c) for c in zmeta["chunks"]]
        dt = np.dtype(zmeta["dtype"])
        order = zmeta.get("order", "C")
        sep = "." if zmeta.get("dimension_separator") in (None, "") else zmeta["dimension_separator"]
        ndim = len(shape)
        out_dt = np.dtype("float64") if dt.kind == "f" else dt

        if axes_spec is not None and len(axes_spec) == ndim:
            sel = [_zarr_resolve_axis(axes_spec[d], shape[d]) for d in range(ndim)]
        else:
            sel = [list(range(shape[d])) for d in range(ndim)]

        fill = zmeta.get("fill_value", 0.0) or 0.0
        full = np.full(shape, fill, dtype=out_dt)
        nchunks = [-(-shape[d] // chunks[d]) for d in range(ndim)]
        for cidx in itertools.product(*[range(n) for n in nchunks]):
            raw = obj_bytes(f"{base}/{array}/" + sep.join(str(c) for c in cidx))
            if raw is None:
                continue  # absent chunk object → keep fill
            carr = np.frombuffer(_zarr_decompress(zmeta.get("compressor"), raw),
                                 dtype=dt).reshape(chunks, order=order)
            reg = tuple(slice(cidx[d] * chunks[d],
                              min(cidx[d] * chunks[d] + chunks[d], shape[d]))
                        for d in range(ndim))
            loc = tuple(slice(0, r.stop - r.start) for r in reg)
            full[reg] = carr[loc].astype(out_dt, copy=False)
        out[array] = full[np.ix_(*sel)]
    return out, {}


def read_shapefile(path, expected, decode=None):
    """Shapefile decode oracle: the reader contract stated independently.

    Extracts the case's `.shp` member (+ `.shx`/`.dbf`/`.prj` sidecars) from the
    zip blob, decodes with pyshp, and applies the pinned rules: a `.dbf` row is
    deleted iff its flag byte is ``*`` (pyshp calls any non-space flag deleted,
    so the flags are read here and normalized); each record's PARTS become their
    own rows with the attributes replicated; rings are right-padded to the
    longest by REPEATING the final vertex; the record's STORED bbox is
    replicated to its parts; `N`/`F` -> float64 (blank -> NaN), `L` -> bool, the
    rest -> str, with ``numeric_columns`` forcing a text column to float64.
    """
    import io
    import shapefile as pyshp
    import zipfile

    decode = decode or {}
    member = decode.get("member")
    numeric = set(decode.get("numeric_columns") or [])
    blobs = {}
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        target = member or sorted(n for n in names if n.lower().endswith(".shp"))[0]
        stem = target[: -len(".shp")].lower()
        for n in names:
            key = "shp" if n == target else (
                n.lower()[len(stem) + 1:] if n.lower().startswith(stem + ".") else None)
            if key in ("shp", "dbf", "shx", "prj"):
                blobs[key] = zf.read(n)

    raw = bytearray(blobs["dbf"])
    hdr = int.from_bytes(raw[8:10], "little")
    rlen = int.from_bytes(raw[10:12], "little")
    nrec = int.from_bytes(raw[4:8], "little")
    deleted = []
    for i in range(nrec):
        off = hdr + i * rlen
        deleted.append(raw[off] == 0x2A)
        if not deleted[-1]:
            raw[off] = 0x20

    kw = {"shp": io.BytesIO(blobs["shp"]), "dbf": io.BytesIO(bytes(raw))}
    if "shx" in blobs:
        kw["shx"] = io.BytesIO(blobs["shx"])
    with pyshp.Reader(**kw) as rdr:
        shapes = list(rdr.iterShapes())
        columns = [f[0] for f in rdr.fields if f[0] != "DeletionFlag"]
        rows = [list(r) for r in rdr.iterRecords()]
        stype = int(rdr.shapeType)

    rings, meta, attrs = [], [], []
    live = 0
    for si, shp in enumerate(shapes):
        if deleted[si]:
            continue
        pts = list(shp.points)
        offs = [int(o) for o in (shp.parts or [])] or [0]
        bounds = offs + [len(pts)]
        parts = [pts[bounds[k]:bounds[k + 1]] for k in range(len(offs))]
        bb = list(shp.bbox)
        for pi, ring in enumerate(parts):
            rings.append(ring)
            meta.append((si, pi, len(parts), bb))
            attrs.append(rows[live])
        live += 1

    nvert = max(len(r) for r in rings)
    geom = np.empty((len(rings), nvert, 2), dtype="f8")
    for i, ring in enumerate(rings):
        padded = list(ring) + [ring[-1]] * (nvert - len(ring))
        geom[i] = np.asarray(padded, dtype="f8")

    out = {
        "geometry": geom,
        "shape_type": [{5: "Polygon", 3: "PolyLine", 1: "Point"}.get(stype, str(stype))],
        "n_vertices": np.array([len(r) for r in rings], dtype="i8"),
        "shape_index": np.array([m[0] for m in meta], dtype="i8"),
        "part_index": np.array([m[1] for m in meta], dtype="i8"),
        "n_parts": np.array([m[2] for m in meta], dtype="i8"),
        "xmin": np.array([m[3][0] for m in meta], dtype="f8"),
        "ymin": np.array([m[3][1] for m in meta], dtype="f8"),
        "xmax": np.array([m[3][2] for m in meta], dtype="f8"),
        "ymax": np.array([m[3][3] for m in meta], dtype="f8"),
    }
    if "prj" in blobs:
        out["crs_wkt"] = [blobs["prj"].decode("utf-8").strip()]
    for j, name in enumerate(columns):
        vals = [a[j] for a in attrs]
        spec = expected["variables"].get(name)
        dt = spec["dtype"] if spec else "string"
        if name in numeric or dt == "float64":
            out[name] = np.array(
                [math.nan if v is None or str(v).strip() == "" else float(v) for v in vals],
                dtype="f8")
        elif dt == "bool":
            out[name] = np.array([bool(v) for v in vals], dtype=bool)
        else:
            out[name] = [str(v).strip() for v in vals]
    return out, {}


def read_parquet(path, expected, decode=None):
    """Reference decode of a Parquet blob — ``spec/conformance.md`` §3.

    Driven by **pyarrow directly**, not through ``earthsciio``: the oracle has to
    be an independent expression of the contract, or it only proves the reader
    agrees with itself. The projection is the expected variable set (which is
    the case's ``variables``), and the target dtype is derived here from the
    Arrow type + ``float_columns`` rather than read off the expected field.
    """
    import pyarrow.parquet as pq
    import pyarrow.types as pt

    dec = decode or {}
    forced = set(dec.get("float_columns") or [])
    null_int = dec.get("null_int")
    null_string = dec.get("null_string")

    wanted = list(expected["variables"])
    table = pq.read_table(path, columns=wanted)

    out = {}
    for name in wanted:
        col = table.column(name)
        typ = col.type
        if pt.is_dictionary(typ):          # a categorical reads as its VALUE type
            col = col.cast(typ.value_type)
            typ = col.type
        if name in forced:
            kind = "float64"
        elif pt.is_boolean(typ):
            kind = "bool"
        elif pt.is_string(typ) or pt.is_large_string(typ):
            kind = "string"
        elif (pt.is_int8(typ) or pt.is_int16(typ) or pt.is_int32(typ)
              or pt.is_uint8(typ) or pt.is_uint16(typ) or pt.is_date32(typ)
              or pt.is_time32(typ)):
            kind = "int32"
        elif (pt.is_int64(typ) or pt.is_uint32(typ) or pt.is_uint64(typ)
              or pt.is_date64(typ) or pt.is_time64(typ) or pt.is_timestamp(typ)
              or pt.is_duration(typ)):
            kind = "int64"
        else:                              # floats, decimals, and the Null type
            kind = "float64"

        # Temporal columns ride as their RAW stored integer (the unit is NOT
        # applied) — casting is what keeps it; to_pylist alone yields datetimes.
        if kind in ("int32", "int64") and not (
            pt.is_integer(typ) or pt.is_null(typ)
        ):
            import pyarrow as pa
            col = col.cast(pa.int32() if kind == "int32" else pa.int64())
        cells = col.to_pylist()

        if kind == "float64":
            vals = []
            for c in cells:
                if c is None:
                    vals.append(math.nan)
                elif isinstance(c, str):   # fixed-decimal TEXT under float_columns
                    vals.append(math.nan if not c.strip() else float(c.strip()))
                else:
                    vals.append(float(c))  # Decimal -> float64 is unscaled/10^scale
            out[name] = np.array(vals, dtype="f8")
        elif kind in ("int32", "int64"):
            out[name] = np.array(
                [null_int if c is None else int(c) for c in cells], dtype=kind)
        elif kind == "bool":
            out[name] = np.array([bool(c) for c in cells], dtype=bool)
        else:
            out[name] = [null_string if c is None else str(c) for c in cells]
    return out, {}
def read_geotiff(path, expected):
    """Independent GeoTIFF decode: tifffile for the container, tags parsed here.

    Deliberately NOT the production reader's path. `earthsciio`'s geotiff reader
    prefers rasterio/GDAL and only falls back to tifffile, so decoding here with
    tifffile and deriving the axes from the raw IFD tags keeps the oracle an
    independent implementation rather than a second call into the same code —
    the same split as the zarr oracle's numcodecs vs the reader's zarr-python.
    """
    import tifffile

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        arr = np.asarray(page.asarray(), dtype="float64")
        tags = {t.name: t.value for t in page.tags.values()}

    scale, tie = tags["ModelPixelScaleTag"], tags["ModelTiepointTag"]
    sx, sy = float(scale[0]), float(scale[1])
    # The tiepoint maps raster point (i0, j0) to model point (x0, y0); for these
    # rasters that is the top-left CORNER, so cell CENTRES sit half a cell in.
    i0, j0, x0, y0 = float(tie[0]), float(tie[1]), float(tie[3]), float(tie[4])
    nlat, nlon = arr.shape
    # Model space is y-up while raster rows run downward: lat DECREASES with row.
    lon = x0 + (np.arange(nlon, dtype="float64") - i0 + 0.5) * sx
    lat = y0 - (np.arange(nlat, dtype="float64") - j0 + 0.5) * sy

    nodata = tags.get("GDAL_NODATA")
    if nodata is not None:
        sentinel = float(str(nodata).strip().strip("\x00").strip())
        arr[arr == sentinel] = np.nan

    band = next(iter(expected["variables"]))  # single-band: `Band1`
    return {band: arr}, {"lon": lon, "lat": lat}


READERS = {"netcdf": read_netcdf, "csv": read_csv, "ff10": read_ff10,
           "zarr": read_zarr, "shapefile": read_shapefile,
           "parquet": read_parquet, "geotiff": read_geotiff}


def _verify_zarr_objects(case) -> list:
    """Checks 1+2 PER OBJECT for a store-backed (zarr) case: a Zarr store is many
    objects, not one blob, so key-agreement + integrity are verified for each
    object in the case's ``objects`` array (sha256(url)==cache_key,
    sha256(blob)==content_sha256==manifest.sha256_content, len==bytes, url match)."""
    errs: list = []
    for o in case["objects"]:
        key = hashlib.sha256(o["url"].encode("utf-8")).hexdigest()
        if key != o["cache_key"]:
            errs.append(f"cache-key: sha256({o['url']})={key} != {o['cache_key']}")
        blob = (CORPUS / o["blob_path"]).read_bytes()
        content_sha = hashlib.sha256(blob).hexdigest()
        if content_sha != o["content_sha256"]:
            errs.append(f"integrity: {o['url']} blob sha256 {content_sha} != {o['content_sha256']}")
        if len(blob) != o["bytes"]:
            errs.append(f"integrity: {o['url']} blob bytes {len(blob)} != {o['bytes']}")
        man_path = CORPUS / "cache" / "v1" / "meta" / f"{o['cache_key']}.json"
        manifest = json.loads(man_path.read_text())
        if manifest["sha256_content"] != o["content_sha256"]:
            errs.append(f"integrity: {o['url']} manifest.sha256_content mismatch")
        if manifest["bytes"] != o["bytes"]:
            errs.append(f"integrity: {o['url']} manifest.bytes mismatch")
        if manifest["url"] != o["url"]:
            errs.append(f"integrity: {o['url']} manifest.url mismatch")
    return errs


def verify_case(case_path: pathlib.Path) -> list:
    errs: list = []
    case = json.loads(case_path.read_text())

    if case.get("format") == "zarr":
        # 1 + 2 per object (a Zarr case's resolved_url is a store base, not a blob).
        errs += _verify_zarr_objects(case)
    else:
        # 1. cache-key agreement
        key = hashlib.sha256(case["resolved_url"].encode("utf-8")).hexdigest()
        if key != case["cache_key"]:
            errs.append(f"cache-key: sha256(resolved_url)={key} != case.cache_key={case['cache_key']}")

        # 2. manifest integrity
        blob = (CORPUS / case["blob_path"]).read_bytes()
        content_sha = hashlib.sha256(blob).hexdigest()
        if content_sha != case["content_sha256"]:
            errs.append(f"integrity: blob sha256 {content_sha} != case.content_sha256")
        if len(blob) != case["bytes"]:
            errs.append(f"integrity: blob bytes {len(blob)} != case.bytes {case['bytes']}")
        manifest = json.loads((CORPUS / case["manifest_path"]).read_text())
        if manifest["sha256_content"] != case["content_sha256"]:
            errs.append("integrity: manifest.sha256_content != case.content_sha256")
        if manifest["bytes"] != case["bytes"]:
            errs.append("integrity: manifest.bytes != case.bytes")
        if manifest["url"] != case["resolved_url"]:
            errs.append("integrity: manifest.url != case.resolved_url")

    # 3 + 4. decode + native-array equality
    reader = READERS.get(case["format"])
    if reader is None:
        errs.append(f"format: no reference reader for '{case['format']}' (stub-only?)")
        return errs
    # A zarr case is a store (many objects), not a single blob: the oracle needs
    # the objects + selection from the whole case, not just one blob_path.
    if case["format"] == "zarr":
        got, coords = read_zarr(CORPUS, case)
    elif case["format"] == "ff10":
        # ff10 needs the case's decode block (zip member selection + header skip).
        got, coords = read_ff10(CORPUS / case["blob_path"], case["expected"],
                                case.get("decode"))
    elif case["format"] == "shapefile":
        # shapefile needs the case's decode block (zip member + numeric_columns).
        got, coords = read_shapefile(CORPUS / case["blob_path"], case["expected"],
                                     case.get("decode"))
    elif case["format"] == "parquet":
        # parquet needs the case's decode block (float_columns + the two null
        # gates); the projection is the expected variable set.
        got, coords = read_parquet(CORPUS / case["blob_path"], case["expected"],
                                   case.get("decode"))
    elif case["format"] == "netcdf":
        # netcdf takes the case's orthogonal `select` (the decode-time window);
        # a case without one reads the blob whole, as before.
        got, coords = read_netcdf(CORPUS / case["blob_path"], case["expected"],
                                  case.get("select"))
    else:
        got, coords = reader(CORPUS / case["blob_path"], case["expected"])
    for name, spec in case["expected"]["variables"].items():
        if name not in got:
            errs.append(f"{name}: missing from reader output")
            continue
        if spec["dtype"] == "string":
            _cmp_string(got[name], spec["data"], name, errs)
        else:
            _cmp_numeric(got[name], spec["data"], name, errs)
    for name, spec in case["expected"].get("coords", {}).items():
        if name not in coords:
            errs.append(f"coord {name}: missing from reader output")
            continue
        _cmp_numeric(coords[name], spec["data"], f"coord {name}", errs)
    return errs


def validate_schemas() -> int:
    """Optional: validate manifests + cases against ../spec/schemas (if jsonschema
    is installed). Cross-language tracks rely on the schemas as the contract; this
    is the Python convenience check. Returns the number of schema failures."""
    try:
        from jsonschema import Draft202012Validator
        from referencing import Registry, Resource
    except Exception:
        print("schema-validation: SKIP (jsonschema/referencing not installed)")
        return 0
    sdir = HERE.parent / "spec" / "schemas"
    schemas = {json.loads(p.read_text())["$id"]: json.loads(p.read_text())
               for p in sdir.glob("*.json")}
    resources = [(sid, Resource.from_contents(s)) for sid, s in schemas.items()]
    resources += [(p.name, Resource.from_contents(json.loads(p.read_text())))
                  for p in sdir.glob("*.json")]
    registry = Registry().with_resources(resources)
    man = schemas["https://earthsci.dev/earthsciio/schemas/manifest.schema.json"]
    case = schemas["https://earthsci.dev/earthsciio/schemas/cache-case.schema.json"]
    fails = 0
    for p in (CORPUS / "cache").rglob("meta/*.json"):
        errs = list(Draft202012Validator(man, registry=registry).iter_errors(json.loads(p.read_text())))
        if errs:
            fails += 1
            print(f"schema FAIL manifest {p.name}: {errs[0].message}")
    for p in (CORPUS / "cases").glob("*.json"):
        errs = list(Draft202012Validator(case, registry=registry).iter_errors(json.loads(p.read_text())))
        if errs:
            fails += 1
            print(f"schema FAIL case {p.name}: {errs[0].message}")
    print(f"schema-validation: {'OK' if not fails else str(fails) + ' FAILED'}")
    return fails


def main() -> int:
    index = json.loads((CORPUS / "cases.json").read_text())
    schema_fails = validate_schemas()
    failed = 0
    for entry in index["cases"]:
        case_path = CORPUS / entry["file"]
        errs = verify_case(case_path)
        if errs:
            failed += 1
            print(f"FAIL  {entry['id']}")
            for e in errs:
                print(f"        - {e}")
        else:
            print(f"PASS  {entry['id']}")
    print(f"\n{len(index['cases']) - failed}/{len(index['cases'])} cases passed (offline)")
    return 1 if (failed or schema_fails) else 0


if __name__ == "__main__":
    sys.exit(main())

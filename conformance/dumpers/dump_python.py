#!/usr/bin/env python3
"""Python track's native-array dumper for the cross-language conformance harness.

Drives the **Python Provider** (:class:`earthsciio.Provider`) over every committed
corpus case, fully OFFLINE (the cache is rooted at the corpus and refuses the
network), and emits the decoded native arrays as a canonical JSON dump. The
cross-language comparator (:mod:`conformance.crosscheck`) diffs this dump against
the Julia and Rust dumps and the corpus oracle to prove array equality across all
three tracks (``esio-9nb.9``).

This is the **reference dumper**: the Julia (``dump_julia.jl``) and Rust
(``rust/examples/conformance_dump.rs``) dumpers emit the *same* schema.

Dump schema — ``earthsciio/native-dump/v1`` (see ``conformance/CROSSLANG.md``):

    {
      "schema": "earthsciio/native-dump/v1",
      "language": "python",
      "provider": "earthsciio.Provider",
      "readers": ["csv", "netcdf"],          # formats this track can decode HERE
                                             # (active AND its extra installed)
      "cases": {
        "<case_id>": {
          "format": "netcdf",
          "status": "decoded",
          "variables": {"<name>": {"dtype","dims","shape","data"}},
          "coords":    {"<name>": {"dtype","dims","shape","data","units?","calendar?"}}
        },
        "<case_id>": {"format":"csv","status":"skipped","reason":"..."}  # no reader
      }
    }

``data`` is the field flattened **row-major (C order)** per ``shape``; a masked /
``_FillValue`` cell is ``null`` (== NaN); strings are emitted verbatim. A case
whose ``format`` has no reader in this track is ``status="skipped"`` (explicit,
never silently dropped) so the comparator can tell a real coverage gap from a bug.

Usage:  python3 conformance/dumpers/dump_python.py [out.json]   # default: stdout
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
from typing import Any, Dict, List

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
CORPUS = HERE.parent / "corpus"
REPO_ROOT = HERE.parent.parent

# Run standalone (no install needed) by putting the repo root on the path, so the
# harness driver can invoke this from anywhere — mirrors verify.py's offline ethos.
sys.path.insert(0, str(REPO_ROOT))

from earthsciio import Cache, DataSource, Provider  # noqa: E402
from earthsciio.registry import format_registry  # noqa: E402


def _dtype_str(arr: np.ndarray) -> str:
    """The native-field schema dtype name for a numeric numpy array."""
    if np.issubdtype(arr.dtype, np.floating):
        return "float64"
    if arr.dtype == np.bool_:
        return "bool"
    if arr.dtype == np.int32:
        return "int32"
    if np.issubdtype(arr.dtype, np.integer):
        return "int64"
    raise TypeError(f"unexpected numeric dtype {arr.dtype}")


def _encode_field(field: Any) -> Dict[str, Any]:
    """Encode a :class:`earthsciio.native.NativeField` to the dump schema.

    Numeric arrays flatten row-major with ``null`` for NaN; string columns become
    a flat list of ``str``. ``dims``/``shape`` are carried in file order.
    """
    data = field.data
    dims = list(field.dims)
    if isinstance(data, np.ndarray):
        flat = np.asarray(data).reshape(-1)
        shape = list(data.shape)
        if flat.dtype.kind in ("S", "U"):
            # A netcdf TEXT variable (spec/conformance.md §3): xarray decodes a
            # `char` array to fixed-width BYTES (`|S4`) and an `NC_STRING` to
            # `<U`, both of them `string` fields in the dump schema. They must go
            # through the string branch, not the numeric one: `int(b'ab')` is a
            # TypeError, which is why no corpus case could carry a char variable
            # before. `shape` is the field's own, so a SCALAR string (a `char
            # label(strlen)` on a private dimension) keeps `shape == []` while its
            # one value still flattens to a one-element `data` list.
            values = [
                v.decode("utf-8") if isinstance(v, bytes) else str(v)
                for v in flat.tolist()
            ]
            enc = {"dtype": "string", "dims": dims, "shape": shape, "data": values}
            return _with_fill_value(enc, field)
        if np.issubdtype(flat.dtype, np.floating):
            values: List[Any] = [
                None if (math.isnan(x)) else float(x) for x in flat.tolist()
            ]
        elif flat.dtype == np.bool_:
            values = [bool(x) for x in flat.tolist()]
        else:
            values = [int(x) for x in flat.tolist()]
        enc = {"dtype": _dtype_str(flat), "dims": dims, "shape": shape, "data": values}
        return _with_fill_value(enc, field)
    # string column: a plain Python list of str
    values = [str(x) for x in data]
    enc = {"dtype": "string", "dims": dims, "shape": [len(values)], "data": values}
    return _with_fill_value(enc, field)


def _with_fill_value(enc: Dict[str, Any], field: Any) -> Dict[str, Any]:
    """Carry a surviving fill/missing sentinel into the dump under ONE key.

    ``spec/conformance.md`` §3 pins where each track reports it: Rust has a
    dedicated ``NativeField.fill_value`` field, Python and Julia carry
    ``attrs["fill_value"]``. Those are the SAME datum in three spellings, not
    three decisions, so every dumper normalises to the dump schema's single
    ``fill_value`` key — otherwise the comparator would read a difference that
    is not one (or, worse, miss one track dropping the sentinel entirely).
    """
    fv = getattr(field, "attrs", {}).get("fill_value")
    if fv is not None:
        enc["fill_value"] = float(fv) if isinstance(fv, float) else int(fv)
    return enc


def _encode_coord(field: Any) -> Dict[str, Any]:
    """A coord is a field plus the CF ``units``/``calendar`` it carries (if any)."""
    enc = _encode_field(field)
    for k in ("units", "calendar"):
        if k in field.attrs:
            enc[k] = str(field.attrs[k])
    return enc


def dump_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """Run the Python Provider over one corpus case and encode its native arrays.

    Skips (without error) a case whose ``format`` has no registered reader, so the
    harness reports the gap instead of failing — matching the Rust track, which
    ships ``netcdf`` only.

    A reader whose optional decode stack is not installed skips the same way, and
    says which extra is missing. Registration alone does NOT mean this
    environment can decode the format: `geotiff` is always registered but needs
    rasterio or tifffile, so without them an unguarded run raised ImportError
    from inside the reader and the coverage gate reported the confusing
    "registers a reader but did not decode this case".
    """
    fmt = case["format"]
    if fmt not in format_registry or format_registry.status(fmt) != "active":
        return {
            "format": fmt,
            "status": "skipped",
            "reason": f"no active reader registered for format '{fmt}' in the Python track",
        }
    missing = format_registry.missing_requirements(fmt)
    if missing:
        return {
            "format": fmt,
            "status": "skipped",
            "reason": (
                f"the '{fmt}' reader needs {', '.join(missing)}, not installed "
                f"in this environment"
            ),
        }

    # An OFFLINE cache rooted at the corpus: every case resolves from disk by its
    # sha256(resolved_url) key; a network attempt would raise (verify=True checks
    # the blob against the manifest on read).
    cache = Cache(root=CORPUS / "cache", offline=True, verify=True)

    reader_kwargs: Dict[str, Any] = {}
    variables: List[str] = []
    if fmt == "csv":
        # numeric_columns is REQUIRED by the loader spec (digit-only text columns
        # like location_id must stay strings); the corpus case pins the list.
        reader_kwargs["numeric_columns"] = list(case["decode"]["numeric_columns"])
    elif fmt == "ff10":
        # FF10 point: the case pins the 42 numeric columns, the schema kind, and
        # the zip member selection — member (singular), members/member_glob
        # (multi-member; sorted-name concat), skip_header_row (drop one asserted
        # `country_cd` header line per member). member=null decodes the bare blob.
        dec = case["decode"]
        reader_kwargs["numeric_columns"] = list(dec["numeric_columns"])
        reader_kwargs["kind"] = dec.get("kind", "point")
        reader_kwargs["member"] = dec.get("member")
        if dec.get("members") is not None:
            reader_kwargs["members"] = list(dec["members"])
        if dec.get("member_glob") is not None:
            reader_kwargs["member_glob"] = str(dec["member_glob"])
        reader_kwargs["skip_header_row"] = bool(dec.get("skip_header_row") or False)
    elif fmt == "shapefile":
        # ESRI shapefile: the case pins the `.shp` member inside the zip blob and
        # the text code column the model wants as a number.
        dec = case["decode"]
        if dec.get("member") is not None:
            reader_kwargs["member"] = str(dec["member"])
        if dec.get("numeric_columns"):
            reader_kwargs["numeric_columns"] = list(dec["numeric_columns"])
    elif fmt == "parquet":
        # Columnar table. `variables` is the loader's PROJECTION and is pushed
        # into the reader (only those column chunks come off disk); the case's
        # decode block pins the three decode options — `float_columns` (an
        # integer measurement, or a column of fixed-decimal TEXT, read as
        # float64) and the two null gates `null_int` / `null_string`.
        dec = case["decode"]
        variables = list(case.get("variables") or [])
        if dec.get("float_columns") is not None:
            reader_kwargs["float_columns"] = list(dec["float_columns"])
        if dec.get("null_int") is not None:
            reader_kwargs["null_int"] = int(dec["null_int"])
        if dec.get("null_string") is not None:
            reader_kwargs["null_string"] = str(dec["null_string"])
    elif fmt == "netcdf":
        # Whole-file, but selection-capable: a case carrying an orthogonal
        # `select` is a DECODE-TIME window — the same blob under the same cache
        # key, with only the requested hyperslab materialised. A case without
        # `axes` reads the blob whole (the `{all_records: true}` form).
        if (case.get("select") or {}).get("axes") is not None:
            reader_kwargs["select"] = case["select"]
    elif fmt == "zarr":
        # Store-backed: the reader is handed (cache, base_url, variables, select).
        # `variables` names the arrays (no .zmetadata to enumerate); `select` (the
        # orthogonal selection) rides in reader_kwargs and drives lazy chunk fetch.
        variables = list(case["variables"])
        reader_kwargs["select"] = case.get("select")

    loader = DataSource(
        name=case["loader"],
        format=fmt,
        url=case["resolved_url"],
        variables=variables,
        reader_kwargs=reader_kwargs,
    )
    provider = Provider(loader, cache)
    nds = provider.materialize()  # CONST: read the single corpus blob once

    return {
        "format": fmt,
        "status": "decoded",
        "variables": {n: _encode_field(f) for n, f in nds.variables.items()},
        "coords": {n: _encode_coord(f) for n, f in nds.coords.items()},
    }


def selected_ids() -> Optional[set]:
    """The case ids ``$ESIO_CONFORMANCE_CASES`` restricts this run to, or ``None``.

    A comma-separated list. Every dumper and :mod:`conformance.crosscheck` honour
    the SAME variable, so a filtered run is still a complete cross-check of the
    cases it names — it just narrows the corpus. It exists for an environment
    where one track's backend for some *other* format is missing or too old (a
    too-old ``zarr``, say): without it, one unrelated broken case makes the whole
    gate unrunnable and a real divergence elsewhere invisible. Unset ⇒ all cases,
    which is what CI runs.
    """
    raw = os.environ.get("ESIO_CONFORMANCE_CASES", "").strip()
    if not raw:
        return None
    return {c.strip() for c in raw.split(",") if c.strip()}


def main(argv: List[str]) -> int:
    index = json.loads((CORPUS / "cases.json").read_text())
    only = selected_ids()
    cases: Dict[str, Any] = {}
    for entry in index["cases"]:
        case = json.loads((CORPUS / entry["file"]).read_text())
        if only is not None and case["id"] not in only:
            continue
        cases[case["id"]] = dump_case(case)

    out = {
        "schema": "earthsciio/native-dump/v1",
        "language": "python",
        "provider": "earthsciio.Provider",
        # What this environment can actually decode, not merely what the library
        # implements. The comparator's coverage gate holds a track to every
        # format it lists here, so listing a reader whose extra is missing turns
        # an honest "not installed" into a spurious failure.
        "readers": sorted(
            k
            for k in format_registry.keys()
            if format_registry.status(k) == "active" and format_registry.available(k)
        ),
        "cases": cases,
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    if len(argv) > 1:
        pathlib.Path(argv[1]).write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

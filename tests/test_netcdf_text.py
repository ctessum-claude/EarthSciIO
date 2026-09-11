"""NetCDF TEXT variables — the PYTHON track's reference decode (spec §3).

A ``char`` array is not a native array of characters. Its last dimension is the
string LENGTH exactly when nothing else claims it as an axis (no coordinate
variable of its own, and every variable using it is a ``char`` using it last) —
xarray's ``conventions.stackable``, which gates its ``CharacterArrayCoder``. The
same spelling therefore means different things depending on the rest of the
FILE, and ``spec/conformance.md`` §3 makes what xarray produces here the
reference the Julia and Rust readers are held to.

These tests exist because that reference had no test of its own: the shapes
below were measured against xarray and then reproduced by hand in two other
tracks, so a future xarray change (or an accidental edit to
``_field_from_dataarray``) would move the contract silently and be discovered
only as a three-way corpus failure with no track obviously at fault.

The four classic fixtures are byte-for-byte the CDF-1 blobs of
``rust/src/format/netcdf.rs`` (``CHAR_VAR_CDF1``, ``CHAR_HOLES_CDF1``,
``STRING_ROWS_CDF1``, ``SCALAR_STRING_CDF1``) and of
``julia/test/test_readers.jl``, so all three tracks are held to the same bytes.
Cross-language equality on a COMMITTED blob is the corpus's job —
``station-labels-text`` / ``station-labels-window``.
"""

from __future__ import annotations

import numpy as np
import pytest

from earthsciio import NetCDFReader

# dim `n=3`; `float value(n)` and `char label(n) = "abc"`. `n` is a REAL axis
# (`value` lives on it), so `label` is three ONE-character strings.
CHAR_VAR_CDF1 = bytes.fromhex(
    "43444601000000000000000a00000001"
    "000000016e0000000000000300000000"
    "000000000000000b0000000200000005"
    "76616c75650000000000000100000000"
    "0000000000000000000000050000000c"
    "0000007c000000056c6162656c000000"
    "00000001000000000000000000000000"
    "0000000200000004000000883f800000"
    "400000004040000061626300"
)

# The same file with an interior NUL: `char label(n) = "a\0c"`.
CHAR_HOLES_CDF1 = bytes.fromhex(
    "43444601000000000000000a00000001"
    "000000016e0000000000000300000000"
    "000000000000000b0000000200000005"
    "76616c75650000000000000100000000"
    "0000000000000000000000050000000c"
    "0000007c000000056c6162656c000000"
    "00000001000000000000000000000000"
    "0000000200000004000000883f800000"
    "400000004040000061006300"
)

# dims `n=4` and a PRIVATE `strlen=4`; `float value(n)` and `char label(n,
# strlen)` = "ab", "cd  ", "efgh", "" — four strings on dims ("n",).
STRING_ROWS_CDF1 = bytes.fromhex(
    "43444601000000000000000a00000002"
    "000000016e0000000000000400000006"
    "7374726c656e00000000000400000000"
    "000000000000000b0000000200000005"
    "76616c75650000000000000100000000"
    "00000000000000000000000500000010"
    "00000090000000056c6162656c000000"
    "00000002000000000000000100000000"
    "000000000000000200000010000000a0"
    "3f800000400000004040000040800000"
    "61620000636420206566676800000000"
)

# dims `n=3` and a private `strlen=5`; a ONE-dimensional `char label(strlen)` =
# "hi", whose only dimension is the length: a SCALAR string.
SCALAR_STRING_CDF1 = bytes.fromhex(
    "43444601000000000000000a00000002"
    "000000016e0000000000000300000006"
    "7374726c656e00000000000500000000"
    "000000000000000b0000000200000005"
    "76616c75650000000000000100000000"
    "0000000000000000000000050000000c"
    "0000008c000000056c6162656c000000"
    "00000001000000010000000000000000"
    "0000000200000008000000983f800000"
    "40000000404000006869000000000000"
)


@pytest.fixture
def blob(tmp_path):
    """Write raw bytes to an extension-less path, as the content-addressed cache
    does, and hand back the path."""

    def _write(data, name="blob"):
        path = tmp_path / name
        path.write_bytes(data)
        return path

    return _write


def _label(path):
    field = NetCDFReader().read_native(path)["label"]
    return list(field.dims), list(np.asarray(field.data).shape), [
        v.decode() if isinstance(v, bytes) else str(v)
        for v in np.asarray(field.data).reshape(-1).tolist()
    ]


def test_a_char_variable_on_a_shared_axis_is_one_string_per_cell(blob):
    # `n` counts elements (a float lives on it), so it is not a length.
    assert _label(blob(CHAR_VAR_CDF1)) == (["n"], [3], ["a", "b", "c"])


def test_a_nul_cell_in_a_char_axis_is_the_empty_string(blob):
    # numpy renders a `|S1` NUL byte as `b""` — never `"\0"`.
    assert _label(blob(CHAR_HOLES_CDF1)) == (["n"], [3], ["a", "", "c"])


def test_a_private_last_dimension_is_the_string_length_not_an_axis(blob):
    # The length dimension is CONSUMED: four strings, not a 4x4 character grid.
    # Trailing NULs are stripped and trailing SPACES are data.
    assert _label(blob(STRING_ROWS_CDF1)) == (
        ["n"], [4], ["ab", "cd  ", "efgh", ""],
    )


def test_a_one_dimensional_char_on_a_private_dimension_is_a_scalar_string(blob):
    # The 1-D case of the same rule: one string, so `dims` and `shape` are empty.
    assert _label(blob(SCALAR_STRING_CDF1)) == ([], [], ["hi"])


def test_read_everything_returns_the_text_variable_alongside_the_numeric_one(blob):
    # A track that skipped text would hand back a different SET of fields for the
    # same bytes — a divergence, not a gap (spec/conformance.md §3).
    for data in (CHAR_VAR_CDF1, STRING_ROWS_CDF1, SCALAR_STRING_CDF1):
        nds = NetCDFReader().read_native(blob(data))
        assert sorted(nds.variables) == ["label", "value"]


def test_the_projection_selects_a_text_variable_like_any_other(blob):
    path = blob(STRING_ROWS_CDF1)
    nds = NetCDFReader().read_native(path, variables=["label"])
    assert sorted(nds.variables) == ["label"]
    with pytest.raises(KeyError, match="nope"):
        NetCDFReader().read_native(path, variables=["nope"])


def test_an_nc_string_is_already_one_string_per_element(tmp_path):
    # NetCDF-4 `NC_STRING`: nothing is consumed, `dims`/`shape` are the
    # variable's own. Authored here rather than committed, because an HDF5 blob
    # is not byte-reproducible across libhdf5 versions the way CDF-1 is.
    netCDF4 = pytest.importorskip("netCDF4")
    path = tmp_path / "nc4strings.nc"
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("n", 3)
        var = ds.createVariable("label", str, ("n",))
        for i, s in enumerate(["alpha", "be", "gamma"]):
            var[i] = s
        ds.createVariable("value", "f8", ("n",))[:] = [1.0, 2.0, 3.0]
    assert _label(path) == (["n"], [3], ["alpha", "be", "gamma"])


def test_a_select_windows_the_strings_not_their_characters(blob):
    # A selection is applied by dimension NAME to every array, text included; the
    # positional axis match is over the DECODED dims, and xarray has already
    # folded the string length away by the time this reader sees the dataset.
    path = blob(STRING_ROWS_CDF1)
    select = {"axes": [{"indices": [0, 2]}]}
    nds = NetCDFReader().read_native(path, select=select)
    assert [v.decode() for v in nds["label"].data.tolist()] == ["ab", "efgh"]
    assert nds["value"].data.tolist() == [1.0, 3.0]

    # `char label(n, strlen)` is a RANK-1 field on two on-disk dimensions, so a
    # 2-axis select matches nothing in this blob and says so rather than binding
    # an axis to a string length.
    with pytest.raises(ValueError, match="no variable in the blob has rank 2"):
        NetCDFReader().read_native(
            path, select={"axes": ["all", {"indices": [0, 1]}]}
        )


def test_a_scalar_string_is_rank_0_under_a_select(blob):
    # `value(n)` and `char label(strlen)` are both rank 1 ON DISK, but `label`
    # decodes to a scalar. The single axis binds `n` alone — binding `strlen` too
    # would truncate "hi" to "h", a wrong STRING no shape assertion would catch.
    nds = NetCDFReader().read_native(
        blob(SCALAR_STRING_CDF1), select={"axes": [{"indices": [0, 2]}]}
    )
    assert nds["value"].data.tolist() == [1.0, 3.0]
    assert list(nds["label"].dims) == []
    assert nds["label"].data.reshape(-1).tolist() == [b"hi"]

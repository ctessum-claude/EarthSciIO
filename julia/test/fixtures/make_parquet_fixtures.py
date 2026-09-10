#!/usr/bin/env python3
"""Regenerate the committed `parquet_*.parquet` fixtures for the Julia
`parquet` reader tests (`julia/test/test_parquet_reader.jl`).

They are written by **pyarrow** — the reference Apache Arrow/Parquet
implementation and the peer of the Rust track's `parquet` crate — on purpose:
the Julia reader's job is to decode files the wider world writes, and a fixture
written by the same library the Rust test uses is what makes the two tracks
comparable. Parquet2.jl (the Julia decode backend) can also write, but its
writer cannot emit several of the shapes the contract is about (a `date32`, a
sub-second `time64`, a `decimal128`, a nested column), so it cannot generate its
own test corpus.

Every blob is a few kilobytes. Run from anywhere:

    python3 julia/test/fixtures/make_parquet_fixtures.py

Requires `pyarrow>=14`. The fixtures are committed, so this script is only
needed when a case changes.
"""

import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

OUT = os.path.dirname(os.path.abspath(__file__))


def write(name, table, **kw):
    path = os.path.join(OUT, name)
    pq.write_table(table, path, **kw)
    print(f"{name}: {os.path.getsize(path)} bytes")


# One column of every supported family, three rows. Mirrors the Rust track's
# `every_supported_arrow_type_maps_onto_the_native_contract` fixture, plus the
# time32/time64/timestamp(us) columns that pin the temporal widths.
write(
    "parquet_types.parquet",
    pa.table(
        {
            "b": pa.array([True, False, True], pa.bool_()),
            "i16": pa.array([1, -2, 3], pa.int16()),
            "i32": pa.array([10, -20, 30], pa.int32()),
            "i64": pa.array([100, -200, 300], pa.int64()),
            "u16": pa.array([1, 2, 3], pa.uint16()),
            "u32": pa.array([1, 2, 3], pa.uint32()),
            "u64": pa.array([1, 2, 3], pa.uint64()),
            "f32": pa.array([1.5, 2.5, 3.5], pa.float32()),
            "f64": pa.array([1.25, 2.5, 3.75], pa.float64()),
            "s": pa.array(["2260000000", "x", ""], pa.string()),
            "cat": pa.array(["gas", "diesel", "gas"]).dictionary_encode(),
            "d32": pa.array([19000, 19001, 19002], pa.date32()),
            "ts": pa.array([1700000000000, 0, -5], pa.timestamp("ms")),
            "dec": pa.array(
                ["261.000000000000", "-1.500000000000", "0.000000000000"]
            ).cast(pa.decimal128(30, 12)),
            "t32": pa.array([1000, 2000, 3000], pa.time32("ms")),
            "t64": pa.array([1000, 2000, 3000], pa.time64("us")),
            "tsu": pa.array([1700000000000000, 0, -5000], pa.timestamp("us")),
        }
    ),
    compression="none",
)

# The null policy + `float_columns`: a null in each family, an all-null (arrow
# `null`) column, and a column of fixed-decimal TEXT with a blank cell — the
# shape the MOVES snapshots use for byte-reproducible floats.
write(
    "parquet_nulls.parquet",
    pa.table(
        {
            "f": pa.array([1.0, None, 3.0], pa.float64()),
            "i": pa.array([1, None, 3], pa.int64()),
            "i32": pa.array([1, None, 3], pa.int32()),
            "s": pa.array(["a", None, "c"], pa.string()),
            "bo": pa.array([True, None, False], pa.bool_()),
            "nul": pa.array([None, None, None], pa.null()),
            "dtxt": pa.array(["261.000000000000", "   ", "-1.5"], pa.string()),
            "badtxt": pa.array(["1.0", "not a number", "3.0"], pa.string()),
        }
    ),
    compression="none",
)

# A zero-row table is TYPED, not absent: the schema lives in the footer.
write(
    "parquet_empty.parquet",
    pa.table(
        {
            "id": pa.array([], pa.int64()),
            "code": pa.array([], pa.string()),
            "val": pa.array([], pa.float64()),
        }
    ),
    compression="none",
)

# A binary column has no rank-1 reading: skipped when unrequested, refused when
# named in `variables`.
write(
    "parquet_binary.parquet",
    pa.table(
        {
            "id": pa.array([7, 8], pa.int64()),
            "blob": pa.array([b"x", b"y"], pa.binary()),
        }
    ),
    compression="none",
)

# A NESTED column. Parquet2.jl cannot open such a file at all (it builds a
# column for every schema node when it opens the footer, and a nested node has
# no column metadata), so this fixture pins the reader's clear refusal.
write(
    "parquet_nested.parquet",
    pa.table(
        {
            "id": pa.array([7, 8], pa.int64()),
            "st": pa.array([{"a": 1}, {"a": 2}], pa.struct([("a", pa.int32())])),
        }
    ),
    compression="none",
)

# A `uint64` past `int64::MAX` is an error naming the column and row, never a
# wraparound into a negative ID.
write(
    "parquet_bigu64.parquet",
    pa.table({"u": pa.array([2**64 - 1], pa.uint64())}),
    compression="none",
)

# Projection pushdown, proved at the BYTE level: two SNAPPY-compressed columns
# whose chunks the test corrupts in place. Enough rows that each column is its
# own multi-page chunk, few enough that the blob stays a few KB.
n = 512
write(
    "parquet_pushdown.parquet",
    pa.table(
        {
            "keep": pa.array(list(range(n)), pa.int64()),
            "poison": pa.array([f"row-{i}" for i in range(n)], pa.string()),
        }
    ),
    compression="snappy",
)

# NEGATIVE decimals at every FIXED_LEN_BYTE_ARRAY width pyarrow uses. Parquet
# stores an FLBA decimal as a big-endian TWO'S COMPLEMENT integer, and
# Parquet2.jl folds those bytes into an Int64 with no sign extension, so every
# negative in a column narrower than 8 bytes came back as its huge unsigned
# reinterpretation until `EarthSciIOParquet2Ext` repaired it. The precisions
# below are one per width: 2 -> 1 byte, 4 -> 2, 9 -> 4, 12 -> 6, 18 -> 8 (the
# width that never needed repairing). Each column carries its extreme negative,
# its extreme positive, and the cells either side of zero.
write(
    "parquet_decimals.parquet",
    pa.table(
        {
            "d02": pa.array(["99", "-99", "0", "-1", "1"]).cast(pa.decimal128(2, 0)),
            "d04": pa.array(["99.99", "-99.99", "0.00", "-0.01", "0.01"])
            .cast(pa.decimal128(4, 2)),
            "d09": pa.array(
                ["9999999.99", "-9999999.99", "0.00", "-0.01", "0.01"]
            ).cast(pa.decimal128(9, 2)),
            "d12": pa.array(
                ["9999999999.99", "-9999999999.99", "0.00", "-0.01", "0.01"]
            ).cast(pa.decimal128(12, 2)),
            "d18": pa.array(
                ["999999.999999", "-999999.999999", "0.000000", "-0.000001",
                 "0.000001"]
            ).cast(pa.decimal128(18, 6)),
        }
    ),
    compression="none",
)

# The ONE width the repair cannot reach: a 7-byte FLBA (pyarrow precision
# 15-16). The unsigned fold of a negative is 2^56 — 17 decimal digits, which
# overflows a Dec64's 16 — so Parquet2 throws during page decode, before any
# repair could run. The reader turns that into a message naming the limit.
write(
    "parquet_decimal7.parquet",
    pa.table(
        {
            "pos": pa.array(["1.25", "2.50"]).cast(pa.decimal128(15, 2)),
            "neg": pa.array(["-1.25", "2.50"]).cast(pa.decimal128(15, 2)),
        }
    ),
    compression="none",
)

print("wrote fixtures to", OUT, file=sys.stderr)

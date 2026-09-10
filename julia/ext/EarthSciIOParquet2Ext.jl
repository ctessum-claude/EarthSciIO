# Parquet decode backend for the `parquet` reader (Julia track).
#
# Loaded via `using Parquet2` (a weakdep extension): it supplies the
# more-specific `read_native(::ParquetReader, path; …)` method, keeping a base
# EarthSciIO install free of the Parquet stack — the Julia mirror of the Python
# reader's lazy `pyarrow` import and the Rust track's `parquet` crate. The pure
# Julia Parquet2.jl owns the container (footer, row groups, page encodings,
# snappy/zstd/gzip/lz4/brotli decompression); this extension only pulls each
# column's decoded values out, normalizes the few types Parquet2 hands back in a
# DECODED form the §3 contract wants RAW (a `Date`/`Time`/`DateTime` back to its
# stored integer), REPAIRS the one place its decode is simply wrong (a narrow
# `FIXED_LEN_BYTE_ARRAY` decimal read unsigned — `_decimal_sign_fix`), and hands
# them to the core `_assemble_parquet`, where the dtype table, the null policy
# and `float_columns` live (shared with any future backend).
#
# Projection pushdown is REAL here: `_assemble_parquet` calls `loadcolumn` only
# for the columns that survive `variables`, and `Parquet2.load(ds, name)` reads
# and decompresses exactly that column's chunk out of the memory-mapped file. A
# skipped column's pages are never touched.
module EarthSciIOParquet2Ext

import EarthSciIO: read_native, ParquetReader, _assemble_parquet, _parquet_kind
import Parquet2
using Dates: Date, DateTime, Time, value

# The parquet `TimeType`/`TimestampType` unit exponent (-3 millis, -6 micros,
# -9 nanos), or `nothing` for a column that is not temporal.
_unit_exponent(pt) = pt isa Union{Parquet2.ParqTime,Parquet2.ParqDateTime} ? pt.exponent : nothing

# The parquet unit exponent, refused when the file does not declare one (there
# is no raw integer to recover without knowing what it counts).
function _exponent_or_throw(pt, name::AbstractString)
    e = _unit_exponent(pt)
    e === nothing && throw(ArgumentError(
        "column $(repr(name)): temporal column with an unknown time unit"))
    return e
end

# Undo the backend's logical decode, so the cell reaches `_assemble_parquet` as
# the RAW stored integer §3 requires. `nothing` when the decoded value is
# already a cell (the common case: integers, floats, decimals, text, booleans,
# and the `missing` of an all-null column), so those columns pass through with
# no per-row work at all.
function _rawconv(pt, name::AbstractString, decwidth = nothing)
    if pt isa Parquet2.ParqDecimal
        return _decimal_sign_fix(pt, decwidth)
    elseif pt isa Parquet2.ParqDate
        # Date32: days since the epoch.
        return x -> Int64(value(x - Date(1970, 1, 1)))
    elseif pt isa Parquet2.ParqTime
        # Time32/Time64: elapsed time since midnight, in the STORED unit.
        # `Dates.value(::Time)` counts NANOSECONDS, so divide by 10^(9+e).
        f = Int64(10)^(9 + _exponent_or_throw(pt, name))
        return x -> div(Int64(value(x)), f)
    elseif pt isa Parquet2.ParqDateTime
        # Timestamp: the stored offset from the epoch, in the STORED unit. NOTE
        # that Parquet2 decodes every timestamp into a `DateTime`, whose
        # resolution is one MILLISECOND, so a MICROS/NANOS column's sub-
        # millisecond digits are already gone by the time this runs — the one
        # known divergence from the Rust track's byte-exact raw integer.
        f = Int64(10)^(-3 - _exponent_or_throw(pt, name))   # units per millisecond
        return x -> Int64(value(x - DateTime(1970, 1, 1))) * f
    end
    return nothing
end

# A DECIMAL column needs no ARITHMETIC here: Parquet2 decodes it into a `Dec64`,
# which is `<: AbstractFloat`, so the core's `Float64(::Real)` fold turns it into
# the unscaled value ÷ 10^scale that §3 asks for.
#
# The Rust and Python tracks spell that arithmetic out as
# `f64(unscaled) / 10.0^scale`, and this IS the same number, not a "better" one:
# for `|unscaled| < 2^53` and `scale <= 22` both operands of their division are
# exact in binary64, so IEEE division returns the correctly-rounded quotient —
# which is precisely what a correctly-rounded decimal → binary64 conversion
# returns. Outside that range the two can differ by an ulp, but so can either
# from the true value, and Parquet2's own `Dec64` (16 significant decimal digits)
# has already bounded the fidelity: the unscaled integer is consumed during page
# decode and never reaches this extension, so a wider `Decimal128`/`Decimal256`
# cannot be read here at full precision by any arithmetic.
#
# What a decimal DOES need is a SIGN repair, below.

# The `FIXED_LEN_BYTE_ARRAY` width of a DECIMAL column in bytes, or `nothing`
# when the column is not FLBA-backed (an `INT32`/`INT64`-backed decimal reaches
# Parquet2 as a signed `Integer` and needs no repair). Read off the schema, so
# it costs no page decode.
function _decimal_flba_width(ds, name::AbstractString)
    col = Parquet2.Column(Parquet2.RowGroup(ds, 1), name)
    bt = Parquet2.parqbasetype(col)
    bt isa Parquet2.ParqFixedByteArray || return nothing
    return Parquet2.valuesize(bt)
end

# Repair Parquet2.jl's UNSIGNED reading of a narrow FLBA decimal.
#
# Parquet stores a `FIXED_LEN_BYTE_ARRAY` decimal as a big-endian TWO'S
# COMPLEMENT integer, so the top bit of the first byte is the sign.
# Parquet2.jl's `concat_integer` folds those `w` bytes into an `Int64` with no
# sign extension (`utils.jl`), which is only accidentally right at `w == 8`,
# where the sign bit lands in the `Int64`'s own. At `w < 8` every NEGATIVE value
# comes back as its unsigned reinterpretation: with pyarrow's widths a
# `decimal128(9, 2)` cell of `-2.50` decodes as `42949670.46` (2^32 - 250, ÷
# 100). That is a SILENTLY WRONG number — the worst failure mode there is, and a
# straight disagreement with the Rust and Python tracks, which both read the
# unscaled integer signed.
#
# The repair is exact, not a heuristic. A valid `w`-byte unscaled value lies in
# `[-2^(8w-1), 2^(8w-1))`, so the unsigned fold lands at or above `2^(8w-1)` if
# and only if the true value was negative, and subtracting `2^(8w)` recovers it
# exactly. Non-negative cells are untouched, so a file that reads correctly
# today still does.
#
# `w == 7` is beyond repair here: the unsigned fold of a negative reaches
# `2^56`, which is 17 decimal digits and does not fit `Dec64`'s 16, so Parquet2
# throws an `InexactError` during page decode before this ever runs (see the
# `read_native` catch). `w >= 8` never needed repairing.
function _decimal_sign_fix(pt, width)
    # `nothing`/>= 8: never needed one. 7: beyond reach — `read_native`'s catch
    # explains the failure Parquet2 raises before this would run.
    (width === nothing || width >= 7) && return nothing
    scale = -pt.scale                       # the §3 (positive) decimal scale
    den = 10.0^scale
    limit = Float64(Int64(1) << (8 * width - 1))
    wrap = Float64(Int64(1) << (8 * width))
    return function (x)
        # The unscaled integer, recovered exactly: |unscaled| < 2^48 here, well
        # inside binary64's exact range, so the round-trip through Float64 and
        # back is lossless.
        u = round(Float64(x) * den)
        u < limit && return Float64(x)
        return (u - wrap) / den
    end
end

"""
    read_native(::ParquetReader, path; variables=nothing, float_columns=nothing,
                null_int=nothing, null_string=nothing) -> NativeDataset

Decode a Parquet blob as a flat table. See [`EarthSciIO.ParquetReader`] for the
contract; every option is documented there. The keyword list is also the reader's
declared option set (`reader_option_keys`), so a [`EarthSciIO.Provider`] rejects
an unrecognised `reader_kwargs` key at construction rather than ignoring it.
"""
function read_native(::ParquetReader, path::AbstractString; variables = nothing,
                     float_columns = nothing, null_int = nothing, null_string = nothing)
    ds = try
        Parquet2.Dataset(String(path))
    catch e
        # Parquet2.jl builds a `Column` for every schema node when it opens the
        # file, and a nested (list/struct/map) node has no column metadata — so
        # such a file cannot be OPENED, let alone have its flat columns read.
        # Say so, rather than leaking a `FieldError` about thrift internals.
        throw(ArgumentError(
            "could not open $(repr(String(path))) as parquet: $(sprint(showerror, e)) " *
            "(note that the Parquet2.jl backend also cannot open a file containing a " *
            "NESTED list/struct/map column, even to read that file's flat columns)"))
    end
    try
        names = String[String(n) for n in Base.names(ds)]
        kinds = Symbol[]
        types = Any[]
        pts = Any[]
        for n in names
            pt = Parquet2.parqtype(ds, n)
            T = Parquet2.juliatype(ds, n)          # already excludes `Missing`
            push!(types, T)
            push!(pts, pt)
            push!(kinds, _parquet_kind(T, _unit_exponent(pt)))
        end
        # A zero-row table is TYPED, not absent: the schema is in the footer, so
        # every column comes back empty with its declared dtype. Parquet2 cannot
        # load a zero-row column chunk (it trips over the page encoding some
        # writers emit for one), and there is nothing there to load anyway.
        nrows = Parquet2.nrow(ds)
        index = Dict(n => j for (j, n) in enumerate(names))

        # Called ONLY for the columns that survive the projection, so a skipped
        # column costs nothing at all — not its chunk, and not even the unit
        # check `_rawconv` does for a temporal one.
        function loadcolumn(name::AbstractString)
            nrows == 0 && return Union{Missing,Int64}[]
            pt = pts[index[name]]
            # A DECIMAL also needs its FLBA width, to repair Parquet2.jl's
            # unsigned reading of a narrow one (`_decimal_sign_fix`). Schema
            # only — no page is touched for it.
            width = pt isa Parquet2.ParqDecimal ? _decimal_flba_width(ds, name) : nothing
            conv = _rawconv(pt, name, width)
            v = try
                Parquet2.load(ds, name)            # this column's chunk, only
            catch e
                # A 7-byte FLBA decimal is the one width the repair cannot
                # reach: the unsigned fold of a negative is 2^56, 17 digits,
                # which overflows `Dec64`'s 16 and throws in here. Say what
                # happened instead of leaking a bare `InexactError`.
                width === nothing || width != 7 || throw(ArgumentError(
                    "column $(repr(name)): the Parquet2.jl backend cannot decode a " *
                    "negative DECIMAL stored in a 7-byte FIXED_LEN_BYTE_ARRAY " *
                    "(pyarrow precision 15-16) — it reads the two's-complement bytes " *
                    "UNSIGNED, and 2^56 does not fit a Dec64's 16 digits. Store the " *
                    "column at precision >= 17 (an 8-byte decimal), or read it in " *
                    "the Python or Rust track. Backend error: $(sprint(showerror, e))"))
                rethrow()
            end
            conv === nothing && return v
            return Any[x === missing ? missing : conv(x) for x in v]
        end

        return _assemble_parquet(names, kinds, types, loadcolumn;
                                 variables = variables, float_columns = float_columns,
                                 null_int = null_int, null_string = null_string)
    finally
        close(ds)
    end
end

end

# The active `parquet` reader (Julia track) — an Apache Parquet file as a flat
# table. Sibling of `rust/tests/parquet_reader.rs`, case for case.
#
# The decode backend is Parquet2.jl, supplied by the `EarthSciIOParquet2Ext`
# weakdep extension (`using Parquet2` below triggers it). The fixtures in
# `test/fixtures/parquet_*.parquet` are written by **pyarrow** — the reference
# Arrow/Parquet implementation, and the peer of the Rust track's `parquet` crate
# — so the two tracks decode files of the same provenance; see
# `test/fixtures/make_parquet_fixtures.py` for how to regenerate them.
#
# What is pinned here is spec/conformance.md §3 "Parquet decode notes": the
# type table (the narrow/wide integer split is the NetCDF reader's, verbatim),
# temporal columns carried as their RAW integer, a categorical expanded, the
# null policy and its two declared gates, `float_columns` including
# decimal-TEXT parsing, a zero-row file being TYPED rather than absent, and
# projection pushdown — proved at the BYTE level, not by inspecting the result.
using EarthSciIO
using Test
using Parquet2          # triggers EarthSciIOParquet2Ext — the decode backend
using Dates: Date, DateTime, Time
using Logging: NullLogger, with_logger

const PQ_FIX = joinpath(@__DIR__, "fixtures")
_pq(name) = joinpath(PQ_FIX, "parquet_$name.parquet")

# A private copy of a fixture, so a test that mutates bytes never touches the
# committed blob.
function _pq_copy(name, dir)
    dst = joinpath(dir, "copy.parquet")
    cp(_pq(name), dst; force = true)
    chmod(dst, 0o644)
    return dst
end

# Something read off a Parquet2 `Dataset`, with the dataset closed afterwards.
function _with_dataset(f, path)
    ds = Parquet2.Dataset(path)
    try
        return f(ds)
    finally
        close(ds)
    end
end

# The `<fixture>/tables/` directory of the MOVES snapshots — from the env
# override or the sibling checkout the downstream `.esm` project uses — and the
# `.parquet` in it whose name ends in `__<suffix>`. `nothing` when absent, so
# the smoke test skips rather than fails outside that checkout.
function _moves_table(suffix::AbstractString)
    roots = String[]
    haskey(ENV, "EARTHSCIIO_MOVES_SNAPSHOTS") && push!(roots, ENV["EARTHSCIIO_MOVES_SNAPSHOTS"])
    # The sibling checkout, from the repo root and from a git worktree of it
    # (one directory deeper).
    for up in ("..", joinpath("..", ".."))
        push!(roots, normpath(joinpath(@__DIR__, "..", "..", up, "moves.rs",
                                       "characterization", "snapshots")))
    end
    for root in roots
        isdir(root) || continue
        dirs = isdir(joinpath(root, "tables")) ? [joinpath(root, "tables")] :
               sort!(String[joinpath(root, d, "tables") for d in readdir(root)
                            if isdir(joinpath(root, d, "tables"))])
        for dir in dirs
            hits = sort!(String[joinpath(dir, f) for f in readdir(dir)
                                if endswith(f, "__" * suffix * ".parquet")])
            isempty(hits) || return hits[1]
        end
    end
    return nothing
end

@testset "parquet reader (Parquet2.jl backend)" begin
    @testset "registered active in the format registry" begin
        @test haskey(EarthSciIO.FORMAT_REGISTRY, "parquet")
        @test EarthSciIO.status_of(EarthSciIO.FORMAT_REGISTRY, "parquet") == :active
        @test EarthSciIO.FORMAT_REGISTRY["parquet"] isa ParquetReader
        # The declared option set IS the extension method's keyword list, so a
        # Provider can reject an unrecognised `reader_kwargs` key at construction.
        @test Set(EarthSciIO.reader_option_keys(ParquetReader())) ==
              Set([:variables, :float_columns, :null_int, :null_string])
        @test_throws ArgumentError EarthSciIO._check_reader_kwargs(
            ParquetReader(), "parquet", Dict{Symbol,Any}(:numeric_columns => ["x"]))
    end

    # The dtype table WITHOUT the backend: `_parquet_kind` is the core's copy of
    # the §3 mapping, shared with any future Parquet backend.
    @testset "the dtype table is the netcdf reader's integer split, verbatim" begin
        k = EarthSciIO._parquet_kind
        @test k(Bool) == :bool
        @test all(k(T) == :int32 for T in (Int8, Int16, Int32, UInt8, UInt16))
        @test all(k(T) == :int64 for T in (Int64, UInt32, UInt64))
        @test all(k(T) == :float64 for T in (Float16, Float32, Float64))
        @test k(String) == :string
        @test k(Missing) == :float64            # an all-null column
        @test k(Vector{UInt8}) == :unsupported  # binary
        @test k(NamedTuple) == :unsupported     # nested
        # Temporal: raw integers at their stored width. Only the unit exponent
        # separates a millisecond Time32 from a microsecond Time64.
        @test k(Date) == :int32
        @test k(DateTime) == :int64
        @test k(Time, -3) == :int32
        @test k(Time, -6) == :int64
        @test k(Time, -9) == :int64
    end

    @testset "every supported type maps onto the native contract" begin
        ds = read_native(ParquetReader(), _pq("types"))
        @test ds isa EarthSciIO.NativeDataset
        @test coord_names(ds) == String[]          # a table has no coordinates
        for name in variable_names(ds)
            @test ds[name].dims == ["index"]
            @test size(ds[name].data) == (3,)
        end

        el(n) = eltype(ds[n].data)
        @test el("b") == Bool
        # The narrow/wide integer split is the NetCDF reader's, verbatim.
        @test el("i16") == Int32
        @test el("i32") == Int32
        @test el("u16") == Int32
        @test el("i64") == Int64
        @test el("u32") == Int64
        @test el("u64") == Int64
        @test el("f32") == Float64
        @test el("f64") == Float64
        @test el("dec") == Float64
        @test el("s") == String
        # A categorical reads as its VALUE type, expanded — never as its key.
        @test el("cat") == String
        # Temporal columns ride as their raw integer, at their stored width.
        @test el("d32") == Int32
        @test el("t32") == Int32
        @test el("t64") == Int64
        @test el("ts") == Int64
        @test el("tsu") == Int64

        @test ds["b"].data == [true, false, true]
        @test ds["i16"].data == Int32[1, -2, 3]
        @test ds["u32"].data == Int64[1, 2, 3]
        @test ds["f32"].data == [1.5, 2.5, 3.5]
        # A leading-zero-safe code column stays text, and an empty cell is "" —
        # an empty string is a VALUE here, not a missing one (that is a null).
        @test ds["s"].data == ["2260000000", "x", ""]
        @test ds["cat"].data == ["gas", "diesel", "gas"]
        # decimal(30,12): unscaled ÷ 10^12.
        @test ds["dec"].data == [261.0, -1.5, 0.0]
        # NOT decoded to an instant: the raw stored value, verbatim.
        @test ds["d32"].data == Int32[19000, 19001, 19002]
        @test ds["ts"].data == Int64[1_700_000_000_000, 0, -5]
        @test ds["t32"].data == Int32[1000, 2000, 3000]
        @test ds["t64"].data == Int64[1000, 2000, 3000]
        @test ds["tsu"].data == Int64[1_700_000_000_000_000, 0, -5000]
        # No sentinel survives into a float column, and no attrs are invented.
        @test isempty(ds["f64"].attrs)
    end

    @testset "a uint64 too large for int64 is an error, not a wraparound" begin
        err = try
            read_native(ParquetReader(), _pq("bigu64"))
            nothing
        catch e
            sprint(showerror, e)
        end
        @test err !== nothing
        @test occursin("\"u\"", err)      # names the column
        @test occursin("row 0", err)      # names the row
    end

    @testset "a binary column is skipped when unrequested, refused when named" begin
        ds = read_native(ParquetReader(), _pq("binary"))
        @test variable_names(ds) == ["id"]      # binary is simply not a field
        @test ds["id"].data == Int64[7, 8]
        err = try
            read_native(ParquetReader(), _pq("binary"); variables = ["blob"])
            nothing
        catch e
            sprint(showerror, e)
        end
        @test err !== nothing && occursin("blob", err)

        # `float_columns` is a statement about how to READ a column, not a claim
        # that an opaque blob is a number: an unrequested binary column stays a
        # NON-FIELD even when named there, rather than registering a field that
        # then hard-errors. (The Rust track shipped that bug and fixed it in
        # `a30c9e4`; the core checks `:unsupported` BEFORE the forced-float
        # override, so this track cannot grow it.)
        ds = read_native(ParquetReader(), _pq("binary"); float_columns = ["blob"])
        @test variable_names(ds) == ["id"]
        @test ds["id"].data == Int64[7, 8]
    end

    # Parquet2.jl builds a column for every schema node when it opens the footer,
    # and a nested node has no column metadata — so a file carrying a
    # list/struct/map column cannot be OPENED at all by this backend, not even to
    # read its flat columns. That is a backend limit, and the reader states it
    # rather than leaking a thrift `FieldError`.
    @testset "a nested column: a clear refusal naming the backend limit" begin
        err = try
            read_native(ParquetReader(), _pq("nested"))
            nothing
        catch e
            sprint(showerror, e)
        end
        @test err !== nothing
        @test occursin("could not open", err)
        @test occursin("NESTED", err)
    end

    @testset "null policy" begin
        f = _pq("nulls")
        # A null FLOAT is NaN — the same fold a CF _FillValue gets — and an
        # all-null (arrow `null`) column is float64 with every cell NaN.
        ds = read_native(ParquetReader(), f; variables = ["f", "nul"])
        @test ds["f"].data[1] == 1.0
        @test isnan(ds["f"].data[2])
        @test ds["f"].data[3] == 3.0
        @test eltype(ds["nul"].data) == Float64
        @test all(isnan, ds["nul"].data)
        @test isempty(ds["f"].attrs)      # no sentinel survives a NaN fold

        # A null in a type with NO missing value is refused by default, and the
        # refusal names the column, the row and the way out.
        for (col, gate) in (("i", "null_int"), ("s", "null_string"))
            err = try
                read_native(ParquetReader(), f; variables = [col])
                nothing
            catch e
                sprint(showerror, e)
            end
            @test err !== nothing
            @test occursin("\"$col\"", err)
            @test occursin("row 1", err)
            @test occursin(gate, err)
        end

        # The declared gates: an integer sentinel is substituted AND reported
        # back as the field's fill_value (an integer sentinel cannot be NaN, so
        # it survives, exactly as a CF integer fill does); a declared string
        # stands in for a null text cell.
        ds = read_native(ParquetReader(), f; variables = ["i", "i32", "s"],
                         null_int = -9, null_string = "NA")
        @test ds["i"].data == Int64[1, -9, 3]
        @test ds["i"].attrs["fill_value"] == -9
        @test eltype(ds["i32"].data) == Int32
        @test ds["i32"].data == Int32[1, -9, 3]
        @test ds["i32"].attrs["fill_value"] === Int64(-9)   # one spelling, both widths
        @test ds["s"].data == ["a", "NA", "c"]

        # A boolean has no such option: a third state is a float64 column.
        err = try
            read_native(ParquetReader(), f; variables = ["bo"],
                        null_int = -9, null_string = "NA")
            nothing
        catch e
            sprint(showerror, e)
        end
        @test err !== nothing
        @test occursin("\"bo\"", err) && occursin("float_columns", err)
    end

    @testset "float_columns: a nullable integer, and floats stored as TEXT" begin
        f = _pq("nulls")
        ds = read_native(ParquetReader(), f; variables = ["i", "dtxt"],
                         float_columns = ["i", "dtxt"])
        # The integer column is float64 now, so its null folds to NaN and no
        # sentinel is reported.
        @test eltype(ds["i"].data) == Float64
        @test ds["i"].data[1] == 1.0 && isnan(ds["i"].data[2]) && ds["i"].data[3] == 3.0
        @test isempty(ds["i"].attrs)
        # Decimal TEXT is trimmed and parsed; an all-whitespace cell is NaN (the
        # FF10/shapefile blank→NaN rule). The MOVES snapshots store floats this
        # way for byte reproducibility.
        @test eltype(ds["dtxt"].data) == Float64
        @test ds["dtxt"].data[1] == 261.0
        @test isnan(ds["dtxt"].data[2])
        @test ds["dtxt"].data[3] == -1.5
        # Without the option the column stays text: the reader never guesses.
        plain = read_native(ParquetReader(), f; variables = ["dtxt"])
        @test plain["dtxt"].data == ["261.000000000000", "   ", "-1.5"]

        # Anything else unparseable is an error naming column, row and text.
        err = try
            read_native(ParquetReader(), f; variables = ["badtxt"],
                        float_columns = ["badtxt"])
            nothing
        catch e
            sprint(showerror, e)
        end
        @test err !== nothing
        @test occursin("\"badtxt\"", err) && occursin("row 1", err) &&
              occursin("not a number", err)
    end

    # The schema lives in the footer, so an empty table is TYPED, not absent.
    # Most of a MOVES fixture's ~770 tables are empty, and a document binding one
    # must still see the array it named.
    # Parquet stores a FIXED_LEN_BYTE_ARRAY decimal as a big-endian TWO'S
    # COMPLEMENT integer. Parquet2.jl's `concat_integer` folds those bytes into
    # an `Int64` with NO sign extension, which is only accidentally right at 8
    # bytes; narrower than that, every NEGATIVE cell came back as its unsigned
    # reinterpretation — `-2.50` in a `decimal128(9, 2)` read as `42949670.46`,
    # a silently wrong number and a straight disagreement with the Rust and
    # Python tracks, which both read the unscaled integer signed.
    # `EarthSciIOParquet2Ext` repairs it exactly: the unsigned fold lands at or
    # above 2^(8w-1) if and only if the true value was negative. One column per
    # FLBA width pyarrow uses, each with its extreme negative.
    @testset "a narrow FLBA decimal keeps its sign" begin
        ds = read_native(ParquetReader(), _pq("decimals"))
        @test ds["d02"].data == [99.0, -99.0, 0.0, -1.0, 1.0]
        @test ds["d04"].data == [99.99, -99.99, 0.0, -0.01, 0.01]
        @test ds["d09"].data == [9999999.99, -9999999.99, 0.0, -0.01, 0.01]
        @test ds["d12"].data == [9999999999.99, -9999999999.99, 0.0, -0.01, 0.01]
        @test ds["d18"].data ==
              [999999.999999, -999999.999999, 0.0, -1.0e-6, 1.0e-6]
        for n in variable_names(ds)
            @test eltype(ds[n].data) == Float64
        end
    end

    # The fixture above reaches the widths pyarrow picks for those precisions —
    # 1, 2, 4, 6 and 8 bytes — which leaves widths 3 and 5 (and every scale but
    # the two it uses) untested, on a repair that had NO passing coverage at all
    # until the `round(Float64, ...)` MethodError above it was fixed. So drive
    # `_decimal_sign_fix`'s closure DIRECTLY over every width the guard admits,
    # at every scale Parquet permits for it, against the extremes where a sign
    # repair can go wrong.
    #
    # The oracle is the OTHER TWO TRACKS' arithmetic, spelled out: Rust and
    # Python both compute `f64(unscaled) / 10.0^scale`, so that expression is
    # what the closure has to reproduce BIT-FOR-BIT (spec/conformance.md §3 —
    # a wrong number is never a permitted divergence). The input is what
    # Parquet2 actually hands over: the `w` bytes folded UNSIGNED into a
    # `Dec64`, i.e. `unscaled + 2^(8w)` for a negative, scaled by `10^-scale`.
    @testset "the FLBA sign repair is exact at every width and scale" begin
        ext = Base.get_extension(EarthSciIO, :EarthSciIOParquet2Ext)
        @test ext !== nothing
        sign_fix = ext._decimal_sign_fix
        # Parquet2's `ParqDecimal` carries the scale as a NEGATIVE exponent, and
        # `_decimal_sign_fix` reads nothing else off it.
        pt(scale) = (scale = -scale,)

        # The largest number of decimal digits a `w`-byte two's-complement FLBA
        # can hold: floor(log10(2^(8w-1) - 1)). Parquet caps a column's
        # precision at this, so it also caps the scale.
        maxprec = [2, 4, 6, 9, 11, 14]

        # `unscaled`/`scale` as an exact `Dec64`, built from the decimal string
        # so no binary arithmetic sits between the integer and the value.
        function dec(unscaled::Integer, scale::Integer)
            digits = string(abs(unscaled))
            if scale > 0
                digits = lpad(digits, scale + 1, '0')
                digits = digits[1:(end - scale)] * "." * digits[(end - scale + 1):end]
            end
            return Parquet2.Dec64((unscaled < 0 ? "-" : "") * digits)
        end

        for width in 1:6
            prec = maxprec[width]
            top = 10^prec - 1              # the extreme |unscaled| at this width
            wrap = Int64(1) << (8 * width) # the unsigned fold's offset
            for scale in 0:prec
                fix = sign_fix(pt(scale), width)
                @test fix !== nothing
                for unscaled in unique(Int64[0, 1, -1, top, -top, top - 1,
                                             -(top - 1), 10^scale, -(10^scale),
                                             div(top, 3), -div(top, 3)])
                    abs(unscaled) > top && continue   # not a legal cell here
                    stored = unscaled < 0 ? unscaled + wrap : unscaled
                    # A negative's unsigned fold must still fit `Dec64`'s 16
                    # digits — the whole reason the guard stops at 7 bytes.
                    @test ndigits(stored) <= 16
                    @test fix(dec(stored, scale)) === Float64(unscaled) / 10.0^scale
                end
            end
        end

        # The guard's other side. 7 bytes is beyond reach (2^56 overflows a
        # `Dec64`, so Parquet2 throws first — the testset below pins the message);
        # 8 bytes and wider never needed a repair, and applying one there would
        # CREATE the wrongness it exists to remove. All of them must decline.
        for width in (nothing, 7, 8, 9, 16)
            @test sign_fix(pt(2), width) === nothing
        end
    end

    # The ONE width the repair cannot reach: a 7-byte FLBA (pyarrow precision
    # 15-16). The unsigned fold of a negative is 2^56 — 17 decimal digits, which
    # overflows a `Dec64`'s 16 — so Parquet2 throws during page decode, before
    # any repair could run. Positives are unaffected; a negative gets a message
    # naming the limit rather than a bare `InexactError`.
    @testset "a 7-byte FLBA decimal names the backend limit" begin
        ds = read_native(ParquetReader(), _pq("decimal7"); variables = ["pos"])
        @test ds["pos"].data == [1.25, 2.5]
        err = try
            with_logger(NullLogger()) do
                read_native(ParquetReader(), _pq("decimal7"); variables = ["neg"])
            end
            nothing
        catch e
            sprint(showerror, e)
        end
        @test err !== nothing
        @test occursin("\"neg\"", err)
        @test occursin("7-byte", err)
    end

    @testset "a zero-row table yields typed empty columns" begin
        ds = read_native(ParquetReader(), _pq("empty"))
        @test variable_names(ds) == ["code", "id", "val"]
        @test eltype(ds["id"].data) == Int64
        @test eltype(ds["code"].data) == String
        @test eltype(ds["val"].data) == Float64
        for name in variable_names(ds)
            @test ds[name].dims == ["index"]
            @test isempty(ds[name].data)
        end
    end

    @testset "variables select columns; an unknown name is refused" begin
        # An EMPTY list is "every column", not "no columns" — the same reading
        # the Python and Rust tracks give it (spec/conformance.md §3).
        every = read_native(ParquetReader(), _pq("empty"); variables = String[])
        @test variable_names(every) == ["code", "id", "val"]

        ds = read_native(ParquetReader(), _pq("types"); variables = ["s", "i64"])
        @test variable_names(ds) == ["i64", "s"]
        @test ds["i64"].data == Int64[100, -200, 300]
        err = try
            read_native(ParquetReader(), _pq("types"); variables = ["i64", "nope"])
            nothing
        catch e
            sprint(showerror, e)
        end
        @test err !== nothing
        @test occursin("nope", err)      # names what is missing
        @test occursin("i64", err)       # lists what is present
    end

    # Projection pushdown is REAL, not read-then-discard, and this proves it at
    # the byte level rather than by inspecting the result.
    #
    # A two-column SNAPPY fixture is copied, and the bytes of one column's chunk
    # — located from the file's OWN metadata — are scribbled over in place. The
    # footer is untouched, so the file still opens and its schema is intact, but
    # any reader that actually fetches and decompresses that chunk must fail.
    # Reading both columns fails; reading only the other column succeeds and
    # returns correct values. The skipped chunk was therefore never read, which
    # is the whole point on tables dozens of columns wide where a document wants
    # three.
    @testset "projection never reads the column chunks it skips" begin
        mktempdir() do dir
            path = _pq_copy("pushdown", dir)
            lo, hi = _with_dataset(path) do ds
                c = Parquet2.Column(Parquet2.RowGroup(ds, 1), "poison")
                (Parquet2.startindex(c), Parquet2.endindex(c))
            end
            @test hi > lo               # the poison chunk has bytes to corrupt

            bytes = read(path)
            for i in lo:hi
                bytes[i] = ~bytes[i]
            end
            write(path, bytes)

            # Sanity: the footer survived, so a failure below is about the CHUNK.
            cols = _with_dataset(ds -> String[String(n) for n in Base.names(ds)], path)
            @test cols == ["keep", "poison"]

            # Reading everything must touch the poisoned chunk and fail.
            poisoned() = with_logger(NullLogger()) do
                read_native(ParquetReader(), path)
            end
            @test_throws Exception poisoned()

            # Reading only `keep` must never touch it.
            ds = read_native(ParquetReader(), path; variables = ["keep"])
            @test variable_names(ds) == ["keep"]
            @test ds["keep"].data == collect(Int64, 0:511)
        end
    end

    # The same proof one level up, through the PROVIDER — because that is where
    # a document's `variables` actually enter. A Provider that read every column
    # and then selected would decode the poisoned chunk and fail; one that
    # pushes the projection into the reader never touches it. (The Python and
    # Rust tracks push it down through the same seam.)
    @testset "a Provider's variables reach the reader as a projection" begin
        mktempdir() do dir
            path = _pq_copy("pushdown", dir)
            lo, hi = _with_dataset(path) do ds
                c = Parquet2.Column(Parquet2.RowGroup(ds, 1), "poison")
                (Parquet2.startindex(c), Parquet2.endindex(c))
            end
            bytes = read(path)
            for i in lo:hi
                bytes[i] = ~bytes[i]
            end
            write(path, bytes)

            cachedir = joinpath(dir, "cache")
            mkpath(cachedir)
            provider = const_provider(Cache(LocalStore(cachedir)), "file://" * path;
                                      format = "parquet", variables = ["keep"])
            nds = with_logger(NullLogger()) do
                materialize(provider)
            end
            @test variable_names(nds) == ["keep"]
            @test nds["keep"].data == collect(Int64, 0:511)

            # ...and an EMPTY `variables` on the Provider is "every variable",
            # not "no variables" — it used to select nothing at all.
            all_p = const_provider(Cache(LocalStore(cachedir)),
                                   "file://" * _pq("empty");
                                   format = "parquet", variables = String[])
            @test variable_names(materialize(all_p)) == ["code", "id", "val"]
        end
    end

    @testset "a non-parquet blob is an error, not a crash" begin
        mktempdir() do dir
            path = joinpath(dir, "not.parquet")
            write(path, "definitely not parquet")
            @test_throws ArgumentError read_native(ParquetReader(), path)
        end
    end

    # A real MOVES table: int64 ID columns, an `SCC` code that must NOT become a
    # number, and a rate stored as fixed-decimal TEXT. The snapshots are
    # gigabytes and live outside this repo, so the test locates them and skips
    # when they are absent — it can never be the only thing covering a behaviour.
    # Point `EARTHSCIIO_MOVES_SNAPSHOTS` at a directory of `<fixture>/tables/`
    # to run it elsewhere.
    @testset "MOVES snapshot smoke" begin
        path = _moves_table("nremissionrate")
        if path === nothing
            @info "skipping MOVES snapshot smoke: no nremissionrate table found"
        else
            whole = read_native(ParquetReader(), path)
            n = length(whole["SCC"].data)
            @test n > 0
            for name in variable_names(whole)
                @test whole[name].dims == ["index"]
                @test length(whole[name].data) == n
            end
            # The snapshots write every column as int64 or text — floats
            # included, as fixed-decimal text — so nothing decodes as a native
            # float until a document says `float_columns`.
            @test eltype(whole["polProcessID"].data) == Int64
            @test eltype(whole["SCC"].data) == String
            @test eltype(whole["meanBaseRate"].data) == String

            # Now the way a document actually reads it: three columns of nine,
            # with the decimal-text rate declared.
            want = ["polProcessID", "SCC", "meanBaseRate"]
            ds = read_native(ParquetReader(), path; variables = want,
                             float_columns = ["meanBaseRate"])
            @test variable_names(ds) == sort(want)
            @test eltype(ds["meanBaseRate"].data) == Float64
            @test length(ds["SCC"].data) == n          # the projection keeps every row
            @test all(x -> isfinite(x) && x >= 0, ds["meanBaseRate"].data)
            for i in (1, n ÷ 2, n)
                @test ds["meanBaseRate"].data[i] ==
                      parse(Float64, strip(whole["meanBaseRate"].data[i]))
            end
            @test ds["SCC"].data == whole["SCC"].data  # no reordering
        end
    end
end

# Format-reader decode parity — conformance checks 3 (decode) + 4 (native-array
# equality), the half component (a)'s test_conformance.jl left to component (b)
# (esio-9nb.5). This is the Julia mirror of conformance/verify.py: open each
# committed corpus blob with the FORMAT_REGISTRY reader its case names, CF-decode
# per spec/conformance.md §3, and assert the native arrays equal the case's
# `expected` (the Python/xarray oracle) within the spec's tolerances.

# --- comparison helpers (mirror verify.py _flat / _cmp_numeric / _cmp_string) -

# Recursive flatten of a nested `expected.data` array (C/row-major order).
function _flat(x)
    out = Any[]
    rec(y) = y isa AbstractVector ? foreach(rec, y) : push!(out, y)
    rec(x)
    return out
end

# C-order (row-major) flatten of a native array whose axes are in file (`dims`)
# order — matches numpy `.reshape(-1)` on xarray's `.values`. NCDatasets/Julia
# are column-major, so reverse the dims before `vec`.
_corder(a::AbstractVector) = collect(a)
_corder(a::AbstractArray) = vec(permutedims(a, reverse(1:ndims(a))))
# A 0-dimensional field — a netcdf SCALAR string, `char label(strlen)` on a
# private dimension — has no axes to permute, and needs its own method:
# `permutedims` with an EMPTY permutation throws on Julia 1.10 (CI's version)
# while working on 1.12, so without this the suite passes locally and fails there.
_corder(a::AbstractArray{<:Any,0}) = collect(vec(a))

const READER_ATOL = 1e-6
const READER_RTOL = 1e-9

# Returns `nothing` on match, else an error string (so the testset can @test it).
function cmp_native_numeric(got::AbstractArray, expected_nested)
    g = Float64[ismissing(x) ? NaN : Float64(x) for x in _corder(got)]
    e = Float64[v === nothing ? NaN : Float64(v) for v in _flat(expected_nested)]
    length(g) == length(e) || return "shape $(length(g)) != expected $(length(e))"
    gn, en = isnan.(g), isnan.(e)
    gn == en || return "NaN/fill mask mismatch"
    keep = .!gn
    if !all(isapprox.(g[keep], e[keep]; atol = READER_ATOL, rtol = READER_RTOL))
        return "value mismatch (max abs diff $(maximum(abs.(g[keep] .- e[keep]); init = 0.0)))"
    end
    return nothing
end

function cmp_native_string(got, expected_nested)
    g = String[string(x) for x in _corder(collect(got))]
    e = String[string(v) for v in _flat(expected_nested)]
    g == e || return "string mismatch $g != $e"
    return nothing
end

# The native-field schema's `dtype` ↔ the Julia element type the reader returns.
function dtype_ok(data, dt::AbstractString)
    dt == "string"  && return eltype(data) <: AbstractString
    dt == "float64" && return eltype(data) == Float64
    dt == "int32"   && return eltype(data) == Int32
    dt == "int64"   && return eltype(data) == Int64
    dt == "bool"    && return eltype(data) == Bool
    return true
end

# --- the decode conformance pass -------------------------------------------

@testset "format readers — decode + native-array equality (checks 3–4)" begin
    index = JSON.parsefile(joinpath(CORPUS, "cases.json"))
    @test length(index["cases"]) >= 1

    for entry in index["cases"]
        case = JSON.parsefile(joinpath(CORPUS, entry["file"]))
        id = case["id"]
        fmt = case["format"]
        blob = joinpath(CORPUS, case["blob_path"])

        @testset "$id ($fmt)" begin
            # the reader is resolved by name through the registry — dispatch is
            # the architectural seam (a new format is one register! line).
            @test haskey(FORMAT_REGISTRY, fmt)
            @test status_of(FORMAT_REGISTRY, fmt) == :active
            reader = FORMAT_REGISTRY[fmt]

            if store_backed(reader)
                # Store-backed (zarr): the reader is handed (cache, base_url;
                # variables, select) — a Zarr store is many objects, not one blob.
                zcache = Cache(LocalStore(joinpath(CORPUS, "cache")); offline = true, verify = true)
                vars = String[String(v) for v in case["variables"]]
                nds = read_store(reader, zcache, case["resolved_url"];
                                 variables = vars, select = case["select"])
            else
                kwargs = if fmt == "csv"
                    (; numeric_columns = String.(case["decode"]["numeric_columns"]))
                elseif fmt == "ff10"
                    # forward the zip member selection + header handling the case
                    # pins (member/members/member_glob/skip_header_row).
                    dec = case["decode"]
                    kw = Dict{Symbol,Any}(
                        :numeric_columns => String.(dec["numeric_columns"]),
                        :kind => String(get(dec, "kind", "point")),
                        :member => get(dec, "member", nothing),
                    )
                    ms = get(dec, "members", nothing)
                    ms === nothing || (kw[:members] = String.(ms))
                    mg = get(dec, "member_glob", nothing)
                    mg === nothing || (kw[:member_glob] = String(mg))
                    shr = get(dec, "skip_header_row", false)
                    kw[:skip_header_row] = shr === nothing ? false : Bool(shr)
                    (; kw...)
                elseif fmt == "shapefile"
                    # forward the `.shp` member inside the zip blob + the text
                    # code column the case wants as a number.
                    dec = case["decode"]
                    kw = Dict{Symbol,Any}()
                    m = get(dec, "member", nothing)
                    m === nothing || (kw[:member] = String(m))
                    nc = get(dec, "numeric_columns", nothing)
                    nc === nothing || (kw[:numeric_columns] = String.(nc))
                    (; kw...)
                elseif fmt == "parquet"
                    # forward the case's PROJECTION (`variables`, pushed into the
                    # decode) plus `float_columns` and the two null gates — a
                    # columnar case decodes to a different field set and a
                    # different dtype without them.
                    dec = case["decode"]
                    kw = Dict{Symbol,Any}()
                    vars = get(case, "variables", nothing)
                    vars === nothing || (kw[:variables] = String.(vars))
                    fc = get(dec, "float_columns", nothing)
                    fc === nothing || (kw[:float_columns] = String.(fc))
                    ni = get(dec, "null_int", nothing)
                    ni === nothing || (kw[:null_int] = Int64(ni))
                    ns = get(dec, "null_string", nothing)
                    ns === nothing || (kw[:null_string] = String(ns))
                    (; kw...)
                elseif fmt == "netcdf"
                    # A netcdf case may pin a DECODE-TIME window: same blob, same
                    # cache key, only the requested hyperslab materialised. A case
                    # without `axes` (the `{all_records: true}` form) reads whole.
                    sel = get(case, "select", nothing)
                    axes = sel === nothing ? nothing : get(sel, "axes", nothing)
                    axes === nothing ? NamedTuple() : (; select = sel)
                else
                    NamedTuple()
                end
                nds = read_native(reader, blob; kwargs...)
            end

            for (name, spec) in case["expected"]["variables"]
                @test haskey(nds, name)
                field = nds[name]
                @test dtype_ok(field.data, spec["dtype"])
                err = spec["dtype"] == "string" ?
                      cmp_native_string(field.data, spec["data"]) :
                      cmp_native_numeric(field.data, spec["data"])
                err === nothing || @info "decode mismatch" id name err
                @test err === nothing
            end

            for (name, spec) in case["expected"]["coords"]
                @test haskey(nds, name)
                field = nds[name]
                @test dtype_ok(field.data, spec["dtype"])
                @test cmp_native_numeric(field.data, spec["data"]) === nothing
                # a CF time axis stays RAW with units/calendar carried (not decoded)
                if haskey(spec, "units")
                    @test field.attrs["units"] == spec["units"]
                end
                if haskey(spec, "calendar")
                    @test field.attrs["calendar"] == spec["calendar"]
                end
            end
        end
    end
end

@testset "netcdf reader — `variables` projection pushdown" begin
    era5_case = JSON.parsefile(joinpath(CORPUS, "cases", "era5-grid-sub-tile.json"))
    blob = joinpath(CORPUS, era5_case["blob_path"])
    reader = FORMAT_REGISTRY["netcdf"]

    full = read_native(reader, blob)
    @test Set(variable_names(full)) == Set(["t2m", "sp"])

    # The projection is field-for-field identical to decode-then-select — the
    # acceptance gate: a pushed-down read may be cheaper, never different.
    one = read_native(reader, blob; variables = ["t2m"])
    @test variable_names(one) == ["t2m"]
    # `isequal`, not `==`: t2m carries a masked cell decoded to NaN, and
    # NaN != NaN would make an identical array compare unequal.
    @test isequal(one["t2m"].data, full["t2m"].data)
    @test one["t2m"].dims == full["t2m"].dims
    @test one["t2m"].attrs == full["t2m"].attrs
    # coords are ALWAYS kept (they are the grid the array lives on), attrs and all
    @test coord_names(one) == coord_names(full)
    for c in coord_names(full)
        @test isequal(one[c].data, full[c].data)
        @test one[c].dims == full[c].dims
        @test one[c].attrs == full[c].attrs
    end

    # An EMPTY list reads every variable, NOT none (spec/conformance.md §3).
    @test variable_names(read_native(reader, blob; variables = String[])) ==
          variable_names(full)
    @test variable_names(read_native(reader, blob; variables = nothing)) ==
          variable_names(full)

    # An absent name is an error naming what IS present, never a missing array.
    err = try
        read_native(reader, blob; variables = ["t2m", "nope"])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("nope", sprint(showerror, err))
    @test occursin("sp", sprint(showerror, err))

    # Declaring the keyword is what makes the Provider push it down: the option
    # set is READ OFF the method, and `_load` only merges `variables` into the
    # reader call when the reader declares it.
    @test :variables in reader_option_keys(reader)
    cache = Cache(LocalStore(joinpath(CORPUS, "cache")); offline = true, verify = true)
    p = const_provider(cache, era5_case["resolved_url"];
                       format = "netcdf", variables = ["t2m"])
    pushed = materialize(p)
    @test variable_names(pushed) == ["t2m"]
    @test isequal(pushed["t2m"].data, full["t2m"].data)
    @test coord_names(pushed) == coord_names(full)
end

@testset "netcdf reader — decode-time `select` (windowed read)" begin
    era5_case = JSON.parsefile(joinpath(CORPUS, "cases", "era5-grid-sub-tile.json"))
    blob = joinpath(CORPUS, era5_case["blob_path"])
    reader = FORMAT_REGISTRY["netcdf"]
    full = read_native(reader, blob)

    # The acceptance gate: a windowed read equals the FULL read sliced afterwards,
    # cell for cell — anything else is an off-by-one. `isequal` because the corpus
    # blob carries a masked cell that decodes to NaN.
    sel = Dict("axes" => Any["all", Dict("slice" => [1, 3]), Dict("indices" => [0, 2])])
    w = read_native(reader, blob; select = sel)
    for name in ("t2m", "sp")
        @test isequal(w[name].data, full[name].data[:, 2:3, [1, 3]])
        @test w[name].dims == full[name].dims          # dims are NAMES, not lengths
        @test w[name].attrs == full[name].attrs
    end

    # Coordinates are sliced WITH the data — a windowed variable beside a
    # full-length lon/lat would be a silent trap.
    @test w["latitude"].data == full["latitude"].data[2:3]
    @test w["longitude"].data == full["longitude"].data[[1, 3]]
    @test w["time"].data == full["time"].data          # the time axis is untouched
    @test w["time"].attrs == full["time"].attrs        # raw, with units + calendar

    # An explicit index list is returned in the ORDER GIVEN (the zarr reader's
    # rule): a reader that sorted the indices, or read a bounding slab and forgot
    # to gather, fails here.
    perm = read_native(reader, blob;
                       select = Dict("axes" => Any["all", "all", Dict("indices" => [2, 0])]))
    @test isequal(perm["t2m"].data, full["t2m"].data[:, :, [3, 1]])
    @test perm["longitude"].data == full["longitude"].data[[3, 1]]

    # `variables` and `select` compose: project, then window.
    both = read_native(reader, blob; variables = ["t2m"], select = sel)
    @test variable_names(both) == ["t2m"]
    @test isequal(both["t2m"].data, full["t2m"].data[:, 2:3, [1, 3]])

    # Time is the PROVIDER's axis (it owns the cadence), so a time selection is
    # refused rather than quietly honoured.
    e = try
        read_native(reader, blob;
                    select = Dict("axes" => Any[Dict("indices" => [0]), "all", "all"]))
        nothing
    catch err
        err
    end
    @test e isa ArgumentError
    @test occursin("time", sprint(showerror, e))

    # An axis count matching no array is an error, not a silently ignored select.
    @test_throws ArgumentError read_native(reader, blob;
                                           select = Dict("axes" => Any["all", "all"]))

    # A FULL-EXTENT selection must be the no-selection read, exactly — both
    # spellings of "everything" still go down the windowed code path, so this is
    # the read that catches an off-by-one in the slab planner.
    for axes in (Any["all", Dict("slice" => [0, 3]), Dict("slice" => [0, 3])],
                 Any["all", Dict("indices" => [0, 1, 2]), Dict("indices" => [0, 1, 2])])
        everything = read_native(reader, blob; select = Dict("axes" => axes))
        for name in ("t2m", "sp")
            @test size(everything[name].data) == size(full[name].data)
            @test isequal(everything[name].data, full[name].data)
        end
        for name in ("latitude", "longitude", "time")
            @test everything[name].data == full[name].data
        end
    end

    # This vocabulary NEVER drops a dimension: a one-index axis comes back at
    # length 1, not squeezed away (a track that squeezed would diverge in rank).
    for ax in (Dict("indices" => [1]), Dict("slice" => [1, 2]))
        one = read_native(reader, blob; select = Dict("axes" => Any["all", ax, "all"]))
        @test size(one["t2m"].data) == (2, 1, 3)
        @test one["t2m"].dims == ["time", "latitude", "longitude"]
        @test size(one["latitude"].data) == (1,)
        @test isequal(one["t2m"].data, full["t2m"].data[:, 2:2, :])
    end

    # An axis may legally resolve to NOTHING: a zero-length axis, KEPT in `dims`.
    # `extrema` of an empty index list would throw, and the sibling tracks return
    # a correctly-shaped empty array here.
    for ax in (Dict("indices" => Int[]), Dict("slice" => [1, 1]), Dict("slice" => [2, 0]))
        none = read_native(reader, blob; select = Dict("axes" => Any["all", ax, "all"]))
        for name in ("t2m", "sp")
            @test size(none[name].data) == (2, 0, 3)
            @test none[name].dims == ["time", "latitude", "longitude"]
        end
        @test size(none["latitude"].data) == (0,)
        # ...and the axes NOT selected keep their full length.
        @test size(none["longitude"].data) == (3,)
        @test size(none["time"].data) == (2,)
    end

    # A slice is bounds-checked like an `indices` list: an over-long window is an
    # error, never a silent clamp, and a negative bound never wraps around.
    for ax in (Dict("slice" => [1, 99]), Dict("slice" => [-2, 3]), Dict("indices" => [5]))
        e = try
            read_native(reader, blob; select = Dict("axes" => Any["all", ax, "all"]))
            nothing
        catch err
            err
        end
        @test e !== nothing
        @test occursin("out of range", sprint(showerror, e))
    end

    # Through the Provider, per-call and baked, on a reader that is NOT
    # store-backed: same blob, same cache key, only the hyperslab materialised.
    cache = Cache(LocalStore(joinpath(CORPUS, "cache")); offline = true, verify = true)
    p = const_provider(cache, era5_case["resolved_url"]; format = "netcdf")
    @test supports_selection(p)
    @test !store_backed(p)
    m = materialize(p; select = sel)
    @test isequal(m["t2m"].data, full["t2m"].data[:, 2:3, [1, 3]])
    @test m["latitude"].data == full["latitude"].data[2:3]
    # ...and the per-call select is a peek: the next plain read is the full array.
    @test size(materialize(p)["t2m"].data) == size(full["t2m"].data)

    pb = const_provider(cache, era5_case["resolved_url"]; format = "netcdf",
                        reader_kwargs = (; select = sel))
    @test isequal(materialize(pb)["t2m"].data, full["t2m"].data[:, 2:3, [1, 3]])
    # a per-call select OVERRIDES the baked one
    @test size(materialize(pb; select = Dict("axes" => Any["all", "all", "all"]))["t2m"].data) ==
          size(full["t2m"].data)
end

import NCDatasets   # authors the NC_STRING fixture; also read directly below

# --- NetCDF TEXT variables (spec/conformance.md §3) -------------------------
#
# A `char` array is not a native array of characters. Its LAST dimension is the
# string length exactly when nothing else claims it as an axis (no coordinate
# variable of its own, and every variable using it is a `char` using it last) —
# xarray's `conventions.stackable`, which gates its `CharacterArrayCoder`. So the
# same spelling means two different things depending on the rest of the FILE, and
# NCDatasets hands back a raw `Char` array of the on-disk shape either way.
#
# The four blobs below are byte-for-byte the CDF-1 fixtures of
# `rust/src/format/netcdf.rs` (`CHAR_VAR_CDF1`, `CHAR_HOLES_CDF1`,
# `STRING_ROWS_CDF1`, `SCALAR_STRING_CDF1`), so the two tracks are held to the
# same bytes here and not merely to the same prose. The expectations are xarray's
# own output for those bytes (`xr.open_dataset(decode_times=False,
# mask_and_scale=True)`), which spec/conformance.md §3 makes the reference.

# dim `n=3`; `float value(n)` and `char label(n) = "abc"`. `n` is a REAL axis
# (`value` lives on it), so `label` is three ONE-character strings.
const CHAR_VAR_CDF1 = hex2bytes(
    "43444601000000000000000a00000001" *
    "000000016e0000000000000300000000" *
    "000000000000000b0000000200000005" *
    "76616c75650000000000000100000000" *
    "0000000000000000000000050000000c" *
    "0000007c000000056c6162656c000000" *
    "00000001000000000000000000000000" *
    "0000000200000004000000883f800000" *
    "400000004040000061626300")

# The same file with an interior NUL: `char label(n) = "a\0c"`. A NUL cell is the
# EMPTY string (numpy's `|S1` of a NUL byte is `b""`), never `"\0"`.
const CHAR_HOLES_CDF1 = hex2bytes(
    "43444601000000000000000a00000001" *
    "000000016e0000000000000300000000" *
    "000000000000000b0000000200000005" *
    "76616c75650000000000000100000000" *
    "0000000000000000000000050000000c" *
    "0000007c000000056c6162656c000000" *
    "00000001000000000000000000000000" *
    "0000000200000004000000883f800000" *
    "400000004040000061006300")

# dims `n=4` and a PRIVATE `strlen=4`; `float value(n)` and `char label(n,
# strlen)` = "ab", "cd  ", "efgh", "". Nothing but `label` uses `strlen`, so
# `strlen` is the string length: four strings on `dims == ["n"]`.
const STRING_ROWS_CDF1 = hex2bytes(
    "43444601000000000000000a00000002" *
    "000000016e0000000000000400000006" *
    "7374726c656e00000000000400000000" *
    "000000000000000b0000000200000005" *
    "76616c75650000000000000100000000" *
    "00000000000000000000000500000010" *
    "00000090000000056c6162656c000000" *
    "00000002000000000000000100000000" *
    "000000000000000200000010000000a0" *
    "3f800000400000004040000040800000" *
    "61620000636420206566676800000000")

# dims `n=3` and a private `strlen=5`; `float value(n)` and a ONE-dimensional
# `char label(strlen)` = "hi". Its only dimension is the length, so the field is
# a SCALAR string: `dims == []`.
const SCALAR_STRING_CDF1 = hex2bytes(
    "43444601000000000000000a00000002" *
    "000000016e0000000000000300000006" *
    "7374726c656e00000000000500000000" *
    "000000000000000b0000000200000005" *
    "76616c75650000000000000100000000" *
    "0000000000000000000000050000000c" *
    "0000008c000000056c6162656c000000" *
    "00000001000000010000000000000000" *
    "0000000200000008000000983f800000" *
    "40000000404000006869000000000000")

# Run `f` over a temp file holding `bytes` (the corpus blobs are extension-less,
# and so are these — the reader is handed a path, never a name to sniff).
function _with_nc_blob(f, bytes)
    mktempdir() do dir
        path = joinpath(dir, "blob.nc")
        write(path, bytes)
        return f(path)
    end
end

@testset "netcdf reader — text variables are `string` fields (spec §3)" begin
    reader = FORMAT_REGISTRY["netcdf"]

    # A `char label(n)` beside a `float value(n)`: `n` counts elements, so every
    # cell is its own one-character string. xarray: `dims=('n',) shape=(3,) |S1`
    # holding `[b'a', b'b', b'c']`.
    _with_nc_blob(CHAR_VAR_CDF1) do path
        nds = read_native(reader, path)
        # Read-everything RETURNS the text variable: a track that skipped it hands
        # back a different set of fields for the same bytes, which is a divergence.
        @test variable_names(nds) == ["label", "value"]
        f = nds["label"]
        @test eltype(f.data) <: AbstractString
        @test f.dims == ["n"]
        @test size(f.data) == (3,)
        @test f.data == ["a", "b", "c"]
    end

    # ...and a NUL cell in such a variable is the EMPTY string, never `"\0"`.
    # xarray: `[b'a', b'', b'c']`.
    _with_nc_blob(CHAR_HOLES_CDF1) do path
        f = read_native(reader, path)["label"]
        @test f.dims == ["n"]
        @test f.data == ["a", "", "c"]
    end

    # A PRIVATE last dimension is the string length and is CONSUMED: four strings
    # on `dims == ["n"]`, not a 4x4 `Char` matrix. Trailing NULs are stripped and
    # trailing SPACES are data, so `"cd  "` survives whole and an all-NUL row is
    # `""`. xarray: `dims=('n',) shape=(4,) |S4` = `[b'ab', b'cd  ', b'efgh', b'']`.
    _with_nc_blob(STRING_ROWS_CDF1) do path
        f = read_native(reader, path)["label"]
        @test f.dims == ["n"]
        @test size(f.data) == (4,)
        @test f.data == ["ab", "cd  ", "efgh", ""]
    end

    # The 1-D case of the same rule: one string, and therefore a SCALAR field —
    # `dims` and `size` both empty. xarray: `dims=() shape=() |S5` = `b'hi'`.
    _with_nc_blob(SCALAR_STRING_CDF1) do path
        f = read_native(reader, path)["label"]
        @test f.dims == String[]
        @test size(f.data) == ()
        @test f.data[] == "hi"
    end

    # A NetCDF-4 `NC_STRING` is already one string per element: `dims`/`shape` are
    # the variable's own, nothing is consumed. xarray: `dims=('n',) <U5`.
    mktempdir() do dir
        path = joinpath(dir, "nc4strings.nc")
        NCDatasets.NCDataset(path, "c") do ds
            NCDatasets.defDim(ds, "n", 3)
            NCDatasets.defVar(ds, "label", String, ("n",))[:] = ["alpha", "be", "gamma"]
            NCDatasets.defVar(ds, "value", Float64, ("n",))[:] = [1.0, 2.0, 3.0]
        end
        f = read_native(reader, path)["label"]
        @test f.dims == ["n"]
        @test f.data == ["alpha", "be", "gamma"]
    end

    # A text variable is PROJECTABLE by name like any other field, and a typo is
    # an error listing what is present — never a silently missing array.
    _with_nc_blob(STRING_ROWS_CDF1) do path
        one = read_native(reader, path; variables = ["label"])
        @test variable_names(one) == ["label"]
        @test one["label"].data == ["ab", "cd  ", "efgh", ""]
        @test_throws ArgumentError read_native(reader, path; variables = ["nope"])
    end
end

@testset "netcdf reader — a string LENGTH is not an axis a `select` can bind to" begin
    reader = FORMAT_REGISTRY["netcdf"]

    # A selection is applied by dimension NAME to every array, text included: a
    # `char label(n)` must lose the same cells `value(n)` loses.
    _with_nc_blob(STRING_ROWS_CDF1) do path
        full = read_native(reader, path)
        w = read_native(reader, path; select = Dict("axes" => Any[Dict("indices" => [0, 2])]))
        @test w["label"].dims == ["n"]
        @test w["label"].data == ["ab", "efgh"]
        @test w["value"].data == full["value"].data[[1, 3]]

        # The positional rank match is over the DECODED field's dims, not the
        # on-disk ones. `char label(n, strlen)` is a RANK-1 field on TWO on-disk
        # dimensions, so a 2-axis select matches nothing here and says so. Counting
        # on-disk dims instead would let `label` answer for rank 2, bind axis 1 to a
        # string LENGTH, and then apply that selector BY NAME to every other array
        # in the blob — here it would hand back "ab"/"cd"/"ef" and call it a window.
        e = try
            read_native(reader, path;
                        select = Dict("axes" => Any["all", Dict("indices" => [0, 1])]))
            nothing
        catch err
            err
        end
        @test e isa ArgumentError
        @test occursin("no variable in the blob has rank 2", sprint(showerror, e))
    end

    # The same hazard's other face: here `value(n)` and `char label(strlen)` are
    # both rank 1 ON DISK, but `label` decodes to a SCALAR. A single-axis select
    # must bind `n` alone; binding `strlen` too would truncate "hi" to "h" — a
    # wrong STRING, invisible to any shape assertion.
    _with_nc_blob(SCALAR_STRING_CDF1) do path
        full = read_native(reader, path)
        w = read_native(reader, path; select = Dict("axes" => Any[Dict("indices" => [0, 2])]))
        @test w["value"].data == full["value"].data[[1, 3]]
        @test w["label"].dims == String[]
        @test w["label"].data[] == "hi"
    end

    # The guard on that invariant. `_netcdf_dim_selection` can no longer produce a
    # consumed dimension's name, so the refusal is exercised directly: naming a
    # string length is an ERROR, not a no-op — honouring it would slice characters
    # off every string, and ignoring it would hand back the full array while the
    # caller believes they asked for a window.
    _with_nc_blob(STRING_ROWS_CDF1) do path
        NCDatasets.NCDataset(path, "r") do ds
            v = ds["label"]
            @test EarthSciIO._netcdf_consumed_dim(ds, v) == "strlen"
            @test EarthSciIO._netcdf_field_dims(ds, v) == ["n"]
            # ...while a variable whose last dimension is a real axis consumes none.
            @test EarthSciIO._netcdf_consumed_dim(ds, ds["value"]) === nothing
            data = EarthSciIO._netcdf_text_data(ds, v)
            e = try
                EarthSciIO._netcdf_text_select(ds, v, ["n"], data, Dict("strlen" => [0, 1]))
                nothing
            catch err
                err
            end
            @test e isa ArgumentError
            @test occursin("string LENGTH", sprint(showerror, e))
            # a selector on a REAL axis of the field still applies, of course
            @test EarthSciIO._netcdf_text_select(ds, v, ["n"], data,
                                                 Dict("n" => [0, 2])) == ["ab", "efgh"]
        end
    end
end

# A callable STRUCT in the `records["indices"]` position — the resolver form spelled
# without a closure. Top level, because a struct cannot be defined inside a
# `@testset begin` block.
struct LastRecord end
(::LastRecord)(len::Integer) = [len - 1]

@testset "netcdf reader — `records` pushdown (the cadence owner's record axis)" begin
    era5_case = JSON.parsefile(joinpath(CORPUS, "cases", "era5-grid-sub-tile.json"))
    blob = joinpath(CORPUS, era5_case["blob_path"])
    reader = FORMAT_REGISTRY["netcdf"]
    full = read_native(reader, blob)                       # 2 records (hours 0, 1)

    # `dim_length` is the metadata-only half of the pushdown: the record index a
    # Provider wants is `mod1(tick, len)`, so `len` has to be knowable BEFORE the
    # decode it is meant to narrow. It reads the header, never an array.
    @test dim_length(reader, blob, "time") == 2
    @test dim_length(reader, blob, "latitude") == 3
    @test dim_length(reader, blob, "no-such-dim") === nothing
    # every other reader inherits "I cannot answer that"
    @test dim_length(FORMAT_REGISTRY["csv"], blob, "time") === nothing
    # ...and it is declared, so the Provider knows it may push records down
    @test :records in reader_option_keys(reader)

    # The acceptance gate: a record-selected read equals the FULL read sliced
    # afterwards, cell for cell, with dims/attrs untouched and the record axis
    # RETAINED at the requested length.
    one = read_native(reader, blob; records = Dict("dim" => "time", "indices" => [1]))
    for name in ("t2m", "sp")
        @test isequal(one[name].data, full[name].data[2:2, :, :])
        @test one[name].dims == full[name].dims            # dims are NAMES, not lengths
        @test one[name].attrs == full[name].attrs
    end
    # the record axis's own coordinate is sliced WITH the data, raw, attrs kept
    @test one["time"].data == full["time"].data[2:2]
    @test eltype(one["time"].data) == eltype(full["time"].data)
    @test one["time"].attrs == full["time"].attrs
    @test one["latitude"].data == full["latitude"].data    # untouched axes stay whole

    # Order is the order given, and DUPLICATES are legal — the end-of-data bracket
    # [last, last] is exactly a duplicate pair, and a reader that de-duplicated or
    # sorted would silently hand back one record where two were asked for.
    rev = read_native(reader, blob; records = Dict("dim" => "time", "indices" => [1, 0]))
    @test isequal(rev["t2m"].data, full["t2m"].data[[2, 1], :, :])
    dup = read_native(reader, blob; records = Dict("dim" => "time", "indices" => [1, 1]))
    @test size(dup["t2m"].data, 1) == 2
    @test isequal(dup["t2m"].data, full["t2m"].data[[2, 2], :, :])
    @test dup["time"].data == full["time"].data[[2, 2]]

    # `variables`, `select` and `records` all compose: project, window, then records.
    sel = Dict("axes" => Any["all", Dict("slice" => [1, 3]), Dict("indices" => [0, 2])])
    all3 = read_native(reader, blob; variables = ["t2m"], select = sel,
                       records = Dict("dim" => "time", "indices" => [1]))
    @test variable_names(all3) == ["t2m"]
    @test isequal(all3["t2m"].data, full["t2m"].data[2:2, 2:3, [1, 3]])
    @test all3["latitude"].data == full["latitude"].data[2:3]
    @test all3["time"].data == full["time"].data[2:2]

    # The reader is told the RECORDS, never the cadence: it does no wrapping, so an
    # out-of-range index is an error rather than a silently `mod1`'d record.
    for bad in ([2], [-1], [0, 2])
        e = try
            read_native(reader, blob; records = Dict("dim" => "time", "indices" => bad))
            nothing
        catch err
            err
        end
        @test e isa ArgumentError
        @test occursin("outside 0:1", sprint(showerror, e))
    end
    # a dimension the blob does not have, an empty selection, and a malformed shape
    @test_throws ArgumentError read_native(reader, blob;
        records = Dict("dim" => "nope", "indices" => [0]))
    @test_throws ArgumentError read_native(reader, blob;
        records = Dict("dim" => "time", "indices" => Int[]))
    @test_throws ArgumentError read_native(reader, blob; records = Dict("dim" => "time"))
    # `select` and `records` may not both narrow the same axis
    @test_throws ArgumentError read_native(reader, blob;
        select = Dict("axes" => Any["all", Dict("slice" => [1, 3]), "all"]),
        records = Dict("dim" => "latitude", "indices" => [0]))

    # The CALLABLE `indices`: resolved against the axis length inside this
    # reader's own open, so a caller whose record choice depends on that length
    # (`mod1(tick, len)`) does not have to open the blob a second time to learn
    # it. Same answer as the equivalent literal list.
    seen = Int[]
    cb = read_native(reader, blob; records = Dict("dim" => "time",
        "indices" => len -> (push!(seen, len); [len - 1])))
    @test seen == [2]                                  # told the real axis length
    @test isequal(cb["t2m"].data, full["t2m"].data[2:2, :, :])
    @test cb["time"].data == full["time"].data[2:2]
    # a resolver that returns something out of range is checked like any other list
    @test_throws ArgumentError read_native(reader, blob;
        records = Dict("dim" => "time", "indices" => len -> [len]))
    # a blob with no such dimension does NOT call the resolver and narrows
    # nothing — the answer a caller that had probed with `dim_length` would give
    # itself — while a LITERAL list against a missing dimension stays an error.
    called = Ref(false)
    absent = read_native(reader, blob; records = Dict("dim" => "no-such-dim",
        "indices" => len -> (called[] = true; Int[])))
    @test !called[]
    @test isequal(absent["t2m"].data, full["t2m"].data)
    # "Callable" means callable, not `isa Function`: a callable STRUCT is the
    # spelling a caller with per-file state reaches for, and it must resolve
    # rather than fall through to the list branch and die in `iterate`.
    fn = read_native(reader, blob;
                     records = Dict("dim" => "time", "indices" => LastRecord()))
    @test isequal(fn["t2m"].data, full["t2m"].data[2:2, :, :])
    @test isequal(read_native(reader, blob;
        records = Dict("dim" => "no-such-dim", "indices" => LastRecord()))["t2m"].data,
        full["t2m"].data)
end

@testset "reader edge cases" begin
    # zarr is now active + store-backed: read_store requires an explicit variable
    # list (the store cannot be enumerated without a consolidated .zmetadata).
    @test status_of(FORMAT_REGISTRY, "zarr") == :active
    @test store_backed(FORMAT_REGISTRY["zarr"])
    @test_throws ErrorException read_store(FORMAT_REGISTRY["zarr"], Cache(; offline = true),
                                           "s3://b/z"; variables = nothing)

    # CSV inference fallback: with no numeric_columns, digit-only TEXT would be
    # mis-inferred as numeric — which is exactly why the loader must pass the
    # list. Here every value is a real number, so inference is safe.
    tmp = joinpath(mktempdir(), "nums.csv")
    write(tmp, "a,b\n1.5,2\n3.5,4\n")
    nds = read_native(CSVReader(), tmp)            # no numeric_columns => infer
    @test eltype(nds["a"].data) == Float64
    @test nds["a"].data == [1.5, 3.5]
    @test eltype(nds["b"].data) == Float64
end

# --- FF10 point reader unit tests -------------------------------------------

# A tiny FF10 point blob: a `#` header block + 3 data rows. Two rows (NOX/SO2)
# share ONE stack (same F001/U1/R1/P1 + stack params + lon/lat), differing only
# in POLID/ANN_VALUE — the reader must NOT pivot/aggregate. Row 1 has a quoted
# FACILITY_NAME with an embedded comma and a blank DESIGN_CAPACITY (numeric->NaN).
function _ff10_fixture_text()
    cols = length(EarthSciIO.FF10_POINT_COLUMNS)
    idx = Dict(n => j for (j, n) in enumerate(EarthSciIO.FF10_POINT_COLUMNS))
    function mkrow(over)
        r = fill("", cols)
        for (k, v) in over
            r[idx[k]] = v
        end
        # FACILITY_NAME may embed a comma -> RFC-4180 quote it.
        nm = r[idx["FACILITY_NAME"]]
        occursin(',', nm) && (r[idx["FACILITY_NAME"]] = "\"" * nm * "\"")
        return join(r, ',')
    end
    stack = ["COUNTRY_CD"=>"US", "REGION_CD"=>"01001", "FACILITY_ID"=>"F001",
             "UNIT_ID"=>"U1", "REL_POINT_ID"=>"R1", "PROCESS_ID"=>"P1",
             "SCC"=>"0030700101", "FACILITY_NAME"=>"Autauga Plant, Unit 1",
             "STKHGT"=>"100.0", "STKTEMP"=>"500.0", "LONGITUDE"=>"-86.51045",
             "LATITUDE"=>"32.43878", "ZIPCODE"=>"00000"]
    lines = ["#FORMAT=FF10_POINT", "#COUNTRY US", "",
             mkrow([stack; ["POLID"=>"NOX", "ANN_VALUE"=>"123.45"]]),
             mkrow([stack; ["POLID"=>"SO2", "ANN_VALUE"=>"67.89"]]),
             mkrow(["COUNTRY_CD"=>"US", "REGION_CD"=>"01001", "FACILITY_ID"=>"F002",
                    "POLID"=>"PM25", "ANN_VALUE"=>"4.2",
                    "FACILITY_NAME"=>"Plain Name"])]
    return join(lines, '\n') * '\n'
end

@testset "FF10 reader — header/quote/empty typing" begin
    tmp = joinpath(mktempdir(), "ff10_point.csv")
    write(tmp, _ff10_fixture_text())
    nds = read_native(FF10Reader(), tmp)

    # 77 columns, all on a single `index` dim, no coords.
    @test length(nds.variables) == 77
    @test isempty(nds.coords)
    @test nds["ANN_VALUE"].dims == ["index"]

    # `#` header + blank line skipped -> exactly 3 data rows.
    @test length(nds["POLID"].data) == 3

    # numeric vs string typing.
    @test eltype(nds["ANN_VALUE"].data) == Float64
    @test nds["ANN_VALUE"].data == [123.45, 67.89, 4.2]
    @test eltype(nds["POLID"].data) <: AbstractString

    # leading-zero codes stay strings (never floats).
    @test nds["REGION_CD"].data == ["01001", "01001", "01001"]
    @test nds["SCC"].data[1] == "0030700101"
    @test nds["ZIPCODE"].data[1] == "00000"

    # quoted comma preserved verbatim (quotes stripped).
    @test nds["FACILITY_NAME"].data[1] == "Autauga Plant, Unit 1"
    @test nds["FACILITY_NAME"].data[3] == "Plain Name"

    # blank numeric cell -> NaN; blank string cell -> "".
    @test isnan(nds["DESIGN_CAPACITY"].data[1])
    @test nds["TRIBAL_CODE"].data[1] == ""

    # multi-pollutant-same-stack: rows 1 & 2 share the stack, differ in POLID/ANN.
    @test nds["FACILITY_ID"].data[1] == nds["FACILITY_ID"].data[2] == "F001"
    @test nds["STKHGT"].data[1] == nds["STKHGT"].data[2] == 100.0
    @test nds["POLID"].data[1:2] == ["NOX", "SO2"]
    @test nds["ANN_VALUE"].data[1:2] == [123.45, 67.89]
end

import ZipFile

@testset "FF10 reader — zip member extraction" begin
    dir = mktempdir()
    csvpath = joinpath(dir, "point.csv")
    text = _ff10_fixture_text()
    write(csvpath, text)
    # bare-csv decode (the conformance path).
    bare = read_native(FF10Reader(), csvpath)

    # build a zip holding the CSV as member `inv/point.csv`.
    zippath = joinpath(dir, "2016fd_inputs_point.zip")
    w = ZipFile.Writer(zippath)
    f = ZipFile.addfile(w, "inv/point.csv")
    write(f, text)
    close(w)

    zipped = read_native(FF10Reader(), zippath; member = "inv/point.csv")
    @test zipped["ANN_VALUE"].data == bare["ANN_VALUE"].data
    @test zipped["POLID"].data == bare["POLID"].data
    @test zipped["FACILITY_NAME"].data == bare["FACILITY_NAME"].data
    # a missing member is a clear error, not a silent empty.
    @test_throws ArgumentError read_native(FF10Reader(), zippath; member = "nope.csv")
end

# The lowercase `country_cd,region_cd,…` header line the EPA 2016fd members
# carry (77 fields, NOT a `#` comment).
_ff10_header_line() = join(lowercase.(EarthSciIO.FF10_POINT_COLUMNS), ',')

# One 77-field row with the given FACILITY_ID/POLID/ANN_VALUE.
function _ff10_tiny_row(fac, polid, ann)
    idx = Dict(n => j for (j, n) in enumerate(EarthSciIO.FF10_POINT_COLUMNS))
    r = fill("", length(EarthSciIO.FF10_POINT_COLUMNS))
    r[idx["COUNTRY_CD"]] = "US"
    r[idx["REGION_CD"]] = "01001"
    r[idx["FACILITY_ID"]] = fac
    r[idx["POLID"]] = polid
    r[idx["ANN_VALUE"]] = ann
    return join(r, ',')
end

# A zip with two `*egu*` members + one excluded member, each carrying a `#`
# comment block and the `country_cd` header line. Written in NON-sorted order to
# prove the read order is sorted-name.
function _ff10_egu_zip(dir)
    member(rows) = "#FORMAT=FF10_POINT\n" * _ff10_header_line() * "\n" *
                   join(rows, '\n') * "\n"
    zippath = joinpath(dir, "2016fd_inputs_point.zip")
    w = ZipFile.Writer(zippath)
    # a glob-matching DIRECTORY placeholder entry (like the real 2016fd
    # `…/ptegu/`) — selection must ignore it (file members only).
    ZipFile.addfile(w, "point_egu/")
    for (name, rows) in [
        ("point/egu_beta.csv", [_ff10_tiny_row("F202", "NOX", "333.3")]),
        ("point/egu_alpha.csv", [_ff10_tiny_row("F101", "NOX", "111.1"),
                                 _ff10_tiny_row("F101", "SO2", "22.2")]),
        ("point/ptnonipm.csv", [_ff10_tiny_row("F999", "NOX", "999.9")]),
    ]
        f = ZipFile.addfile(w, name)
        write(f, member(rows))
    end
    close(w)
    return zippath
end

@testset "FF10 reader — multi-member glob + header skip" begin
    dir = mktempdir()
    zippath = _ff10_egu_zip(dir)

    # glob selects BOTH egu members (not ptnonipm); rows concatenate in sorted
    # member-name order (alpha's 2 rows, then beta's 1); one header dropped each.
    nds = read_native(FF10Reader(), zippath;
                      member_glob = "*egu*", skip_header_row = true)
    @test nds["FACILITY_ID"].data == ["F101", "F101", "F202"]
    @test nds["ANN_VALUE"].data == [111.1, 22.2, 333.3]
    @test !("F999" in nds["FACILITY_ID"].data)

    # a glob matching zero members is an error, not a silent empty.
    @test_throws ArgumentError read_native(FF10Reader(), zippath;
                                           member_glob = "*nope*")

    # explicit members ∪ glob, deduplicated, sorted.
    both = read_native(FF10Reader(), zippath;
                       members = ["point/ptnonipm.csv", "point/egu_alpha.csv"],
                       member_glob = "*egu*", skip_header_row = true)
    @test both["FACILITY_ID"].data == ["F101", "F101", "F202", "F999"]

    # an explicit member absent from the archive is an error.
    @test_throws ArgumentError read_native(FF10Reader(), zippath;
                                           members = ["point/absent.csv"],
                                           skip_header_row = true)

    # without skip_header_row the header line is a 77-field data row that dies
    # at the numeric parse of ann_value (it is NOT silently accepted).
    @test_throws ArgumentError read_native(FF10Reader(), zippath;
                                           member_glob = "*egu*")

    # singular member + skip_header_row composes.
    one = read_native(FF10Reader(), zippath;
                      member = "point/egu_beta.csv", skip_header_row = true)
    @test one["FACILITY_ID"].data == ["F202"]

    # `member` is mutually exclusive with `members`/`member_glob`.
    @test_throws ArgumentError read_native(FF10Reader(), zippath;
                                           member = "point/egu_beta.csv",
                                           member_glob = "*egu*")

    # asserting a header on an input that has none errors — never a silently
    # dropped data row.
    bare = joinpath(dir, "noheader.csv")
    write(bare, _ff10_fixture_text())
    @test_throws ArgumentError read_native(FF10Reader(), bare;
                                           skip_header_row = true)
end

@testset "FF10 reader — glob matcher semantics" begin
    m(p, t) = occursin(EarthSciIO._glob_regex(p), t)
    @test m("*egu*", "point/egucems_2016fd.csv")
    @test m("*egu*", "egu")
    @test !m("*egu*", "point/ptnonipm.csv")
    @test m("?gu", "egu")
    @test !m("?gu", "eegu")
    @test m("egu[12].csv", "egu1.csv")
    @test !m("egu[12].csv", "egu3.csv")
    @test m("egu[!12].csv", "egu3.csv")
    @test m("egu[a-c].csv", "egub.csv")
    @test !m("egu[a-c].csv", "egud.csv")
    @test m("lit[", "lit[")            # unclosed class is a literal
    @test !m("*EGU*", "point/egucems.csv")  # case-sensitive
end

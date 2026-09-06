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

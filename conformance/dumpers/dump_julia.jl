# Julia track's native-array dumper for the cross-language conformance harness.
#
# Drives the **Julia Provider** (`EarthSciIO.const_provider`) over every committed
# corpus case, fully OFFLINE (the cache is rooted at the corpus and refuses the
# network), and emits the decoded native arrays as a canonical JSON dump in the
# SAME schema as the Python (`dump_python.py`) and Rust
# (`rust/examples/conformance_dump.rs`) dumpers. The cross-language comparator
# (`conformance/crosscheck.py`) diffs the three dumps + the corpus oracle to prove
# native-array equality across all three tracks (`esio-9nb.9`).
#
# Dump schema — `earthsciio/native-dump/v1` (see `conformance/CROSSLANG.md`).
# `data` is flattened **row-major (C order)** per `shape`; NCDatasets/Julia store
# column-major, so each array is permuted to file order before a C-order `vec`
# (mirrors `julia/test/test_readers.jl` `_corder`). A masked / `_FillValue` cell
# is `null` (== NaN); strings are emitted verbatim. A case whose `format` has no
# active reader in this track is `status="skipped"` (explicit, never dropped).
#
# Usage:  julia --project="$(julia conformance/dumpers/julia_env.jl)" \
#             conformance/dumpers/dump_julia.jl [out.json]
#         (`conformance/run_conformance.sh` does both steps.)

using EarthSciIO
import JSON

# Four of the corpus formats decode through weakdep EXTENSIONS: `zarr` through
# `EarthSciIOBloscExt` (`Blosc`), `shapefile` through `EarthSciIOShapefileExt`,
# `parquet` through `EarthSciIOParquet2Ext`, `geotiff` through
# `EarthSciIOTiffImagesExt`. They are `[weakdeps]` of `julia/Project.toml` (a
# base install stays light), so they are NOT importable under `--project=julia`.
#
# This dumper therefore runs in the environment `dumpers/julia_env.jl` prepares,
# where EarthSciIO and every one of its weakdeps are resolved TOGETHER and a
# plain import just works. It deliberately does NOT resolve anything itself: the
# fallback that used to live here — a throwaway env pushed onto `LOAD_PATH` —
# looks harmless and is not, because Julia resolves a stacked package's
# dependencies through the PRIMARY environment's manifest and the two
# independent resolutions silently mix (see `julia_env.jl` for the case that
# fired: WeakRefStrings built against the primary `Parsers`). A missing weakdep
# is a setup error, and says so.
function _load_ext(extname::Symbol, pkg::String)
    Base.get_extension(EarthSciIO, extname) === nothing || return
    try
        @eval import $(Symbol(pkg))
    catch err
        error("""
              cannot import `$pkg`, the trigger for EarthSciIO's `$extname`.
              Run this dumper in the prepared conformance environment:
                  julia --project="\$(julia conformance/dumpers/julia_env.jl)" $(basename(@__FILE__)) ...
              or just run conformance/run_conformance.sh.
              active project: $(Base.active_project())
              underlying: $err
              """)
    end
    Base.retry_load_extensions()
end

_load_ext(:EarthSciIOBloscExt, "Blosc")
_load_ext(:EarthSciIOShapefileExt, "Shapefile")
_load_ext(:EarthSciIOParquet2Ext, "Parquet2")
_load_ext(:EarthSciIOTiffImagesExt, "TiffImages")

# Row-major (C-order) flatten of a native array whose axes are in file (`dims`)
# order — matches numpy `.reshape(-1)` on the Python track's arrays.
_corder(a::AbstractVector) = collect(a)
_corder(a::AbstractArray) = vec(permutedims(a, reverse(1:ndims(a))))

# Encode one NativeField to the dump schema (dtype/dims/shape/data).
function encode_field(field)
    data = field.data
    dims = collect(String.(field.dims))
    if eltype(data) <: AbstractString
        vals = Any[String(x) for x in data]
        return _with_fill_value(Dict("dtype" => "string", "dims" => dims,
                                     "shape" => [length(vals)], "data" => vals), field)
    end
    flat = _corder(data)
    et = eltype(data)
    if et <: AbstractFloat
        dtype = "float64"
        vals = Any[isnan(x) ? nothing : Float64(x) for x in flat]
    elseif et === Bool
        dtype = "bool"
        vals = Any[Bool(x) for x in flat]
    elseif et <: Integer
        dtype = et == Int32 ? "int32" : "int64"
        vals = Any[Int(x) for x in flat]
    else
        error("unexpected numeric eltype $et in field with dims $dims")
    end
    return _with_fill_value(Dict("dtype" => dtype, "dims" => dims,
                                 "shape" => collect(Int, size(data)), "data" => vals),
                            field)
end

# Carry a surviving fill/missing sentinel into the dump under ONE key.
# spec/conformance.md §3 pins where each track reports it: Rust has a dedicated
# `NativeField.fill_value` field, Python and Julia carry `attrs["fill_value"]`.
# Those are the SAME datum in three spellings, not three decisions, so every
# dumper normalises to the dump schema's single `fill_value` key — otherwise the
# comparator would read a difference that is not one (or miss one track dropping
# the sentinel entirely).
function _with_fill_value(enc::Dict, field)
    fv = get(field.attrs, "fill_value", nothing)
    fv === nothing || (enc["fill_value"] = fv isa AbstractFloat ? Float64(fv) : Int64(fv))
    return enc
end

# A coord is a field plus the CF units/calendar it carries (if any).
function encode_coord(field)
    enc = encode_field(field)
    for k in ("units", "calendar")
        haskey(field.attrs, k) && (enc[k] = String(field.attrs[k]))
    end
    return enc
end

# Run the Julia Provider over one corpus case and encode its native arrays. Skips
# (without error) a case whose format has no active reader, matching the Rust
# track (netcdf only) so the harness reports the gap instead of failing.
function dump_case(corpus, case)
    fmt = String(case["format"])
    if !haskey(FORMAT_REGISTRY, fmt) || status_of(FORMAT_REGISTRY, fmt) != :active
        return Dict("format" => fmt, "status" => "skipped",
                    "reason" => "no active reader registered for format '$fmt' in the Julia track")
    end
    # An OFFLINE cache rooted at the corpus: each case resolves from disk by its
    # sha256(resolved_url) key; verify=true checks the blob against its manifest.
    cache = Cache(LocalStore(joinpath(corpus, "cache")); offline = true, verify = true)
    url = String(case["resolved_url"])
    provider = if fmt == "csv"
        # numeric_columns is REQUIRED (digit-only text like location_id must stay
        # a string); the corpus case pins the list.
        nc = String.(case["decode"]["numeric_columns"])
        const_provider(cache, url; format = fmt, reader_kwargs = (; numeric_columns = nc))
    elseif fmt == "ff10"
        # FF10 point: the case pins the 42 numeric columns, schema kind, and the
        # zip member selection — member (singular), members/member_glob (multi-
        # member; sorted-name concat), skip_header_row (drop one asserted
        # `country_cd` header line per member). member=null decodes the bare blob.
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
        const_provider(cache, url; format = fmt, reader_kwargs = (; kw...))
    elseif fmt == "shapefile"
        # ESRI shapefile: the case pins the `.shp` member inside the zip blob and
        # the text code column the model wants as a number.
        dec = case["decode"]
        kw = Dict{Symbol,Any}()
        m = get(dec, "member", nothing)
        m === nothing || (kw[:member] = String(m))
        nc = get(dec, "numeric_columns", nothing)
        nc === nothing || (kw[:numeric_columns] = String.(nc))
        const_provider(cache, url; format = fmt, reader_kwargs = (; kw...))
    elseif fmt == "parquet"
        # Columnar table. `variables` is the loader's PROJECTION and is pushed
        # into the reader (only those column chunks come off disk); the case's
        # decode block pins the three decode options — `float_columns` (an
        # integer measurement, or a column of fixed-decimal TEXT, read as
        # float64) and the two null gates `null_int` / `null_string`.
        dec = case["decode"]
        kw = Dict{Symbol,Any}()
        fc = get(dec, "float_columns", nothing)
        fc === nothing || (kw[:float_columns] = String.(fc))
        ni = get(dec, "null_int", nothing)
        ni === nothing || (kw[:null_int] = Int64(ni))
        ns = get(dec, "null_string", nothing)
        ns === nothing || (kw[:null_string] = String(ns))
        vars = String[String(v) for v in get(case, "variables", String[])]
        const_provider(cache, url; format = fmt,
                       variables = isempty(vars) ? nothing : vars,
                       reader_kwargs = (; kw...))
    elseif fmt == "zarr"
        # Store-backed: `url` is the store base; `variables` names the arrays (no
        # .zmetadata to enumerate); `select` (the orthogonal selection) rides in
        # reader_kwargs and drives lazy chunk fetch.
        vars = String[String(v) for v in case["variables"]]
        const_provider(cache, url; format = fmt, variables = vars,
                       reader_kwargs = (; select = case["select"]))
    else
        const_provider(cache, url; format = fmt)
    end
    nds = materialize(provider)  # CONST: read the single corpus blob once
    return Dict(
        "format" => fmt, "status" => "decoded",
        "variables" => Dict(n => encode_field(f) for (n, f) in nds.variables),
        "coords" => Dict(n => encode_coord(f) for (n, f) in nds.coords),
    )
end

# The case ids `$ESIO_CONFORMANCE_CASES` restricts this run to, or `nothing`.
# A comma-separated list, honoured identically by every dumper and by
# `crosscheck.py`, so a filtered run is still a complete cross-check of the cases
# it names. It exists for an environment where one track's backend for some
# OTHER format is missing or too old: without it a single unrelated broken case
# makes the gate unrunnable and a real divergence elsewhere invisible. Unset ⇒
# every case, which is what CI runs.
function selected_ids()
    raw = strip(get(ENV, "ESIO_CONFORMANCE_CASES", ""))
    isempty(raw) && return nothing
    return Set(String[strip(c) for c in split(raw, ",") if !isempty(strip(c))])
end

function main()
    corpus = normpath(joinpath(@__DIR__, "..", "corpus"))
    index = JSON.parsefile(joinpath(corpus, "cases.json"))
    only = selected_ids()
    cases = Dict{String,Any}()
    for entry in index["cases"]
        case = JSON.parsefile(joinpath(corpus, entry["file"]))
        id = String(case["id"])
        only === nothing || id in only || continue
        cases[id] = dump_case(corpus, case)
    end
    active = sort!([n for n in registered_names(FORMAT_REGISTRY)
                    if status_of(FORMAT_REGISTRY, n) == :active])
    out = Dict(
        "schema" => "earthsciio/native-dump/v1",
        "language" => "julia",
        "provider" => "EarthSciIO.const_provider",
        "readers" => active,
        "cases" => cases,
    )
    text = JSON.json(out, 2)
    if !isempty(ARGS)
        open(ARGS[1], "w") do io
            write(io, text)
            write(io, "\n")
        end
    else
        println(text)
    end
end

main()

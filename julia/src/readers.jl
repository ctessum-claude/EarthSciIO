# Format readers — component (b) (esio-9nb.5). A reader opens a cached blob and
# returns RAW native-grid arrays keyed by the on-disk `file_variable` name. It
# applies ONLY the format/CF decode pinned by spec/conformance.md §3; it does
# NOT remap variable names or convert units (those stay in ESS — Risk R3).
#
# Readers register into `FORMAT_REGISTRY` by name (`netcdf`, `csv`) and are the
# decode half (conformance checks 3–4) that the cache layer (component a) left
# to this bead. A new format plugs in by adding a `Reader` subtype + a
# `read_native` method + one `register!` line — never a Provider change.

# --- native array containers ------------------------------------------------

"""
    NativeField(data, dims, attrs)

One native-grid array exactly as a reader decodes it: `data` is a Julia array
whose axes correspond, in order, to `dims` (the on-disk dimension names, file
order — e.g. `["time","latitude","longitude"]`). Numeric fields are `Float64`
with `NaN` for masked/`_FillValue` cells; raw integer reads (e.g. an undecoded
time axis) keep their stored integer type; text columns are `String`. `attrs`
carries decode-relevant metadata the reader must NOT act on but ESS needs —
notably a time axis's `units`/`calendar` (calendar decoding is ESS's job)."""
struct NativeField
    data::AbstractArray
    dims::Vector{String}
    attrs::Dict{String,Any}
end
NativeField(data::AbstractArray, dims::AbstractVector) =
    NativeField(data, String.(collect(dims)), Dict{String,Any}())

Base.size(f::NativeField) = size(f.data)
Base.eltype(f::NativeField) = eltype(f.data)

function Base.show(io::IO, f::NativeField)
    print(io, "NativeField(", eltype(f.data), " ", join(string.(size(f.data)), "×"),
          " dims=", f.dims, ")")
end

"""
    NativeDataset(variables, coords)

The native arrays a reader returns from one blob: `variables` (data fields keyed
by `file_variable`) and `coords` (the dimension-coordinate fields — e.g. the
grid's `latitude`/`longitude`/`time`). Both map `String` name → [`NativeField`].
`getindex` looks in `variables` then `coords`, so `nds["t2m"]` and `nds["time"]`
both resolve."""
struct NativeDataset
    variables::Dict{String,NativeField}
    coords::Dict{String,NativeField}
end
NativeDataset() = NativeDataset(Dict{String,NativeField}(), Dict{String,NativeField}())

function Base.getindex(nds::NativeDataset, name::AbstractString)
    k = String(name)
    haskey(nds.variables, k) && return nds.variables[k]
    haskey(nds.coords, k) && return nds.coords[k]
    throw(KeyError(k))
end
Base.haskey(nds::NativeDataset, name::AbstractString) =
    haskey(nds.variables, String(name)) || haskey(nds.coords, String(name))
"""Names of the data variables (not coordinates)."""
variable_names(nds::NativeDataset) = sort!(collect(keys(nds.variables)))
"""Names of the coordinate fields."""
coord_names(nds::NativeDataset) = sort!(collect(keys(nds.coords)))

function Base.show(io::IO, nds::NativeDataset)
    print(io, "NativeDataset(variables=", variable_names(nds),
          ", coords=", coord_names(nds), ")")
end

# --- NetCDF reader (NCDatasets) ---------------------------------------------

"""
    NetCDFReader()

The `netcdf` format reader, backed by NCDatasets. CF-decodes per
spec/conformance.md §3: applies `scale_factor`/`add_offset` (math in float64),
maps `_FillValue`/`missing_value` cells to `NaN`, and returns the time axis
**raw** (the stored integers/floats) with `units`+`calendar` carried in
`attrs` — calendar→wall-clock decoding is ESS's job, never the reader's.

NCDatasets exposes arrays in column-major (reversed) dimension order; this
reader permutes each array back to **file order** so `field.dims` and
`size(field.data)` match the on-disk layout (and the Python/xarray track).

`reader_kwargs`: `variables=[...]` restricts the returned **data variables** and
is a real PROJECTION PUSHDOWN — an unrequested variable is never decoded, rather
than decoded and thrown away. Coordinate fields are always returned (they are the
native grid the arrays live on). An empty list, like `nothing`, means **every**
data variable (spec/conformance.md §3), never none; a requested name absent from
the blob is an error listing what is present.

`select=Dict("axes"=>[...])` is a DECODE-TIME orthogonal selection: the reader
still fetches the whole blob (one URL, one cache key — nothing about the fetch or
the cache changes), but materialises only the requested hyperslab, reading only
the intersecting HDF5 chunks off local disk. `supports_selection` is `true` while
`store_backed` stays `false`, and that pair is how a caller tells a DECODE-scoped
selection from the zarr reader's FETCH-scoped one.

The vocabulary is the zarr reader's, unchanged and **0-based**: each axis is
`"all"`, `Dict("indices"=>[...])` (an explicit, possibly non-contiguous, ordered
index list, returned in the order given) or `Dict("slice"=>[start, stop, step?])`
(half-open, `step` defaults to 1). Zero-based is chosen over the AST side's
1-based convention on purpose: it is ONE `select` spelling across every reader and
both sibling tracks, so a caller that translates for zarr does not translate
differently here. Applied thus:

  * the axes are **positional over FILE-order dims** (`[time, lev, lat, lon]` for
    a GEOS-FP A3dyn variable, i.e. the order `field.dims` reports) of every array
    whose rank equals the axis count — the zarr rule;
  * that induces a **dimension-name → selector** map, which is then applied by
    NAME to every other array **and to the coordinate fields**, so a windowed
    variable can never come back beside a full-length `lon`/`lat`. Two arrays of
    matching rank that disagree about a dimension are an error, not a coin toss;
  * a `select` whose **time axis** is anything but `"all"` is REFUSED: record
    selection belongs to the Provider, which owns the cadence
    (`records_per_sample`), not to the reader (spec/conformance.md §3);
  * an axis count matching no array in the blob is an error rather than a
    silently-ignored selection;
  * every resolved index is **bounds-checked** (`0 <= i < dim_len`) for a `slice`
    exactly as for an `indices` list: an over-long or negative `[start, stop)` is
    an error, never a silent clamp and never a negative wrap-around;
  * a **dimension is never dropped** — a one-index axis comes back at length 1 —
    and an axis may legally select NOTHING (`Dict("indices"=>[])`, or an empty
    half-open `Dict("slice"=>[1, 1])`), giving a **zero-length axis** kept in
    `dims` rather than an error."""
struct NetCDFReader <: Reader end

# Widened capability (spec/registries.md §2): "honours `select` without
# materialising the whole array". `store_backed` (still `false` here) is what
# separates a fetch-scoped selection — the zarr reader's, which shrinks the
# DOWNLOAD — from this decode-scoped one, which shrinks only the decode.
supports_selection(::NetCDFReader) = true

# A CF time axis is one whose `units` is "<step> since <reference>" (hours since
# …, days since …). Matching xarray `decode_times=false`, such variables are
# returned raw; `mask_and_scale` still applies to everything else.
function _is_cf_time(attrib)::Bool
    haskey(attrib, "units") || return false
    return occursin(r"\bsince\b"i, strip(String(attrib["units"])))
end

# Reverse NCDatasets' storage (column-major) order to the file's logical order.
_to_file_order(a::AbstractArray) =
    ndims(a) > 1 ? permutedims(a, reverse(1:ndims(a))) : a

# Decode rule (spec/conformance.md §3): masked → Float64 with NaN; an unpacked
# pure-integer field keeps its integer logical type; every other numeric read is
# normalized to Float64 so float32-vs-float64 never diverges across languages.
function _finalize_numeric(a::AbstractArray)
    if Missing <: eltype(a)
        return map(x -> ismissing(x) ? NaN : Float64(x), a)   # → Array{Float64}, shape kept
    elseif eltype(a) <: Integer
        return collect(a)
    elseif eltype(a) <: AbstractFloat
        return Float64.(a)
    else
        return collect(a)
    end
end

function _carry_attrs(attrib)
    d = Dict{String,Any}()
    for k in ("units", "calendar")
        haskey(attrib, k) && (d[k] = String(attrib[k]))
    end
    return d
end

# The set of DATA variables a `variables` projection asks for, or `nothing` for
# "all of them" (`variables === nothing` or an EMPTY list — spec/conformance.md
# §3, which the parquet reader and both sibling tracks already spell this way).
# A requested name absent from the blob is an error naming what is present, so a
# typo'd `file_variable` cannot read back as a silently missing array.
function _netcdf_wanted(ds, dimset::AbstractSet{String}, variables)
    (variables === nothing || isempty(variables)) && return nothing
    want = Set(String[String(v) for v in variables])
    present = sort!(String[String(vn) for vn in keys(ds) if !(String(vn) in dimset)])
    absent = sort!(String[v for v in want if !(v in present)])
    isempty(absent) || throw(ArgumentError(
        "requested variables not in blob: $absent; present data variables: $present"))
    return want
end

# Is dimension `d` the file's TIME axis? A same-named coordinate variable whose
# `units` is CF "<step> since <ref>" settles it; a dimension literally named
# `time` with no coordinate variable (the GEOS-FP shape) is taken at its word.
# Only used to REFUSE a time selection — record selection is the Provider's job —
# so erring towards "yes" costs a clear error, never a wrong array.
function _netcdf_is_time_dim(ds, d::AbstractString)
    if haskey(ds, d)
        _is_cf_time(ds[d].attrib) && return true
    end
    return lowercase(String(d)) == "time"
end

# `select` -> Dict(dimension name => ordered 0-based index list), or `nothing`
# when nothing is selected. Axes are positional over the FILE-order dims of every
# array whose rank matches the axis count (the zarr rule); the induced map is what
# the read applies BY NAME, which is what keeps the coordinates in step with the
# data. netCDF dimension lengths are file-global, so each axis resolves once.
function _netcdf_dim_selection(ds, select)
    axes_spec = _select_axes(select)
    axes_spec === nothing && return nothing
    parsed = [_parse_axis(a) for a in axes_spec]
    naxes = length(parsed)

    bydim = Dict{String,Any}()
    matched = false
    for vn in keys(ds)
        dims = reverse(String.(collect(NCDatasets.dimnames(ds[vn]))))
        length(dims) == naxes || continue
        matched = true
        for (i, d) in enumerate(dims)
            if haskey(bydim, d) && bydim[d] != parsed[i]
                throw(ArgumentError(
                    "select is ambiguous: dimension '$d' is asked for two different " *
                    "selectors by two rank-$naxes variables in this blob; a netcdf " *
                    "`select` is positional over file-order dims and must agree"))
            end
            bydim[d] = parsed[i]
        end
    end
    matched || throw(ArgumentError(
        "select has $naxes axes but no variable in the blob has rank $naxes; " *
        "a netcdf `select` is positional over the file-order dims of the arrays " *
        "it applies to"))

    out = Dict{String,Vector{Int}}()
    for (d, ax) in bydim
        ax[1] === :all && continue
        _netcdf_is_time_dim(ds, d) && throw(ArgumentError(
            "select asks for a subset of the time dimension '$d'; record selection " *
            "is the Provider's (it owns the cadence: `records_per_sample`), not the " *
            "reader's — the time axis of a netcdf `select` must be \"all\""))
        out[d] = _resolve_axis(ax, Int(ds.dim[d]))
    end
    return isempty(out) ? nothing : out
end

# An ordered index list as (start, step, needs_gather): an arithmetic progression
# reads as ONE strided hyperslab; anything else (a permuted or irregular list)
# reads its bounding slab and gathers out of it — still never the whole array.
function _netcdf_progression(idxs::Vector{Int})
    length(idxs) <= 1 && return (1, false)
    step = idxs[2] - idxs[1]
    step >= 1 || return (1, true)
    for k in 3:length(idxs)
        idxs[k] - idxs[k-1] == step || return (1, true)
    end
    return (step, false)
end

# Read one variable under the dimension selection `sel` (`nothing` ⇒ whole array).
# `raw` reads the underlying variable with no CF transform (a time axis).
function _netcdf_read(v, file_dims::Vector{String}, sel, raw::Bool)
    src = raw ? v.var : v
    if sel === nothing || !any(haskey(sel, d) for d in file_dims)
        data = _to_file_order(Array(src))
        return raw ? data : _finalize_numeric(data)
    end
    # Per-axis slab range + local gather positions, in FILE order.
    nd = length(file_dims)
    ranges = Vector{Any}(undef, nd)
    gathers = Vector{Any}(undef, nd)
    needs_gather = false
    for (i, d) in enumerate(file_dims)
        idxs = get(sel, d, nothing)
        if idxs === nothing
            ranges[i] = Colon()
            gathers[i] = Colon()
            continue
        end
        if isempty(idxs)
            # An axis may legally resolve to NOTHING (`{"indices": []}`, or an
            # empty half-open `{"slice": [1, 1]}`): a zero-length axis, KEPT in
            # `dims`, never an error and never a dropped dimension. Read an empty
            # range on it — `extrema` of an empty list would throw, and the two
            # sibling tracks return a correctly-shaped empty array here.
            ranges[i] = 1:0
            gathers[i] = Colon()
            continue
        end
        lo, hi = extrema(idxs)
        step, gather = _netcdf_progression(idxs)
        ranges[i] = gather ? ((lo + 1):(hi + 1)) : ((lo + 1):step:(hi + 1))
        gathers[i] = gather ? (idxs .- lo .+ 1) : Colon()
        needs_gather |= gather
    end
    # NCDatasets indexes in its own (reversed) axis order; permute the slab back.
    data = _to_file_order(Array(src[reverse(ranges)...]))
    raw || (data = _finalize_numeric(data))
    return needs_gather ? data[gathers...] : data
end

function read_native(::NetCDFReader, path::AbstractString; variables = nothing,
                     select = nothing)
    nds = NativeDataset()
    NCDatasets.NCDataset(String(path), "r") do ds
        dimset = Set(String.(collect(keys(ds.dim))))
        # PROJECTION PUSHDOWN (spec/conformance.md §3): the loader's `variables`
        # reach the READER, so an unrequested variable is never decoded — a
        # GEOS-FP A1 file carries 47 variables and a provider wants one. An EMPTY
        # list means "every variable", NOT "no variables" (the same rule the
        # parquet reader and the other two tracks follow); an absent name is an
        # error listing what is present, never a silently missing array.
        want = _netcdf_wanted(ds, dimset, variables)
        # DECODE-TIME SELECTION: one blob is still fetched, but only the requested
        # hyperslab is materialised (NCDatasets reads just the intersecting HDF5
        # chunks). Resolved ONCE, per dimension name, so the coordinate fields are
        # sliced with the data they index rather than left at full length.
        sel = _netcdf_dim_selection(ds, select)
        for vn in keys(ds)
            name = String(vn)
            # Coordinates are always kept; data variables honour the projection.
            (want === nothing || name in dimset || name in want) || continue
            v = ds[vn]
            attrs = _carry_attrs(v.attrib)
            file_dims = reverse(String.(collect(NCDatasets.dimnames(v))))
            # A CF time axis is read RAW (no CF transform), so a "hours since …"
            # axis stays the stored integers; everything else is mask_and_scale'd
            # (NCDatasets applies scale/offset + _FillValue→missing).
            data = _netcdf_read(v, file_dims, sel, _is_cf_time(v.attrib))
            field = NativeField(data, file_dims, attrs)
            if String(vn) in dimset
                nds.coords[String(vn)] = field
            else
                nds.variables[String(vn)] = field
            end
        end
    end
    return nds
end

# --- CSV reader -------------------------------------------------------------

"""
    CSVReader()

The `csv` format reader — a second reader proving a non-NetCDF format plugs into
`FORMAT_REGISTRY` unchanged (spec/conformance.md). Columns named in
`numeric_columns` parse to `Float64` 1-D arrays keyed by the column
(`file_variable`) name; every other column is returned as a `String` array. All
fields have a single dimension `index`; there are no coordinates.

`numeric_columns` is REQUIRED by the loader spec and is not inferred: the corpus
`location_id` column is digit-only text (`"1"`,`"2"`) yet must stay a string, so
"parses as a number" is not a safe signal. When `numeric_columns === nothing`
the reader falls back to best-effort inference (every value parses as a float),
which the loader/`.esm` node should override. Quoted fields with embedded
delimiters are not handled (the points corpus has none); add that when a loader
needs it."""
struct CSVReader <: Reader end

_parses_float(s::AbstractString) = tryparse(Float64, strip(s)) !== nothing

function read_native(::CSVReader, path::AbstractString;
                     numeric_columns = nothing, delimiter::AbstractString = ",",
                     header_row::Integer = 1)
    rows = Vector{String}[]
    for ln in eachline(String(path))
        isempty(ln) && continue
        push!(rows, String.(split(rstrip(ln, ['\r']), delimiter)))
    end
    isempty(rows) && return NativeDataset()
    header = rows[header_row]
    body = rows[header_row+1:end]
    numset = numeric_columns === nothing ? nothing : Set(String.(collect(numeric_columns)))

    vars = Dict{String,NativeField}()
    for (j, col) in enumerate(header)
        name = String(col)
        vals = String[r[j] for r in body]
        isnum = numset === nothing ? all(_parses_float, vals) : (name in numset)
        data = isnum ? Float64[parse(Float64, strip(v)) for v in vals] : vals
        vars[name] = NativeField(data, ["index"], Dict{String,Any}())
    end
    return NativeDataset(vars, Dict{String,NativeField}())
end

# --- GeoTIFF reader (TiffImages backend via a weakdep extension) ------------

"""
    GeoTIFFReader()

The `geotiff` format reader — raster bands on a native lon/lat (geographic) or
x/y (projected) grid. The decode half for the ArcGIS ImageServer `exportImage`
rasters the ESS loaders fetch (LANDFIRE fuel model, USGS 3DEP elevation). One
data variable per band keyed `Band1`..`BandN` (1-based, the GDAL convention; the
LANDFIRE loader's `file_variable: "Band1"` matches), plus the cell-center
coordinate fields. Geographic rasters (`imageSR=4326`) get `lon`/`lat` axes;
projected rasters get `x`/`y`. Band arrays are `Float64` with the `GDAL_NODATA`
sentinel mapped to `NaN` (spec/conformance.md §3). Reader-only: no variable-name
remap, no unit conversion, no reprojection — those stay in ESS/ESD.

The decode backend is the pure-Julia `TiffImages` package, loaded LAZILY via a
weakdep extension (`EarthSciIOTiffImagesExt`) — mirroring the Python reader's lazy
`tifffile` import, so a base EarthSciIO install stays light. Calling `read_native`
without `using TiffImages` throws a clear install hint.

`reader_kwargs`: `band_names=[...]` renames the bands positionally (e.g. a
single-band elevation raster → `["elevation"]`); `variables=[...]` restricts the
returned bands (a requested-but-absent band is a `KeyError`)."""
struct GeoTIFFReader <: Reader end

# The real decode lives in ext/EarthSciIOTiffImagesExt.jl, whose method is typed
# `path::AbstractString` — strictly MORE specific than this untyped-`path` fallback,
# so when `using TiffImages` is active it wins by dispatch (no method overwrite,
# which precompilation forbids). This fallback fires only when the backend is absent.
read_native(::GeoTIFFReader, path; kwargs...) = error(
    "the geotiff reader needs the TiffImages backend: add `using TiffImages` so the " *
    "EarthSciIOTiffImagesExt extension supplies the decode (kept a weakdep to keep a " *
    "base EarthSciIO install light, mirroring the Python tifffile-optional path).")

# GTModelTypeGeoKey-style lookup (key 1024: 1=projected, 2=geographic) from a flat
# `GeoKeyDirectoryTag`. The directory is [version,keyRev,minorRev,nKeys,
# (KeyID,loc,count,value)*nKeys]; only INLINE keys (loc==0) carry their value in
# the 4th slot. Returns the value or `nothing`.
function _geotiff_geokey(geokeys, key_id::Integer)
    geokeys === nothing && return nothing
    g = Int[Int(v) for v in geokeys]
    length(g) < 4 && return nothing
    n = g[4]
    for k in 0:(n - 1)
        off = 4 + 4k                      # 0-based offset of the k-th key entry
        off + 4 <= length(g) || break
        g[off + 1] == key_id && g[off + 2] == 0 && return g[off + 4]
    end
    return nothing
end

# Parse the GDAL_NODATA sentinel (an ASCII tag, often null-terminated) → Float64,
# or `nothing` when absent/unparseable.
function _geotiff_nodata(raw)
    raw === nothing && return nothing
    s = raw isa AbstractVector{UInt8} ? String(copy(raw)) : String(raw)
    s = strip(replace(s, '\0' => ""))
    isempty(s) && return nothing
    return tryparse(Float64, s)
end

"""
    _assemble_geotiff(bands, tags; variables=nothing, band_names=nothing) -> NativeDataset

Build the GeoTIFF [`NativeDataset`] from decoded `bands` (each a `(height,width)`
`Matrix{Float64}` in file order, rows = y/lat) plus the raw IFD `tags` (Int tag id
→ value): cell-center axes from `ModelPixelScaleTag` (33550) + `ModelTiepointTag`
(33922) — `x = x0 + (col − i0 + 0.5)·sx`, `y = y0 − (row − j0 + 0.5)·sy`, GeoTIFF
model space being y-up while raster rows increase downward — the geographic vs
projected flag from `GeoKeyDirectoryTag` (34735) GTModelTypeGeoKey, and
`GDAL_NODATA` (42113) → `NaN`. Shared decode CONTRACT: the TiffImages backend (and
any future GDAL one) only supplies `bands`+`tags`, so the georef math lives once."""
function _assemble_geotiff(bands::AbstractVector, tags::AbstractDict;
                           variables = nothing, band_names = nothing)
    isempty(bands) && throw(ArgumentError("GeoTIFF has no raster bands"))
    nbands = length(bands)
    scale = get(tags, 33550, nothing)
    tie = get(tags, 33922, nothing)
    (scale === nothing || tie === nothing) && throw(ArgumentError(
        "GeoTIFF lacks ModelPixelScaleTag/ModelTiepointTag; cannot derive a grid " *
        "(a non-tiepoint affine georeferencing needs the GDAL backend)."))
    sx, sy = Float64(scale[1]), Float64(scale[2])
    i0, j0 = Float64(tie[1]), Float64(tie[2])
    x0, y0 = Float64(tie[4]), Float64(tie[5])
    H, W = size(bands[1])
    xs = Float64[x0 + (c - i0 + 0.5) * sx for c in 0:(W - 1)]
    ys = Float64[y0 - (r - j0 + 0.5) * sy for r in 0:(H - 1)]
    geographic = _geotiff_geokey(get(tags, 34735, nothing), 1024) != 1
    nodata = _geotiff_nodata(get(tags, 42113, nothing))

    names = band_names === nothing ? ["Band$(i)" for i in 1:nbands] :
            String[String(n) for n in band_names]
    length(names) == nbands || throw(ArgumentError(
        "band_names has $(length(names)) entries but the GeoTIFF has $nbands band(s)"))
    ydim, xdim = geographic ? ("lat", "lon") : ("y", "x")
    want = variables === nothing ? nothing : Set(String[String(v) for v in variables])
    if want !== nothing
        miss = sort!(String[v for v in want if !(v in names)])
        isempty(miss) || throw(KeyError(
            "requested bands not in GeoTIFF: $miss; present bands: $names"))
    end
    vars = Dict{String,NativeField}()
    for (nm, band) in zip(names, bands)
        want !== nothing && !(nm in want) && continue
        data = Array{Float64}(band)                  # copy: we may write NaN below
        if nodata !== nothing && !isnan(nodata)
            @inbounds for k in eachindex(data)
                data[k] == nodata && (data[k] = NaN)
            end
        end
        vars[nm] = NativeField(data, [ydim, xdim], Dict{String,Any}())
    end
    coords = Dict{String,NativeField}(
        xdim => NativeField(xs, [xdim], Dict{String,Any}()),
        ydim => NativeField(ys, [ydim], Dict{String,Any}()))
    return NativeDataset(vars, coords)
end

# --- FF10 point reader (SMOKE FF10_POINT / Emissions.jl oracle) --------------

# The 77 FF10 point column names, in file order. Copied from Emissions.jl
# `src/ff10.jl` `FF10_POINT_COLUMNS`; the first two use the SMOKE FF10_POINT spec
# names COUNTRY_CD / REGION_CD (Emissions.jl names them COUNTRY / FIPS — identical
# values, a positional alias documented in conformance/ff10_oracle_emissions.jl).
const FF10_POINT_COLUMNS = String[
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

# The 42 FF10 point columns decoded to Float64 (blank → NaN). Everything else
# (IDs, codes, free-text FACILITY_NAME, temporal tokens CALC_YEAR/DATE_UPDATED)
# stays String so leading-zero codes (REGION_CD "01001", ZIPCODE "00000", SCC,
# POLID) never become floats. Overridable via the `numeric_columns` kwarg.
const FF10_POINT_NUMERIC = Set{String}([
    "ANN_VALUE", "ANN_PCT_RED", "STKHGT", "STKDIAM", "STKTEMP", "STKFLOW",
    "STKVEL", "LONGITUDE", "LATITUDE", "DESIGN_CAPACITY", "CURRENT_COST",
    "CUMULATIVE_COST", "PROJECTION_FACTOR", "FUG_HEIGHT", "FUG_WIDTH_XDIM",
    "FUG_LENGTH_YDIM", "FUG_ANGLE", "ANNUAL_AVG_HOURS_PER_YEAR",
    "JAN_VALUE", "FEB_VALUE", "MAR_VALUE", "APR_VALUE", "MAY_VALUE", "JUN_VALUE",
    "JUL_VALUE", "AUG_VALUE", "SEP_VALUE", "OCT_VALUE", "NOV_VALUE", "DEC_VALUE",
    "JAN_PCTRED", "FEB_PCTRED", "MAR_PCTRED", "APR_PCTRED", "MAY_PCTRED",
    "JUN_PCTRED", "JUL_PCTRED", "AUG_PCTRED", "SEP_PCTRED", "OCT_PCTRED",
    "NOV_PCTRED", "DEC_PCTRED",
])

"""
    FF10Reader()

The `ff10` format reader — the RAW long-format FF10 **point** table (SMOKE /
Emissions.jl `FF10_POINT`) as a `points` [`NativeDataset`] in **native units**.

Unlike [`CSVReader`] (which only skips empty lines and splits naively), this
reader (a) skips the leading `#` comment header block (`#FORMAT=…`, `#COUNTRY`,
…), (b) applies the fixed 77-column [`FF10_POINT_COLUMNS`] schema — FF10 data
rows carry no clean header row, so the names come from the schema constant
exactly as Emissions.jl supplies them — and (c) does RFC-4180 quote handling so a
free-text `FACILITY_NAME` may embed the delimiter (`"Autauga Plant, Unit 1"`).

Each of the 77 columns becomes one [`NativeField`] on a single `index` dim (one
index per data row); there are no coordinates (`LONGITUDE`/`LATITUDE` are ordinary
variables — a points table has no gridded axis). The 42 numeric columns parse to
`Float64` (blank → `NaN`); the other 35 (IDs/codes/free-text) stay `String`
(blank → `""`).

READER-ONLY (Risk R3): NO pollutant pivot (POLID stays a data column, rows are
not reshaped), NO unit conversion (`STKHGT`/`STKDIAM` stay feet, `STKTEMP` °F,
`STKFLOW` ft³/s, `STKVEL` ft/s, `ANN_VALUE` tons/yr), NO FIPS/SCC normalization,
NO EGU/pollutant filter — those transforms move DOWNSTREAM into the `.esm`.

`reader_kwargs`: `member="path/in/zip"` extracts a named member from a `.zip`
blob (the whole zip stays the cached content-addressed blob; the member is reader
config so it never enters the cache key). `members=[…]` (an explicit list of
member names) and/or `member_glob="*egu*"` (fnmatch-style — `*`, `?`, `[...]`,
case-sensitive, matched against the full member path) select MULTIPLE members:
the selection is the union of the explicit list and the glob matches,
deduplicated, and the members are read and their rows concatenated in ascending
lexicographic (byte) order of member name. An explicit name absent from the
archive, and a glob matching zero members, are errors; directory placeholder
entries (names ending in `/`) are never selected; `member` (singular) is
mutually exclusive with `members`/`member_glob`. None of these enter the cache
key. `skip_header_row=true` handles the EPA 2016fd-style column-header line:
after comment (`#`) and blank lines are dropped, the first remaining line of
each selected input (each selected zip member, or the bare file) must be a
header row — its first delimiter-separated field, compared case-insensitively,
must be `country_cd` — and exactly that one line is skipped per member; if the
first field is anything else the reader errors (the option asserts a header row
and never silently drops a data row). `kind="point"` selects the schema (only
point ships). `numeric_columns`, `delimiter`, `comment` override the defaults;
`variables=[…]` restricts the returned columns (default = all 77)."""
struct FF10Reader <: Reader end

# RFC-4180 quote-aware split of ONE line into fields. A field may be wrapped in
# `"`, may contain the delimiter inside quotes (the free-text FACILITY_NAME), and
# `""` inside a quoted field is a literal quote. Quotes are stripped; the inner
# content is verbatim. The CSVReader's naive `split` cannot do any of this — the
# documented reason ff10 needs its own reader.
function _split_ff10_fields(line::AbstractString, delim::AbstractChar)
    fields = String[]
    buf = IOBuffer()
    inquote = false
    chars = collect(line)
    k = 1
    L = length(chars)
    while k <= L
        c = chars[k]
        if inquote
            if c == '"'
                if k < L && chars[k+1] == '"'   # escaped quote ""
                    write(buf, '"'); k += 1
                else
                    inquote = false
                end
            else
                write(buf, c)
            end
        else
            if c == '"'
                inquote = true
            elseif c == delim
                push!(fields, String(take!(buf)))
            else
                write(buf, c)
            end
        end
        k += 1
    end
    push!(fields, String(take!(buf)))
    return fields
end

# Read a named member of a zip archive as text (UTF-8), via ZipFile.jl. The
# member is reader config — NOT part of the cache key — because one cached
# `2016fd_inputs_point.zip` holds many member CSVs several loaders read.
function _ff10_member_text(path::AbstractString, member::AbstractString)
    reader = ZipFile.Reader(String(path))
    try
        for f in reader.files
            f.name == member && return String(read(f))
        end
        throw(ArgumentError("zip member $(repr(member)) not found in $(path); " *
            "members: $(String[f.name for f in reader.files])"))
    finally
        close(reader)
    end
end

# Translate an fnmatch-style glob (`*` any run incl. empty, `?` one char,
# `[...]`/`[!...]` character class; case-sensitive; an unclosed `[` is literal)
# into an anchored Regex. Semantics match Python `fnmatch.fnmatchcase` and the
# Rust reader's matcher — the cross-language `member_glob` contract.
function _glob_regex(pat::AbstractString)
    io = IOBuffer()
    write(io, '^')
    chars = collect(pat)
    n = length(chars)
    i = 1
    while i <= n
        c = chars[i]
        if c == '*'
            write(io, ".*")
        elseif c == '?'
            write(io, '.')
        elseif c == '['
            j = i + 1
            j <= n && chars[j] == '!' && (j += 1)
            j <= n && chars[j] == ']' && (j += 1)
            while j <= n && chars[j] != ']'
                j += 1
            end
            if j > n
                write(io, "\\[")                    # unclosed class -> literal '['
            else
                inner = join(chars[i+1:j-1])
                inner = replace(inner, "\\" => "\\\\")
                startswith(inner, "!") && (inner = "^" * inner[2:end])
                write(io, '[')
                write(io, inner)
                write(io, ']')
                i = j
            end
        elseif c in ('\\', '^', '$', '.', '|', '+', '(', ')', '{', '}', ']')
            write(io, '\\')
            write(io, c)
        else
            write(io, c)
        end
        i += 1
    end
    write(io, '$')
    return Regex(String(take!(io)))
end

# Resolve `members`/`member_glob` against the archive and return the selected
# member TEXTS in ascending lexicographic (byte) order of member name — the
# deterministic concatenation order. An explicit name absent from the archive,
# and a glob matching zero members, are errors. Selection considers only FILE
# members: directory placeholder entries (names ending in `/`, e.g. the real
# 2016fd zip's `…/ptegu/`) are ignored. Like `member`, this is reader config —
# NOT part of the cache key (the blob is the whole zip).
function _ff10_member_texts(path::AbstractString, members, member_glob)
    reader = ZipFile.Reader(String(path))
    try
        names = String[f.name for f in reader.files if !endswith(f.name, '/')]
        selected = Set{String}()
        if members !== nothing
            want = String[String(m) for m in members]
            miss = sort!(String[m for m in want if !(m in names)])
            isempty(miss) || throw(ArgumentError(
                "zip members $(repr(miss)) not found in $(path); members: $(sort(names))"))
            union!(selected, want)
        end
        if member_glob !== nothing
            re = _glob_regex(String(member_glob))
            hits = String[n for n in names if occursin(re, n)]
            isempty(hits) && throw(ArgumentError(
                "member_glob $(repr(member_glob)) matched no members in $(path); " *
                "members: $(sort(names))"))
            union!(selected, hits)
        end
        isempty(selected) && throw(ArgumentError(
            "members/member_glob selected no zip members"))
        byname = Dict(f.name => f for f in reader.files)
        return [String(read(byname[n])) for n in sort!(collect(selected))]
    finally
        close(reader)
    end
end

# Drop the asserted `country_cd` header line from one input's data lines. The
# first non-comment, non-empty line's first delimiter-separated field must equal
# `country_cd` (case-insensitive) — anything else is an error, so the option can
# never silently drop a data row.
function _ff10_skip_header(data_lines::Vector{<:AbstractString}, delimiter::AbstractString)
    isempty(data_lines) && throw(ArgumentError(
        "skip_header_row: no non-comment lines — the asserted header row is missing"))
    first_field = lowercase(strip(first(split(first(data_lines), delimiter; limit = 2))))
    first_field == "country_cd" || throw(ArgumentError(
        "skip_header_row: first non-comment line does not start with a " *
        "'country_cd' header field (got $(repr(first_field))); refusing to " *
        "drop a data row"))
    return data_lines[2:end]
end

function read_native(::FF10Reader, path::AbstractString;
                     member = nothing, members = nothing, member_glob = nothing,
                     skip_header_row::Bool = false, kind::AbstractString = "point",
                     numeric_columns = FF10_POINT_NUMERIC,
                     delimiter::AbstractString = ",", comment::AbstractString = "#",
                     variables = nothing)
    kind == "point" || throw(ArgumentError(
        "FF10Reader only supports kind=\"point\" (got $(repr(kind))); the 45-col " *
        "nonpoint/onroad/nonroad schemas are not implemented yet"))
    texts = if members !== nothing || member_glob !== nothing
        member === nothing || throw(ArgumentError(
            "`member` is mutually exclusive with `members`/`member_glob`"))
        _ff10_member_texts(String(path), members, member_glob)
    elseif member !== nothing
        [_ff10_member_text(String(path), String(member))]
    else
        [read(String(path), String)]
    end
    delim = first(delimiter)
    ncol = length(FF10_POINT_COLUMNS)

    # Per selected input: skip empty + '#' comment lines, drop the asserted
    # `country_cd` header line (if skip_header_row), then RFC-4180 parse.
    rows = Vector{String}[]
    for text in texts
        data_lines = String[rstrip(ln, ['\r']) for ln in split(text, '\n')
                            if !isempty(strip(ln)) && !startswith(strip(ln), comment)]
        skip_header_row && (data_lines = _ff10_skip_header(data_lines, delimiter))
        for ln in data_lines
            fields = _split_ff10_fields(ln, delim)
            length(fields) == ncol || throw(ArgumentError(
                "FF10 point row has $(length(fields)) fields, expected $ncol; " *
                "row=$(repr(ln))"))
            push!(rows, fields)
        end
    end

    numset = Set(String.(collect(numeric_columns)))
    want = variables === nothing ? nothing : Set(String[String(v) for v in variables])
    if want !== nothing
        miss = sort!(String[v for v in want if !(v in FF10_POINT_COLUMNS)])
        isempty(miss) || throw(KeyError("requested FF10 columns not in schema: $miss"))
    end

    vars = Dict{String,NativeField}()
    for (j, name) in enumerate(FF10_POINT_COLUMNS)
        want !== nothing && !(name in want) && continue
        col = String[r[j] for r in rows]
        data = if name in numset
            Float64[isempty(strip(v)) ? NaN : parse(Float64, strip(v)) for v in col]
        else
            col
        end
        vars[name] = NativeField(data, ["index"], Dict{String,Any}())
    end
    return NativeDataset(vars, Dict{String,NativeField}())
end

# --- Shapefile reader (ESRI shapefile feature table) -------------------------

"""Shape-type code -> name (ESRI Shapefile Technical Description, page 4). The
name is carried in the geometry field's `attrs` so ESS can tell a polygon layer
from a polyline one without re-sniffing the blob."""
const SHAPE_TYPE_NAMES = Dict{Int,String}(
    0 => "Null", 1 => "Point", 3 => "PolyLine", 5 => "Polygon", 8 => "MultiPoint",
    11 => "PointZ", 13 => "PolyLineZ", 15 => "PolygonZ", 18 => "MultiPointZ",
    21 => "PointM", 23 => "PolyLineM", 25 => "PolygonM", 28 => "MultiPointM",
    31 => "MultiPatch")

"""Field names the reader itself produces. A `.dbf` column of the same name is a
collision the reader refuses rather than silently shadowing either side."""
const SHAPEFILE_RESERVED = String[
    "geometry", "n_vertices", "shape_index", "part_index", "n_parts",
    "xmin", "ymin", "xmax", "ymax", "shape_type", "crs_wkt"]

"""
    _shapefile_members(path, member) -> Dict{String,Vector{UInt8}}

The `.shp` + sidecar byte blobs of one shapefile, keyed by lowercase extension.
A shapefile is a **file set** but the content-addressed cache holds ONE blob, so
the fetchable form is a `.zip`; a bare `.shp` blob decodes too, with geometry
only (no `.dbf` attributes, no `.prj`). `member` names the `.shp` inside a zip;
when omitted the archive must contain exactly one."""
function _shapefile_members(path::AbstractString, member)
    magic = open(io -> read(io, 2), path)
    if length(magic) < 2 || magic != UInt8['P', 'K']
        return Dict{String,Vector{UInt8}}("shp" => read(path))
    end
    out = Dict{String,Vector{UInt8}}()
    r = ZipFile.Reader(path)
    try
        names = String[f.name for f in r.files if !endswith(f.name, "/")]
        shps = sort!(String[n for n in names if endswith(lowercase(n), ".shp")])
        target = if member !== nothing
            m = String(member)
            m in names || throw(KeyError(
                "zip member '$m' not in the archive; .shp members present: $shps"))
            m
        elseif length(shps) == 1
            shps[1]
        elseif isempty(shps)
            throw(KeyError("the zip contains no .shp member"))
        else
            throw(KeyError("the zip contains $(length(shps)) .shp members; name one " *
                           "with reader_options.member: $shps"))
        end
        stem = lowercase(target[1:(end - length(".shp"))])
        for f in r.files
            lf = lowercase(f.name)
            if f.name == target
                out["shp"] = read(f)
            elseif startswith(lf, stem * ".")
                ext = lf[(length(stem) + 2):end]
                ext in ("dbf", "shx", "prj") && (out[ext] = read(f))
            end
        end
    finally
        close(r)
    end
    return out
end

"""
    _assemble_shapefile(shapecode, parts, boxes, colnames, colvalues, deleted, crs_wkt;
                        variables=nothing, numeric_columns=nothing) -> NativeDataset

Build the shapefile [`NativeDataset`] from decoded geometry + attributes. The
backend (Shapefile.jl, via `EarthSciIOShapefileExt`) supplies only `parts` (per
shapefile record, its vertex rings in file order as `(x, y)` tuples), `boxes`
(per record, its stored `(xmin, ymin, xmax, ymax)`), the `.dbf` columns and the
`.prj` text; the CONTRACT — one row per PART, attribute replication, the
`esm-spec` §8.6.1 repeat-final-vertex padding, the `N`/`F`→Float64 rule and the
`*`-only deletion rule — lives here, shared with any future backend."""
function _assemble_shapefile(shapecode::Integer,
                             parts::AbstractVector,
                             boxes::AbstractVector,
                             colnames::AbstractVector{<:AbstractString},
                             colvalues::AbstractVector,
                             deleted::AbstractVector{Bool},
                             crs_wkt;
                             variables = nothing, numeric_columns = nothing,
                             nvert_max = nothing)
    clash = sort!(String[c for c in colnames if String(c) in SHAPEFILE_RESERVED])
    isempty(clash) || throw(ArgumentError(
        ".dbf column name(s) $clash collide with the reader's own fields $SHAPEFILE_RESERVED"))
    nshape = length(parts)
    (isempty(colvalues) || all(v -> length(v) == nshape, colvalues)) || throw(ArgumentError(
        "shapefile has $nshape shapes but the .dbf column lengths disagree"))

    # Explode to one row per part, dropping `*`-deleted records whole.
    rings = Vector{Vector{Tuple{Float64,Float64}}}()
    shape_ix, part_ix, nparts, row_of = Int[], Int[], Int[], Int[]
    for si in 1:nshape
        (si <= length(deleted) && deleted[si]) && continue
        ps = parts[si]
        for (pi, ring) in enumerate(ps)
            push!(rings, ring); push!(shape_ix, si - 1); push!(part_ix, pi - 1)
            push!(nparts, length(ps)); push!(row_of, si)
        end
    end

    n = length(rings)
    nvert = isempty(rings) ? 0 : maximum(length, rings)
    if nvert_max !== nothing
        if nvert > Int(nvert_max)
            w = argmax(length.(rings))          # 1-based; the row index is 0-based
            throw(ArgumentError(
                "declared nvert_max=$(Int(nvert_max)) but row $(w - 1) " *
                "(shape $(shape_ix[w]), part $(part_ix[w])) has $nvert vertices"))
        end
        nvert = Int(nvert_max)
    end
    geom = fill(NaN, n, max(nvert, 1), 2)
    for (i, ring) in enumerate(rings)
        for (v, pt) in enumerate(ring)
            geom[i, v, 1] = pt[1]
            geom[i, v, 2] = pt[2]
        end
        if !isempty(ring)   # right-pad by repeating the final vertex (esm-spec 8.6.1)
            geom[i, (length(ring) + 1):end, 1] = fill(ring[end][1], max(nvert, 1) - length(ring))
            geom[i, (length(ring) + 1):end, 2] = fill(ring[end][2], max(nvert, 1) - length(ring))
        end
    end

    vars = Dict{String,NativeField}(
        "geometry" => NativeField(geom, ["index", "vertex", "xy"]),
        "shape_type" => NativeField(
            String[get(SHAPE_TYPE_NAMES, Int(shapecode), string(Int(shapecode)))], ["meta"]),
        "n_vertices" => NativeField(Int64[length(r) for r in rings], ["index"]),
        "shape_index" => NativeField(Int64.(shape_ix), ["index"]),
        "part_index" => NativeField(Int64.(part_ix), ["index"]),
        "n_parts" => NativeField(Int64.(nparts), ["index"]))
    crs_wkt === nothing ||
        (vars["crs_wkt"] = NativeField(String[strip(String(crs_wkt))], ["meta"]))
    for (k, nm) in enumerate(("xmin", "ymin", "xmax", "ymax"))
        vars[nm] = NativeField(Float64[boxes[si][k] for si in row_of], ["index"])
    end

    numset = numeric_columns === nothing ? Set{String}() :
             Set{String}(String(c) for c in numeric_columns)
    unknown = sort!(String[c for c in numset if !(c in String.(colnames))])
    isempty(unknown) || throw(KeyError("numeric_columns names no such .dbf column: $unknown"))
    for (j, nm) in enumerate(colnames)
        name = String(nm)
        col = colvalues[j]
        vals = [col[si] for si in row_of]
        # Bool BEFORE Real: `Bool <: Real` in Julia, so a `.dbf` `L` column would
        # otherwise decode to Float64 here and to bool in the Python/Rust tracks.
        if !(name in numset) && all(v -> v === missing || v isa Bool, vals)
            vars[name] = NativeField(Bool[v === missing ? false : v for v in vals], ["index"])
        elseif name in numset || all(v -> v === missing || v isa Real, vals)
            vars[name] = NativeField(Float64[_shp_float(v) for v in vals], ["index"])
        else
            vars[name] = NativeField(String[_shp_text(v) for v in vals], ["index"])
        end
    end

    if variables !== nothing
        want = Set{String}(String(v) for v in variables)
        missing_names = sort!(String[v for v in want if !haskey(vars, v)])
        isempty(missing_names) && (vars = Dict(k => v for (k, v) in vars if k in want))
        isempty(missing_names) || throw(KeyError(
            "requested variables not in the shapefile: $missing_names; " *
            "present: $(sort!(collect(keys(vars))))"))
    end
    return NativeDataset(vars, Dict{String,NativeField}())
end

"A `.dbf` cell as Float64: blank / missing / unparseable -> `NaN`."
function _shp_float(v)
    v === missing && return NaN
    v isa Bool && return Float64(v)
    v isa Real && return Float64(v)
    s = strip(string(v))
    isempty(s) && return NaN
    p = tryparse(Float64, s)
    return p === nothing ? NaN : p
end

"A `.dbf` cell as String: missing -> `\"\"`; a `D` date -> `YYYYMMDD`."
function _shp_text(v)
    v === missing && return ""
    v isa Dates.Date && return Dates.format(v, "yyyymmdd")
    return v isa AbstractString ? String(strip(v)) : string(v)
end

"""
    ShapefileReader()

The `shapefile` format reader — an ESRI shapefile as a feature table.

Decode is delegated to **Shapefile.jl**, loaded LAZILY via a weakdep extension
(`EarthSciIOShapefileExt`, active on `using Shapefile`) — mirroring the Python
reader's lazy `pyshp` import and the Rust `shapefile` crate. Calling
`read_native` without `using Shapefile` throws a clear install hint.

**One row per PART.** A shapefile record may carry several parts — a polygon's
outer ring plus its holes, a county's mainland plus its islands. The op that
consumes this geometry (`polygon_intersection_area`, `intersect_polygon`) takes
ONE ring, so a reader that surfaced only the first part would silently drop the
islands. Each part becomes one row of the `index` axis, with the record's `.dbf`
attributes REPLICATED across its parts; a single-part layer decodes 1:1.

Variables: `geometry` (`Float64[index, vertex, xy]`, right-PADDED to the longest
part by REPEATING the final vertex — the `esm-spec` §8.6.1 rectangular-storage
convention, which a binding evaluates as the deduplicated ring; a Null shape's
row is all `NaN`), `shape_type` and — when the archive carries a `.prj` —
`crs_wkt` (`String[meta]`, one element each: the layer's shape type and the
projection WKT verbatim; the native CRS is DECLARED, not acted on),
`n_vertices`/`shape_index`/`part_index`/
`n_parts` (`Int64[index]`), `xmin`/`ymin`/`xmax`/`ymax` (`Float64[index]`, the
parent record's STORED bounding box replicated to its parts) and one field per
`.dbf` column (`N`/`F` -> Float64 with blank as `NaN`, `L` -> Bool, `D`/`C` ->
String). A row whose deletion flag is `*` is dropped, and no other flag byte
means deleted (spec/conformance.md §3).

READER-ONLY (Risk R3): no reprojection, no unit conversion, no ring orientation
fix, no polygon/hole classification, no name remap.

`reader_kwargs`: `member="path/in/zip"` names the `.shp` inside a zip blob
(sidecars are the same stem with `.dbf`/`.shx`/`.prj`); it never enters the
cache key. `numeric_columns=[...]` parses the named `C` columns as Float64 — the
`CSVReader`/`FF10Reader` spelling, for a text-typed code column (a FIPS `GEOID`)
a model wants as a number. `nvert_max=N` pads `geometry` to exactly `N` vertex
slots instead of to the longest part, so a DOCUMENT declares the vertex-axis
length rather than inheriting a number the file happens to have (a longer part
is an error naming it, never a silent truncation). `variables=[...]` restricts
the returned fields."""
struct ShapefileReader <: Reader end

# The real decode lives in ext/EarthSciIOShapefileExt.jl, whose method is typed
# `path::AbstractString` — strictly MORE specific than this untyped-`path` fallback,
# so when `using Shapefile` is active it wins by dispatch (no method overwrite,
# which precompilation forbids). This fallback fires only when the backend is absent.
read_native(::ShapefileReader, path; kwargs...) = error(
    "the shapefile reader needs the Shapefile.jl backend: add `using Shapefile` so " *
    "the EarthSciIOShapefileExt extension supplies the decode (kept a weakdep to " *
    "keep a base EarthSciIO install light, mirroring the Python pyshp-optional path).")

# --- Parquet reader (columnar table) ----------------------------------------

"""
    _parquet_kind(T, time_exponent=nothing) -> Symbol

The native dtype an on-disk Parquet column maps onto — `:bool`, `:int32`,
`:int64`, `:float64`, `:string`, or `:unsupported` — keyed by the Julia element
type the decode backend produces for it (with `missing` already stripped off).
This function IS spec/conformance.md §3's "Parquet decode notes" table, and it
lives in the core rather than in the backend so any future Parquet backend
decodes identically.

The narrow/wide integer split is the [`NetCDFReader`]'s VERBATIM (`Int8`/`Int16`/
`Int32`/`UInt8`/`UInt16` → int32; `Int64`/`UInt32`/`UInt64` → int64), restated
here so a MOVES `int32` ID column and a CF `int32` time axis cannot drift apart.
A temporal column rides as its RAW stored integer at its stored width (`Date32`
and a millisecond `Time32` are int32; `Date64`/`Time64`/`Timestamp`/`Duration`
are int64) — the unit and any timezone are NOT applied and NOT reported, the
same rule a CF time axis gets, because turning an epoch offset into a wall-clock
instant is ESS's job (Risk R3). `time_exponent` is the parquet `TimeType` unit
exponent (-3 millis, -6 micros, -9 nanos), the only thing separating a `Time32`
from a `Time64` once a backend has turned both into a `Dates.Time`.

An all-null column (element type `Missing`) is float64 with every cell `NaN` —
the one logical type that can represent every cell of it. A nested or binary
column (`Vector{UInt8}`, `SVector`, `NamedTuple`, …) has no rank-1 reading at
all and is `:unsupported`.
"""
function _parquet_kind(T::Type, time_exponent = nothing)
    T === Missing && return :float64          # an all-null column: float64, all NaN
    T === Bool && return :bool                # BEFORE Integer: Bool <: Integer here
    T <: Union{Int8,Int16,Int32,UInt8,UInt16} && return :int32
    T <: Union{Int64,UInt32,UInt64} && return :int64
    T === Date && return :int32               # raw days since 1970-01-01
    T === Time && return time_exponent == -3 ? :int32 : :int64
    T === DateTime && return :int64
    T <: AbstractFloat && return :float64     # Float16/32/64 and the DecFP decimals
    T <: AbstractString && return :string
    return :unsupported
end

"""
    _assemble_parquet(names, kinds, types, loadcolumn; variables=nothing,
                      float_columns=nothing, null_int=nothing,
                      null_string=nothing) -> NativeDataset

Build the parquet [`NativeDataset`] from a backend's per-column metadata plus a
`loadcolumn(name)` callback that materializes ONE column's cells. The backend
(Parquet2.jl, via `EarthSciIOParquet2Ext`) supplies only `names` (the on-disk
column names in file order), `kinds` ([`_parquet_kind`] per column), `types`
(the backend's element type, for error messages) and the callback; the whole
decode CONTRACT — projection, the null policy, `float_columns` including
decimal-TEXT parsing, the int32 range check — lives here, shared with any future
backend and testable without the weakdep loaded.

`loadcolumn` is called ONLY for the columns that survive the projection, which
is what makes `variables` a real pushdown rather than a read-then-discard: an
unread column chunk is never fetched off disk. A cell is `missing` (the null),
a `Bool`, an `Integer`, a `Real`, or an `AbstractString`.
"""
function _assemble_parquet(names::AbstractVector{<:AbstractString},
                           kinds::AbstractVector{Symbol},
                           types::AbstractVector,
                           loadcolumn;
                           variables = nothing, float_columns = nothing,
                           null_int = nothing, null_string = nothing)
    allnames = String[String(n) for n in names]
    (length(kinds) == length(allnames) && length(types) == length(allnames)) ||
        throw(ArgumentError("parquet column metadata lengths disagree"))

    # Projection. Resolved against the file's own schema FIRST, so an unknown
    # name is this reader's error listing what is present, never a silently
    # missing array (and never the backend's own message).
    # An EMPTY `variables` reads every column (spec/conformance.md §3), the same
    # as passing none at all — never "no columns", which is what a bare
    # `Set()` would have meant.
    want = nothing
    if variables !== nothing && !isempty(variables)
        asked = String[String(v) for v in variables]
        absent = sort!(unique!(String[v for v in asked if !(v in allnames)]))
        isempty(absent) || throw(KeyError(
            "requested variables not in the parquet file: $absent; present: $allnames"))
        want = Set(asked)
    end
    forced = float_columns === nothing ? Set{String}() :
             Set{String}(String(c) for c in float_columns)

    vars = Dict{String,NativeField}()
    nrows = -1
    for (j, name) in enumerate(allnames)
        want === nothing || name in want || continue
        kind = kinds[j]
        if kind === :unsupported
            # Silently skipping a column the document NAMED would hand back a
            # dataset missing an array it asked for, so that is an error;
            # unrequested, it is simply not a native field (the NetCDF reader
            # skips its non-numeric variables the same way).
            want === nothing && continue
            throw(ArgumentError(
                "column $(repr(name)) decodes as $(types[j]), which has no rank-1 " *
                "native reading (nested and binary columns are not supported)"))
        end
        # `float_columns` is a statement about the SOURCE, so it applies to any
        # column that can produce a number — decimal text included.
        dtype = name in forced ? :float64 : kind
        data = _parquet_column(loadcolumn(name), name, dtype, null_int, null_string)
        attrs = Dict{String,Any}()
        # A NaN-folded float carries no surviving sentinel; a DECLARED integer
        # sentinel does, and is reported back exactly as a CF integer fill is.
        # `attrs["fill_value"]` is the spelling because [`NativeField`] has no
        # `fill_value` slot of its own — the same place the Python track reports
        # it, and one `Int64` for both integer widths so the two agree on the
        # value's type as well as the number.
        if null_int !== nothing && (dtype === :int32 || dtype === :int64)
            attrs["fill_value"] = Int64(null_int)
        end
        vars[name] = NativeField(data, ["index"], attrs)
        nrows < 0 && (nrows = length(data))
        length(data) == nrows || throw(ArgumentError(
            "parquet column $(repr(name)) has $(length(data)) rows, but the table has $nrows"))
    end
    return NativeDataset(vars, Dict{String,NativeField}())
end

# One column's cells coerced into its target native dtype. Row numbers in errors
# are 0-BASED, so they name the row as the document (and the Rust/Python tracks)
# count it.
function _parquet_column(cells, name::AbstractString, dtype::Symbol, null_int, null_string)
    n = length(cells)
    if dtype === :float64
        out = Vector{Float64}(undef, n)
        for (i, c) in enumerate(cells)
            out[i] = _parquet_float(c, name, i - 1)
        end
        return out
    elseif dtype === :int64
        out = Vector{Int64}(undef, n)
        for (i, c) in enumerate(cells)
            out[i] = _parquet_int(c, name, i - 1, null_int)
        end
        return out
    elseif dtype === :int32
        out = Vector{Int32}(undef, n)
        for (i, c) in enumerate(cells)
            v = _parquet_int(c, name, i - 1, null_int)
            typemin(Int32) <= v <= typemax(Int32) || throw(ArgumentError(
                "column $(repr(name)) row $(i - 1): value $v does not fit the int32 " *
                "native dtype"))
            out[i] = v % Int32
        end
        return out
    elseif dtype === :string
        out = Vector{String}(undef, n)
        for (i, c) in enumerate(cells)
            out[i] = _parquet_text(c, name, i - 1, null_string)
        end
        return out
    elseif dtype === :bool
        out = Vector{Bool}(undef, n)
        for (i, c) in enumerate(cells)
            out[i] = _parquet_bool(c, name, i - 1)
        end
        return out
    end
    throw(ArgumentError("unknown parquet native dtype $(repr(dtype))"))
end

# A null float is `NaN` — the same fold a CF `_FillValue` gets. A cell of DECIMAL
# TEXT (the column was declared in `float_columns`) is trimmed and parsed; blank
# is `NaN`, matching the FF10/shapefile rule; anything else is an error naming
# the column, the row and the text.
function _parquet_float(c, name::AbstractString, row::Integer)
    c === missing && return NaN
    c isa Bool && throw(ArgumentError(
        "column $(repr(name)) row $row: a boolean cell cannot be read as float64"))
    c isa Real && return Float64(c)
    if c isa AbstractString
        t = strip(String(c))
        isempty(t) && return NaN
        v = tryparse(Float64, t)
        v === nothing && throw(ArgumentError(
            "column $(repr(name)) row $row: $(repr(String(c))) is not a float64 " *
            "(the column is declared in float_columns)"))
        return v
    end
    throw(ArgumentError(
        "column $(repr(name)) row $row: a $(typeof(c)) cell cannot be read as float64"))
end

function _parquet_int(c, name::AbstractString, row::Integer, null_int)
    if c === missing
        null_int === nothing && throw(_parquet_null_error(name, row, "integer", "null_int"))
        return Int64(null_int)
    end
    if c isa Integer && !(c isa Bool)
        # The one integer width that does not fit: refuse rather than wrap into a
        # negative ID.
        c isa Unsigned && c > typemax(Int64) && throw(ArgumentError(
            "column $(repr(name)) row $row: uint64 value $c exceeds the int64 native dtype"))
        return Int64(c)
    end
    throw(ArgumentError(
        "column $(repr(name)) row $row: expected an integer cell, got $(typeof(c))"))
end

function _parquet_text(c, name::AbstractString, row::Integer, null_string)
    if c === missing
        null_string === nothing && throw(_parquet_null_error(name, row, "string", "null_string"))
        return String(null_string)
    end
    c isa AbstractString && return String(c)
    throw(ArgumentError(
        "column $(repr(name)) row $row: expected a text cell, got $(typeof(c))"))
end

function _parquet_bool(c, name::AbstractString, row::Integer)
    c isa Bool && return c
    # No sentinel option: a third boolean state is a float64 column.
    c === missing && throw(ArgumentError(
        "column $(repr(name)) row $row is null, and a boolean native field has no " *
        "missing value; declare the column in the `float_columns` reader option " *
        "if a third state is meant"))
    throw(ArgumentError(
        "column $(repr(name)) row $row: expected a boolean cell, got $(typeof(c))"))
end

# The refusal a null in a type with no missing value gets. It names the way out,
# because "declare a sentinel" is a decision only the document can make.
_parquet_null_error(name::AbstractString, row::Integer, kind::AbstractString,
                    option::AbstractString) = ArgumentError(
    "column $(repr(name)) row $row is null, and a $kind native field has no missing " *
    "value; declare the `$option` reader option to substitute one, or list the " *
    "column in `float_columns` to read it as float64 with NaN")

"""
    ParquetReader()

The `parquet` format reader — an Apache Parquet file as a **flat table**.

Every column becomes a rank-1 field over `index`, keyed by its **on-disk column
name**, and a table produces **no coordinates** (like [`CSVReader`] and
[`ZarrReader`]). `index` has length `num_rows`; a zero-row file still yields
every column, empty and correctly typed (the schema is in the footer, so an
empty table is TYPED, not absent — most of a MOVES fixture's ~770 tables are
empty, and a document binding one must still see the array it named).

Decode is delegated to **Parquet2.jl**, loaded LAZILY via a weakdep extension
(`EarthSciIOParquet2Ext`, active on `using Parquet2`) — mirroring the shapefile
reader's `Shapefile.jl` and the Python track's lazy `pyarrow` import. Calling
`read_native` without `using Parquet2` throws a clear install hint.

**Type mapping** (spec/conformance.md §3, and [`_parquet_kind`]): Parquet carries
an explicit logical type per column, so the dtype is a total function of it —
`Boolean`→Bool; `Int8`/`Int16`/`Int32`/`UInt8`/`UInt16`→Int32 and
`Int64`/`UInt32`/`UInt64`→Int64 (the SAME narrow/wide split as [`NetCDFReader`]);
floats and `Decimal`→Float64 (unscaled ÷ 10^scale); `Utf8`→String; a
dictionary-encoded column reads as its VALUE type, expanded to one value per row;
an all-null column is Float64/all-NaN. **Temporal columns ride as their RAW
stored integer**, undecoded (`Date32`/millisecond-`Time32`→Int32,
`Date64`/`Time64`/`Timestamp`→Int64): the unit and any timezone are not applied,
because an epoch offset → instant is ESS's job (Risk R3). A `uint64` past
`typemax(Int64)` is an error naming the column and row, never a wraparound.
Nested and binary columns have no rank-1 reading: naming one in `variables` is an
error, and unrequested it is simply not a field.

**Null policy.** Nearly every Parquet column is nullable in its schema whether or
not it holds a null (a table exported from a relational database usually marks
every column nullable), so nullability cannot pick the dtype. A null in a FLOAT
column becomes `NaN`; a null in an INTEGER, STRING or BOOLEAN column is an
ERROR naming the column and the row, because those types have no NaN and any
default would be a real value silently standing in for a missing one.

`reader_kwargs`: `variables=[...]` is the projection and is PUSHED DOWN — only
those column chunks are read off disk (these tables are wide and a document
typically wants three columns); an absent name is an error listing what is
present. `float_columns=[...]` forces the named columns to Float64 whatever
their on-disk type — the Parquet twin of [`ShapefileReader`]'s `numeric_columns`
— and does double duty: an integer column whose missing cells must be `NaN`, AND
a column of fixed-decimal TEXT (a corpus needing byte-reproducible floats stores
them as decimal strings; the MOVES snapshots write `meanBaseRate` as
`"261.000000000000"`), trimmed and parsed, blank→`NaN`, anything else
unparseable an error naming column/row/text. `null_int=n` substitutes an integer
sentinel for a null AND reports it back in the field's `attrs["fill_value"]`
(an integer sentinel cannot be NaN, so it survives exactly as a CF integer fill
does); `null_string=s` substitutes text, which is then indistinguishable from a
real cell holding it — which is why the document has to choose it. A boolean has
no such option.

READER-ONLY (Risk R3): row selection, `record_filter`, `codes` and `extent` are
esm-spec §8.9 work DOWNSTREAM of the decode (and a whole-file reader never even
sees a `select`), and there is no name remap and no unit conversion.

!!! note "Parquet2.jl backend limits"
    A file containing a nested (`list`/`struct`/`map`) column cannot be OPENED by
    Parquet2.jl at all, so such a file errors rather than decoding its readable
    columns — the one place this track cannot reach the §3 contract's "unrequested,
    it is simply not a field". Parquet2.jl decodes a `Timestamp` into a
    millisecond-resolution `DateTime` before this reader sees it, so the raw
    integer of a MICROS/NANOS timestamp column is recovered only to millisecond
    granularity. A `Float16` column is not decoded at all, so it is simply not a
    field here where the other two tracks return float64. And it decodes a
    `Decimal` into a `Dec64` (16 significant decimal digits) during page decode,
    so a wider `Decimal128`/`Decimal256` cannot be read here at the full precision
    the Rust and Python tracks get from the unscaled integer; within binary64's
    exact range the three agree bit for bit (see `ext/EarthSciIOParquet2Ext.jl`).

    One further Parquet2.jl defect is REPAIRED rather than tolerated: it folds a
    `FIXED_LEN_BYTE_ARRAY` decimal's big-endian two's-complement bytes into an
    `Int64` with no sign extension, so every negative cell in a column narrower
    than 8 bytes decoded as its huge unsigned reinterpretation. The extension
    undoes that exactly. The one width it cannot reach is 7 bytes (pyarrow
    precision 15-16), where the unsigned fold overflows `Dec64` inside the
    backend; a negative there is an error naming the limit, never a wrong number.
"""
struct ParquetReader <: Reader end

# The real decode lives in ext/EarthSciIOParquet2Ext.jl, whose method is typed
# `path::AbstractString` — strictly MORE specific than this untyped-`path`
# fallback, so when `using Parquet2` is active it wins by dispatch (no method
# overwrite, which precompilation forbids). This fallback fires only when the
# backend is absent.
read_native(::ParquetReader, path; kwargs...) = error(
    "the parquet reader needs the Parquet2.jl backend: add `using Parquet2` so " *
    "the EarthSciIOParquet2Ext extension supplies the decode (kept a weakdep to " *
    "keep a base EarthSciIO install light, mirroring the Shapefile.jl path).")

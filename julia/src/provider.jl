# The cadence Provider — component (b) (esio-9nb.5). A Provider fulfils the ESS
# data-loader CONTRACT for one `.esm` node: it resolves a URL (per time, for a
# time-varying source), fetches it through the content-addressed cache
# (component a), decodes it with the named `FORMAT_REGISTRY` reader, and returns
# RAW native-grid arrays. Variable remap / unit conversion / regrid are NOT here
# — they stay in ESS/ESD (Risk R3).
#
# It provides DATA, not a solver (the sanctioned impure I/O boundary). The
# library EXPOSES the provider and its `refresh_times`; the USER/solver drives
# the discrete-cadence update (e.g. a DifferentialEquations `PresetTimeCallback`
# that calls `refresh(provider, t)` at each tick). No solver is embedded here —
# `library-exposes-rhs-not-solver`.

"""
    Cadence

Temporal cadence of a provider's data (the `.esm` node's `temporal-cadence`):

  * `CONST` — time-invariant. `refresh_times` is empty; the data is materialized
    once and never refreshed.
  * `DISCRETE` — changes at known time points. `refresh_times` returns that grid;
    the solver refreshes at each tick."""
@enum Cadence CONST DISCRETE

"""
    Provider(cache, url; format, cadence=CONST, times=Float64[], time_dim=nothing,
             variables=nothing, reader_kwargs=NamedTuple(),
             source_loader=nothing, auth_realm=nothing)

A data provider over the shared cache.

  * `url` — a resolved-URL `String` (constant source) OR a function `t -> url`
    (a time-varying source whose file/key changes per tick).
  * `format` — reader name in [`FORMAT_REGISTRY`] (`"netcdf"`, `"csv"`, …).
  * `cadence` — [`CONST`] or [`DISCRETE`]. `DISCRETE` requires non-empty `times`;
    `CONST` requires empty `times`.
  * `times` — the discrete cadence grid (sorted on construction); what
    [`refresh_times`] returns. It may span MANY files: pair it with a `t -> url`
    resolver and the cadence walks from one file to the next (see `time_dim`).
  * `time_dim` — when set on a `DISCRETE` provider whose files hold the cadence
    on an internal axis (e.g. a daily file of hourly steps),
    [`materialize`]`(p, t)` slices that dimension to the tick's record.

    The record is located WITHIN its file, not within the whole cadence: tick
    `i` is record `mod1(i, L)` of the file the resolver returns for it, where `L`
    is that file's length along `time_dim`. So a cadence of 56 three-hourly ticks
    over seven 8-record daily files reads record 1 of day 2 at tick 9, not
    record 9 of day 1 (which does not exist). A single file holding the whole
    cadence is the `L >= length(times)` case, where `mod1` is the identity — the
    behaviour is unchanged. This assumes files are UNIFORM in length and that
    their records are consecutive on the cadence; a short/trimmed file mid-run
    shifts every later tick (the Python track avoids that by matching the file's
    own decoded time axis, which the Julia track cannot do because `times` is a
    caller-defined grid rather than the file's raw axis).
  * `records_per_sample` — `nothing` or `1` (default) returns the SINGLE
    at-or-before record with `time_dim` DROPPED (held piecewise-constant); `2`
    returns the TWO bracketing records (floor + successor) with `time_dim`
    RETAINED at length 2 and a canonical 2-element `time_dim` coordinate of Unix
    epoch seconds, so a downstream model interpolates in time. `2` requires a
    `time_dim`; the successor is read across a file boundary when needed, and at
    the last cadence tick the bracket degenerates to `[last, last]` (equal
    timestamps) so the downstream weight clamps. The provider does pure I/O
    (returns N records) and performs no interpolation itself.
  * `variables` — restrict the returned data variables (coords always kept);
    `nothing` returns all.
  * `reader_kwargs` — extra keywords forwarded to [`read_native`] (e.g. the CSV
    reader's `numeric_columns`).
  * `source_loader` / `auth_realm` — recorded in the cache manifest on fetch."""
struct Provider
    cache::Cache
    format::String
    cadence::Cadence
    times::Vector{Float64}
    url_for::Function
    time_dim::Union{Nothing,String}
    variables::Union{Nothing,Vector{String}}
    reader_kwargs::Dict{Symbol,Any}
    source_loader::Union{Nothing,String}
    auth_realm::Union{Nothing,String}
    records_per_sample::Union{Nothing,Int}
end

function Provider(cache::Cache, url; format::AbstractString,
                  cadence::Cadence = CONST, times = Float64[],
                  time_dim = nothing, variables = nothing,
                  reader_kwargs = NamedTuple(),
                  source_loader = nothing, auth_realm = nothing,
                  records_per_sample = nothing)
    haskey(FORMAT_REGISTRY, format) || throw(ArgumentError(
        "format '$format' is not registered in the format registry; " *
        "registered: $(registered_names(FORMAT_REGISTRY))"))
    tvec = sort!(Float64[t for t in times])
    if cadence == CONST && !isempty(tvec)
        throw(ArgumentError(
            "a CONST provider has no refresh times, got $(length(tvec)) — use cadence=DISCRETE"))
    elseif cadence == DISCRETE && isempty(tvec)
        throw(ArgumentError(
            "a DISCRETE provider requires a non-empty cadence (times=...)"))
    end
    time_dim === nothing || cadence == DISCRETE ||
        throw(ArgumentError("time_dim only applies to a DISCRETE provider"))
    records_per_sample === nothing || records_per_sample == 1 || records_per_sample == 2 ||
        throw(ArgumentError(
            "records_per_sample must be 1 or 2, got $(repr(records_per_sample))"))
    records_per_sample != 2 || time_dim !== nothing ||
        throw(ArgumentError(
            "records_per_sample=2 needs a time_dim to bracket along"))
    kw = Dict{Symbol,Any}(pairs(reader_kwargs))
    _check_reader_kwargs(FORMAT_REGISTRY[format], String(format), kw)
    url_for = url isa AbstractString ? (let u = String(url); _ -> u; end) : url
    return Provider(cache, String(format), cadence, tvec, url_for,
                    time_dim === nothing ? nothing : String(time_dim),
                    variables === nothing ? nothing : String.(collect(variables)),
                    kw,
                    source_loader === nothing ? nothing : String(source_loader),
                    auth_realm === nothing ? nothing : String(auth_realm),
                    records_per_sample === nothing ? nothing : Int(records_per_sample))
end

# The loader's declared decode options, checked against the bound reader AT
# CONSTRUCTION (spec/registries.md §2.1). An unrecognised option is an ERROR
# here, never an ignored key, so a mis-typed `member_filter` cannot quietly
# select nothing and surface much later as wrong numbers. The accepted set is
# the reader's own keyword declaration (`reader_option_keys`), so this can never
# disagree with what `read_native` / `read_store` will actually honour.
function _check_reader_kwargs(reader, format::String, kw::AbstractDict{Symbol,<:Any})
    isempty(kw) && return nothing
    accepted = reader_option_keys(reader)
    accepted === nothing && return nothing        # a reader that slurps everything
    unknown = sort!(String[String(k) for k in keys(kw) if !(k in accepted)])
    isempty(unknown) && return nothing
    throw(ArgumentError(
        "the '$format' reader does not recognise reader option(s) " *
        "$(join(unknown, ", ")); it takes $(join(sort!(String.(accepted)), ", ")). " *
        "An unrecognised option is an error rather than an ignored key " *
        "(spec/registries.md §2.1) — a mis-typed option would otherwise decode " *
        "something else, silently"))
end

"""Thin constructor for a time-invariant ([`CONST`]) provider."""
const_provider(cache::Cache, url; kwargs...) =
    Provider(cache, url; cadence = CONST, kwargs...)

"""Thin constructor for a time-varying ([`DISCRETE`]) provider over `times`."""
discrete_provider(cache::Cache, url, times; kwargs...) =
    Provider(cache, url; cadence = DISCRETE, times = times, kwargs...)

"""
    refresh_times(p::Provider) -> Vector{Float64}

The discrete time points at which the data changes and the solver must
[`refresh`]. Empty for a [`CONST`] provider; the sorted cadence grid for a
[`DISCRETE`] one. The library exposes these; the user wires them into the
solver (e.g. `PresetTimeCallback(refresh_times(p), …)`)."""
refresh_times(p::Provider) = copy(p.times)

"""True if `p`'s data is time-invariant ([`CONST`])."""
is_const(p::Provider) = p.cadence == CONST

# The format-reader instance backing `p` (mirrors the `_load` lookup so
# `supports_selection`/`array_shape` resolve the SAME reader the sample path uses).
_reader_for(p::Provider) = FORMAT_REGISTRY[p.format]

# Fetch (cache) + decode (format reader) for the URL resolved at `t`.
#
# `select` is an optional PER-CALL projection override (component (b) pushdown):
# `nothing` uses the baked `reader_kwargs[:select]`, a non-`nothing` value
# OVERRIDES it for this call only. Any reader that answers `supports_selection`
# can honour it — a store-backed one by fetching fewer objects, a whole-file one
# by materialising only the requested hyperslab of the blob it already fetched.
# A `select` handed to a reader that does neither is a clear error, raised BEFORE
# any fetch (the fetch-full fallback belongs to the EarthSciAST caller, not here).
function _load(p::Provider, t; select = nothing, records = nothing)
    reader = _reader_for(p)
    # Store-backed readers (e.g. zarr) are handed (cache, base_url; variables,
    # select): a Zarr `url` is a directory-like prefix, not a fetchable blob, so
    # the reader fetches individual objects on demand. Active whole-file readers
    # inherit `store_backed = false`.
    if store_backed(reader)
        # Effective select: a per-call `select` OVERRIDES the baked
        # reader_kwargs[:select]. Forward it explicitly and splat the REST of
        # reader_kwargs (with `:select` removed, so `select` is never passed twice).
        records === nothing || throw(ArgumentError(
            "a store-backed reader takes its record selection through `select`, " *
            "not `records`"))
        effective = select === nothing ? get(p.reader_kwargs, :select, nothing) : select
        rest = Dict{Symbol,Any}(k => v for (k, v) in p.reader_kwargs if k !== :select)
        return read_store(reader, p.cache, p.url_for(t);
                          variables = p.variables, select = effective, rest...)
    end
    # Whole-file reader. A `select` — per-call, else the baked one — reaches it
    # only if it declares `supports_selection`; the netcdf reader does, and honours
    # it at DECODE time (same blob, same cache key, only the requested hyperslab
    # materialised). Checked before the fetch so a refusal costs nothing.
    effective = select === nothing ? get(p.reader_kwargs, :select, nothing) : select
    effective === nothing || supports_selection(reader) || throw(ArgumentError(
        "reader $(typeof(reader)) does not support select/pushdown"))
    entry = fetch_blob(p.cache, p.url_for(t);
                       source_loader = p.source_loader, auth_realm = p.auth_realm)
    # PROJECTION PUSHDOWN for a whole-file reader that declares a `variables`
    # decode option (the `parquet` reader): the loader's `variables` reach the
    # READER, so only those column chunks come off disk — spec/conformance.md §3
    # "Projection pushdown" — instead of being applied to an already-decoded
    # dataset. That is not only a speed matter on a table dozens of columns
    # wide: a column the document never named must not be able to FAIL the read
    # (a null-bearing integer column with no `null_int` declared would), which
    # is what the Python and Rust tracks already do. A reader with no such
    # option keeps the read-everything-then-`_select` path below.
    #
    # An EMPTY `variables` means "every variable", NOT "no variables"
    # (spec/conformance.md §3; the Python and Rust tracks both test the list for
    # emptiness before using it). Normalised once here so the pushdown and the
    # `_select` below agree — an empty list used to hand back an empty dataset.
    wanted = (p.variables === nothing || isempty(p.variables)) ? nothing : p.variables
    # `:select` is forwarded explicitly (the per-call value wins over the baked
    # one), so it is stripped from the splat rather than passed twice.
    kw = Dict{Symbol,Any}(k => v for (k, v) in p.reader_kwargs if k !== :select)
    effective === nothing || (kw[:select] = effective)
    if wanted !== nothing && !haskey(kw, :variables)
        opts = reader_option_keys(reader)
        if opts !== nothing && :variables in opts
            kw[:variables] = wanted
        end
    end
    # RECORD PUSHDOWN (`_load_ticks`): the records this file must yield, already
    # resolved to the file's own 0-based numbering by the caller — the cadence
    # never reaches the reader. Only set on the pushdown path, which has already
    # checked the reader declares the option.
    records === nothing || (kw[:records] = records)
    nds = read_native(reader, entry.path; kw...)
    return wanted === nothing ? nds : _select(nds, wanted)
end

"""
    supports_selection(p::Provider) -> Bool

True when `p`'s format reader can honour an orthogonal `select` **without
materialising the whole array**. A caller uses this to decide whether to push a
projection down (via `materialize(p, t; select=…)`) or to read whole and slice on
its own side.

It says nothing about what is FETCHED — pair it with [`store_backed`] for that:

| `supports_selection` | `store_backed` | what a `select` saves | reader |
|---|---|---|---|
| `true` | `true`  | fetch **and** decode — only the intersecting objects are downloaded | `zarr` |
| `true` | `false` | decode only — the whole blob is still fetched under the same cache key | `netcdf` |
| `false` | — | nothing; a `select` is an error | `csv`, `ff10`, `parquet`, … |"""
supports_selection(p::Provider) = supports_selection(_reader_for(p))

"""
    store_backed(p::Provider) -> Bool

True when `p`'s format reader reads a directory-like STORE (a Zarr v2 store, whose
`.zarray`/`.zattrs`/chunks are each their own object) rather than one fetchable
blob. Read with [`supports_selection`] it tells a caller whether a pushed-down
`select` shrinks the DOWNLOAD or only the decode — see the table there."""
store_backed(p::Provider) = store_backed(_reader_for(p))

"""
    array_shape(p::Provider, var::AbstractString) -> Union{Nothing,NTuple{N,Int}}

The full native (dims-order) shape of on-disk array `var`, for a honour/refuse
pushdown decision. For a store-backed zarr provider this reads ONLY the array's
`.zarray` metadata (never a chunk); `nothing` for a whole-file reader (whose
shape is not knowable without reading the blob)."""
function array_shape(p::Provider, var::AbstractString)
    reader = _reader_for(p)
    store_backed(reader) || return nothing
    t0 = p.cadence == CONST ? 0.0 : first(p.times)
    return array_shape(reader, p.cache, p.url_for(t0), var)
end

function _select(nds::NativeDataset, want::Vector{String})
    keep = Dict{String,NativeField}()
    for name in want
        haskey(nds.variables, name) || throw(ArgumentError(
            "requested variable '$name' not in blob; present: $(variable_names(nds))"))
        keep[name] = nds.variables[name]
    end
    return NativeDataset(keep, nds.coords)
end

# Index of the cadence tick active at `t`: exact match, else the last tick ≤ t
# (the currently-in-effect record). Errors if `t` precedes the first tick.
function _tick_index(p::Provider, t::Real)
    tf = Float64(t)
    i = findfirst(==(tf), p.times)
    i === nothing || return i
    j = searchsortedlast(p.times, tf)
    j >= 1 && return j
    throw(ArgumentError(
        "t=$t precedes the provider's first refresh time $(first(p.times))"))
end

# The record index of cadence tick `tick` WITHIN its own file, given that file's
# length `len` along `time_dim`. The cadence is global and files are local: tick
# 9 of a 56-tick cadence laid over 8-record daily files is record 1 of day 2, not
# record 9 of day 1. `len <= 0` means nothing in the blob carries `time_dim`, so
# there is no wrap to apply and the tick passes through (the caller's slice then
# reports the real out-of-range error rather than a silently wrong record).
_file_record(tick::Integer, len::Integer) = len <= 0 ? Int(tick) : mod1(Int(tick), Int(len))

# --- record pushdown --------------------------------------------------------
# Can this provider hand its record CHOICE to the reader, instead of decoding the
# whole cadence axis and slicing afterwards? Three things must hold: the reader
# declares a `records` decode option (the same `reader_option_keys` rule the
# `variables` projection uses), the caller has not baked a `records` of its own
# into `reader_kwargs`, and the source is a whole-file blob — a store-backed
# reader narrows records through `select`, whose chunk arithmetic already shrinks
# the FETCH, and must not be handed a second, conflicting mechanism.
function _records_pushdown(p::Provider)
    p.time_dim === nothing && return false
    reader = _reader_for(p)
    store_backed(reader) && return false
    haskey(p.reader_kwargs, :records) && return false
    opts = reader_option_keys(reader)
    return opts !== nothing && :records in opts
end

# Decode the file holding cadence tick `ticks[1]`, keeping ONLY the records those
# ticks name, and return `(nds, positions)` — `positions[k]` is where `ticks[k]`
# sits along `time_dim` in `nds`.
#
# Every tick in `ticks` must land in the SAME file; the caller checks that by URL
# before asking. The cadence→record arithmetic (`_file_record`, i.e. `mod1` over
# the file's own length) stays HERE, on the side that owns the cadence: the
# reader is handed absolute 0-based record indices and never learns of `times`,
# `records_per_sample` or a file boundary. That is why the length has to be read
# from metadata first — a record cannot be chosen before the decode without it.
#
# A reader that takes no `records` option falls back to the historical path:
# decode the axis whole, return the record indices to slice at. The two paths
# differ in what is DECODED and in nothing else.
function _load_ticks(p::Provider, ticks::AbstractVector{<:Integer}; select = nothing)
    dim = p.time_dim::String
    t_url = p.times[first(ticks)]
    if !_records_pushdown(p)
        nds = _load(p, t_url; select = select)
        len = _time_len(nds, dim)
        return (nds, Int[_file_record(tk, len) for tk in ticks])
    end
    # The record a tick names is `mod1(tick, len)`, so the pushdown needs the
    # file's length along `dim` — which only the decode's own open knows. Rather
    # than open the blob twice (measured: on a GEOS-FP A1 file the extra open
    # costs more than the records it saves), hand the reader THIS CLOSURE: it is
    # called with `len` inside the reader's single open, and the cadence
    # arithmetic still happens here, in the Provider, exactly as before.
    fired = Ref(false)
    resolve = function (len::Integer)
        fired[] = true
        return Int[_file_record(tk, Int(len)) - 1 for tk in ticks]   # 0-based
    end
    nds = _load(p, t_url; select = select,
                records = Dict{String,Any}("dim" => dim, "indices" => resolve))
    # The resolver is not called when the blob has no such dimension. That is the
    # `len <= 0` case of `_file_record`, and it must behave exactly as it always
    # did: nothing was narrowed, so the tick passes through to the slice.
    fired[] && return (nds, collect(1:length(ticks)))
    return (nds, Int[_file_record(tk, _time_len(nds, dim)) for tk in ticks])
end

# Slice `dim` out of every variable that carries it, at record `idx`; drop the
# now-singular dimension and its coordinate. Used for an internal-axis DISCRETE
# source (many time records per file).
function _slice_dim(nds::NativeDataset, dim::String, idx::Integer)
    vars = Dict{String,NativeField}()
    for (name, f) in nds.variables
        pos = findfirst(==(dim), f.dims)
        if pos === nothing
            vars[name] = f
        else
            sliced = collect(selectdim(f.data, pos, idx))
            vars[name] = NativeField(sliced, [d for d in f.dims if d != dim], f.attrs)
        end
    end
    coords = Dict(k => v for (k, v) in nds.coords if k != dim)
    return NativeDataset(vars, coords)
end

# --- 2-record bracket mode (records_per_sample=2) ---------------------------
# Mirrors the Python `Provider._refresh_bracket`: return the TWO records that
# bracket `t` (the floor tick + its successor) with `time_dim` RETAINED at length
# 2 and a canonical 2-element epoch-seconds `time_dim` coordinate, so a downstream
# model interpolates in time. The floor record is `_tick_index` (the same data
# index `_slice_dim` uses); the successor is the next data index along the axis,
# or — when that overruns the file — record 1 of the next file (`url_for(t_next)`,
# re-decoded since the Julia provider keeps no file buffer). At the last cadence
# tick there is no successor: the bracket degenerates to `[last, last]` (equal
# timestamps) so the downstream weight clamps — bracket mode never throws at the
# end of data.

const _CF_UNIT_SECONDS = Dict{String,Float64}(
    "second" => 1.0, "seconds" => 1.0, "sec" => 1.0, "secs" => 1.0, "s" => 1.0,
    "minute" => 60.0, "minutes" => 60.0, "min" => 60.0, "mins" => 60.0,
    "hour" => 3600.0, "hours" => 3600.0, "hr" => 3600.0, "hrs" => 3600.0, "h" => 3600.0,
    "day" => 86400.0, "days" => 86400.0, "d" => 86400.0)

# Parse a CF reference date (the `<ref>` in "<unit> since <ref>") → Unix epoch
# seconds, or `nothing`. Handles "yyyy-mm-dd[ HH[:MM[:SS]]]" with an optional
# `T`/space separator and a trailing `Z`/`UTC`/±offset (treated as UTC).
function _parse_cf_reference(s)
    str = strip(replace(String(s), 'T' => ' '))
    str = strip(replace(str, r"\s*(Z|UTC|[+-]\d{2}:?\d{2}(:\d{2})?)\s*$" => ""))
    for fmt in (dateformat"yyyy-mm-dd HH:MM:SS", dateformat"yyyy-mm-dd HH:MM",
                dateformat"yyyy-mm-dd HH", dateformat"yyyy-mm-dd")
        dt = tryparse(DateTime, str, fmt)
        dt === nothing || return datetime2unix(dt)
    end
    return nothing
end

# Parse a CF "<unit> since <reference>" units string → (ref_epoch_seconds,
# unit_seconds), or `nothing` when it can't be decoded (non-"since" units, an
# unknown step, or an unparseable reference) — the caller then emits raw times.
function _cf_time_scale(units)
    units === nothing && return nothing
    m = match(r"^\s*([A-Za-z]+)\s+since\s+(.+?)\s*$", String(units))
    m === nothing && return nothing
    step = get(_CF_UNIT_SECONDS, lowercase(m.captures[1]), nothing)
    step === nothing && return nothing
    ref = _parse_cf_reference(m.captures[2])
    ref === nothing && return nothing
    return (ref, step)
end

# Convert a raw cadence-grid value to Unix epoch seconds via a decoded CF scale;
# with no scale (units absent/undecodable) fall back to the raw value (documented
# deviation from the epoch-seconds contract).
_raw_to_epoch(raw, scale) =
    scale === nothing ? Float64(raw) : (scale[1] + Float64(raw) * scale[2])

# The CF `units` string of the `dim` coordinate in `nds`, or `nothing`.
function _time_units(nds::NativeDataset, dim::String)
    haskey(nds.coords, dim) || return nothing
    return get(nds.coords[dim].attrs, "units", nothing)
end

# Length of `nds` along `dim` (from the coordinate, else the first variable
# carrying it); 0 if nothing carries it.
function _time_len(nds::NativeDataset, dim::String)
    if haskey(nds.coords, dim)
        c = nds.coords[dim]
        pos = findfirst(==(dim), c.dims)
        pos === nothing || return size(c.data, pos)
    end
    for f in values(nds.variables)
        pos = findfirst(==(dim), f.dims)
        pos === nothing || return size(f.data, pos)
    end
    return 0
end

# Stack record `i0` of `a0` and record `i1` of `a1` along axis `pos`, keeping that
# axis at length 2 (floor, then successor). Range-index each so the axis survives,
# then `cat` — this is uniform across same-file, cross-file, and degenerate cases.
_stack_bracket(a0, i0::Integer, a1, i1::Integer, pos::Integer) =
    collect(cat(selectdim(a0, pos, i0:i0), selectdim(a1, pos, i1:i1); dims = pos))

# Assemble the 2-record bracket: every variable carrying `dim` gets records
# (`nds0`,`i0`) and (`nds1`,`i1`) stacked to a size-2 `dim` axis (dims/order/attrs
# preserved); non-temporal variables and non-`dim` coords pass through from
# `nds0`; `dim` becomes a 2-element epoch-seconds coordinate `[t0, t1]`.
function _bracket_build(nds0::NativeDataset, i0::Integer,
                        nds1::NativeDataset, i1::Integer,
                        dim::String, t0::Float64, t1::Float64)
    vars = Dict{String,NativeField}()
    for (name, f0) in nds0.variables
        pos = findfirst(==(dim), f0.dims)
        if pos === nothing
            vars[name] = f0                                  # non-temporal: unchanged
        else
            stacked = _stack_bracket(f0.data, i0, nds1.variables[name].data, i1, pos)
            vars[name] = NativeField(stacked, copy(f0.dims), f0.attrs)
        end
    end
    coords = Dict{String,NativeField}()
    for (k, v) in nds0.coords
        k == dim && continue                                 # replaced below
        coords[k] = v
    end
    coords[dim] = NativeField(Float64[t0, t1], [dim],
        Dict{String,Any}("units" => "seconds since 1970-01-01T00:00:00Z",
                         "calendar" => "standard"))
    return NativeDataset(vars, coords)
end

function _bracket(p::Provider, t::Real; select = nothing)
    dim  = p.time_dim::String
    tick = _tick_index(p, t)                 # floor tick on the GLOBAL cadence
    # Resolve the URL at the TICK time, not at the query time `t`: the record we
    # are about to slice is the tick's, so it must come from the tick's file. The
    # two agree for any resolver that is constant across a file's own span, but
    # only the tick time is guaranteed to name the file that holds the record.
    u0 = p.url_for(p.times[tick])
    last_tick = tick >= length(p.times)      # no successor → degenerate bracket
    same_file = !last_tick && p.url_for(p.times[tick + 1]) == u0
    # ONE read covers the same-file bracket: both records come out of the same
    # decode, and with a `records` pushdown that decode materialises those two
    # records and no others. The end-of-data and cross-file cases ask this file
    # for the floor record alone — the successor is either absent (it degenerates
    # to the floor) or in the next file, read below.
    ticks0 = same_file ? Int[tick, tick + 1] : Int[tick]
    nds0, pos0 = _load_ticks(p, ticks0; select = select)
    scale = _cf_time_scale(_time_units(nds0, dim))
    t0 = _raw_to_epoch(p.times[tick], scale)

    if last_tick                             # last tick / past end → degenerate
        return _bracket_build(nds0, pos0[1], nds0, pos0[1], dim, t0, t0)
    end

    t1 = _raw_to_epoch(p.times[tick + 1], scale)
    if same_file                             # successor in the same file
        return _bracket_build(nds0, pos0[1], nds0, pos0[2], dim, t0, t1)
    end
    # Successor: record 1 of the NEXT file, re-decoded because the provider keeps
    # no file buffer (and, with the pushdown, one record of it rather than all).
    nds1, pos1 = _load_ticks(p, Int[tick + 1]; select = select)
    return _bracket_build(nds0, pos0[1], nds1, pos1[1], dim, t0, t1)
end

"""
    materialize(p::Provider, t::Real; select=nothing) -> NativeDataset
    materialize(p::Provider; select=nothing) -> NativeDataset

Return the native arrays for the source at time `t`. For a [`DISCRETE`] provider
with `time_dim`, the internal cadence axis is sliced to `t`'s record — unless
`records_per_sample=2`, in which case the two bracketing records (floor +
successor) are returned with `time_dim` retained at length 2 and a 2-element
epoch-seconds `time_dim` coordinate (see the [`Provider`] `records_per_sample`
field). The no-argument form is for a [`CONST`] provider (a `DISCRETE` provider
must be given a time).

`select` is an optional PER-CALL projection pushdown (in EarthSciIO's native
select shape, e.g. `Dict("axes"=>[...])` with 0-based indices). When supplied it
OVERRIDES any baked `reader_kwargs[:select]` for this call only, letting a caller
(EarthSciAST) push a projection down at sample time without rebuilding the
provider. Any reader that answers [`supports_selection`] can honour it — the
`zarr` reader by fetching only the intersecting chunk objects, the `netcdf` one by
materialising only the requested hyperslab of the blob it already fetched;
passing `select` to a reader that cannot is an `ArgumentError`."""
function materialize(p::Provider, t::Real; select = nothing)
    if p.records_per_sample == 2 && p.time_dim !== nothing
        return _bracket(p, t; select = select)
    end
    if p.time_dim !== nothing
        tick = _tick_index(p, t)
        # The TICK's file (see `_bracket`), and — where the reader takes a
        # `records` pushdown — only the tick's record out of it.
        nds, pos = _load_ticks(p, Int[tick]; select = select)
        return _slice_dim(nds, p.time_dim, pos[1])
    end
    return _load(p, t; select = select)
end

function materialize(p::Provider; select = nothing)
    p.cadence == CONST || throw(ArgumentError(
        "a DISCRETE provider needs a time: materialize(p, t) / refresh(p, t)"))
    return materialize(p, 0.0; select = select)
end

"""
    refresh(p::Provider, t::Real; select=nothing) -> NativeDataset

Re-materialize at the cadence tick `t`. This is the call the solver makes from
its `PresetTimeCallback` at each of [`refresh_times`]`(p)`. Identical to
[`materialize`]`(p, t)`; named for the solver-side update site. `select` is the
per-call projection pushdown override (see [`materialize`])."""
refresh(p::Provider, t::Real; select = nothing) = materialize(p, t; select = select)

"""
    prefetch(p::Provider) -> Vector{CacheEntry}

Warm the cache for every URL the provider will need — the single URL for a
[`CONST`] provider, the unique per-tick URLs for a [`DISCRETE`] one — WITHOUT
decoding. Lets a caller pull all blobs up front (e.g. before a solve, or while
online) so later [`materialize`]/[`refresh`] calls hit a warm, offline-readable
cache. Returns the [`CacheEntry`] for each."""
function prefetch(p::Provider)
    urls = p.cadence == CONST ? String[p.url_for(0.0)] :
           unique(String[p.url_for(t) for t in p.times])
    return CacheEntry[fetch_blob(p.cache, u; source_loader = p.source_loader,
                                 auth_realm = p.auth_realm) for u in urls]
end

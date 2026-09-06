# The three extensibility registries (spec/registries.md, registries.json).
#
# Each registry is a name -> implementation lookup. The single load-bearing
# rule: a new backend registers under a NEW NAME without touching the Provider
# API. The Provider depends only on the three interfaces below and resolves the
# concrete implementation by name at runtime.

# --- interfaces -------------------------------------------------------------
# Language-neutral pseudo-signatures bound to Julia idiom (abstract type +
# methods). Each concrete backend subtypes one of these and adds methods to the
# generic functions declared further down.

"""Fetch a resolved URL's bytes into the cache. Keyed by URL scheme.
Bypassed entirely in offline mode (never constructed when `offline=true`)."""
abstract type Transport end

"""Physical home of the content-addressed cache (blobs/meta/locks). Keyed by
store name. Realizes `spec/cache-format.md`; the cache key is store-independent."""
abstract type Store end

"""Open a cached blob and return CF-decoded native-grid arrays keyed by the
on-disk `file_variable` name. Keyed by format name. Implemented by component
(b) (`esio-9nb.5`); declared here so the registry seam is complete."""
abstract type Reader end

# --- generic name -> implementation registry --------------------------------

"""
    Registry{T}(kind)

A name → implementation lookup with per-entry `:active`/`:stub` status. Adding a
backend is `register!(reg, name, impl)` — never an edit to a consumer.
"""
struct Registry{T}
    kind::String
    items::Dict{String,T}
    status::Dict{String,Symbol}
end
Registry{T}(kind::AbstractString) where {T} =
    Registry{T}(String(kind), Dict{String,T}(), Dict{String,Symbol}())

"""
    register!(reg, name_or_names, impl; status=:active)

Register `impl` under one name or several (e.g. the `http` transport registers
for both `"http"` and `"https"`). Returns `impl`.
"""
function register!(r::Registry{T}, names, impl::T; status::Symbol = :active) where {T}
    namelist = names isa AbstractString ? (names,) : names
    for nm in namelist
        key = String(nm)
        r.items[key] = impl
        r.status[key] = status
    end
    return impl
end

Base.haskey(r::Registry, name::AbstractString) = haskey(r.items, String(name))

function Base.getindex(r::Registry, name::AbstractString)
    key = String(name)
    haskey(r.items, key) && return r.items[key]
    throw(ArgumentError(
        "'$key' is not registered in the $(r.kind) registry — a registration " *
        "gap, not a Provider change. Registered: $(registered_names(r))."))
end

Base.get(r::Registry, name::AbstractString, default) = get(r.items, String(name), default)

"""Sorted list of registered names."""
registered_names(r::Registry) = sort!(collect(keys(r.items)))

"""`:active` or `:stub` for a registered name."""
status_of(r::Registry, name::AbstractString) = r.status[String(name)]

# --- the three registries ---------------------------------------------------
# Populated by `EarthSciIO._register_defaults` in `__init__`.

"""Transport registry, keyed by URL scheme (`http`/`https`/`file`/`s3`-stub)."""
const TRANSPORT_REGISTRY = Registry{Transport}("transport")

"""Format/reader registry, keyed by format name. Readers are component (b);
the `zarr` stub is registered now to prove the seam."""
const FORMAT_REGISTRY = Registry{Any}("format")

"""Store registry, keyed by store name. Values are factories `(; root, …) ->
Store` so a store's configuration (e.g. the cache root) is supplied at use
site, not baked into a global instance."""
const STORE_REGISTRY = Registry{Any}("store")

"""
    make_store(name; kwargs...) -> Store

Resolve a store backend by name through [`STORE_REGISTRY`] and construct it.
`make_store("local"; root=dir)` is the local-disk cache rooted at `dir`.
"""
make_store(name::AbstractString; kwargs...) = STORE_REGISTRY[name](; kwargs...)

# --- interface generic functions (concrete backends add methods) ------------

# Transport: schemes() + fetch!(transport, url, dest; conditional, auth)
function schemes end
function fetch! end

# Store: realizes spec/cache-format.md §2 on-disk layout.
function store_name end
function get_blob end        # key -> path | nothing  (nothing => miss)
function blob_exists end     # key -> Bool
function put_blob! end       # key, staged -> committed path (atomic rename)
function get_meta end        # key -> Manifest | nothing
function put_meta! end       # key, manifest -> meta path
function lock_key end        # f, key -> f() under the per-blob advisory lock
function staging_path end    # -> a fresh tmp/<uuid>.part staging path

# Reader (format) — component (b)/esio-9nb.5.
function read_native end

# Store-backed reader capability (additive; default-off). A reader whose source
# is NOT one fetchable blob but a directory-like store (a Zarr v2 store, whose
# `.zarray`/`.zattrs`/chunks are each their own object) declares
# `store_backed(::TheReader) = true` and implements `read_store(reader, cache,
# base_url; variables, select, kwargs...)`. The Provider gates on
# `store_backed(reader)` and, when true, hands the reader `(cache, base_url)` so
# it fetches individual objects on demand — instead of pre-fetching one blob and
# calling `read_native`. Active whole-file readers inherit the `false` default.
store_backed(::Any) = false
function read_store end

# Orthogonal-selection (projection-pushdown) capability (additive; default-off).
# A reader that can honour a per-axis `select` at read time WITHOUT MATERIALISING
# THE WHOLE ARRAY declares `supports_selection(::TheReader) = true`. The Provider
# exposes this (`supports_selection(p::Provider)`) so a caller (EarthSciAST) can
# decide whether to push a projection down or fall back to a full read on its side.
#
# NOTE — it does not say what is FETCHED, and the two are genuinely different:
#
#   * store-backed + supports_selection (`zarr`): the selection decides which chunk
#     OBJECTS are downloaded, so it shrinks the transfer as well;
#   * whole-file + supports_selection (`netcdf`): the same blob is fetched under the
#     same cache key and only the requested hyperslab is MATERIALISED, so it shrinks
#     the decode and the resident arrays, not the download.
#
# `store_backed` is how a caller tells those apart — deliberately one existing
# boolean rather than a second capability, because the AST-side
# `provider_supports_selection` is a single boolean and a second one would have to
# be threaded through it. A reader that can do neither inherits the `false` default.
supports_selection(::Any) = false

"""
    reader_option_keys(reader) -> Union{Nothing,Vector{Symbol}}

The decode options `reader` accepts — `spec/registries.md` §2.1's
`reader_kwargs`, the Julia spelling of an `.esm` loader's `reader_options`.
[`Provider`] checks a declared option set against this AT CONSTRUCTION, because
an option the reader does not recognise MUST be an error rather than a silently
ignored key: a mis-typed `member_filter` that quietly selects nothing is
discovered arbitrarily far from its cause, as wrong numbers. (Rust spells the
same rule `Reader::configured`.)

The list is READ OFF the reader's own `read_native` / `read_store` method — its
keyword declaration is the single source of truth, so it cannot drift from the
code that consumes it. `nothing` means "this reader slurps every keyword"
(only [`StubReader`], which errors on any call), and disables the check.
"""
function reader_option_keys(reader)
    m = try
        store_backed(reader) ?
            which(read_store, Tuple{typeof(reader),Cache,String}) :
            which(read_native, Tuple{typeof(reader),String})
    catch
        return nothing            # no applicable method: let the call report it
    end
    decl = Base.kwarg_decl(m)
    any(endswith(String(k), "...") for k in decl) && return nothing
    return decl
end

"""A registered-but-unimplemented reader (e.g. the `zarr` stub). Calling it is a
clear error pointing at the bead that will implement it."""
struct StubReader
    name::String
    reason::String
end
read_native(s::StubReader, args...; kwargs...) =
    error("format '$(s.name)' is a registered STUB: $(s.reason)")

# The `zarr` store-backed reader (Zarr v2) — chunk math, orthogonal selection,
# the partial edge chunk, fill_value != NaN, and the load-bearing LAZINESS
# capability (a runtime index list fetches ONLY the intersecting chunk objects).
#
# Two complementary checks:
#   * against the COMMITTED corpus store (blosc-encoded by numcodecs in the Python
#     generator) — so this also proves numcodecs <-> Blosc.jl byte-compatibility;
#   * against a Julia-built POISON store — non-selected chunks hold undecodable
#     garbage, so any over-fetch blosc-errors instead of silently succeeding.

import Blosc   # activates the EarthSciIOBloscExt weakdep extension

const ZCORPUS = normpath(joinpath(@__DIR__, "..", "..", "conformance", "corpus"))
const ZBASE = "s3://earthsci-fixtures/mini.zarr"

# C-order flatten of a native array in file (dims) order (mirrors dump_julia).
_corder(a::AbstractVector) = collect(a)
_corder(a::AbstractArray) = vec(permutedims(a, reverse(1:ndims(a))))

@testset "zarr: chunk math" begin
    @test EarthSciIO._chunk_key((0, 5, 0), ".") == "0.5.0"
    @test EarthSciIO._chunk_key((3,), ".") == "3"
    @test EarthSciIO._chunk_key((1, 2), "/") == "1/2"

    # dim1 chunk_len 100: indices [0,250,260] -> chunks {0,2}; chunk 1 skipped.
    got = EarthSciIO._needed_chunks([[1], [0, 250, 260], [0]], [1, 100, 1])
    @test Set(got) == Set([(1, 0, 0), (1, 2, 0)])

    # a 3-index selection over 525 chunks touches <= 3 chunks, never 525.
    got2 = EarthSciIO._needed_chunks([[0], [50, 12345, 52000], [0]], [1, 100, 52411])
    @test Set(c[2] for c in got2) == Set([0, 123, 520])
    @test length(got2) == 3

    @test EarthSciIO._resolve_axis(EarthSciIO._parse_axis("all"), 4) == [0, 1, 2, 3]
    @test EarthSciIO._resolve_axis(EarthSciIO._parse_axis(Dict("indices" => [3, 0, 1])), 4) == [3, 0, 1]
    @test EarthSciIO._resolve_axis(EarthSciIO._parse_axis(Dict("slice" => [1, 8, 2])), 10) == [1, 3, 5, 7]
end

@testset "zarr: read_store over the committed corpus (numcodecs<->Blosc.jl)" begin
    cache = Cache(LocalStore(joinpath(ZCORPUS, "cache")); offline = true, verify = true)
    sel = Dict("axes" => Any[Dict("indices" => [1]), Dict("indices" => [1, 4]), "all"])
    nds = read_store(ZarrReader(), cache, "s3://earthsci-fixtures/isrm-mini.zarr";
                     variables = ["field3d", "pop1d"], select = sel)

    f3 = nds.variables["field3d"]
    @test f3.dims == ["layer", "y", "x"]
    @test size(f3.data) == (1, 2, 4)
    @test eltype(f3.data) == Float64
    @test _corder(f3.data) == Float64[110, 111, 112, 113, 140, 141, 142, 143]

    p1 = nds.variables["pop1d"]
    @test p1.dims == ["cell"]
    @test p1.data == Float64[1, 3, 5, 7, 9, 11, 13, 15]  # rank 1 != 3 axes -> whole
end

@testset "zarr: full read + partial edge chunk (fill_value 0.0 not -> NaN)" begin
    cache = Cache(LocalStore(joinpath(ZCORPUS, "cache")); offline = true, verify = true)
    nds = read_store(ZarrReader(), cache, "s3://earthsci-fixtures/isrm-mini.zarr";
                     variables = ["field3d"], select = nothing)
    f3 = nds.variables["field3d"]
    @test size(f3.data) == (2, 5, 4)               # full array
    @test vec(f3.data[2, 5, :]) == Float64[140, 141, 142, 143]  # partial edge chunk row
    @test !any(isnan, f3.data)                     # zeros stay zeros, never NaN
end

# --- Julia-built poison store: the LAZINESS proof --------------------------- #

function _z_zarray(shape, chunks, dtype)
    d = Dict("zarr_format" => 2, "shape" => collect(shape), "chunks" => collect(chunks),
             "dtype" => dtype,
             "compressor" => Dict("id" => "blosc", "cname" => "lz4", "clevel" => 5,
                                  "shuffle" => 1, "blocksize" => 0),
             "fill_value" => 0.0, "order" => "C", "filters" => nothing,
             "dimension_separator" => nothing)
    return Vector{UInt8}(codeunits(JSON.json(d)))
end

function _z_encode(chunk::AbstractArray)
    Blosc.set_compressor("lz4")
    flatC = _corder(chunk)                          # C-order bytes
    return Blosc.compress(flatC; level = 5, shuffle = true, itemsize = sizeof(eltype(chunk)))
end

function _z_populate(root, objects)
    store = LocalStore(root)
    for (url, data) in objects
        key = cache_key(url)
        staged = EarthSciIO.staging_path(store)
        write(staged, data)
        EarthSciIO.put_blob!(store, key, staged; ext = "")
        m = EarthSciIO.Manifest(url, nothing, nothing, bytes2hex(sha256(data)),
                                length(data), "2026-06-26T00:00:00Z", nothing, nothing)
        EarthSciIO.put_meta!(store, key, m)
    end
    return store
end

@testset "zarr: laziness never touches unselected (poison) chunks" begin
    tmp = mktempdir()
    objs = Dict{String,Vector{UInt8}}()
    objs["$ZBASE/sr/.zarray"] = _z_zarray((3, 500, 1), (1, 100, 1), "<f4")
    objs["$ZBASE/sr/.zattrs"] =
        Vector{UInt8}(codeunits(JSON.json(Dict("_ARRAY_DIMENSIONS" => ["layer", "source", "receptor"]))))
    # 3 layers x 5 source-chunks x 1 = 15 chunks. Only layer 0, source-chunks {0,3}
    # are valid; every other chunk is poison (garbage that fails blosc decode).
    for c0 in 0:2, c1 in 0:4
        key = "$ZBASE/sr/$c0.$c1.0"
        if c0 == 0 && (c1 == 0 || c1 == 3)
            objs[key] = _z_encode(fill(Float32(c0 * 1000 + c1), (1, 100, 1)))
        else
            objs[key] = Vector{UInt8}(b"\x00POISON-not-a-blosc-container\xff")
        end
    end
    store = _z_populate(tmp, objs)
    cache = Cache(store; offline = true, verify = true)

    sel = Dict("axes" => Any[Dict("indices" => [0]),
                             Dict("indices" => [5, 12, 305, 340]), "all"])
    nds = read_store(ZarrReader(), cache, ZBASE; variables = ["sr"], select = sel)
    f = nds.variables["sr"]
    @test size(f.data) == (1, 4, 1)
    @test vec(f.data) == Float64[0, 0, 3, 3]   # sources 5,12->chunk0; 305,340->chunk3

    # Control: a selection that DOES hit a poison chunk decode-errors.
    badsel = Dict("axes" => Any[Dict("indices" => [0]), Dict("indices" => [150]), "all"])
    @test_throws Exception read_store(ZarrReader(), cache, ZBASE;
                                      variables = ["sr"], select = badsel)
end

@testset "zarr: registry dispatch + store-backed provider seam" begin
    @test status_of(FORMAT_REGISTRY, "zarr") == :active
    @test store_backed(FORMAT_REGISTRY["zarr"])
    @test !store_backed(FORMAT_REGISTRY["netcdf"])

    cache = Cache(LocalStore(joinpath(ZCORPUS, "cache")); offline = true, verify = true)
    p = const_provider(cache, "s3://earthsci-fixtures/isrm-mini.zarr";
                       format = "zarr", variables = ["field3d"],
                       reader_kwargs = (; select = Dict("axes" =>
                           Any[Dict("indices" => [1]), Dict("indices" => [1, 4]), "all"])))
    nds = materialize(p)
    @test size(nds.variables["field3d"].data) == (1, 2, 4)
end

@testset "zarr: s3 transport rewrite" begin
    @test status_of(TRANSPORT_REGISTRY, "s3") == :active
    @test EarthSciIO.s3_https_url("s3://inmap-model/isrm_v1.2.1.zarr/PrimaryPM25/0.5.0") ==
          "https://inmap-model.s3.us-east-2.amazonaws.com/isrm_v1.2.1.zarr/PrimaryPM25/0.5.0"
    @test EarthSciIO.s3_https_url("s3://b/k/o"; ) isa String
    withenv("EARTHSCI_S3_REGION" => "eu-west-1") do
        @test EarthSciIO.resolve_s3_region() == "eu-west-1"
    end
end

# --- Phase 1a: per-call `select` pushdown, supports_selection, array_shape --- #

# A Store that records every `get_blob` KEY, so a test can prove the reader
# fetched ONLY the objects it needed (each on-demand object fetch is exactly one
# `get_blob` on the fast offline path). Everything else forwards to a LocalStore.
mutable struct CountingStore <: EarthSciIO.Store
    inner::LocalStore
    gets::Vector{String}
end
CountingStore(inner::LocalStore) = CountingStore(inner, String[])
EarthSciIO.store_name(s::CountingStore) = EarthSciIO.store_name(s.inner)
function EarthSciIO.get_blob(s::CountingStore, key::AbstractString)
    push!(s.gets, String(key))
    return EarthSciIO.get_blob(s.inner, key)
end
EarthSciIO.blob_exists(s::CountingStore, key::AbstractString) = EarthSciIO.blob_exists(s.inner, key)
EarthSciIO.get_meta(s::CountingStore, key::AbstractString) = EarthSciIO.get_meta(s.inner, key)
EarthSciIO.staging_path(s::CountingStore) = EarthSciIO.staging_path(s.inner)
EarthSciIO.put_blob!(s::CountingStore, key::AbstractString, staged::AbstractString; kwargs...) =
    EarthSciIO.put_blob!(s.inner, key, staged; kwargs...)
EarthSciIO.put_meta!(s::CountingStore, key::AbstractString, m::EarthSciIO.Manifest) =
    EarthSciIO.put_meta!(s.inner, key, m)
EarthSciIO.lock_key(f::Function, s::CountingStore, key::AbstractString) =
    EarthSciIO.lock_key(f, s.inner, key)

const ZSR = "s3://earthsci-fixtures/sr-mini.zarr"

# A VALID (non-poison) `sr` store: shape (3,500,1), chunks (1,100,1). Element at
# global (layer, source, 0) encodes its indices: value = layer*1_000_000 + source
# (exact in Float32 for these ranges), so a selection's values are self-checking.
function _z_sr_store(root)
    objs = Dict{String,Vector{UInt8}}()
    objs["$ZSR/sr/.zarray"] = _z_zarray((3, 500, 1), (1, 100, 1), "<f4")
    objs["$ZSR/sr/.zattrs"] = Vector{UInt8}(codeunits(JSON.json(
        Dict("_ARRAY_DIMENSIONS" => ["layer", "source", "receptor"]))))
    for c0 in 0:2, c1 in 0:4
        a = Array{Float32}(undef, 1, 100, 1)
        for j in 0:99
            a[1, j + 1, 1] = Float32(c0 * 1_000_000 + (c1 * 100 + j))
        end
        objs["$ZSR/sr/$c0.$c1.0"] = _z_encode(a)
    end
    return _z_populate(root, objs)
end

@testset "zarr: per-call select pushes down + fetches only needed chunks" begin
    store = CountingStore(_z_sr_store(mktempdir()))
    cache = Cache(store; offline = true, verify = false)
    p = const_provider(cache, ZSR; format = "zarr", variables = ["sr"])

    # layer 1, sources {5,12}∈chunk0 and {305,340}∈chunk3, all receptors.
    sel = Dict("axes" => Any[Dict("indices" => [1]),
                             Dict("indices" => [5, 12, 305, 340]), "all"])
    nds = materialize(p; select = sel)
    f = nds.variables["sr"]
    @test f.dims == ["layer", "source", "receptor"]
    @test size(f.data) == (1, 4, 1)
    @test vec(f.data) == Float64[1_000_005, 1_000_012, 1_000_305, 1_000_340]

    # Laziness: fetched ONLY .zarray + .zattrs + chunks (1,0,0) and (1,3,0) — the
    # 13 other chunks (layers 0/2, source-chunks 1/2/4) were never touched.
    expected = Set(cache_key.([
        "$ZSR/sr/.zarray", "$ZSR/sr/.zattrs", "$ZSR/sr/1.0.0", "$ZSR/sr/1.3.0"]))
    @test Set(store.gets) == expected
    @test length(store.gets) == 4
end

@testset "zarr: per-call select OVERRIDES baked reader_kwargs[:select]" begin
    cache = Cache(_z_sr_store(mktempdir()); offline = true, verify = false)
    baked = Dict("axes" => Any[Dict("indices" => [0]), Dict("indices" => [7]), "all"])
    p = const_provider(cache, ZSR; format = "zarr", variables = ["sr"],
                       reader_kwargs = (; select = baked))

    # No per-call select ⇒ the baked select still applies (regression).
    @test vec(materialize(p).variables["sr"].data) == Float64[7]            # layer 0, src 7

    # A per-call select OVERRIDES the baked one for this call only.
    over = Dict("axes" => Any[Dict("indices" => [2]), Dict("indices" => [7]), "all"])
    @test vec(materialize(p; select = over).variables["sr"].data) == Float64[2_000_007]
    # ... and the baked default is untouched afterwards.
    @test vec(materialize(p).variables["sr"].data) == Float64[7]
end

@testset "zarr: array_shape reads only .zarray (no chunk fetch)" begin
    store = CountingStore(_z_sr_store(mktempdir()))
    cache = Cache(store; offline = true, verify = false)
    p = const_provider(cache, ZSR; format = "zarr", variables = ["sr"])

    @test array_shape(p, "sr") == (3, 500, 1)
    @test store.gets == [cache_key("$ZSR/sr/.zarray")]   # ONLY .zarray, never a chunk
end

@testset "zarr: supports_selection / array_shape capability surface" begin
    cache = Cache(LocalStore(joinpath(ZCORPUS, "cache")); offline = true, verify = true)

    # store-backed zarr provider CAN push down
    pz = const_provider(cache, "s3://earthsci-fixtures/isrm-mini.zarr";
                        format = "zarr", variables = ["field3d"])
    @test supports_selection(ZarrReader())
    @test supports_selection(pz)

    @test store_backed(pz)

    # a whole-file reader that honours `select` at DECODE time: it answers
    # supports_selection but NOT store_backed, which is how a caller tells "the
    # download shrank" from "only the decode did". array_shape is still nothing
    # (a blob's shape is not knowable without reading it).
    pn = const_provider(cache, "file:///dev/null"; format = "netcdf")
    @test supports_selection(pn)
    @test !store_backed(pn)
    @test array_shape(pn, "anything") === nothing
    @test supports_selection(NetCDFReader())

    # readers that cannot honour a select at all
    for fmt in ("csv", "ff10")
        pw = const_provider(cache, "file:///dev/null"; format = fmt)
        @test !supports_selection(pw)
        @test !store_backed(pw)
        @test array_shape(pw, "anything") === nothing
    end
    @test !supports_selection(CSVReader())
    @test !supports_selection(FF10Reader())
end

# --- the gather -> scatter assembly inversion (peak-memory contract) -------- #
#
# `_read_v2_array` used to decompress EVERY needed chunk into a `Dict` and only
# then GATHER the output from it, which pins every chunk simultaneously: on the
# real ISRM SR array one read held 416 chunks x ~21 MB = ~8.7 GB to produce a
# 0.59 GB result (~15x amplification) and OOM-killed a production run. The reader
# is now CHUNK-driven — one chunk decoded, scattered into the output, and dropped
# before the next fetch. Two things must hold, and both are checked below:
#   * the decoded arrays are BIT-identical to the gather's (same cells, same
#     conversion, same fill_value; only the write ORDER changed), and
#   * the peak simultaneous residency is O(one chunk), not O(all chunks).

# The retired OUTPUT-driven read path (the old `_read_v2_array` body), over the
# same cache: this test's oracle for bit-identity AND its buffer-everything
# memory baseline. `EarthSciIO._assemble` is retained only for this.
function _zarr_gather_read(cache, base, arr, axes_spec)
    meta = EarthSciIO._parse_zarray(EarthSciIO._fetch_bytes(cache, "$base/$arr/.zarray"))
    ndim = EarthSciIO._ndim(meta)
    sel = [EarthSciIO._resolve_axis(EarthSciIO._parse_axis(axes_spec[d]), meta.shape[d])
           for d in 1:ndim]
    buffers = Dict{NTuple{ndim,Int},Any}()
    for ck in EarthSciIO._needed_chunks(sel, meta.chunks)
        raw = EarthSciIO._fetch_bytes_optional(
            cache, "$base/$arr/" * EarthSciIO._chunk_key(ck, meta.dim_sep))
        buffers[ck] = raw === nothing ? nothing :
                      EarthSciIO._chunk_array(meta, EarthSciIO._decompress(meta, raw))
    end
    return EarthSciIO._assemble(sel, meta, buffers)
end

# A Store that force-collects and records the LIVE heap at every object fetch, so
# a read's peak simultaneous residency is observable in-process (no RSS, no
# subprocess): under the gather, the k-th chunk fetch sees k-1 decoded chunks
# still alive; under the scatter it sees at most one.
mutable struct ProbeStore <: EarthSciIO.Store
    inner::LocalStore
    live::Vector{Int}
    on::Bool
end
ProbeStore(inner::LocalStore) = ProbeStore(inner, Int[], false)
EarthSciIO.store_name(s::ProbeStore) = EarthSciIO.store_name(s.inner)
function EarthSciIO.get_blob(s::ProbeStore, key::AbstractString)
    if s.on
        GC.gc(true)
        push!(s.live, Int(Base.gc_live_bytes()))
    end
    return EarthSciIO.get_blob(s.inner, key)
end
EarthSciIO.blob_exists(s::ProbeStore, key::AbstractString) = EarthSciIO.blob_exists(s.inner, key)
EarthSciIO.get_meta(s::ProbeStore, key::AbstractString) = EarthSciIO.get_meta(s.inner, key)
EarthSciIO.staging_path(s::ProbeStore) = EarthSciIO.staging_path(s.inner)
EarthSciIO.put_blob!(s::ProbeStore, key::AbstractString, staged::AbstractString; kwargs...) =
    EarthSciIO.put_blob!(s.inner, key, staged; kwargs...)
EarthSciIO.put_meta!(s::ProbeStore, key::AbstractString, m::EarthSciIO.Manifest) =
    EarthSciIO.put_meta!(s.inner, key, m)
EarthSciIO.lock_key(f::Function, s::ProbeStore, key::AbstractString) =
    EarthSciIO.lock_key(f, s.inner, key)

const ZSCAT = "s3://earthsci-fixtures/scatter-mini.zarr"

# A many-chunk store: (1, NCH*ROWS, NCOL) f8 in (1, ROWS, NCOL) chunks, so the
# selection below spans every chunk and no single chunk can serve two output
# rows' worth of the answer. Chunk NCH-8's object is deliberately ABSENT (its
# cells must keep fill_value under BOTH paths). Element (0,j,k) = j*10_000 + k,
# exact in Float64, so every decoded value is self-checking.
const ZS_NCH, ZS_ROWS, ZS_NCOL = 40, 50, 2000
const ZS_CHUNKBYTES = ZS_ROWS * ZS_NCOL * 8          # 800_000 B decompressed
const ZS_ABSENT = 7

function _z_scatter_store(root)
    objs = Dict{String,Vector{UInt8}}()
    objs["$ZSCAT/sr/.zarray"] =
        _z_zarray((1, ZS_NCH * ZS_ROWS, ZS_NCOL), (1, ZS_ROWS, ZS_NCOL), "<f8")
    objs["$ZSCAT/sr/.zattrs"] = Vector{UInt8}(codeunits(JSON.json(
        Dict("_ARRAY_DIMENSIONS" => ["layer", "source", "receptor"]))))
    for c in 0:(ZS_NCH - 1)
        c == ZS_ABSENT && continue                   # absent object -> fill_value
        a = Array{Float64}(undef, 1, ZS_ROWS, ZS_NCOL)
        for j in 0:(ZS_ROWS - 1), k in 0:(ZS_NCOL - 1)
            a[1, j + 1, k + 1] = (c * ZS_ROWS + j) * 10_000 + k
        end
        objs["$ZSCAT/sr/0.$c.0"] = _z_encode(a)
    end
    return _z_populate(root, objs)
end

@testset "zarr: chunk-driven scatter is bit-identical to the old gather" begin
    store = ProbeStore(_z_scatter_store(mktempdir()))
    cache = Cache(store; offline = true, verify = false)
    # one source row out of every chunk => the selection spans all 40 chunks.
    rows = [c * ZS_ROWS + 3 for c in 0:(ZS_NCH - 1)]
    axes_spec = Any[Dict("indices" => [0]), Dict("indices" => rows), "all"]

    store.live = Int[]; store.on = true
    got = read_store(ZarrReader(), cache, ZSCAT;
                     variables = ["sr"], select = Dict("axes" => axes_spec)).variables["sr"].data
    scatter_live = copy(store.live)

    store.live = Int[]
    ref = _zarr_gather_read(cache, ZSCAT, "sr", axes_spec)
    gather_live = copy(store.live)
    store.on = false

    # (1) same shape, and BIT-identical values — not merely approximately equal.
    @test size(got) == (1, ZS_NCH, ZS_NCOL) == size(ref)
    @test eltype(got) == eltype(ref) == Float64
    @test reinterpret(UInt64, vec(got)) == reinterpret(UInt64, vec(ref))

    # (2) the values are the ones the store encodes, and the ABSENT chunk's cells
    #     kept fill_value 0.0 (never NaN) under the scatter.
    values_ok = true
    for c in 0:(ZS_NCH - 1)
        c == ZS_ABSENT && continue
        for k in 0:(ZS_NCOL - 1)
            values_ok &= got[1, c + 1, k + 1] == (c * ZS_ROWS + 3) * 10_000 + k
        end
    end
    @test values_ok
    @test all(iszero, got[1, ZS_ABSENT + 1, :])
    @test !any(isnan, got)

    # (3) the peak-memory contract. `store.live` is the live heap at each object
    #     fetch: the gather accumulates a decoded chunk per fetch (~linear growth
    #     to ~39 chunks), the scatter drops each chunk before the next fetch.
    scatter_span = maximum(scatter_live) - minimum(scatter_live)
    gather_span = maximum(gather_live) - minimum(gather_live)
    @info "zarr scatter vs gather live-heap span" chunk_MB=ZS_CHUNKBYTES/2^20 scatter_MB=scatter_span/2^20 gather_MB=gather_span/2^20
    @test gather_span > 20 * ZS_CHUNKBYTES        # old path really did pin them all
    @test scatter_span < 4 * ZS_CHUNKBYTES        # new path holds O(1) chunks
    @test scatter_span < gather_span ÷ 5
end

@testset "zarr: per-call select on a non-store reader is a clear error" begin
    cache = Cache(LocalStore(joinpath(ZCORPUS, "cache")); offline = true)
    pw = const_provider(cache, "file:///dev/null"; format = "csv")
    # raised before any fetch — the reader can't honour a projection pushdown
    @test_throws ArgumentError materialize(pw; select = Dict("axes" => Any["all"]))
    @test_throws ArgumentError refresh(pw, 0.0; select = Dict("axes" => Any["all"]))
end

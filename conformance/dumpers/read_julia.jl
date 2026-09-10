# Julia track's READBACK driver for the write-conformance harness.
#
# Reads a produced Zarr v3 sharded store (written by ANY track's writer) with the
# **Julia store-backed reader** (`EarthSciIO.ZarrReader` / `read_store`) — the same
# reader the read-conformance harness proves conformant — and emits its decoded
# native arrays as a canonical JSON dump (schema `earthsciio/write-native-dump/v1`,
# identical shape to `read_python.py`). `conformance/crosscheck_write.py` then
# asserts each writer's store, decoded here, agrees with the spec oracle and
# pairwise within tolerance (RFC §16.6).
#
# The reader is store-backed: handed `(cache, base_url; variables)`, it fetches
# each object through a content-addressed `Cache`. A produced LOCAL store is read
# by pointing a plain (online, no-network) `Cache` at a `file://` base URL — the
# FileTransport copies each local object into a throwaway cache dir. Blosc DECode
# rides the same `EarthSciIOBloscExt` weakdep the reader always uses; the plain v3
# `zstd` DECode (what the `wasm` output profile writes) rides `EarthSciIOZstdExt`.
# Loading BOTH means this one driver cross-reads every codec profile variant.
#
# Usage:  julia --project="$(julia conformance/dumpers/julia_env.jl)" \
#             conformance/dumpers/read_julia.jl STORE_DIR WRITER_LABEL [OUT.json] [SPEC.json]
#         (`conformance/run_write_conformance.sh` does both steps.)

using EarthSciIO
import JSON

include(joinpath(@__DIR__, "codec_weakdeps.jl"))
load_codec_weakdeps!()

# Row-major (C order) flatten of a native array whose axes are in file order.
_corder(a::AbstractVector) = collect(a)
_corder(a::AbstractArray) = vec(permutedims(a, reverse(1:ndims(a))))

function encode_field(field)
    data = field.data
    dims = collect(String.(field.dims))
    flat = _corder(data)
    et = eltype(data)
    if et <: AbstractFloat
        dtype = "float64"
        vals = Any[isnan(x) ? nothing : Float64(x) for x in flat]
    elseif et <: Integer
        dtype = et == Int32 ? "int32" : "int64"
        vals = Any[Int(x) for x in flat]
    else
        error("unexpected numeric eltype $et in field with dims $dims")
    end
    return Dict("dtype" => dtype, "dims" => dims,
                "shape" => collect(Int, size(data)), "data" => vals)
end

function main()
    length(ARGS) >= 2 ||
        error("usage: read_julia.jl STORE_DIR WRITER_LABEL [OUT.json] [SPEC.json]")
    store_dir = abspath(ARGS[1])
    writer_label = ARGS[2]
    conf = normpath(joinpath(@__DIR__, ".."))
    spec_path = length(ARGS) >= 4 ? ARGS[4] : joinpath(conf, "write_spec.json")
    spec = JSON.parsefile(spec_path)

    arrays = String[]
    for co in spec["coords"]; push!(arrays, String(co["name"])); end
    for v in spec["vars"];   push!(arrays, String(v["name"]));  end

    # A plain online cache over a throwaway dir; a `file://` base URL means every
    # object fetch is a local copy (no network). verify=false: freshly-written
    # objects carry no sidecar manifest to check against.
    cachedir = mktempdir()
    cache = Cache(LocalStore(cachedir); offline = false, verify = false)
    base_url = "file://" * store_dir

    nds = read_store(ZarrReader(), cache, base_url; variables = arrays)

    fields = Dict{String,Any}()
    for (n, f) in nds.variables
        fields[n] = encode_field(f)
    end
    out = Dict(
        "schema" => "earthsciio/write-native-dump/v1",
        "writer" => writer_label,
        "reader" => "julia",
        "reader_impl" => "EarthSciIO.read_store(ZarrReader)",
        "fields" => fields,
    )
    text = JSON.json(out, 2)
    if length(ARGS) >= 3
        open(ARGS[3], "w") do io
            write(io, text); write(io, "\n")
        end
    else
        println(text)
    end
end

main()
